"""Fault-injection and support-preservation tests for optional House conditioning."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np

from t2solution import conditioning
from t2solution.text import _house_json
from test_candidate import fixture


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(9).normal(size=(300, 2, 3))
        self.labels, self.summaries = conditioning.clusters(self.samples)
        self.docs = [{'doc_id': 'd1', 'text': 'The policy outlook remains uncertain and conditional on new data.'}]

    def response(self, factor=1.2, quote=None):
        return {'weights': [{'cluster_id': s['cluster_id'], 'relative_weight': factor,
                             'evidence': [{'doc_id': 'd1', 'quote': quote or self.docs[0]['text']}],
                             'mechanism': 'Test evidence makes this path more plausible.'}
                            for s in self.summaries]}

    def test_hallucinated_quotes_are_neutral(self):
        values = conditioning.weights(self.response(quote='A made-up quotation not present in the document.'), self.summaries, self.docs)
        np.testing.assert_array_equal(values, np.ones(len(self.summaries)))

    def test_missing_mechanisms_are_neutral(self):
        raw = self.response()
        for row in raw['weights']:
            row.pop('mechanism')
        values = conditioning.weights(raw, self.summaries, self.docs)
        np.testing.assert_array_equal(values, np.ones(len(self.summaries)))

    def test_unaudited_proposal_is_not_applied(self):
        unit = fixture()
        with patch.dict('os.environ', {'MODEL_ENDPOINT': 'https://house.invalid', 'MODEL_TOKEN': 'fake', 'MODEL_NAME': 'house'}, clear=True), \
             patch('t2solution.conditioning.documents', return_value=self.docs), \
             patch('t2solution.conditioning._house_json', return_value=self.response()) as call, \
             patch('t2solution.conditioning.time.monotonic', side_effect=[0, 121]):
            out, info = conditioning.condition(unit, '2024-01-01', self.samples, {}, 0)
        np.testing.assert_array_equal(out, self.samples)
        self.assertFalse(info['text_used'])
        self.assertEqual(call.call_count, 1)

    def test_incomplete_and_duplicate_clusters_refused(self):
        raw = self.response()
        raw['weights'].pop()
        self.assertIsNone(conditioning.weights(raw, self.summaries, self.docs))
        raw = self.response()
        raw['weights'].append(raw['weights'][0])
        self.assertIsNone(conditioning.weights(raw, self.summaries, self.docs))

    def test_arbitrary_weights_cannot_escape_bounds(self):
        values = conditioning.weights(self.response(1000), self.summaries, self.docs)
        np.testing.assert_array_equal(values, np.full(len(self.summaries), 1.25))

    def test_neutral_response_preserves_samples_bit_exactly(self):
        out, info = conditioning.apply(self.samples, self.labels, self.summaries, np.ones(len(self.summaries)), 0)
        np.testing.assert_array_equal(out, self.samples)
        self.assertFalse(info['applied'])

    def test_conditioning_only_selects_original_whole_draws(self):
        values = np.linspace(0.75, 1.25, len(self.summaries))
        out, info = conditioning.apply(self.samples, self.labels, self.summaries, values, 0)
        original = {row.tobytes() for row in self.samples}
        self.assertTrue(all(row.tobytes() in original for row in out))
        self.assertGreaterEqual(info['ess_ratio'], 0.95)
        self.assertLessEqual(info['expected_max_mean_shift_sd'], 0.15)
        again, _ = conditioning.apply(self.samples, self.labels, self.summaries, values, 0)
        np.testing.assert_array_equal(out, again)

    def test_no_credentials_means_no_network_and_exact_fallback(self):
        unit = fixture()
        with patch.dict('os.environ', {}, clear=True), patch('t2solution.conditioning._house_json') as call:
            out, info = conditioning.condition(unit, '2024-01-01', self.samples, {}, 0)
        np.testing.assert_array_equal(out, self.samples)
        self.assertEqual(info['house_calls'], 0)
        call.assert_not_called()

    def test_future_missing_date_and_duplicate_documents_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            rows = []
            for i, stamp in enumerate(['2024-01-01', '2024-02-01', '', '2024-01-01']):
                file = f'{i}.txt'
                (root / file).write_text('same eligible document', encoding='utf-8')
                rows.append({'file': file, 'doc_id': str(i), 'timestamp': stamp})
            (root / 'corpus_index.json').write_text(json.dumps({'documents': rows}), encoding='utf-8')
            unit = fixture()
            unit.text_dir = root
            docs = conditioning.documents(unit, '2024-01-01')
        self.assertEqual([d['doc_id'] for d in docs], ['0'])

    def test_mocked_house_and_audit_pipeline(self):
        unit = fixture()
        raw = self.response()
        for i, row in enumerate(raw['weights']):
            row['relative_weight'] = 0.75 if i % 2 else 1.25
        with patch.dict('os.environ', {'MODEL_ENDPOINT': 'https://house.invalid', 'MODEL_TOKEN': 'fake', 'MODEL_NAME': 'house'}, clear=True), \
             patch('t2solution.conditioning.documents', return_value=self.docs), \
             patch('t2solution.conditioning._house_json', return_value=raw) as call:
            out, info = conditioning.condition(unit, '2024-01-01', self.samples, {}, 0)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(info['house_calls'], 2)
        self.assertTrue(info['text_used'])
        self.assertEqual(out.shape, self.samples.shape)

    def test_uniform_relative_weight_changes_are_neutral(self):
        out, info = conditioning.apply(self.samples, self.labels, self.summaries,
                                        np.full(len(self.summaries), 1.25), 0)
        np.testing.assert_array_equal(out, self.samples)
        self.assertFalse(info['applied'])

    def test_disagreeing_audit_is_neutral(self):
        unit = fixture()
        with patch.dict('os.environ', {'MODEL_ENDPOINT': 'https://house.invalid', 'MODEL_TOKEN': 'fake', 'MODEL_NAME': 'house'}, clear=True), \
             patch('t2solution.conditioning.documents', return_value=self.docs), \
             patch('t2solution.conditioning._house_json', side_effect=[self.response(1.2), self.response(0.8)]):
            out, info = conditioning.condition(unit, '2024-01-01', self.samples, {}, 0)
        np.testing.assert_array_equal(out, self.samples)
        self.assertFalse(info['text_used'])

    def test_http_errors_do_not_expose_credentials(self):
        capture = io.StringIO()
        error = urllib.error.HTTPError('https://private.invalid', 403, 'secret credential', {}, None)
        with patch('urllib.request.urlopen', side_effect=error), contextlib.redirect_stderr(capture):
            result = _house_json('https://private.invalid', 'secret-token', 'house', 'test', {}, max_tokens=100)
        self.assertIsNone(result)
        self.assertIn('403', capture.getvalue())
        self.assertNotIn('secret', capture.getvalue())
        self.assertNotIn('private.invalid', capture.getvalue())

    def test_output_cap_enforced_before_network(self):
        with patch('urllib.request.urlopen') as call:
            with self.assertRaises(ValueError):
                _house_json('https://house.invalid', 'fake', 'house', 'test', {}, max_tokens=4001)
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
