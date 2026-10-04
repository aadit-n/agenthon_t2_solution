from __future__ import annotations

import unittest
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd

from t2solution.model import _historical_width, _nearest_correlation
from t2solution.text import _read_documents, _url


class CoreTests(unittest.TestCase):
    def test_house_endpoint_origin_gets_v1(self) -> None:
        self.assertEqual(_url("http://house:8000"), "http://house:8000/v1/chat/completions")
        self.assertEqual(_url("http://house:8000/v1"), "http://house:8000/v1/chat/completions")
        self.assertEqual(
            _url("http://house:8000/v1/chat/completions"),
            "http://house:8000/v1/chat/completions",
        )

    def test_corpus_index_file_metadata_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "opaque.txt").write_text("signal", encoding="utf-8")
            (root / "corpus_index.json").write_text(json.dumps({"documents": [{
                "file": "opaque.txt", "doc_id": "official-id", "timestamp": "2024-01-02",
                "source": "source", "doc_type": "statement",
            }]}), encoding="utf-8")
            docs = _read_documents(root)
        self.assertEqual(docs[0]["doc_id"], "official-id")
        self.assertEqual(docs[0]["timestamp"], "2024-01-02")

    def test_correlation_is_psd_and_unit_diagonal(self) -> None:
        raw = np.array([[1.0, 1.2], [1.2, 1.0]])
        fixed = _nearest_correlation(raw)
        self.assertTrue(np.allclose(np.diag(fixed), 1.0))
        self.assertGreater(np.linalg.eigvalsh(fixed).min(), 0.0)

    def test_width_calibration_is_bounded(self) -> None:
        rng = np.random.default_rng(3)
        steps = pd.Series(rng.standard_normal(1000))
        value = _historical_width(steps, [21, 63], False, float(steps.std()))
        self.assertGreaterEqual(value, 0.75)
        self.assertLessEqual(value, 1.50)


if __name__ == "__main__":
    unittest.main()
