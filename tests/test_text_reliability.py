"""Offline fault injection: no test connects to House or uses withheld outcomes."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import numpy as np
import pandas as pd

from t2solution import legacy_text
from t2solution.cli import main as forecast_main
from t2solution.house import HouseClient, MAX_CALLS
from t2solution.model import prepare, simulate
from t2solution.text import TextView, _ask, infer_text_view, output_budget, validate_signal, validate_synthesis
from t2solution.text_context import excerpt, read_documents, task_context
from t2solution.unit import Unit

QUOTE = "The outlook remains highly uncertain."
ENV = {"MODEL_ENDPOINT": "http://mock-house", "MODEL_TOKEN": "SECRET-TEST-TOKEN",
       "MODEL_NAME": "house", "T2_DISABLE_TEXT": ""}


def unit(assets=None, monthly=False) -> Unit:
    assets = assets or ["EUR", "JPY"]
    dates = pd.date_range("2020-01-01", periods=35, freq="MS") if monthly else pd.bdate_range("2022-01-03", periods=400)
    frame = pd.concat([pd.DataFrame({"date": dates, "asset": asset,
                                    "value": 1 + np.arange(len(dates)) * .0001 + .01 * np.sin(np.arange(len(dates)))})
                       for asset in assets], ignore_index=True)
    card = {"targets": {"asset_ids": assets, "horizons": [140, 160] if monthly else [21, 63]},
            "panels": {"g10_fx_daily": {"asset_ids": assets, "series": ["DEXUSEU", "DEXJPUS"]}}}
    if monthly:
        card["targets"]["observation_periods"] = ["2023-07", "2023-08"]
    return Unit(pathlib.Path("."), card, {}, {"g10_fx_daily": frame}, pathlib.Path("text"),
                "test", "T2-F3", assets, card["targets"]["horizons"], "level",
                "monthly" if monthly else "daily", "native", 200)


def document(doc_id="doc1") -> dict:
    return {"doc_id": doc_id, "timestamp": "2023-07-01", "source": "test", "doc_type": "statement",
            "text": QUOTE, "_full_text": QUOTE}


def signal(u, neutral=False) -> dict:
    return {"relevance": 0.0 if neutral else .8, "confidence": .9, "event": False, "established": True,
            "summary": "uncertain outlook", "evidence_quotes": [] if neutral else [QUOTE],
            "per_asset": {asset: {"direction": 0.0 if neutral else .5,
                                   "uncertainty": 0.0 if neutral else .6, "tail_tilt": 0.0}
                          for asset in u.assets}}


def synthesis(u, neutral=False, ids=None) -> dict:
    return {"scenarios": [{"name": "base", "probability": 1.0,
                            "drift_sd": {a: 0.0 if neutral else .4 for a in u.assets},
                            "vol_mult": {a: 1.0 if neutral else 1.5 for a in u.assets},
                            "tail_tilt": {a: 0.0 for a in u.assets},
                            "evidence": ids or [], "explanation": "document-based"}]}


def envelope(value, finish="stop") -> bytes:
    return json.dumps({"choices": [{"finish_reason": finish, "message": {"content": json.dumps(value)}}],
                       "usage": {"completion_tokens": 200}}).encode()


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size):
        return self.body[:size]


class DocumentTests(unittest.TestCase):
    def write_docs(self, root, rows):
        for row in rows:
            (root / row["file"]).write_text(row.get("body", QUOTE), encoding="utf-8")
        (root / "corpus_index.json").write_text(json.dumps({"documents": rows}), encoding="utf-8")

    def test_future_undated_and_duplicates_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write_docs(root, [{"file": "old.txt", "doc_id": "old", "timestamp": "2024-01-01"},
                                   {"file": "new.txt", "doc_id": "new", "timestamp": "2024-01-02"},
                                   {"file": "future.txt", "timestamp": "2024-01-03"},
                                   {"file": "unknown.txt"}])
            audit = {}
            docs = read_documents(root, asof="2024-01-02", audit=audit)
        self.assertEqual([d["doc_id"] for d in docs], ["new"])
        self.assertEqual(audit["document_exclusions"], {"after_cutoff": 1, "undated": 1, "duplicate": 1})

    def test_utc_cutoff_includes_entire_asof_day(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write_docs(root, [{"file": "a.txt", "body": "a", "timestamp": "2024-01-02T23:59:59Z"},
                                   {"file": "b.txt", "body": "b", "timestamp": "2024-01-03T01:00:00+02:00"},
                                   {"file": "c.txt", "body": "c", "timestamp": "2024-01-03T00:00:00Z"}])
            docs = read_documents(root, asof="2024-01-02")
        self.assertEqual({d["doc_id"] for d in docs}, {"a", "b"})

    def test_duplicate_id_even_with_different_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write_docs(root, [{"file": "a.txt", "body": "a", "doc_id": "one", "timestamp": "2024-01-01"},
                                   {"file": "b.txt", "body": "b", "doc_id": "one", "timestamp": "2024-01-02"}])
            docs = read_documents(root)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["text"], "b")

    def test_missing_or_malformed_index_does_not_guess_filename_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "2024-01-01.txt").write_text(QUOTE, encoding="utf-8")
            self.assertEqual(read_documents(root), [])
            (root / "corpus_index.json").write_text("invalid", encoding="utf-8")
            audit = {}
            self.assertEqual(read_documents(root, audit=audit), [])
            self.assertEqual(audit["document_exclusions"]["invalid_index"], 1)

    def test_excerpt_retains_head_and_tail_without_exceeding_budget(self):
        body = "HEAD " + "a" * 10_000 + " TAIL"
        result = excerpt(body, 9000)
        self.assertEqual(len(result), 9000)
        self.assertTrue(result.startswith("HEAD"))
        self.assertTrue(result.endswith("TAIL"))
        self.assertIn("omitted text", result)

    def test_total_character_and_document_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.write_docs(root, [{"file": f"{i}.txt", "body": str(i) + "x" * 10_000,
                                   "timestamp": "2024-01-01"} for i in range(15)])
            docs = read_documents(root)
        self.assertLessEqual(len(docs), 10)
        self.assertEqual(sum(len(d["text"]) for d in docs), 70_000)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.u, self.doc = unit(), document()
        self.cards = [validate_signal(signal(self.u), document(doc_id), self.u)[0] for doc_id in ("one", "two")]

    def test_complete_signal_preserves_legacy_features(self):
        raw = signal(self.u)
        actual, reason = validate_signal(raw, self.doc, self.u)
        expected = legacy_text._signal_card(raw, self.doc, self.u)
        self.assertEqual(reason, "valid_signal")
        self.assertEqual({k: v for k, v in actual.items() if k != "evidence_quotes"}, expected)

    def test_missing_asset_is_rejected_not_silently_zero_filled(self):
        raw = signal(self.u)
        del raw["per_asset"]["JPY"]
        self.assertIsNotNone(legacy_text._signal_card(raw, self.doc, self.u))
        self.assertEqual(validate_signal(raw, self.doc, self.u)[1], "incomplete_signal_assets")

    def test_missing_value_and_nonfinite_or_coerced_numbers_rejected(self):
        for bad in (None, "0.5", True, float("nan"), float("inf"), 10 ** 1000, 1.1):
            with self.subTest(bad_type=type(bad).__name__):
                raw = signal(self.u)
                raw["per_asset"]["EUR"]["direction"] = bad
                self.assertIsNone(validate_signal(raw, self.doc, self.u)[0])

    def test_string_false_is_not_true(self):
        raw = signal(self.u)
        raw["event"] = "false"
        self.assertEqual(validate_signal(raw, self.doc, self.u)[1], "invalid_signal_flags")

    def test_missing_or_invented_quote_rejected(self):
        raw = signal(self.u)
        raw["evidence_quotes"] = []
        self.assertEqual(validate_signal(raw, self.doc, self.u)[1], "missing_signal_evidence")
        raw["evidence_quotes"] = ["This sentence was never in the document."]
        self.assertEqual(validate_signal(raw, self.doc, self.u)[1], "unsupported_quote")

    def test_quote_must_be_present_in_both_excerpt_and_full_text(self):
        raw = signal(self.u)
        doc = {**self.doc, "_full_text": "different source"}
        self.assertEqual(validate_signal(raw, doc, self.u)[1], "unsupported_quote")
        doc = {**self.doc, "text": "excerpt did not contain evidence"}
        self.assertEqual(validate_signal(raw, doc, self.u)[1], "unsupported_quote")

    def test_quote_whitespace_and_case_normalization(self):
        raw = signal(self.u)
        raw["evidence_quotes"] = ["THE outlook   remains highly uncertain."]
        self.assertIsNotNone(validate_signal(raw, self.doc, self.u)[0])

    def test_neutral_signal_can_have_no_quote(self):
        self.assertIsNotNone(validate_signal(signal(self.u, neutral=True), self.doc, self.u)[0])

    def test_complete_synthesis_preserves_v02_guardrails(self):
        raw = synthesis(self.u, ids=["one", "two"])
        validated, reason = validate_synthesis(raw, self.u, self.cards)
        self.assertEqual(reason, "valid_synthesis")
        actual = legacy_text._parse(validated, self.u, self.cards)
        expected = legacy_text._parse(raw, self.u, self.cards)
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(actual[0].drift_sd["EUR"], .32 * .4)
        self.assertAlmostEqual(actual[0].vol_mult["EUR"], 1.175)

    def test_incomplete_synthesis_or_unsupported_evidence_rejected(self):
        for key in ("drift_sd", "vol_mult", "tail_tilt"):
            raw = synthesis(self.u, ids=["one"])
            del raw["scenarios"][0][key]["JPY"]
            self.assertEqual(validate_synthesis(raw, self.u, self.cards)[1], "incomplete_scenario_assets")
        raw = synthesis(self.u, ids=["invented"])
        self.assertEqual(validate_synthesis(raw, self.u, self.cards)[1], "unsupported_scenario_evidence")
        raw["scenarios"][0]["evidence"] = []
        self.assertEqual(validate_synthesis(raw, self.u, self.cards)[1], "missing_scenario_evidence")

    def test_probability_and_volatility_bounds(self):
        for key, bad, reason in (("probability", .4, "invalid_probability_sum"),
                                 ("probability", True, "invalid_probability"),
                                 ("vol_mult", .9, "invalid_scenario_numbers")):
            raw = synthesis(self.u, ids=["one"])
            if key == "vol_mult":
                raw["scenarios"][0][key]["EUR"] = bad
            else:
                raw["scenarios"][0][key] = bad
            self.assertEqual(validate_synthesis(raw, self.u, self.cards)[1], reason)


class ContextTests(unittest.TestCase):
    def test_opposite_fx_quote_directions(self):
        ctx = task_context(unit(), "2024-01-01")["asset_context"]
        self.assertEqual(ctx["EUR"]["native_quote"], "USD per unit of EUR")
        self.assertIn("strengthens", ctx["EUR"]["positive_direction_means"])
        self.assertEqual(ctx["JPY"]["native_quote"], "JPY units per USD")
        self.assertIn("weakens", ctx["JPY"]["positive_direction_means"])

    def test_native_unknown_target_is_not_assumed_to_be_fx(self):
        u = unit(["X", "Y"])
        u.card["panels"]["g10_fx_daily"]["series"] = ["unknown", "unknown"]
        ctx = task_context(u, "2024-01-01")["asset_context"]
        self.assertEqual(ctx["X"]["native_quote"], "native")

    def test_monthly_leads_resolved_without_renaming_labels(self):
        ctx = task_context(unit(monthly=True), "2023-07-01")
        leads = ctx["asset_context"]["EUR"]["horizon_leads"]
        self.assertEqual(ctx["horizons"], [140, 160])
        self.assertEqual(leads["140"]["steps_from_last_observation"], 8)
        self.assertEqual(leads["160"]["steps_from_last_observation"], 9)
        self.assertEqual(leads["140"]["observation_period"], "2023-07")


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.stderr = io.StringIO()
        self.redirect = contextlib.redirect_stderr(self.stderr)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.client = HouseClient("http://SECRET-ENDPOINT", "SECRET-TOKEN", "house")

    def request(self):
        return self.client.request("SECRET-PROMPT", {}, 850, phase="extraction", attempt=1)

    def test_request_route_thinking_and_proxy_environment(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy"}), patch(
            "urllib.request.urlopen", return_value=Response(envelope({"ok": True}))
        ) as mocked:
            self.assertEqual(self.request().value, {"ok": True})
            self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy")
        req = mocked.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "http://SECRET-ENDPOINT/v1/chat/completions")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["temperature"], 0)

    def test_parseable_truncated_json_is_retried_with_larger_budget(self):
        with patch("urllib.request.urlopen", side_effect=[Response(envelope({"ok": True}, "length")),
                                                          Response(envelope({"ok": True}))]) as mocked:
            value = _ask(self.client, "test", {}, "extraction", 10, lambda x: (x, "valid_signal"))
        self.assertEqual(value, {"ok": True})
        tokens = [json.loads(call.args[0].data)["max_tokens"] for call in mocked.call_args_list]
        self.assertGreater(tokens[1], tokens[0])
        self.assertEqual(self.client.calls, 2)
        self.assertEqual(self.client.attempts[0]["reason"], "truncated")

    def test_403_blocks_route_and_is_not_retried(self):
        error = urllib.error.HTTPError("SECRET-URL", 403, "SECRET-BODY", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            self.assertIsNone(_ask(self.client, "x", {}, "extraction", 2, lambda x: (x, "valid")))
            self.assertEqual(self.request().reason, "route_blocked")
        self.assertEqual(self.client.calls, 1)

    def test_429_has_one_bounded_retry(self):
        error = urllib.error.HTTPError("url", 429, "body", {"Retry-After": "999999"}, None)
        with patch("urllib.request.urlopen", side_effect=[error, Response(envelope({"ok": True}))]), patch(
            "t2solution.text.time.sleep"
        ) as sleep:
            value = _ask(self.client, "x", {}, "extraction", 2, lambda x: (x, "valid"))
        self.assertEqual(value, {"ok": True})
        sleep.assert_called_once_with(3.0)
        self.assertEqual(self.client.calls, 2)

    def test_two_network_failures_do_not_loop_forever(self):
        with patch("urllib.request.urlopen", side_effect=OSError("SECRET-ERROR")), patch("t2solution.text.time.sleep"):
            self.assertIsNone(_ask(self.client, "x", {}, "extraction", 2, lambda x: (x, "valid")))
        self.assertEqual(self.client.calls, 2)

    def test_invalid_envelope_and_content_are_retryable(self):
        for body in (b"invalid", b"{}", b'{"choices": []}', b'{"choices": [{"message": {"content": "invalid"}}]}'):
            with patch("urllib.request.urlopen", return_value=Response(body)):
                self.assertTrue(self.request().retryable)

    def test_oversized_response_rejected(self):
        with patch("urllib.request.urlopen", return_value=Response(b" " * 1_048_577)):
            self.assertEqual(self.request().reason, "oversized_response")

    def test_hard_call_and_time_budgets_do_not_send_requests(self):
        self.client.calls = MAX_CALLS
        with patch("urllib.request.urlopen") as mocked:
            self.assertEqual(self.request().reason, "call_budget")
            self.client.calls = 0
            self.client.deadline = 0
            self.assertEqual(self.request().reason, "time_budget")
            mocked.assert_not_called()

    def test_every_token_budget_within_organizer_cap(self):
        for phase in ("extraction", "synthesis"):
            for count in (1, 4, 10, 100):
                for attempt in (0, 1):
                    self.assertLessEqual(output_budget(phase, count, attempt), 4000)
        self.assertGreater(output_budget("extraction", 10), 550)
        with self.assertRaises(ValueError):
            self.client.request("x", {}, 4001, phase="test", attempt=1)

    def test_diagnostics_do_not_expose_secrets_or_raw_finish_reason(self):
        with patch("urllib.request.urlopen", side_effect=OSError("SECRET-ERROR")):
            self.request()
        with patch("urllib.request.urlopen", return_value=Response(envelope({}, "SECRET-FINISH"))):
            self.request()
        self.assertNotIn("SECRET", self.stderr.getvalue())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.u = unit()
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.stderr = io.StringIO()
        self.redirect = contextlib.redirect_stderr(self.stderr)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def pipeline(self, docs, replies):
        with patch("t2solution.text._read_documents", return_value=docs), patch(
            "urllib.request.urlopen", side_effect=[Response(reply) for reply in replies]
        ) as mocked:
            view = infer_text_view(self.u, "2024-01-01", {})
        return view, mocked

    def test_successful_pipeline_applies_only_existing_v02_caps(self):
        view, mocked = self.pipeline([document("one"), document("two")], [envelope(signal(self.u)),
                                     envelope(signal(self.u)), envelope(synthesis(self.u, ids=["one", "two"]))])
        self.assertTrue(view.used)
        self.assertEqual(view.call_count, 3)
        self.assertAlmostEqual(view.scenarios[0].drift_sd["EUR"], .128)
        self.assertAlmostEqual(view.scenarios[0].vol_mult["EUR"], 1.175)
        self.assertEqual(view.reliability["status"], "applied")
        first = json.loads(mocked.call_args_list[0].args[0].data)
        self.assertNotIn("_full_text", json.loads(first["messages"][1]["content"])["document"])

    def test_successful_text_path_reaches_cli_outputs_and_diagnostics(self):
        replies = [envelope(signal(self.u)), envelope(signal(self.u)),
                   envelope(synthesis(self.u, ids=["one", "two"]))]
        with tempfile.TemporaryDirectory() as temp, patch("t2solution.cli.load_unit", return_value=self.u), patch(
            "t2solution.text._read_documents", return_value=[document("one"), document("two")]
        ), patch("urllib.request.urlopen", side_effect=[Response(reply) for reply in replies]), contextlib.redirect_stdout(io.StringIO()):
            output = pathlib.Path(temp)
            code = forecast_main(["--panels", ".", "--text", ".", "--asof", "2024-01-01",
                                  "--out", str(output / "forecast.parquet"), "--n-draws", "200", "--seed", "19"])
            self.assertEqual(code, 0)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"forecast.parquet", "forecast_meta.json", "forecast_rationale.md"})
            meta = json.loads((output / "forecast_meta.json").read_text())
            self.assertEqual(meta["agent"]["text_diagnostics"]["status"], "applied")
            self.assertEqual(meta["agent"]["house_calls"], 3)
            self.assertEqual(meta["agent"]["text_mode"], "reliable")
            frame = pd.read_parquet(output / "forecast.parquet")
            self.assertTrue(np.isfinite(frame["value"]).all())
            self.assertEqual(len(frame), 200 * len(self.u.assets) * len(self.u.horizons))
            self.assertIn("Text reliability", (output / "forecast_rationale.md").read_text())
        self.assertNotIn("SECRET", self.stderr.getvalue())

    def test_deadline_reserve_skips_extraction_without_calling_house(self):
        client = HouseClient("http://mock", "test", "house", seconds=30)
        with patch("t2solution.text._read_documents", return_value=[document()]), patch(
            "t2solution.text.HouseClient", return_value=client
        ), patch("urllib.request.urlopen") as mocked:
            view = infer_text_view(self.u, "2024-01-01", {})
        self.assertFalse(view.used)
        self.assertEqual(view.call_count, 0)
        self.assertEqual(view.reliability["extraction_stop_reason"], "time_budget")
        self.assertEqual(view.reliability["documents_unprocessed"], 1)
        mocked.assert_not_called()

    def test_incomplete_extraction_gets_one_retry(self):
        bad = signal(self.u)
        del bad["per_asset"]["JPY"]
        view, _ = self.pipeline([document()], [envelope(bad), envelope(signal(self.u)),
                                               envelope(synthesis(self.u, ids=["doc1"]))])
        self.assertEqual(view.call_count, 3)
        self.assertEqual(view.reliability["call_outcomes"]["incomplete_signal_assets"], 1)
        self.assertTrue(view.used)

    def test_no_valid_extraction_does_not_synthesize_empty_evidence(self):
        view, mocked = self.pipeline([document()], [envelope({}), envelope({})])
        self.assertFalse(view.used)
        self.assertEqual(view.call_count, 2)
        self.assertEqual(view.reliability["status"], "extraction_failed")
        self.assertEqual(mocked.call_count, 2)

    def test_worst_case_repair_path_stays_below_25_calls(self):
        docs = [document(f"doc{i}") for i in range(10)]
        replies = []
        for _ in docs:
            replies.extend([envelope({}), envelope(signal(self.u))])
        replies.extend([envelope({}), envelope(synthesis(self.u, ids=["doc0", "doc1"]))])
        view, mocked = self.pipeline(docs, replies)
        self.assertEqual(view.call_count, 22)
        self.assertEqual(mocked.call_count, 22)
        self.assertTrue(view.used)

    def test_invalid_synthesis_retries_once_then_exact_neutral_view(self):
        view, _ = self.pipeline([document()], [envelope(signal(self.u)), envelope({}), envelope({})])
        self.assertFalse(view.used)
        self.assertEqual(view.call_count, 3)
        self.assertEqual(view.scenarios, TextView.neutral().scenarios)
        self.assertEqual(view.reliability["status"], "synthesis_failed")

    def test_all_neutral_cards_and_synthesis_keep_exact_numerical_fallback(self):
        view, _ = self.pipeline([document()], [envelope(signal(self.u, neutral=True)),
                                               envelope(synthesis(self.u, neutral=True))])
        self.assertFalse(view.used)
        self.assertEqual(view.scenarios, TextView.neutral().scenarios)
        self.assertEqual(view.reliability["status"], "valid_neutral_synthesis")

    def test_auth_failure_stops_entire_pipeline(self):
        with patch("t2solution.text._read_documents", return_value=[document(), document("two")]), patch(
            "urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 403, "body", {}, None)
        ) as mocked:
            view = infer_text_view(self.u, "2024-01-01", {})
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(view.used)

    def test_disabled_or_missing_house_credentials_never_connect(self):
        for extra, status in (({"T2_DISABLE_TEXT": "1"}, "disabled"),
                              ({"MODEL_TOKEN": ""}, "missing_house_access")):
            with patch.dict(os.environ, extra), patch("urllib.request.urlopen") as mocked:
                view = infer_text_view(self.u, "2024-01-01", {})
                self.assertEqual(view.reliability["status"], status)
                mocked.assert_not_called()

    def test_no_eligible_documents_never_connect(self):
        with patch("t2solution.text._read_documents", return_value=[]), patch("urllib.request.urlopen") as mocked:
            self.assertFalse(infer_text_view(self.u, "2024-01-01", {}).used)
            mocked.assert_not_called()

    def test_legacy_mode_delegates_without_changing_v02_pipeline(self):
        expected = legacy_text.TextView.neutral()
        with patch("t2solution.legacy_text.infer_text_view", return_value=expected) as mocked:
            self.assertIs(infer_text_view(self.u, "2024-01-01", {}, mode="legacy"), expected)
        mocked.assert_called_once_with(self.u, "2024-01-01", {})

    def test_neutral_samples_are_bit_exact_with_v02(self):
        states = prepare(self.u, "2024-01-01")
        old, _ = simulate(self.u, "2024-01-01", states, legacy_text.TextView.neutral(), 200, 19)
        new, _ = simulate(self.u, "2024-01-01", states, TextView.neutral(), 200, 19)
        np.testing.assert_array_equal(new, old)


if __name__ == "__main__":
    unittest.main()
