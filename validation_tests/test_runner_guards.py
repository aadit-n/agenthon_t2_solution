"""Fail-closed provenance, result-integrity, environment and failure-accounting checks."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "validation_tests")]

from test_validation import rows, synthetic
from compare_validation import load_run
from validation.cases import build_window, step_counts
from validation.metrics import CAP, score_samples
from validation.runner import digest, execute, file_hash, input_files, neutral_environment, run_agent, safe_output, verify_plan
from validation.reporting import compare_frames


class GuardTests(unittest.TestCase):
    def test_output_outside_ignored_backtests_is_refused(self):
        for out in (ROOT / "src", ROOT / "submissions" / "new-validation", ROOT / "backtest_runs"):
            with self.assertRaises(ValueError):
                safe_output(out)

    def test_existing_output_is_not_overwritten(self):
        (ROOT / "backtest_runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "backtest_runs") as temp:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                safe_output(pathlib.Path(temp))

    def test_credentials_and_hyperparameter_overrides_are_not_used_and_are_restored(self):
        settings = {"MODEL_TOKEN": "private-test-value", "MODEL_ENDPOINT": "http://no-request",
                    "T2_M0_WEIGHT": ".95", "T2_TEXT_MODE": "legacy"}
        with patch.dict(os.environ, settings):
            with neutral_environment():
                self.assertNotIn("MODEL_TOKEN", os.environ)
                self.assertNotIn("T2_M0_WEIGHT", os.environ)
                self.assertEqual(os.environ["T2_DISABLE_TEXT"], "1")
            for key, value in settings.items():
                self.assertEqual(os.environ[key], value)

    def test_source_panel_mutation_cannot_affect_the_next_seed(self):
        u = synthetic(periods=600)
        window = build_window(u, pd.Timestamp("2001-07-02"), step_counts(u), min_train=300)
        before = copy.deepcopy(window.trial.panels)
        def mutating_prepare(trial, asof):
            next(iter(trial.panels.values()))["value"] = 999
            return []
        with patch("validation.runner.model.prepare", side_effect=mutating_prepare), patch(
            "validation.runner.model.simulate", return_value=(np.zeros((200, 2, 2)), {"house_calls": 0, "text_used": False})
        ):
            run_agent(window.trial, "2001-07-02", 200, 0)
        for name in before:
            pd.testing.assert_frame_equal(before[name], window.trial.panels[name])

    def test_plan_hash_and_input_changes_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = pathlib.Path(temp)
            root = repo / "units" / "unit"
            root.mkdir(parents=True)
            (root / "card.toml").write_text("schema_version = '2.0'", encoding="utf-8")
            body = {"format_version": 1, "sources": {"unit": {"files": input_files(root)}},
                    "cases": [{"case_id": "case"}]}
            plan = {**body, "plan_sha256": digest(body)}
            verify_plan(plan, repo)
            changed = copy.deepcopy(plan)
            changed["cases"][0]["case_id"] = "tampered"
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_plan(changed, repo)
            (root / "card.toml").write_text("schema_version = 'different'", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                verify_plan(plan, repo)

    def test_completed_result_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp)
            pd.concat([rows(), rows().assign(engine="m0")]).to_parquet(out / "results.parquet", index=False)
            count = len(pd.read_parquet(out / "results.parquet"))
            meta = {"state": "complete", "rows": count, "expected_rows": count,
                    "results_sha256": file_hash(out / "results.parquet")}
            (out / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
            load_run(out)
            rows().to_parquet(out / "results.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "modified"):
                load_run(out)

    def test_partial_run_cannot_be_presented_as_a_complete_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp)
            (out / "manifest.json").write_text(json.dumps({"state": "running", "rows": 1, "expected_rows": 10}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial"):
                load_run(out)

    def test_missing_family_coverage_prevents_promotion_even_with_positive_evidence(self):
        base, candidate = rows(seeds=(0, 19, 73)), rows(losses=(.8, 1.8), seeds=(0, 19, 73))
        report = compare_frames(base, candidate, bootstrap=100, seed=19)
        self.assertLess(report["holdout_block_interval"]["high"], 0)
        self.assertFalse(report["coverage_gate_passed"])
        self.assertFalse(report["promotion_gate_passed"])

    def test_score_clips_whole_composite_not_individual_components(self):
        u = synthetic(assets=["X"], horizons=[5], periods=600)
        result = score_samples(np.zeros((200, 1, 1)), np.array([1000.]), u,
                               {"marginal": 1., "joint": 1., "tail": 1.})
        self.assertGreater(result["unclipped_loss"], CAP)
        self.assertEqual(result["local_loss"], CAP)
        self.assertTrue(result["clip_applied"])

    def test_executed_engine_failure_counts_for_every_seed(self):
        u = synthetic(periods=600)
        counts = step_counts(u)
        w = build_window(u, pd.Timestamp("2001-07-02"), counts, min_train=300)
        from validation.metrics import gaussian_reference, expected_scales
        ref = gaussian_reference(w.trial, counts)
        case = {"case_id": "1" * 64, "group_id": "group", "unit": "synthetic", "origin": "2001-07-02",
                "label_end": w.label_end, "target_dates": w.target_dates, "split": "tune", "family": "T2-F3",
                "target_type": "level", "frequency": "daily", "tier": "history_proxy", "counts": counts.tolist(),
                "scales": expected_scales(ref.mean, ref.covariance)}
        plan = {"plan_sha256": "test", "cases": [case], "sources": {"synthetic": {"cutoff": u.card["provenance"]["data_cutoff"]}}}
        with tempfile.TemporaryDirectory() as temp, patch("validation.runner.load_unit", return_value=u), patch(
            "validation.runner.run_agent", side_effect=ValueError("sensitive text must not be logged")
        ):
            out = pathlib.Path(temp)
            report = execute(plan, ROOT, out, draws=200, seeds=[0, 19], identity={})
            frame = pd.read_parquet(out / "results.parquet")
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["failures"], 2)
            self.assertTrue((frame[frame.engine == "agent"].local_loss == CAP).all())
            self.assertNotIn("sensitive text", (out / "results.partial.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
