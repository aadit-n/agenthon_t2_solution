"""Validation-tool tests: synthetic histories and mocked failures, never sealed outcomes."""

from __future__ import annotations

import copy
import math
import pathlib
import sys
from statistics import NormalDist
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from t2solution import model
from t2solution.text import TextView
from t2solution.unit import Unit
from validation.cases import build_window, choose_origins, split_for, step_counts
from validation.metrics import expected_scales, gaussian_reference, score_samples
from validation.reporting import compare_frames, summarize
from validation.runner import run_agent


def synthetic(*, monthly=False, returns=False, assets=None, horizons=None, periods=None):
    assets = assets or ["Y", "X"]
    horizons = horizons or ([140, 160] if monthly else [5, 21])
    n = periods or (240 if monthly else 6000)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS") if monthly else pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(17)
    frames = []
    for asset in assets:
        steps = rng.normal(.0001, .01, n)
        values = steps if returns else 2 + steps.cumsum()
        frames.append(pd.DataFrame({"date": dates, "asset": asset, "value": values}))
    panel = "macro_monthly" if monthly else "rates_daily"
    card = {"task": {"id": "synthetic-future-title-2024", "title": "Future event"},
            "metadata": {"category": "T2-F3", "description": "Future information"},
            "provenance": {"data_cutoff": str(dates[-1].date())},
            "panels": {panel: {"asset_ids": assets}}, "targets": {},
            "scoring": {"params": {"joint": "variogram", "tail_metric": "pinball"}}}
    if monthly:
        start = dates[-1].to_period("M")
        card["targets"]["observation_periods"] = [str(start + 8), str(start + 9)]
    return Unit(pathlib.Path("unavailable-real-root"), card, {}, {panel: pd.concat(frames)},
                pathlib.Path("unavailable-text"), "synthetic-future-title-2024", "T2-F3", assets,
                horizons, "log_return" if returns else "level", "monthly" if monthly else "daily", "native", 200)


class WindowTests(unittest.TestCase):
    def test_training_prefix_is_cut_and_future_labels_are_separate(self):
        u = synthetic(periods=600)
        origin = pd.Timestamp("2001-07-02")
        window = build_window(u, origin, step_counts(u), min_train=300)
        self.assertTrue(all(frame.date.max() <= origin for frame in window.trial.panels.values()))
        self.assertTrue(all(pd.Timestamp(d) > origin for d in window.target_dates))
        self.assertLessEqual(pd.Timestamp(window.label_end), pd.Timestamp(u.card["provenance"]["data_cutoff"]))
        self.assertNotIn("description", window.trial.card["metadata"])
        self.assertNotIn("title", window.trial.card["task"])
        self.assertNotEqual(window.trial.unit_id, u.unit_id)

    def test_future_changes_cannot_change_training_or_reference(self):
        u = synthetic(periods=600)
        origin = pd.Timestamp("2001-07-02")
        counts = step_counts(u)
        before = build_window(u, origin, counts, min_train=300)
        changed = copy.deepcopy(u)
        for frame in changed.panels.values():
            frame.loc[frame.date > origin, "value"] *= 10
        after = build_window(changed, origin, counts, min_train=300)
        for panel in before.trial.panels:
            pd.testing.assert_frame_equal(before.trial.panels[panel], after.trial.panels[panel])
        p, q = gaussian_reference(before.trial, counts), gaussian_reference(after.trial, counts)
        np.testing.assert_array_equal(p.mean, q.mean)
        np.testing.assert_array_equal(p.covariance, q.covariance)
        self.assertFalse(np.array_equal(before.actual, after.actual))

    def test_return_targets_are_sum_log1p_not_raw_sum(self):
        u = synthetic(returns=True, assets=["X"], horizons=[5], periods=600)
        origin = pd.Timestamp("2001-07-02")
        window = build_window(u, origin, step_counts(u), min_train=300)
        future = u.panels["rates_daily"].query("date > @origin")["value"].iloc[:5]
        self.assertAlmostEqual(window.actual[0], np.log1p(future).sum())
        self.assertNotAlmostEqual(window.actual[0], future.sum(), places=8)

    def test_missing_target_values_or_long_gaps_are_rejected(self):
        u = synthetic(assets=["X"], horizons=[21], periods=600)
        origin = pd.Timestamp("2001-07-02")
        frame = u.panels["rates_daily"]
        u.panels["rates_daily"] = frame[(frame.date <= origin) | (frame.date >= origin + pd.Timedelta(days=30))]
        with self.assertRaisesRegex(ValueError, "future_gap"):
            build_window(u, origin, np.array([[21]]), min_train=300)

    def test_labels_must_fit_inside_supplied_real_cutoff(self):
        u = synthetic(assets=["X"], horizons=[21], periods=600)
        origin = u.panels["rates_daily"].date.max()
        with self.assertRaisesRegex(ValueError, "insufficient_future"):
            build_window(u, origin, np.array([[21]]), min_train=300)

    def test_monthly_leads_and_authored_labels_are_preserved(self):
        u = synthetic(monthly=True)
        counts = step_counts(u)
        np.testing.assert_array_equal(counts, [[8, 9], [8, 9]])
        origin = pd.Timestamp("2010-01-01")
        window = build_window(u, origin, counts, min_train=60)
        self.assertEqual(window.trial.horizons, [140, 160])
        np.testing.assert_array_equal(step_counts(window.trial), counts)
        self.assertEqual(window.target_dates[:2], ["2010-09-01", "2010-10-01"])

    def test_missing_month_is_not_treated_as_one_month_step(self):
        u = synthetic(monthly=True, assets=["X"])
        frame = u.panels["macro_monthly"]
        u.panels["macro_monthly"] = frame[frame.date != pd.Timestamp("2010-03-01")]
        with self.assertRaisesRegex(ValueError, "future_gap"):
            build_window(u, pd.Timestamp("2010-01-01"), np.array([[8, 9]]), min_train=60)

    def test_split_is_global_chronological_and_windows_cannot_cross_it(self):
        self.assertEqual(split_for("2011-01-01", "2012-12-31"), "tune")
        self.assertEqual(split_for("2013-01-01", "2017-12-31"), "holdout_1")
        self.assertEqual(split_for("2018-01-01", "2024-12-31"), "holdout_2")
        self.assertIsNone(split_for("2012-12-01", "2013-01-02"))
        self.assertIsNone(split_for("2017-12-01", "2018-01-02"))

    def test_chosen_origins_are_nonoverlapping_and_embargoed(self):
        u = synthetic()
        cases, exclusions = choose_origins(u, step_counts(u), per_split=4, embargo_days=5)
        self.assertEqual({c["split"] for c in cases}, {"tune", "holdout_1", "holdout_2"})
        previous = None
        for row in sorted(cases, key=lambda c: c["origin"]):
            if previous:
                self.assertGreater(pd.Timestamp(row["origin"]), pd.Timestamp(previous["label_end"]) + pd.offsets.BDay(5))
            self.assertEqual(split_for(row["origin"], row["label_end"]), row["split"])
            previous = row


class MetricTests(unittest.TestCase):
    def test_expected_scales_have_no_realized_outcome_argument(self):
        scales = expected_scales(np.array([0.0]), np.array([[.01]]), (.01, .05, .95, .99))
        self.assertAlmostEqual(scales["marginal"], .1 / np.sqrt(np.pi))
        z = [NormalDist().inv_cdf(q) for q in (.01, .05, .95, .99)]
        expected_tail = .1 * np.mean([math.exp(-value**2 / 2) / math.sqrt(2 * math.pi) for value in z])
        self.assertAlmostEqual(scales["tail"], expected_tail, places=13)
        self.assertEqual(scales["joint"], 1.0)

    def test_gaussian_joint_scale_matches_independent_monte_carlo(self):
        mean, cov = np.array([0.0, .4]), np.array([[.1, .04], [.04, .3]])
        scale = expected_scales(mean, cov, (.01, .05, .95, .99))["joint"]
        rng = np.random.default_rng(23)
        values = rng.multivariate_normal(mean, cov, 300_000)
        mc = 2 * np.var(np.sqrt(np.abs(values[:, 0] - values[:, 1])))
        self.assertAlmostEqual(scale, mc, delta=scale * .012)

    def test_scoring_delegates_to_organizer_and_handles_single_cell_weights(self):
        u = synthetic(assets=["X"], horizons=[5], periods=600)
        samples = np.random.default_rng(5).normal(size=(200, 1, 1))
        actual, scales = np.array([.3]), {"marginal": 1.0, "joint": 1.0, "tail": 1.0}
        from qfbench2_track_forecasting.scoring import _composite
        expected = _composite(samples.reshape(200, 1), actual, weights=(5/7, 0, 2/7),
                              tail_levels=(.01, .05, .95, .99), joint="variogram", tail_metric="pinball", ref_scale=scales)
        result = score_samples(samples, actual, u, scales)
        self.assertAlmostEqual(result["unclipped_loss"], expected["composite"])
        self.assertEqual(result["joint_weight"], 0)

    def test_reference_uses_only_trailing_history_and_path_covariance(self):
        u = synthetic(assets=["X"], periods=600)
        reference = gaussian_reference(u, np.array([[5, 21]]))
        self.assertEqual(reference.mean.shape, (2,))
        self.assertAlmostEqual(reference.covariance[0, 1] / (reference.covariance[0, 0] - 1.1e-9), 1.0)
        changed = copy.deepcopy(u)
        changed.panels["rates_daily"].loc[:100, "value"] += 100
        other = gaussian_reference(changed, np.array([[5, 21]]))
        np.testing.assert_array_equal(reference.mean, other.mean)
        np.testing.assert_array_equal(reference.covariance, other.covariance)

    def test_extreme_native_quote_differences_have_finite_positive_joint_scale(self):
        scales = expected_scales(np.array([1.0, 150.0]), np.array([[.0001, .0], [.0, .01]]), (.01, .05, .95, .99))
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in scales.values()))

    def test_nonfinite_samples_fail_not_get_dropped(self):
        u = synthetic(assets=["X"], horizons=[5], periods=600)
        with self.assertRaises(ValueError):
            score_samples(np.full((200, 1, 1), np.nan), np.array([.3]), u,
                          {"marginal": 1., "joint": 1., "tail": 1.})

    def test_one_exact_engine_call_not_200_draw_batches(self):
        u = synthetic(periods=600)
        window = build_window(u, pd.Timestamp("2001-07-02"), step_counts(u), min_train=300)
        states = model.prepare(window.trial, "2001-07-02")
        expected, _ = model.simulate(window.trial, "2001-07-02", states, TextView.neutral(), 200, 19)
        with patch("validation.runner.model.simulate", wraps=model.simulate) as mocked:
            actual = run_agent(window.trial, "2001-07-02", 200, 19)
        mocked.assert_called_once()
        np.testing.assert_array_equal(actual, expected)


def rows(losses=(1.0, 2.0), seeds=(0, 19)):
    output = []
    for year in range(2000, 2024, 2):
        for group, loss in enumerate(losses):
            for seed in seeds:
                output.append({"case_id": f"{group}-{year}", "group_id": str(group), "unit": "synthetic",
                               "origin": f"{year}-01-03", "label_end": f"{year}-02-01",
                               "split": "tune" if year < 2013 else ("holdout_1" if year < 2018 else "holdout_2"),
                               "family": "T2-F1", "target_type": "level", "frequency": "daily",
                               "tier": "history_proxy", "seed": seed, "engine": "agent", "status": "ok",
                               "local_loss": loss, "unclipped_loss": loss,
                               "marginal_normalized": loss, "joint_normalized": loss, "tail_normalized": loss,
                               "coverage90": .9, "clip_applied": False})
    return pd.DataFrame(output)


class ReportingTests(unittest.TestCase):
    def test_seeds_are_not_independent_historical_cases(self):
        report = summarize(rows())
        self.assertEqual(report["rows"], 48)
        self.assertEqual(report["historical_cases"], 24)
        self.assertAlmostEqual(report["overall"][0]["local_loss"], 1.5)

    def test_group_weighting_does_not_favor_repeated_origins(self):
        frame = rows()
        frame = pd.concat([frame, frame[frame.group_id == "0"].assign(case_id=lambda f: f.case_id + "-copy")])
        self.assertAlmostEqual(summarize(frame)["overall"][0]["local_loss"], 1.5)

    def test_failures_stay_in_denominator_at_cap(self):
        frame = rows(losses=(1.0,))
        frame.loc[frame.index[:2], ["status", "local_loss"]] = ["engine_failed", 8.0]
        report = summarize(frame)
        self.assertEqual(report["failures"], 2)
        self.assertGreater(report["overall"][0]["local_loss"], 1.0)

    def test_paired_comparison_rejects_missing_or_duplicate_cases(self):
        base = rows()
        for bad in (base.iloc[1:], pd.concat([base, base.iloc[:1]])):
            with self.assertRaises(ValueError):
                compare_frames(base, bad, bootstrap=100, seed=19)

    def test_identical_runs_cannot_be_called_an_improvement(self):
        base = rows()
        report = compare_frames(base, base.copy(), bootstrap=100, seed=19)
        self.assertFalse(report["promotion_gate_passed"])
        self.assertEqual(report["overall_delta"], 0.0)

    def test_block_interval_does_not_treat_extra_seeds_as_more_dates(self):
        base, candidate = rows(), rows(losses=(.8, 1.8))
        a = compare_frames(base, candidate, bootstrap=100, seed=19)
        b = compare_frames(rows(seeds=(0, 19, 73, 101)), rows(losses=(.8, 1.8), seeds=(0, 19, 73, 101)),
                           bootstrap=100, seed=19)
        self.assertEqual(a["holdout_block_interval"], b["holdout_block_interval"])
        self.assertAlmostEqual(a["overall_delta"], -.2)


if __name__ == "__main__":
    unittest.main()
