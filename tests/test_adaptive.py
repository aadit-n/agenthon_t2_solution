"""Adaptive numerical experts must be cutoff-safe, coherent and reproducible."""

from __future__ import annotations

import dataclasses
import pathlib
import unittest

import numpy as np
import pandas as pd

from t2solution.adaptive import _CACHE, _experts, _training_loss, fit, sample
from t2solution.model import prepare, simulate, simulate_adaptive, simulate_incumbent
from t2solution.text import TextView
from t2solution.unit import Unit


def fixture(n=1800, returns=False):
    rng = np.random.default_rng(101)
    dates = pd.bdate_range("2000-01-03", periods=n)
    shocks = rng.normal(0, .03 if not returns else .008, (n, 2))
    shocks[:, 1] = .8 * shocks[:, 0] + .6 * shocks[:, 1]
    values = shocks if returns else 4 + np.cumsum(shocks, axis=0)
    frame = pd.DataFrame({"date": np.repeat(dates, 2), "asset": np.tile(["A", "B"], n), "value": values.reshape(-1)})
    return Unit(pathlib.Path("not-readable"), {}, {}, {"rates": frame}, pathlib.Path("not-readable"),
                "fixture", "T2-F3", ["A", "B"], [21, 63], "log_return" if returns else "level",
                "daily", "decimal_return" if returns else "percent_per_annum", 200), str(dates[-1].date())


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        _CACHE.clear()

    def test_completed_selection_labels_precede_cutoff(self):
        unit, cutoff = fixture()
        experts, weights, audit = fit(unit, cutoff)
        self.assertLessEqual(audit["last_training_label"], cutoff)
        self.assertLess(audit["last_training_origin"], audit["last_training_label"])
        self.assertGreaterEqual(audit["completed_nonoverlapping_origins"], 4)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue((weights > 0).all())
        for expert in experts:
            self.assertGreater(np.linalg.eigvalsh(expert.covariance).min(), 0)

    def test_future_rows_cannot_change_fit_or_samples(self):
        full, _ = fixture()
        cutoff = "2005-01-03"
        truncated = dataclasses.replace(full, panels={"rates": full.panels["rates"].loc[lambda f: f.date <= cutoff].copy()})
        a, b = fit(full, cutoff), fit(truncated, cutoff)
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(sample(a, 1000, 19)[0], sample(b, 1000, 19)[0])
        self.assertEqual(a[2]["input_fingerprint"], b[2]["input_fingerprint"])

    def test_seed_repeatability_and_native_return_transform(self):
        unit, cutoff = fixture(returns=True)
        fitted = fit(unit, cutoff)
        self.assertTrue(fitted[2]["native_log_returns"])
        a, _, _ = sample(fitted, 1000, 19)
        b, _, _ = sample(fitted, 1000, 19)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (1000, 4))
        self.assertTrue(np.isfinite(a).all())

    def test_daily_simple_returns_are_not_summed_as_logs(self):
        raw = np.full((800, 1), .01)
        raw[:, 0] += np.random.default_rng(7).normal(0, .0001, 800)
        expert = _experts(raw, np.array([21]), True, False)[0]
        expected = np.log1p(raw[-299:]).mean() * 21
        self.assertAlmostEqual(expert.mean[0], expected, places=12)
        self.assertGreater(abs(expert.mean[0] - raw[-299:].mean() * 21), .0005)

    def test_short_monthly_and_gapped_histories_fallback(self):
        unit, cutoff = fixture(400)
        self.assertIsNone(fit(unit, cutoff))
        unit, cutoff = fixture()
        self.assertIsNone(fit(dataclasses.replace(unit, target_frequency="monthly"), cutoff))
        frame = unit.panels["rates"].copy()
        frame = frame.loc[~frame.date.between("2002-01-01", "2002-03-01")]
        self.assertIsNone(fit(dataclasses.replace(unit, panels={"rates": frame}), cutoff))

    def test_gaussian_training_loss_prefers_correct_center(self):
        experts = _experts(fixture()[0].panels["rates"].pivot(index="date", columns="asset", values="value").to_numpy(), np.array([21]), False, True)
        correct = experts[1]
        wrong = dataclasses.replace(correct, mean=correct.mean + 5)
        scale = np.sqrt(np.diag(correct.covariance))
        self.assertLess(_training_loss(correct, correct.mean, scale), _training_loss(wrong, correct.mean, scale))

    def test_simulator_shape_fallback_and_text_overlay(self):
        short, cutoff = fixture(400)
        states = prepare(short, cutoff)
        a, _ = simulate_adaptive(short, cutoff, states, TextView.neutral(), 500, 19)
        b, _ = simulate_incumbent(short, cutoff, states, TextView.neutral(), 500, 19)
        np.testing.assert_array_equal(a, b)
        unit, cutoff = fixture()
        states = prepare(unit, cutoff)
        neutral, diagnostics = simulate_adaptive(unit, cutoff, states, TextView.neutral(), 500, 19)
        view = TextView.neutral()
        view.scenarios[0].drift_sd = {"A": .2, "B": .2}
        shifted, _ = simulate_adaptive(unit, cutoff, states, view, 500, 19)
        self.assertEqual(neutral.shape, (500, 2, 2))
        self.assertTrue(np.isfinite(neutral).all())
        self.assertTrue((shifted - neutral > 0).all())
        self.assertEqual(diagnostics["numerical_engine"], "cutoff_adaptive_experimental")

    def test_default_remains_bit_exact_incumbent_even_with_long_history(self):
        unit, cutoff = fixture()
        states = prepare(unit, cutoff)
        a, _ = simulate(unit, cutoff, states, TextView.neutral(), 2000, 19)
        b, _ = simulate_incumbent(unit, cutoff, states, TextView.neutral(), 2000, 19)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
