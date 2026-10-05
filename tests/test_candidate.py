"""Regression checks for researched candidates, without outcome or network access."""

from __future__ import annotations

import copy
import pathlib
import unittest

import numpy as np
import pandas as pd

from t2solution import candidate
from t2solution.model import prepare as legacy_prepare, simulate as legacy_simulate
from t2solution.text import TextView
from t2solution.unit import Unit


def fixture(monthly: bool = False, returns: bool = False) -> Unit:
    dates = pd.date_range('2000-01-01', periods=160 if monthly else 640,
                          freq='MS' if monthly else 'B')
    rng = np.random.default_rng(7)
    values = (0.002 + 0.02 * rng.standard_normal(len(dates)) if returns
              else 100 + np.cumsum(0.6 + 0.2 * rng.standard_normal(len(dates))))
    frame = pd.DataFrame({'date': dates, 'asset': 'X', 'value': values})
    targets = {'asset_ids': ['X'], 'horizons': [140, 160] if monthly else [21, 63]}
    if monthly:
        targets['observation_periods'] = [str(dates[-1].to_period('M') + n) for n in (8, 9)]
    return Unit(pathlib.Path('.'), {'targets': targets}, {}, {'macro_monthly' if monthly else 'factors_daily': frame},
                pathlib.Path('missing-text'), 'synthetic', 'T2-F1', ['X'], targets['horizons'],
                'log_return' if returns else 'level', 'monthly' if monthly else 'daily', 'native', 200)


class CandidateTests(unittest.TestCase):
    def test_returns_are_log1p_not_raw_simple_returns(self):
        unit = fixture(returns=True)
        cutoff = str(unit.panels['factors_daily'].date.max().date())
        states = candidate.prepare(unit, cutoff)
        source = unit.panels['factors_daily'].set_index('date').value
        np.testing.assert_allclose(states[0].steps.to_numpy(), np.log1p(source.iloc[1:]).to_numpy())
        self.assertFalse(np.allclose(states[0].steps, legacy_prepare(unit, cutoff)[0].steps))

    def test_return_conversion_never_mutates_input(self):
        unit = fixture(returns=True)
        original = unit.panels['factors_daily'].copy(deep=True)
        candidate.prepare(unit, str(original.date.max().date()))
        pd.testing.assert_frame_equal(unit.panels['factors_daily'], original)

    def test_invalid_simple_returns_refused(self):
        unit = fixture(returns=True)
        unit.panels['factors_daily'].loc[20, 'value'] = -1.0
        with self.assertRaises(ValueError):
            candidate.prepare(unit, str(unit.panels['factors_daily'].date.max().date()))

    def test_monthly_drift_uses_actual_months(self):
        unit = fixture(monthly=True)
        cutoff = str(unit.panels['macro_monthly'].date.max().date())
        state = candidate.prepare(unit, cutoff)[0]
        self.assertGreater(state.drift_step, 0.1)
        self.assertEqual(legacy_prepare(unit, cutoff)[0].drift_step, 0.0)

    def test_copula_postprocessing_preserves_joint_ranks(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(200, 2, 3))
        target = rng.normal(1, 2, size=source.shape)
        result = candidate.quantile_blend(source, target, 0.2)
        np.testing.assert_array_equal(np.argsort(result, axis=0), np.argsort(source, axis=0))
        expected = 0.8 * np.sort(source, axis=0) + 0.2 * np.sort(target, axis=0)
        np.testing.assert_allclose(np.sort(result, axis=0), expected)

    def test_zero_postprocessing_is_bit_identical(self):
        source = np.random.default_rng(2).normal(size=(200, 1, 2))
        np.testing.assert_array_equal(candidate.quantile_blend(source, -source, 0), source)

    def test_future_rows_do_not_affect_candidate(self):
        unit = fixture(returns=True)
        cutoff = str(unit.panels['factors_daily'].date.iloc[500].date())
        other = copy.deepcopy(unit)
        other.panels['factors_daily'].loc[other.panels['factors_daily'].date > cutoff, 'value'] = 1000
        for left, right in zip(candidate.prepare(unit, cutoff), candidate.prepare(other, cutoff), strict=True):
            pd.testing.assert_series_equal(left.steps, right.steps)

    def test_rollback_and_daily_corrected_profiles_preserve_incumbent_samples(self):
        unit = fixture()
        cutoff = str(unit.panels['factors_daily'].date.max().date())
        expected, _ = legacy_simulate(unit, cutoff, legacy_prepare(unit, cutoff), TextView.neutral(), 200, 3)
        for profile in ('v2', 'corrected'):
            states = candidate.prepare(unit, cutoff, profile)
            actual, _ = candidate.simulate(unit, cutoff, states, TextView.neutral(), 200, 3, profile)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
