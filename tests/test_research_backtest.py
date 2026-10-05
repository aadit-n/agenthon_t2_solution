"""Check the causal benchmark's targets and calendar exclusions on synthetic inputs."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from research_backtest import pseudo_task, summarize
from test_candidate import fixture


class ResearchBacktestTests(unittest.TestCase):
    def test_pseudo_return_outcome_matches_official_log_units(self):
        unit = fixture(returns=True)
        frame = unit.panels['factors_daily']
        origin = frame.date.iloc[500]
        trial, actual = pseudo_task(unit, origin, np.array([[21, 63]]))
        future = frame[frame.date > origin].value
        np.testing.assert_allclose(actual, [np.log1p(future.iloc[:h]).sum() for h in (21, 63)])
        self.assertLessEqual(trial.panels['factors_daily'].date.max(), origin)

    def test_missing_multi_year_window_is_rejected(self):
        unit = fixture()
        frame = unit.panels['factors_daily']
        origin = frame.date.iloc[300]
        unit.panels['factors_daily'] = frame[(frame.index <= 300) | (frame.index >= 450)]
        with self.assertRaisesRegex(ValueError, 'unavailable gap'):
            pseudo_task(unit, origin, np.array([[21, 63]]))

    def test_monthly_grid_uses_new_explicit_origin_relative_periods(self):
        unit = fixture(monthly=True)
        frame = unit.panels['macro_monthly']
        origin = frame.date.iloc[100]
        trial, actual = pseudo_task(unit, origin, np.array([[8, 9]]))
        questions = trial.spec['questions']
        self.assertEqual([q['observation_period'] for q in questions],
                         [str(origin.to_period('M') + h) for h in (8, 9)])
        np.testing.assert_allclose(actual, frame.value.iloc[[108, 109]])

    def test_normalization_clips_whole_composite_not_terms(self):
        base = {'group': 0, 'origin': '2001-01-01', 'split': 'holdout', 'coverage90': 0.9}
        frame = pd.DataFrame([
            {**base, 'profile': 'm0', 'marginal': 1, 'joint': 1, 'pinball': 1, 'coverage_tail': 1},
            {**base, 'profile': 'v2', 'marginal': 5, 'joint': 0.5, 'pinball': 0.5, 'coverage_tail': 0.5},
        ])
        result = summarize(frame)
        self.assertAlmostEqual(result[result.profile == 'v2'].local_loss.iloc[0], 2.75)


if __name__ == '__main__':
    unittest.main()
