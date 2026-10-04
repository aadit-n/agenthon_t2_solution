from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from t2solution.model import _historical_width, _nearest_correlation
from t2solution.text import _url


class CoreTests(unittest.TestCase):
    def test_house_endpoint_origin_gets_v1(self) -> None:
        self.assertEqual(_url("http://house:8000"), "http://house:8000/v1/chat/completions")
        self.assertEqual(_url("http://house:8000/v1"), "http://house:8000/v1/chat/completions")

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
