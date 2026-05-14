"""
Tests rápidos para los módulos de bench (sin correr el sweep completo).
"""
from __future__ import annotations

import math
import time
import unittest

import numpy as np

from bench.regression import bootstrap_ci, fit_exponential_in_k, fit_polynomial
from bench.timing import run_with_timeout, time_call


class TestTiming(unittest.TestCase):
    def test_time_call_basic(self):
        def fn(x):
            time.sleep(0.005)
            return 42

        res = time_call(fn, instance=None, warmup=1, repeats=3, timeout_s=2.0)
        self.assertFalse(res.timed_out)
        self.assertEqual(res.last_value, 42)
        self.assertGreater(res.median_s, 0.003)
        self.assertLess(res.median_s, 0.5)

    def test_run_with_timeout(self):
        def fast():
            return 7

        def slow():
            time.sleep(2.0)
            return 7

        ok, val = run_with_timeout(fast, timeout_s=1.0)
        self.assertTrue(ok)
        self.assertEqual(val, 7)

        ok2, _ = run_with_timeout(slow, timeout_s=0.2)
        self.assertFalse(ok2)


class TestRegression(unittest.TestCase):
    def test_fit_polynomial(self):
        x = np.arange(10, dtype=float)
        y = 3.0 * x + 5.0
        fit = fit_polynomial(x, y, degree=1)
        self.assertAlmostEqual(fit.coeffs[0], 3.0, places=6)
        self.assertAlmostEqual(fit.coeffs[1], 5.0, places=6)
        self.assertGreater(fit.r2, 0.999)

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(0)
        x = np.arange(40, dtype=float)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, size=40)
        boot = bootstrap_ci(x, y, degree=1, n_resamples=200, seed=0)
        self.assertLess(boot.coeffs_ci_low[0], 2.0)
        self.assertGreater(boot.coeffs_ci_high[0], 2.0)

    def test_fit_exponential_in_k(self):
        k = np.arange(5, 15, dtype=float)
        t = np.exp(0.7 * k + 0.1)
        fit = fit_exponential_in_k(k, t)
        self.assertAlmostEqual(fit.coeffs[0], 0.7, places=2)


if __name__ == "__main__":
    unittest.main()
