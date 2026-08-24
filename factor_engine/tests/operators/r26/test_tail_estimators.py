# -*- coding: utf-8 -*-
"""R26-060..071: tail / spread estimator numerical semantics.

* Hill lower tail on a strictly-positive level is unsupported (NaN);
* Hill on downside losses stays finite positive;
* Roll spread with positive covariance is NaN (undefined), not 0;
* Roll rejects non-positive price columns.
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.cleaned_operators.extreme_tail import _hill_series
from factor_engine.cleaned_operators.spread_estimators import _roll_spread_series


def test_hill_upper_pareto_positive():
    rng = np.random.default_rng(0)
    x = (1.0 / rng.uniform(size=3000)) ** 1.5
    xi = _hill_series(x, 500, "upper", 0.2, 20)
    assert np.nanmedian(xi) > 0.0


def test_hill_downside_loss_positive():
    rng = np.random.default_rng(1)
    x = -((1.0 / rng.uniform(size=3000)) ** 1.5)  # negative returns
    xi = _hill_series(x, 500, "lower", 0.2, 20)
    assert np.nanmedian(xi) > 0.0


def test_hill_positive_level_lower_tail_unsupported():
    rng = np.random.default_rng(2)
    x = rng.uniform(100.0, 200.0, size=3000)  # strictly-positive level
    xi = _hill_series(x, 500, "lower", 0.2, 20)
    assert np.all(np.isnan(xi)), "R26-061: positive-level lower tail must be unsupported"


def test_roll_positive_covariance_is_nan():
    rng = np.random.default_rng(3)
    n = 300
    # prices whose log-returns have POSITIVE autocovariance -> Roll model invalid
    r = rng.normal(0.0, 0.01, n)
    r[1:] += 0.5 * r[:-1]  # positive AR(1) -> cov(dx, lag dx) > 0
    price = 100.0 * np.exp(np.cumsum(r))
    out = _roll_spread_series(price[:, None], 60, 20)
    assert np.isnan(out[-1, 0]), "R26-069: positive covariance -> undefined -> NaN"


def test_roll_negative_covariance_finite():
    rng = np.random.default_rng(4)
    n = 300
    r = rng.normal(0.0, 0.01, n)
    r[1:] += -0.5 * r[:-1]  # negative AR(1) -> valid Roll
    price = 100.0 * np.exp(np.cumsum(r))
    out = _roll_spread_series(price[:, None], 60, 20)
    assert np.isfinite(out[-1, 0]) and out[-1, 0] > 0.0


def test_roll_rejects_non_positive_price():
    price = np.array([100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 100.0])[:, None]
    out = _roll_spread_series(price, 5, 3)
    assert np.all(np.isnan(out)), "R26-071: non-positive price column fails closed"
