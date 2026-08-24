# -*- coding: utf-8 -*-
"""R26-122..125: DMD log-domain numerical stability.

* log_rho = 2·log|λ| (never abs(λ)² which overflows);
* log_b2 = 2·log|b| (never abs_b² which underflows);
* huge lambda -> finite; tiny lambda -> 0; rho=0 -> 0;
* all-zero-energy modes -> explicit fail-closed.
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.cleaned_operators.dmd import _log_finite_horizon_sum


def test_log_sum_huge_lambda_finite():
    # |λ| ~ e^350 -> rho = |λ|² overflows float64; the log-domain form stays finite.
    assert np.isfinite(_log_finite_horizon_sum(700.0, 10))


def test_log_sum_tiny_lambda_is_zero():
    assert _log_finite_horizon_sum(-800.0, 10) == pytest.approx(0.0)


def test_log_sum_unit_lambda():
    # |λ| = 1 -> sum_{t=0}^{K-1} 1 = K
    assert _log_finite_horizon_sum(0.0, 10) == pytest.approx(np.log(10.0))


def test_log_sum_zero_lambda():
    # λ = 0 (log_rho = -inf) -> only the t=0 term survives -> log 1 = 0
    assert _log_finite_horizon_sum(float("-inf"), 10) == 0.0


def test_log_sum_matches_direct_where_representable():
    for r, K in [(0.5, 50), (0.9, 50), (1.1, 50), (2.0, 50)]:
        direct = np.log(np.sum(r ** np.arange(K, dtype=float)))
        assert abs(direct - _log_finite_horizon_sum(float(np.log(r)), K)) < 1e-9
