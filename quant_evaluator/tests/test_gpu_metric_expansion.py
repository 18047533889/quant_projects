"""GPU parity tests for the metric-expansion families (spec §28-30, Phase 7).

Predictive (spec §28) and quantile-shape (spec §29) metrics have cheap GPU
kernels that must match their CPU reference to rtol<=1e-8 / atol<=1e-10.
These tests SKIP when CuPy is not importable so CPU-only CI still passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.metrics.predictive import (
    compute_ic_positive_ratio,
    compute_ic_recent_vs_history_delta,
    compute_ic_sign_consistency,
    compute_monthly_rank_ic,
    compute_quarterly_rank_ic,
    compute_rank_ic_decay,
    compute_rank_ic_positive_ratio,
    compute_recent_12m_rank_ic,
    compute_recent_3m_rank_ic,
    compute_recent_6m_rank_ic,
    compute_rolling_rank_ic_ir,
    compute_rolling_rank_ic_mean,
    compute_worst_quarter_rank_ic,
    compute_worst_year_rank_ic,
    compute_yearly_rank_ic,
)
from quant_evaluator.metrics.quantile_shape import (
    compute_bottom_quantile_cliff,
    compute_quantile_adjacent_spread,
    compute_quantile_curvature,
    compute_quantile_extreme_cliff,
    compute_quantile_monotonicity,
    compute_quantile_tail_asymmetry,
    compute_top_quantile_cliff,
)

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.predictive import (  # noqa: E402
    ic_positive_ratio,
    ic_recent_vs_history_delta,
    ic_sign_consistency,
    monthly_rank_ic,
    quarterly_rank_ic,
    rank_ic_decay,
    rank_ic_positive_ratio,
    recent_12m_rank_ic,
    recent_3m_rank_ic,
    recent_6m_rank_ic,
    rolling_rank_ic_ir,
    rolling_rank_ic_mean,
    worst_quarter_rank_ic,
    worst_year_rank_ic,
    yearly_rank_ic,
)
from quant_evaluator.kernels.gpu.quantile_shape import (  # noqa: E402
    bottom_quantile_cliff,
    quantile_adjacent_spread,
    quantile_curvature,
    quantile_extreme_cliff,
    quantile_monotonicity,
    quantile_tail_asymmetry,
    top_quantile_cliff,
)

RTOL = 1e-8
ATOL = 1e-10


def _ic_series():
    rng = np.random.default_rng(11)
    T, F = 300, 5
    ic = rng.normal(size=(T, F)) * 0.4
    ic[10, 1] = np.nan
    ic[40:44, 2] = np.nan
    ic[3, 4] = np.nan
    return ic


def _quantile_returns():
    rng = np.random.default_rng(12)
    nq, F = 10, 5
    qr = rng.normal(size=(nq, F)) * 0.1
    qr[0, 1] = np.nan
    qr[2, 2] = np.nan
    return qr


def test_gpu_predictive_parity():
    ic = _ic_series()
    cases = [
        (ic_positive_ratio, compute_ic_positive_ratio, {}),
        (rank_ic_positive_ratio, compute_rank_ic_positive_ratio, {}),
        (yearly_rank_ic, compute_yearly_rank_ic, {}),
        (monthly_rank_ic, compute_monthly_rank_ic, {}),
        (quarterly_rank_ic, compute_quarterly_rank_ic, {}),
        (rolling_rank_ic_mean, compute_rolling_rank_ic_mean, {}),
        (rolling_rank_ic_ir, compute_rolling_rank_ic_ir, {}),
        (recent_3m_rank_ic, compute_recent_3m_rank_ic, {}),
        (recent_6m_rank_ic, compute_recent_6m_rank_ic, {}),
        (recent_12m_rank_ic, compute_recent_12m_rank_ic, {}),
        (worst_year_rank_ic, compute_worst_year_rank_ic, {}),
        (worst_quarter_rank_ic, compute_worst_quarter_rank_ic, {}),
        (rank_ic_decay, compute_rank_ic_decay, {}),
        (ic_sign_consistency, compute_ic_sign_consistency, {}),
        (ic_recent_vs_history_delta, compute_ic_recent_vs_history_delta, {}),
    ]
    for gpu_fn, cpu_fn, kw in cases:
        gpu = gpu_fn(ic, **kw)
        cpu = cpu_fn(ic, **kw)
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_quantile_shape_parity():
    qr = _quantile_returns()
    cases = [
        (quantile_monotonicity, compute_quantile_monotonicity),
        (quantile_curvature, compute_quantile_curvature),
        (quantile_tail_asymmetry, compute_quantile_tail_asymmetry),
        (quantile_adjacent_spread, compute_quantile_adjacent_spread),
        (quantile_extreme_cliff, compute_quantile_extreme_cliff),
        (top_quantile_cliff, compute_top_quantile_cliff),
        (bottom_quantile_cliff, compute_bottom_quantile_cliff),
    ]
    for gpu_fn, cpu_fn in cases:
        gpu = gpu_fn(qr)
        cpu = cpu_fn(qr)
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_predictive_1d_input():
    ic = _ic_series()[:, 0]
    gpu = ic_positive_ratio(ic)
    cpu = compute_ic_positive_ratio(ic)
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_gpu_quantile_shape_1d_input():
    qr = _quantile_returns()[:, 0]
    gpu = quantile_monotonicity(qr)
    cpu = compute_quantile_monotonicity(qr)
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL, equal_nan=True)
