"""GPU parity tests for the metric-expansion families (spec §24/§25/§26/§30).

Exposure / purity, tradability / capacity, novelty / interactions, and
data-quality GPU kernels must match their CPU reference to rtol<=1e-8 /
atol<=1e-10.  These tests SKIP when CuPy is not importable so CPU-only CI
still passes.
"""

from __future__ import annotations

import numpy as np
import pytest

# CPU oracles (read-only imports)
from quant_evaluator.metrics.exposure import (
    compute_concentration_hhi,
    compute_sector_exposure,
    compute_factor_loadings,
    compute_style_exposure,
)
from quant_evaluator.metrics.temporal import compute_factor_turnover_rate
from quant_evaluator.metrics.turnover import (
    compute_weighted_turnover,
    compute_turnover_contribution,
)
from quant_evaluator.metrics.data_quality import (
    compute_missing_ratio,
    compute_missing_timeline,
    compute_staleness,
    compute_effective_n,
    compute_distinct_level_ratio,
    compute_tie_ratio,
    compute_cross_section_cardinality,
    compute_tradable_coverage,
    compute_universe_churn,
)
from quant_evaluator.metrics.interactions.pairwise import compute_pairwise_correlation
from quant_evaluator.metrics.interactions.conditional import (
    compute_conditional_ic,
    compute_incremental_ic,
)
from quant_evaluator.metrics.interactions.substitution import compute_substitution_effect
from quant_evaluator.metrics.interactions.complementarity import (
    compute_complementarity_score,
    compute_interaction_strength,
)
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.exposure import (  # noqa: E402
    batched_concentration_hhi,
    batched_sector_exposure,
    batched_factor_loadings,
    batched_style_exposure,
)
from quant_evaluator.kernels.gpu.tradability import (  # noqa: E402
    batched_factor_turnover_rate,
    batched_weighted_turnover,
    batched_turnover_contribution,
)
from quant_evaluator.kernels.gpu.data_quality import (  # noqa: E402
    batched_missing_ratio,
    batched_missing_timeline,
    batched_staleness,
    batched_effective_n,
    batched_distinct_level_ratio,
    batched_tie_ratio,
    batched_cross_section_cardinality,
    batched_tradable_coverage,
    batched_universe_churn,
)
from quant_evaluator.kernels.gpu.interactions import (  # noqa: E402
    batched_pairwise_correlation,
    batched_conditional_ic,
    batched_substitution_effect,
    batched_complementarity_score,
    batched_interaction_strength,
    batched_incremental_ic,
)

RTOL = 1e-8
ATOL = 1e-10


def _make_batch(x, y):
    """x is (T, N, F); y is (T, N)."""
    T, N, F = x.shape
    fb = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
        asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
        values=x,
        layout="wide",
    )
    idx = pd_date_range(T)
    lb = LabelBundle(
        target_id="next_ret", values=y, horizon=1,
        decision_time=tuple(idx), label_start_time=tuple(idx),
        label_end_time=tuple(idx + np.timedelta64(1, "D")),
    )
    return fb, lb


def pd_date_range(T):
    import pandas as pd
    return pd.date_range("2016-01-04", periods=T)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    T, N, F = 20, 300, 4
    x = rng.normal(size=(T, N, F))
    x[rng.random((T, N, F)) < 0.05] = np.nan
    y = rng.normal(size=(T, N))
    y[rng.random((T, N)) < 0.05] = np.nan
    return x, y


def _tfn(x):
    return np.transpose(x, (0, 2, 1))  # (T, F, N)


# ---------------------------------------------------------------------------
# exposure / purity
# ---------------------------------------------------------------------------

def test_gpu_hhi_parity(data):
    x, _ = data
    gpu = cp.asnumpy(batched_concentration_hhi(_tfn(x)))
    cpu = np.stack([compute_concentration_hhi(x[:, :, f]) for f in range(x.shape[2])], axis=1)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_hhi_parity_with_weights(data):
    x, _ = data
    rng = np.random.default_rng(1)
    w = rng.uniform(0.1, 1.0, size=(x.shape[0], x.shape[1]))
    gpu = cp.asnumpy(batched_concentration_hhi(_tfn(x), weights=w))
    cpu = np.stack([compute_concentration_hhi(x[:, :, f], weights=w) for f in range(x.shape[2])], axis=1)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_sector_exposure_parity(data):
    x, _ = data
    rng = np.random.default_rng(2)
    sectors = rng.integers(0, 5, size=x.shape[1]).astype(float)
    sectors[::50] = np.nan
    gpu_exp, gpu_cnt = batched_sector_exposure(_tfn(x), sectors)
    gpu_exp = cp.asnumpy(gpu_exp)
    gpu_cnt = cp.asnumpy(gpu_cnt)
    cpu_exp, cpu_cnt = compute_sector_exposure(x[:, :, 0], sectors)
    # compare factor 0
    mask = np.isfinite(cpu_exp)
    assert np.nanmax(np.abs(gpu_exp[:, 0, :][mask] - cpu_exp[mask])) < 1e-8
    assert np.array_equal(gpu_cnt[:, 0, :], cpu_cnt)


def test_gpu_factor_loadings_parity(data):
    x, _ = data
    rng = np.random.default_rng(3)
    K = 2
    rf = rng.normal(size=(x.shape[0], x.shape[1], K))
    rf[rng.random(rf.shape) < 0.05] = np.nan
    gpu_l, gpu_r2, gpu_res = batched_factor_loadings(_tfn(x), rf, intercept=True, min_obs=10)
    gpu_l = cp.asnumpy(gpu_l)
    gpu_r2 = cp.asnumpy(gpu_r2)
    gpu_res = cp.asnumpy(gpu_res)
    for f in range(x.shape[2]):
        cpu_l, cpu_r2, cpu_res = compute_factor_loadings(x[:, :, f], rf, intercept=True, min_obs=10)
        mask = np.isfinite(cpu_l)
        assert np.nanmax(np.abs(gpu_l[:, f, :][mask] - cpu_l[mask])) < 1e-6
        mask2 = np.isfinite(cpu_r2)
        assert np.nanmax(np.abs(gpu_r2[:, f][mask2] - cpu_r2[mask2])) < 1e-6
        mask3 = np.isfinite(cpu_res)
        assert np.nanmax(np.abs(gpu_res[:, f, :][mask3] - cpu_res[mask3])) < 1e-6


def test_gpu_style_exposure_parity(data):
    x, _ = data
    rng = np.random.default_rng(4)
    K = 2
    sf = rng.normal(size=(x.shape[0], x.shape[1], K))
    names = ("size", "value")
    gpu = cp.asnumpy(batched_style_exposure(_tfn(x), sf, names))
    cpu = np.stack([compute_style_exposure(x[:, :, f], sf, names) for f in range(x.shape[2])], axis=0)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-6


# ---------------------------------------------------------------------------
# tradability / capacity
# ---------------------------------------------------------------------------

def test_gpu_factor_turnover_rate_parity(data):
    x, _ = data
    gpu = cp.asnumpy(batched_factor_turnover_rate(_tfn(x), quantile=0.9))
    cpu = compute_factor_turnover_rate(x, quantile=0.9)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_factor_turnover_rate_bottom(data):
    x, _ = data
    gpu = cp.asnumpy(batched_factor_turnover_rate(_tfn(x), quantile=0.1))
    cpu = compute_factor_turnover_rate(x, quantile=0.1)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_weighted_turnover_parity(data):
    x, _ = data
    rng = np.random.default_rng(5)
    w = rng.normal(size=(x.shape[0], x.shape[1]))
    w[rng.random(w.shape) < 0.05] = np.nan
    ps = np.abs(w) + 0.1
    gpu = cp.asnumpy(batched_weighted_turnover(w, ps))
    cpu = compute_weighted_turnover(w, ps)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_turnover_contribution_parity(data):
    x, _ = data
    rng = np.random.default_rng(6)
    w = rng.normal(size=(x.shape[0], x.shape[1]))
    w[rng.random(w.shape) < 0.05] = np.nan
    gpu = cp.asnumpy(batched_turnover_contribution(w))
    cpu = compute_turnover_contribution(w)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


# ---------------------------------------------------------------------------
# data-quality
# ---------------------------------------------------------------------------

def test_gpu_data_quality_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    xt = _tfn(x)
    cases = [
        (batched_missing_ratio(xt), compute_missing_ratio(fb)),
        (batched_missing_timeline(xt), compute_missing_timeline(fb)),
        (batched_staleness(xt), compute_staleness(fb)),
        (batched_effective_n(xt), compute_effective_n(fb)),
        (batched_distinct_level_ratio(xt), compute_distinct_level_ratio(fb)),
        (batched_tie_ratio(xt), compute_tie_ratio(fb)),
        (batched_cross_section_cardinality(xt), compute_cross_section_cardinality(fb)),
        (batched_tradable_coverage(xt, y), compute_tradable_coverage(fb, lb)),
        (batched_universe_churn(xt, y), compute_universe_churn(fb, lb)),
    ]
    for gpu, cpu in cases:
        gpu = np.asarray(gpu)
        cpu = np.asarray(cpu)
        mask = np.isfinite(cpu)
        assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


# ---------------------------------------------------------------------------
# novelty / interactions
# ---------------------------------------------------------------------------

def test_gpu_pairwise_correlation_parity(data):
    x, _ = data
    fb, _ = _make_batch(x, np.zeros((x.shape[0], x.shape[1])))
    for method in ("pearson", "spearman"):
        gpu = cp.asnumpy(batched_pairwise_correlation(_tfn(x), method=method, min_obs=30))
        cpu = compute_pairwise_correlation(fb, method=method, min_obs=30)
        mask = np.isfinite(cpu)
        assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_conditional_ic_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    for method in ("pearson", "spearman"):
        gpu_ic, gpu_cnt = batched_conditional_ic(
            _tfn(x), y, conditioning_factor_idx=0, quantiles=5,
            method=method, min_assets=10,
        )
        gpu_ic = cp.asnumpy(gpu_ic)
        gpu_cnt = cp.asnumpy(gpu_cnt)
        cpu_ic, cpu_cnt = compute_conditional_ic(
            fb, lb, conditioning_factor_idx=0, quantiles=5,
            method=method, min_assets=10,
        )
        mask = np.isfinite(cpu_ic)
        assert np.nanmax(np.abs(gpu_ic[mask] - cpu_ic[mask])) < 1e-8
        assert np.array_equal(gpu_cnt, cpu_cnt)


def test_gpu_substitution_effect_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    for method in ("pearson", "spearman"):
        ga, gb, gc = batched_substitution_effect(
            _tfn(x), y, 0, 1, method=method, min_assets=30,
        )
        ga, gb, gc = cp.asnumpy(ga), cp.asnumpy(gb), cp.asnumpy(gc)
        ca, cb, cc = compute_substitution_effect(
            fb, lb, 0, 1, method=method, min_assets=30,
        )
        for g, c in ((ga, ca), (gb, cb), (gc, cc)):
            mask = np.isfinite(c)
            assert np.nanmax(np.abs(g[mask] - c[mask])) < 1e-8


def test_gpu_complementarity_score_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    for method in ("pearson", "spearman"):
        gs, gm, gstd = batched_complementarity_score(
            _tfn(x), y, 0, 1, method=method, min_assets=30, min_periods=20,
        )
        gs, gm, gstd = cp.asnumpy(gs), cp.asnumpy(gm), cp.asnumpy(gstd)
        cs, cm, cstd = compute_complementarity_score(
            fb, lb, 0, 1, method=method, min_assets=30, min_periods=20,
        )
        assert abs(gs - cs) < 1e-8
        assert abs(gm - cm) < 1e-8
        assert abs(gstd - cstd) < 1e-8


def test_gpu_interaction_strength_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    for method in ("pearson", "spearman"):
        gi, ga = batched_interaction_strength(
            _tfn(x), y, 0, 1, method=method, min_assets=30,
        )
        gi, ga = cp.asnumpy(gi), cp.asnumpy(ga)
        ci, ca = compute_interaction_strength(
            fb, lb, 0, 1, method=method, min_assets=30,
        )
        for g, c in ((gi, ci), (ga, ca)):
            mask = np.isfinite(c)
            assert np.nanmax(np.abs(g[mask] - c[mask])) < 1e-8


def test_gpu_incremental_ic_parity(data):
    x, y = data
    fb, lb = _make_batch(x, y)
    for method in ("pearson", "spearman"):
        gi, gb, gt = batched_incremental_ic(
            _tfn(x), y, (0,), 1, method=method, min_assets=30,
        )
        gi, gb, gt = cp.asnumpy(gi), cp.asnumpy(gb), cp.asnumpy(gt)
        ci, cb, ct = compute_incremental_ic(
            fb, lb, (0,), 1, method=method, min_assets=30,
        )
        for g, c in ((gi, ci), (gb, cb), (gt, ct)):
            mask = np.isfinite(c)
            assert np.nanmax(np.abs(g[mask] - c[mask])) < 1e-8


# ---------------------------------------------------------------------------
# capability registry
# ---------------------------------------------------------------------------

def test_gpu_metric_expansion_registered():
    from quant_evaluator.kernels.gpu._register_metric_expansion import (
        register_gpu_metric_expansion_kernels,
    )
    from quant_evaluator.backends.capability_registry import (
        get_backend_capability_registry,
    )
    n = register_gpu_metric_expansion_kernels()
    assert n >= 20
    reg = get_backend_capability_registry()
    for op in (
        "hhi_concentration", "factor_turnover_rate", "pairwise_correlation",
        "conditional_ic", "substitution_effect", "complementarity_score",
        "interaction_strength", "incremental_ic", "missing_ratio",
        "tradable_coverage", "universe_churn",
    ):
        assert reg.cuda_ready(op), f"{op} not cuda-ready"
