"""GPU parity tests for rank / quantile / turnover / rank-stability / IC.

These tests SKIP when CuPy is not importable so CPU-only CI still passes.
They verify exact parity with the CPU reference (spec §54-55).
"""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.kernels.gpu.rank import (
    batched_rank,
    batched_distinct_level_count,
    batched_quantile_assignment,
    batched_rank_weights,
)
from quant_evaluator.kernels.gpu.correlation import (
    batched_pearson_ic,
    batched_spearman_ic,
)
from quant_evaluator.metrics.ic import _pearson_correlation, compute_daily_ic
from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns
from quant_evaluator.kernels.gpu.turnover import batched_turnover
from quant_evaluator.kernels.gpu.stability import batched_rank_stability
from quant_evaluator.metrics.quantile import assign_quantiles, compute_quantile_returns
from quant_evaluator.metrics.turnover import compute_turnover, estimate_turnover_from_ranks
from quant_evaluator.metrics.registry_adapters import compute_turnover_value
from quant_evaluator.metrics.temporal import compute_rank_stability
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


def _cpu_spearman(x, y, min_obs=20):
    T, N, F = x.shape
    out = np.full((T, F), np.nan)
    for t in range(T):
        for f in range(F):
            m = np.isfinite(x[t, :, f]) & np.isfinite(y[t, :])
            if m.sum() < min_obs:
                continue
            rx = pd.Series(x[t, m, f]).rank().values
            ry = pd.Series(y[t, m]).rank().values
            if np.unique(x[t, m, f]).size < 2 or np.unique(y[t, m]).size < 2:
                continue
            out[t, f] = np.corrcoef(rx, ry)[0, 1]
    return out


def _cpu_pearson(x, y, min_obs=20):
    T, N, F = x.shape
    out = np.full((T, F), np.nan)
    for t in range(T):
        for f in range(F):
            m = np.isfinite(x[t, :, f]) & np.isfinite(y[t, :])
            if m.sum() < min_obs:
                continue
            out[t, f] = np.corrcoef(x[t, m, f], y[t, m])[0, 1]
    return out


def _make_batch(x, y):
    T, N, F = x.shape
    fb = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
        asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
        values=x,
        layout="wide",
    )
    idx = pd.date_range("2016-01-04", periods=T)
    lb = LabelBundle(
        target_id="next_ret", values=y, horizon=1,
        decision_time=tuple(idx), label_start_time=tuple(idx),
        label_end_time=tuple(idx + pd.Timedelta(days=1)),
    )
    return fb, lb


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    T, N, F = 20, 300, 4
    x = rng.normal(size=(T, N, F))
    x[rng.random((T, N, F)) < 0.05] = np.nan
    y = rng.normal(size=(T, N))
    y[rng.random((T, N)) < 0.05] = np.nan
    return x, y


def test_gpu_rank_ic_parity(data):
    x, y = data
    xt = np.transpose(x, (0, 2, 1))
    gpu, _ = batched_spearman_ic(xt, y, min_obs=20)
    gpu = cp.asnumpy(gpu)
    cpu = _cpu_spearman(x, y)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


def test_gpu_pearson_ic_parity(data):
    x, y = data
    xt = np.transpose(x, (0, 2, 1))
    gpu, _ = batched_pearson_ic(xt, y, min_obs=20)
    gpu = cp.asnumpy(gpu)
    cpu = _cpu_pearson(x, y)
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_gpu_ic_seeded_pairwise_oracle(method):
    """Independent CPU oracle across ties, missing values and constant subsets."""
    rng = np.random.default_rng(20260928)
    kernel = batched_pearson_ic if method == "pearson" else batched_spearman_ic
    max_error = 0.0
    for case in range(20):
        x = rng.normal(size=(3, 4, 53))
        y = rng.normal(size=(3, 53))
        if case % 3 == 0:
            x = np.round(x, 1)
        if case % 4 == 0:
            y = np.round(y, 1)
        x[rng.random(x.shape) < 0.18] = np.nan
        y[rng.random(y.shape) < 0.13] = np.nan
        if case % 7 == 0:
            x[0, 0, :35] = 1e12
            x[0, 0, 35:] = np.nan
        if case % 11 == 0:
            y[1, :40] = 1e12
            x[1, 0, 40:] = np.nan
        gpu, counts = kernel(x, y, min_obs=20)
        gpu, counts = cp.asnumpy(gpu), cp.asnumpy(counts)
        expected = np.full((3, 4), np.nan)
        expected_counts = np.zeros((3, 4), dtype=np.int32)
        for ti in range(3):
            for fi in range(4):
                valid = np.isfinite(x[ti, fi]) & np.isfinite(y[ti])
                expected_counts[ti, fi] = valid.sum()
                if valid.sum() < 20:
                    continue
                xv, yv = x[ti, fi, valid], y[ti, valid]
                if np.ptp(xv) == 0 or np.ptp(yv) == 0:
                    continue
                if method == "spearman":
                    xv = pd.Series(xv).rank(method="average").to_numpy()
                    yv = pd.Series(yv).rank(method="average").to_numpy()
                expected[ti, fi] = np.corrcoef(xv, yv)[0, 1]
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(np.isnan(gpu), np.isnan(expected))
        max_error = max(max_error, np.nanmax(np.abs(gpu - expected)))
    assert max_error < 1e-8


@pytest.mark.parametrize("label_offset", [0.0, 10_000.0])
def test_gpu_pearson_ic_float32_large_offset(label_offset):
    rng = np.random.default_rng(42)
    n = 100_000
    signal = rng.normal(size=n).astype(np.float32)
    noise = rng.normal(size=n).astype(np.float32)
    factors = (10_000.0 + signal).astype(np.float32)
    labels = (label_offset + signal + noise).astype(np.float32)

    gpu, counts = batched_pearson_ic(
        factors[None, None, :], labels[None, :], min_obs=20
    )
    expected = np.corrcoef(factors, labels)[0, 1]
    gpu_value = cp.asnumpy(gpu)[0, 0]
    assert cp.asnumpy(counts)[0, 0] == n
    assert np.isfinite(gpu_value)
    assert abs(gpu_value - expected) < 1e-5


def _stable_pearson_oracle(x, y):
    """Independent scale-normalized Pearson oracle for extreme magnitudes."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    x = x - x.mean()
    y = y - y.mean()
    return np.dot(x, y) / np.sqrt(np.dot(x, x)) / np.sqrt(np.dot(y, y))


@pytest.mark.parametrize("scale_x,scale_y", [
    (1e-200, 1e200),
    (1e-160, 1e160),
    (1e-150, 1e150),
    (1e150, 1e-150),
])
def test_pearson_scale_invariance_across_float64_range(scale_x, scale_y):
    z = np.arange(1, 65, dtype=np.float64)
    x = scale_x * z
    y = scale_y * z[::-1]
    expected = _stable_pearson_oracle(x, y)

    cpu = _pearson_correlation(x, y, min_obs=2)
    gpu, _ = batched_pearson_ic(
        x[None, None, :], y[None, :], min_obs=2
    )
    assert expected == pytest.approx(-1.0, abs=1e-14)
    assert cpu == pytest.approx(expected, abs=1e-14)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(expected, abs=1e-14)


def test_pearson_intermediate_bias_threshold_matches_cpu_gpu():
    y = np.arange(64, dtype=np.float64)
    x = 2e14 + 1e7 * y
    mean = np.mean(x)
    sum_std = np.sqrt(np.sum((x - mean) ** 2))
    sample_std = np.std(x, ddof=1)
    # This lies between the formerly inconsistent sum-std and sample-std
    # threshold checks; both CPU and GPU must now take the stable route.
    assert abs(mean) / sum_std < 1e6
    assert abs(mean) / sample_std > 1e6
    np.testing.assert_array_equal(np.diff(x), np.full(63, 1e7))

    fb, lb = _make_batch(x[None, :, None], y[None, :])
    cpu, counts = compute_daily_ic(fb, lb, method="pearson", min_assets=2)
    gpu, _ = batched_pearson_ic(x[None, None, :], y[None, :], min_obs=2)
    assert counts.tolist() == [[64]]
    assert cpu[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(1.0, abs=1e-12)


def test_pearson_large_offset_exact_linear_relationship():
    y = np.arange(64, dtype=np.float64)
    x = 1e16 + 2.0 * y
    # Every increment is exactly representable at this offset.
    np.testing.assert_array_equal(np.diff(x), np.full(63, 2.0))

    cpu = _pearson_correlation(x, y, min_obs=2)
    gpu, _ = batched_pearson_ic(
        x[None, None, :], y[None, :], min_obs=2
    )
    assert cpu == pytest.approx(1.0, abs=1e-12)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(1.0, abs=1e-12)


def test_pearson_extreme_scale_uses_pairwise_finite_values():
    z = np.arange(1, 65, dtype=np.float64)
    x = 1e-200 * z
    y = 1e200 * z[::-1]
    x[[1, 7]] = [np.nan, np.inf]
    y[[3, 9]] = [-np.inf, np.nan]
    expected = _stable_pearson_oracle(x, y)

    cpu = _pearson_correlation(x, y, min_obs=2)
    gpu, counts = batched_pearson_ic(
        x[None, None, :], y[None, :], min_obs=2
    )
    assert expected == pytest.approx(-1.0, abs=1e-14)
    assert cpu == pytest.approx(expected, abs=1e-14)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(expected, abs=1e-14)
    assert int(cp.asnumpy(counts)[0, 0]) == 60


def test_pearson_opposite_finite_extremes_use_scaling_fallback():
    x = np.array([-1.7e308, 1.7e308, -1.6e308, 1.6e308], dtype=np.float64)
    y = x.copy()

    cpu = _pearson_correlation(x, y, min_obs=2)
    gpu, _ = batched_pearson_ic(
        x[None, None, :], y[None, :], min_obs=2
    )
    assert cpu == pytest.approx(1.0, abs=1e-14)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(1.0, abs=1e-14)


def test_pearson_stable_path_preserves_zero_correlation_and_constants():
    x = np.tile(np.arange(1, 33, dtype=np.float64), 2)
    y = np.repeat(np.array([-1.0, 1.0]), 32)
    expected = _stable_pearson_oracle(x, y)
    cpu = _pearson_correlation(x, y, min_obs=2)
    gpu, _ = batched_pearson_ic(x[None, None, :], y[None, :], min_obs=2)
    assert abs(expected) < 1e-15
    assert cpu == pytest.approx(expected, abs=1e-15)
    assert float(cp.asnumpy(gpu)[0, 0]) == pytest.approx(expected, abs=1e-15)

    constant = np.ones(64, dtype=np.float64) * 1e-200
    assert np.isnan(_pearson_correlation(constant, y, min_obs=2))
    gpu_constant, _ = batched_pearson_ic(
        constant[None, None, :], y[None, :], min_obs=2
    )
    assert np.isnan(float(cp.asnumpy(gpu_constant)[0, 0]))


def test_gpu_rank_heavy_ties():
    x = np.array([[3.0, 1, 2, 2, 5, 1, 2, 2]])
    gpu = cp.asnumpy(batched_rank(x))[0]
    cpu = pd.Series(x[0]).rank(method="average").values
    assert np.abs(gpu - cpu).max() < 1e-12


def test_gpu_rank_nan_excluded():
    x = np.array([[1.0, np.nan, 2, 2, 1, 5, 3, 1, np.nan]])
    gpu = cp.asnumpy(batched_rank(x))[0]
    cpu = pd.Series(x[0]).rank(method="average").values
    ok = np.isfinite(x[0])
    assert np.abs(gpu[ok] - cpu[ok]).max() < 1e-12
    assert np.all(np.isnan(gpu) == np.isnan(x[0]))


def test_gpu_rank_distinct_level_floor():
    x = np.array([[1.0, 1, 1, 1, 1, 1]])
    dl = int(cp.asnumpy(batched_distinct_level_count(x))[0])
    assert dl == 1


def test_gpu_rank_constant_reject():
    # constant factor -> spearman NaN
    x = np.ones((1, 1, 50))
    y = np.random.default_rng(0).normal(size=(1, 50))
    gpu, _ = batched_spearman_ic(x, y, min_obs=20)
    assert np.isnan(cp.asnumpy(gpu)[0, 0])


def test_gpu_quantile_parity():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 200))
    x[0, ::7] = np.nan
    gpu = cp.asnumpy(batched_quantile_assignment(x, n_quantiles=10))[0]
    cpu = assign_quantiles(x, n_quantiles=10, method="max")[0]
    assert np.array_equal(gpu, cpu)


def test_gpu_quantile_tie_policy():
    # value exactly on boundary -> MAX policy puts it in higher bin
    x = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    gpu = cp.asnumpy(batched_quantile_assignment(x, n_quantiles=5))[0]
    cpu = assign_quantiles(x, n_quantiles=5, method="max")[0]
    assert np.array_equal(gpu, cpu)


def test_gpu_quantile_returns_parity(data):
    x, y = data
    xt = np.transpose(x, (0, 2, 1))
    fb, lb = _make_batch(x, y)
    qr_cpu, _ = compute_quantile_returns(fb, lb, n_quantiles=10, min_assets=10)
    qr_gpu = cp.asnumpy(batched_quantile_returns(xt, y, n_quantiles=10, min_assets=10))
    mask = np.isfinite(qr_cpu)
    assert np.nanmax(np.abs(qr_gpu[mask] - qr_cpu[mask])) < 1e-8


def test_single_quantile_assignment_and_returns_match_cpu():
    x = np.tile(np.arange(12, dtype=float)[None, :, None], (2, 1, 1))
    x[0, 1, 0] = np.nan
    x[1, 2, 0] = np.inf
    y = np.tile(np.arange(12, dtype=float), (2, 1))
    y[0, 3] = np.nan
    y[1, 4] = np.inf
    cpu_assignment = assign_quantiles(x[:, :, 0], n_quantiles=1)
    gpu_assignment = cp.asnumpy(batched_quantile_assignment(
        x[:, :, 0], n_quantiles=1))
    np.testing.assert_array_equal(gpu_assignment, cpu_assignment)
    fb, lb = _make_batch(x, y)
    cpu_values, cpu_counts = compute_quantile_returns(
        fb, lb, n_quantiles=1, min_assets=10)
    gpu_values, gpu_counts = batched_quantile_returns(
        np.transpose(x, (0, 2, 1)), y, n_quantiles=1,
        min_assets=10, return_counts=True)
    np.testing.assert_allclose(cp.asnumpy(gpu_values), cpu_values, equal_nan=True)
    np.testing.assert_array_equal(cp.asnumpy(gpu_counts), cpu_counts)


def test_gpu_turnover_parity(data):
    x, _ = data
    xt = np.transpose(x, (0, 2, 1))
    w = cp.asnumpy(batched_rank_weights(xt))
    eligible = np.isfinite(xt).sum(axis=2) >= 2
    w = np.where(eligible[..., None], np.nan_to_num(w, nan=0.0), np.nan)
    turn_gpu = cp.asnumpy(batched_turnover(xt))
    turn_cpu = []
    for f in range(x.shape[2]):
        vals = [compute_turnover(w[t - 1, f], w[t, f]) for t in range(1, x.shape[0])]
        turn_cpu.append(np.nanmean(vals))
    turn_cpu = np.array(turn_cpu)
    assert np.abs(turn_gpu - turn_cpu).max() < 1e-8


@pytest.mark.parametrize("n_assets", [2, 9, 10, 11])
def test_gpu_turnover_matches_cpu_minimum_cross_section_contract(n_assets):
    """Rank turnover remains unknown below CPU's 10-asset floor."""
    x = np.empty((4, n_assets, 1), dtype=np.float64)
    base = np.arange(n_assets, dtype=np.float64)
    x[:, :, 0] = np.stack((base, base[::-1], np.roll(base, 1), base), axis=0)
    # One non-finite value takes a 10-asset cross-section below the floor.
    if n_assets >= 10:
        x[2, 0, 0] = np.inf
    fb, _ = _make_batch(x, np.zeros((4, n_assets), dtype=np.float64))

    cpu = estimate_turnover_from_ranks(fb, window=1)
    gpu = cp.asnumpy(
        batched_turnover(np.transpose(x, (0, 2, 1)), return_series=True)
    )
    np.testing.assert_allclose(gpu, cpu, equal_nan=True)


@pytest.mark.parametrize("n_assets", [9, 10])
def test_gpu_registered_turnover_matches_cpu_adapter_floor(n_assets):
    """The registered turnover metric and GPU kernel share the 10-value floor."""
    base = np.arange(n_assets, dtype=np.float64)
    x = np.stack((base, base[::-1], np.roll(base, 1)))[:, :, None]
    fb, _ = _make_batch(x, np.zeros((3, n_assets), dtype=np.float64))

    cpu = compute_turnover_value(fb, min_periods=2)
    gpu = cp.asnumpy(batched_turnover(np.transpose(x, (0, 2, 1))))
    np.testing.assert_allclose(gpu, cpu, equal_nan=True)
    assert np.isfinite(cpu[0]) == (n_assets == 10)


def test_gpu_rank_stability_parity(data):
    x, _ = data
    xt = np.transpose(x, (0, 2, 1))
    gpu = cp.asnumpy(batched_rank_stability(xt, lag=1, min_obs=10))
    cpu = compute_rank_stability(x, lag=1, method="spearman")
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8
