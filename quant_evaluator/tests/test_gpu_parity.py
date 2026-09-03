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
from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns
from quant_evaluator.kernels.gpu.turnover import batched_turnover
from quant_evaluator.kernels.gpu.stability import batched_rank_stability
from quant_evaluator.metrics.quantile import assign_quantiles, compute_quantile_returns
from quant_evaluator.metrics.turnover import compute_turnover
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


def test_gpu_turnover_parity(data):
    x, _ = data
    xt = np.transpose(x, (0, 2, 1))
    w = cp.asnumpy(batched_rank_weights(xt))
    turn_gpu = cp.asnumpy(batched_turnover(xt))
    turn_cpu = []
    for f in range(x.shape[2]):
        vals = [compute_turnover(w[t - 1, f], w[t, f]) for t in range(1, x.shape[0])]
        turn_cpu.append(np.nanmean(vals))
    turn_cpu = np.array(turn_cpu)
    assert np.abs(turn_gpu - turn_cpu).max() < 1e-8


def test_gpu_rank_stability_parity(data):
    x, _ = data
    xt = np.transpose(x, (0, 2, 1))
    gpu = cp.asnumpy(batched_rank_stability(xt, lag=1, min_obs=10))
    cpu = compute_rank_stability(x, lag=1, method="spearman")
    mask = np.isfinite(cpu)
    assert np.nanmax(np.abs(gpu[mask] - cpu[mask])) < 1e-8
