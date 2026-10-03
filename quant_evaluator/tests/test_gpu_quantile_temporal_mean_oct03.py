"""GPU temporal quantile-mean overflow regressions with Decimal oracles."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _decimal_mean(values):
    finite = [float(value) for value in np.asarray(values).reshape(-1)
              if math.isfinite(float(value))]
    if not finite:
        return math.nan
    with localcontext() as context:
        context.prec = 2000
        total = sum((Decimal.from_float(value) for value in finite), Decimal(0))
        return float(total / Decimal(len(finite)))


@pytest.mark.parametrize("values", [
    np.full((25, 1), 1e308),
    np.array([[1e308], [3.0], [-1e308], [0.0]]),
    np.array([[float.fromhex("0x0.0000000000001p-1022")], [0.0]]),
    np.array([[-float.fromhex("0x0.0000000000001p-1022")], [0.0]]),
])
def test_gpu_finite_temporal_mean_matches_decimal_extremes(values):
    cp = _cuda()
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0

    means, counts = finite_mean_axis0(
        cp.asarray(values, dtype=cp.float64),
        workspace_bytes=1 << 20,
    )
    got = cp.asnumpy(means).reshape(-1)
    got_counts = cp.asnumpy(counts).reshape(-1)
    assert got_counts[0] == values.shape[0]
    assert got[0].hex() == _decimal_mean(values).hex()


def test_gpu_finite_temporal_mean_masks_nonfinite_and_keeps_threshold_gate():
    cp = _cuda()
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0

    values = np.array([
        [1e308, 1e308], [np.nan, 3.0], [np.inf, -1e308],
        [-np.inf, 0.0], [1e308, np.nan],
    ])
    means, counts = finite_mean_axis0(
        cp.asarray(values, dtype=cp.float64), workspace_bytes=1 << 20)
    means = cp.asnumpy(means)
    counts = cp.asnumpy(counts)
    np.testing.assert_array_equal(counts, [2, 4])
    assert means[0] == 1e308
    assert means[1] == _decimal_mean([1e308, 3.0, -1e308, 0.0])
    gated = np.where(counts >= 4, means, np.nan)
    assert math.isnan(gated[0])
    assert gated[1] == means[1]


def test_gpu_finite_temporal_mean_fails_closed_on_small_workspace():
    cp = _cuda()
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0

    values = cp.ones((25, 10), dtype=cp.float64)
    with pytest.raises(MemoryError, match="requires"):
        finite_mean_axis0(values, workspace_bytes=1)


def test_gpu_finite_temporal_mean_rejects_noncurrent_device():
    cp = _cuda()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("device mismatch requires at least two CUDA devices")
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0

    current = cp.cuda.Device().id
    other = (current + 1) % cp.cuda.runtime.getDeviceCount()
    with cp.cuda.Device(other):
        values = cp.ones((25, 1), dtype=cp.float64)
    with pytest.raises(ValueError, match="current CUDA device"):
        finite_mean_axis0(values, workspace_bytes=1 << 20)


def test_public_cuda_quantile_temporal_means_keep_huge_constant_finite():
    _cuda()
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    times, assets, factors = 25, 10, 1
    time_ids = tuple(range(times))
    asset_ids = np.asarray([f"a{i}" for i in range(assets)])
    factor_values = np.broadcast_to(
        np.arange(assets, dtype=np.float64)[None, :, None],
        (times, assets, factors),
    ).copy()
    batch = FactorBatch(
        ("f0",), AxisRef("time", "int64", times, np.asarray(time_ids)),
        AxisRef("asset", "str", assets, asset_ids), factor_values,
    )
    labels = np.full((times, assets), 1e308, dtype=np.float64)
    label = LabelBundle(
        "huge-temporal-label", labels, 1, decision_time=time_ids,
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=batch.asset_axis,
    )
    metrics = ("quantile_returns_full",)
    parameters = {"quantile_returns_full": {"n_quantiles": 1}}
    result = evaluate(
        batch, label, metrics=metrics, metric_parameters=parameters,
        backend="cuda_strict",
    )
    full = result.artifacts["quantile_returns_full"].values
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(full, np.full((1, 1), 1e308))
def test_gpu_finite_temporal_mean_passes_budget_after_live_arrays(monkeypatch):
    cp = _cuda()
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0
    from quant_evaluator.kernels.gpu import quantile_numeric

    captured = {}

    def capture_budget(bucket, labels, means, counts, min_assets, *, workspace_bytes):
        captured["live_bytes"] = labels.nbytes + bucket.nbytes + means.nbytes + counts.nbytes
        captured["workspace_bytes"] = workspace_bytes

    monkeypatch.setattr(quantile_numeric, "repair_quantile_means_gpu", capture_budget)
    values = cp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=cp.float64)
    initial_workspace = 1 << 20
    finite_mean_axis0(values, workspace_bytes=initial_workspace)

    expected_live = 3 * 2 * (8 + 4) + 2 * (8 + 8)
    assert captured["live_bytes"] == expected_live
    assert captured["workspace_bytes"] == initial_workspace - expected_live
    assert captured["workspace_bytes"] < initial_workspace


def test_gpu_finite_temporal_mean_rejects_insufficient_repair_remainder():
    cp = _cuda()
    from quant_evaluator.kernels.gpu.finite_mean import finite_mean_axis0

    # 96 bytes passes the helper's 40*T*R+16R preflight. After live arrays,
    # 56 bytes remain, below the quantile repair guard's minimum working set.
    values = cp.asarray([[1.0], [2.0]], dtype=cp.float64)
    with pytest.raises(MemoryError, match="requires"):
        finite_mean_axis0(values, workspace_bytes=96)


def _public_cuda_quantile_metrics(labels):
    _cuda()
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    times, assets = labels.shape
    time_ids = tuple(range(times))
    asset_ids = np.asarray([f"a{i}" for i in range(assets)])
    factor_values = np.broadcast_to(
        np.arange(assets, dtype=np.float64)[None, :, None],
        (times, assets, 1),
    ).copy()
    batch = FactorBatch(
        ("f0",), AxisRef("time", "int64", times, np.asarray(time_ids)),
        AxisRef("asset", "str", assets, asset_ids), factor_values,
    )
    label = LabelBundle(
        "huge-temporal-label", labels, 1, decision_time=time_ids,
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=batch.asset_axis,
    )
    metrics = ("quantile_returns_full", "quantile_spread", "quantile_monotonicity")
    parameters = {
        "quantile_returns_full": {"n_quantiles": 5},
        "quantile_spread": {"n_quantiles": 5},
    }
    return evaluate(
        batch, label, metrics=metrics, metric_parameters=parameters,
        backend="cuda_strict",
    )


def test_public_cuda_quantile_full_spread_monotonicity_constant_profile():
    result = _public_cuda_quantile_metrics(np.full((25, 50), 1e308))
    full = result.artifacts["quantile_returns_full"].values
    spread = result.artifacts["quantile_spread"].values
    monotonicity = result.artifacts["quantile_monotonicity"]

    np.testing.assert_array_equal(full, np.full((5, 1), 1e308))
    np.testing.assert_array_equal(spread, np.array([0.0]))
    np.testing.assert_array_equal(monotonicity.values, np.array([0.0]))
    assert monotonicity.provenance["observation_counts"] == (4,)


def test_public_cuda_quantile_spread_huge_top_bottom_difference():
    labels = np.full((25, 50), 1e308)
    labels[:, :10] = 0.0
    result = _public_cuda_quantile_metrics(labels)
    full = result.artifacts["quantile_returns_full"].values
    spread = result.artifacts["quantile_spread"].values
    monotonicity = result.artifacts["quantile_monotonicity"]

    expected_profile = np.array([[0.0], [1e308], [1e308], [1e308], [1e308]])
    np.testing.assert_array_equal(full, expected_profile)
    np.testing.assert_array_equal(spread, np.array([1e308]))
    np.testing.assert_array_equal(monotonicity.values, np.array([0.25]))
    assert monotonicity.provenance["observation_counts"] == (4,)
