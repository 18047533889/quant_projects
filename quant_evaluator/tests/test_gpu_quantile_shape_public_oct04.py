"""Public explicit-CUDA contracts for the six quantile-shape metrics.

The oracle below starts from the known asset ordering and label panel.  It
does not call either evaluator or a production quantile/profile builder.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - hardware-dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.api.evaluate_many import evaluate_many


SHAPE_METRICS = (
    "quantile_curvature",
    "quantile_tail_asymmetry",
    "quantile_adjacent_spread",
    "quantile_extreme_cliff",
    "top_quantile_cliff",
    "bottom_quantile_cliff",
)


def _contracts(label_sign=1.0):
    # Three deliberately different cross-sectional orderings.  The first two
    # contain extreme-but-finite endpoints; missingness is explicit and stable.
    t_count, n_assets, n_factors = 32, 72, 3
    times = np.arange(t_count, dtype=np.int64)
    assets = np.arange(n_assets, dtype=np.int64)
    time_axis = AxisRef("time", "int64", t_count, times)
    asset_axis = AxisRef("asset", "int64", n_assets, assets)
    ranks = np.column_stack((assets, n_assets - 1 - assets,
                             (assets * 7) % n_assets)).astype(np.float64)
    values = np.broadcast_to(ranks[None, :, :],
                             (t_count, n_assets, n_factors)).copy()
    values[:, 0, 0] = -1e300
    values[:, -1, 0] = 1e300
    values[:, 0, 1] = 1e300
    values[:, -1, 1] = -1e300
    validity = np.ones(values.shape, dtype=bool)
    validity[::4, 1, 0] = False
    validity[1::4, 34, 1] = False
    values[2::5, 2, 2] = np.nan

    labels = np.broadcast_to(
        ((assets % 9) - 4).astype(np.float64)[None, :] * 0.002,
        (t_count, n_assets),
    ).copy()
    labels[:, 0] = 0.02 * label_sign
    labels[:, -1] = -0.03 * label_sign
    label_validity = np.ones(labels.shape, dtype=bool)
    label_validity[::7, 35] = False
    factor_batch = FactorBatch(
        ("ascending", "descending", "permuted"), time_axis, asset_axis,
        values, validity=validity,
    )
    label_bundle = LabelBundle(
        "ret_up" if label_sign > 0 else "ret_down", labels, 1,
        decision_time=tuple(times), label_start_time=tuple(times),
        label_end_time=tuple(times + 1),
        validity=label_validity, asset_axis=asset_axis,
    )
    return factor_batch, label_bundle


def _manual_profile(batch, label, n_quantiles, min_assets=10, min_periods=20):
    """Independent percentile buckets -> daily means -> long-run profile."""
    t_count, n_assets, n_factors = batch.values.shape
    daily = np.full((t_count, n_quantiles, n_factors), np.nan)
    for t in range(t_count):
        for f in range(n_factors):
            factor_usable = (batch.validity[t, :, f]
                             & np.isfinite(batch.values[t, :, f]))
            ordered = np.sort(batch.values[t, factor_usable, f], kind="mergesort")
            if len(ordered) < n_quantiles:
                continue
            boundaries = []
            for q in range(1, n_quantiles):
                position = (q / n_quantiles) * (len(ordered) - 1)
                low = int(position)
                fraction = position - low
                boundaries.append(ordered[low] + fraction *
                                  (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]))
            bins = np.searchsorted(
                boundaries, batch.values[t, factor_usable, f], side="right")
            label_usable = label.validity[t] & np.isfinite(label.values[t])
            label_mask = label_usable[factor_usable]
            bins = bins[label_mask]
            selected_labels = label.values[t, factor_usable][label_mask]
            for q in range(n_quantiles):
                values_q = selected_labels[bins == q]
                if len(values_q) >= min_assets:
                    daily[t, q, f] = np.mean(values_q)
    profile = np.full((n_quantiles, n_factors), np.nan)
    for q in range(n_quantiles):
        for f in range(n_factors):
            observations = daily[:, q, f]
            finite = observations[np.isfinite(observations)]
            if len(finite) >= min_periods:
                profile[q, f] = np.mean(finite)
    return profile


def _manual_shape_values(profile):
    q_count, n_factors = profile.shape
    result = {name: np.full(n_factors, np.nan) for name in SHAPE_METRICS}
    for f in range(n_factors):
        column = profile[:, f]
        finite = np.isfinite(column)
        if q_count >= 3:
            interior = finite[:-2] & finite[1:-1] & finite[2:]
            if interior.any():
                result["quantile_curvature"][f] = np.mean(
                    column[2:][interior] - 2 * column[1:-1][interior]
                    + column[:-2][interior])
            mid = q_count // 2
            if finite[0] and finite[mid] and finite[-1]:
                result["quantile_tail_asymmetry"][f] = (
                    column[-1] - 2 * column[mid] + column[0])
        pairs = finite[:-1] & finite[1:]
        if pairs.any():
            result["quantile_adjacent_spread"][f] = np.mean(
                np.abs(column[1:][pairs] - column[:-1][pairs]))
        if q_count >= 2 and finite[0] and finite[1] and finite[-1] and finite[-2]:
            result["quantile_extreme_cliff"][f] = (
                (column[-1] - column[-2]) + (column[1] - column[0])) / 2
        if q_count >= 2 and finite[-1] and finite[-2]:
            result["top_quantile_cliff"][f] = column[-1] - column[-2]
        if q_count >= 2 and finite[0] and finite[1]:
            result["bottom_quantile_cliff"][f] = column[1] - column[0]
    return result


def _assert_public_values(bundle, expected):
    for metric, values in expected.items():
        np.testing.assert_allclose(bundle.artifacts[metric].values, values,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)


def _assert_cuda_contract(gpu, cpu, metrics):
    assert gpu.metadata["shape_kernel_backend"] == "cuda_strict"
    assert gpu.metadata["shape_kernel_no_fallback"] is True
    assert gpu.metadata["shape_kernel_dispatches"] >= len(metrics)
    for metric in metrics:
        for factor_id in gpu.factor_ids:
            actual = gpu.get_metric(metric, factor_id)
            expected = cpu.get_metric(metric, factor_id)
            assert actual.observation_count == expected.observation_count
            assert actual.sample_unit == expected.sample_unit


@pytest.mark.parametrize("n_quantiles", [2, 3])
@pytest.mark.parametrize("tile_size", [1, 2, 3])
def test_public_cuda_shape_metrics_match_manual_profile_oracle(
        monkeypatch, n_quantiles, tile_size):
    # Catches missing public CUDA dispatch, wrong Q/min-period propagation,
    # cross-factor tile leakage, and shape-specific missing/extreme mistakes.
    batch, label = _contracts()
    from quant_evaluator.runtime import gpu_executor
    host_shapes = []
    original_to_host = gpu_executor._to_cpu

    def capture_to_host(value):
        host_shapes.append(tuple(value.shape))
        return original_to_host(value)

    monkeypatch.setattr(gpu_executor, "_to_cpu", capture_to_host)
    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile",
                        lambda self, *args, **kwargs: min(tile_size, batch.num_factors))
    profile = _manual_profile(batch, label, n_quantiles)
    expected = _manual_shape_values(profile)
    if n_quantiles == 2:
        assert not np.isfinite(expected["quantile_curvature"]).any()
        assert not np.isfinite(expected["quantile_tail_asymmetry"]).any()
        expected_finite_metrics = set(SHAPE_METRICS) - {
            "quantile_curvature", "quantile_tail_asymmetry"}
    else:
        expected_finite_metrics = set(SHAPE_METRICS)
    assert all(np.isfinite(expected[mid]).any() for mid in expected_finite_metrics)
    cpu = evaluate(
        batch, label, backend="cpu", metrics=SHAPE_METRICS,
        quantile_builder_parameters={"n_quantiles": n_quantiles, "min_assets": 10},
    )
    _assert_public_values(cpu, expected)
    result = evaluate(
        batch, label, backend="cuda_strict", metrics=SHAPE_METRICS,
        quantile_builder_parameters={"n_quantiles": n_quantiles, "min_assets": 10},
    )
    _assert_public_values(result, expected)
    assert result.factor_ids == batch.factor_ids
    _assert_cuda_contract(result, cpu, SHAPE_METRICS)
    widths = [min(tile_size, batch.num_factors - start)
              for start in range(0, batch.num_factors, tile_size)]
    assert host_shapes == [(2 * len(SHAPE_METRICS), width) for width in widths]
    assert result.metadata["factor_tiles_processed"] == int(
        np.ceil(batch.num_factors / tile_size))


@pytest.mark.parametrize("metrics", [SHAPE_METRICS, tuple(reversed(SHAPE_METRICS))])
def test_evaluate_many_cuda_keeps_label_profiles_and_metric_order_isolated(
        monkeypatch, metrics):
    # Catches a shared profile cache keyed without label identity, or a public
    # evaluate_many route that drops shape metric outputs/order.
    batch, up = _contracts(1.0)
    _, down = _contracts(-1.0)
    from collections import Counter
    from quant_evaluator.runtime import gpu_executor
    host_shapes = []
    original_to_host = gpu_executor._to_cpu

    def capture_to_host(value):
        host_shapes.append(tuple(value.shape))
        return original_to_host(value)

    monkeypatch.setattr(gpu_executor, "_to_cpu", capture_to_host)
    uploads = []
    opens = []
    original_stage = DeviceEvaluationSession.stage_masked_factors
    original_open = DeviceEvaluationSession._open

    def stage(self, values, validity, factor_ids, layout="T,F,N"):
        uploads.append(tuple(factor_ids))
        return original_stage(self, values, validity, factor_ids, layout)

    def opened(self):
        opens.append(True)
        return original_open(self)

    monkeypatch.setattr(DeviceEvaluationSession, "estimate_tile",
                        lambda *args, **kwargs: 2)
    monkeypatch.setattr(DeviceEvaluationSession, "stage_masked_factors", stage)
    monkeypatch.setattr(DeviceEvaluationSession, "_open", opened)
    results = evaluate_many(batch, (up, down), metrics=metrics,
                             backend="cuda_strict")
    cpu_results = evaluate_many(batch, (up, down), metrics=metrics,
                                backend="cpu")
    for label in (up, down):
        expected = _manual_shape_values(_manual_profile(batch, label, 5))
        _assert_public_values(results[label.target_id], expected)
        _assert_cuda_contract(results[label.target_id],
                              cpu_results[label.target_id], metrics)
        assert results[label.target_id].factor_ids == batch.factor_ids
        metadata = results[label.target_id].metadata
        assert metadata["multi_label_factor_tile_reuse"] is True
        assert metadata["device_session_counter_scope"] == "shared_session_total"
        assert metadata["shared_session_label_count"] == 2
        assert metadata["shared_session_factor_tiles_processed"] == 2
    assert opens == [True]
    assert uploads == [("ascending", "descending"), ("permuted",)]
    assert Counter(host_shapes) == Counter([(12, 2), (12, 1)] * 2)


def test_shape_profile_cache_isolated_from_quantile_returns_metric_parameter():
    # Shape metrics consume the request-wide Q=3 profile while the vector
    # metric explicitly requests Q=2; a shared cache key must not cross them.
    batch, label = _contracts()
    metrics = SHAPE_METRICS + ("quantile_returns_full",)
    result = evaluate(
        batch, label, backend="cuda_strict", metrics=metrics,
        quantile_builder_parameters={"n_quantiles": 3, "min_assets": 10},
        metric_parameters={"quantile_returns_full": {"n_quantiles": 2}},
    )
    expected_shape = _manual_shape_values(_manual_profile(batch, label, 3))
    _assert_public_values(result, expected_shape)
    cpu = evaluate(
        batch, label, backend="cpu", metrics=metrics,
        quantile_builder_parameters={"n_quantiles": 3, "min_assets": 10},
        metric_parameters={"quantile_returns_full": {"n_quantiles": 2}},
    )
    _assert_public_values(cpu, expected_shape)
    _assert_cuda_contract(result, cpu, SHAPE_METRICS)
    expected_q2 = _manual_profile(batch, label, 2)
    np.testing.assert_allclose(result.artifacts["quantile_returns_full"].values,
                               expected_q2, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("metric,kernel_name", [
    ("quantile_curvature", "quantile_curvature"),
    ("quantile_tail_asymmetry", "quantile_tail_asymmetry"),
    ("quantile_adjacent_spread", "quantile_adjacent_spread"),
    ("quantile_extreme_cliff", "quantile_extreme_cliff"),
    ("top_quantile_cliff", "top_quantile_cliff"),
    ("bottom_quantile_cliff", "bottom_quantile_cliff"),
])
def test_shape_kernels_can_return_device_arrays_without_host_materialization(
        metric, kernel_name):
    # Catches kernels that unconditionally call .get()/asnumpy on public
    # device-return requests; public bundle host conversion is tested above.
    from quant_evaluator.kernels.gpu import quantile_shape

    kernel = getattr(quantile_shape, kernel_name)
    profile = cp.asarray([[0.01, -0.02], [0.02, -0.01], [0.04, 0.03]],
                         dtype=cp.float64)
    result = kernel(profile, return_device=True)
    assert isinstance(result, cp.ndarray), metric
    assert result.shape == (2,)


def test_shape_repair_wrapper_only_gets_when_device_return_is_false(monkeypatch):
    # Catches an unconditional wrapper .get() even if a caller reuploads the
    # host result and disguises the round trip as a device return.
    from quant_evaluator.kernels.gpu import quantile_shape, shape_linear_numeric

    class DeviceSentinel:
        def __init__(self):
            self.get_calls = 0

        def get(self):
            self.get_calls += 1
            return np.array([7.0])

    device_result = DeviceSentinel()
    monkeypatch.setattr(
        shape_linear_numeric, "repair_shape_linear_gpu",
        lambda *_args, **_kwargs: device_result,
    )
    values = np.ones((3, 1), dtype=np.float64)
    result = np.ones(1, dtype=np.float64)
    assert quantile_shape._repair_linear(
        values, result, "curvature", 4096, return_device=True) is device_result
    assert device_result.get_calls == 0
    host_result = quantile_shape._repair_linear(
        values, result, "curvature", 4096, return_device=False)
    np.testing.assert_array_equal(host_result, [7.0])
    assert device_result.get_calls == 1
def test_shape_tile_results_and_counts_use_one_packed_host_transfer():
    # Catches one D2H conversion per metric instead of one bounded packed copy.
    from quant_evaluator.runtime.gpu_quantile_shape_adapter import (
        materialize_linear_quantile_shape_results,
    )

    values = cp.asarray([
        [1.0, cp.nan, 3.0],
        [cp.nan, 2.0, 4.0],
        [-1.0, -2.0, cp.nan],
        [0.0, 0.5, 1.5],
        [cp.nan, cp.nan, 7.0],
        [2.0, 3.0, cp.nan],
    ], dtype=cp.float64)
    expected = np.asarray([
        [1.0, np.nan, 3.0],
        [np.nan, 2.0, 4.0],
        [-1.0, -2.0, np.nan],
        [0.0, 0.5, 1.5],
        [np.nan, np.nan, 7.0],
        [2.0, 3.0, np.nan],
    ], dtype=np.float64)
    results = {name: values[i] for i, name in enumerate(SHAPE_METRICS)}
    host_calls = []
    host_arrays = []

    def to_host(packed):
        host_calls.append((packed.shape, packed.dtype))
        assert isinstance(packed, cp.ndarray)
        host = cp.asnumpy(packed)
        host_arrays.append(host)
        return host

    scalars, counts = materialize_linear_quantile_shape_results(
        results, to_host=to_host)
    assert host_calls == [((2 * len(SHAPE_METRICS), 3), cp.float64)]
    assert tuple(scalars) == SHAPE_METRICS
    assert tuple(counts) == SHAPE_METRICS
    for index, name in enumerate(SHAPE_METRICS):
        np.testing.assert_allclose(scalars[name], expected[index],
                                   equal_nan=True)
        np.testing.assert_array_equal(counts[name],
                                      np.isfinite(expected[index]).astype(np.int64))
        assert not np.shares_memory(scalars[name], host_arrays[0][index])
        assert not np.shares_memory(counts[name], host_arrays[0][len(SHAPE_METRICS) + index])
    assert all(array.dtype == np.int64 for array in counts.values())


def test_shape_tile_materializer_rejects_invalid_results_before_host_copy():
    from quant_evaluator.runtime.gpu_quantile_shape_adapter import (
        materialize_linear_quantile_shape_results,
    )

    def forbidden_to_host(_packed):
        pytest.fail("invalid or empty result requested host materialization")

    assert materialize_linear_quantile_shape_results(
        {}, to_host=forbidden_to_host) == ({}, {})
    with pytest.raises(ValueError, match="unknown linear"):
        materialize_linear_quantile_shape_results(
            {"not_a_shape_metric": object()}, to_host=forbidden_to_host)
    with pytest.raises(ValueError, match="same-device float64 vectors"):
        materialize_linear_quantile_shape_results(
            {SHAPE_METRICS[0]: cp.zeros(3, dtype=cp.float32)},
            to_host=forbidden_to_host)


@pytest.mark.parametrize(
    "factor_validity, label_validity, expected_bins, expected_counts, expected_means",
    [
        (None, [True, True, True, True, True, False],
         [0, 0, 0, 1, 1, 1], [3, 2], [20.0, 45.0]),
        ([True, True, True, True, True, False], None,
         [0, 0, 1, 1, 1, -1], [2, 3], [15.0, 40.0]),
    ],
)
def test_factor_memberships_are_independent_of_label_validity(
        factor_validity, label_validity, expected_bins, expected_counts,
        expected_means):
    from quant_evaluator.metrics.quantile import (
        assign_quantiles_fast,
        compute_quantile_returns_optimized,
    )

    time_axis = AxisRef("time", "int64", 1, np.array([0], dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", 6, np.arange(6, dtype=np.int64))
    values = np.arange(6, dtype=np.float64).reshape(1, 6, 1)
    validity = (None if factor_validity is None else
                np.asarray(factor_validity, dtype=bool).reshape(1, 6, 1))
    batch = FactorBatch(("f",), time_axis, asset_axis, values,
                        validity=validity)
    labels = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
    label_mask = (None if label_validity is None else
                  np.asarray(label_validity, dtype=bool).reshape(1, 6))
    label = LabelBundle("y", labels, horizon=1, validity=label_mask,
                        asset_axis=asset_axis, decision_time=(0,),
                        label_start_time=(0,), label_end_time=(1,))

    factor_values = values if validity is None else np.where(validity, values, np.nan)
    bins = assign_quantiles_fast(factor_values, n_quantiles=2)
    returns, counts = compute_quantile_returns_optimized(
        batch, label, n_quantiles=2, min_assets=1)

    np.testing.assert_array_equal(bins[0, :, 0], expected_bins)
    np.testing.assert_array_equal(counts[0, :, 0], expected_counts)
    np.testing.assert_allclose(returns[0, :, 0], expected_means)
