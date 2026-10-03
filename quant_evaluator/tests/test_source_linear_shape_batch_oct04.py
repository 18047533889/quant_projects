"""Bounded CPU source evaluation for the linear quantile-shape metrics."""

import numpy as np
import pytest

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.errors import UnsupportedMetricError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


LINEAR_SHAPE_METRICS = (
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
)


class _TinySource:
    def __init__(self, values, time_axis, asset_axis, factor_ids, max_tile_size=2):
        self.values = values
        self.time_axis = time_axis
        self.asset_axis = asset_axis
        self.factor_ids = tuple(factor_ids)
        self.dtype = "float64"
        self.snapshot_id = "tiny-linear-shape-source-v1"
        self.max_tile_size = max_tile_size
        self.reads = []
        self.closed = False

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.values[:, :, start:end],
        )
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        self.closed = True


def _inputs():
    rng = np.random.default_rng(20261004)
    t, n, f = 32, 72, 3
    times = np.arange(t, dtype=np.int64)
    assets = AxisRef("asset", "str", n,
                     np.asarray([f"A{i:03d}" for i in range(n)]))
    time_axis = AxisRef("time", "int64", t, times)
    values = rng.normal(size=(t, n, f))
    values[2, 0, 0] = np.nan
    values[7, 4, 1] = np.nan
    values[11, 25, 2] = np.nan
    factors = FactorBatch(tuple(f"f{i}" for i in range(f)), time_axis,
                          assets, values)
    label_values = rng.normal(size=(t, n))
    label_values[4, 3] = np.nan
    validity = rng.random((t, n)) > 0.08
    labels = LabelBundle(
        "tiny-label", label_values, 1,
        decision_time=tuple(times), observation_time=tuple(times),
        signal_available_time=tuple(times), execution_time=tuple(times),
        label_start_time=tuple(times), label_end_time=tuple(times + 1),
        asset_axis=assets, validity=validity,
    )
    source = _TinySource(values, time_axis, assets, factors.factor_ids)
    return factors, labels, source


def test_cpu_source_linear_shape_metrics_match_manual_tile_evaluations():
    factors, labels, source = _inputs()
    try:
        result = evaluate_factor_source_batch(
            source, labels, metrics=LINEAR_SHAPE_METRICS, backend="cpu",
            max_tile_size=2,
        )
    finally:
        source.close()

    assert source.closed
    assert source.reads == [(0, 2), (2, 3)]
    assert result.scalar_metrics.keys() == set(LINEAR_SHAPE_METRICS)
    assert result.series_metrics == {}
    assert result.metadata["factor_tiles_processed"] == 2
    assert result.metadata["host_result_bytes_reserved"] <= \
        result.metadata["host_result_budget_bytes"]

    manual_values = {metric: np.empty(len(factors.factor_ids))
                     for metric in LINEAR_SHAPE_METRICS}
    manual_counts = {metric: np.empty(len(factors.factor_ids), dtype=np.int64)
                     for metric in LINEAR_SHAPE_METRICS}
    for start, end in ((0, 2), (2, 3)):
        tile = FactorBatch(
            factors.factor_ids[start:end], factors.time_axis,
            factors.asset_axis, factors.values[:, :, start:end],
        )
        expected = evaluate(tile, labels, metrics=LINEAR_SHAPE_METRICS, backend="cpu")
        for metric in LINEAR_SHAPE_METRICS:
            artifact = expected.artifacts[metric]
            assert artifact.artifact_kind == "scalar"
            manual_values[metric][start:end] = artifact.values
            for local_index, factor_id in enumerate(tile.factor_ids):
                manual_counts[metric][start + local_index] = (
                    expected.grouped_metrics[factor_id][metric].observation_count)

    for metric in LINEAR_SHAPE_METRICS:
        np.testing.assert_allclose(result.scalar_metrics[metric], manual_values[metric],
                                   rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(result.observation_counts[metric],
                                      manual_counts[metric])
        assert np.isfinite(result.scalar_metrics[metric]).any()


def test_unqualified_auto_keeps_new_shape_request_on_cpu():
    _, labels, source = _inputs()
    try:
        result = evaluate_factor_source_batch(
            source, labels, metrics=LINEAR_SHAPE_METRICS, backend="auto",
            max_tile_size=2, gpu_policy=GPUExecutionPolicy(),
        )
    finally:
        source.close()

    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    assert result.metadata["source_qualification_applied"] is False
    assert result.metadata["source_qualification_winner"] is None
    assert source.reads == [(0, 2), (2, 3)]


def test_vector_quantile_metric_remains_unsupported_before_source_read():
    _, labels, source = _inputs()
    try:
        with pytest.raises(UnsupportedMetricError, match="quantile_returns_daily"):
            evaluate_factor_source_batch(
                source, labels, metrics=("quantile_returns_daily",),
                backend="cpu", max_tile_size=2,
            )
    finally:
        source.close()

    assert source.reads == []
