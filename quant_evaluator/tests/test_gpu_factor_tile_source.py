"""One-session GPU factor-source tiles versus the existing in-memory path."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.device_session import DeviceEvaluationSession
from quant_evaluator.runtime.gpu_executor import GPUExecutor


class ArraySource:
    def __init__(self, batch):
        self.batch = batch
        self.factor_ids = batch.factor_ids
        self.time_axis = batch.time_axis
        self.asset_axis = batch.asset_axis
        self.dtype = batch.dtype
        self.snapshot_id = "verified-test-snapshot"
        self.max_tile_size = 2
        self.reads = []

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.batch.values[:, :, start:end],
            validity=self.batch.validity[:, :, start:end],
        )
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


def _inputs():
    rng = np.random.default_rng(991)
    x = rng.normal(size=(8, 48, 5))
    y = rng.normal(size=(8, 48))
    valid = rng.random(x.shape) > 0.07
    lvalid = rng.random(y.shape) > 0.04
    x[1, 4, 2] = np.nan
    times = np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]")
    assets = np.arange(48, dtype=np.int64)
    time_axis = AxisRef("time", "datetime64[D]", 8, times)
    asset_axis = AxisRef("asset", "int64", 48, assets)
    batch = FactorBatch(tuple(f"f{i}" for i in range(5)), time_axis, asset_axis,
                        x, validity=valid)
    label = LabelBundle(
        "ret", y, 1, decision_time=tuple(times),
        label_start_time=tuple(times),
        label_end_time=tuple(times + np.timedelta64(1, "D")),
        validity=lvalid, asset_axis=asset_axis,
    )
    return batch, label


def _run_memory(batch, label, metrics):
    with DeviceEvaluationSession(GPUExecutionPolicy()) as session:
        return GPUExecutor(session).run_tiled(batch, label, metrics)


def _run_source(source, label, metrics, **kwargs):
    with DeviceEvaluationSession(GPUExecutionPolicy()) as session:
        return GPUExecutor(session).run_source_tiled(source, label, metrics, **kwargs)


def test_source_tiles_match_memory_gpu_for_scalar_series_and_vector():
    batch, label = _inputs()
    source = ArraySource(batch)
    metrics = ("rank_ic", "rank_ic_series", "coverage", "quantile_spread")
    reference = _run_memory(batch, label, metrics)
    result = _run_source(source, label, metrics, max_tile_size=2)
    assert result.factor_ids == batch.factor_ids
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert result.metadata["factor_tiles_processed"] == 3
    assert result.metadata["source_snapshot_id"] == source.snapshot_id
    for field in ("scalar_metrics", "series_metrics", "vector_metrics", "observation_counts"):
        actual = getattr(result, field)
        expected = getattr(reference, field)
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_allclose(actual[name], expected[name],
                                       rtol=0, atol=1e-12, equal_nan=True)


def test_source_label_axis_mismatch_rejected_before_read():
    batch, label = _inputs()
    source = ArraySource(batch)
    source.asset_axis = AxisRef("asset", "int64", 48, np.arange(1, 49, dtype=np.int64))
    with pytest.raises(InvalidContractError, match="asset coordinates"):
        _run_source(source, label, ("rank_ic",))
    assert source.reads == []


def test_source_tiling_oom_retries_with_smaller_read(monkeypatch):
    batch, label = _inputs()
    source = ArraySource(batch)
    metrics = ("rank_ic",)
    reference = _run_memory(batch, label, metrics)
    original = GPUExecutor.run

    def run(self, factor_ids, metric_ids, label_id="next_ret"):
        if len(factor_ids) > 1:
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return original(self, factor_ids, metric_ids, label_id)

    monkeypatch.setattr(GPUExecutor, "run", run)
    result = _run_source(source, label, metrics, max_tile_size=2)
    assert source.reads[0] == (0, 2)
    assert source.reads[1] == (0, 1)
    assert result.metadata["factor_tiles_processed"] == 5
    np.testing.assert_allclose(result.scalar_metrics["rank_ic"],
                               reference.scalar_metrics["rank_ic"],
                               rtol=0, atol=1e-12, equal_nan=True)
