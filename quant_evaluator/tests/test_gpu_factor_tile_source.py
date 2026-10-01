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
    def __init__(self, batch, max_tile_size=2):
        self.batch = batch
        self.factor_ids = batch.factor_ids
        self.time_axis = batch.time_axis
        self.asset_axis = batch.asset_axis
        self.dtype = batch.dtype
        self.snapshot_id = "verified-test-snapshot"
        self.max_tile_size = max_tile_size
        self.reads = []

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.batch.values[:, :, start:end],
            validity=(None if self.batch.validity is None
                      else self.batch.validity[:, :, start:end]),
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


def _many_factor_inputs():
    rng = np.random.default_rng(20260929)
    x = rng.normal(size=(8, 48, 48))
    y = rng.normal(size=(8, 48))
    times = np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]")
    assets = np.arange(48, dtype=np.int64)
    time_axis = AxisRef("time", "datetime64[D]", 8, times)
    asset_axis = AxisRef("asset", "int64", 48, assets)
    batch = FactorBatch(tuple(f"f{i}" for i in range(48)), time_axis,
                        asset_axis, x)
    label = LabelBundle(
        "ret", y, 1, decision_time=tuple(times),
        label_start_time=tuple(times),
        label_end_time=tuple(times + np.timedelta64(1, "D")),
        asset_axis=asset_axis,
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


def test_source_tiling_oom_retries_without_rereading_one_pass_source(monkeypatch):
    batch, label = _inputs()
    class OnePassSource(ArraySource):
        def __init__(self, batch):
            super().__init__(batch)
            self.next_start = 0

        def read_tile(self, start, end):
            assert start == self.next_start, "source was reread after OOM"
            tile = super().read_tile(start, end)
            self.next_start = end
            return tile

    source = OnePassSource(batch)
    metrics = ("rank_ic",)
    reference = _run_memory(batch, label, metrics)
    original = GPUExecutor.run
    original_stage = DeviceEvaluationSession.stage_masked_factors
    stage_attempts = []
    attempts = []

    def stage_masked(self, values, validity, factor_ids, layout="T,F,N"):
        factor_ids = tuple(factor_ids)
        stage_attempts.append(factor_ids)
        if len(factor_ids) > 1:
            raise cp.cuda.memory.OutOfMemoryError(100, 100, 100)
        return original_stage(self, values, validity, factor_ids, layout)

    def run(self, factor_ids, metric_ids, label_id="next_ret"):
        attempts.append(tuple(factor_ids))
        return original(self, factor_ids, metric_ids, label_id)

    monkeypatch.setattr(DeviceEvaluationSession, "stage_masked_factors", stage_masked)
    monkeypatch.setattr(GPUExecutor, "run", run)
    result = _run_source(source, label, metrics, max_tile_size=2)
    assert source.reads == [(0, 2), (2, 3), (3, 4), (4, 5)]
    assert stage_attempts == [("f0", "f1"), ("f0",), ("f1",),
                              ("f2",), ("f3",), ("f4",)]
    assert attempts == [("f0",), ("f1",), ("f2",), ("f3",), ("f4",)]
    assert source.next_start == batch.num_factors
    assert result.metadata["factor_tiles_processed"] == 5
    np.testing.assert_allclose(result.scalar_metrics["rank_ic"],
                               reference.scalar_metrics["rank_ic"],
                               rtol=0, atol=1e-12, equal_nan=True)


def test_source_gpu_values_match_independent_public_cpu_evaluation():
    from quant_evaluator.runtime.evaluator import evaluate

    batch, label = _inputs()
    metrics = ("rank_ic", "rank_ic_series", "coverage", "quantile_spread")
    cpu = evaluate(batch, label, metrics=metrics, backend="cpu")
    gpu = _run_source(ArraySource(batch), label, metrics)
    for metric in metrics:
        actual = (gpu.series_metrics if metric == "rank_ic_series" else gpu.scalar_metrics)[metric]
        np.testing.assert_allclose(actual, cpu.artifacts[metric].values,
                                   rtol=1e-8, atol=1e-10, equal_nan=True)


def test_source_gpu_cpu_parity_on_constant_nan_tied_and_masked_factors():
    from quant_evaluator.runtime.evaluator import evaluate

    original, label = _inputs()
    values = np.array(original.values)
    validity = np.array(original.validity)
    values[:, :, 0] = 1.0
    values[:, :, 1] = np.nan
    values[:, :, 2] = np.round(values[:, :, 2], 1)
    validity[:, :, 3] = False
    batch = FactorBatch(original.factor_ids, original.time_axis,
                        original.asset_axis, values, validity=validity)
    metrics = ("rank_ic", "rank_ic_series", "coverage", "quantile_spread")
    cpu = evaluate(batch, label, metrics=metrics, backend="cpu")
    gpu = _run_source(ArraySource(batch), label, metrics)
    for metric in metrics:
        actual = (gpu.series_metrics if metric == "rank_ic_series" else gpu.scalar_metrics)[metric]
        np.testing.assert_allclose(actual, cpu.artifacts[metric].values,
                                   rtol=1e-8, atol=1e-10, equal_nan=True)


def test_many_factor_source_tiles_aggregate_to_one_cpu_request():
    from quant_evaluator.runtime.evaluator import evaluate

    batch, label = _many_factor_inputs()
    source = ArraySource(batch, max_tile_size=8)
    metrics = ("rank_ic", "quantile_spread", "factor_turnover_rate")
    cpu = evaluate(batch, label, metrics=metrics, backend="cpu")
    gpu = _run_source(source, label, metrics, max_tile_size=8)

    assert source.reads == [(start, start + 8) for start in range(0, 48, 8)]
    assert gpu.metadata["factor_tiles_processed"] == 6
    assert gpu.factor_ids == batch.factor_ids
    assert gpu.metadata["source_snapshot_id"] == source.snapshot_id
    for metric in metrics:
        np.testing.assert_allclose(
            gpu.scalar_metrics[metric], cpu.artifacts[metric].values,
            rtol=1e-8, atol=1e-10, equal_nan=True,
        )
        np.testing.assert_array_equal(
            gpu.observation_counts[metric],
            [cpu.grouped_metrics[fid][metric].observation_count
             for fid in batch.factor_ids],
        )
