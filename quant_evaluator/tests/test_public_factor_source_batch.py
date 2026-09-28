"""Public bounded source-batch CPU/GPU and option contracts."""
import numpy as np
import pytest

from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.errors import InvalidContractError, UnsupportedMetricError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


class Source:
    def __init__(self, batch):
        self.batch = batch
        self.factor_ids = batch.factor_ids
        self.time_axis = batch.time_axis
        self.asset_axis = batch.asset_axis
        self.dtype = batch.dtype
        self.snapshot_id = "verified-test-source"
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
    rng = np.random.default_rng(983)
    T, N, F = 8, 48, 5
    x = rng.normal(size=(T, N, F))
    x[:, :, 0] = 1.0
    x[1, 2, 1] = np.nan
    valid = rng.random(x.shape) > 0.08
    y = rng.normal(size=(T, N))
    lvalid = rng.random(y.shape) > 0.06
    times = np.arange(T, dtype=np.int64)
    time_axis = AxisRef("time", "int64", T, times)
    asset_axis = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
    batch = FactorBatch(tuple(f"f{i}" for i in range(F)), time_axis, asset_axis,
                        x, validity=valid)
    label = LabelBundle("ret", y, 1, decision_time=tuple(times),
                        label_start_time=tuple(times),
                        label_end_time=tuple(times + 1),
                        validity=lvalid, asset_axis=asset_axis)
    return batch, label


def _assert_against_full_cpu(result, reference, metrics):
    for metric in metrics:
        group = result.series_metrics if metric == "rank_ic_series" else result.scalar_metrics
        np.testing.assert_allclose(group[metric], reference.artifacts[metric].values,
                                   rtol=1e-8, atol=1e-10, equal_nan=True)
        expected_counts = []
        for index, fid in enumerate(reference.factor_ids):
            if metric == "rank_ic_series":
                expected_counts.append(int(np.isfinite(reference.artifacts[metric].values[:, index]).sum()))
            else:
                expected_counts.append(reference.grouped_metrics[fid][metric].observation_count)
        np.testing.assert_array_equal(result.observation_counts[metric], expected_counts)


@pytest.mark.parametrize("backend", ["cpu", "auto", "cuda_strict"])
def test_public_source_batch_matches_full_cpu(backend):
    if backend == "cuda_strict":
        pytest.importorskip("cupy")
    batch, label = _inputs()
    metrics = ("rank_ic", "rank_ic_series", "coverage", "quantile_spread")
    reference = evaluate(batch, label, metrics=metrics, backend="cpu")
    source = Source(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend=backend, max_tile_size=2)
    _assert_against_full_cpu(result, reference, metrics)
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert result.factor_ids == batch.factor_ids
    assert result.metadata["source_snapshot_id"] == source.snapshot_id
    assert result.metadata["backend_used"] == ("cuda" if backend == "cuda_strict" else "cpu")
    assert result.to_dict()["series_metrics"]["rank_ic_series"]


def test_public_source_batch_invalid_options_fail_before_reads():
    batch, label = _inputs()
    source = Source(batch)
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    for kwargs, error in (
        ({"metrics": ("rank_ic", "rank_ic")}, InvalidContractError),
        ({"metrics": "rank_ic"}, InvalidContractError),
        ({"metrics": None}, InvalidContractError),
        ({"metrics": ("not_a_metric",)}, UnsupportedMetricError),
        ({"backend": "cuda"}, InvalidContractError),
        ({"max_tile_size": 0}, InvalidContractError),
        ({"gpu_policy": False}, InvalidContractError),
        ({"gpu_policy": GPUExecutionPolicy(precision_policy="invalid")}, InvalidContractError),
    ):
        with pytest.raises(error):
            evaluate_factor_source_batch(source, label, **kwargs)
    assert source.reads == []


def test_auto_source_exact_profile_uses_cuda_when_gate_passes(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")

    batch, label = _inputs()
    monkeypatch.setattr(source_api, "_F8_SHAPE", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *_: None)
    result = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"))
    reference = evaluate(batch, label, metrics=("rank_ic", "rank_ic_series"), backend="cpu")
    _assert_against_full_cpu(result, reference, ("rank_ic", "rank_ic_series"))
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "bounded_f8_rank_pair_gpu"


def test_auto_source_resource_rejection_falls_back_to_cpu(monkeypatch):
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")

    batch, label = _inputs()
    monkeypatch.setattr(source_api, "_F8_SHAPE", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    result = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"))
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_auto_source_nondefault_precision_does_not_use_uncertified_gpu(monkeypatch):
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    source_api = importlib.import_module("quant_evaluator.api.factor_source")

    batch, label = _inputs()
    monkeypatch.setattr(source_api, "_F8_SHAPE", batch.values.shape)
    policy = GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64)
    result = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"), gpu_policy=policy)
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"


def test_source_batch_requires_typed_label_before_read():
    batch, _ = _inputs()
    source = Source(batch)
    with pytest.raises(InvalidContractError, match="typed LabelBundle"):
        evaluate_factor_source_batch(source, object())
    assert source.reads == []


def test_source_request_fingerprint_is_backend_and_tile_invariant():
    batch, label = _inputs()
    first = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"),
        backend="cpu", max_tile_size=1)
    second = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"),
        backend="cpu", max_tile_size=2)
    assert first.metadata["source_request_fingerprint"] == second.metadata["source_request_fingerprint"]
    assert first.metadata["execution_receipt"]["receipt_hash"] != second.metadata["execution_receipt"]["receipt_hash"]
    assert first.metadata["request_id"] != second.metadata["request_id"]

    changed = Source(batch)
    changed.snapshot_id = "another-verified-test-source"
    third = evaluate_factor_source_batch(
        changed, label, metrics=("rank_ic", "rank_ic_series"), backend="cpu")
    assert third.metadata["source_request_fingerprint"] != first.metadata["source_request_fingerprint"]


def test_source_request_fingerprint_accepts_datetime_axes():
    batch, _ = _inputs()
    dates = np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]")
    time_axis = AxisRef("time", "datetime64[D]", len(dates), dates)
    dated = FactorBatch(batch.factor_ids, time_axis, batch.asset_axis,
                        batch.values, validity=batch.validity)
    label = LabelBundle(
        "ret", np.ones((8, 48)), 1, decision_time=tuple(dates),
        label_start_time=tuple(dates),
        label_end_time=tuple(dates + np.timedelta64(1, "D")),
        asset_axis=batch.asset_axis,
    )
    result = evaluate_factor_source_batch(
        Source(dated), label, metrics=("rank_ic",), backend="cpu")
    assert len(result.metadata["source_request_fingerprint"]) == 64


def test_source_request_fingerprint_matches_cpu_and_cuda():
    pytest.importorskip("cupy")
    batch, label = _inputs()
    cpu = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"), backend="cpu")
    cuda = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"), backend="cuda_strict")
    assert cpu.metadata["source_request_fingerprint"] == cuda.metadata["source_request_fingerprint"]
    assert cpu.metadata["execution_receipt"]["receipt_hash"] != cuda.metadata["execution_receipt"]["receipt_hash"]
