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


def _patch_source_evidence_shape(monkeypatch, family, shape):
    """Inject small test shapes through the evidence registry itself."""
    from dataclasses import replace
    import importlib

    registry = importlib.import_module("quant_evaluator.runtime.source_auto_evidence")
    records = []
    for item in registry.SOURCE_AUTO_EVIDENCE:
        if family in item.evidence_id:
            adjusted_shape = item.shape
            if isinstance(adjusted_shape, frozenset):
                adjusted_shape = frozenset({shape})
            else:
                adjusted_shape = shape
            item = replace(item, shape=adjusted_shape)
        records.append(item)
    monkeypatch.setattr(registry, "SOURCE_AUTO_EVIDENCE", tuple(records))


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
    if backend == "auto":
        assert result.metadata["execution_receipt"]["auto_backend_evidence_id"] is None
        assert result.metadata["execution_receipt"]["auto_backend_evidence_version"] is None
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
    _patch_source_evidence_shape(monkeypatch, "f8", batch.values.shape)
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
    _patch_source_evidence_shape(monkeypatch, "f8", batch.values.shape)
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
    _patch_source_evidence_shape(monkeypatch, "f8", batch.values.shape)
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

_EXTENDED_SOURCE_METRICS = (
    "ic_ir", "ic_std", "ic_median", "pearson_ic", "pearson_ic_series",
    "pearson_ic_std", "pearson_ic_ir", "factor_turnover_rate", "turnover",
    "quantile_monotonicity", "daily_quantile_monotonicity_rate",
)


def test_all_public_source_metrics_can_share_one_cpu_cuda_request():
    pytest.importorskip("cupy")
    from quant_evaluator.scripts.benchmark_real_cos_source_batch import ALL_SOURCE_METRICS

    batch, label = _inputs()
    cpu = evaluate_factor_source_batch(
        Source(batch), label, metrics=ALL_SOURCE_METRICS, backend="cpu")
    cuda = evaluate_factor_source_batch(
        Source(batch), label, metrics=ALL_SOURCE_METRICS, backend="cuda_strict")
    for metric in ALL_SOURCE_METRICS:
        cpu_values = (cpu.series_metrics if metric in cpu.series_metrics
                      else cpu.scalar_metrics)[metric]
        cuda_values = (cuda.series_metrics if metric in cuda.series_metrics
                       else cuda.scalar_metrics)[metric]
        np.testing.assert_allclose(cpu_values, cuda_values, rtol=1e-8, atol=1e-10,
                                   equal_nan=True)
        np.testing.assert_array_equal(cpu.observation_counts[metric],
                                      cuda.observation_counts[metric])
    assert cpu.metadata["source_request_fingerprint"] == cuda.metadata["source_request_fingerprint"]


def test_auto_f61_all_source_profile_uses_cuda_only_with_certified_gate(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    from quant_evaluator.scripts.benchmark_real_cos_source_batch import ALL_SOURCE_METRICS

    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    admitted = []
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda _policy, minimum: admitted.append(minimum) or None)
    source = Source(batch)
    source.max_tile_size = 32
    metrics = tuple(reversed(ALL_SOURCE_METRICS))
    routed = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend="auto", max_tile_size=32)
    cpu = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="cpu")
    assert routed.metadata["backend_used"] == "cuda"
    assert routed.metadata["auto_backend_reason"] == "bounded_f61_all_source_15_gpu_tile16"
    assert routed.metadata["effective_max_tile_size"] == 16
    assert admitted == [14 * 1024 ** 3]
    for metric in metrics:
        cpu_values = (cpu.series_metrics if metric in cpu.series_metrics else cpu.scalar_metrics)[metric]
        gpu_values = (routed.series_metrics if metric in routed.series_metrics
                      else routed.scalar_metrics)[metric]
        np.testing.assert_allclose(cpu_values, gpu_values, rtol=1e-8, atol=1e-10,
                                   equal_nan=True)
        np.testing.assert_array_equal(cpu.observation_counts[metric],
                                      routed.observation_counts[metric])

    for changed_metrics, width in ((metrics[:-1], 16), (metrics, 8)):
        candidate = Source(batch)
        candidate.max_tile_size = 32
        rejected = evaluate_factor_source_batch(
            candidate, label, metrics=changed_metrics, backend="auto", max_tile_size=width)
        assert rejected.metadata["backend_used"] == "cpu"
    assert admitted == [14 * 1024 ** 3]

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    candidate = Source(batch)
    candidate.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        candidate, label, metrics=metrics, backend="auto", max_tile_size=16)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"

    _patch_source_evidence_shape(
        monkeypatch, "f61", (batch.values.shape[0] + 1, *batch.values.shape[1:]))
    candidate = Source(batch)
    candidate.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        candidate, label, metrics=metrics, backend="auto", max_tile_size=16)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

@pytest.mark.parametrize("case", ["mixed", "all_missing", "tied", "masked"])
def test_extended_source_metrics_cpu_cuda_parity(case):
    pytest.importorskip("cupy")
    batch, label = _inputs()
    if case != "mixed":
        values = batch.values.copy()
        validity = batch.validity.copy()
        if case == "all_missing":
            values[:] = np.nan
        elif case == "tied":
            values[:] = 1.0
        else:
            validity[:, :, 0] = False
        batch = FactorBatch(batch.factor_ids, batch.time_axis, batch.asset_axis,
                            values, validity=validity)
    reference = evaluate(batch, label, metrics=_EXTENDED_SOURCE_METRICS, backend="cpu")
    cpu = evaluate_factor_source_batch(Source(batch), label, metrics=_EXTENDED_SOURCE_METRICS, backend="cpu")
    cuda = evaluate_factor_source_batch(Source(batch), label, metrics=_EXTENDED_SOURCE_METRICS, backend="cuda_strict")
    for metric in _EXTENDED_SOURCE_METRICS:
        artifact = reference.artifacts[metric]
        group = cpu.series_metrics if artifact.artifact_kind == "series" else cpu.scalar_metrics
        gpu_group = cuda.series_metrics if artifact.artifact_kind == "series" else cuda.scalar_metrics
        np.testing.assert_allclose(group[metric], artifact.values, rtol=1e-8, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(gpu_group[metric], artifact.values, rtol=1e-8, atol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(cpu.observation_counts[metric], cuda.observation_counts[metric])


def test_auto_source_f32_pair_uses_cuda_only_for_certified_tile_width(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f32", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *_: None)
    metrics = ("rank_ic", "rank_ic_series")
    cuda = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="auto")
    cpu = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="cpu")
    _assert_against_full_cpu(cuda, evaluate(batch, label, metrics=metrics, backend="cpu"), metrics)
    for metric in metrics:
        group = cuda.series_metrics if metric.endswith("_series") else cuda.scalar_metrics
        np.testing.assert_allclose(
            group[metric], (cpu.series_metrics if metric.endswith("_series") else cpu.scalar_metrics)[metric],
            rtol=1e-8, atol=1e-10, equal_nan=True)
    assert cuda.metadata["backend_used"] == "cuda"
    assert cuda.metadata["auto_backend_reason"] == "bounded_f32_rank_pair_gpu"
    narrow = evaluate_factor_source_batch(Source(batch), label, metrics=metrics,
                                          backend="auto", max_tile_size=1)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="auto")
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_auto_source_f32_mixed_three_uses_cuda_under_exact_gate(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f32", batch.values.shape)
    admitted = []
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda _policy, minimum: admitted.append(minimum) or None,
    )
    metrics = ("factor_turnover_rate", "rank_ic", "quantile_spread")
    source = Source(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend="auto", max_tile_size=2)
    reference = evaluate(batch, label, metrics=metrics, backend="cpu")
    _assert_against_full_cpu(result, reference, metrics)
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert admitted == [14 * 1024 ** 3]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "bounded_f32_mixed_three_gpu"


def test_auto_source_f32_mixed_three_fails_closed_outside_gate(monkeypatch):
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f32", batch.values.shape)
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("uncertified request must not probe CUDA admission"),
    )
    metrics = ("rank_ic", "quantile_spread", "factor_turnover_rate")
    narrow = evaluate_factor_source_batch(
        Source(batch), label, metrics=metrics, backend="auto", max_tile_size=1)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    wrong_metrics = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "quantile_spread"), backend="auto")
    assert wrong_metrics.metadata["backend_used"] == "cpu"
    assert wrong_metrics.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    nondefault = evaluate_factor_source_batch(
        Source(batch), label, metrics=metrics, backend="auto",
        gpu_policy=GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64),
    )
    assert nondefault.metadata["backend_used"] == "cpu"
    assert nondefault.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected = evaluate_factor_source_batch(
        Source(batch), label, metrics=metrics, backend="auto")
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"

def test_auto_source_f61_mixed_three_uses_cuda_under_exact_gate(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    admitted = []
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda _policy, minimum: admitted.append(minimum) or None,
    )
    metrics = ("factor_turnover_rate", "rank_ic", "quantile_spread")
    source = Source(batch)
    source.max_tile_size = 8
    result = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend="auto", max_tile_size=8)
    reference = evaluate(batch, label, metrics=metrics, backend="cpu")
    _assert_against_full_cpu(result, reference, metrics)
    assert source.reads == [(0, 5)]
    assert admitted == [14 * 1024 ** 3]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "bounded_f61_mixed_three_gpu"
    assert result.metadata["effective_max_tile_size"] == 8


@pytest.mark.parametrize("cap,expected", [(8, 8), (15, 8), (16, 16), (32, 16)])
def test_auto_source_f61_selects_certified_width_within_cap(monkeypatch, cap, expected):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    seed, label = _inputs()
    batch = FactorBatch(
        tuple(f"wide_{i}" for i in range(21)), seed.time_axis, seed.asset_axis,
        np.tile(seed.values, (1, 1, 5))[:, :, :21],
        validity=np.tile(seed.validity, (1, 1, 5))[:, :, :21],
    )
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *_: None)
    source = Source(batch)
    source.max_tile_size = 32
    result = evaluate_factor_source_batch(
        source, label, metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
        backend="auto", max_tile_size=cap)
    assert source.reads == [(start, min(start + expected, 21)) for start in range(0, 21, expected)]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["effective_max_tile_size"] == expected
    assert result.metadata["execution_receipt"]["effective_max_tile_size"] == expected
    assert result.metadata["execution_receipt"]["max_tile_size"] == cap
    assert all(end - start <= expected for start, end in source.reads)
    assert result.metadata["auto_backend_reason"] == (
        "bounded_f61_mixed_three_gpu_tile16" if expected == 16
        else "bounded_f61_mixed_three_gpu")


def test_auto_source_f61_pearson_single_uses_certified_tile_and_fails_closed(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    admitted = []
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda _policy, minimum: admitted.append(minimum) or None,
    )
    source = Source(batch)
    source.max_tile_size = 32
    result = evaluate_factor_source_batch(
        source, label, metrics=("pearson_ic",), backend="auto", max_tile_size=32)
    reference = evaluate(batch, label, metrics=("pearson_ic",), backend="cpu")
    _assert_against_full_cpu(result, reference, ("pearson_ic",))
    assert admitted == [14 * 1024 ** 3]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "bounded_f61_pearson_ic_gpu_tile16"
    assert result.metadata["effective_max_tile_size"] == 16

    narrow = evaluate_factor_source_batch(
        Source(batch), label, metrics=("pearson_ic",), backend="auto",
        max_tile_size=8)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    assert admitted == [14 * 1024 ** 3]

    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda _policy, _minimum: "insufficient_cuda_memory",
    )
    rejected_source = Source(batch)
    rejected_source.max_tile_size = 16
    rejected = evaluate_factor_source_batch(
        rejected_source, label, metrics=("pearson_ic",), backend="auto",
        max_tile_size=16)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_auto_source_f61_pearson_chain_exact_route_and_negative_cases(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    gates = []
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda _policy, minimum: gates.append(minimum) or None)
    metrics = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
    source = Source(batch)
    source.max_tile_size = 32
    routed = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend="auto", max_tile_size=32)
    cpu = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="cpu")
    for metric in metrics:
        group = "series_metrics" if metric.endswith("_series") else "scalar_metrics"
        np.testing.assert_allclose(getattr(routed, group)[metric], getattr(cpu, group)[metric],
                                   rtol=1e-8, atol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(routed.observation_counts[metric], cpu.observation_counts[metric])
    assert gates == [14 * 1024 ** 3]
    assert routed.metadata["backend_used"] == "cuda"
    assert routed.metadata["effective_max_tile_size"] == 16
    assert routed.metadata["auto_backend_reason"] == "bounded_f61_pearson_chain_gpu_tile16"

    reversed_metrics = tuple(reversed(metrics))
    reversed_source = Source(batch)
    reversed_source.max_tile_size = 32
    reversed_routed = evaluate_factor_source_batch(
        reversed_source, label, metrics=reversed_metrics, backend="auto", max_tile_size=32)
    reversed_cpu = evaluate_factor_source_batch(
        Source(batch), label, metrics=reversed_metrics, backend="cpu")
    for metric in reversed_metrics:
        group = "series_metrics" if metric.endswith("_series") else "scalar_metrics"
        np.testing.assert_allclose(getattr(reversed_routed, group)[metric],
                                   getattr(reversed_cpu, group)[metric],
                                   rtol=1e-8, atol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(reversed_routed.observation_counts[metric],
                                      reversed_cpu.observation_counts[metric])
    assert reversed_routed.metadata["backend_used"] == "cuda"
    assert reversed_routed.metadata["auto_backend_reason"] == "bounded_f61_pearson_chain_gpu_tile16"

    assert reversed_routed.metadata["execution_receipt"]["metric_backends"] == {
        metric: "cuda" for metric in reversed_metrics
    }
    assert routed.metadata["source_request_fingerprint"] != reversed_routed.metadata["source_request_fingerprint"]
    assert routed.metadata["execution_receipt"]["receipt_hash"] != reversed_routed.metadata["execution_receipt"]["receipt_hash"]
    for changed_metrics, width in ((metrics, 8), (metrics[:-1], 16)):
        case = Source(batch)
        case.max_tile_size = 32
        rejected = evaluate_factor_source_batch(
            case, label, metrics=changed_metrics, backend="auto", max_tile_size=width)
        assert rejected.metadata["backend_used"] == "cpu"
        assert rejected.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    assert gates == [14 * 1024 ** 3, 14 * 1024 ** 3]

    policy = GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64)
    case = Source(batch)
    case.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        case, label, metrics=metrics, backend="auto", max_tile_size=16, gpu_policy=policy)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    case = Source(batch)
    case.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        case, label, metrics=metrics, backend="auto", max_tile_size=16)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_auto_source_f61_mixed_three_fails_closed_outside_gate(monkeypatch):
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    monkeypatch.setattr(
        evaluator, "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("uncertified request must not probe CUDA admission"),
    )
    metrics = ("rank_ic", "quantile_spread", "factor_turnover_rate")
    source = Source(batch)
    source.max_tile_size = 8
    narrow = evaluate_factor_source_batch(
        source, label, metrics=metrics, backend="auto", max_tile_size=2)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    wrong_source = Source(batch)
    wrong_source.max_tile_size = 8
    wrong_metrics = evaluate_factor_source_batch(
        wrong_source, label, metrics=("rank_ic", "quantile_spread"),
        backend="auto", max_tile_size=8)
    assert wrong_metrics.metadata["backend_used"] == "cpu"
    assert wrong_metrics.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    policy_source = Source(batch)
    policy_source.max_tile_size = 8
    nondefault = evaluate_factor_source_batch(
        policy_source, label, metrics=metrics, backend="auto", max_tile_size=8,
        gpu_policy=GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64),
    )
    assert nondefault.metadata["backend_used"] == "cpu"
    assert nondefault.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected_source = Source(batch)
    rejected_source.max_tile_size = 8
    rejected = evaluate_factor_source_batch(
        rejected_source, label, metrics=metrics, backend="auto", max_tile_size=8)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"
