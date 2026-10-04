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


def _patch_source_evidence_shape(monkeypatch, family, shape, *, evidence_status=None):
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
            if evidence_status is not None:
                item = replace(item, evidence_status=evidence_status)
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
def test_source_memory_hint_bounds_execution_without_changing_declared_cap(backend):
    if backend == "cuda_strict":
        pytest.importorskip("cupy")
    batch, label = _inputs()
    source = Source(batch)
    source.max_tile_size = 5
    source.admitted_max_tile_size = 1
    metrics = ("rank_ic", "rank_ic_series", "coverage")
    result = evaluate_factor_source_batch(source, label, metrics=metrics, backend=backend)
    assert source.max_tile_size == 5
    assert source.reads == [(i, i + 1) for i in range(5)]
    assert result.metadata["effective_max_tile_size"] == 1
    assert result.metadata["admitted_source_tile_size"] == 1
    assert result.metadata["execution_receipt"]["admitted_source_tile_size"] == 1
    _assert_against_full_cpu(result, evaluate(batch, label, metrics=metrics, backend="cpu"), metrics)


@pytest.mark.parametrize("hint", [0, -1, 6, True, 1.0, "1"])
def test_invalid_source_memory_hint_fails_before_reads(hint):
    batch, label = _inputs()
    source = Source(batch)
    source.max_tile_size = 5
    source.admitted_max_tile_size = hint
    with pytest.raises(InvalidContractError, match="admitted source tile width"):
        evaluate_factor_source_batch(source, label, backend="cpu")
    assert source.reads == []


def test_source_memory_hint_revokes_gpu_width_instead_of_silently_reducing_it(monkeypatch):
    import quant_evaluator.runtime.evaluator as evaluator
    batch, label = _inputs()
    source = Source(batch)
    source.admitted_max_tile_size = 1
    _patch_source_evidence_shape(monkeypatch, "synthetic_f32_rank_pair_tile2", (8, 48, 5))
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *args: None)
    result = evaluate_factor_source_batch(source, label, metrics=("rank_ic", "rank_ic_series"), source_auto_policy="legacy_measured")
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "source_memory_budget_outside_certified_tile"
    assert result.metadata["effective_max_tile_size"] == 1
    assert source.reads == [(i, i + 1) for i in range(5)]


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


def test_default_auto_policy_skips_legacy_static_selector_for_known_envelope(monkeypatch):
    """A measured static envelope must not select CUDA without explicit authority."""
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    metrics = ("rank_ic", "rank_ic_series")
    _patch_source_evidence_shape(monkeypatch, "synthetic_f32_rank_pair_tile2", batch.values.shape)
    monkeypatch.setattr(source_api, "get_cached_source_route_profile_records", lambda **kwargs: None)
    monkeypatch.setattr(source_api, "qualify_source_route",
        lambda **kwargs: (_ for _ in ()).throw(
            source_api.SourceQualificationError("qualified_cache_miss")))
    gate_calls = []
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
        lambda *_: gate_calls.append(True) or "insufficient_cuda_memory")

    result = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="auto")
    receipt = result.metadata["execution_receipt"]

    assert result.metadata["backend_used"] == "cpu"
    assert receipt["auto_backend_reason"] == "source_qualification_required"
    assert receipt["auto_backend_evidence_id"] is None
    assert gate_calls == []


@pytest.mark.parametrize("invalid_policy", [None, True, "legacy", "QUALIFIED_ONLY"])
def test_invalid_source_auto_policy_fails_before_source_reads(invalid_policy):
    batch, label = _inputs()
    source = Source(batch)
    with pytest.raises(InvalidContractError):
        evaluate_factor_source_batch(source, label, backend="auto",
                                     source_auto_policy=invalid_policy)
    assert source.reads == []


def test_auto_source_legacy_f8_profile_falls_back_to_cpu(monkeypatch):
    batch, label = _inputs()
    _patch_source_evidence_shape(
        monkeypatch, "f8", batch.values.shape,
        evidence_status="legacy_unverified_source_performance")
    result = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "rank_ic_series"), source_auto_policy="legacy_measured")
    reference = evaluate(batch, label, metrics=("rank_ic", "rank_ic_series"), backend="cpu")
    _assert_against_full_cpu(result, reference, ("rank_ic", "rank_ic_series"))
    receipt = result.metadata["execution_receipt"]
    assert result.metadata["backend_used"] == "cpu"
    assert receipt["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    assert receipt["auto_backend_evidence_id"] is None


def test_auto_source_resource_rejection_falls_back_to_cpu(monkeypatch):
    import importlib
    source_api = importlib.import_module("quant_evaluator.api.factor_source")
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")

    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    source = Source(batch)
    source.max_tile_size = 16
    result = evaluate_factor_source_batch(
        source, label,
        metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
        max_tile_size=16, source_auto_policy="legacy_measured")
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_auto_source_nondefault_precision_does_not_use_uncertified_gpu(monkeypatch):
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    source_api = importlib.import_module("quant_evaluator.api.factor_source")

    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f61", batch.values.shape)
    source = Source(batch)
    source.max_tile_size = 16
    policy = GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64)
    result = evaluate_factor_source_batch(
        source, label,
        metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
        max_tile_size=16, gpu_policy=policy, source_auto_policy="legacy_measured")
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
        source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=32)
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
            candidate, label, metrics=changed_metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=width)
        assert rejected.metadata["backend_used"] == "cpu"
    assert admitted == [14 * 1024 ** 3]

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    candidate = Source(batch)
    candidate.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        candidate, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=16)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"

    _patch_source_evidence_shape(
        monkeypatch, "f61", (batch.values.shape[0] + 1, *batch.values.shape[1:]))
    candidate = Source(batch)
    candidate.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        candidate, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=16)
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
    cuda = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured")
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
                                          backend="auto", source_auto_policy="legacy_measured", max_tile_size=1)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected = evaluate_factor_source_batch(Source(batch), label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured")
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
        source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=2)
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
        Source(batch), label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=1)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    wrong_metrics = evaluate_factor_source_batch(
        Source(batch), label, metrics=("rank_ic", "quantile_spread"), backend="auto", source_auto_policy="legacy_measured")
    assert wrong_metrics.metadata["backend_used"] == "cpu"
    assert wrong_metrics.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    nondefault = evaluate_factor_source_batch(
        Source(batch), label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured",
        gpu_policy=GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64),
    )
    assert nondefault.metadata["backend_used"] == "cpu"
    assert nondefault.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected = evaluate_factor_source_batch(
        Source(batch), label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured")
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"

def test_auto_source_f8_default_cap_uses_measured_tile2_and_preserves_request(monkeypatch):
    pytest.importorskip("cupy")
    import importlib
    evaluator = importlib.import_module("quant_evaluator.runtime.evaluator")
    batch, label = _inputs()
    _patch_source_evidence_shape(monkeypatch, "f8", batch.values.shape)
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *_: None)
    metrics = ("rank_ic", "rank_ic_series")
    source = Source(batch)
    source.max_tile_size = 8
    routed = evaluate_factor_source_batch(source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured")
    explicit = Source(batch)
    explicit.max_tile_size = 8
    reference = evaluate_factor_source_batch(
        explicit, label, metrics=metrics, backend="cuda_strict", max_tile_size=2)
    _assert_against_full_cpu(routed, evaluate(batch, label, metrics=metrics, backend="cpu"), metrics)
    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert routed.metadata["backend_used"] == "cuda"
    assert routed.metadata["effective_max_tile_size"] == 2
    assert routed.metadata["auto_backend_reason"] == "bounded_f8_rank_pair_gpu_cap8_tile2"
    assert routed.metadata["auto_backend_evidence_id"] == "real_cos_f8_rank_pair_cap8_tile2"
    assert routed.metadata["source_request_fingerprint"] == reference.metadata["source_request_fingerprint"]
    for metric in metrics:
        if metric == "rank_ic_series":
            left, right = routed.series_metrics[metric], reference.series_metrics[metric]
        else:
            left, right = routed.scalar_metrics[metric], reference.scalar_metrics[metric]
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(routed.observation_counts[metric], reference.observation_counts[metric])

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection", lambda *_: "insufficient_cuda_memory")
    rejected = Source(batch)
    rejected.max_tile_size = 8
    fallback = evaluate_factor_source_batch(rejected, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured")
    assert fallback.metadata["backend_used"] == "cpu"
    assert fallback.metadata["auto_backend_reason"] == "insufficient_cuda_memory"
    assert fallback.metadata["effective_max_tile_size"] == 8
    assert rejected.reads == [(0, 5)]


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
        source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=8)
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
        backend="auto", source_auto_policy="legacy_measured", max_tile_size=cap)
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
        source, label, metrics=("pearson_ic",), backend="auto", source_auto_policy="legacy_measured", max_tile_size=32)
    reference = evaluate(batch, label, metrics=("pearson_ic",), backend="cpu")
    _assert_against_full_cpu(result, reference, ("pearson_ic",))
    assert admitted == [14 * 1024 ** 3]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["auto_backend_reason"] == "bounded_f61_pearson_ic_gpu_tile16"
    assert result.metadata["effective_max_tile_size"] == 16

    narrow = evaluate_factor_source_batch(
        Source(batch), label, metrics=("pearson_ic",), backend="auto", source_auto_policy="legacy_measured",
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
        rejected_source, label, metrics=("pearson_ic",), backend="auto", source_auto_policy="legacy_measured",
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
        source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=32)
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
        reversed_source, label, metrics=reversed_metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=32)
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
            case, label, metrics=changed_metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=width)
        assert rejected.metadata["backend_used"] == "cpu"
        assert rejected.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"
    assert gates == [14 * 1024 ** 3, 14 * 1024 ** 3]

    policy = GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64)
    case = Source(batch)
    case.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        case, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=16, gpu_policy=policy)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"
    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    case = Source(batch)
    case.max_tile_size = 32
    rejected = evaluate_factor_source_batch(
        case, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=16)
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
        source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=2)
    assert narrow.metadata["backend_used"] == "cpu"
    assert narrow.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    wrong_source = Source(batch)
    wrong_source.max_tile_size = 8
    wrong_metrics = evaluate_factor_source_batch(
        wrong_source, label, metrics=("rank_ic", "quantile_spread"),
        backend="auto", source_auto_policy="legacy_measured", max_tile_size=8)
    assert wrong_metrics.metadata["backend_used"] == "cpu"
    assert wrong_metrics.metadata["auto_backend_reason"] == "source_shape_or_metrics_not_certified"

    policy_source = Source(batch)
    policy_source.max_tile_size = 8
    nondefault = evaluate_factor_source_batch(
        policy_source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=8,
        gpu_policy=GPUExecutionPolicy(precision_policy=PrecisionPolicy.GPU_FP64),
    )
    assert nondefault.metadata["backend_used"] == "cpu"
    assert nondefault.metadata["auto_backend_reason"] == "gpu_precision_policy_outside_certified_range"

    monkeypatch.setattr(evaluator, "_auto_batch_cuda_rejection",
                        lambda *_: "insufficient_cuda_memory")
    rejected_source = Source(batch)
    rejected_source.max_tile_size = 8
    rejected = evaluate_factor_source_batch(
        rejected_source, label, metrics=metrics, backend="auto", source_auto_policy="legacy_measured", max_tile_size=8)
    assert rejected.metadata["backend_used"] == "cpu"
    assert rejected.metadata["auto_backend_reason"] == "insufficient_cuda_memory"


def test_current_cpu_qualification_precedes_legacy_static_policy(monkeypatch):
    """An accepted current CPU winner remains authoritative under legacy opt-in."""
    import importlib
    from test_source_qualification_provider_api_oct04 import _typed_pair
    from test_source_profile_default_cache_oct04 import _request
    from quant_evaluator.api import factor_source as source_api

    source, labels, _, _, records = _typed_pair(monkeypatch)
    monkeypatch.setattr(source_api, "select_source_auto_route",
        lambda **kwargs: pytest.fail("accepted qualification must bypass static routing"))
    monkeypatch.setattr(importlib.import_module("quant_evaluator.runtime.evaluator"),
        "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("a qualified CPU winner must not probe CUDA"))

    result = source_api.evaluate_factor_source_batch(
        source, labels, **_request(source), source_qualification=records,
        source_auto_policy="legacy_measured")
    receipt = result.metadata["execution_receipt"]

    assert result.metadata["backend_used"] == "cpu"
    assert receipt["source_auto_policy"] == "legacy_measured"
    assert receipt["source_qualification_status"] == "qualified_current_source"
    assert receipt["source_qualification_applied"] is True
    assert receipt["source_qualification_winner"] == "cpu"
    assert receipt["auto_backend_reason"] == "qualified_source_cpu_winner"
def _wrong_namespace_qualification_pair():
    from quant_evaluator.runtime.source_route_qualification import (
        BackendMeasurement, CounterbalancedABRecord, SourceRouteContext,
    )
    digest = lambda char: char * 64
    context = SourceRouteContext(
        request_content_sha256=digest("a"), source_identity_sha256=digest("b"),
        source_content_sha256=digest("c"), executable_source_sha256=digest("d"),
        runtime_fingerprint_sha256=digest("e"), package_fingerprint_sha256=digest("f"),
        thread_fingerprint_sha256=digest("1"), device_fingerprint_sha256=digest("2"),
        config_fingerprint_sha256=digest("3"), request_shape=(8, 48, 5),
        metric_ids=("rank_ic", "rank_ic_series"), expected_coverage_count=8 * 48 + 5,
        requested_tile_size=2, effective_tile_size=2)
    cpu = BackendMeasurement("cpu", 1.0, 0, digest("4"), True, 53, 53)
    cuda = BackendMeasurement("cuda", 1.1, 0, digest("5"), True, 53, 53)
    return (
        CounterbalancedABRecord(context, ("cpu", "cuda"), cpu, cuda),
        CounterbalancedABRecord(context, ("cuda", "cpu"), cpu, cuda),
    )


def test_malformed_cached_profile_is_not_legacy_reauthorization(monkeypatch):
    """A wrong-namespace validated pair must block V1 and static GPU fallback."""
    import importlib
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.runtime import source_qualification_cache as cache_api
    from quant_evaluator.runtime import source_profile_router
    from quant_evaluator.runtime.source_qualification_provider import SourceQualificationLookup
    from quant_evaluator.api import factor_source as source_api

    batch, label = _inputs()
    metrics = ("rank_ic", "rank_ic_series")
    source = Source(batch)
    baseline = evaluate_factor_source_batch(source, label, metrics=metrics, backend="cpu")
    metadata = source_api.capture_factor_tile_source(source)
    policy = GPUExecutionPolicy()
    key_args = dict(source=source, metadata=metadata, metrics=metrics,
        request_fingerprint=baseline.metadata["source_request_fingerprint"],
        requested_tile_size=2, policy=policy)
    cache_key = source_profile_router.source_profile_cache_key(**key_args)
    wrong_pair = _wrong_namespace_qualification_pair()
    cache_api.discard_validated_records(cache_key)
    cache_api.remember_validated_records(cache_key, wrong_pair)

    # The historical public getter remains tolerant for callers that do not opt in to strict lookup.
    assert source_profile_router.get_cached_source_route_profile_records(**key_args) is None
    assert cache_api.get_validated_records(cache_key) is None
    cache_api.remember_validated_records(cache_key, wrong_pair)

    provider_api = importlib.import_module("quant_evaluator.runtime.source_qualification_provider")
    monkeypatch.setattr(provider_api, "lookup_source_qualification_candidate",
        lambda _: SourceQualificationLookup(None, "provider_not_configured"))
    monkeypatch.setattr(source_api, "qualify_source_route",
        lambda **kwargs: (_ for _ in ()).throw(
            source_api.SourceQualificationError("qualified_cache_miss")))
    monkeypatch.setattr(source_api, "select_source_auto_route",
        lambda **kwargs: pytest.fail("malformed profile cache must not authorize static routing"))
    monkeypatch.setattr(importlib.import_module("quant_evaluator.runtime.evaluator"),
        "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("malformed profile cache must not check or launch CUDA"))
    source.reads.clear()

    try:
        result = evaluate_factor_source_batch(source, label, metrics=metrics, backend="auto",
                                              source_auto_policy="legacy_measured")
        receipt = result.metadata["execution_receipt"]
        assert result.metadata["backend_used"] == "cpu"
        assert receipt["source_auto_policy"] == "legacy_measured"
        assert receipt["source_qualification_status"] == "rejected_legacy_fallback"
        assert receipt["source_qualification_provider_status"] == "profile_cache_rejected"
        assert receipt["auto_backend_reason"] == "source_qualification_rejected"
        assert receipt["auto_backend_evidence_id"] is None
        assert source.reads == [(0, 2), (2, 4), (4, 5)]
        assert cache_api.get_validated_records(cache_key) is None
    finally:
        cache_api.discard_validated_records(cache_key)


@pytest.mark.parametrize("provider_status", ["report_invalid", "report_unreadable", "provider_error"])
def test_legacy_optin_rejects_provider_failures_before_static_route(monkeypatch, provider_status):
    """Provider errors are not pure absence and cannot reauthorize static CUDA."""
    import importlib
    from quant_evaluator.runtime.source_qualification_provider import SourceQualificationLookup
    from quant_evaluator.api import factor_source as source_api

    batch, label = _inputs()
    source = Source(batch)
    metrics = ("rank_ic", "rank_ic_series")
    monkeypatch.setattr(source_api, "get_cached_source_route_profile_records", lambda **kwargs: None)
    monkeypatch.setattr(source_api, "qualify_source_route",
        lambda **kwargs: (_ for _ in ()).throw(
            source_api.SourceQualificationError("qualified_cache_miss")))
    provider_api = importlib.import_module("quant_evaluator.runtime.source_qualification_provider")
    if provider_status == "provider_error":
        def lookup(_fingerprint):
            raise OSError("untrusted report path")
    else:
        lookup = lambda _fingerprint: SourceQualificationLookup(None, provider_status)
    monkeypatch.setattr(provider_api, "lookup_source_qualification_candidate", lookup)
    monkeypatch.setattr(source_api, "select_source_auto_route",
        lambda **kwargs: pytest.fail("provider failure must not authorize static routing"))
    monkeypatch.setattr(importlib.import_module("quant_evaluator.runtime.evaluator"),
        "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("provider failure must not probe CUDA"))

    result = evaluate_factor_source_batch(source, label, metrics=metrics, backend="auto",
                                          source_auto_policy="legacy_measured")
    receipt = result.metadata["execution_receipt"]
    assert result.metadata["backend_used"] == "cpu"
    assert receipt["source_qualification_provider_status"] == provider_status
    assert receipt["source_qualification_status"] == "not_available_legacy_fallback"
    assert receipt["auto_backend_reason"] == "source_qualification_rejected"
    assert receipt["auto_backend_evidence_id"] is None


def test_legacy_optin_rejects_live_candidate_before_static_route(monkeypatch):
    """A present but rejected report candidate is not equivalent to a cache miss."""
    import importlib
    from quant_evaluator.runtime.source_qualification_provider import (
        SourceQualificationCandidate, SourceQualificationLookup,
    )
    from quant_evaluator.runtime.source_route_profiles import CounterbalancedRouteProfileRecord
    from quant_evaluator.api import factor_source as source_api

    batch, label = _inputs()
    source = Source(batch)
    metrics = ("rank_ic", "rank_ic_series")
    fake_pair = (CounterbalancedRouteProfileRecord(None, (), None, None, ()),
                 CounterbalancedRouteProfileRecord(None, (), None, None, ()))
    candidate = SourceQualificationCandidate("candidate:rejected", fake_pair)
    provider_api = importlib.import_module("quant_evaluator.runtime.source_qualification_provider")
    monkeypatch.setattr(provider_api, "lookup_source_qualification_candidate",
        lambda _fingerprint: SourceQualificationLookup(candidate, "candidate_found"))
    monkeypatch.setattr(source_api, "get_cached_source_route_profile_records", lambda **kwargs: None)
    monkeypatch.setattr(source_api, "qualify_source_route_profiles",
        lambda **kwargs: (_ for _ in ()).throw(
            source_api.SourceProfileQualificationError("qualified_profile_context_mismatch")))
    monkeypatch.setattr(source_api, "select_source_auto_route",
        lambda **kwargs: pytest.fail("rejected candidate must not authorize static routing"))
    monkeypatch.setattr(importlib.import_module("quant_evaluator.runtime.evaluator"),
        "_auto_batch_cuda_rejection",
        lambda *_: pytest.fail("rejected candidate must not probe CUDA"))

    result = evaluate_factor_source_batch(source, label, metrics=metrics, backend="auto",
                                          source_auto_policy="legacy_measured")
    receipt = result.metadata["execution_receipt"]
    assert result.metadata["backend_used"] == "cpu"
    assert receipt["source_qualification_provider_status"] == "candidate_rejected"
    assert receipt["source_qualification_status"] == "rejected_legacy_fallback"
    assert receipt["auto_backend_reason"] == "source_qualification_rejected"
    assert receipt["auto_backend_evidence_id"] is None
