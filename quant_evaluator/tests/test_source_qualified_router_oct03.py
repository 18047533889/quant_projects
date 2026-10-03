"""Focused identity, routing and cache tests for current source qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.adapters.cos_factor_tile_source import (
    BoundCosFactor, CosFactorTileSource,
)
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTileSourceMetadata
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.api import factor_source as source_api
from quant_evaluator.runtime import source_qualified_router as router
from quant_evaluator.runtime.source_qualification_cache import (
    clear_validated_records, get_validated_records,
)
from quant_evaluator.runtime.source_route_qualification import (
    BackendMeasurement, CounterbalancedABRecord,
)


def _h(char):
    return char * 64


@pytest.fixture
def live(monkeypatch):
    factor_ids = ("f0", "f1")
    time_axis = AxisRef("time", "int64", 3, np.arange(3, dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", 4, np.arange(4, dtype=np.int64))
    records = tuple(BoundCosFactor(fid, f"cos://bound/{fid}", _h("b"), 128)
                    for fid in factor_ids)
    rows = {r.factor_id: {"uri": r.uri, "sha256": r.sha256,
                          "bytes": r.size_bytes} for r in records}
    manifest = SimpleNamespace(manifest_sha256=_h("a"), factors=rows)
    source = object.__new__(CosFactorTileSource)
    source.records = records
    source.factor_ids = factor_ids
    source.manifest_sha256 = manifest.manifest_sha256
    source.manifest_snapshot = manifest
    source.snapshot_id = _h("c")
    source._closed = False
    source._controller_close = lambda: None
    source.verify_manifest = lambda snapshot: None
    source.admitted_max_tile_size = 2
    source.max_tile_size = 2
    source.prefetch_mode = "auto"
    source.prefetch_workers = 2
    source.max_source_memory_bytes = 4 * 1024**3
    source.max_prefetch_memory_bytes = 512 * 1024**2
    source.extra_assembly_bytes_per_cell = 0
    metadata = FactorTileSourceMetadata(
        factor_ids, time_axis, asset_axis, "float64", source.snapshot_id, 2)

    runtime = {
        "schema": "source-runtime-identity-v3", "complete": True,
        "incomplete_reasons": [], "python": "3.12.3",
        "packages": {name: {"loaded": True, "version": "1.0"}
                     for name in ("numpy", "pandas", "numba", "cupy", "threadpoolctl",
                                  "scipy", "llvmlite", "duckdb", "pyarrow", "polars")},
        "numba": {"loaded": True, "threads": 2, "threading_layer": "omp"},
        "threadpools": {"status": "captured", "pools": []},
        "thread_environment": {"OMP_NUM_THREADS": "2"},
        "cuda": {"status": "captured", "context_initialized": True,
                 "device": {"id": 0, "identity_scope": "host_pci_bus_v1",
                            "host_identity_sha256": _h("d"), "pci_bus_id": "0000:01:00.0",
                            "name": "NVIDIA L20", "total_memory_bytes": 48 * 1024**3,
                            "compute_capability": [8, 9], "uuid": "e" * 32,
                            "uuid_status": "captured_full_16_bytes"}},
        "identity_digest": _h("e"),
    }
    monkeypatch.setattr(router, "capture_source_runtime_identity", lambda: runtime)
    monkeypatch.setattr(router, "is_qualified_source_runtime_identity", lambda _: True)
    monkeypatch.setattr(router, "source_runtime_identity_digest", lambda _: _h("e"))

    class _Identity:
        def __init__(self, *args, **kwargs):
            pass

        def identify(self, *, strict_full_content):
            assert strict_full_content is True
            return SimpleNamespace(
                digest=_h("f"), full_content_checked=True,
                strategy="strict_full_content", drifted=False)

    monkeypatch.setattr(router, "ProcessSourceIdentity", _Identity)
    monkeypatch.setattr(router, "capture_source_dependency_identity", lambda: SimpleNamespace(
        schema=router.DEPENDENCY_IDENTITY_SCHEMA, digest=_h("3"),
        component_digests=(("quant_evaluator", _h("f")),
                          ("data_access", _h("a")),
                          ("factor_optimizer", _h("b")),
                          ("factor_preprocess", _h("c"))),
    ))
    return source, metadata, GPUExecutionPolicy()


def _route_args(live, *, records=None, maximum=2):
    source, metadata, policy = live
    return dict(source=source, metadata=metadata, metrics=("rank_ic",),
                request_fingerprint=_h("9"), requested_tile_size=2,
                maximum_effective_tile_size=maximum, policy=policy,
                records=records)


def _records(context, winner="cuda"):
    cpu_seconds, cuda_seconds = ((2.0, 1.0) if winner == "cuda" else (1.0, 2.0))

    def measurement(backend, seconds):
        return BackendMeasurement(backend, seconds, 0, _h("8"), True, 2, 2)

    return (
        CounterbalancedABRecord(
            context, ("cpu", "cuda"), measurement("cpu", cpu_seconds),
            measurement("cuda", cuda_seconds)),
        CounterbalancedABRecord(
            context, ("cuda", "cpu"), measurement("cpu", cpu_seconds + 0.1),
            measurement("cuda", cuda_seconds + 0.1)),
    )


def _matching_records(live):
    source, metadata, policy = live
    state = router._capture_live_context_state(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=_h("9"), requested_tile_size=2,
        maximum_effective_tile_size=2, policy=policy,
        lookup_key=_h("7"))
    return _records(state.context)


def test_dependency_digest_drift_rejects_cached_evidence(live, monkeypatch):
    records = _matching_records(live)
    router.qualify_source_route(**_route_args(live, records=records))
    monkeypatch.setattr(router, "capture_source_dependency_identity", lambda: SimpleNamespace(
        schema=router.DEPENDENCY_IDENTITY_SCHEMA, digest=_h("4"),
        component_digests=(("quant_evaluator", _h("f")),
                          ("data_access", _h("a")),
                          ("factor_optimizer", _h("b")),
                          ("factor_preprocess", _h("c"))),
    ))
    with pytest.raises(router.SourceQualificationError, match="qualified_receipt_mismatch"):
        router.qualify_source_route(**_route_args(live, records=None))
    with pytest.raises(router.SourceQualificationError, match="qualified_cache_miss"):
        router.qualify_source_route(**_route_args(live, records=None))


def test_dependency_scan_failure_rejects_and_evicts_cached_evidence(live, monkeypatch):
    records = _matching_records(live)
    router.qualify_source_route(**_route_args(live, records=records))
    def fail():
        raise RuntimeError("private source location must not become a reason code")
    monkeypatch.setattr(router, "capture_source_dependency_identity", fail)
    with pytest.raises(router.SourceQualificationError,
                       match="current_dependency_source_identity_unavailable") as error:
        router.qualify_source_route(**_route_args(live, records=None))
    assert "private source location" not in str(error.value)
    with pytest.raises(router.SourceQualificationError, match="qualified_cache_miss"):
        router.qualify_source_route(**_route_args(live, records=None))


def test_public_context_builder_counts_all_scalar_and_series_outputs(live):
    source, metadata, policy = live
    context = router.capture_source_route_context(
        source=source, metadata=metadata,
        metrics=("rank_ic", "rank_ic_series"),
        request_fingerprint=_h("9"), requested_tile_size=2,
        maximum_effective_tile_size=2, effective_tile_size=2, policy=policy)
    assert context.expected_coverage_count == 2 + 3 * 2
    assert context.metric_ids == ("rank_ic", "rank_ic_series")


@pytest.mark.parametrize("winner", ["cpu", "cuda"])
def test_accepts_exact_live_counterbalanced_pair(live, winner):
    records = _matching_records(live)
    if winner == "cpu":
        records = _records(records[0].context, winner="cpu")
    result = router.qualify_source_route(**_route_args(live, records=records))
    assert result.qualification.winning_backend == winner
    assert result.effective_tile_size == 2
    assert result.scope == "exact_request_bound_cos_runtime_policy_v1"
    assert result.source_content_scope == "qe_da_fo_fp_python_dependency_content_v1"
    assert len(result.evidence_sha256) == 64


@pytest.mark.parametrize("field,value", [
    ("runtime_fingerprint_sha256", _h("1")),
    ("package_fingerprint_sha256", _h("2")),
    ("thread_fingerprint_sha256", _h("3")),
    ("device_fingerprint_sha256", _h("4")),
    ("config_fingerprint_sha256", _h("5")),
    ("source_content_sha256", _h("6")),
    ("executable_source_sha256", _h("7")),
])
def test_rejects_any_stale_record_context_identity(live, field, value):
    records = _matching_records(live)
    changed = replace(records[0].context, **{field: value})
    stale = (replace(records[0], context=changed), records[1])
    with pytest.raises(router.SourceQualificationError, match="qualified_receipt_mismatch"):
        router.qualify_source_route(**_route_args(live, records=stale))


@pytest.mark.parametrize("bad_width", [True, 2.0, 3])
def test_rejects_malformed_or_out_of_admission_measured_width(live, bad_width):
    records = _matching_records(live)
    malformed_context = replace(records[0].context, effective_tile_size=bad_width)
    malformed = (replace(records[0], context=malformed_context), records[1])
    with pytest.raises(router.SourceQualificationError):
        router.qualify_source_route(**_route_args(live, records=malformed))


def test_refuses_non_cos_snapshot_source(live):
    _, metadata, policy = live
    fake = SimpleNamespace(snapshot_id=metadata.snapshot_id)
    with pytest.raises(router.SourceQualificationError, match="source_is_not_bound_cos"):
        router.qualify_source_route(**dict(
            source=fake, metadata=metadata, metrics=("rank_ic",),
            request_fingerprint=_h("9"), requested_tile_size=2,
            maximum_effective_tile_size=2, policy=policy,
            records=_matching_records(live)))


def test_rejects_manifest_selected_record_mismatch(live):
    source, metadata, policy = live
    source.manifest_snapshot.factors["f0"]["sha256"] = _h("1")
    with pytest.raises(router.SourceQualificationError, match="bound_cos_selected_record_mismatch"):
        router.qualify_source_route(**_route_args(live, records=_matching_records(live)))


def test_cache_only_contains_live_validated_pairs(live):
    clear_validated_records()
    records = _matching_records(live)
    decision = router.qualify_source_route(**_route_args(live, records=records))
    assert decision.cache_status == "supplied"
    looked_up = router.qualify_source_route(**_route_args(live, records=None))
    assert looked_up.cache_status == "cache_hit"
    assert looked_up.evidence_sha256 == decision.evidence_sha256
    assert get_validated_records(router._preliminary_cache_key(**{
        "source": live[0], "metadata": live[1], "metrics": ("rank_ic",),
        "request_fingerprint": _h("9"), "requested_tile_size": 2,
        "maximum_effective_tile_size": 2, "policy": live[2]})) == records


def test_effective_width_must_fit_current_memory_and_policy_cap(live):
    records = _matching_records(live)
    with pytest.raises(router.SourceQualificationError, match="effective_tile_outside_live_admission"):
        router.qualify_source_route(**_route_args(live, records=records, maximum=1))


def test_smaller_width_receipt_cannot_certify_larger_live_maximum(live):
    source, metadata, policy = live
    args = _route_args(live)
    state = router._capture_live_context_state(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=_h("9"), requested_tile_size=2,
        maximum_effective_tile_size=2, policy=policy, lookup_key=_h("7"))
    smaller_context = state.for_effective_width(1)
    with pytest.raises(router.SourceQualificationError,
                       match="qualified_effective_tile_below_live_maximum"):
        router.qualify_source_route(**{**args, "records": _records(smaller_context)})


class _ArraySource:
    def __init__(self, batch):
        self.batch = batch
        self.factor_ids = batch.factor_ids
        self.time_axis = batch.time_axis
        self.asset_axis = batch.asset_axis
        self.dtype = batch.dtype
        self.snapshot_id = "api-source-fixture"
        self.max_tile_size = 2
        self.reads = []

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.batch.values[:, :, start:end],
            validity=(None if self.batch.validity is None
                      else self.batch.validity[:, :, start:end]))
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


def _small_api_inputs():
    time = np.arange(4, dtype=np.int64)
    assets = np.arange(8, dtype=np.int64)
    time_axis = AxisRef("time", "int64", 4, time)
    asset_axis = AxisRef("asset", "int64", 8, assets)
    values = np.arange(64, dtype=np.float64).reshape(4, 8, 2) + np.random.default_rng(4).normal(
        size=(4, 8, 2))
    batch = FactorBatch(("f0", "f1"), time_axis, asset_axis, values)
    label_values = np.random.default_rng(5).normal(size=(4, 8))
    label = LabelBundle(
        "ret", label_values, 1, decision_time=tuple(time),
        label_start_time=tuple(time), label_end_time=tuple(time + 1),
        asset_axis=asset_axis)
    return batch, label


def _api_decision(winner, effective=1):
    from quant_evaluator.runtime.source_qualified_router import QualifiedSourceRoute
    from quant_evaluator.runtime.source_route_qualification import (
        SourceRouteContext, SourceRouteQualification,
    )

    context = SourceRouteContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        config_fingerprint_sha256=_h("3"), request_shape=(4, 8, 2),
        metric_ids=("rank_ic",), expected_coverage_count=2,
        requested_tile_size=2, effective_tile_size=effective)
    qualification = SourceRouteQualification(
        context, winner, (("cpu", "cuda"), ("cuda", "cpu")),
        ((2.0, 1.0), (2.1, 1.1)), _h("4"))
    return QualifiedSourceRoute(qualification, effective, _h("5"), "supplied")


def test_explicit_api_backend_ignores_optional_qualification(monkeypatch):
    monkeypatch.setattr(source_api, "qualify_source_route",
                        lambda **_: pytest.fail("explicit backend must ignore qualification"))
    batch, label = _small_api_inputs()
    source = _ArraySource(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=("rank_ic",), backend="cpu",
        source_qualification=object())
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["source_qualification_status"] == "not_used_explicit_backend"
    assert result.metadata["execution_receipt"]["source_qualification_applied"] is False


def test_api_applies_qualified_cpu_width_and_records_exact_scope(monkeypatch):
    decision = _api_decision("cpu", effective=1)
    monkeypatch.setattr(source_api, "qualify_source_route", lambda **_: decision)
    batch, label = _small_api_inputs()
    source = _ArraySource(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=("rank_ic",), backend="auto")
    assert source.reads == [(0, 1), (1, 2)]
    assert result.metadata["effective_max_tile_size"] == 1
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["source_qualification_winner"] == "cpu"
    assert result.metadata["source_qualification_applied"] is True
    assert result.metadata["source_qualification_content_scope"] == "qe_da_fo_fp_python_dependency_content_v1"


def test_cuda_winner_resource_rejection_routes_cpu_without_claiming_applied(monkeypatch):
    decision = _api_decision("cuda", effective=1)
    monkeypatch.setattr(source_api, "qualify_source_route", lambda **_: decision)
    monkeypatch.setattr(source_api, "cuda_route_rejection", lambda _: "insufficient_cuda_memory")
    batch, label = _small_api_inputs()
    source = _ArraySource(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=("rank_ic",), backend="auto")
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["source_qualification_winner"] == "cuda"
    assert result.metadata["source_qualification_status"] == "qualified_cuda_ineligible"
    assert result.metadata["source_qualification_applied"] is False


@pytest.mark.parametrize("actual_width,oom_retries,deviates", [
    (1, 0, True), (2, 1, True), (True, 0, True), (2, False, True), (2, 0, False),
])
def test_cuda_execution_deviation_revokes_application_and_cache(
    live, monkeypatch, actual_width, oom_retries, deviates,
):
    decision = _api_decision("cuda", effective=2)
    monkeypatch.setattr(source_api, "qualify_source_route", lambda **_: decision)
    monkeypatch.setattr(source_api, "cuda_route_rejection", lambda _: None)
    evictions = []
    monkeypatch.setattr(source_api, "discard_source_route_cache",
                        lambda **kwargs: evictions.append(kwargs))

    class Session:
        def __init__(self, policy):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Executor:
        def __init__(self, session):
            pass

        def run_source_tiled(self, *args, **kwargs):
            from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
            out = BatchEvaluationBundle(("f0", "f1"), "ret")
            out.scalar_metrics = {"rank_ic": np.zeros(2, dtype=np.float64)}
            out.observation_counts = {"rank_ic": np.zeros(2, dtype=np.int64)}
            out.metadata = {"factor_tile_size": actual_width,
                            "oom_retries": oom_retries}
            return out

    import quant_evaluator.runtime.device_session as device_session
    import quant_evaluator.runtime.gpu_executor as gpu_executor
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", Session)
    monkeypatch.setattr(gpu_executor, "GPUExecutor", Executor)
    batch, label = _small_api_inputs()
    source = _ArraySource(batch)
    result = evaluate_factor_source_batch(
        source, label, metrics=("rank_ic",), backend="auto")
    assert result.metadata["source_qualification_applied"] is (not deviates)
    assert len(evictions) == int(deviates)
    if deviates:
        assert result.metadata["source_qualification_status"] == "execution_configuration_deviated"
        assert result.metadata["source_qualification_reason"] == (
            "qualified_cuda_execution_width_or_oom_deviated")
    else:
        assert result.metadata["source_qualification_status"] == "qualified_current_source"


def test_runtime_drift_after_cache_insert_discards_pair(live, monkeypatch):
    clear_validated_records()
    records = _matching_records(live)
    router.qualify_source_route(**_route_args(live, records=records))
    key = router._preliminary_cache_key(**{
        "source": live[0], "metadata": live[1], "metrics": ("rank_ic",),
        "request_fingerprint": _h("9"), "requested_tile_size": 2,
        "maximum_effective_tile_size": 2, "policy": live[2]})
    assert get_validated_records(key) is not None
    monkeypatch.setattr(router, "source_runtime_identity_digest", lambda _: _h("9"))
    with pytest.raises(router.SourceQualificationError, match="qualified_receipt_mismatch"):
        router.qualify_source_route(**_route_args(live, records=None))
    assert get_validated_records(key) is None
