from types import SimpleNamespace

import numpy as np

from quant_evaluator.api import factor_source as api
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.source_qualified_router import SourceQualificationError


class _Source:
    def __init__(self):
        T, N, F = 8, 12, 5
        self.factor_ids = tuple(f"f{i}" for i in range(F))
        self.time_axis = AxisRef("time", "int64", T, np.arange(T, dtype=np.int64))
        self.asset_axis = AxisRef("asset", "int64", N, np.arange(N, dtype=np.int64))
        rng = np.random.default_rng(61004)
        self.values = rng.normal(size=(T, N, F))
        self.dtype = str(self.values.dtype)
        self.snapshot_id = "api-profile-test"
        self.max_tile_size = 16
        self.admitted_max_tile_size = 5
        self.reads = []

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(self.factor_ids[start:end], self.time_axis, self.asset_axis,
                            self.values[:, :, start:end])
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        pass


def _labels(source):
    values = np.random.default_rng(61005).normal(
        size=(source.time_axis.size, source.asset_axis.size))
    times = tuple(source.time_axis.values.tolist())
    return LabelBundle(
        "ret", values, 1, decision_time=times, label_start_time=times,
        label_end_time=tuple(value + 1 for value in times),
        asset_axis=source.asset_axis)


def _decision(winner):
    cpu = SimpleNamespace(source_tile_size=5, source_ranges=((0, 5),))
    cuda = SimpleNamespace(source_tile_size=2,
                           source_ranges=((0, 2), (2, 4), (4, 5)))
    return SimpleNamespace(
        winning_backend=winner, cpu_profile=cpu, cuda_profile=cuda,
        winner_profile=cpu if winner == "cpu" else cuda,
        evidence_sha256="a" * 64, cache_status="supplied",
        scope="test-profile-v2", source_content_scope="test-source-v1",
        context=object())


def _use_fake_profile(monkeypatch, winner):
    monkeypatch.setattr(api, "is_source_route_profile_pair", lambda records: True)
    monkeypatch.setattr(api, "qualify_source_route_profiles",
                        lambda **kwargs: _decision(winner))
    monkeypatch.setattr(api, "validate_source_route_profile_execution",
                        lambda **kwargs: None)
    monkeypatch.setattr(api, "select_source_auto_route", lambda **kwargs: None)
    monkeypatch.setattr(api, "qualify_source_route",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError("v2 evidence leaked into the v1 guard")))


def test_v2_cpu_winner_uses_source_ceiling_independent_of_gpu_cap(monkeypatch):
    source = _Source()
    _use_fake_profile(monkeypatch, "cpu")
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: (object(), object()))

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2))

    assert source.reads == [(0, 5)]
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["effective_max_tile_size"] == 5
    assert result.metadata["source_qualification_applied"] is True
    assert result.metadata["source_qualification_winner"] == "cpu"


def test_v2_cuda_ineligible_falls_back_to_measured_cpu_width(monkeypatch):
    source = _Source()
    _use_fake_profile(monkeypatch, "cuda")
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: (object(), object()))
    monkeypatch.setattr(api, "cuda_route_rejection", lambda policy: "test_resource_gate")

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2))

    assert source.reads == [(0, 5)]
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["effective_max_tile_size"] == 5
    assert result.metadata["source_qualification_status"] == "qualified_cuda_ineligible"
    assert result.metadata["source_qualification_applied"] is False


def test_v2_cache_miss_does_not_enter_qualification_or_read_a_pilot(monkeypatch):
    source = _Source()
    cache_calls = []
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: cache_calls.append(kwargs) or None)
    monkeypatch.setattr(api, "qualify_source_route_profiles",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError("profile cache miss must not qualify")))
    monkeypatch.setattr(api, "qualify_source_route",
                        lambda **kwargs: (_ for _ in ()).throw(
                            SourceQualificationError("qualified_cache_miss")))
    monkeypatch.setattr(api, "select_source_auto_route", lambda **kwargs: None)

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2))

    assert len(cache_calls) == 1
    assert result.metadata["source_qualification_status"] == "not_available_legacy_fallback"
    assert source.reads == [(0, 5)]  # Only the requested CPU evaluation, no pilot read.


def test_explicit_backend_ignores_v2_evidence(monkeypatch):
    source = _Source()
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError("explicit backend must skip profile cache")))
    monkeypatch.setattr(api, "qualify_source_route_profiles",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError("explicit backend must skip profile validation")))

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="cpu",
        source_qualification=(object(), object()), max_tile_size=16)

    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["source_qualification_status"] == "not_used_explicit_backend"
    assert result.metadata["source_qualification_applied"] is False
    assert source.reads == [(0, 5)]


def test_v2_cuda_winner_dispatches_measured_width_and_checks_real_receipt(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from quant_evaluator.contracts._hashutil import stable_content_hex
    from quant_evaluator.runtime import source_profile_router as router
    from quant_evaluator.runtime.source_route_profiles import (
        BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
        MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
        route_profile_execution_config_sha256,
    )

    source = _Source()
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    h = lambda char: char * 64
    context = SourceRouteProfileContext(
        request_content_sha256=h("a"), source_identity_sha256=h("b"),
        source_content_sha256=h("c"), executable_source_sha256=h("d"),
        runtime_fingerprint_sha256=h("e"), package_fingerprint_sha256=h("f"),
        thread_fingerprint_sha256=h("1"), device_fingerprint_sha256=h("2"),
        common_config_fingerprint_sha256=h("3"), request_shape=(8, 12, 5),
        metric_ids=("rank_ic",), expected_coverage_count=5,
        requested_tile_size=16, timing_scope=router.TIMING_SCOPE,
        metric_coverage=(("rank_ic", 5),),
        metric_error_tolerances=(("rank_ic", 1e-10),),
        live_source_admitted_max_tile_size=5, gpu_admitted_max_tile_size=2)
    comparison = (MetricComparisonReceipt(
        metric_id="rank_ic", cpu_values_sha256=h("4"), cuda_values_sha256=h("5"),
        cpu_finite_mask_sha256=h("8"), cuda_finite_mask_sha256=h("8"),
        cpu_observation_counts_sha256=h("9"), cuda_observation_counts_sha256=h("9"),
        compared_value_count=5, finite_mask_equal=True, observation_counts_equal=True,
        max_abs_error=1e-12, max_abs_error_tolerance=1e-10,
        evidence_sha256=h("6")),)

    def make_profile(backend, width, seconds, digest):
        ranges = tuple((start, min(start + width, 5)) for start in range(0, 5, width))
        outputs = (MetricOutputReceipt("rank_ic", h(digest), h("8"), h("9"), 5, 5),)
        schedule = stable_content_hex(
            tag="RouteProfileExecutionSchedule.v2",
            fields={"scope": router.EXECUTION_SCHEDULE_SCOPE,
                    "source_ranges": ranges, "compute_ranges": ranges})
        return BackendRouteProfileMeasurement(
            backend=backend, seconds=seconds, oom_count=0,
            source_tile_size=width, source_ranges=ranges,
            actual_tile_size=width, compute_ranges=ranges,
            execution_config_sha256=route_profile_execution_config_sha256(
                context, backend, width),
            execution_schedule_sha256=schedule,
            execution_schedule_scope=router.EXECUTION_SCHEDULE_SCOPE,
            correctness_validated=True, coverage_expected=5,
            coverage_observed=5, outputs=outputs)

    cpu, cuda = make_profile("cpu", 5, 6.0, "4"), make_profile("cuda", 2, 4.0, "5")
    cpu2, cuda2 = replace(cpu, seconds=6.1), replace(cuda, seconds=4.1)
    records = (
        CounterbalancedRouteProfileRecord(context, ("cpu", "cuda"), cpu, cuda, comparison),
        CounterbalancedRouteProfileRecord(context, ("cuda", "cpu"), cpu2, cuda2, comparison),
    )
    router.profile_cache.clear_validated_records()
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    monkeypatch.setattr(api, "cuda_route_rejection", lambda policy: None)

    class _Session:
        def __init__(self, policy):
            self.policy = policy
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class _Executor:
        def __init__(self, session):
            self.session = session
        def run_source_tiled(self, src, label, metrics, *, max_tile_size, source_metadata):
            output = api._cpu_source_batch(
                src, source_metadata, label, metrics, policy, max_tile_size)
            output.metadata.update({"factor_tile_size": 2, "oom_retries": 0})
            return output

    import quant_evaluator.runtime.device_session as device_session
    import quant_evaluator.runtime.gpu_executor as gpu_executor
    monkeypatch.setattr(device_session, "DeviceEvaluationSession", _Session)
    monkeypatch.setattr(gpu_executor, "GPUExecutor", _Executor)

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=policy, source_qualification=records)

    assert source.reads == [(0, 2), (2, 4), (4, 5)]
    assert result.metadata["backend_used"] == "cuda"
    assert result.metadata["effective_max_tile_size"] == 2
    assert result.metadata["source_qualification_applied"] is True
    router.profile_cache.clear_validated_records()

def test_v2_cache_hit_is_requalified_as_cache_origin(monkeypatch):
    source = _Source()
    records = (object(), object())
    seen = []
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: records)
    monkeypatch.setattr(api, "is_source_route_profile_pair", lambda value: True)
    monkeypatch.setattr(api, "qualify_source_route_profiles",
                        lambda **kwargs: seen.append(kwargs) or _decision("cpu"))
    monkeypatch.setattr(api, "validate_source_route_profile_execution",
                        lambda **kwargs: None)
    monkeypatch.setattr(api, "select_source_auto_route", lambda **kwargs: None)

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2))

    assert len(seen) == 1
    assert seen[0]["records"] is None
    assert result.metadata["source_qualification_applied"] is True


def test_v2_post_execution_mismatch_revokes_result_and_discards_cache(monkeypatch):
    source = _Source()
    discards = []
    _use_fake_profile(monkeypatch, "cpu")
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: (object(), object()))
    monkeypatch.setattr(api, "validate_source_route_profile_execution",
                        lambda **kwargs: "qualified_profile_live_context_changed")
    monkeypatch.setattr(api, "discard_source_route_profile_cache",
                        lambda **kwargs: discards.append(kwargs))

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2))

    assert result.metadata["source_qualification_applied"] is False
    assert result.metadata["source_qualification_status"] == "execution_configuration_deviated"
    assert result.metadata["source_qualification_reason"] == "qualified_profile_live_context_changed"
    assert len(discards) == 1

