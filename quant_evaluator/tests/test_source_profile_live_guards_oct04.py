from dataclasses import replace
from types import SimpleNamespace

import pytest

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import source_profile_router as router
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    route_profile_execution_config_sha256,
)


def _h(char):
    return char * 64


class _Source:
    def __init__(self, source_ceiling):
        self.admitted_max_tile_size = source_ceiling
        self.reads = []


class _Metadata:
    def __init__(self, factor_count, source_ceiling):
        self.time_axis = SimpleNamespace(size=2)
        self.asset_axis = SimpleNamespace(size=3)
        self.factor_ids = tuple(f"f{i}" for i in range(factor_count))
        self.max_tile_size = 16
        self.snapshot_id = "live-profile-guard-test"
        self.dtype = "float64"


def _context(factor_count, source_ceiling, policy):
    return SourceRouteProfileContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        common_config_fingerprint_sha256=_h("3"),
        request_shape=(2, 3, factor_count), metric_ids=("rank_ic",),
        expected_coverage_count=factor_count, requested_tile_size=16,
        timing_scope=router.TIMING_SCOPE,
        metric_coverage=(("rank_ic", factor_count),),
        metric_error_tolerances=(("rank_ic", 1e-10),),
        live_source_admitted_max_tile_size=source_ceiling,
        gpu_admitted_max_tile_size=min(
            source_ceiling, policy.max_factor_tile_size or source_ceiling))


def _profile(context, backend, width, seconds, digest):
    factor_count = context.request_shape[-1]
    ranges = tuple((start, min(start + width, factor_count))
                   for start in range(0, factor_count, width))
    output = MetricOutputReceipt(
        "rank_ic", _h(digest), _h("8"), _h("9"), factor_count, factor_count)
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
        correctness_validated=True, coverage_expected=factor_count,
        coverage_observed=factor_count, outputs=(output,))


def _records(context, cpu_width, gpu_width):
    comparison = (MetricComparisonReceipt(
        metric_id="rank_ic", cpu_values_sha256=_h("4"),
        cuda_values_sha256=_h("5"), cpu_finite_mask_sha256=_h("8"),
        cuda_finite_mask_sha256=_h("8"),
        cpu_observation_counts_sha256=_h("9"),
        cuda_observation_counts_sha256=_h("9"),
        compared_value_count=context.request_shape[-1],
        finite_mask_equal=True, observation_counts_equal=True,
        max_abs_error=1e-12, max_abs_error_tolerance=1e-10,
        evidence_sha256=_h("6")),)
    cpu = _profile(context, "cpu", cpu_width, 4.0, "4")
    cuda = _profile(context, "cuda", gpu_width, 5.0, "5")
    cpu2 = replace(cpu, seconds=4.2)
    cuda2 = replace(cuda, seconds=5.1)
    return (
        CounterbalancedRouteProfileRecord(context, ("cpu", "cuda"), cpu, cuda,
                                          comparison),
        CounterbalancedRouteProfileRecord(context, ("cuda", "cpu"), cpu2, cuda2,
                                          comparison),
    )


def _qualify(monkeypatch, factor_count, source_ceiling, policy, gpu_width):
    source, metadata = _Source(source_ceiling), _Metadata(factor_count, source_ceiling)
    context = _context(factor_count, source_ceiling, policy)
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    result = router.qualify_source_route_profiles(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=_h("a"), requested_tile_size=16,
        policy=policy, records=_records(context, source_ceiling, gpu_width))
    return source, metadata, context, result


@pytest.mark.parametrize("factor_count,source_ceiling,policy,gpu_width", [
    (48, 5, GPUExecutionPolicy(), 5),
    (5, 5, GPUExecutionPolicy(), 5),
    (48, 5, GPUExecutionPolicy(max_factor_tile_size=2), 2),
])
def test_live_router_accepts_clipped_default_candidate_widths(
        monkeypatch, factor_count, source_ceiling, policy, gpu_width):
    router.profile_cache.clear_validated_records()
    _, _, _, result = _qualify(
        monkeypatch, factor_count, source_ceiling, policy, gpu_width)
    assert result.cpu_profile.source_tile_size == 5
    assert result.cuda_profile.source_tile_size == gpu_width


def test_live_router_rejects_width_unreachable_under_gpu_policy_cap(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=6)
    source, metadata = _Source(6), _Metadata(48, 6)
    context = _context(48, 6, policy)
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    with pytest.raises(router.SourceProfileQualificationError,
                       match="receipt_mismatch"):
        router.qualify_source_route_profiles(
            source=source, metadata=metadata, metrics=("rank_ic",),
            request_fingerprint=_h("a"), requested_tile_size=16,
            policy=policy, records=_records(context, 6, 5))


def test_cache_origin_stale_context_is_revalidated_and_evicted(monkeypatch):
    router.profile_cache.clear_validated_records()
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    source, metadata, context, _ = _qualify(monkeypatch, 48, 5, policy, 2)
    key = router.source_profile_cache_key(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=_h("a"), requested_tile_size=16, policy=policy)
    assert router.profile_cache.get_validated_records(key) is not None
    changed_context = replace(context, source_content_sha256=_h("0"))
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: changed_context)
    with pytest.raises(router.SourceProfileQualificationError,
                       match="receipt_mismatch"):
        router.qualify_source_route_profiles(
            source=source, metadata=metadata, metrics=("rank_ic",),
            request_fingerprint=_h("a"), requested_tile_size=16,
            policy=policy, records=None)
    assert router.profile_cache.get_validated_records(key) is None


@pytest.mark.parametrize("bad_range", [((0, True),), ((0, 5.0),)])
def test_execution_guard_rejects_bool_or_float_range_endpoints(
        monkeypatch, bad_range):
    policy = GPUExecutionPolicy()
    context = _context(5, 5, policy)
    profile = _profile(context, "cpu", 5, 4.0, "4")
    source = _Source(5)
    source.reads = list(bad_range)
    output = SimpleNamespace(metadata={"factor_tiles_processed": 1})
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    result = router.validate_source_route_profile_execution(
        source=source, output=output, profile=profile,
        expected_context=context, metadata=_Metadata(5, 5),
        metrics=("rank_ic",), request_fingerprint=_h("a"),
        requested_tile_size=16, policy=policy, backend="cpu")
    assert result == "qualified_profile_execution_receipt_deviated"


def test_execution_guard_rejects_post_execution_context_drift(monkeypatch):
    policy = GPUExecutionPolicy()
    context = _context(5, 5, policy)
    profile = _profile(context, "cpu", 5, 4.0, "4")
    source = _Source(5)
    source.reads = [(0, 5)]
    output = SimpleNamespace(metadata={"factor_tiles_processed": 1})
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: replace(
                            context, source_content_sha256=_h("0")))
    result = router.validate_source_route_profile_execution(
        source=source, output=output, profile=profile,
        expected_context=context, metadata=_Metadata(5, 5),
        metrics=("rank_ic",), request_fingerprint=_h("a"),
        requested_tile_size=16, policy=policy, backend="cpu")
    assert result == "qualified_profile_live_context_changed"


def test_cuda_execution_guard_accepts_exact_width_count_and_zero_oom(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=4)
    context = _context(5, 5, policy)
    profile = _profile(context, "cuda", 4, 4.0, "4")
    source = _Source(5)
    source.reads = [(0, 4), (4, 5)]
    output = SimpleNamespace(metadata={
        "factor_tiles_processed": 2, "factor_tile_size": 4, "oom_retries": 0})
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    assert router.validate_source_route_profile_execution(
        source=source, output=output, profile=profile,
        expected_context=context, metadata=_Metadata(5, 5),
        metrics=("rank_ic",), request_fingerprint=_h("a"),
        requested_tile_size=16, policy=policy, backend="cuda") is None


@pytest.mark.parametrize("metadata", [
    {"factor_tiles_processed": 2, "factor_tile_size": True, "oom_retries": 0},
    {"factor_tiles_processed": 2, "factor_tile_size": 2, "oom_retries": 0},
    {"factor_tiles_processed": 2, "factor_tile_size": 4, "oom_retries": True},
    {"factor_tiles_processed": 2, "factor_tile_size": 4, "oom_retries": 1},
    {"factor_tiles_processed": 2.0, "factor_tile_size": 4, "oom_retries": 0},
])
def test_cuda_execution_guard_rejects_width_oom_or_count_deviation(
        monkeypatch, metadata):
    policy = GPUExecutionPolicy(max_factor_tile_size=4)
    context = _context(5, 5, policy)
    profile = _profile(context, "cuda", 4, 4.0, "4")
    source = _Source(5)
    source.reads = [(0, 4), (4, 5)]
    monkeypatch.setattr(router, "capture_source_route_profile_context",
                        lambda **kwargs: context)
    result = router.validate_source_route_profile_execution(
        source=source, output=SimpleNamespace(metadata=metadata), profile=profile,
        expected_context=context, metadata=_Metadata(5, 5),
        metrics=("rank_ic",), request_fingerprint=_h("a"),
        requested_tile_size=16, policy=policy, backend="cuda")
    assert result == "qualified_profile_execution_receipt_deviated"


def test_execution_guard_rejects_bool_start_even_when_equal_to_zero(monkeypatch):
    policy = GPUExecutionPolicy()
    context = _context(5, 5, policy)
    profile = _profile(context, "cpu", 5, 4.0, "4")
    source = _Source(5)
    source.reads = [(False, 5)]  # bool compares equal to 0; strict type check must reject it.
    output = SimpleNamespace(metadata={"factor_tiles_processed": 1})
    result = router.validate_source_route_profile_execution(
        source=source, output=output, profile=profile,
        expected_context=context, metadata=_Metadata(5, 5),
        metrics=("rank_ic",), request_fingerprint=_h("a"),
        requested_tile_size=16, policy=policy, backend="cpu")
    assert result == "qualified_profile_execution_receipt_deviated"
