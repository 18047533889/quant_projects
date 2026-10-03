"""End-to-end contract for default reuse of a validated v2 route profile."""

from dataclasses import replace

import pytest

from quant_evaluator.api import factor_source as api
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import source_profile_router as router
from quant_evaluator.runtime.source_qualified_router import SourceQualificationError
from test_source_profile_live_guards_oct04 import _context, _records
from test_source_route_profiles_api_oct03 import _Source, _labels


@pytest.fixture(autouse=True)
def clear_profile_cache():
    router.profile_cache.clear_validated_records()
    yield
    router.profile_cache.clear_validated_records()


def _install_live_context_capture(monkeypatch, policy):
    def capture(*, source, metadata, metrics, request_fingerprint,
                requested_tile_size, policy):
        factors = len(metadata.factor_ids)
        ceiling = source.admitted_max_tile_size
        context = _context(factors, ceiling, policy)
        return replace(
            context,
            request_content_sha256=request_fingerprint,
            request_shape=(metadata.time_axis.size, metadata.asset_axis.size, factors),
            metric_ids=tuple(metrics),
            expected_coverage_count=factors,
            metric_coverage=tuple((metric, factors) for metric in metrics),
            metric_error_tolerances=tuple((metric, 1e-10) for metric in metrics),
        )

    monkeypatch.setattr(router, "capture_source_route_profile_context", capture)


def _request(source):
    return dict(
        metrics=("rank_ic",), backend="auto", max_tile_size=16,
        gpu_policy=GPUExecutionPolicy(max_factor_tile_size=2),
    )


def test_explicit_typed_pair_is_cached_then_default_auto_reports_cache_hit(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    _install_live_context_capture(monkeypatch, policy)

    first_source = _Source()
    labels = _labels(first_source)
    metadata = api.capture_factor_tile_source(first_source)
    fingerprint = api._source_request_fingerprint(
        metadata, labels, ("rank_ic",))
    context = router.capture_source_route_profile_context(
        source=first_source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=fingerprint, requested_tile_size=16, policy=policy)
    pair = _records(context, cpu_width=5, gpu_width=2)

    first = api.evaluate_factor_source_batch(
        first_source, labels, **_request(first_source), source_qualification=pair)
    assert first.metadata["source_qualification_status"] == "qualified_current_source"
    assert first.metadata["source_qualification_cache_status"] == "supplied"
    assert first.metadata["source_qualification_applied"] is True
    assert first.metadata["source_qualification_winner"] == "cpu"

    second_source = _Source()
    second = api.evaluate_factor_source_batch(second_source, labels, **_request(second_source))
    assert second.metadata["source_qualification_status"] == "qualified_current_source"
    assert second.metadata["source_qualification_cache_status"] == "cache_hit"
    assert second.metadata["source_qualification_applied"] is True
    assert second.metadata["source_qualification_winner"] == "cpu"
    assert second.metadata["effective_max_tile_size"] == first.metadata["effective_max_tile_size"] == 5


@pytest.mark.parametrize("identity_change", ["snapshot", "prefetch_config"])
def test_default_auto_does_not_reuse_pair_after_cache_key_identity_changes(
    monkeypatch, identity_change,
):
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    _install_live_context_capture(monkeypatch, policy)
    first_source = _Source()
    first_source.prefetch_mode = "auto"
    labels = _labels(first_source)
    metadata = api.capture_factor_tile_source(first_source)
    fingerprint = api._source_request_fingerprint(metadata, labels, ("rank_ic",))
    context = router.capture_source_route_profile_context(
        source=first_source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=fingerprint, requested_tile_size=16, policy=policy)
    pair = _records(context, cpu_width=5, gpu_width=2)
    first = api.evaluate_factor_source_batch(
        first_source, labels, **_request(first_source), source_qualification=pair)
    assert first.metadata["source_qualification_applied"] is True

    changed = _Source()
    changed.prefetch_mode = "auto"
    if identity_change == "snapshot":
        changed.snapshot_id = "different-source-snapshot"
    else:
        changed.prefetch_mode = "off"
    monkeypatch.setattr(api, "qualify_source_route",
        lambda **kwargs: (_ for _ in ()).throw(
            SourceQualificationError("qualified_cache_miss")))
    result = api.evaluate_factor_source_batch(changed, labels, **_request(changed))
    assert result.metadata["source_qualification_status"] == "not_available_legacy_fallback"
    assert result.metadata["source_qualification_applied"] is False
    assert result.metadata["source_qualification_winner"] is None
