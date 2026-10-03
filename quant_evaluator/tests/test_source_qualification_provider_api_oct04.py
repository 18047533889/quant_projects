"""Public API contracts for report-candidate source qualification."""
from dataclasses import replace
import pytest

from quant_evaluator.api import factor_source as api
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import source_profile_router as router
from quant_evaluator.runtime import source_qualification_provider as provider
from test_source_profile_default_cache_oct04 import (
    _install_live_context_capture, _request,
)
from test_source_profile_live_guards_oct04 import _records
from test_source_route_profiles_api_oct03 import _Source, _labels
from source_profile_fixture_identity_oct04 import rebind_pair_to_cpu_outputs


@pytest.fixture(autouse=True)
def clear_process_qualification_state():
    provider.clear_source_qualification_provider()
    router.profile_cache.clear_validated_records()
    yield
    provider.clear_source_qualification_provider()
    router.profile_cache.clear_validated_records()


def _typed_pair(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    _install_live_context_capture(monkeypatch, policy)
    source = _Source()
    labels = _labels(source)
    metadata = api.capture_factor_tile_source(source)
    request_fingerprint = api._source_request_fingerprint(
        metadata, labels, ("rank_ic",))
    context = router.capture_source_route_profile_context(
        source=source, metadata=metadata, metrics=("rank_ic",),
        request_fingerprint=request_fingerprint, requested_tile_size=16,
        policy=policy)
    records = _records(context, cpu_width=5, gpu_width=2)
    actual_cpu = api.evaluate_factor_source_batch(
        source, labels, metrics=("rank_ic",), backend="cpu",
        max_tile_size=16, gpu_policy=policy)
    source.reads.clear()
    return source, labels, policy, request_fingerprint, rebind_pair_to_cpu_outputs(
        records, actual_cpu)


def _install_candidate_lookup(monkeypatch, request_fingerprint, records):
    candidate = provider.SourceQualificationCandidate("candidate:test", records)
    outcome = provider.SourceQualificationLookup(candidate, "candidate_found")
    calls = []

    def lookup(self, fingerprint):
        calls.append(fingerprint)
        return outcome

    monkeypatch.setattr(provider.FileSourceQualificationProvider, "lookup", lookup)
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({}))
    return calls


def test_report_candidate_is_live_validated_cached_and_skips_second_lookup(monkeypatch):
    source, labels, _, fingerprint, records = _typed_pair(monkeypatch)
    calls = _install_candidate_lookup(monkeypatch, fingerprint, records)

    first = api.evaluate_factor_source_batch(
        source, labels, **_request(source))
    assert calls == [fingerprint]
    assert first.metadata["source_qualification_status"] == "qualified_current_source"
    assert first.metadata["source_qualification_applied"] is True
    assert first.metadata["source_qualification_winner"] == "cpu"
    assert first.metadata["source_qualification_origin"] == "report_candidate"
    assert first.metadata["source_qualification_provider_status"] == "candidate_validated"
    assert first.metadata["source_qualification_candidate_id"] == "candidate:test"

    second_source = _Source()
    second = api.evaluate_factor_source_batch(second_source, labels, **_request(second_source))
    assert calls == [fingerprint]
    assert second.metadata["source_qualification_status"] == "qualified_current_source"
    assert second.metadata["source_qualification_cache_status"] == "cache_hit"
    assert second.metadata["source_qualification_origin"] == "process_cache"
    assert second.metadata["source_qualification_provider_status"] == "not_checked"
    assert second.metadata["source_qualification_candidate_id"] is None


def test_explicit_receipts_and_explicit_cpu_backend_do_not_query_provider(monkeypatch):
    source, labels, _, fingerprint, records = _typed_pair(monkeypatch)
    calls = _install_candidate_lookup(monkeypatch, fingerprint, records)

    explicit = api.evaluate_factor_source_batch(
        source, labels, **_request(source), source_qualification=records)
    assert explicit.metadata["source_qualification_origin"] == "explicit_receipts"
    assert explicit.metadata["source_qualification_applied"] is True
    assert calls == []

    cpu_source = _Source()
    cpu = api.evaluate_factor_source_batch(
        cpu_source, labels, metrics=("rank_ic",), backend="cpu",
        max_tile_size=16, source_qualification=records)
    assert cpu.metadata["backend_used"] == "cpu"
    assert cpu.metadata["source_qualification_origin"] == "explicit_receipts"
    assert cpu.metadata["source_qualification_provider_status"] == "not_used_explicit_backend"
    assert cpu.metadata["source_qualification_candidate_id"] is None
    assert calls == []


def test_invalid_report_candidate_context_fails_closed_to_cpu_fallback(monkeypatch):
    source, labels, _, fingerprint, records = _typed_pair(monkeypatch)
    wrong_context = replace(records[0].context,
                            request_content_sha256="0" * 64)
    invalid_records = tuple(replace(record, context=wrong_context)
                            for record in records)
    calls = _install_candidate_lookup(monkeypatch, fingerprint, invalid_records)

    result = api.evaluate_factor_source_batch(source, labels, **_request(source))
    assert calls == [fingerprint]
    assert result.metadata["backend_used"] == "cpu"
    assert result.metadata["source_qualification_applied"] is False
    assert result.metadata["source_qualification_origin"] == "report_candidate"
    assert result.metadata["source_qualification_candidate_id"] == "candidate:test"
    assert result.metadata["source_qualification_status"] in {
        "rejected_legacy_fallback", "not_available_legacy_fallback"}


def test_provider_lookup_exception_is_hidden_and_falls_back_safely(monkeypatch):
    source, labels, _, fingerprint, _ = _typed_pair(monkeypatch)
    secret = "/private/reports/top-secret-report.json"
    calls = []

    def fail_lookup(self, requested_fingerprint):
        calls.append(requested_fingerprint)
        raise OSError(f"failed to open {secret}: credential=do-not-leak")

    monkeypatch.setattr(provider.FileSourceQualificationProvider, "lookup", fail_lookup)
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({}))

    result = api.evaluate_factor_source_batch(source, labels, **_request(source))
    metadata = result.metadata
    assert calls == [fingerprint]
    assert metadata["backend_used"] == "cpu"
    assert metadata["source_qualification_applied"] is False
    assert metadata["source_qualification_status"] == "not_available_legacy_fallback"
    assert metadata["source_qualification_origin"] == "none"
    assert metadata["source_qualification_provider_status"] == "provider_error"
    assert metadata["source_qualification_candidate_id"] is None
    assert secret not in repr(metadata)
    assert "credential=do-not-leak" not in repr(metadata)
