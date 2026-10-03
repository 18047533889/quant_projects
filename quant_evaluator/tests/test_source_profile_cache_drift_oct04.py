from types import SimpleNamespace

import pytest

from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime import source_profile_router as router

# Reuse the typed immutable profile fixtures so these exercise the production cache.
from test_source_profile_live_guards_oct04 import _h, _qualify


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    router.profile_cache.clear_validated_records()
    yield
    router.profile_cache.clear_validated_records()


def _cache_kwargs(source, metadata, policy):
    return dict(source=source, metadata=metadata, metrics=("rank_ic",),
                request_fingerprint=_h("a"), requested_tile_size=16, policy=policy)


def _qualified(monkeypatch):
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    source, metadata, _, decision = _qualify(monkeypatch, 48, 5, policy, 2)
    return source, metadata, policy, decision


def test_post_run_cache_discard_uses_original_key_after_budget_drift(monkeypatch):
    source, metadata, policy, decision = _qualified(monkeypatch)
    kwargs = _cache_kwargs(source, metadata, policy)
    original_key = decision.cache_key
    assert original_key == router.source_profile_cache_key(**kwargs)
    assert router.profile_cache.get_validated_records(original_key) is not None

    source.admitted_max_tile_size = 4
    router.discard_source_route_profile_cache(**kwargs, cache_key=decision.cache_key)

    assert router.profile_cache.get_validated_records(original_key) is None


def test_post_run_cache_discard_keeps_no_drift_behavior(monkeypatch):
    source, metadata, policy, _ = _qualified(monkeypatch)
    kwargs = _cache_kwargs(source, metadata, policy)
    original_key = router.source_profile_cache_key(**kwargs)

    router.discard_source_route_profile_cache(**kwargs, cache_key=original_key)

    assert router.profile_cache.get_validated_records(original_key) is None


def test_post_run_cache_discard_cleans_original_entry_when_live_key_rebuild_fails(
        monkeypatch):
    source, metadata, policy, _ = _qualified(monkeypatch)
    kwargs = _cache_kwargs(source, metadata, policy)
    original_key = router.source_profile_cache_key(**kwargs)
    source.admitted_max_tile_size = 0  # Current-key recomputation now rejects admission.

    router.discard_source_route_profile_cache(**kwargs, cache_key=original_key)

    assert router.profile_cache.get_validated_records(original_key) is None


def test_api_forwards_qualification_cache_key_on_execution_mismatch(monkeypatch):
    from quant_evaluator.api import factor_source as api
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
    from quant_evaluator.tests.test_source_route_profiles_api_oct03 import (
        _Source, _decision, _labels)

    source = _Source()
    policy = GPUExecutionPolicy(max_factor_tile_size=2)
    decision = _decision("cpu")
    decision.cache_key = "original-qualified-cache-key"
    observed = []
    monkeypatch.setattr(api, "get_cached_source_route_profile_records",
                        lambda **kwargs: (object(), object()))
    monkeypatch.setattr(api, "is_source_route_profile_pair", lambda value: True)
    monkeypatch.setattr(api, "qualify_source_route_profiles",
                        lambda **kwargs: decision)
    monkeypatch.setattr(api, "validate_source_route_profile_execution",
                        lambda **kwargs: "qualified_profile_live_context_changed")
    monkeypatch.setattr(api, "discard_source_route_profile_cache",
                        lambda **kwargs: observed.append(kwargs))
    monkeypatch.setattr(api, "select_source_auto_route", lambda **kwargs: None)

    result = api.evaluate_factor_source_batch(
        source, _labels(source), metrics=("rank_ic",), backend="auto",
        max_tile_size=16, gpu_policy=policy)

    assert result.metadata["source_qualification_applied"] is False
    assert len(observed) == 1
    assert observed[0]["cache_key"] == "original-qualified-cache-key"
