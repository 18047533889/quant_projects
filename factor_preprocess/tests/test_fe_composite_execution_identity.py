"""Scoped execution-identity bridge tests for FE-owned preprocessing composites."""
import builtins
import pandas as pd
import pytest

from factor_preprocess.errors import GovernanceError
from factor_preprocess.registry.transforms import create_default_registry


def _block_factor_engine_imports(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("factor_engine"):
            raise ModuleNotFoundError("simulated FP-only environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_ols_identity_is_scoped_and_attached_to_actual_fe_adapter():
    registry = create_default_registry()
    meta = registry.get("ols_neutralize")
    legacy_hashes = (
        meta.signature_hash, meta.implementation_hash, meta.numeric_policy_hash,
    )
    seal_digest = registry.seal()

    identity = registry.execution_identity("ols_neutralize")
    executor = registry.get_execution("ols_neutralize")

    assert identity["status"] == "bound"
    assert identity["execution_origin"] == "FE_COMPOSITE"
    assert identity["identity_kind"] == "FE_COMPOSITE_SCOPED"
    assert identity["recipe_identity"] == meta.fe_equivalent_semantics
    assert identity["adapter"].endswith("execute_ols_neutralize")
    assert identity["fe_scoped_digest"] == identity["scoped_identity"]["digest"]
    assert "not a full transitive runtime closure" in identity["coverage_marker"]
    assert executor.execution_identity == identity
    assert registry.snapshot_identity == seal_digest
    assert (
        meta.signature_hash, meta.implementation_hash, meta.numeric_policy_hash,
    ) == legacy_hashes


def test_missing_fe_fails_closed_unless_research_fallback_is_explicit(monkeypatch):
    registry = create_default_registry()
    with monkeypatch.context() as blocked:
        _block_factor_engine_imports(blocked)
        with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
            registry.get_execution("ols_neutralize")

    with monkeypatch.context() as blocked:
        _block_factor_engine_imports(blocked)
        identity = registry.execution_identity("ols_neutralize", allow_research=True)
        executor = registry.get_execution("ols_neutralize", allow_research=True)

    assert identity["identity_kind"] == "FP_RESEARCH_FALLBACK"
    assert identity["execution_origin"] == "FP_RESEARCH_FALLBACK"
    assert identity["execution_origin"] != "FE_COMPOSITE"
    assert "not an FE composite identity" in identity["coverage_marker"]
    assert executor.execution_identity == identity


def test_smoothing_composite_has_explicit_unbound_identity():
    registry = create_default_registry()

    with pytest.raises(GovernanceError, match="No scoped FE composite identity provider"):
        registry.execution_identity("rolling_zscore")
    with pytest.raises(GovernanceError, match="No scoped FE composite identity provider"):
        registry.execution_identity("ewma")


def test_research_opt_in_still_uses_fe_when_composite_executor_is_available(monkeypatch):
    from factor_preprocess.adapters import fe_composite

    sentinel = lambda *args, **kwargs: "FE"
    monkeypatch.setattr(
        fe_composite, "get_fe_composite_executor", lambda name, recipe: sentinel
    )
    monkeypatch.setattr(
        fe_composite, "get_fe_composite_identity",
        lambda name, recipe: {
            "status": "bound", "execution_origin": "FE_COMPOSITE",
            "identity_kind": "FE_COMPOSITE_SCOPED", "digest": "scoped-test",
        },
    )
    registry = create_default_registry()

    executor = registry.get_execution("ols_neutralize", allow_research=True)

    assert executor.executor is sentinel
    assert executor.execution_identity["execution_origin"] == "FE_COMPOSITE"


def test_smoothing_execution_route_survives_unavailable_scoped_identity(monkeypatch):
    from factor_preprocess.adapters import fe_composite

    sentinel = lambda *args, **kwargs: "FE smoothing"
    monkeypatch.setattr(
        fe_composite, "get_fe_composite_executor",
        lambda name, recipe: sentinel if name == "rolling_zscore" else None,
    )
    registry = create_default_registry()

    executor = registry.get_execution("rolling_zscore")

    assert executor.executor is sentinel
    assert executor.execution_identity is None


def test_live_fe_entrypoint_and_helper_changes_identity_not_seal(monkeypatch):
    from factor_engine.backend import long_neutralization

    registry = create_default_registry()
    seal_digest = registry.seal()
    original = registry.execution_identity("ols_neutralize")
    executor = registry.get_execution("ols_neutralize")
    assert executor.execution_identity["digest"] == original["digest"]

    def changed_entrypoint(
        values, exposures, date_col="date", asset_col="asset_id",
        value_col="value", min_observations=10, add_intercept=True,
    ):
        return "patched FE body"

    monkeypatch.setattr(long_neutralization, "ols_effective_rank", changed_entrypoint)
    changed_entry = executor.execution_identity
    values = pd.DataFrame({"date": [1], "asset_id": ["a"], "value": [1.0]})
    exposures = pd.DataFrame({"date": [1], "asset_id": ["a"], "x": [0.0]})
    assert executor(values, exposures) == "patched FE body"

    def changed_helper(base, used):
        return f"x_{base}"

    monkeypatch.setattr(long_neutralization, "_private_name", changed_helper)
    changed_helper_identity = executor.execution_identity

    assert original["digest"] != changed_entry["digest"]
    assert changed_entry["digest"] != changed_helper_identity["digest"]
    assert registry.snapshot_identity == seal_digest
