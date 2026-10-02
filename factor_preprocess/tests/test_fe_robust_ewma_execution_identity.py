"""Scoped identity coverage for the FE robust EWMA composite."""
from factor_preprocess.registry.transforms import create_default_registry


def test_robust_ewma_identity_binds_live_fe_route_and_preserves_registry_seal(monkeypatch):
    from factor_engine.backend import native_long_robust_ewma

    registry = create_default_registry()
    metadata = registry.get("robust_ewma")
    legacy_hashes = (
        metadata.signature_hash, metadata.implementation_hash,
        metadata.numeric_policy_hash,
    )
    seal = registry.seal()
    executor = registry.get_execution("robust_ewma")
    identity = executor.execution_identity

    assert identity["status"] == "bound"
    assert identity["execution_origin"] == "FE_COMPOSITE"
    assert identity["identity_kind"] == "FE_COMPOSITE_SCOPED"
    assert identity["recipe_identity"] == metadata.fe_equivalent_semantics
    assert identity["adapter"].endswith("execute_robust_ewma")
    assert identity["fe_scoped_digest"] == identity["scoped_identity"]["digest"]
    assert "not a full transitive runtime closure" in identity["coverage_marker"]
    assert registry.snapshot_identity == seal

    original = identity["digest"]

    def changed_kernel(frame, **kwargs):
        return frame

    monkeypatch.setattr(native_long_robust_ewma, "lagged_robust_ewma", changed_kernel)
    changed_entry = executor.execution_identity
    assert changed_entry["digest"] != original

    def changed_helper(value):
        return float(value) + 1.0

    monkeypatch.setattr(native_long_robust_ewma, "_winsor_width", changed_helper)
    changed_helper_identity = executor.execution_identity
    assert changed_helper_identity["digest"] != changed_entry["digest"]
    assert registry.snapshot_identity == seal
    assert (
        metadata.signature_hash, metadata.implementation_hash,
        metadata.numeric_policy_hash,
    ) == legacy_hashes


def test_robust_ewma_identity_rejects_wrong_name_or_recipe():
    import pytest

    from factor_preprocess.adapters.fe_robust_ewma_identity import (
        get_robust_ewma_identity,
    )
    from factor_preprocess.adapters.fe_smoothing import ROBUST_EWMA_RECIPE
    from factor_preprocess.errors import GovernanceError

    with pytest.raises(GovernanceError):
        get_robust_ewma_identity("ewma", ROBUST_EWMA_RECIPE)
    with pytest.raises(GovernanceError):
        get_robust_ewma_identity("robust_ewma", "wrong-recipe")


def test_robust_ewma_identity_rejects_fe_recipe_drift(monkeypatch):
    import pytest

    from factor_engine.backend import robust_ewma_execution_identity as fe_identity
    from factor_preprocess.adapters.fe_robust_ewma_identity import (
        get_robust_ewma_identity,
    )
    from factor_preprocess.adapters.fe_smoothing import ROBUST_EWMA_RECIPE
    from factor_preprocess.errors import GovernanceError

    monkeypatch.setattr(fe_identity, "ROBUST_EWMA_RECIPE", "FE_COMPOSITE:wrong:v1")
    with pytest.raises(GovernanceError, match="schema or recipe"):
        get_robust_ewma_identity("robust_ewma", ROBUST_EWMA_RECIPE)


def test_identity_tracks_wrapper_bound_helpers_not_source_module_rebinding(monkeypatch):
    from factor_engine.backend import long_robust_ewm, long_ewm
    from factor_preprocess.adapters.fe_robust_ewma_identity import (
        get_robust_ewma_identity,
    )
    from factor_preprocess.adapters.fe_smoothing import ROBUST_EWMA_RECIPE

    original = get_robust_ewma_identity("robust_ewma", ROBUST_EWMA_RECIPE)["digest"]

    def irrelevant_rebinding(value):
        return value

    monkeypatch.setattr(long_ewm, "halflife_to_alpha", irrelevant_rebinding)
    assert get_robust_ewma_identity("robust_ewma", ROBUST_EWMA_RECIPE)["digest"] == original

    monkeypatch.setattr(long_robust_ewm, "halflife_to_alpha", irrelevant_rebinding)
    changed = get_robust_ewma_identity("robust_ewma", ROBUST_EWMA_RECIPE)["digest"]
    assert changed != original


def test_identity_tracks_constants_and_rejects_invalid_or_missing_state(monkeypatch):
    import pytest

    from factor_engine.backend import native_long_robust_ewma
    from factor_preprocess.adapters.fe_robust_ewma_identity import (
        get_robust_ewma_identity,
    )
    from factor_preprocess.adapters.fe_smoothing import ROBUST_EWMA_RECIPE
    from factor_preprocess.errors import GovernanceError

    registry = create_default_registry()
    original = registry.execution_identity("robust_ewma")["digest"]
    monkeypatch.setattr(native_long_robust_ewma, "_WINDOW", 11)
    assert registry.execution_identity("robust_ewma")["digest"] != original

    monkeypatch.setattr(native_long_robust_ewma, "_STD_FLOOR", float("nan"))
    with pytest.raises(GovernanceError, match="identity is unbound"):
        registry.execution_identity("robust_ewma")

    monkeypatch.delattr(native_long_robust_ewma, "_WINDOW")
    with pytest.raises(GovernanceError, match="identity is unbound"):
        get_robust_ewma_identity("robust_ewma", ROBUST_EWMA_RECIPE)


def test_identity_rejects_kernel_polars_binding_drift(monkeypatch):
    import pytest

    from factor_engine.backend import native_long_robust_ewma
    from factor_preprocess.errors import GovernanceError

    registry = create_default_registry()
    monkeypatch.setattr(native_long_robust_ewma, "pl", object())
    with pytest.raises(GovernanceError, match="identity is unbound"):
        registry.execution_identity("robust_ewma")


def test_identity_rejects_scoped_schema_drift(monkeypatch):
    from dataclasses import replace
    import pytest
    from factor_engine.backend import robust_ewma_execution_identity as provider
    from factor_preprocess.errors import GovernanceError

    identity = provider.build_robust_ewma_identity()
    monkeypatch.setattr(provider, "build_robust_ewma_identity", lambda: replace(
        identity, schema="unsupported-schema",
    ))
    with pytest.raises(GovernanceError, match="schema or recipe"):
        create_default_registry().execution_identity("robust_ewma")


def test_identity_rejects_invalid_constants_independently(monkeypatch):
    import pytest
    from factor_engine.backend import native_long_robust_ewma
    from factor_preprocess.errors import GovernanceError

    registry = create_default_registry()
    for name, value in (
        ("_WINDOW", True), ("_WINDOW", 0), ("_WINDOW", 10.0),
        ("_ROLLING_MIN_SAMPLES", -1), ("_ROLLING_MIN_SAMPLES", False),
        ("_STD_FLOOR", 0.0), ("_STD_FLOOR", float("inf")),
        ("_STD_FLOOR", True), ("_STD_FLOOR", "1e-10"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(native_long_robust_ewma, name, value)
            with pytest.raises(GovernanceError, match="identity is unbound"):
                registry.execution_identity("robust_ewma")
