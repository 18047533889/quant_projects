"""Narrow execution identity checks for FP's EWMA and IIR FE routes."""
import pytest

from factor_preprocess.errors import GovernanceError
from factor_preprocess.registry.transforms import create_default_registry


@pytest.mark.parametrize("name", ["ewma", "one_sided_iir_lowpass"])
def test_smoothing_identity_binds_live_fe_route_and_fp_adapter_without_registry_seal(
    monkeypatch, name,
):
    registry = create_default_registry()
    metadata = registry.get(name)
    seal = registry.seal()
    identity = registry.execution_identity(name)
    executor = registry.get_execution(name)
    assert executor.execution_identity["digest"] == identity["digest"]
    assert identity["status"] == "bound"
    assert identity["execution_origin"] == "FE_COMPOSITE"
    assert identity["identity_kind"] == "FE_COMPOSITE_SCOPED"
    assert identity["recipe_identity"] == metadata.fe_equivalent_semantics
    assert identity["fp_adapter_digest"]

    from factor_engine.backend import long_ewm, long_smoothing
    target_module, target_name = (
        (long_smoothing, "lagged_ewma") if name == "ewma"
        else (long_ewm, "lagged_iir_lowpass")
    )
    before = identity["digest"]
    monkeypatch.setattr(target_module, target_name, lambda *a, **k: None)
    assert registry.execution_identity(name)["digest"] != before
    assert registry.seal() == seal


@pytest.mark.parametrize("name,recipe", [
    ("ewma", "FE_COMPOSITE:long_smoothing.lagged_ewma:v1"),
    ("one_sided_iir_lowpass", "FE_COMPOSITE:long_ewm.lagged_iir_lowpass:v1"),
])
def test_smoothing_identity_rejects_cross_name_or_recipe(name, recipe):
    from factor_preprocess.adapters.fe_smoothing_identity import get_smoothing_identity

    other = "ewma" if name == "one_sided_iir_lowpass" else "one_sided_iir_lowpass"
    with pytest.raises(GovernanceError):
        get_smoothing_identity(other, recipe)
    with pytest.raises(GovernanceError):
        get_smoothing_identity(name, "wrong-recipe")


@pytest.mark.parametrize("name", ["ewma", "one_sided_iir_lowpass"])
def test_smoothing_identity_tracks_actual_fp_adapter_callable(monkeypatch, name):
    from factor_preprocess.adapters import fe_smoothing

    registry = create_default_registry()
    before = registry.execution_identity(name)["digest"]
    adapter_name = "execute_ewma" if name == "ewma" else "execute_iir_lowpass"
    monkeypatch.setattr(fe_smoothing, adapter_name, lambda *a, **k: None)
    assert registry.execution_identity(name)["digest"] != before


@pytest.mark.parametrize("name", ["ewma", "one_sided_iir_lowpass"])
def test_smoothing_identity_fails_closed_if_fe_runtime_is_missing(monkeypatch, name):
    import sys

    registry = create_default_registry()
    monkeypatch.setitem(sys.modules, "polars", None)
    with pytest.raises(GovernanceError, match="identity is unbound"):
        registry.execution_identity(name)


def test_smoothing_identity_checks_native_polars_global_if_kernel_uses_one(monkeypatch):
    from factor_engine.backend import native_long_ewm

    kernel_globals = native_long_ewm.collect_lagged_ewma.__globals__
    if "pl" not in kernel_globals:
        # Current kernel imports Polars locally; no module-level alias exists.
        assert not hasattr(native_long_ewm, "pl")
        return

    registry = create_default_registry()
    monkeypatch.setitem(kernel_globals, "pl", object())
    with pytest.raises(GovernanceError, match="identity is unbound"):
        registry.execution_identity("ewma")
