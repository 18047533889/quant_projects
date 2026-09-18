"""Numerical repairs must not reuse pre-repair factor identities."""
import pytest
from factor_engine.backend import operator_semantic_version as versions
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.factor_identity import OperatorSemanticContractDigest


@pytest.mark.parametrize("canonical,current", [
    ("ts_poly2_resid", 2), ("ts_extremal_index", 2),
    ("ts_gpd_shape_pwm", 2), ("ts_deviation_from_mean", 2),
    ("ts_km_equilibrium_distance", 3),
])
def test_repaired_formula_versions_reach_identity_and_change_hash(monkeypatch, canonical, current):
    load_all()
    assert versions.semantic_version(canonical) == current
    repaired = OperatorSemanticContractDigest.for_canonical(canonical)
    assert repaired.semantic_version == f"{current}.0"
    # The catalog snapshots this version during registration. Simulate an old
    # loaded catalog as well as the old version map, not a live hot reload.
    monkeypatch.setitem(versions.OPERATOR_SEMANTIC_VERSIONS, canonical, current - 1)
    old_row = {**OperatorRegistry._catalog[canonical], "semantic_version": f"{current - 1}.0"}
    old_catalog = {**OperatorRegistry._catalog, canonical: old_row}
    monkeypatch.setattr(OperatorRegistry, "_catalog", old_catalog)
    previous = OperatorSemanticContractDigest.for_canonical(canonical)
    assert previous.contract_hash() != repaired.contract_hash()
