from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend import operator_semantic_version as versions
from factor_engine.runtime.factor_identity import OperatorSemanticContractDigest
import pytest


def test_authoritative_semantic_bump_changes_identity_without_wrapper_edit(monkeypatch):
    load_all()
    canonical = "ts_kalman_beta"
    catalog = dict(OperatorRegistry._catalog)
    catalog[canonical] = {**catalog[canonical], "semantic_version": None}
    monkeypatch.setattr(OperatorRegistry, "_catalog", catalog)
    monkeypatch.setitem(versions.OPERATOR_SEMANTIC_VERSIONS, canonical, 2)
    first = OperatorSemanticContractDigest.for_canonical(canonical)
    monkeypatch.setitem(versions.OPERATOR_SEMANTIC_VERSIONS, canonical, 3)
    second = OperatorSemanticContractDigest.for_canonical(canonical)
    assert first.semantic_version == "2.0"
    assert second.semantic_version == "3.0"
    assert first.implementation_hash == second.implementation_hash
    assert first.contract_hash() != second.contract_hash()


def test_explicit_declared_version_is_preserved(monkeypatch):
    load_all()
    canonical = "ts_kalman_beta"
    catalog = dict(OperatorRegistry._catalog)
    catalog[canonical] = {**catalog[canonical], "semantic_version": "custom-3"}
    monkeypatch.setattr(OperatorRegistry, "_catalog", catalog)
    assert OperatorSemanticContractDigest.for_canonical(canonical).semantic_version == "custom-3"


@pytest.mark.parametrize(("canonical", "version"), [
    ("ts_sample_entropy", 2),
    ("ts_market_liquidity_beta", 2),
    ("ts_ar_prior_coeff", 2),
    ("ts_multi_regression_r2", 2),
    ("ts_ridge_regression_coeff", 3),
    ("ts_ridge_regression_forecast_error_z", 3),
])
def test_v8_changed_contract_versions_reach_public_identity(canonical, version):
    load_all()
    assert versions.semantic_version(canonical) == version
    assert OperatorSemanticContractDigest.for_canonical(canonical).semantic_version == f"{version}.0"
