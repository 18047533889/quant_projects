"""Parity tests for the public FP-to-FE effective-rank OLS composite."""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.neutralization import ols_neutralize
from factor_preprocess.registry.transforms import get_default_registry
from factor_preprocess.adapters.fe_neutralization import (
    OLS_NEUTRALIZATION_RECIPE, get_fe_neutralization_executor,
)


_FE_AVAILABLE = get_fe_neutralization_executor("ols_neutralize", OLS_NEUTRALIZATION_RECIPE) is not None
needs_fe = pytest.mark.skipif(
    not _FE_AVAILABLE,
    reason="FactorEngine is not importable in this FP-only environment",
)


def _inputs(exposure_columns):
    n = 12
    assets = [f"a{i}" for i in range(n)]
    dates = pd.Timestamp("2026-01-02")
    values = pd.DataFrame({
        "date": [dates] * n,
        "asset_id": assets,
        "value": [0.2, 1.7, -0.4, 3.1, 2.0, -1.2, 4.4, 0.8, 5.2, -2.1, 3.7, 1.3],
    })
    exposures = pd.DataFrame({"date": [dates] * n, "asset_id": assets})
    for name, column in exposure_columns.items():
        exposures[name] = column
    return values, exposures


def _fe_registry(values, exposures, *, min_observations):
    registry = get_default_registry()
    assert registry.resolve_origin("ols_neutralize") == "FE_COMPOSITE"
    return registry.get_execution("ols_neutralize")(
        values, exposures, min_observations=min_observations
    )


@needs_fe
@pytest.mark.parametrize("case", ["duplicated", "zero"])
def test_fe_composite_matches_rank_deficient_designs(case):
    x = np.arange(12, dtype=float)
    if case == "duplicated":
        columns = {"x": x, "x_duplicate": 2.0 * x + 1.0}
    else:
        columns = {"x": x, "zero": np.zeros_like(x)}
    values, exposures = _inputs(columns)

    fp = ols_neutralize(values, exposures, min_observations=4, add_intercept=True)
    fe = _fe_registry(values, exposures, min_observations=4)

    np.testing.assert_allclose(fe, fp, rtol=1e-12, atol=1e-12, equal_nan=True)


@needs_fe
def test_fe_composite_matches_ill_conditioned_design():
    x = np.linspace(-1.0, 1.0, 12)
    perturbation = np.resize(np.array([-1.0, 1.0]), len(x)) * 1e-14
    design = np.column_stack((np.ones_like(x), x, x + perturbation))
    assert np.linalg.cond(design) > 1e12
    values, exposures = _inputs({"x": x, "near_duplicate": x + perturbation})

    fp = ols_neutralize(values, exposures, min_observations=4, add_intercept=True)
    fe = _fe_registry(values, exposures, min_observations=4)

    np.testing.assert_allclose(fe, fp, rtol=1e-12, atol=1e-12, equal_nan=True)


@needs_fe
def test_fe_composite_matches_minimum_at_coefficient_count():
    x = np.array([0.0, 1.0, 2.0])
    values, exposures = _inputs({"x": np.resize(x, 12)})
    values = values.iloc[:3].copy()
    exposures = exposures.iloc[:3].copy()

    fp = ols_neutralize(values, exposures, min_observations=2, add_intercept=True)
    fe = _fe_registry(values, exposures, min_observations=2)
    np.testing.assert_allclose(fe, fp, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(fe.to_numpy()).all()


@needs_fe
@pytest.mark.parametrize(
    "name", ["ols_neutralize", "industry_neutral", "size_neutral", "dual_neutral"]
)
def test_all_ols_aliases_use_same_effective_rank_composite(name):
    x = np.arange(12, dtype=float)
    values, exposures = _inputs({"x": x, "duplicate": 2 * x + 1})
    registry = get_default_registry()
    assert registry.resolve_origin(name) == "FE_COMPOSITE"
    meta = registry.get(name)
    assert meta.fe_equivalent_semantics == OLS_NEUTRALIZATION_RECIPE
    actual = registry.get_execution(name)(values, exposures, min_observations=4)
    expected = ols_neutralize(values, exposures, min_observations=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_neutralization_composite_fails_closed_without_FE_and_research_fallback_is_explicit(monkeypatch):
    import factor_preprocess.adapters.fe_composite as composite_adapter
    from factor_preprocess.errors import GovernanceError

    registry = get_default_registry()
    monkeypatch.setattr(composite_adapter, "get_fe_composite_executor", lambda *args: None)
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("ols_neutralize")
    research = registry.get_execution("ols_neutralize", allow_research=True)
    values, exposures = _inputs({"x": np.arange(12, dtype=float)})
    actual = research(values, exposures, min_observations=4)
    expected = ols_neutralize(values, exposures, min_observations=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_neutralization_resolver_rejects_wrong_recipe_identity():
    from factor_preprocess.errors import GovernanceError
    with pytest.raises(GovernanceError, match="not registered"):
        get_fe_neutralization_executor("ols_neutralize", "wrong:v9")
