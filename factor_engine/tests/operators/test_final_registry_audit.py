# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def test_duplicate_regression_canonicals_are_direct_aliases() -> None:
    canonicals = set(OperatorRegistry.list_canonical())
    for old, target in {
        "intercept": "ts_regression_intercept",
        "residual": "ts_regression_resid",
        "r_squared": "ts_regression_r2",
    }.items():
        assert old not in canonicals
        assert OperatorRegistry._aliases[old] == target
        assert target in canonicals


def test_explicit_ewm_names_and_hidden_compatibility_aliases() -> None:
    canonicals = set(OperatorRegistry.list_canonical())
    for old, new in {
        "ewm_std": "ts_ewm_std",
        "ewm_var": "ts_ewm_var",
        "ewm_cov": "ts_ewm_cov",
        "ewm_corr": "ts_ewm_corr",
    }.items():
        if new in canonicals:
            assert old not in canonicals
            assert OperatorRegistry._aliases[old] == new
    assert OperatorRegistry._aliases.get("ewm") == "ts_ema"


def test_extended_surface_is_not_default_authoring() -> None:
    from backend.cleaned_bridge import build_cleaned_dsl_allowlist
    from cleaned_operators import operator_surface

    daily = build_cleaned_dsl_allowlist(surface="daily")
    extended = build_cleaned_dsl_allowlist(surface="extended")
    for name in ("acos", "cos", "fix", "reverse", "ts_moment", "winsorize_mean"):
        if name in OperatorRegistry._operators:
            assert operator_surface.classify_canonical(name) == "extended"
            assert name not in daily
            assert name in extended
    for hidden in ("intercept", "ewm", "quantile", "div_or_null", "ts_top_n_avg"):
        assert hidden not in daily


def test_size_neutralize_is_recipe_not_primitive() -> None:
    from factor_recipes.registry import FactorRecipeRegistry

    assert "size_neutralize" not in OperatorRegistry._operators
    recipe = FactorRecipeRegistry.get("size_neutralize")
    assert recipe is not None
    assert "cs_neutralize" in recipe.expression and "log" in recipe.expression


def test_backend_capability_contract_is_closed() -> None:
    from backend.active_capabilities import backend_contract_errors, capability_matrix

    assert backend_contract_errors() == []
    assert set(capability_matrix()) == set(OperatorRegistry.list_canonical())


def test_misleading_polars_registrations_are_removed() -> None:
    for name in ("period_lag", "cs_rank_gaussian", "ts_tail_mean", "ts_time_slope"):
        if name in OperatorRegistry._operators:
            assert "polars" not in OperatorRegistry.backends_for(name)
    for name in (
        "ts_last_if", "ts_true_streak", "ts_argmax", "ts_argmin",
        "ts_topk_mean", "ts_topk_sum", "ts_bottomk_mean", "ts_bottomk_sum",
    ):
        if name in OperatorRegistry._operators:
            assert "polars" in OperatorRegistry.backends_for(name)
            source = OperatorRegistry.catalog()[name]["backend_meta"]["polars"]["source"]
            assert source == "final_expression_native_polars"


def test_production_recipes_execute() -> None:
    from factor_recipes.verification import verify_recipes

    results = verify_recipes(statuses=("production",))
    failures = {name: item.error for name, item in results.items() if not item.pandas_execution_verified}
    assert results and failures == {}
