import pytest
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula


@pytest.mark.parametrize("suffix", [", 'monthly'", ", correction='monthly'"])
def test_explicit_fixed_monthly_policy_migrates_to_current_default(suffix):
    source = "ts_abdi_ranaldo_spread(close, high, low, 20" + suffix + ")"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == "ts_abdi_ranaldo_spread(close, high, low, 20)"
    assert result.changes
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula


@pytest.mark.parametrize("source", [
    "ts_abdi_ranaldo_spread(close, high, low, 20, 'daily')",
    "ts_abdi_ranaldo_spread(close, high, low, 20, policy)",
    "ts_abdi_ranaldo_spread(close, high, low, 20, 'monthly', correction='monthly')",
    "ts_abdi_ranaldo_spread(close, high, low, 20, 'monthly', other=True)",
])
def test_ambiguous_or_unsupported_policy_is_never_discarded(source):
    assert migrate_catalog_recipe_formula(source).formula == source


def test_real_catalog_nested_factor_only_removes_fixed_policy():
    source = "subtract(rank(ichimoku_cloud_width(high, low, 5, 10, 20)), rank(ts_abdi_ranaldo_spread(close, high, low, 20, 'monthly')))"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == source.replace(", 'monthly'", "")


def test_malformed_catalog_ohlc_signature_is_repaired_exactly():
    source = "ts_abdi_ranaldo_spread(open, high, low, close, 20)"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == "ts_abdi_ranaldo_spread(close, high, low, 20)"
    assert result.changes
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula


def test_malformed_catalog_ohlc_signature_is_repaired_inside_formula():
    source = "add(ts_abdi_ranaldo_spread(open, high, low, close, 20), close)"
    result = migrate_catalog_recipe_formula(source)
    assert result.formula == "add(ts_abdi_ranaldo_spread(close, high, low, 20), close)"
    assert result.changes


@pytest.mark.parametrize("source", [
    "ts_abdi_ranaldo_spread(close, high, low, open, 20)",
    "ts_abdi_ranaldo_spread(open, high, low, add(close, 0), 20)",
    "ts_abdi_ranaldo_spread(o, high, low, close, 20)",
    "ts_abdi_ranaldo_spread(open, high, low, close, 20.0)",
    "ts_abdi_ranaldo_spread(open, high, low, close, True)",
    "ts_abdi_ranaldo_spread(open, high, low, close, window=20)",
    "ts_abdi_ranaldo_spread(open, high, low, close, *windows)",
    "ts_abdi_ranaldo_spread(open, high, low, close, 20, 'monthly')",
])
def test_malformed_catalog_signature_is_not_guessed(source):
    assert migrate_catalog_recipe_formula(source).formula == source


def test_nested_outer_recipe_never_uses_stale_overlapping_offsets():
    import ast
    source = "add(ROC(ts_abdi_ranaldo_spread(close, high, low, 20, 'monthly'), 5), close)"
    result = migrate_catalog_recipe_formula(source)
    tree = ast.parse(result.formula, mode="eval")
    assert ast.unparse(tree.body.args[1]) == "close"
    assert "'monthly'" not in result.formula
    assert "ts_abdi_ranaldo_spread(close, high, low, 20)" in result.formula


def test_default_matches_explicit_monthly_numerically():
    import numpy as np
    import pandas as pd
    from factor_engine.cleaned_operators.ohlc_spread import _ts_abdi_ranaldo_spread
    close = pd.DataFrame(100 + np.sin(np.arange(50) / 3.0))
    explicit = _ts_abdi_ranaldo_spread(close, close + 2, close - 2, 20, "monthly")
    implicit = _ts_abdi_ranaldo_spread(close, close + 2, close - 2, 20)
    pd.testing.assert_frame_equal(explicit, implicit)
    assert np.isfinite(implicit.to_numpy()).any()
