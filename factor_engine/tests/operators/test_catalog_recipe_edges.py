import ast
import pytest
from factor_engine.tools.catalog_recipe_migration import migrate_catalog_recipe_formula

@pytest.mark.parametrize("formula", ["ROC(close,)", "ROC(close, # trailing comment\n)", "MOM(close,)"])
def test_legacy_default_with_trailing_comma_remains_valid(formula):
    result = migrate_catalog_recipe_formula(formula)
    parsed = ast.parse(result.formula, mode="eval").body
    if formula.startswith("MOM"):
        assert parsed.func.id == "ts_delta"
        assert parsed.args[1].value == 10
    else:
        assert parsed.func.id == "multiply"
        assert parsed.args[0].func.id == "ts_pct"
        assert parsed.args[0].args[1].value == 10

def test_keyword_comment_does_not_move_keyword_edit():
    result = migrate_catalog_recipe_formula("BollingerUpper(close, std_dev= # std_dev\n 2)")
    parsed = ast.parse(result.formula, mode="eval").body
    assert parsed.func.id == "add"
    assert parsed.args[1].args[0].value == 2

def test_hill_support_count_preserved_and_idempotent():
    result = migrate_catalog_recipe_formula("ts_hill_tail_index(ret, window=252, side='lower', tail_fraction=.2, min_tail=10)")
    assert "min_tail_count=10" in result.formula
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula
    assert not migrate_catalog_recipe_formula(result.formula).changes

@pytest.mark.parametrize("formula", [
    "ts_hill_tail_index(ret, min_tail=10, min_tail_count=5)",
    "ts_hill_tail_index(ret, 252, 'lower', .2, 5, min_tail=10)",
    "other(ret, min_tail=10)",
])
def test_ambiguous_tail_alias_is_not_silently_repaired(formula):
    assert migrate_catalog_recipe_formula(formula).formula == formula
@pytest.mark.parametrize("formula", [
    "ts_spectral_entropy(turnover_ratio, window=60)",
    "ts_spectral_entropy(x=turnover_ratio, window=60)",
    "ts_spectral_entropy(TurnoverRatio, 60)",
])
def test_direct_turnover_entropy_opts_into_activity_canonical(formula):
    result = migrate_catalog_recipe_formula(formula)
    assert result.formula.startswith("ts_activity_spectral_entropy(")
    assert result.changes == (
        "ts_spectral_entropy(turnover activity) -> ts_activity_spectral_entropy",)
    assert migrate_catalog_recipe_formula(result.formula).formula == result.formula


@pytest.mark.parametrize("formula", [
    "ts_spectral_entropy(ret, window=60)",
    "ts_spectral_entropy(add(turnover_ratio, 1), window=60)",
    "ts_spectral_entropy(obj.turnover_ratio, window=60)",
    "ts_spectral_entropy(turnover_ratio, x=turnover_ratio, window=60)",
    "obj.ts_spectral_entropy(turnover_ratio, window=60)",
])
def test_activity_entropy_migration_does_not_guess_expression_types(formula):
    assert migrate_catalog_recipe_formula(formula).formula == formula
