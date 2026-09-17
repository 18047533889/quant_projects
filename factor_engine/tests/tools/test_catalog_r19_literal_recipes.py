import pytest
from factor_engine.tools.catalog_r19_literal_recipes import migrate_formula

def test_registered_enum_is_quoted():
    formula,notes=migrate_formula("ts_quantilogram(ret, side=lower)")
    assert formula=="ts_quantilogram(ret, side='lower')"
    assert notes

@pytest.mark.parametrize("formula",[
    "ts_quantilogram(ret, side=close)",
    "ts_quantilogram(ret, window=close)",
    "missing_operator(ret, side=lower)",
    "rand_normal(ret, side=lower)",
    "ts_quantilogram(ret, side='lower')",
    "ts_quantilogram(ret, side=lower(", 
])
def test_unknown_or_non_enum_is_not_invented(formula):
    assert migrate_formula(formula)==(formula,[])
