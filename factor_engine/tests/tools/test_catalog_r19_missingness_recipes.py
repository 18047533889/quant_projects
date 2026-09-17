from factor_engine.tools.catalog_r19_missingness_recipes import migrate_formula

import pytest

@pytest.mark.parametrize("field", ["roe", "roa", "gross_profit_margin", "net_profit_margin"])
def test_missing_financial_values_are_preserved(field):
    source=f"where(ge(cs_coverage_ratio({field}),0.7), rank(cs_impute_median({field},20)), rank(neg(ts_std(ret,20))))"
    result,notes=migrate_formula(source)
    assert result==source.replace(f"cs_impute_median({field},20)",field)
    assert notes
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    engine.compile(Factor(name="roe",expr=parser.parse(result),source_expr=result,surface="compat_research"))

def test_other_inputs_unchanged():
    source="cs_impute_mean(volume,20)"
    assert migrate_formula(source)==(source,[])
