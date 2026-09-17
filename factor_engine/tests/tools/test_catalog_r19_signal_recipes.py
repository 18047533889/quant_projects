import pytest
from factor_engine.tools.catalog_r19_signal_recipes import migrate_formula

@pytest.mark.parametrize("name",["ret_1d","rel_volume","amihud"])
def test_shorthand_expands_to_registered_real_fields(name):
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from evidence.factor_catalog_20260915.smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    formula,changes=migrate_formula(f"ts_mean({name}, 20)")
    expr=parser.parse(formula)
    assert changes
    assert not bind_fields(expr)[1]
    engine.compile(Factor(name=name,expr=expr,source_expr=formula,surface="compat_research"))

@pytest.mark.parametrize("formula",["ts_mean(oi_spread,20)","amihud(ret)","field('amihud')","x("])
def test_unknown_data_or_function_not_replaced(formula):
    assert migrate_formula(formula)==(formula,[])
