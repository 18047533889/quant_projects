import pytest

@pytest.mark.parametrize("formula",[
    "row_sum_skipna(close, open)",
    "row_sum_skipna(close, 0.25)",
    "row_sum_skipna(close, open, 0.25, min_count=2)",
])
def test_variadic_inputs_do_not_bind_keyword_only_min_count(formula):
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    engine.compile(Factor(name="varargs",expr=parser.parse(formula),
                          source_expr=formula,surface="compat_research"))

@pytest.mark.parametrize("formula",[
    "row_sum_skipna(close, open, min_count=close)",
    "row_sum_skipna(close, open, min_count=0.25)",
])
def test_invalid_keyword_only_control_still_fails(formula):
    from evidence.factor_catalog_20260915.compile_catalog import build_runtime
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    with pytest.raises((NotImplementedError,TypeError,ValueError)):
        engine.compile(Factor(name="invalid",expr=parser.parse(formula),
                              source_expr=formula,surface="compat_research"))
