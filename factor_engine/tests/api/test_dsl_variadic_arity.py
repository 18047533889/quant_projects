import pytest

from factor_engine.api.dsl_parser import DSLParseError, DSLParser, DSLUnknownOperatorError


def _row_sum_formula(count: int) -> str:
    fields = ", ".join(f"f{i}" for i in range(count))
    return f"row_sum_skipna({fields}, min_count=2)"


@pytest.mark.parametrize("count", [23, 27])
def test_declared_variadic_operator_uses_variadic_input_budget(count):
    expr = DSLParser(surface="compat_research").parse(_row_sum_formula(count))
    assert expr.op == "row_sum_skipna"
    assert len(expr.args) == count
    assert dict(expr.kwargs)["min_count"] == 2


def test_fixed_arity_operator_does_not_inherit_variadic_budget():
    fields = ", ".join(f"f{i}" for i in range(13))
    with pytest.raises(DSLParseError, match="max_call_arity"):
        DSLParser(surface="compat_research").parse(f"ts_mean({fields})")


def test_unknown_operator_remains_rejected_before_arity_relaxation():
    fields = ", ".join(f"f{i}" for i in range(23))
    with pytest.raises(DSLUnknownOperatorError, match="not_registered"):
        DSLParser(surface="compat_research").parse(f"not_registered({fields})")
