import pytest

from api.dsl_parser import DSLParseError, parse_expr, parse_factor
from expr.cleaned_call import CleanedCall


def test_parse_expr_builds_cleaned_call_tree():
    expr = parse_expr('rank(ts_mean(col("close"), 3) - delay(col("close"), 1))')
    assert isinstance(expr, CleanedCall)
    assert expr.op == "rank"


def test_parse_expr_rejects_unsafe_names():
    with pytest.raises(DSLParseError):
        parse_expr('__import__("os").system("echo unsafe")')


def test_parse_factor_wraps_expr():
    factor = parse_factor('rank(ts_mean(col("close"), 3))', name="demo")
    assert factor.name == "demo"


def test_parse_expr_cleaned_operators():
    parse_expr('group_rank(col("x"), col("g"))')
    parse_expr('neutralize(col("x"), col("y"))')
    parse_expr('SMA(col("close"), 5)')
    parse_expr('sin(col("x"))', surface="extended")
    parse_expr('cos(col("x"))', surface="extended")
    parse_expr('exp(col("x"))')
    parse_expr('ts_macd(col("c"), line="hist")', surface="compat")


def test_parse_expr_bare_field_names():
    """挖掘侧标准：裸字段名自动视为列引用。"""
    bare = parse_expr("rank(ts_mean(close, 3) - delay(close, 1))")
    explicit = parse_expr('rank(ts_mean(col("close"), 3) - delay(col("close"), 1))')
    assert isinstance(bare, CleanedCall)
    assert bare.op == explicit.op == "rank"


def test_parse_expr_bare_field_arithmetic():
    parse_expr("close - open")
    parse_expr("close / (volume + 1e-9)")


def test_parse_expr_rejects_bare_operator_name():
    with pytest.raises(DSLParseError, match="not a column reference"):
        parse_expr("rank")


def test_parse_expr_comparison_ops():
    parse_expr("close > open")
    parse_expr('col("x") > col("y")')
    parse_expr('col("x") <= 1.0')
