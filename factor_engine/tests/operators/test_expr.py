from api import rank, ts_mean
from api.columns import col
from expr.base import ensure_expr
from expr.cleaned_call import CleanedCall


def test_binop_builds_cleaned_call():
    expr = col("close") - col("open")
    assert isinstance(expr, CleanedCall)
    assert expr.op == "subtract"


def test_rank_ts_mean_chain():
    expr = rank(ts_mean(col("close"), 3))
    assert isinstance(expr, CleanedCall)
    assert expr.op == "rank"
    assert isinstance(expr.args[0], CleanedCall)
    assert expr.args[0].op == "ts_mean"


def test_ensure_expr_wraps_scalar():
    lit = ensure_expr(3)
    from expr.literal import Literal

    assert isinstance(lit, Literal)
    assert lit.value == 3
