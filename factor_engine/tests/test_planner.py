from api.columns import col
from api import rank, ts_mean
from ir.analyzer import Analyzer
from planner.lowerer import Lowerer


def test_analyzer_and_lowerer():
    expr = rank(ts_mean(col("close"), 10))
    analysis = Analyzer().lower(expr)
    plan = Lowerer().to_logical_plan(analysis.ir)

    assert analysis.has_ts_op is True
    assert analysis.has_cs_op is True
    assert analysis.lookback == 10
    assert analysis.referenced_columns == {"close"}
    assert plan.op == "rank"
    assert plan.inputs[0].op == "ts_mean"


def test_analyzer_lookback_includes_operator_policy_lag():
    from api import ts_delay

    analysis = Analyzer().lower(ts_delay(col("close")))
    assert analysis.lookback >= 1
