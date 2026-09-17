import pytest

from factor_engine.api.dsl_parser import DSLParser
from factor_engine.ir.analyzer import Analyzer
from factor_engine.planner.composite_lowering import lower_composite_operators
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerer import Lowerer


@pytest.mark.parametrize(
    "formula",
    [
        "limit_up_close(field('close', table='StockDailyBarAdj'), "
        "field('high_limit', table='StockDailyBarAdj'))",
        "limit_down_close(field('close', table='StockDailyBarAdj'), "
        "field('low_limit', table='StockDailyBarAdj'))",
    ],
)
def test_limit_close_analyzer_and_lowering_preserve_formal_event_semantics(formula):
    ir = Analyzer().lower(DSLParser(surface="compat_research").parse(formula)).ir
    assert ir.semantic_attrs["semantic_kind"] == "EventBool"
    before = Lowerer().to_logical_plan(ir)
    after = lower_composite_operators(before)
    assert after.op in {"ge", "le"}
    assert after.semantic_attrs == before.semantic_attrs
    assert after.semantic_attrs["semantic_kind"] == "EventBool"
