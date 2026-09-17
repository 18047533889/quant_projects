from factor_engine.api.dsl_parser import DSLParser
from factor_engine.ir.analyzer import Analyzer


def _lower(source: str):
    return Analyzer().lower(DSLParser(surface="compat_research").parse(source)).ir


def test_qualified_adjusted_return_source_ref_is_return_decimal():
    node = _lower('field("ret", table="StockDailyBarAdj")')
    assert node.attrs["field_id"] == "StockDailyBarAdj.ret"
    assert node.attrs["price_basis"] == "RETURN"
    assert node.semantic_attrs["semantic_kind"] == "ReturnDecimal"


def test_adjusted_price_field_keeps_continuous_price_semantics():
    node = _lower('field("close", table="StockDailyBarAdj")')
    assert node.attrs["field_id"] == "StockDailyBarAdj.close"
    assert node.semantic_attrs["semantic_kind"] == "PriceContinuous"
