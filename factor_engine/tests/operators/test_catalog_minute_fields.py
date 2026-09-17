"""Bare minute inputs bind the existing adjusted contract, not an untyped column."""
import pytest
from factor_engine.api.columns import field
from factor_engine.api.source_ref import decode_source_ref
from factor_engine.fields.resolver import resolve_market_field

@pytest.mark.parametrize("stem,physical", [
    ("open", "AdjOpen"), ("high", "AdjHigh"), ("low", "AdjLow"),
    ("close", "AdjClose"), ("volume", "Volume"),
    ("amount", "AdjAmount"), ("vwap", "AdjVwap"),
])
@pytest.mark.parametrize("prefix", ["minute_", "m_"])
def test_bare_minute_adjusted_source_identity(stem, physical, prefix):
    ref = field(prefix + stem)
    assert ref.table == "StockMinuteBarAdj"
    assert ref.source_name == physical
    source = decode_source_ref(ref.name)
    assert source.table == "StockMinuteBarAdj" and source.field == physical
    assert resolve_market_field(ref, "ashare", strict=True).spec.source_name == physical

def test_explicit_raw_and_daily_authority_unchanged():
    assert field("minute_close", table="StockMinuteBar").table == "StockMinuteBar"
    assert field("close").table == "StockDailyBarAdj"

def test_unrelated_ambiguity_is_not_invented():
    with pytest.raises(KeyError):
        field("pub_date")
