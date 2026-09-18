"""回归：US StockDailyBar 价量列须为 primary_signal（非 forbidden）。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANON = ROOT / "factor-pool-standard" / "enums" / "canonical_data_fields.json"
if not CANON.is_file():
    CANON = Path(__file__).resolve().parents[2] / "docs" / "canonical_data_fields.json"


def test_us_stock_daily_bar_ohlcv_primary_signal():
    data = json.loads(CANON.read_text(encoding="utf-8"))
    market = data["markets"]["us_stock"]
    table = market["tables_massive_external"]["us_stocks_sip/day_aggs_v1"]
    assert table["logical_table"] == "StockDailyBar"
    assert table["maps_to_local"] == "StockDailyBar"
    cols = table["columns"]
    for canonical, wire in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
        col = cols[canonical]
        assert col["formula_usage"] == "primary_signal", canonical
        assert col["signal_domain"] == "price_volume", canonical
        assert col["canonical"] == canonical
        assert col["physical"] == canonical
        assert table["sip_to_physical"][wire] == canonical.capitalize()
        assert "lqtp_alias_expands_to" not in col, canonical
