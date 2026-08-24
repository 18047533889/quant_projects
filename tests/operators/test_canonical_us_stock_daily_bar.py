"""回归：US StockDailyBar 价量列须为 primary_signal（非 forbidden）。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANON = ROOT / "factor-pool-standard" / "enums" / "canonical_data_fields.json"
if not CANON.is_file():
    CANON = Path(__file__).resolve().parents[2] / "factor_engine" / "docs" / "canonical_data_fields.json"


def test_us_stock_daily_bar_ohlcv_primary_signal():
    data = json.loads(CANON.read_text(encoding="utf-8"))
    cols = data["markets"]["us_stock"]["tables_local"]["StockDailyBar"]["columns"]
    for phys in ("Open", "High", "Low", "Close", "Volume"):
        col = cols[phys]
        assert col["formula_usage"] == "primary_signal", phys
        assert col.get("signal_domain") == "price_volume", phys
        assert col.get("formula_us_dsl") == col["canonical"], phys
        assert "lqtp_alias_expands_to" not in col, phys
