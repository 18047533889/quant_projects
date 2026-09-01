"""A 股冷启动常用字段约定。"""

from __future__ import annotations

PV_CORE_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "amount",
    "ret",
    "pre_close",
    "factor",
)

VALUATION_FIELDS: tuple[str, ...] = (
    "pe",
    "pb",
    "turnover_ratio",
    "market_cap",
    "circulating_market_cap",
)

# AlphaPROBE / mining_integration 默认可加载（ashare_pv_valuation）
ASHARE_DATASOURCE_FIELDS: tuple[str, ...] = PV_CORE_FIELDS + ("preclose",) + VALUATION_FIELDS
