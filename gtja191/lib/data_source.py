"""GTJA-191 与 factor_engine / data_access 对齐的数据源配置。"""
from __future__ import annotations

from typing import Any

# A 股价量主表：ashare_stock_daily（data_access 登记名）
ASHARE_PV_FIELDS: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vwap": "Vwap",
    "amount": "Amount",
    "ret": "Return",
    "preclose": "PreClose",
}

# GTJA 公式中 index_close / index_open 对应 ashare_index_daily（需单独 composite ETL）
ASHARE_INDEX_FIELDS: dict[str, str] = {
    "index_close": "Close",
    "index_open": "Open",
}


def default_ashare_pv_data_source(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    read_auto: bool = True,
) -> dict[str, Any]:
    """A 股价量 data_access 配置（与 factor_engine mining_integration 一致）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "ashare_stock_daily",
        "fields": dict(ASHARE_PV_FIELDS),
        "read_auto": read_auto,
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def campaign_data_source() -> dict[str, Any]:
    """disk.v1 campaign config.json 中的 data_source 块。"""
    return default_ashare_pv_data_source(read_auto=True)


def smoke_data_source(
    *,
    start_date: str = "2016-01-04",
    end_date: str = "2016-01-10",
) -> dict[str, Any]:
    """本地 smoke：read_auto 读数加速。"""
    return default_ashare_pv_data_source(
        start_date=start_date,
        end_date=end_date,
        read_auto=True,
    )
