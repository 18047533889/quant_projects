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

# disk.v1 campaign config 用的本地 / COS 路径提示（AFV 不执行；与 FE §2.8 一致）
CAMPAIGN_LOCAL_ROOT = "data/a_share/lqtp_data/"
CAMPAIGN_COS_ROOT = "cos://qs-cold/clean_data/ashare/lqtp_data/"


def default_ashare_pv_data_source(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    read_auto: bool = True,
) -> dict[str, Any]:
    """A 股价量 **执行层** data_access 配置（与 factor_engine mining_integration 一致）。

    用于 materialize YAML / smoke / FactorEngine.run，**不是** campaign config.json 的 data_source。
    """
    try:
        from api.mining_integration import default_ashare_pv_data_source_config

        cfg = default_ashare_pv_data_source_config(
            start_date=start_date,
            end_date=end_date,
        )
        # mining_integration 默认未强制 read_auto；GTJA 需要 Arrow 加速
        cfg["read_auto"] = read_auto
        if "fields" not in cfg:
            cfg["fields"] = dict(ASHARE_PV_FIELDS)
        return cfg
    except Exception:
        pass

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
    """disk.v1 campaign ``config.json`` 的 ``data_source``（local/cos 路径提示）。

    与 week2_pv_factors / factor_engine ``miner_delivery_spec`` §2.8 对齐。
    真正算因子时用 :func:`default_ashare_pv_data_source` / materialize YAML。
    """
    return {
        "local": CAMPAIGN_LOCAL_ROOT,
        "cos": CAMPAIGN_COS_ROOT,
    }


def production_date_range(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, str]:
    """与 campaign mining_config 对齐的全量落值区间。"""
    return start_date or "2014-01-01", end_date or "2026-06-25"


def production_data_source(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    read_auto: bool = True,
) -> dict[str, Any]:
    """全量落值 data_access 配置。"""
    start, end = production_date_range(start_date=start_date, end_date=end_date)
    return default_ashare_pv_data_source(
        start_date=start,
        end_date=end,
        read_auto=read_auto,
    )


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
