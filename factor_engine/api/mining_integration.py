"""挖掘框架 / 投递校验与 factor_engine 的对接入口。

能力
----
- ``validate_factor_engine_dsl(formula)`` / ``validate_us_dsl``：校验公式能否被 ``parse_expr`` 解析；
- ``validate_manifest_for_execution``：按 ``market`` + ``expression_type`` 决定是否做 DSL 校验；
- ``list_dsl_allowlist()``、``scripts/validate_delivery_formula.py``：投递前 CLI。

执行策略（A 股 / 美股统一）
----------------------------
**A 股与美股均走 factor_engine runtime**（``cleaned_operators`` + ``PandasBackend``）；
仅数据源与 canonical 字段不同（A 股 parquet 常在 ``data/a_share/lqtp_data/``）。
``lqtp_dsl`` / ``platforms.lqtp`` 为历史 LQTP 平台路径，**新 campaign 不再依赖**。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.dsl_parser import DSLParseError, parse_expr
from api.operator_registry import build_dsl_allowlist

# composite 估值源：允许 DSL 使用 col("pe") 等无前缀简写
_VALUATION_FIELD_ALIASES: dict[str, str] = {
    "pe": "valuation.pe",
    "pb": "valuation.pb",
    "turnover_ratio": "valuation.turnover_ratio",
    "market_cap": "valuation.market_cap",
    "circulating_market_cap": "valuation.circulating_market_cap",
}


def validate_production_dsl(formula: str) -> tuple[bool, str]:
    """production 模式公式校验：语法 + 算子 production 允许。"""
    from cleaned_operators.operator_spec import check_production_formula_ops

    ok, msg = validate_factor_engine_dsl(formula)
    if not ok:
        return False, msg
    violations = check_production_formula_ops(formula)
    if violations:
        return False, "; ".join(violations)
    return True, "OK"


def validate_factor_engine_dsl(formula: str) -> tuple[bool, str]:
    """校验 factor_engine DSL 语法与白名单（A 股 / 美股同一套 ``parse_expr``）。"""
    text = str(formula or "").strip()
    if not text:
        return False, "empty formula"
    try:
        parse_expr(text)
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)


def validate_us_dsl(formula: str) -> tuple[bool, str]:
    """兼容旧名；同 :func:`validate_factor_engine_dsl`。"""
    return validate_factor_engine_dsl(formula)


def list_dsl_allowlist() -> list[str]:
    return sorted(build_dsl_allowlist().keys())


def export_dsl_allowlist_json(*, market: str | None = None) -> dict[str, Any]:
    """导出 AFV Gateway 对齐用的算子白名单 JSON。"""
    names = list_dsl_allowlist()
    mkt = str(market or "us").strip().lower()
    policy_by_market = {
        "ashare": "lqtp_pv_daily",
        "a_share": "lqtp_pv_daily",
        "cn": "lqtp_pv_daily",
        "china": "lqtp_pv_daily",
        "us": "afv_us_pv_daily",
        "usa": "afv_us_pv_daily",
    }
    return {
        "schema_version": "factor_engine.dsl_allowlist.v1",
        "operator_policy": policy_by_market.get(mkt, "afv_us_pv_daily"),
        "market": mkt,
        "operators": names,
        "count": len(names),
    }


def default_ashare_pv_data_source_config(
    *,
    max_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """A 股价量数据源配置。

    默认走 ``data_access`` 登记数据集 ``ashare_stock_daily``；
    ``max_files`` 限文件时回退直连 parquet（smoke test）。
    """
    fields = {
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
    if max_files is not None:
        cfg: dict[str, Any] = {
            "type": "parquet",
            "root": "~/quant_projects/data/a_share/lqtp_data/StockDailyBar",
            "timestamp_col": "TradeDate",
            "instrument_col": "Symbol",
            "fields": fields,
            "max_files": max_files,
        }
    else:
        cfg = {
            "type": "data_access",
            "dataset": "ashare_stock_daily",
            "fields": fields,
        }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_ashare_pv_valuation_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """A 股日线 + 估值复合数据源（锚点日线，估值 asof 对齐）。"""
    pv = default_ashare_pv_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    valuation: dict[str, Any] = {
        "type": "data_access",
        "dataset": "ashare_stock_valuation_daily",
        "fields": {
            "pe": "PeRatio",
            "pb": "PbRatio",
            "turnover_ratio": "TurnoverRatio",
            "market_cap": "MarketCap",
            "circulating_market_cap": "CirculatingMarketCap",
        },
    }
    if start_date is not None:
        valuation["start_date"] = start_date
    if end_date is not None:
        valuation["end_date"] = end_date

    return {
        "type": "composite",
        "anchor": "pv",
        "anchor_column": "close",
        "sources": {
            "pv": pv,
            "valuation": valuation,
        },
        "joins": {
            "valuation": "asof_backward",
        },
        "aliases": dict(_VALUATION_FIELD_ALIASES),
    }


def ashare_dataset_registry() -> dict[str, str]:
    """已登记的 A 股 lqtp 数据集名 → COS 子目录名。"""
    from data_access.cos_mirror import ASHARE_DATASET_TABLE_MAP

    return dict(ASHARE_DATASET_TABLE_MAP)


def us_dataset_registry() -> dict[str, str]:
    """已登记的美股数据集名 → COS 子目录名（massive_data 部分）。"""
    from data_access.cos_mirror import DATASET_MIRROR_REGISTRY

    return {
        name: spec.table
        for name, spec in DATASET_MIRROR_REGISTRY.items()
        if name.startswith("us_") and spec.table
    }


def default_us_pv_data_source_config(
    *,
    max_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股价量数据源配置（``data_access`` 数据集 ``us_stock_daily``）。"""
    fields = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "VWAP",
        "amount": "Amount",
        "ret": "Ret",
        "preclose": "PreClose",
        "adj_factor": "AdjFactor",
    }
    if max_files is not None:
        cfg: dict[str, Any] = {
            "type": "parquet",
            "root": "~/quant_projects/data/us_stock/massive_data/StockDailyBar",
            "timestamp_col": "TradeDate",
            "instrument_col": "Ticker",
            "fields": fields,
            "max_files": max_files,
        }
    else:
        cfg = {
            "type": "data_access",
            "dataset": "us_stock_daily",
            "fields": fields,
        }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_pv_valuation_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股日线 + 估值复合数据源（锚点日线，估值 asof 对齐）。"""
    pv = default_us_pv_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    valuation: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stock_valuation_daily",
        "fields": {
            "pe": "PeRatio",
            "pb": "PbRatio",
            "turnover_ratio": "TurnoverRatio",
            "market_cap": "MarketCap",
            "circulating_market_cap": "CirculatingMarketCap",
        },
    }
    if start_date is not None:
        valuation["start_date"] = start_date
    if end_date is not None:
        valuation["end_date"] = end_date

    return {
        "type": "composite",
        "anchor": "pv",
        "anchor_column": "close",
        "sources": {
            "pv": pv,
            "valuation": valuation,
        },
        "joins": {
            "valuation": "asof_backward",
        },
        "aliases": dict(_VALUATION_FIELD_ALIASES),
    }


def default_ashare_pv_universe_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """A 股日线 + 停牌状态 + 指数成分复合数据源（PiT 生产模板）。

    - ``status``：``ashare_stock_status``，asof 对齐停牌/上市状态
    - ``constituent``：``ashare_index_constituent``，asof 对齐成分股权重
    """
    pv = default_ashare_pv_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    status: dict[str, Any] = {
        "type": "data_access",
        "dataset": "ashare_stock_status",
        "fields": {
            "listed_state": "ListedState",
        },
    }
    constituent: dict[str, Any] = {
        "type": "data_access",
        "dataset": "ashare_index_constituent",
        "fields": {
            "index_symbol": "IndexSymbol",
            "weight": "Weight",
        },
    }
    if start_date is not None:
        status["start_date"] = start_date
        constituent["start_date"] = start_date
    if end_date is not None:
        status["end_date"] = end_date
        constituent["end_date"] = end_date

    return {
        "type": "composite",
        "anchor": "pv",
        "anchor_column": "close",
        "sources": {
            "pv": pv,
            "status": status,
            "constituent": constituent,
        },
        "joins": {
            "status": "asof_backward",
            "constituent": "asof_backward",
        },
    }


def default_us_pv_universe_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股日线 + 日度 universe 复合数据源（锚点日线，universe exact 对齐）。"""
    pv = default_us_pv_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    universe: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_universe_daily",
        "fields": {
            "universe_ticker": "ticker",
        },
    }
    if start_date is not None:
        universe["start_date"] = start_date
    if end_date is not None:
        universe["end_date"] = end_date

    return {
        "type": "composite",
        "anchor": "pv",
        "anchor_column": "close",
        "sources": {
            "pv": pv,
            "universe": universe,
        },
        "joins": {
            "universe": "exact",
        },
    }


def default_us_daily_market_summary_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股 Polygon 原始日线聚合（``daily_market_summary``，PR5 登记）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "daily_market_summary",
        "fields": {
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
            "vwap": "vw",
            "ticker": "T",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_stocks_floats_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股流通股快照（``stocks_floats``，PR5 登记）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "stocks_floats",
        "fields": {
            "free_float": "free_float",
            "free_float_percent": "free_float_percent",
            "ticker": "ticker",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_stocks_sip_day_aggs_data_source_config(
    *,
    max_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned SIP 日 K（``us_stocks_sip_day_aggs``，align_time + ticker）。"""
    fields = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "window_start": "window_start",
    }
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stocks_sip_day_aggs",
        "fields": fields,
    }
    if max_files is not None and start_date is None:
        start_date = "2024-01-01"
        end_date = end_date or "2024-01-31"
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_financials_ratios_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned 财务比率（``financials_ratios``，date + ticker）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "financials_ratios",
        "fields": {
            "pe": "price_to_earnings",
            "price_to_earnings": "price_to_earnings",
            "return_on_equity": "return_on_equity",
            "roe": "return_on_equity",
            "debt_to_equity": "debt_to_equity",
            "price_to_book": "price_to_book",
            "free_cash_flow": "free_cash_flow",
            "market_cap": "market_cap",
            "price_to_sales": "price_to_sales",
            "enterprise_value": "enterprise_value",
            "earnings_per_share": "earnings_per_share",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_stocks_sip_minute_aggs_data_source_config(
    *,
    max_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned SIP 分钟 K（``us_stocks_sip_minute_aggs``，window_start ns）。"""
    fields = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "transactions": "transactions",
        "window_start": "window_start",
    }
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stocks_sip_minute_aggs",
        "fields": fields,
        "normalize_timestamp": True,
        "timestamp_unit": "ns",
    }
    if max_files is not None and start_date is None:
        start_date = "2024-01-01"
        end_date = end_date or "2024-01-07"
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_stocks_sip_quotes_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned SIP 报价（``us_stocks_sip_quotes``，sip_timestamp ns）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stocks_sip_quotes",
        "fields": {
            "bid_price": "bid_price",
            "ask_price": "ask_price",
            "bid_size": "bid_size",
            "ask_size": "ask_size",
        },
        "normalize_timestamp": True,
        "timestamp_unit": "ns",
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_stocks_sip_trades_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned SIP 逐笔成交（``us_stocks_sip_trades``，sip_timestamp ns）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stocks_sip_trades",
        "fields": {
            "price": "price",
            "size": "size",
        },
        "normalize_timestamp": True,
        "timestamp_unit": "ns",
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_massive_ticks_data_source_config(
    *,
    kind: str = "trades_v1",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """原始 massive ticks（``massive_ticks`` 参数化，``kind=quotes_v1|trades_v1``）。"""
    fields = (
        {
            "bid_price": "bid_price",
            "ask_price": "ask_price",
        }
        if kind == "quotes_v1"
        else {
            "price": "price",
            "size": "size",
        }
    )
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "massive_ticks",
        "kind": kind,
        "fields": fields,
        "normalize_timestamp": True,
        "timestamp_unit": "ns",
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_fundamentals_balance_sheet_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned 资产负债表（``fundamentals_balance_sheet``）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "fundamentals_balance_sheet",
        "fields": {
            "total_assets": "total_assets",
            "total_equity": "total_equity",
            "total_liabilities": "total_liabilities",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_fundamentals_cash_flow_statement_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned 现金流量表（``fundamentals_cash_flow_statement``）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "fundamentals_cash_flow_statement",
        "fields": {
            "operating_cf": "net_cash_from_operating_activities",
            "investing_cf": "net_cash_from_investing_activities",
            "net_cash_from_operating_activities": "net_cash_from_operating_activities",
            "net_cash_from_investing_activities": "net_cash_from_investing_activities",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_fundamentals_income_statement_data_source_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Massive cleaned 利润表（``fundamentals_income_statement``）。"""
    cfg: dict[str, Any] = {
        "type": "data_access",
        "dataset": "fundamentals_income_statement",
        "fields": {
            "revenue": "revenue",
            "operating_income": "operating_income",
            "net_income": "net_income",
        },
    }
    if start_date is not None:
        cfg["start_date"] = start_date
    if end_date is not None:
        cfg["end_date"] = end_date
    return cfg


def default_us_sip_day_ratios_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """SIP 日 K + 财务比率 asof 复合源（价量 anchor）。"""
    price = default_us_stocks_sip_day_aggs_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    ratios = default_financials_ratios_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "type": "composite",
        "anchor": "price",
        "anchor_column": "close",
        "sources": {"price": price, "ratios": ratios},
        "joins": {"ratios": "asof_backward"},
        "aliases": {
            "pe": "ratios.price_to_earnings",
            "price_to_earnings": "ratios.price_to_earnings",
            "roe": "ratios.return_on_equity",
        },
    }


def default_us_sip_balance_sheet_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """SIP 日 K + 资产负债表 asof 复合源。"""
    price = default_us_stocks_sip_day_aggs_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    balance_sheet = default_fundamentals_balance_sheet_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "type": "composite",
        "anchor": "price",
        "anchor_column": "close",
        "sources": {"price": price, "balance_sheet": balance_sheet},
        "joins": {"balance_sheet": "asof_backward"},
        "aliases": {
            "total_assets": "balance_sheet.total_assets",
            "total_equity": "balance_sheet.total_equity",
            "total_liabilities": "balance_sheet.total_liabilities",
        },
    }


def default_us_sip_cash_flow_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """SIP 日 K + 现金流量表 asof 复合源。"""
    price = default_us_stocks_sip_day_aggs_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    cash_flow = default_fundamentals_cash_flow_statement_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "type": "composite",
        "anchor": "price",
        "anchor_column": "close",
        "sources": {"price": price, "cash_flow": cash_flow},
        "joins": {"cash_flow": "asof_backward"},
        "aliases": {
            "operating_cf": "cash_flow.net_cash_from_operating_activities",
            "investing_cf": "cash_flow.net_cash_from_investing_activities",
        },
    }


def default_us_sip_income_statement_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """SIP 日 K + 利润表 asof 复合源。"""
    price = default_us_stocks_sip_day_aggs_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    income_statement = default_fundamentals_income_statement_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "type": "composite",
        "anchor": "price",
        "anchor_column": "close",
        "sources": {"price": price, "income_statement": income_statement},
        "joins": {"income_statement": "asof_backward"},
        "aliases": {
            "revenue": "income_statement.revenue",
            "operating_income": "income_statement.operating_income",
            "net_income": "income_statement.net_income",
        },
    }


def default_us_sip_floats_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """SIP 日 K + 流通股快照 asof 复合源。"""
    price = default_us_stocks_sip_day_aggs_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    floats = default_us_stocks_floats_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "type": "composite",
        "anchor": "price",
        "anchor_column": "close",
        "sources": {"price": price, "floats": floats},
        "joins": {"floats": "asof_backward"},
        "aliases": {
            "free_float": "floats.free_float",
            "free_float_percent": "floats.free_float_percent",
        },
    }


def default_us_polygon_floats_composite_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """美股 Polygon 日线 + 流通股快照复合源（价量 anchor，floats asof）。"""
    pv = default_us_daily_market_summary_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    floats: dict[str, Any] = {
        "type": "data_access",
        "dataset": "stocks_floats",
        "fields": {
            "free_float": "free_float",
            "free_float_percent": "free_float_percent",
        },
    }
    if start_date is not None:
        floats["start_date"] = start_date
    if end_date is not None:
        floats["end_date"] = end_date
    return {
        "type": "composite",
        "anchor": "pv",
        "anchor_column": "close",
        "sources": {"pv": pv, "floats": floats},
        "joins": {"floats": "asof_backward"},
        "aliases": {
            "free_float": "floats.free_float",
            "free_float_percent": "floats.free_float_percent",
        },
    }


def validate_manifest_for_execution(
    *,
    market: str,
    expression_type: str | None,
    formula: str,
) -> tuple[bool, str]:
    """按 market / expression_type 决定是否做 factor_engine 语法校验。"""
    mkt = str(market or "").strip()
    et = str(expression_type or "dsl").strip() or "dsl"
    if et == "python":
        return True, "skip: python intermediate; translate to dsl before delivery"
    if et in ("dsl", "lqtp_dsl", ""):
        return validate_factor_engine_dsl(formula)
    return False, f"unsupported expression_type: {et!r}"


def write_dsl_allowlist(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(export_dsl_allowlist_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def default_mining_label_config(
    *,
    horizon_bars: int = 5,
    feature_lookback_bars: int = 20,
    gap_bars: int = 1,
) -> dict[str, Any]:
    """挖掘/回测默认标签配置（PIT：标签与特征窗口隔离）。"""
    from api.label_pit import default_mining_label_config as _label_cfg

    return _label_cfg(
        horizon_bars=horizon_bars,
        feature_lookback_bars=feature_lookback_bars,
        gap_bars=gap_bars,
    )


def validate_mining_label_formula(formula: str, *, enforce: bool = True) -> tuple[bool, str]:
    """校验标签 DSL 不含前视算子。"""
    from api.label_pit import validate_label_formula_for_pit

    try:
        report = validate_label_formula_for_pit(formula, enforce=enforce)
        if report["ok"]:
            return True, "OK"
        return False, "; ".join(report.get("violations") or [])
    except Exception as exc:
        return False, str(exc)


def audit_composite_join_policies(config: dict[str, Any]) -> list[str]:
    """审计 composite 配置：非 anchor 源应使用 asof_backward（PiT 安全）。"""
    violations: list[str] = []
    if str(config.get("type", "")).lower() != "composite":
        return violations
    joins = config.get("joins") or {}
    anchor = str(config.get("anchor") or config.get("anchor_source") or "pv")
    for name, mode in joins.items():
        if str(name) == anchor:
            continue
        if str(mode).lower() not in {"asof_backward", "backward", "asof"}:
            violations.append(f"{name}: join={mode!r} (expected asof_backward)")
    for name, sub in (config.get("sources") or {}).items():
        if isinstance(sub, dict):
            violations.extend(audit_composite_join_policies(sub))
    return violations


def audit_default_data_source_configs() -> dict[str, list[str]]:
    """审计所有默认 mining 数据源 preset 的 join 策略。"""
    presets = default_mining_data_source_presets()
    composite_names = {
        "ashare_pv_valuation",
        "us_pv_valuation",
        "ashare_pv_universe",
        "us_pv_universe",
        "us_polygon_floats",
        "us_sip_day_ratios",
        "us_sip_balance_sheet",
        "us_sip_cash_flow",
        "us_sip_income_statement",
        "us_sip_floats",
    }
    return {
        name: audit_composite_join_policies(presets[name])
        for name in composite_names
        if name in presets
    }


def default_mining_data_source_presets() -> dict[str, dict[str, Any]]:
    """全部 mining 默认 ``data_source`` preset（契约审计 / CI 单点注册）。"""
    return {
        "ashare_pv": default_ashare_pv_data_source_config(),
        "ashare_pv_valuation": default_ashare_pv_valuation_data_source_config(),
        "ashare_pv_universe": default_ashare_pv_universe_data_source_config(),
        "us_pv": default_us_pv_data_source_config(),
        "us_pv_valuation": default_us_pv_valuation_data_source_config(),
        "us_pv_universe": default_us_pv_universe_data_source_config(),
        "us_daily_market_summary": default_us_daily_market_summary_data_source_config(),
        "us_stocks_floats": default_us_stocks_floats_data_source_config(),
        "us_stocks_sip_day_aggs": default_us_stocks_sip_day_aggs_data_source_config(),
        "financials_ratios": default_financials_ratios_data_source_config(),
        "us_sip_day_ratios": default_us_sip_day_ratios_composite_config(),
        "us_sip_balance_sheet": default_us_sip_balance_sheet_composite_config(),
        "us_sip_cash_flow": default_us_sip_cash_flow_composite_config(),
        "us_sip_income_statement": default_us_sip_income_statement_composite_config(),
        "us_sip_floats": default_us_sip_floats_composite_config(),
        "us_stocks_sip_minute_aggs": default_us_stocks_sip_minute_aggs_data_source_config(),
        "us_polygon_floats": default_us_polygon_floats_composite_config(),
        "us_stocks_sip_quotes": default_us_stocks_sip_quotes_data_source_config(),
        "us_stocks_sip_trades": default_us_stocks_sip_trades_data_source_config(),
        "massive_ticks_quotes": default_massive_ticks_data_source_config(kind="quotes_v1"),
        "massive_ticks_trades": default_massive_ticks_data_source_config(kind="trades_v1"),
        "fundamentals_balance_sheet": default_fundamentals_balance_sheet_data_source_config(),
        "fundamentals_cash_flow_statement": default_fundamentals_cash_flow_statement_data_source_config(),
        "fundamentals_income_statement": default_fundamentals_income_statement_data_source_config(),
    }


def audit_mining_datasets_contract() -> dict[str, Any]:
    """审计 mining preset 引用的 ``data_access`` 数据集是否在 ``datasets.yaml`` 登记。"""
    from api.datasets_contract import audit_mining_dataset_contract

    return audit_mining_dataset_contract()
