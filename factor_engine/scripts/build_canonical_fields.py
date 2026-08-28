#!/usr/bin/env python3
"""从本地 parquet schema + Massive 数据字典生成 canonical_data_fields.json。

用法（在 quant_projects 根目录或 factor_engine 下）：
  python factor_engine/scripts/build_canonical_fields.py
  python factor_engine/scripts/build_canonical_fields.py --check   # 仅校验现有 JSON 是否与扫描一致
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # quant_projects
FE = ROOT / "factor_engine"
OUT_JSON = FE / "docs" / "canonical_data_fields.json"
SCHEMA_SCAN = FE / "docs" / "_generated_schema_scan.json"
MASSIVE_DICT = FE / "docs" / "massive_parquet_data_dictionary.json"

# --- 显式 canonical 映射（价量核心列；优先于自动 snake_case）---
US_PV_CANONICAL: dict[str, dict[str, str]] = {
    "StockDailyBar": {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "AdjFactor": "adj_factor",
    },
    "FactReturnsDaily": {
        "RetPrice": "ret_price",
        "RetTotal": "ret_total",
    },
}

ASHARE_PV_CANONICAL: dict[str, dict[str, str]] = {
    # ADJ_FIELD_MIGRATION（2026-08-28）：A 股行情权威口径 = 后复权表
    # StockDailyBarAdj（Adj* 字段集）。未复权表列仅保留 Volume/Factor（raw 概念）。
    "StockDailyBarAdj": {
        "AdjOpen": "open",
        "AdjHigh": "high",
        "AdjLow": "low",
        "AdjClose": "close",
        "AdjPreClose": "pre_close",
        "Volume": "volume",
        "AdjAmount": "amount",
        "AdjHighLimit": "high_limit",
        "AdjLowLimit": "low_limit",
        "Return": "ret",
        "Factor": "factor",
        "AdjVwap": "vwap",
        "IsSuspend": "is_suspend",
    },
    "StockCapitalDaily": {
        "TotalCapital": "total_capital",
        "CirculatingCapital": "circulating_capital",
    },
    "StockValuationDaily": {
        "PeRatio": "pe_ratio",
        "PbRatio": "pb_ratio",
        "PsRatio": "ps_ratio",
        "MarketCap": "market_cap",
        "CirculatingMarketCap": "circulating_market_cap",
        "TurnoverRatio": "turnover_ratio",
    },
}
# 未复权原始列（Factor/Volume 用途）
ASHARE_PV_CANONICAL_RAW: dict[str, dict[str, str]] = {
    "StockDailyBar": {
        "Open": "raw_open",
        "High": "raw_high",
        "Low": "raw_low",
        "Close": "raw_close",
        "PreClose": "raw_pre_close",
        "Volume": "volume",
        "Amount": "raw_amount",
        "HighLimit": "raw_high_limit",
        "LowLimit": "raw_low_limit",
        "Return": "ret",
        "Factor": "factor",
        "Vwap": "raw_vwap",
        "IsSuspend": "is_suspend",
    },
}

# LQTP 别名（来自 field_aliases.py 摘要）
LQTP_ALIASES: dict[str, str] = {
    "open": "StockDailyBar.Open * StockDailyBar.Factor",
    "high": "StockDailyBar.High * StockDailyBar.Factor",
    "low": "StockDailyBar.Low * StockDailyBar.Factor",
    "close": "StockDailyBar.Close * StockDailyBar.Factor",
    "pre_close": "StockDailyBar.PreClose * StockDailyBar.Factor",
    "volume": "StockDailyBar.Volume / StockDailyBar.Factor",
    "amount": "StockDailyBar.Amount",
    "ret": "StockDailyBar.Return",
    "vwap": "StockDailyBar.Vwap * StockDailyBar.Factor",
    "is_suspend": "StockDailyBar.IsSuspend",
    "high_limit": "StockDailyBar.HighLimit * StockDailyBar.Factor",
    "low_limit": "StockDailyBar.LowLimit * StockDailyBar.Factor",
}

TABLE_META: dict[str, dict[str, Any]] = {
    # us_stock local
    "FactReturnsDaily": {"domain_root": "price_volume", "role": "primary", "market": "us_stock"},
    "PanelDaily": {"domain_root": "price_volume", "role": "primary", "market": "us_stock", "note": "宽表；含 pit_* / ltm_* / news_*"},
    "DimCalendar": {"domain_root": "price_volume", "role": "auxiliary", "market": "us_stock"},
    "DimSecurityMaster": {"domain_root": "price_volume", "role": "auxiliary", "market": "us_stock"},
    "DimTickerMap": {"domain_root": "price_volume", "role": "auxiliary", "market": "us_stock"},
    "DimTickerAlias": {"domain_root": "price_volume", "role": "auxiliary", "market": "us_stock"},
    # ashare local
    "Calendar": {"domain_root": "price_volume", "role": "auxiliary", "market": "ashare"},
    "StockCapitalDaily": {"domain_root": "price_volume", "role": "primary", "market": "ashare"},
    "StockBalance": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockIncome": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockCashFlow": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockIndicator": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockValuationDaily": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockDividend": {"domain_root": "fundamental", "role": "primary", "market": "ashare"},
    "StockTopTenShareholder": {"domain_root": "alternative", "role": "primary", "market": "ashare"},
    "StockTopTenFloatShareholder": {"domain_root": "alternative", "role": "primary", "market": "ashare"},
    "StockList": {"domain_root": "price_volume", "role": "auxiliary", "market": "ashare"},
    "StockStatus": {"domain_root": "price_volume", "role": "auxiliary", "market": "ashare"},
    "StockIndustry": {"domain_root": "price_volume", "role": "auxiliary", "market": "ashare"},
}

# 同名表在不同 market 下元数据不同（避免 dict 重复键覆盖）
TABLE_META_BY_MARKET: dict[str, dict[str, dict[str, Any]]] = {
    "us_stock": {
        "StockDailyBar": {"domain_root": "price_volume", "role": "primary", "market": "us_stock"},
    },
    "ashare": {
        "StockDailyBar": {"domain_root": "price_volume", "role": "primary", "market": "ashare"},
    },
}


def get_table_meta(table: str, market: str | None = None) -> dict[str, Any]:
    """获取逻辑表的 domain_root / role / market 元数据（支持按 market 覆盖）。"""
    meta = dict(TABLE_META.get(table, {}))
    if market:
        meta.update(TABLE_META_BY_MARKET.get(market, {}).get(table, {}))
    return meta

KEY_COLUMNS = frozenset({
    "TradeDate", "Symbol", "Ticker", "PubDate", "ReportPeriodEndDate",
    "UpdateTime", "ChangeDate", "RightRegDate", "ExDividendDate",
})

METADATA_PATTERNS = (
    re.compile(r"^batch_id", re.I),
    re.compile(r"^year_", re.I),
    re.compile(r"^month_", re.I),
    re.compile(r"^year$", re.I),
    re.compile(r"^month$", re.I),
    re.compile(r"^cik$", re.I),
    re.compile(r"^timeframe$", re.I),
    re.compile(r"^filing_date", re.I),
    re.compile(r"^period_end$", re.I),
    re.compile(r"^is_batch", re.I),
    re.compile(r"^is_negative", re.I),
    re.compile(r"^is_zero", re.I),
    re.compile(r"^is_nonstandard", re.I),
    re.compile(r"^is_decum", re.I),
    re.compile(r"join_log", re.I),
    re.compile(r"missing_inputs", re.I),
    re.compile(r"knowledge_ts", re.I),
    re.compile(r"system_insert", re.I),
    re.compile(r"shares_source", re.I),
)

# PanelDaily 宽表：按列前缀/列名拆 signal_domain（表级仍是 primary，列级约束挖掘类型）
PANEL_PV_PHYSICAL = frozenset({"Open", "High", "Low", "Close", "Volume", "AdjFactor"})

DOMAIN_ROOTS = ("price_volume", "fundamental", "alternative", "microstructure")


def pascal_to_snake(name: str) -> str:
    """PascalCase 列名转 snake_case canonical。"""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


ASHARE_SDB_ALIAS_CANONICAL = frozenset(LQTP_ALIASES.keys())


def infer_signal_domain(physical: str, table: str, role: str, market: str) -> str | None:
    """该列作为因子主信号时所属的 domain_root；非 signal 列返回 None。"""
    if role != "signal":
        return None
    if table == "PanelDaily":
        if physical in PANEL_PV_PHYSICAL:
            return "price_volume"
        if physical.startswith("pit_") or physical.startswith("ltm_"):
            return "fundamental"
        if physical.startswith("news_"):
            return "alternative"
        if physical in ("shares_out",):
            return "fundamental"
        return None
    meta = get_table_meta(table, market)
    return meta.get("domain_root")


def infer_formula_usage(role: str, signal_domain: str | None) -> str:
    """根据列角色与 signal_domain 推断公式中的用法约束。"""
    if role == "metadata":
        return "forbidden"
    if role == "key":
        return "forbidden"
    if role == "auxiliary":
        return "auxiliary_only"
    if role == "filter":
        return "filter_only"
    if signal_domain:
        return "primary_signal"
    return "forbidden"


def column_role(physical: str, table: str, market: str) -> str:
    """推断列角色：key / signal / filter / auxiliary / metadata。"""
    if physical in KEY_COLUMNS:
        return "key"
    if any(p.search(physical) for p in METADATA_PATTERNS):
        return "metadata"
    meta = get_table_meta(table, market)
    if meta.get("role") == "auxiliary":
        return "auxiliary"
    if physical in ("IsSuspend", "IsTradingDay", "IsTradeDay", "ComputeMask"):
        return "filter"
    if table == "StockDailyBar" and physical in ("HighLimit", "LowLimit"):
        return "filter"
    return "signal"


def build_column_entry(
    physical: str,
    dtype: str,
    table: str,
    market: str,
    explicit: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """为单列构建 canonical 字段条目（含 role、formula_usage、LQTP 别名等）。"""
    explicit = explicit or {}
    canonical = explicit.get(table, {}).get(physical)
    if not canonical:
        canonical = pascal_to_snake(physical)
    role = column_role(physical, table, market)
    signal_domain = infer_signal_domain(physical, table, role, market)
    formula_usage = infer_formula_usage(role, signal_domain)
    entry: dict[str, Any] = {
        "physical": physical,
        "dtype": dtype,
        "canonical": canonical,
        "role": role,
        "formula_usage": formula_usage,
    }
    if signal_domain:
        entry["signal_domain"] = signal_domain
    if market == "ashare" and table in (
        "StockIncome", "StockBalance", "StockCashFlow", "StockIndicator",
        "StockValuationDaily", "StockDividend", "StockTopTenShareholder",
        "StockTopTenFloatShareholder", "StockCapitalDaily",
    ):
        entry["lqtp_qualified"] = f"{table}.{physical}"
    if market == "ashare" and canonical in LQTP_ALIASES and table == "StockDailyBar":
        entry["lqtp_alias_expands_to"] = LQTP_ALIASES[canonical]
        if canonical in ASHARE_SDB_ALIAS_CANONICAL:
            entry["formula_usage"] = "forbidden"
            entry.pop("signal_domain", None)
    if market == "us_stock" and role == "signal" and formula_usage == "primary_signal":
        entry["formula_us_dsl"] = canonical
    return entry


def build_table_from_scan(
    table: str,
    scan: dict[str, Any],
    market: str,
    local_path: str,
    explicit: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """将 schema 扫描结果组装为逻辑表定义（含全部列条目）。"""
    meta = get_table_meta(table, market)
    ts = "TradeDate" if "TradeDate" in scan["columns"] else None
    inst = "Symbol" if "Symbol" in scan["columns"] else (
        "Ticker" if "Ticker" in scan["columns"] else None
    )
    cols = {}
    for physical, dtype in scan["columns"].items():
        cols[physical] = build_column_entry(physical, dtype, table, market, explicit)
    return {
        "logical_table": table,
        "domain_root": meta.get("domain_root", "price_volume"),
        "table_role": meta.get("role", "primary"),
        "local_path": local_path,
        "timestamp_col": ts,
        "instrument_col": inst,
        "parquet_file_count": scan.get("file_count"),
        "columns": cols,
        **({"note": meta["note"]} if meta.get("note") else {}),
    }


def load_massive_datasets() -> dict[str, list[dict[str, str]]]:
    """从 Massive 数据字典 JSON 加载外部数据集列清单。"""
    if not MASSIVE_DICT.is_file():
        return {}
    data = json.loads(MASSIVE_DICT.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, str]]] = {}
    for ds in data.get("datasets", []):
        name = ds["dataset"]
        out[name] = [{"name": c["name"], "dtype": c.get("dtype", ""), "meaning_zh": c.get("meaning_zh", "")} for c in ds.get("columns", [])]
    return out


def build_massive_external_tables(massive: dict[str, list]) -> dict[str, Any]:
    """Massive 原始数据集 → 逻辑表 / 外部路径（未必在本地 massive_data 镜像）。"""
    mapping = {
        "us_stocks_sip/day_aggs_v1": {
            "logical_table": "StockDailyBar",
            "domain_root": "price_volume",
            "frequency_bucket": "daily",
            "timestamp_col": "window_start",
            "instrument_col": "ticker",
            "maps_to_local": "StockDailyBar",
            "sip_to_physical": {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "n": "transactions"},
        },
        "us_stocks_sip/minute_aggs_v1": {
            "logical_table": "StockMinuteBar",
            "domain_root": "microstructure",
            "frequency_bucket": "minute",
            "timestamp_col": "window_start",
            "instrument_col": "ticker",
        },
        "us_stocks_sip/trades_v1": {
            "logical_table": "StockTransaction",
            "domain_root": "microstructure",
            "frequency_bucket": "tick",
            "timestamp_col": "sip_timestamp",
            "instrument_col": "ticker",
        },
        "us_stocks_sip/quotes_v1": {
            "logical_table": "StockQuote",
            "domain_root": "microstructure",
            "frequency_bucket": "quote",
            "timestamp_col": "sip_timestamp",
            "instrument_col": "ticker",
        },
        "fundamentals/balance_sheet": {
            "logical_table": "StockBalance",
            "domain_root": "fundamental",
            "timestamp_col": "period_end",
            "instrument_col": "tickers",
        },
        "fundamentals/income_statement": {
            "logical_table": "StockIncome",
            "domain_root": "fundamental",
            "timestamp_col": "period_end",
            "instrument_col": "tickers",
        },
        "fundamentals/cash_flow_statement": {
            "logical_table": "StockCashFlow",
            "domain_root": "fundamental",
            "timestamp_col": "period_end",
            "instrument_col": "tickers",
        },
        "fundamentals/financials_ratios": {
            "logical_table": "StockFinancialRatios",
            "domain_root": "fundamental",
            "timestamp_col": "date",
            "instrument_col": "ticker",
        },
        "fundamentals/short_interest": {
            "logical_table": "StockShortInterest",
            "domain_root": "alternative",
            "timestamp_col": "settlement_date",
            "instrument_col": "ticker",
        },
        "fundamentals/short_volume": {
            "logical_table": "StockShortVolume",
            "domain_root": "alternative",
            "timestamp_col": "date",
            "instrument_col": "ticker",
        },
        "fundamentals/stocks_floats": {
            "logical_table": "StockFloat",
            "domain_root": "fundamental",
            "timestamp_col": "effective_date",
            "instrument_col": "ticker",
        },
    }
    external: dict[str, Any] = {}
    for ds_path, meta in mapping.items():
        cols_raw = massive.get(ds_path, [])
        columns = {}
        for c in cols_raw:
            physical = c["name"]
            canonical = pascal_to_snake(physical) if physical.islower() or "_" in physical else pascal_to_snake(physical)
            if physical in ("o", "h", "l", "c", "v"):
                sip_map = meta.get("sip_to_physical", {})
                canonical = pascal_to_snake(sip_map.get(physical, physical))
            role = column_role(physical, meta["logical_table"], "us_stock")
            sig_dom = meta.get("domain_root") if role == "signal" else None
            if meta["logical_table"] == "StockDailyBar" or ds_path.endswith("day_aggs_v1"):
                sig_dom = "price_volume"
            columns[physical] = {
                "physical": physical,
                "dtype": c.get("dtype", ""),
                "canonical": canonical,
                "role": role,
                "formula_usage": infer_formula_usage(role, sig_dom),
                "meaning_zh": c.get("meaning_zh", ""),
            }
            if sig_dom:
                columns[physical]["signal_domain"] = sig_dom
        external[ds_path] = {**meta, "massive_path": ds_path, "columns": columns}
    return external


def build_domain_scopes(doc: dict[str, Any]) -> dict[str, Any]:
    """按 domain_root × market 汇总可用主信号表/列与禁止跨域引用。"""
    templates = doc["mining_scope_templates"]
    descriptions = {
        "price_volume": "价量/收益/股本等日频行情主信号",
        "fundamental": "财报、估值、PIT/LTM 基本面主信号",
        "alternative": "股东、融券、新闻情绪等另类主信号",
        "microstructure": "分钟/逐笔/报价等微观结构主信号",
    }
    operator_policy = {
        "price_volume": {"ashare": "lqtp_pv_daily", "us_stock": "afv_us_pv_daily"},
        "fundamental": {"ashare": "lqtp_fs_event", "us_stock": "afv_us_pv_daily"},
        "alternative": {"ashare": "lqtp_pv_daily", "us_stock": "afv_us_pv_daily"},
        "microstructure": {"ashare": "lqtp_pv_daily", "us_stock": "afv_us_pv_daily"},
    }
    scopes: dict[str, Any] = {}
    aux_tpl = templates.get("auxiliary", {})

    for domain in DOMAIN_ROOTS:
        tpl = templates.get(domain, {})
        scopes[domain] = {
            "description": descriptions[domain],
            "operator_policy": operator_policy[domain],
            "campaign_rule": (
                "一场 campaign 仅一个 domain_root；manifest 中每个主信号列的 signal_domain "
                "必须等于 campaign.domain_root；不得跨 domain 混挖"
            ),
            "markets": {},
        }
        for market_id in ("us_stock", "ashare"):
            if market_id == "us_stock":
                primary = list(tpl.get("us_primary", []))
                if domain in ("fundamental", "alternative", "microstructure"):
                    primary.extend(f"massive:{p}" for p in tpl.get("us_massive_external", []))
            else:
                primary = list(tpl.get("ashare_primary", tpl.get("ashare_not_local", [])))

            forbidden: list[str] = []
            for other in DOMAIN_ROOTS:
                if other == domain:
                    continue
                ot = templates.get(other, {})
                if market_id == "us_stock":
                    forbidden.extend(ot.get("us_primary", []))
                else:
                    forbidden.extend(ot.get("ashare_primary", ot.get("ashare_not_local", [])))
            # 宽表（如 PanelDaily）可被多个 domain 引用不同列 — 表级不禁用，靠列级 signal_domain 约束
            forbidden = [t for t in forbidden if t not in primary]

            signal_columns: dict[str, list[str]] = {}
            for table, tdef in doc["markets"][market_id].get("tables_local", {}).items():
                in_primary = table in primary
                cols = [
                    c["canonical"]
                    for c in tdef["columns"].values()
                    if c.get("signal_domain") == domain
                    and c.get("formula_usage") == "primary_signal"
                ]
                if cols and (in_primary or table == "PanelDaily"):
                    signal_columns[table] = sorted(set(cols))

            if market_id == "us_stock":
                for ds_path, ext in doc["markets"]["us_stock"].get("tables_massive_external", {}).items():
                    if ext.get("domain_root") != domain and domain != "price_volume":
                        if not (domain == "fundamental" and ds_path.startswith("fundamentals/")):
                            if not (domain == "microstructure" and ds_path.startswith("us_stocks_sip/")):
                                if not (domain == "alternative" and "short" in ds_path):
                                    continue
                    cols = [
                        c["canonical"]
                        for c in ext.get("columns", {}).values()
                        if c.get("signal_domain") == domain or (
                            c.get("role") == "signal" and ext.get("domain_root") == domain
                        )
                    ]
                    if cols:
                        signal_columns[f"massive:{ds_path}"] = sorted(set(cols))

            filter_cols = sorted({
                c["canonical"]
                for tdef in doc["markets"][market_id].get("tables_local", {}).values()
                for c in tdef["columns"].values()
                if c.get("formula_usage") == "filter_only"
            })

            scopes[domain]["markets"][market_id] = {
                "primary_tables": sorted(set(primary)),
                "forbidden_primary_tables": sorted(set(forbidden)),
                "auxiliary_tables": list(aux_tpl.get(market_id, [])),
                "signal_columns_by_table": signal_columns,
                "filter_columns_global": filter_cols,
            }
    return scopes


def rescan_local_parquet() -> dict[str, Any]:
    """扫描 quant_projects 本地 parquet 目录，写入 _generated_schema_scan.json。"""
    import pyarrow.parquet as pq

    roots = {
        "us_stock": ROOT / "data" / "us_stock" / "massive_data",
        "ashare": ROOT / "data" / "a_share" / "lqtp_data",
    }
    result: dict[str, Any] = {}
    for market, root in roots.items():
        result[market] = {}
        if not root.is_dir():
            continue
        for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
            pqs = list(tdir.rglob("*.parquet"))
            if not pqs:
                continue
            cols: dict[str, str] = {}
            for f in pqs[:5]:
                for field in pq.read_schema(f):
                    cols[field.name] = str(field.type)
            result[market][tdir.name] = {"file_count": len(pqs), "columns": cols}
    SCHEMA_SCAN.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build_document() -> dict[str, Any]:
    """扫描本地 parquet 与 Massive 字典，组装完整 canonical_data_fields 文档。"""
    rescan_local_parquet()
    scan = json.loads(SCHEMA_SCAN.read_text(encoding="utf-8"))
    massive = load_massive_datasets()

    us_tables = {}
    for table, info in scan.get("us_stock", {}).items():
        if info:
            us_tables[table] = build_table_from_scan(
                table, info, "us_stock", f"data/us_stock/massive_data/{table}",
                US_PV_CANONICAL,
            )

    ashare_tables = {}
    for table, info in scan.get("ashare", {}).items():
        if info:
            ashare_tables[table] = build_table_from_scan(
                table, info, "ashare", f"data/a_share/lqtp_data/{table}",
                ASHARE_PV_CANONICAL,
            )

    # PanelDaily pit/ltm 常用信号列显式 canonical
    if "PanelDaily" in us_tables:
        pit_canonical = {
            "pit_total_assets": "pit_total_assets",
            "pit_total_liabilities": "pit_total_liabilities",
            "pit_total_equity": "pit_total_equity",
            "pit_revenue": "pit_revenue",
            "pit_net_income": "pit_net_income",
            "pit_eps_basic": "pit_eps_basic",
            "ltm_revenue": "ltm_revenue",
            "ltm_net_income": "ltm_net_income",
            "news_count_1d": "news_count_1d",
            "news_sentiment_last": "news_sentiment_last",
            "shares_out": "shares_out",
        }
        for phys, can in pit_canonical.items():
            if phys in us_tables["PanelDaily"]["columns"]:
                us_tables["PanelDaily"]["columns"][phys]["canonical"] = can
                us_tables["PanelDaily"]["columns"][phys]["formula_us_dsl"] = can

    doc = {
        "schema_version": "canonical_data_fields.v3",
        "generated_by": "factor_engine/scripts/build_canonical_fields.py",
        "sources": {
            "local_parquet_scan": str(SCHEMA_SCAN.relative_to(ROOT)),
            "massive_data_dictionary": str(MASSIVE_DICT.relative_to(ROOT)),
        },
        "domain_roots": ["price_volume", "fundamental", "alternative", "microstructure"],
        "column_roles": {
            "key": "主键/时间/标的列，不在公式中作信号",
            "signal": "可作因子主信号",
            "filter": "样本过滤（停牌、涨跌停、mask）",
            "auxiliary": "中性化/分组/映射，仅 auxiliary_tables",
            "metadata": "审计/批次/血缘，禁止入公式",
        },
        "formula_usage": {
            "primary_signal": "可作该 domain 下因子主信号（须 signal_domain 与 campaign.domain_root 一致）",
            "filter_only": "仅用于 filter 表达式，不得单独构成 alpha 主信号",
            "auxiliary_only": "仅 auxiliary_tables，用于中性化/分组，不得作主信号",
            "forbidden": "禁止出现在 manifest.formula",
        },
        "formula_conventions": {
            "us_stock_dsl": "manifest 公式用小写 canonical（如 close）；factor_engine YAML 用 fields 映射 physical",
            "ashare_lqtp_dsl": "优先 Table.Field（PascalCase）；PV 可用别名 close/volume（见 lqtp_alias_expands_to）",
            "financial_ashare": "StockIncome.OperatingRevenue 等全限定名；配合 ttm/quarter/yoy",
        },
        "markets": {
            "us_stock": {
                "default_data_root": "data/us_stock/massive_data/",
                "cos_data_root": "cos://qs-cold/clean_data/us_stock/massive_data/",
                "execution_engine": "factor_engine",
                "default_operator_policy_pv": "afv_us_pv_daily",
                "tables_local": us_tables,
                "tables_massive_external": build_massive_external_tables(massive),
            },
            "ashare": {
                "default_data_root": "data/a_share/lqtp_data/",
                "cos_data_root": "cos://qs-cold/clean_data/ashare/lqtp_data/",
                "execution_engine": "lqtp",
                "default_operator_policy_pv": "lqtp_pv_daily",
                "default_operator_policy_fs": "lqtp_fs_event",
                "tables_local": ashare_tables,
                "tables_not_synced_locally": {
                    "StockMinuteBar": {
                        "domain_root": "microstructure",
                        "frequency_bucket": "minute",
                        "note": "LQTP StockMinuteBarSQL；本地 parquet 未同步",
                        "lqtp_aliases": ["minute_open", "minute_close", "minute_volume"],
                    },
                    "StockTransaction": {
                        "domain_root": "microstructure",
                        "frequency_bucket": "tick",
                        "note": "LQTP StockTransactionSQL；本地 parquet 未同步",
                        "lqtp_aliases": ["tick_price", "tick_volume", "tick_side"],
                    },
                    "EtfDailyBar": {
                        "domain_root": "price_volume",
                        "note": "LQTP 支持；别名 etf_close 等",
                    },
                    "FuturesDailyBar": {
                        "domain_root": "price_volume",
                        "note": "LQTP 支持；别名 futures_close 等",
                    },
                    "BenchmarkIndexDailyBar": {
                        "domain_root": "price_volume",
                        "role": "auxiliary",
                        "note": "benchmark_index / rolling_beta 等",
                    },
                },
            },
        },
        "mining_scope_templates": {
            "price_volume": {
                "ashare_primary": ["StockDailyBar", "StockCapitalDaily"],
                "us_primary": ["StockDailyBar", "FactReturnsDaily", "PanelDaily"],
            },
            "fundamental": {
                "ashare_primary": [
                    "StockBalance", "StockIncome", "StockCashFlow",
                    "StockIndicator", "StockValuationDaily", "StockDividend",
                ],
                "us_primary": ["PanelDaily"],
                "us_massive_external": [
                    "fundamentals/balance_sheet", "fundamentals/income_statement",
                    "fundamentals/cash_flow_statement", "fundamentals/financials_ratios",
                ],
            },
            "alternative": {
                "ashare_primary": ["StockTopTenShareholder", "StockTopTenFloatShareholder"],
                "us_massive_external": ["fundamentals/short_interest", "fundamentals/short_volume"],
            },
            "microstructure": {
                "ashare_not_local": ["StockMinuteBar", "StockTransaction"],
                "us_massive_external": [
                    "us_stocks_sip/minute_aggs_v1", "us_stocks_sip/trades_v1", "us_stocks_sip/quotes_v1",
                ],
            },
            "auxiliary": {
                "ashare": ["StockList", "StockStatus", "StockIndustry", "Calendar"],
                "us_stock": ["DimCalendar", "DimSecurityMaster", "DimTickerMap", "DimTickerAlias"],
            },
        },
        "generators": {
            "note": "所有 generator 投递前须映射到 canonical；执行引擎由 market 决定，与 generator 无关",
            "registered_in_delivery_spec": [
                "quantaalpha", "cogalpha", "alphacfg", "alphaagentevo", "alphaprobe",
                "alphasage", "quantevolver", "alphaqcm", "rdagent", "evoalpha",
                "factorminer", "manual",
            ],
        },
    }
    doc["domain_scopes"] = build_domain_scopes(doc)
    return doc


def main() -> int:
    """生成或校验 ``canonical_data_fields.json`` 契约文档。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="校验现有 JSON 是否与当前扫描一致（仅比 local 表列数）")
    args = parser.parse_args()

    if not SCHEMA_SCAN.is_file():
        raise SystemExit(f"Missing schema scan: {SCHEMA_SCAN}. Run build without --check first.")

    doc = build_document()

    if args.check:
        if not OUT_JSON.is_file():
            raise SystemExit("canonical_data_fields.json missing")
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        mismatches = []
        for market in ("us_stock", "ashare"):
            old_t = old["markets"][market]["tables_local"]
            new_t = doc["markets"][market]["tables_local"]
            for table in set(old_t) | set(new_t):
                oc = len(old_t.get(table, {}).get("columns", {}))
                nc = len(new_t.get(table, {}).get("columns", {}))
                if oc != nc:
                    mismatches.append(f"{market}.{table}: {oc} -> {nc}")
        if mismatches:
            print("STALE:", *mismatches, sep="\n  ")
            return 1
        print("OK: column counts match")
        return 0

    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # 同步到 factor-pool-standard（单一契约副本）
    fps_copy = ROOT / "factor-pool-standard" / "enums" / "canonical_data_fields.json"
    fps_copy.parent.mkdir(parents=True, exist_ok=True)
    fps_copy.write_text(OUT_JSON.read_text(encoding="utf-8"), encoding="utf-8")

    us_n = sum(len(t["columns"]) for t in doc["markets"]["us_stock"]["tables_local"].values())
    ash_n = sum(len(t["columns"]) for t in doc["markets"]["ashare"]["tables_local"].values())
    ext_n = sum(len(t["columns"]) for t in doc["markets"]["us_stock"]["tables_massive_external"].values())
    print(f"Wrote {OUT_JSON}")
    print(f"  us_stock local: {len(doc['markets']['us_stock']['tables_local'])} tables, {us_n} columns")
    print(f"  ashare local: {len(doc['markets']['ashare']['tables_local'])} tables, {ash_n} columns")
    print(f"  us massive external: {len(doc['markets']['us_stock']['tables_massive_external'])} datasets, {ext_n} columns")
    print(f"  synced -> {fps_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
