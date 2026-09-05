"""挖掘框架 / 投递校验与 factor_engine 的对接入口。

能力
----
- ``validate_factor_engine_dsl(formula)`` / ``validate_us_dsl``：校验公式能否被 ``parse_expr`` 解析；
- ``validate_manifest_for_execution``：按 ``market`` + ``expression_type`` 决定是否做 DSL 校验；
- ``list_dsl_allowlist()``、``scripts/validate_delivery_formula.py``：投递前 CLI。

执行策略（A 股 / 美股统一）
----------------------------
**A 股与美股均走 factor_engine runtime**（``cleaned_operators`` + 混合后端）。
生产执行路由由 **Global Physical Planner**
（``runtime.multibackend.batch_global_optimizer.PhysicalBatchGlobalOptimizer``）
选择一份 **certified ``PhysicalRegionPlan``**（显式 region + ``TransferEdge``），
``BackendRouter`` 仅提供 capability/cost 候选，**不 finalize 生产路由**。

生产路径**无 executor 静默降级**：

- 每个 region 带显式 ``ExecutionKind``，按 certified ``PhysicalRegionPlan`` 执行；
- 运行时失败抛**类型化失败**（``runtime.exceptions`` 的 ``FailureTaxonomy``，
  R37-P0-072，含 retry/replan/shard/fallback/abort 语义）；
- 若策略允许（如 OOM → smaller-shape replan），由上层**显式 replan**，
  绝不由 executor 悄悄降级到 pandas；
- ``assert_no_production_pandas_fallbacks`` 在 production 模式下 hard-gate 任何
  unplanned pandas fallback（除非 ``production_fallback_policy='warn'``）。

仅数据源与 canonical 字段不同（A 股 parquet 常在 ``data/a_share/lqtp_data/``）。
``lqtp_dsl`` / ``platforms.lqtp`` 为历史 LQTP 平台路径，**新 campaign 不再依赖**。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from factor_engine.api.dsl_parser import DSLParseError, parse_expr
from factor_engine.api.operator_registry import build_dsl_allowlist

# composite 估值源：允许 DSL 使用 col("pe") 等无前缀简写
_VALUATION_FIELD_ALIASES: dict[str, str] = {
    "pe": "valuation.pe",
    "pb": "valuation.pb",
    "turnover_ratio": "valuation.turnover_ratio",
    "market_cap": "valuation.market_cap",
    "circulating_market_cap": "valuation.circulating_market_cap",
}
# US valuation uses the lowercase X0 physical names; there is NO turnover /
# circulating concept in us_stock_valuation_daily (those are A-share fields).
_US_VALUATION_FIELD_ALIASES: dict[str, str] = {
    "pe": "valuation.price_to_earnings",
    "pb": "valuation.price_to_book",
    "ps": "valuation.price_to_sales",
    "dividend_yield": "valuation.dividend_yield",
}


def validate_production_dsl(formula: str, *, market: str) -> tuple[bool, str]:
    """production 模式公式校验：语法 + 算子 production 允许。

    R24-158..160: the production validator runs a REAL ``Analyzer(production=True,
    market=resolved_market)`` (never the research Analyzer with an implicit
    market), and the bare-field check resolves through
    ``MULTI_MARKET_FIELD_REGISTRY.registry_for(market)`` — a US formula must
    never be validated against the legacy A-share registry.

    R21-P033: market is now a required parameter. Passing None or omitting it
    will cause a TypeError, preventing implicit defaults in production.

    Parameters
    ----------
    formula : str
        DSL 公式字符串。
    market : str
        The market the formula will run in (``ashare`` / ``us``).  Required.

    Returns
    -------
    tuple[bool, str]
        ``(True, "OK")`` 或 ``(False, 错误说明)``。
    """
    from factor_engine.cleaned_operators.operator_spec import check_production_formula_ops
    from factor_engine.ir.analyzer import Analyzer

    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    import ast
    if market is None:
        raise ValueError("market is required for production validation (R21-P033)")
    market = str(market).strip().lower()
    # R24-160: per-market field registry — never the legacy A-only FIELD_REGISTRY.
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

    registry = MULTI_MARKET_FIELD_REGISTRY.registry_for(market)

    try:
        tree = ast.parse(str(formula), mode="eval")
    except SyntaxError as exc:
        return False, str(exc)
    bare_secondary = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "col"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        spec = registry.get(node.args[0].value, strict=False)
        if spec is not None and spec.table != "StockDailyBar":
            bare_secondary.append(spec.name)
    if bare_secondary:
        return False, (
            "production DSL requires field(...) for cataloged non-anchor fields: "
            + ", ".join(sorted(set(bare_secondary)))
        )

    ok, msg = validate_factor_engine_dsl(formula)
    if not ok:
        return False, msg
    try:
        # R24-158: the production validator must run the PRODUCTION Analyzer
        # with the resolved market — a research/default Analyzer is a bypass.
        Analyzer(production=True, market=market).lower(
            parse_expr(str(formula), surface="daily"), production=True
        )
    except (DSLParseError, KeyError, ValueError, TypeError, SyntaxError) as exc:
        return False, str(exc)
    # Parsing may finalize/bootstrap operator metadata; refresh the certified
    # runtime view afterwards so production evidence is evaluated on the final
    # registry state rather than a partially loaded catalog.
    ensure_cleaned_loaded()
    violations = check_production_formula_ops(formula)
    if violations:
        return False, "; ".join(violations)
    from factor_engine.cleaned_operators.operator_spec import check_financial_grain_contract

    # R34 P0-021: 把 production Analyzer 解析出的 market context 透传给 financial
    # validator——不再硬编码 ASHARE_CONTEXT（默认 A 股，US 显式传入 US_CONTEXT）。
    grain_violations = check_financial_grain_contract(formula, market_context=market)
    if grain_violations:
        return False, "; ".join(grain_violations)
    return True, "OK"


def validate_production_fastpath_dsl(
    formula: str, *, strict: bool | None = None, market: str
) -> tuple[bool, str]:
    """production fast path 公式校验：语法 + 高性能 backend 允许。

    R21-FASTPATH-MARKET: ``market`` 现在是必填参数。与
    ``validate_production_dsl`` 统一 —— 不再允许 ``None`` 缺省为 A 股，
    显式传 ``None`` 或省略会直接失败。US 公式绝不能用默认 A-share
    registry 校验。

    Parameters
    ----------
    formula : str
        DSL 公式字符串。
    strict : bool | None
        传给 ``check_production_fastpath_formula_ops``；``None`` 用模块默认。
    market : str
        ``ashare`` / ``us``。必填（``Market.ASHARE`` / ``Market.US`` 或等价
        字符串）。省略或传 ``None`` 会抛 ``TypeError`` / ``ValueError``。

    Returns
    -------
    tuple[bool, str]
        ``(True, "OK")`` 或 ``(False, 违规说明)``。
    """
    if market is None:
        raise ValueError(
            "market is required for production fastpath validation (R21-FASTPATH-MARKET)"
        )
    market = str(market).strip().lower()
    ok, msg = validate_production_dsl(formula, market=market)
    if not ok:
        return False, msg
    from factor_engine.backend.production_fastpath_gate import check_production_fastpath_formula_ops

    result = check_production_fastpath_formula_ops(formula, strict=strict)
    if not result.ok:
        return False, "; ".join(result.violations)
    return True, "OK"


def list_production_fastpath_allowlist(*, strict: bool = True) -> list[str]:
    """返回 production fastpath 层算子白名单（排序后）。"""
    from factor_engine.backend.fastpath_allowlists import production_fastpath_allowlist

    return sorted(production_fastpath_allowlist(strict=strict))


def export_fastpath_allowlists_json(*, strict: bool = True) -> dict[str, Any]:
    """导出 research / production / production_fastpath 三层 allowlist。

    Returns
    -------
    dict[str, Any]
        含 ``schema_version``、各层算子列表及 ``counts`` 的 JSON 可序列化 dict。
    """
    from factor_engine.backend.fastpath_allowlists import (
        production_allowlist,
        production_fastpath_allowlist,
        research_allowlist,
    )

    research = sorted(research_allowlist())
    production = sorted(production_allowlist())
    fastpath = sorted(production_fastpath_allowlist(strict=strict))
    return {
        "schema_version": "factor_engine.fastpath_allowlists.v1",
        "research_allowlist": research,
        "production_allowlist": production,
        "production_fastpath_allowlist": fastpath,
        "counts": {
            "research": len(research),
            "production": len(production),
            "production_fastpath": len(fastpath),
        },
    }


def write_fastpath_allowlists(path: str | Path, *, strict: bool = True) -> Path:
    """将三层 fastpath allowlist 写入 JSON 文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(export_fastpath_allowlists_json(strict=strict), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def validate_factor_engine_dsl(
    formula: str,
    *,
    surface: str = "daily",
) -> tuple[bool, str]:
    """校验 factor_engine DSL 语法与白名单（A 股 / 美股同一套 ``parse_expr``）。

    Parameters
    ----------
    formula : str
        DSL 公式字符串。
    surface : str
        算子面：``daily``（默认，新投递）或 ``compat``（已发布 / 遗留公式，如 GTJA185）。

    Returns
    -------
    tuple[bool, str]
        ``(True, "OK")`` 或 ``(False, DSLParseError 消息)``。
    """
    text = str(formula or "").strip()
    if not text:
        return False, "empty formula"
    try:
        parse_expr(text, surface=str(surface or "daily"))
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)


def validate_syntax_only_dsl(formula: str, *, surface: str = "daily") -> tuple[bool, str]:
    """仅做 factor_engine DSL **语法** + 白名单解析校验。

    R40 #87: 旧名 ``validate_us_dsl`` 冒充「US 市场校验」——它实际上不校验
    US 市场/字段注册表/算子契约/提供者绑定，只是 ``parse_expr`` 语法解析。
    新名如实表达能力：**syntax-only**。要真正校验 US production 公式，请用
    :func:`validate_production_dsl` 并传 ``market="us"``。
    """
    return validate_factor_engine_dsl(formula, surface=surface)


def validate_us_dsl(formula: str, *, surface: str = "daily") -> tuple[bool, str]:
    """R40 #87 兼容旧名；同 :func:`validate_syntax_only_dsl`。

    Deprecated: 该名字暗示「US 市场校验」，实际只是语法解析。生产 US 校验请
    用 :func:`validate_production_dsl`(``market="us"``)。
    """
    import warnings

    warnings.warn(
        "validate_us_dsl is deprecated; use validate_syntax_only_dsl (syntax-only) "
        "or validate_production_dsl(formula, market='us') for a real US contract check.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_syntax_only_dsl(formula, surface=surface)


def list_dsl_allowlist(*, surface: str = "daily") -> list[str]:
    """返回完整 DSL 算子白名单（排序后的函数名列表）。"""
    return sorted(build_dsl_allowlist(surface=str(surface or "daily")).keys())


def export_dsl_allowlist_json(
    *,
    market: str,
    surface: str = "daily",
) -> dict[str, Any]:
    """导出 AFV Gateway 对齐用的算子白名单 JSON。

    R21-P033: market is now a required parameter. Passing None or omitting it
    will cause a TypeError.

    Parameters
    ----------
    market : str
        市场标识（``us`` / ``ashare`` 等），决定 ``operator_policy`` 字段。Required.
    surface : str
        ``daily`` 或 ``compat``（GTJA185 等 published 包用 compat）。

    Returns
    -------
    dict[str, Any]
        含 ``operators``、``count``、``operator_policy``、``dsl_surface`` 的 dict。
    """
    surf = str(surface or "daily").strip() or "daily"
    names = list_dsl_allowlist(surface=surf)
    if market is None:
        raise ValueError("market is required for export_dsl_allowlist_json (R21-P033)")
    mkt = str(market).strip().lower()
    policy_by_market = {
        "ashare": "lqtp_pv_daily",
        "a_share": "lqtp_pv_daily",
        "cn": "lqtp_pv_daily",
        "china": "lqtp_pv_daily",
        "us": "afv_us_pv_daily",
        "usa": "afv_us_pv_daily",
    }
    # R10-P0-026: an unknown / misspelt market (``ahsare``) must NOT silently
    # fall back to the US policy — that would certify the wrong operator
    # surface for an A-share campaign.  Fail closed.
    if mkt not in policy_by_market:
        raise ValueError(
            f"unknown market {market!r}; expected one of "
            f"{sorted(policy_by_market)}"
        )
    return {
        "schema_version": "factor_engine.dsl_allowlist.v1",
        "operator_policy": policy_by_market[mkt],
        "market": mkt,
        "dsl_surface": surf,
        "operators": names,
        "count": len(names),
    }


def default_ashare_pv_data_source_config(
    *,
    max_files: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """A 股价量数据源配置（复权权威口径）。

    始终走 ``data_access`` 登记数据集 ``ashare_stock_daily_adj``（Adj* 字段集）。
    ADJ_FIELD_MIGRATION（2026-08-28）：A 股行情权威口径 = 后复权表。旧 ``max_files``
    直连 parquet 分支已删除——factor_engine 不允许非 data_access 的 COS/parquet
    直读 IO（即使 smoke test 也走 data_access 的 read）。``max_files`` 保留为
    兼容参数（A 股场景不再改变数据源类型）。
    """
    fields = {
        "open": "AdjOpen",
        "high": "AdjHigh",
        "low": "AdjLow",
        "close": "AdjClose",
        "volume": "Volume",
        "vwap": "AdjVwap",
        "amount": "AdjAmount",
        "ret": "Return",
        "preclose": "AdjPreClose",
    }
    cfg = {
        "type": "data_access",
        "dataset": "ashare_stock_daily_adj",
        "fields": fields,
        "read_auto": True,
    }
    if max_files is not None:
        cfg["max_files"] = max_files
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
    """A 股日线 + 估值复合数据源（锚点日线，估值严格按交易日对齐）。"""
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
            "valuation": "exact",
        },
        "aliases": dict(_VALUATION_FIELD_ALIASES),
    }


def ashare_dataset_registry() -> dict[str, str]:
    """已登记的 A 股 lqtp 数据集名 → COS 子目录名。"""
    from data_access.cos.mirror import ASHARE_DATASET_TABLE_MAP

    return dict(ASHARE_DATASET_TABLE_MAP)


def us_dataset_registry() -> dict[str, str]:
    """已登记的美股数据集名 → COS 子目录名（massive_data 部分）。"""
    from data_access.cos.mirror import DATASET_MIRROR_REGISTRY

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
    """美股日线 + 估值复合数据源（估值仅为 current/research snapshot）。

    P0-013: the previous config used A-share physical field names (PeRatio /
    PbRatio / TurnoverRatio / MarketCap / CirculatingMarketCap) against
    ``us_stock_valuation_daily``, which stores the *lowercase* X0 snapshot names
    (``price_to_earnings`` / ``price_to_book`` / ``price_to_sales`` /
    ``market_cap`` / ``dividend_yield``) and has no turnover / circulating field.
    US Valuation/Indicator is an X0 sparse snapshot (~49 files, current-only) —
    it is marked ``snapshot_only`` and MUST NOT be asof-ffilled into a historical
    daily panel for default alpha mining.  The historical US market cap comes from
    ``Close * weighted_shares_outstanding`` (TickerSharesSnapshot), not this X0.
    """
    pv = default_us_pv_data_source_config(
        start_date=start_date,
        end_date=end_date,
    )
    valuation: dict[str, Any] = {
        "type": "data_access",
        "dataset": "us_stock_valuation_daily",
        "fields": {
            "pe": "price_to_earnings",
            "pb": "price_to_book",
            "ps": "price_to_sales",
            "dividend_yield": "dividend_yield",
        },
        "snapshot_only": True,
        "usage": "research_snapshot",
        "notes": "X0 sparse current-snapshot valuation; do NOT asof-ffill into history",
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
            # current-only join: valuation is a snapshot, never backfilled over
            # the historical panel (asof_backward would invent history).
            "valuation": "current_only",
        },
        "aliases": dict(_US_VALUATION_FIELD_ALIASES),
    }


def default_ashare_pv_universe_data_source_config(
    *,
    index_symbol: str,
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
            "public_status": "ListedState",
            # Compatibility alias for formulas that still request listed_state.
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
        "semantic_filters": {"IndexSymbol": str(index_symbol)},
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
            "constituent": "exact",
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
    require_production: bool = False,
    require_fastpath: bool | None = None,
    surface: str | None = None,
) -> tuple[bool, str]:
    """按 market / expression_type 决定是否做 factor_engine 语法校验。

    ``require_production``：额外校验 production allowlist。
    ``require_fastpath``：额外校验 production fastpath；默认读
    ``FACTOR_ENGINE_MINING_REQUIRE_FASTPATH=1``。
    ``surface``：``daily`` / ``compat``；也可从环境 ``FACTOR_ENGINE_DSL_SURFACE`` 读取。
    已发布包（如 GTJA185）应传 ``surface=\"compat\"`` 或在 campaign 标 ``dsl_surface``。
    """
    import os

    mkt = str(market or "").strip()
    et = str(expression_type or "dsl").strip() or "dsl"
    if et == "python":
        if require_production or require_fastpath:
            return False, "python expression is research-only; translate to validated DSL before production delivery"
        return True, "research-only: translation_required=true; valid_for_production=false"
    if et not in ("dsl", "lqtp_dsl", ""):
        return False, f"unsupported expression_type: {et!r}"

    if require_fastpath is None:
        require_fastpath = os.environ.get("FACTOR_ENGINE_MINING_REQUIRE_FASTPATH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    surf = str(
        surface
        or os.environ.get("FACTOR_ENGINE_DSL_SURFACE", "").strip()
        or "daily"
    ).strip() or "daily"

    if require_fastpath:
        return validate_production_fastpath_dsl(formula, market=mkt)
    if require_production:
        return validate_production_dsl(formula, market=mkt)
    return validate_factor_engine_dsl(formula, surface=surf)


# R22-020: the new semantic mining tiers.  ``research`` / ``production`` /
# ``production_fastpath`` are kept only as backward-compat aliases.
_R22_MINING_TIER_LANES: dict[str, str] = {
    # authoritative tiers
    "direct_standard": "alpha_direct",   # DIRECT_ALPHA / DIRECT_ALPHA_HIGH_COST terminal lane
    "direct_high_cost": "alpha_high_cost",  # DIRECT_ALPHA_HIGH_COST lane
    "direct_all_context": "all",         # every DIRECT_* lane (incl. state/event/intermediate)
    "research_tools": "research_tools",  # research-tool manifest (NOT for automated miners)
    # legacy aliases
    "research": "all",                   # exploration: every retained DIRECT_* incl. uncertified
    "production": "all",                 # eligible, every DIRECT_* context
    "production_fastpath": "all",        # eligible, every DIRECT_* context (+fastpath validation flag)
}

# R22-017..019: fastpath is ONLY an execution preference / performance lane.  It
# never decides which operator is eligible to be mined — a single-backend
# (pandas-only) DIRECT_* operator is eligible as long as DirectUse admits it
# (R22-133..135).  Legacy ``production_fastpath`` keeps its validation flag.
_FASTPATH_VALIDATION_TIERS = frozenset({"production_fastpath"})


def default_mining_operator_allowlist(*, tier: str = "production_fastpath") -> list[str]:
    """挖掘搜索空间分层 allowlist（R18-001 / R22-018 authority 收口）。

    R18-001 / R22-017..019: 自动挖掘**只允许**走 ``get_direct_use_mining_operators()``
    这个 direct-use authority。``backend.fastpath_allowlists`` 只表达 backend
    execution capability（"这个算子能编译/能执行"），**不再决定矿工能看什么**。

    * 所有 tier 的**搜索空间 surface** 都是 retained DIRECT_* 算子（role-derived，
      与 evidence 无关 —— certification 由 R18-006 cold-start 在 execution 时
      逐算子验证，不在搜索空间 surface 层过滤）。
    * R22-020 新 tier：``direct_standard`` / ``direct_high_cost`` /
      ``direct_all_context`` / ``research_tools``；旧 ``research`` / ``production`` /
      ``production_fastpath`` 保留为别名。
    """
    from factor_engine.market.context import Market
    from factor_engine.mining.direct_use import (
        DirectUseContext,
        DirectUseStatus,
        direct_use_matrix_rows,
        get_direct_use_mining_operators,
    )

    tier_key = resolve_mining_allowlist_tier(tier)
    if tier_key == "research_tools":
        rows = direct_use_matrix_rows()
        return sorted(
            r.canonical
            for r in rows
            if r.direct_use_status is DirectUseStatus.RESEARCH_TOOL
        )
    lane = _R22_MINING_TIER_LANES.get(tier_key, "all")
    rows = get_direct_use_mining_operators(
        DirectUseContext(market=Market.ASHARE), admission="all"
    )
    direct = [r for r in rows if lane == "all" or r.mining_lane == lane]
    return sorted(r.canonical for r in direct)


def resolve_mining_allowlist_tier(tier: str | None = None) -> str:
    """解析挖掘 allowlist tier；默认 ``FACTOR_ENGINE_MINING_ALLOWLIST_TIER``。

    R22-020: 新 tier 名（``direct_standard`` / ``direct_high_cost`` /
    ``direct_all_context`` / ``research_tools``）优先；旧名仅作兼容别名。"""
    import os

    if tier is not None and str(tier).strip():
        return str(tier).strip().lower()
    env = os.environ.get("FACTOR_ENGINE_MINING_ALLOWLIST_TIER", "").strip().lower()
    if env:
        return env
    return "direct_all_context"


def get_mining_operators_from_manifest(
    manifest_path: str,
    *,
    run_mode: str = "production",
    allow_stale: bool = False,
    market: str | None = None,
    max_cost: int | None = None,
    available_sources: list[str] | None = None,
    target_frequency: str | None = None,
) -> list[str]:
    """FE-P0-035: Production/cold-start entry point with manifest validation.

    Wires validate_direct_use_manifest() into production and cold-start mining
    paths. Fails closed when:
    - Manifest is missing, malformed, or has invalid schema
    - Manifest fingerprint is stale (unless allow_stale=True)
    - git HEAD lookup fails in production/cold-start mode
    - Context filtering removes all operators

    R61-P0 #64 (fail-closed evidence gate): when ``run_mode`` is
    ``"production"`` or ``"cold_start"`` this function ALSO calls
    ``evidence.gate.require_current_evidence()`` BEFORE returning operators —
    the Agent-direct operator surface may not run on stale CURRENT evidence.
    A stale/missing ``evidence/CURRENT.json`` raises
    ``evidence.gate.StaleAgentOperatorEvidence`` (fail-closed).  In
    ``"research"`` mode the gate is evaluated with ``allow_stale=True``
    (evaluate but never raise).  The gate is a SEPARATE, non-overridable
    production concern: the caller's ``allow_stale`` relaxes ONLY the manifest
    fingerprint check and never bypasses the evidence gate.

    Escape hatch (legacy in-repo test harnesses only): set
    ``FACTOR_ENGINE_EVIDENCE_GATE=off`` to disable the raise (a
    ``warnings.warn`` is emitted instead).  Default (unset or any other value)
    = enforced.  There is no warn-and-proceed path in production.

    Args:
        manifest_path: Path to direct-use manifest JSON
        run_mode: "production" | "cold_start" | "research"
        allow_stale: Allow stale manifest fingerprint (default False);
            relaxes ONLY the manifest-fingerprint check, never the evidence gate
        market: Market context for filtering (e.g., "ashare", "us")
        max_cost: Maximum runtime cost for filtering
        available_sources: Available data sources for filtering
        target_frequency: Target frequency for filtering ("daily", "minute")

    Returns:
        List of canonical operator names after validation and filtering

    Raises:
        ManifestValidationError: When manifest validation fails in
            production/cold-start mode
        StaleAgentOperatorEvidence: When run_mode is production/cold_start and
            evidence/CURRENT.json cannot be proven CURRENT against the live tree
    """
    from factor_engine.market.context import Market
    from factor_engine.mining.direct_use import (
        DirectUseContext,
        get_direct_use_mining_operators_from_manifest,
    )

    # R61-P0 #64: lazy evidence-gate import keeps the module import graph clean.
    # run_mode normalization mirrors get_direct_use_mining_operators_from_manifest.
    from evidence.gate import require_current_evidence

    _mode = str(run_mode or "production").strip().lower()
    if _mode in ("production", "cold_start"):
        require_current_evidence(allow_stale=False)
    elif _mode == "research":
        require_current_evidence(allow_stale=True)
    else:
        raise ValueError(
            f"unknown run_mode {run_mode!r}; expected production|cold_start|research"
        )

    # Build context for filtering (performs real work, not a pass stub)
    # Convert string market to Market enum if provided
    market_enum: Market | None = None
    if market is not None:
        market_str = str(market).strip().lower()
        if market_str in ("ashare", "cn", "china", "a_share"):
            market_enum = Market.ASHARE
        elif market_str in ("us", "usa"):
            market_enum = Market.US
        else:
            raise ValueError(f"unknown market {market!r}; expected ashare|us")

    context = DirectUseContext(
        available_sources=tuple(available_sources) if available_sources else (),
        target_frequency=target_frequency,
        market=market_enum,
        max_cost=max_cost,
    )

    # Validate and load manifest with context filtering
    return get_direct_use_mining_operators_from_manifest(
        manifest_path,
        run_mode=run_mode,
        allow_stale=allow_stale,
        context=context,
    )


def default_mining_search_space_config(
    *,
    tier: str | None = None,
    version: str = "typed_v2",
    fields: list[dict[str, Any]] | None = None,
    max_domains: int = 2,
    max_cost: float | None = None,
    field_dq_policy: str = "drop",
) -> dict[str, Any]:
    """Campaign / DSL 生成器搜索空间。

    ``version='typed_v2'`` 是新 campaign 默认契约；``version='v1'`` 仅保留
    历史 allowlist 兼容。
    """
    tier_key = resolve_mining_allowlist_tier(tier)
    ops = default_mining_operator_allowlist(tier=tier_key)
    if str(version).lower() in {"v2", "2", "typed_v2"}:
        return default_typed_mining_search_space_config(
            tier=tier_key,
            fields=fields,
            max_domains=max_domains,
            max_cost=max_cost,
            field_dq_policy=field_dq_policy,
        )
    return {
        "schema_version": "factor_engine.mining_search_space.v1",
        "allowlist_tier": tier_key,
        "operators": ops,
        "count": len(ops),
        "require_fastpath_validation": tier_key in _FASTPATH_VALIDATION_TIERS,
    }


_TYPED_FIELD_DEFAULTS: tuple[dict[str, Any], ...] = (
    {"name": "open", "frequency": "daily", "domain": "price", "cardinality": "panel", "unit": "price"},
    {"name": "high", "frequency": "daily", "domain": "price", "cardinality": "panel", "unit": "price"},
    {"name": "low", "frequency": "daily", "domain": "price", "cardinality": "panel", "unit": "price"},
    {"name": "close", "frequency": "daily", "domain": "price", "cardinality": "panel", "unit": "price"},
    {"name": "volume", "frequency": "daily", "domain": "liquidity", "cardinality": "panel", "unit": "shares"},
    {"name": "ret", "frequency": "daily", "domain": "return", "cardinality": "panel", "unit": "return"},
    {"name": "pe", "frequency": "daily", "domain": "valuation", "cardinality": "panel", "unit": "ratio"},
    {"name": "pb", "frequency": "daily", "domain": "valuation", "cardinality": "panel", "unit": "ratio"},
    {"name": "free_float_shares", "frequency": "daily", "domain": "capital", "cardinality": "panel", "unit": "shares"},
    {"name": "benchmark_ret", "frequency": "daily", "domain": "benchmark", "cardinality": "broadcast", "unit": "return"},
)


def _typed_tag_values(tags: list[str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for tag in tags or []:
        text = str(tag)
        if ":" in text:
            key, value = text.split(":", 1)
            if key in {"signature", "domain", "unit", "cost"}:
                values[key] = value
    return values


def default_typed_mining_search_space_config(
    *,
    tier: str | None = None,
    fields: list[dict[str, Any]] | None = None,
    max_domains: int = 2,
    max_cost: float | None = None,
    field_dq_policy: str = "drop",
) -> dict[str, Any]:
    """构建 typed mining v2 搜索空间。

    字段显式携带 frequency/domain/cardinality/unit/DQ policy；算子签名优先
    读取 metadata tags，基础层没有 signature/cost 字段时回退 catalog。
    """
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    domains_limit = int(max_domains)
    if domains_limit < 1 or domains_limit > 2:
        raise ValueError("max_domains must be 1 or 2")
    dq = str(field_dq_policy or "drop").lower()
    if dq not in {"drop", "mask", "reject", "allow"}:
        raise ValueError("field_dq_policy must be drop, mask, reject, or allow")

    typed_fields: list[dict[str, Any]] = []
    if fields is None:
        from factor_engine.fields import FIELD_REGISTRY

        raw_fields = [
            {
                "name": spec.name,
                "field_id": spec.field_id,
                "field_expr": f"field({spec.name!r})",
                "table": spec.table,
                "source_name": spec.source_name,
                "frequency": spec.frequency,
                "domain": spec.domain,
                "cardinality": spec.cardinality,
                "unit": spec.canonical_unit,
                "value_kind": spec.value_kind,
                "temporal_model": spec.temporal_model,
                "strict_pit_allowed": spec.strict_pit_allowed,
                "required_filters": list(spec.required_filters),
            }
            for spec in FIELD_REGISTRY.fields()
            if spec.mining_allowed
            and spec.cardinality != "one_to_many"
            # ADJ_FIELD_MIGRATION（2026-08-28）：同一 logical 名同时存在于
            # 未复权 StockDailyBar 与复权 StockDailyBarAdj 时，挖掘字段表只发布
            # 复权（权威口径）版本，避免挖掘面裸名歧义；未复权原始列仍可经显式
            # table/raw_* 概念访问。
            and not (
                spec.table == "StockDailyBar"
                and spec.name in {"open", "high", "low", "close", "vwap", "amount",
                                  "pre_close", "high_limit", "low_limit", "volume", "ret"}
            )
        ]
    else:
        raw_fields = list(fields)
    for raw in raw_fields:
        field = dict(raw)
        missing = [key for key in ("name", "frequency", "domain", "cardinality", "unit") if not field.get(key)]
        if missing:
            raise ValueError(f"typed field missing required metadata: {missing}")
        field.setdefault("dq_policy", dq)
        typed_fields.append(field)

    tier_key = resolve_mining_allowlist_tier(tier)
    allowed = default_mining_operator_allowlist(tier=tier_key)
    catalog = OperatorRegistry.catalog()
    signatures: list[dict[str, Any]] = []
    for canonical in allowed:
        entry = catalog.get(canonical) or {}
        op = OperatorRegistry.get(canonical)
        meta = getattr(op, "metadata", None) if op is not None else None
        tags = list(getattr(meta, "tags", None) or [])
        values = _typed_tag_values(tags)
        params = list(entry.get("param_names") or getattr(meta, "param_names", None) or [])
        # R18-002: data inputs / scalar parameters / context / group / event are
        # FIVE separate lists.  ``inputs`` (kept for backward compat) equals
        # exactly the data-input list — a window/lag/threshold scalar knob is
        # NEVER a data input.
        from factor_engine.mining.direct_use import (
            split_input_slots,
            terminal_allowed_for,
        )

        slot_split = split_input_slots(canonical, entry)
        data_inputs = slot_split["data_inputs"]
        scalar_params = slot_split["scalar_parameters"]
        context_inputs = slot_split["context_inputs"]
        group_inputs = slot_split["group_inputs"]
        event_inputs = slot_split["event_inputs"]
        cost = float(values.get("cost", entry.get("cost", 1.0)))
        if not cost > 0:
            raise ValueError(f"operator {canonical!r} has invalid mining cost {cost!r}")
        if max_cost is not None and cost > float(max_cost):
            continue
        policy = entry.get("contract") or {}
        domains = [values["domain"]] if values.get("domain") else []
        if not domains and getattr(meta, "input_fields", None):
            domains = sorted({str(value) for value in meta.input_fields if value})
        input_units = dict(getattr(meta, "input_units", None) or {})
        output_unit = getattr(meta, "output_unit", None) or values.get("unit") or "inherit"
        scope = str(policy.get("scope") or entry.get("scope") or "unknown")
        frequency = "minute" if scope == "session_intraday" else "daily"
        cardinality = "group" if scope in {"group", "cross_sectional"} else "panel"
        signature_entry = {
            "name": canonical,
            # R18-002: ``inputs`` == data inputs ONLY (never scalar knobs).
            "inputs": data_inputs,
            "data_inputs": data_inputs,
            "scalar_parameters": scalar_params,
            "context_inputs": context_inputs,
            "group_inputs": group_inputs,
            "event_inputs": event_inputs,
            "input_units": input_units,
            "output": getattr(meta, "return_type", None) or entry.get("return_type") or "series",
            "signature": values.get("signature") or f"{','.join(params)}->series",
            "domains": domains,
            "frequency": frequency,
            "cardinality": cardinality,
            "unit": output_unit,
            "cost": cost,
        }
        # R18-004 / R22-028: role / direct-use resolution failure is a HARD
        # ERROR for a mining search space — never emit an operator signature
        # with a missing role (no ``except Exception: pass`` here).
        from factor_engine.mining.direct_use import (
            build_direct_use_operator,
            resolve_direct_use_status,
            terminal_allowed_for,
        )
        from factor_engine.mining.operator_catalog import assign_mining_role

        role = assign_mining_role(canonical, entry)
        row = build_direct_use_operator(canonical, entry)
        signature_entry["mining_role"] = role.value
        # R22-006: allowed AST positions come from the DirectUse positive
        # contract (the single authority), NOT the role's exclusion-based
        # ``_ROLE_AST_POSITIONS`` — an intermediate's positions must not drift
        # into "terminal" just because its role is ALPHA.
        signature_entry["allowed_ast_positions"] = list(row.allowed_ast_positions)
        # R22-006: terminal_allowed is DirectUse positive authority.
        signature_entry["terminal_allowed"] = row.terminal_usable
        signature_entry["direct_use_status"] = row.direct_use_status.value
        # R22-178: complete signature — lane / roles / input slots / output
        # domain / market-source contexts / cost / search grades / default recipe.
        signature_entry["direct_use_status"] = row.direct_use_status.value
        signature_entry["lane"] = row.mining_lane
        signature_entry["mining_visible"] = row.mining_visible
        signature_entry["composition_usable"] = row.composition_usable
        # R23: terminal_usable is the semantic gate; production_terminal_usable
        # is the production gate (terminal_usable AND production_admitted).
        # Mining consumers must use production_terminal_usable as the final
        # flag for terminal placement in production.
        signature_entry["terminal_usable"] = row.terminal_usable
        signature_entry["production_terminal_usable"] = row.production_terminal_usable
        signature_entry["production_admitted"] = row.production_admitted
        signature_entry["context_admitted"] = row.context_admitted
        signature_entry["output_domain"] = row.output_value_domain or ""
        signature_entry["market_contexts"] = list(row.supported_markets)
        signature_entry["source_recipes"] = list(row.source_recipes)
        signature_entry["search_grades"] = dict(row.search_grade_by_param)
        signature_entry["default_recipe"] = dict(row.default_input_recipe)
        # R22-136..137: search budget (configurable defaults — a campaign may
        # override; never hardcoded economic conclusions).
        signature_entry["search_prior"] = row.search_prior
        signature_entry["family_budget"] = row.family_budget
        signature_entry["cost_budget"] = row.cost_budget
        signatures.append(signature_entry)

    return {
        "schema_version": "factor_engine.mining_search_space.v2",
        "typing": "strict",
        "allowlist_tier": tier_key,
        "v1_allowlist": allowed,
        "fields": typed_fields,
        "operators": signatures,
        "constraints": {"max_domains": domains_limit, "max_cost": max_cost},
        "field_dq_policy": dq,
        "field_catalog_hash": FIELD_REGISTRY.catalog_hash(),
        "count": len(signatures),
        "require_fastpath_validation": tier_key in _FASTPATH_VALIDATION_TIERS,
    }


def validate_formula_in_mining_allowlist(
    formula: str,
    *,
    tier: str | None = None,
    market: str,
    max_domains: int = 2,
) -> tuple[bool, str]:
    """校验公式算子是否在指定 tier allowlist 内。

    R21-FASTPATH-MARKET: ``market`` 为必填参数，生产层校验（fastpath /
    production）会透传给 ``validate_production_dsl`` —— 无 A 股缺省。

    Parameters
    ----------
    formula : str
        DSL 公式字符串。
    tier : str | None
        ``research`` | ``production`` | ``production_fastpath``；
        默认读 ``FACTOR_ENGINE_MINING_ALLOWLIST_TIER``。
    market : str
        ``ashare`` / ``us``。必填（省略或 ``None`` 会抛错）。

    Returns
    -------
    tuple[bool, str]
        ``(True, "OK")`` 或 ``(False, 违规说明)``。
    """
    import ast

    from factor_engine.cleaned_operators.registry import OperatorRegistry

    tier_key = resolve_mining_allowlist_tier(tier)
    allowed = frozenset(default_mining_operator_allowlist(tier=tier_key))
    text = str(formula or "").strip()
    if not text:
        return False, "empty formula"
    ok, msg = validate_factor_engine_dsl(text)
    if not ok:
        return False, msg
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        return False, str(exc)
    unknown: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in {"col", "field"}:
                continue
            canon = OperatorRegistry._aliases.get(name, name)
            if canon not in allowed and canon not in {"column", "literal"}:
                unknown.append(canon)
    if unknown:
        return False, f"operators not in {tier_key} allowlist: {sorted(set(unknown))}"
    from factor_engine.ir.analyzer import Analyzer, validate_max_domains

    try:
        analysis = Analyzer().lower(parse_expr(text, surface="daily"))
    except Exception as exc:
        return False, str(exc)
    domain_errors = validate_max_domains(analysis, max_domains=max_domains)
    if domain_errors:
        return False, "; ".join(domain_errors)
    if tier_key in _FASTPATH_VALIDATION_TIERS:
        return validate_production_fastpath_dsl(text, market=market)
    if tier_key == "production":
        return validate_production_dsl(text, market=market)
    return True, "OK"


def write_mining_search_space(
    path: str | Path,
    *,
    tier: str | None = None,
    version: str = "typed_v2",
    fields: list[dict[str, Any]] | None = None,
    max_domains: int = 2,
    max_cost: float | None = None,
    field_dq_policy: str = "drop",
) -> Path:
    """Write a deterministic versioned mining search-space document."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = default_mining_search_space_config(
        tier=tier,
        version=version,
        fields=fields,
        max_domains=max_domains,
        max_cost=max_cost,
        field_dq_policy=field_dq_policy,
    )
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def write_dsl_allowlist(path: str | Path, *, market: str = "us", surface: str = "daily") -> Path:
    """将 ``export_dsl_allowlist_json()`` 写入 JSON 文件。

    R21-P033: ``export_dsl_allowlist_json`` 要求显式 ``market``（默认 ``us``，
    与既有 ``docs/dsl_allowlist.json`` 的 ``operator_policy=afv_us_pv_daily`` 一致）。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            export_dsl_allowlist_json(market=market, surface=surface),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
    from factor_engine.api.label_pit import default_mining_label_config as _label_cfg

    return _label_cfg(
        horizon_bars=horizon_bars,
        feature_lookback_bars=feature_lookback_bars,
        gap_bars=gap_bars,
    )


def validate_mining_label_formula(formula: str, *, enforce: bool = True) -> tuple[bool, str]:
    """校验标签 DSL 不含前视算子。

    Parameters
    ----------
    formula : str
        标签 DSL 公式。
    enforce : bool
        传给 ``validate_label_formula_for_pit`` 的 ``enforce`` 参数。

    Returns
    -------
    tuple[bool, str]
        ``(True, "OK")`` 或 ``(False, 违规说明)``。
    """
    from factor_engine.api.label_pit import validate_label_formula_for_pit

    try:
        report = validate_label_formula_for_pit(formula, enforce=enforce)
        if report["ok"]:
            return True, "OK"
        return False, "; ".join(report.get("violations") or [])
    except Exception as exc:
        return False, str(exc)


def audit_composite_join_policies(config: dict[str, Any]) -> list[str]:
    """Audit composite joins against explicit per-source contracts.

    A-share daily valuation is an exact date×instrument table.  Other legacy
    presets remain backward-asof unless they opt into an explicit ``join_policy``
    contract on the child source.
    """
    violations: list[str] = []
    if str(config.get("type", "")).lower() != "composite":
        return violations
    joins = config.get("joins") or {}
    sources = config.get("sources") or {}
    anchor = str(config.get("anchor") or config.get("anchor_source") or "pv")
    dataset_policies = {
        "ashare_stock_valuation_daily": "exact",
        "ashare_stock_status": "asof_backward",
        "ashare_index_constituent": "exact",
    }
    for name, mode in joins.items():
        if str(name) == anchor:
            continue
        child = sources.get(name) if isinstance(sources, dict) else None
        child = child if isinstance(child, dict) else {}
        expected = str(
            child.get("join_policy")
            or dataset_policies.get(str(child.get("dataset") or ""))
            or "asof_backward"
        ).lower()
        actual = str(mode).lower()
        aliases = {"backward": "asof_backward", "asof": "asof_backward"}
        actual = aliases.get(actual, actual)
        if actual != expected:
            violations.append(f"{name}: join={mode!r} (expected {expected})")
    for name, sub in sources.items():
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
        "ashare_pv_universe": default_ashare_pv_universe_data_source_config(
            index_symbol="000300.SH"
        ),
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
    from factor_engine.api.datasets_contract import audit_mining_dataset_contract

    return audit_mining_dataset_contract()
