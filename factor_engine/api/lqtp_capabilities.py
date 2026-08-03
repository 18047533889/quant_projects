# -*- coding: utf-8 -*-
"""Machine-readable LQTP compatibility states.

recognized != parseable != executable != production-certified. Source-aware
entries may additionally require an explicit external dataset/definition.
"""
from __future__ import annotations
from typing import Any


def build_lqtp_capability_manifest() -> dict[str, Any]:
    from api.lqtp_compat import DEFAULT_LQTP_DIALECT_VERSION
    from backend.backend_certification import backend_certification
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.registry import OperatorRegistry
    from backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    aliases = {
        "decay_linear":"ts_decay_linear","ts_rank_pct":"ts_rank","ts_ewm_mean":"ts_ema",
        "ts_expanding_rank":"expanding_rank","ts_hump_decay":"hump_decay",
        "sma(x,n)":"ts_mean","sma(x,n,m)":"ts_sma_cn","momentum":"ts_delta",
    }
    macros=["safe_log","nullif_zero","clean","historical_var"]
    sources={
        "benchmark_index(...).Field":"BenchmarkIndexDailyBar",
        "BenchmarkIndexDailyBar(...).Field":"BenchmarkIndexDailyBar",
        "market_ret":"BenchmarkIndexDailyBar.Return",
        "rolling_beta_to_market(ret,window)":"BenchmarkIndexDailyBar.Return",
        "tail_beta(ret,window,q)":"BenchmarkIndexDailyBar.Return",
        "residual_momentum_capm(ret,window)":"BenchmarkIndexDailyBar.Return",
        "industry_neutralize(x)":"IndustryDaily.IndustryCode",
        "size_neutralize(x)":"SizeDaily.MarketCap",
        "industry_size_neutralize(x)":"IndustryDaily.IndustryCode + SizeDaily.MarketCap",
        "real_turnover_rate()":"TurnoverBaseDaily",
        "asof(financial_field)":"StockIncome/StockCashFlow/StockBalance",
        "financial_lag(financial_field,q)":"StockIncome/StockCashFlow/StockBalance",
        "minute_at":"StockMinuteBar","minute_range":"StockMinuteBar","minute_bar":"StockMinuteBar",
        "minute_resample":"StockMinuteBar (intraday run only)",
        "intermediate(name,version)":"FactorLake + intermediate registry",
        "enterprise_value":"DerivedFieldRegistry (definition required)",
        "ebitda_approx":"DerivedFieldRegistry (definition required)",
    }
    configuration_required={
        "enterprise_value":"provide a versioned, hashed DerivedFieldRegistry expression; no finance formula is guessed",
        "ebitda_approx":"provide a versioned, hashed DerivedFieldRegistry expression; no finance formula is guessed",
        "intermediate(name,version)":"materialize existing dependency or register a synchronous backfill config",
        "real_turnover_rate()":"TurnoverBaseDaily LQTP mirror must be available",
    }
    recognized_blocked={
        "ts_sumac":"exact semantic definition absent from supplied LQTP manual",
        "ts_regression_slope_sequence":"exact semantic definition absent from supplied LQTP manual",
        "group_minmax":"named by manual but exact compatibility contract not installed",
        "group_quantile_mask":"named by manual but exact compatibility contract not installed",
        "neutralize(x)":"retired public name; use industry_neutralize / size_neutralize / industry_size_neutralize",
        "l2_sum":"requires StockTransaction/L2 source contract",
        "l2_sum_if":"requires StockTransaction/L2 source contract",
        "l2_count":"requires StockTransaction/L2 source contract",
        "l2_count_if":"requires StockTransaction/L2 source contract",
    }

    canonical={}
    for name in sorted(OperatorRegistry.list_canonical()):
        spec=build_operator_spec(name);cert=backend_certification(name)
        canonical[name]={
            "surface":getattr(spec,"status",None) if spec is not None else None,
            "production_allowed":bool(getattr(spec,"allow_in_production",False)) if spec is not None else False,
            "production_backends":[backend for backend,status in {
                "pandas_numpy":cert.pandas_numpy,"polars":cert.polars,"duckdb_sql":cert.duckdb_sql,
            }.items() if status=="production"],
        }
    return {
        "schema_version":"factor_engine.lqtp_compatibility.v2",
        "dialect":"lqtp","dialect_version":DEFAULT_LQTP_DIALECT_VERSION,
        "state_model":["recognized","parseable","executable","production_certified"],
        "exact_aliases":aliases,"macros":macros,"source_aware":sources,
        "configuration_required":configuration_required,
        "recognized_blocked":recognized_blocked,
        "canonical_operators":canonical,
    }
