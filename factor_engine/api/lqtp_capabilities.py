# -*- coding: utf-8 -*-
"""Machine-readable LQTP compatibility states.

The key distinction is intentional: recognized != executable != production.
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
        "decay_linear": "ts_decay_linear",
        "ts_rank_pct": "ts_rank",
        "ts_ewm_mean": "ts_ema",
        "ts_expanding_rank": "expanding_rank",
        "ts_hump_decay": "hump_decay",
        "sma(x,n)": "ts_mean",
        "sma(x,n,m)": "ts_sma_cn",
        "momentum": "ts_delta",
    }
    macros = ["safe_log", "nullif_zero", "clean", "historical_var"]
    sources = {
        "benchmark_index(...).Field": "BenchmarkIndexDailyBar",
        "BenchmarkIndexDailyBar(...).Field": "BenchmarkIndexDailyBar",
        "market_ret": "BenchmarkIndexDailyBar",
        "rolling_beta_to_market(ret,window)": "BenchmarkIndexDailyBar",
        "tail_beta(ret,window,q)": "BenchmarkIndexDailyBar",
        "residual_momentum_capm(ret,window)": "BenchmarkIndexDailyBar",
        "real_turnover_rate()": "TurnoverBaseDaily",
        "asof(financial_field)": "StockIncome/StockCashFlow/StockBalance",
        "financial_lag(financial_field,q)": "StockIncome/StockCashFlow/StockBalance",
        "minute_at": "StockMinuteBar",
        "minute_range": "StockMinuteBar",
        "minute_bar": "StockMinuteBar",
        "minute_resample": "StockMinuteBar (intraday run only)",
        "intermediate(name,version)": "FactorLake + intermediate registry",
    }
    recognized_blocked = {
        "ts_sumac": "exact semantic definition absent from supplied LQTP manual",
        "ts_regression_slope_sequence": "exact semantic definition absent from supplied LQTP manual",
        "enterprise_value": "derived-field definition/source absent; no guessed finance formula",
        "ebitda_approx": "derived-field definition/source absent; no guessed finance formula",
        "group_minmax": "named by manual but exact compatibility contract not installed",
        "group_quantile_mask": "named by manual but exact compatibility contract not installed",
        "l2_sum": "requires StockTransaction/L2 source contract",
        "l2_sum_if": "requires StockTransaction/L2 source contract",
        "l2_count": "requires StockTransaction/L2 source contract",
        "l2_count_if": "requires StockTransaction/L2 source contract",
    }

    canonical: dict[str, Any] = {}
    for name in sorted(OperatorRegistry.list_canonical()):
        spec = build_operator_spec(name)
        cert = backend_certification(name)
        canonical[name] = {
            "surface": getattr(spec, "status", None) if spec is not None else None,
            "production_allowed": bool(getattr(spec, "allow_in_production", False)) if spec is not None else False,
            "production_backends": [
                backend for backend, status in {
                    "pandas_numpy": cert.pandas_numpy,
                    "polars": cert.polars,
                    "duckdb_sql": cert.duckdb_sql,
                }.items() if status == "production"
            ],
        }

    return {
        "schema_version": "factor_engine.lqtp_compatibility.v2",
        "dialect": "lqtp",
        "dialect_version": DEFAULT_LQTP_DIALECT_VERSION,
        "state_model": ["recognized", "parseable", "executable", "production_certified"],
        "exact_aliases": aliases,
        "macros": macros,
        "source_aware": sources,
        "recognized_blocked": recognized_blocked,
        "canonical_operators": canonical,
    }
