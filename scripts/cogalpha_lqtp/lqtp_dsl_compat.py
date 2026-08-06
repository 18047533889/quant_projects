# -*- coding: utf-8 -*-
"""Detect / convert factor_engine DSL for LQTP RunFactor (2026-07 manual)."""
from __future__ import annotations

import re

# Operators that remain FE-only after Jul-2026 LQTP updates.
# Everything else either exists natively on LQTP or is rewritten by dsl_to_lqtp.
FE_ONLY_OPERATOR_PATTERNS: tuple[str, ...] = (
    r"\bADX\s*\(",
    r"\bATR\s*\(",
    r"\bRSI\s*\(",
    r"\bROC\s*\(",
    r"\bSMA\s*\(",
    # protected_div / cs_rank_gaussian / tanh / clip 等由 dsl_to_lqtp 改写后再判定
)


# Names accepted by LQTP after conversion (subset used for leftover scan).
_LQTP_KNOWN_CALLS = frozenset(
    {
        "abs", "log", "sqrt", "sign", "round", "signed_sqrt", "sigmoid", "power",
        "cap", "where", "iif", "is_null", "is_nan", "nan_to_num", "coalesce",
        "rank", "cs_rank", "zscore", "cs_zscore", "cs_demean", "scale", "winsorize",
        "cs_resid", "cs_regression",
        "ts_mean", "ts_sum", "ts_std", "ts_max", "ts_min", "ts_rank", "ts_delta",
        "ts_pct", "delay", "decay_linear", "ts_decay_linear", "ema", "EMA",
        "price_spread_deviation", "ts_corr", "ts_cov", "ts_regression_slope",
        "ts_quantile", "ts_skew", "ts_kurt", "ts_moment", "ts_topk_sum",
        "rank_corr", "rankcorr", "ts_rank_corr", "ts_poly2_coeff", "ts_poly2_resid",
        "digital_count", "ts_max_buildup", "ts_argmax", "ts_argmin",
        "group_mean", "group_demean", "group_rank", "group_zscore", "group_winsorize",
        "group_std", "group_minmax", "group_quantile_mask", "group_decay_linear",
        "market_ret", "historical_var", "historical_cvar",
        "benchmark_index", "rolling_beta_to_market", "downside_beta", "tail_beta",
        "residual_momentum_capm", "coskewness_to_market", "idio_vol", "idio_skew",
        "asof", "financial_lag", "lag", "ttm", "quarter", "yoy", "avg2",
        "industry_neutralize", "industry_neutral", "ind_neutralize",
        "size_neutralize", "market_cap_neutralize", "cap_neutralize", "neutralize",
        "size_industry_neutralize", "industry_size_neutralize",
        "minute_at", "minute_range", "minute_resample", "minute_bar",
        "l2_sum", "l2_sum_if", "l2_count", "l2_count_if", "real_turnover_rate",
        "safe_div", "nullif_zero", "safe_log", "clean", "ma", "sum_n", "std_n",
        "delta", "pct_change", "intermediate",
        # Handbook 2026-07-19: NO scalar min/max/minimum/maximum — use where(...).
        # NO ts_true_streak in operator tables (platform rejects). Use where for binary extrema.
        # Prefer power (not pow), cap (not clip), ts_quantile(...,0.5) (not ts_median).
    }
)


def fe_only_operators(dsl: str) -> list[str]:
    """Return FE-only operator names found in *dsl* (post or pre conversion)."""
    hits: list[str] = []
    for pattern in FE_ONLY_OPERATOR_PATTERNS:
        if re.search(pattern, dsl or ""):
            name = pattern.replace(r"\b", "").replace(r"\s*\(", "")
            hits.append(name)
    return hits


def unknown_lqtp_calls(dsl: str) -> list[str]:
    """Call names that are neither known LQTP ops nor fields/literals."""
    names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", dsl or "")
    skip = {
        "and", "or", "not", "if", "True", "False", "true", "false",
        "open", "high", "low", "close", "volume", "amount", "vwap", "ret",
        "pre_close", "factor",
    }
    out: list[str] = []
    for name in names:
        if name in skip or name in _LQTP_KNOWN_CALLS:
            continue
        if name not in out:
            out.append(name)
    return out


def is_lqtp_native_dsl(dsl: str) -> bool:
    """True when *dsl* can be passed to LQTP RunFactor after naming alignment."""
    text = (dsl or "").strip()
    if not text:
        return False
    # Multi-statement / assignment packs are not RunFactor-ready until expanded.
    if ";" in text:
        return False
    if re.search(r"\bnumerator\s*=|\bdenominator\s*=|\bspan\s*=", text):
        return False
    if re.search(r"\b(intraday_return|intraday_ret|daily_return)\b", text):
        return False
    if fe_only_operators(text):
        return False
    # Ban FE-only logic wrappers if still present.
    if re.search(r"\b(and_|or_|not_|clip|tanh|ts_delay|ts_ema|ewm_mean|protected_div|ewm)\s*\(", text):
        return False
    return not unknown_lqtp_calls(text)


def eval_route_for_entry(*, status: str, dsl: str) -> str:
    """Catalog eval route: lqtp_dsl | local_dsl | local_python."""
    if status in {"python", "hard"} or not (dsl or "").strip():
        return "local_python"
    if status == "ready" and is_lqtp_native_dsl(dsl):
        return "lqtp_dsl"
    return "local_dsl"
