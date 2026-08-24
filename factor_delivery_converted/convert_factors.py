#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor DSL Converter v4 — converts factor formulas from source DSL to:
  1. LQTP DSL  (lqtp-backtest 因子服务)
  2. factor_engine DSL (quant_projects cleaned_operators)

Handles:
  - Window-at-end: MA(x, d) → ts_mean(d, x)
  - Subscript syntax: EMA5(x) → ema(5, x), MA20(x) → ts_mean(20, x),
    mean_5d(x) → ts_mean(5, x), sum_20(x) → ts_sum(20, x)
  - Textual notation: I[...], R(...), C, E — marked as non-executable in LQTP
  - Pandas methods: ifnan, daily_std, daily_corr → preserved in code (not formula)
"""

import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass


# ─── Core operator table ────────────────────────────────────────────────────

@dataclass
class OpRule:
    lqtp: str
    fe: str
    swap: bool = False

OP_RULES = {
    # Time-series: window-first → window-last
    "MA":            OpRule("ts_mean",        "ts_mean",     True),
    "rolling_mean":  OpRule("ts_mean",        "ts_mean",     True),
    "rolling_avg":   OpRule("ts_mean",        "ts_mean",     True),
    "ewm":          OpRule("ema",            "ema",         True),
    "ewma":         OpRule("ema",            "ema",         True),
    "ewm_span":     OpRule("ema",            "ema",         True),
    "ts_ema":      OpRule("ema",            "ema",         True),
    "ts_mean":      OpRule("ts_mean",        "ts_mean",     True),
    "ts_std":       OpRule("ts_std",         "ts_std",      True),
    "ts_std_dev":   OpRule("ts_std",         "ts_std",      True),
    "rolling_std":  OpRule("ts_std",         "ts_std",      True),
    "ts_median":    OpRule("ts_median",      "ts_median",   True),
    "rolling_median":OpRule("ts_median",     "ts_median",   True),
    "rolling_max":  OpRule("ts_max",         "ts_max",      True),
    "rolling_min":  OpRule("ts_min",         "ts_min",      True),
    "ts_max":       OpRule("ts_max",         "ts_max",      True),
    "ts_min":       OpRule("ts_min",         "ts_min",      True),
    "ts_sum":       OpRule("ts_sum",         "ts_sum",      True),
    "rolling_sum":  OpRule("ts_sum",         "ts_sum",      True),
    "ts_skew":      OpRule("ts_skew",        "ts_skew",     True),
    "ts_kurt":      OpRule("ts_kurt",        "ts_kurt",     True),
    "ts_rank":      OpRule("ts_rank",        "ts_rank",     True),
    "ts_delta":     OpRule("ts_delta",       "ts_delta",    True),
    "ts_pct":       OpRule("ts_pct",         "ts_pct",      True),
    "ts_corr":      OpRule("ts_corr",        "ts_corr",     True),
    "ts_cov":       OpRule("ts_cov",         "ts_cov",      True),
    "ts_decay_linear":OpRule("ts_decay_linear","ts_decay_linear",True),
    "ts_decay_exp": OpRule("ts_decay_exp",   "ts_decay_exp",True),
    "ts_argmax":    OpRule("ts_argmax",      "ts_argmax",   True),
    "ts_argmin":    OpRule("ts_argmin",      "ts_argmin",   True),
    "ts_topk_sum":  OpRule("ts_topk_sum",   "ts_topk_sum", True),
    "ts_top_k_sum": OpRule("ts_topk_sum",   "ts_topk_sum", True),
    "ts_quantile":  OpRule("ts_quantile",    "ts_quantile", True),
    "ts_regression_slope":OpRule("ts_regression","ts_regression",True),
    # Cross-sectional: data-first, no swap
    "rank":         OpRule("cs_rank",        "rank",        False),
    "cs_rank":      OpRule("cs_rank",        "rank",        False),
    "zscore":       OpRule("cs_zscore",      "zscore",      False),
    "cs_zscore":    OpRule("cs_zscore",      "zscore",      False),
    "cs_demean":    OpRule("cs_demean",      "cs_demean",   False),
    "cs_resid":     OpRule("cs_resid",       "cs_resid",    False),
    "cs_regression":OpRule("cs_regression",  "cs_regression",False),
    "scale":        OpRule("scale",          "scale",       False),
    "winsorize":    OpRule("winsorize",      "winsorize",   False),
    "clip":         OpRule("clip",           "clip",        False),
    "cap":          OpRule("clip",           "clip",        False),
    # Group
    "group_mean":   OpRule("group_mean",    "group_mean",  False),
    "group_neutralize":OpRule("group_neutralize","group_neutralize",False),
    "group_rank":   OpRule("group_rank",    "group_rank",  False),
    "group_zscore": OpRule("group_zscore",  "group_zscore",False),
    "group_winsorize":OpRule("group_winsorize","group_winsorize",False),
    # Other
    "delay":        OpRule("delay",          "ts_delay",    False),
    "ts_delay":     OpRule("delay",          "ts_delay",    False),
    "ref":          OpRule("delay",          "ts_delay",    False),
    "delta":        OpRule("ts_delta",       "ts_delta",    False),
    "pct_change":   OpRule("ts_pct",        "ts_pct",      False),
    "abs":          OpRule("abs",            "abs",         False),
    "log":          OpRule("log",            "log",         False),
    "ln":           OpRule("log",            "log",         False),
    "sqrt":         OpRule("sqrt",           "sqrt",        False),
    "sign":         OpRule("sign",           "sign",        False),
    "where":        OpRule("where",          "where",       False),
    "iif":          OpRule("where",          "where",       False),
    "na_if":        OpRule("nan_to_num",     "nan_to_num",  False),
    "nan_to_num":   OpRule("nan_to_num",     "nan_to_num",  False),
    "is_nan":       OpRule("is_nan",         "is_nan",      False),
    "is_null":      OpRule("is_null",        "is_null",     False),
    "coalesce":     OpRule("coalesce",       "coalesce",    False),
    "signed_sqrt":  OpRule("signed_sqrt",    "signed_sqrt", False),
    "tanh":         OpRule("tanh",           "tanh",        False),
    "exp":          OpRule("exp",            "exp",         False),
    "pow":          OpRule("power",          "power",       False),
    "power":        OpRule("power",          "power",       False),
    "sigmoid":      OpRule("sigmoid",        "sigmoid",     False),
    # Financial
    "ttm":          OpRule("ttm",            "ttm",         False),
    "TTM":          OpRule("ttm",            "ttm",         False),
    "yoy":          OpRule("yoy",            "yoy",         False),
    "YOY":          OpRule("yoy",            "yoy",         False),
    "quarter":      OpRule("quarter",        "quarter",     False),
    "financial_lag":OpRule("financial_lag",  "financial_lag",False),
    "avg2":         OpRule("avg2",           "avg2",        False),
    "asof":         OpRule("asof",           "asof",        False),
    # Market risk
    "market_ret":   OpRule("market_ret",    "market_ret",  False),
    "rolling_beta_to_market":OpRule("rolling_beta_to_market","rolling_beta_to_market",False),
    "downside_beta":OpRule("downside_beta", "downside_beta",False),
    "tail_beta":    OpRule("tail_beta",      "tail_beta",   False),
    "residual_momentum_capm":OpRule("residual_momentum_capm","residual_momentum_capm",False),
    "idio_vol":     OpRule("idio_vol",        "idio_vol",    False),
    "coskewness_to_market":OpRule("coskewness_to_market","coskewness_to_market",False),
    "idio_skew":    OpRule("idio_skew",     "idio_skew",   False),
    # Neutralization
    "industry_neutralize":OpRule("industry_neutralize","industry_neutralize",False),
    "size_neutralize":OpRule("size_neutralize","size_neutralize",False),
    "neutralize":   OpRule("neutralize",     "neutralize",  False),
    # Technical
    "ts_rsi":       OpRule("ts_rsi",        "RSI",         False),
    "ts_macd":      OpRule("ts_macd",        "MACD",        False),
    "ts_atr":       OpRule("ts_atr",        "ATR",         False),
    "ts_adx":       OpRule("ts_adx",        "ADX",         False),
    "RSI":          OpRule("ts_rsi",        "RSI",         False),
    "MACD":         OpRule("ts_macd",        "MACD",        False),
    "ATR":          OpRule("ts_atr",        "ATR",         False),
    "ADX":          OpRule("ts_adx",        "ADX",         False),
    "SMA":          OpRule("ts_mean",        "ts_mean",     True),
    "EMA":          OpRule("ema",            "ema",         True),
    "EWMA":         OpRule("ema",            "ema",         True),
    "EWM":          OpRule("ema",            "ema",         True),
    "SUM":          OpRule("ts_sum",         "ts_sum",      False),
    "MEAN":         OpRule("ts_mean",        "ts_mean",     True),
    "STD":          OpRule("ts_std",         "ts_std",      True),
    "CORR":         OpRule("ts_corr",        "ts_corr",     True),
    "COV":          OpRule("ts_cov",         "ts_cov",      True),
    "ROC":          OpRule("ts_pct",         "ts_pct",      True),
    "RET":          OpRule("ret",            "ts_pct",      True),
    # Lowercase
    "mean":         OpRule("ts_mean",        "ts_mean",     True),
    "sum":          OpRule("ts_sum",         "ts_sum",      True),
    "std":          OpRule("ts_std",         "ts_std",      True),
    "max":          OpRule("max",            "flex_max",    False),
    "min":          OpRule("min",            "flex_min",    False),
    "corr":         OpRule("ts_corr",        "ts_corr",     True),
    "cov":          OpRule("ts_cov",         "ts_cov",      True),
    "roc":          OpRule("ts_pct",         "ts_pct",      True),
    "shift":        OpRule("delay",          "ts_delay",    False),
    "round":        OpRule("round",          "round",       False),
    "clip":         OpRule("clip",           "clip",        False),
    "winsorize":    OpRule("winsorize",      "winsorize",   False),
    "tanh":         OpRule("tanh",           "tanh",        False),
    "exp":          OpRule("exp",            "exp",         False),
    "pow":          OpRule("power",          "power",       False),
    "vol_ratio":   OpRule("<vol_ratio>",   "<vol_ratio>", False),
    "ifnan":       OpRule("<ifnan>",       "<ifnan>",    False),
    "rolling_rank": OpRule("ts_rank",       "ts_rank",    True),
    "rolling":     OpRule("<rolling>",     "<rolling>",  False),
    # Additional patterns from source formulas
    "sm5":         OpRule("ts_mean",        "ts_mean",     True),
    "sma":         OpRule("ts_mean",        "ts_mean",     False),  # sma(x) = ts_mean(x), no window
    "m_median":    OpRule("ts_median",     "ts_median",   True),
    "skewness":    OpRule("ts_skew",       "ts_skew",     True),
    "rank_pct":    OpRule("cs_rank",       "rank",        False),
    "daily_mean":  OpRule("<daily_mean>",   "<daily_mean>",False),
    "ts_skew":     OpRule("ts_skew",       "ts_skew",     True),
    "z":           OpRule("cs_zscore",    "zscore",     False),  # z(x) = zscore(x)
    "m_median":    OpRule("ts_median",    "ts_median",   True),
}


def _split_args(s: str) -> list[str]:
    args, depth, current = [], 0, ""
    for ch in s:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def _find_matching_paren(s: str, start: int) -> int:
    """Find closing paren for the '(' at position start (depth starts at 1)."""
    depth = 1
    for i in range(start + 1, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_all_subscript_calls(s):
    """Find all subscript-pattern function calls in string s.
    Returns list of (start, end, name, digits, inner) where:
    - start/end: slice in s
    - name: the function name (may include {digits} suffix)
    - digits: the numeric suffix if present (e.g. '20' from ts_rank_{20})
    - inner: the content between outer parens
    """
    results = []
    i = 0
    while i < len(s):
        if not s[i].isalpha():
            i += 1
            continue
        j = i
        # Include letters, digits, underscores, hyphens, and braces in names
        while j < len(s) and (s[j].isalnum() or s[j] in '_-{}'):
            j += 1
        name = s[i:j]
        # Extract {digits} suffix if present
        digits = None
        k = j
        while k < len(s) and s[k] in ' \t':
            k += 1
        if k < len(s) and s[k] == '{':
            k2 = k + 1
            while k2 < len(s) and s[k2].isdigit():
                k2 += 1
            if k2 > k + 1 and k2 < len(s) and s[k2] == '}':
                digits = s[k+1:k2]
                j = k2 + 1
        # Now expect '('
        while j < len(s) and s[j] in ' \t':
            j += 1
        if j < len(s) and s[j] == '(':
            paren_start = j
            depth = 0
            k = j + 1
            while k < len(s):
                if s[k] == '(':
                    depth += 1
                elif s[k] == ')':
                    if depth == 0:
                        inner = s[paren_start+1:k]
                        results.append((i, k+1, name, digits, inner))
                        i = k + 1
                        break
                    depth -= 1
                k += 1
            else:
                i += 1
        else:
            i += 1
    return results


def _convert_subscripts(s: str) -> str:
    """
    Convert subscript/decorated patterns: EMA5(x) → ema(5, x).
    Handles nested parentheses correctly.
    """
    matches = _find_all_subscript_calls(s)
    if not matches:
        return s

    # Build substitution dict
    subs = {}  # (start, end) → replacement string

    for start, end, name, digits, inner in matches:
        # Parse inner args
        args = _split_args(inner)

        # ── Handle {digits} suffix: ts_rank_{20}, rolling_mean_{20} ──
        if digits is not None:
            d = digits
            # ts_rank_{20}(x) → ts_rank(20, x)
            if name == 'ts_rank_':
                new_call = f"ts_rank({d}, {inner})"
                subs[(start, end)] = new_call
            elif name.startswith('rolling_mean_'):
                subs[(start, end)] = f"ts_mean({d}, {inner})"
            elif name.startswith('rolling_std_'):
                subs[(start, end)] = f"ts_std({d}, {inner})"
            elif name.startswith('rolling_max_'):
                subs[(start, end)] = f"ts_max({d}, {inner})"
            elif name.startswith('rolling_min_'):
                subs[(start, end)] = f"ts_min({d}, {inner})"
            continue

        # ── Handle NAME{N}(...) patterns ──
        # e.g. EMA5, SMA10, MA20, EWMA5, EWM10, ewm_span21
        for prefix, target in [
            ('EMA', 'ema'), ('SMA', 'ts_mean'), ('MA', 'ts_mean'),
            ('EWMA', 'ema'), ('EWM', 'ema'), ('ewm_span', 'ema'),
        ]:
            if name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+$', rest):
                    d = rest
                    # args[0] is the data expression
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    break

        # ── Handle NAME_DIGIT patterns: EMA_5, SMA_20 ──
        if '_' in name:
            parts = name.split('_')
            if len(parts) == 2 and re.match(r'^\d+$', parts[1]):
                prefix, d = parts
                if prefix in ('EMA', 'SMA', 'MA', 'EWMA', 'EWM'):
                    target = {'EMA': 'ema', 'SMA': 'ts_mean', 'MA': 'ts_mean',
                               'EWMA': 'ema', 'EWM': 'ema'}.get(prefix, 'ts_mean')
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    continue

        # ── Handle _N patterns without underscore: mean5, sum20 ──
        for prefix, target in [
            ('mean', 'ts_mean'), ('sum', 'ts_sum'),
            ('std', 'ts_std'), ('median', 'ts_median'),
        ]:
            if name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+$', rest):
                    d = rest
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    break

        # ── Handle _Nd suffix: mean_5d, sum_20d ──
        for prefix, target in [
            ('mean_', 'ts_mean'), ('sum_', 'ts_sum'),
            ('std_', 'ts_std'), ('median_', 'ts_median'),
        ]:
            if name.startswith(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+d?$', rest):
                    d = re.match(r'^(\d+)', rest).group(1)
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    break

        # ── shift{d}, lag{d} ──
        for prefix in ('shift', 'lag'):
            if name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+$', rest):
                    d = rest
                    if len(args) == 1:
                        subs[(start, end)] = f"delay({d}, {args[0].strip()})"
                    break

        # ── PLUS_DI, MINUS_DI ──
        if name == 'PLUS_DI' and len(args) == 1:
            subs[(start, end)] = f"<PLUS_DI({args[0].strip()})>"
        elif name == 'MINUS_DI' and len(args) == 1:
            subs[(start, end)] = f"<MINUS_DI({args[0].strip()})>"

        # ── percentile_rank(x) ──
        if name == 'percentile_rank' and len(args) == 1:
            subs[(start, end)] = f"cs_rank({args[0].strip()})"

        # ── robust_z(x), robust_zscore(x) ──
        if name in ('robust_z', 'robust_zscore') and len(args) >= 1:
            subs[(start, end)] = f"cs_zscore({inner})"

        # ── rolling_skew(x, d) ──
        if name == 'rolling_skew' and len(args) == 2:
            subs[(start, end)] = f"ts_skew({args[1].strip()}, {args[0].strip()})"

        # ── m_skew(x, d) ──
        if name == 'm_skew' and len(args) == 2:
            subs[(start, end)] = f"ts_skew({args[1].strip()}, {args[0].strip()})"

        # ── EMA_volume, MA_volume ──
        if name in ('EMA_volume', 'MA_volume') and len(args) == 1:
            subs[(start, end)] = f"<{name}({args[0].strip()})>"

        # ── ewts_mean(x, d) ──
        if name == 'ewts_mean' and len(args) == 2:
            subs[(start, end)] = f"ema({args[1].strip()}, {args[0].strip()})"

        # ── Max10, Min5, Median30 etc. ──
        for prefix, target in [('Max', 'ts_max'), ('Min', 'ts_min'), ('Median', 'ts_median')]:
            if name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+$', rest):
                    d = rest
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    break

        # ── max_10, min_5, rank_20 ──
        for prefix, target in [
            ('max_', 'ts_max'), ('min_', 'ts_min'), ('rank_', 'ts_rank'),
        ]:
            if name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if re.match(r'^\d+$', rest):
                    d = rest
                    if len(args) == 1:
                        subs[(start, end)] = f"{target}({d}, {args[0].strip()})"
                    break

    # Apply substitutions in reverse order (to preserve positions)
    result = s
    for (start, end), replacement in sorted(subs.items(), key=lambda x: -x[0][0]):
        result = result[:start] + replacement + result[end:]

    return result


def _lqtp_target_names() -> set:
    return {r.lqtp for r in OP_RULES.values()}

_LQTP_TARGETS = _lqtp_target_names()

def _convert_calls(s: str, target: str) -> tuple[str, list[str]]:
    notes = []
    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
    matches = []
    for m in re.finditer(pattern, s):
        name = m.group(1)
        paren_end = _find_matching_paren(s, m.end() - 1)
        if paren_end == -1:
            continue
        inner = s[m.end():paren_end]
        matches.append((m.start(), paren_end, name, inner))

    if not matches:
        return s, notes

    result = s
    for start, pend, name, inner in reversed(matches):
        if name not in OP_RULES:
            continue
        # Skip if name is already a target name (to avoid double-swap from subscript conversion)
        lqtp_targets = _LQTP_TARGETS
        if target == "lqtp" and name in lqtp_targets:
            continue
        # For FE target, skip FE canonical names too
        if target == "fe":
            fe_targets = {r.fe for r in OP_RULES.values()}
            if name in fe_targets:
                continue
        rule = OP_RULES[name]
        target_name = rule.lqtp if target == "lqtp" else rule.fe
        args = _split_args(inner)
        if rule.swap and len(args) >= 2:
            args = [args[1], args[0]] + args[2:]
        new_call = target_name + '(' + ', '.join(args) + ')'
        result = result[:start] + new_call + result[pend+1:]

    return result, notes


def _convert_formula(formula: str, target: str = "lqtp") -> tuple[str, list[str]]:
    if not formula or not formula.strip():
        return formula, []

    result = formula
    notes = []

    # Step 1: subscript/decorated patterns
    result = _convert_subscripts(result)

    # Step 2: function calls
    result, conv_notes = _convert_calls(result, target)
    notes.extend(conv_notes)

    # Step 3: check for textual/non-executable patterns
    textual = []
    if re.search(r'\bI\s*\[', result):
        textual.append("I[...]-notation (indicator function, use Python code)")
    if re.search(r'\bR\s*\(', result):
        textual.append("R(...)-notation (rank, use Python code)")
    if re.search(r'(?<![A-Za-z_])C(?![A-Za-z_0-9])', result):
        textual.append("C (close shorthand in text, not executable)")
    if re.search(r'\bdaily_std\b', result):
        textual.append("daily_std (pandas method, not LQTP executable)")
    if re.search(r'\bdaily_corr\b', result):
        textual.append("daily_corr (pandas method, not LQTP executable)")
    if re.search(r'\bdaily_skew\b', result):
        textual.append("daily_skew (pandas method, not LQTP executable)")
    if re.search(r'\bdaily_last\b', result):
        textual.append("daily_last (pandas method, not LQTP executable)")
    if re.search(r'\bifnan\b', result):
        textual.append("ifnan (pandas method, not LQTP executable)")
    if re.search(r'\bvwap_deviation\b', result):
        textual.append("vwap_deviation (custom function, not in LQTP)")
    if re.search(r'\blag\d+\b', result):
        textual.append("lagN (variable delay, use Python code)")
    if re.search(r'<PLUS_DI|<MINUS_DI', result):
        textual.append("PLUS_DI/MINUS_DI (ADX sub-component, not in LQTP)")
    if textual:
        notes.append("Textual non-executable: " + "; ".join(textual))

    return result, notes


def _known_functions() -> set:
    return set(OP_RULES.keys()) | {
        # LQTP/factor_engine builtins
        'max', 'min', 'abs', 'log', 'sqrt', 'sign', 'where', 'exp', 'tanh',
        'pow', 'round', 'coalesce', 'is_nan', 'is_null', 'clip', 'cap',
        'scale', 'winsorize', 'nan_to_num', 'signed_sqrt', 'flex_max', 'flex_min',
        # LQTP fields
        'DailyBar', 'StockDailyBar', 'StockMinuteBar', 'MinuteBar',
        'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap',
        'pre_close', 'open_close', 'open_close_return', 'ret',
        'real_turnover_rate',
        # LQTP time-series
        'ema', 'ts_mean', 'ts_sum', 'ts_std', 'ts_max', 'ts_min',
        'ts_rank', 'ts_delta', 'ts_pct', 'ts_corr', 'ts_cov',
        'ts_decay_linear', 'ts_decay_exp', 'ts_median',
        'ts_skew', 'ts_kurt', 'ts_quantile', 'ts_topk_sum',
        'ts_argmax', 'ts_argmin', 'ts_regression',
        'delay', 'rank', 'cs_rank', 'cs_zscore', 'cs_demean',
        'zscore', 'cs_resid', 'cs_regression',
        # LQTP group
        'group_mean', 'group_rank', 'group_neutralize', 'group_zscore',
        'group_winsorize', 'group_demean', 'group_std',
        # LQTP financial
        'ttm', 'yoy', 'quarter', 'avg2', 'asof', 'financial_lag',
        # LQTP market risk
        'market_ret', 'rolling_beta', 'rolling_beta_to_market',
        'downside_beta', 'tail_beta', 'residual_momentum_capm',
        'idio_vol', 'coskewness_to_market', 'idio_skew',
        'historical_var', 'historical_cvar',
        # LQTP neutralization
        'industry_neutralize', 'size_neutralize', 'neutralize',
        # LQTP minute
        'minute_at', 'minute_range', 'minute_resample', 'minute_bar',
        'l2_sum', 'l2_count',
        # factor_engine canonical
        'RSI', 'MACD', 'ATR', 'ADX', 'ADXR', 'CCI', 'MOM', 'OBV',
        'WMA', 'ALMA', 'Beta', 'ACF',
        # alpha_tools custom
        'amount_weighted_mean', 'volume_weighted_mean',
        'classify_volume_regime', 'decompose_overnight_intraday',
        # Pandas / numpy (in code)
        'pd', 'np', 'df', 'copy', 'astype', 'fillna', 'ffill',
        'rolling', 'ewm', 'shift', 'pct_change', 'rank',
        'groupby', 'cumsum', 'rename', 'replace',
        'mean', 'std', 'median', 'sum', 'count', 'min', 'max',
        'True', 'False', 'None', 'and', 'or', 'not', 'if', 'else',
        'def', 'return', 'lambda', 'len', 'range', 'enumerate', 'zip',
        'print', 'sorted', 'reversed', 'any', 'all',
        'factor_', 'factor_name', 'out', 'value', 'day',
        'StyleGate', 'StyleGates', 'minute_tools',
    }


def convert_factor(factor_json: dict) -> dict:
    formula = factor_json.get("formula") or ""
    code = factor_json.get("code") or ""
    name = factor_json.get("factor_name", "unknown")

    lqtp_formula, lqtp_notes = _convert_formula(formula, "lqtp")
    fe_formula, fe_notes = _convert_formula(formula, "fe")

    # Check for remaining unknown function calls in formula
    known = _known_functions()
    for converted, notes_key in [(lqtp_formula, 'lqtp_notes'), (fe_formula, 'fe_notes')]:
        funcs = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', converted))
        unconverted = funcs - known
        # Filter out common non-function patterns
        unconverted = {u for u in unconverted if u not in (
            'where', 'if', 'for', 'while', 'def', 'return', 'lambda',
            'sum', 'min', 'max', 'abs', 'round', 'pow', 'sorted',
            'reversed', 'map', 'filter', 'zip', 'range', 'enumerate',
            'list', 'dict', 'tuple', 'set', 'frozenset',
            'get', 'items', 'keys', 'values',
            'factor_', 'factor_name',
            'open', 'close',  # field names that look like functions
        )}
        if unconverted:
            if notes_key == 'lqtp_notes':
                lqtp_notes.append(f"Unconverted: {sorted(unconverted)}")
            else:
                fe_notes.append(f"Unconverted: {sorted(unconverted)}")

    return {
        "lqtp_formula": lqtp_formula,
        "fe_formula": fe_formula,
        "lqtp_notes": "; ".join(lqtp_notes),
        "fe_notes": "; ".join(fe_notes),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/sunhaiwei/factor_delivery_converted")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    index_path = "/tmp/factor_delivery/factor_delivery/all_factors_index.csv"
    rows = list(csv.DictReader(open(index_path)))

    results = []
    for r in rows:
        fp = "/tmp/factor_delivery/factor_delivery/" + r["file_path"].replace("\\", "/")
        if not os.path.exists(fp):
            continue
        factor = json.load(open(fp))
        conv = convert_factor(factor)
        results.append({
            "factor_name": r["factor_name"],
            "source": r["source"],
            "pool": r["pool"],
            "domain": r.get("domain_agent", ""),
            "ic": r.get("ic", ""),
            "rank_ic": r.get("rank_ic", ""),
            "source_formula": factor.get("formula") or "",
            "lqtp_formula": conv["lqtp_formula"],
            "fe_formula": conv["fe_formula"],
            "lqtp_notes": conv["lqtp_notes"],
            "fe_notes": conv["fe_notes"],
        })

    csv_path = os.path.join(output_dir, "converted_factors.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    has_notes = sum(1 for r in results if r["lqtp_notes"] or r["fe_notes"])
    print(f"Conversion complete: {len(results)} factors")
    print(f"  Fully converted (no notes): {len(results) - has_notes}")
    print(f"  With unconverted patterns:  {has_notes}")
    print(f"  Output: {csv_path}")

    src_counts = Counter(r["source"] for r in results)
    print("\nBy source:")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src}: {cnt}")

    summary = {
        "total": len(results),
        "fully_converted": len(results) - has_notes,
        "with_unconverted": has_notes,
        "by_source": dict(src_counts),
    }
    summary_path = os.path.join(output_dir, "conversion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
