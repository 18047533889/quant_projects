#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python code → factor_engine DSL 真实转换器（替换详情页假「DSL 语法近似」）。

核心
----
- 读取 formula_lqtp_all.json 的 dsl 字段，把**纯表达式型** LQTP DSL 编译为
  factor_engine 可执行的 ``Factor(name=..., expr=o[...](...))`` 构建代码。
- 非表达式（自然语言/赋值链/数学下标/分支）→ can_use_factor_engine=False。
- 只更新 can_use_factor_engine / fe_formula 两个字段。

验证
----
--smoke：随机抽 N 个可转因子，用 factor_engine 落值（2018-06-01..2019-03-01，
取末 5 天），与 factor_matrices_all/ 对拍 3 只股票 5 天，|corr|>0.95 通过。

用法
----
.venv/bin/python jobs/fe_code_translator.py [--convert] [--smoke] [--sample N] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
LQTP_PATH = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
LOG_PATH = Path("/tmp/agent_dsl_conv.log")

os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.join(os.path.expanduser("~"), "cos_data"))
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")
for _p in (str(PROJECT), str(PROJECT / "jobs"),
           str(PROJECT / "scripts" / "archive" / "jobs"),
           str(PROJECT / "vectorbt_qs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# =====================================================================
# 1. 算子映射
# =====================================================================
DIRECT_OPS = {
    "abs": "abs", "clip": "clip", "exp": "exp", "log": "log", "sign": "sign",
    "sigmoid": "sigmoid", "tanh": "tanh", "sqrt": "sqrt", "signed_sqrt": "signed_sqrt",
    "power": "power", "maximum": "maximum", "minimum": "minimum",
    "ts_mean": "ts_mean", "ts_std": "ts_std", "ts_rank": "ts_rank",
    "ts_corr": "ts_corr", "ts_delay": "ts_delay", "ts_sum": "ts_sum",
    "ts_max": "ts_max", "ts_min": "ts_min", "ts_median": "ts_median",
    "ts_skew": "ts_skew", "ts_zscore": "ts_zscore", "ts_pct": "ts_pct",
    "ts_delta": "ts_delta", "ts_kurt": "ts_kurt", "ts_mad": "ts_mad",
    "ts_product": "ts_product", "ts_argmax": "ts_argmax", "ts_argmin": "ts_argmin",
    "ts_decay_linear": "ts_decay_linear", "ts_quantile": "ts_quantile",
    "rank": "rank", "zscore": "zscore", "cs_rank": "rank", "cs_zscore": "zscore",
    "where": "where", "delay": "ts_delay", "delta": "ts_delta",
    "safe_div_null": "safe_div_null",
    "ts_ema": "ts_ema", "ts_ewm_std": "ts_ewm_std", "ts_ewm_var": "ts_ewm_var",
    "ts_max_drawdown": "ts_max_drawdown", "cs_weighted_mean": "cs_weighted_mean",
    "true_range": "true_range", "ADX": "ADX", "ATR_WILDER": "ATR_WILDER",
    "RSI_WILDER": "RSI_WILDER", "NATR": "NATR", "atr_pct": "atr_pct",
}

OP_ALIAS = {
    "ewm_mean": "ts_ema", "ewm_std": "ts_ewm_std", "ewm_var": "ts_ewm_var",
    "EWM_std": "ts_ewm_std", "corr": "ts_corr", "correlation": "ts_corr",
    "rolling_corr": "ts_corr", "skew": "ts_skew", "cummax": "expanding_max",
    "rolling_max": "ts_max", "rolling_mean": "ts_mean", "rolling_std": "ts_std",
    "rolling_median": "ts_median", "rolling_skew": "ts_skew",
    "rolling_zscore": "ts_zscore", "max": "ts_max", "min": "ts_min",
    "std": "ts_std", "mean": "ts_mean", "median": "ts_median",
    "sum": "ts_sum", "sma": "ts_mean", "lag": "ts_delay", "shift1": "ts_delay",
    "z_score": "zscore", "m_zscore": "zscore", "m_skew": "ts_skew",
    "rank_ts": "ts_rank", "rank_trailing": "ts_rank",
    "ema_10": "ts_ema",
    "r": "ts_delta",
}

RECIPE_OPS = {
    "ts_atr": "ts_mean(true_range({h}, {l}, {c}), {w})",
    "ATR": "ts_mean(true_range({h}, {l}, {c}), {w})",
    "ATR90": "ts_mean(true_range({h}, {l}, {c}), {w})",
    "ATR20": "ts_mean(true_range({h}, {l}, {c}), 20)",
    "ATR14": "ts_mean(true_range({h}, {l}, {c}), 14)",
    "ATR21": "ts_mean(true_range({h}, {l}, {c}), 21)",
    "ATR_20": "ts_mean(true_range({h}, {l}, {c}), 20)",
    "ATR_14": "ts_mean(true_range({h}, {l}, {c}), 14)",
    "ATR_21": "ts_mean(true_range({h}, {l}, {c}), 21)",
    "ATR5": "ts_mean(true_range({h}, {l}, {c}), 5)",
    "ts_rsi": "100 - 100 / (1 + safe_div_null(ts_mean(maximum(ts_delta({x}, 1), 0), {w}), ts_mean(maximum(-ts_delta({x}, 1), 0), {w})))",
    "RSI": "100 - 100 / (1 + safe_div_null(ts_mean(maximum(ts_delta({x}, 1), 0), {w}), ts_mean(maximum(-ts_delta({x}, 1), 0), {w})))",
    "ts_roc": "100 * ts_pct({x}, {w})",
    "ROC": "100 * ts_pct({x}, {w})",
    "roc": "100 * ts_pct({x}, {w})",
    "max_drawdown": "ts_max_drawdown({x}, {w})",
    "rolling_vwap": "safe_div_null(ts_mean(col('AdjAmount'), {w}), ts_mean(col('Volume'), {w}))",
    "volatility": "ts_std(ts_pct(col('close'), 1), {w})",
    "vwap": "safe_div_null(ts_mean(col('AdjAmount'), {w}), ts_mean(col('Volume'), {w}))",
    "SMA10": "ts_mean({x}, 10)",
    "SMA20": "ts_mean({x}, 20)",
    "SMA_10": "ts_mean({x}, 10)",
    "SMA_5": "ts_mean({x}, 5)",
    "SMA_21": "ts_mean({x}, 21)",
    "SMA90": "ts_mean({x}, 90)",
    "SMA60": "ts_mean({x}, 60)",
    "EMA20": "ts_ema({x}, 20)",
    "EMA50": "ts_ema({x}, 50)",
    "EMA5": "ts_ema({x}, 5)",
    "EMA200": "ts_ema({x}, 200)",
    "EMA_volume": "ts_ema({x}, {w})",
    "MA_volume": "ts_mean({x}, {w})",
    "MA20": "ts_mean({x}, 20)",
    "MA30": "ts_mean({x}, 30)",
    "EWMA_up_vol": "ts_ema({x}, {w})",
    "EWMA_down_vol": "ts_ema({x}, {w})",
    "vol_ratio": "safe_div_null({x}, ts_mean({x}, {w}))",
    "vol_ratio_EMA": "safe_div_null({x}, ts_ema({x}, {w}))",
    "up_vol": "ts_std(maximum(ts_pct(col('close'), 1), 0), {w})",
    "down_vol": "ts_std(maximum(-ts_pct(col('close'), 1), 0), {w})",
    "upside_vol": "ts_std(maximum(ts_pct(col('close'), 1), 0), {w})",
    "downside_vol": "ts_std(maximum(-ts_pct(col('close'), 1), 0), {w})",
    "up_vol_60": "ts_std(maximum(ts_pct(col('close'), 1), 0), 60)",
    "down_vol_60": "ts_std(maximum(-ts_pct(col('close'), 1), 0), 60)",
    "up_vol_30": "ts_std(maximum(ts_pct(col('close'), 1), 0), 30)",
    "down_vol_30": "ts_std(maximum(-ts_pct(col('close'), 1), 0), 30)",
    "Z": "zscore({x})",
    "drawdown": "ts_max_drawdown({x}, {w})",
    "rank_corr": "rank_corr({x}, {y}, {d})",
    "daily_mean": "ts_mean({x}, 390)",
    "get": "identity({x})",
    "m_skew": "ts_skew({x}, 20)",
    "m_zscore": "ts_zscore({x}, 20)",
    "d": "ts_delta({x}, 1)",
}

COL_MAP = {
    "close": "AdjClose", "open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
    "volume": "Volume", "amount": "AdjAmount", "vwap": "AdjVwap", "ret": "Return",
    "vol": "Volume",
    "C": "AdjClose", "V": "Volume", "O": "AdjOpen", "H": "AdjHigh", "L": "AdjLow",
    "close_price": "AdjClose", "AdjClose": "AdjClose", "AdjOpen": "AdjOpen",
    "AdjHigh": "AdjHigh", "AdjLow": "AdjLow", "AdjVwap": "AdjVwap",
    "AdjAmount": "AdjAmount", "Volume": "Volume", "Return": "Return",
    "Close": "AdjClose",
}

UNSUPPORTED_MARKERS = [
    "MYDSL", "group_weighted_mean", "daily_corr", "daily_kurt", "daily_std",
    "daily_skew", "daily_last", "classify_volume_regime",
    "consensus", "trailing_max", "trailing_vol",
    "rolling20", "rolling_40_std", "rolling_20_corr",
    "rolling_20_mean", "rolling_20_median", "median_vol", "mean20", "max20",
    "sum_20", "std_20", "lag4", "normalized", "persistence",
    "directional_efficiency", "vwap_deviation", "regime_adaptive_trailing_high",
    "replace", "ifnan", "ATR_EMA90", "EMA_gated", "Corr20",
    "Cov_40", "Cov_60", "Var_40", "Var_60", "Rng", "RollingStd60",
    "MINUS_DI", "PLUS_DI", "CV", "I", "binary", "smoothed", "days",
    "close_std", "neg_std", "pos_std", "sigma_down", "sigma_up",
    "downside_ratio", "downside_stress_ratio", "score", "robust_z",
    "max_10", "max_20", "mean_5", "mean_10", "mean_20",
    "median_20", "tr", "E20", "E30",
    "ts_adx", "ts_adxr", "ts_rsi_wilder", "ts_aroon", "ts_cci", "ts_mom",
    "ts_obv", "ts_trix", "ts_willr", "ts_stoch", "ts_bbands",
]

# 后复权单表可物理读取的列；其余需要多表 composite 或中间派生
ADJ_SINGLE_TABLE_COLS = {
    "close", "open", "high", "low", "volume", "amount", "vwap", "ret",
    "AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "AdjVwap", "AdjAmount",
    "Volume", "Return", "Factor",
}

MULTI_TABLE_OR_INTERMEDIATE = {
    "free_turn", "pb_lf", "pe_ttm", "ps_ttm", "pcf_ocf_ttm", "mkt_cap_float",
    "roe_ttm2", "roa2_ttm2", "free_float_shares", "qfa_yoygr",
    "forecast_incap_chgr_mid", "debttoassets",
    "DD", "ATR", "ATR20", "ATR14", "ATR21", "ATR_20", "ATR_14", "ATR_21",
    "ATR5", "ATR_20d", "mom", "delta_depth_10", "down_vol_sum_40",
    "up_vol_sum_40", "volume_tail_asymmetry", "drawdown_depth_50",
    "downside_damage_12", "duration_20", "depth_20", "vol_down", "vol_up",
    "efficiency", "volume_momentum", "hold_persistence", "res20",
    "down_vol_60", "up_vol_60", "ret_5d", "ret_5", "pct_return_5",
    "price_ret_5", "vol_ma20", "vol_ma60", "vol_ma40", "vol_ma_20",
    "vol_ma_10", "transition_into_high_volume", "close_return",
    "overnight_ret", "intraday_ret", "close_ret", "overnight_return",
    "intraday_return", "close_10d_chg", "close_10d_return",
    "rolling_20d_mean_volume", "range_ratio_z", "sym_z", "avg_abs_ret",
    "vol_z", "norm_pressure", "net_dir", "herding", "shift5", "sma_20",
    "sma_30", "ma_vol21", "close_5d_ago", "range20", "median_20",
    "mean_16", "lower_wick", "close_location",
    "ema_imbalance", "std_imbalance", "MA20_V", "ma_volume_20",
    "pos_ret", "neg_ret", "neg_returns", "pos_returns", "up_range",
    "down_range", "total_range", "down_ret", "up_ret", "down_avg",
    "up_avg", "avg_down_range", "avg_up_range", "close_pct_change",
    "log_return", "pct_change", "daily_ret", "vol_pct", "ret_daily",
    "vol_chg", "symmetry", "vol_trend", "vol_rank", "close_loc",
    "close_loc_t2", "R", "B", "F", "S", "H20", "H20_shift", "CL",
    "TR", "typical", "momentum", "short_trend", "long_trend",
    "smooth_corr", "asym_ratio", "scale", "blend", "drawdown",
    "drawdown_depth", "drawdown_duration", "normalized_drawdown",
    "exp_ratio", "stress", "weight", "tight", "dry", "signal_t",
    "vol_20d", "vol_60d", "resistance", "resistance_shift",
    "prior_bar_close", "bar_low", "prior_15", "shift_close_5",
    "lag_ret", "lagged_ret", "ret_10", "pos_ret_10", "downside",
    "upside", "overnight_downside", "intraday_downside",
    "downside_energy_share_20", "lower", "upper", "midpoint",
    "typical_price", "size_dev", "cheapness", "roe", "20d", "resvol",
    "vol_regime", "low_volume_regime", "high_volume_regime", "vol_ratio",
    "vol_ratio_20", "vol_ratio_30", "vol_ratio_40", "vol_ratio_60",
    "vol_ratio_EMA", "close_ma20", "close_ma60", "close_ma200",
    "close_std20", "close_std60", "range_20", "range_60",
    "resvol_high", "gate", "conditional", "window", "Range",
    "close_return", "TradingDay", "epsilon",
    "over", "day", "filled", "on", "forward", "direction", "returns",
    "r", "cumulative", "return", "get", "d", "classify_volume_regime",
    "classify_volume_regime_multiplier", "style_gate_resvol_high",
    "style_gate_size_large", "style_gate_size_small",
    "style_gate_liquidity_high", "style_gate_momentum_high",
    "prior_high", "prior_low", "prior_close", "prior_high5",
    "prior_high20", "prior_low20", "prior_high_shift", "prior_low_shift",
    "atr_20", "atr_14", "atr_21", "close_std", "std_20", "vol_std_20",
    "vol_ma30", "vol_ma40", "vol_ma60", "vol_ma20", "vol_ma",
    "vol_ma_20", "vol_ma_10", "vol_ma_5", "close_5d_ago",
    "EWMA_up_vol", "EWMA_down_vol", "up_vol", "down_vol",
    "upside_vol", "downside_vol", "down_ret", "up_ret", "neg_ret",
    "pos_ret",
}

_FE_OPS: set[str] = set()


def _fe_op(name: str) -> str | None:
    if name in DIRECT_OPS:
        return DIRECT_OPS[name]
    if name in OP_ALIAS:
        return OP_ALIAS[name]
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name in _FE_OPS else None


def _init_fe_ops() -> None:
    global _FE_OPS
    try:
        from factor_engine.cleaned_operators import REGISTRY_BOOTSTRAP
        try:
            REGISTRY_BOOTSTRAP.ensure_ready(include_research=True)
        except Exception:
            pass
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        ops, aliases, catalog = OperatorRegistry._read_state()
        _FE_OPS = set(ops.keys())
    except Exception:
        _FE_OPS = set()


_KNOWN_FIELDS: set[str] = set()


def _init_known_fields() -> None:
    global _KNOWN_FIELDS
    _KNOWN_FIELDS = set(COL_MAP) | set(ADJ_SINGLE_TABLE_COLS) | set(MULTI_TABLE_OR_INTERMEDIATE)
    _KNOWN_FIELDS |= {"factor", "signal", "close_price"}


class DslParseError(ValueError):
    pass


_NUM = r"\d+(?:\.\d+)?"
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def _is_convertible_dsl(dsl: str) -> tuple[bool, str]:
    s = dsl.strip()
    if not s:
        return False, "empty dsl"
    if "MYDSL" in s:
        return False, "MYDSL"
    if re.search(r"[α-ωσμΔΣσ]", s):
        return False, "greek-symbol"
    if re.search(r"\{|_\{|\]\(|\)\[|\[[^\]]*\(|\bmax_\{", s):
        return False, "mathjax-index"
    if ";" in s or re.search(r"(?<![<>!=])=(?!=)", s):
        return False, "assignment-or-semicolon"
    if re.search(r"\b(if|else|where|when|gated|then)\b", s, re.I) or "?" in s:
        return False, "branch"
    if re.search(r"over\s+\d+\s+days|smoothed|with\s+(window|span|volume)|rolling\s+\d+", s, re.I):
        return False, "natural-language"
    if re.search(r"(\bof\b|\bthe\b|\band\b|\bto\b|\bfor\b|\bfrom\b)", s, re.I):
        return False, "natural-language-words"
    if "<" in s or ">" in s:
        stripped = re.sub(r"<([^<>]*)>", r"\1", s)
        if "<" in stripped or ">" in stripped:
            return False, "angle-template"
        s = stripped
    fn_names = {m.group(1) for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)", s)}
    attr_fns = {m.group(2) for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)", s)}
    fn_names |= attr_fns
    unsupported = [f for f in fn_names if f in UNSUPPORTED_MARKERS]
    if unsupported:
        return False, f"unsupported-op:{sorted(unsupported)[:6]}"
    for f in fn_names:
        if f in DIRECT_OPS or f in OP_ALIAS or f in RECIPE_OPS:
            continue
        if _fe_op(f) is not None:
            continue
        if f in COL_MAP or f in _KNOWN_FIELDS:
            continue
        if f in ("C", "V", "O", "H", "L"):
            continue
        if f in ("Z", "volatility", "drawdown", "rank_corr", "daily_mean",
                 "get", "m_skew", "m_zscore", "d"):
            continue
        return False, f"unknown-op:{f}"
    bare = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", s)) - fn_names
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\.", s):
        bare.discard(m.group(1))
    for b in bare:
        if b in ("factor", "signal", "x", "y", "w", "h", "l", "c", "v", "o", "t", "i", "n"):
            continue
        if b in COL_MAP or b in _KNOWN_FIELDS:
            continue
        if b in DIRECT_OPS or b in OP_ALIAS or b in RECIPE_OPS:
            continue
        if re.fullmatch(_NUM, b):
            continue
        return False, f"unknown-bare:{b}"
    return True, ""


class _DslCompiler:
    def __init__(self) -> None:
        self._pos = 0

    def compile(self, dsl: str) -> str:
        s = dsl.strip()
        if "<" in s or ">" in s:
            s = re.sub(r"<([^<>]*)>", r"\1", s)
        neg = False
        if s.startswith("-(") and _wrapped_in_outer_paren(s[1:]):
            neg = True
            s = s[1:][1:-1].strip()
        elif s.startswith("-"):
            neg = True
            s = s[1:].strip()
        node = self._parse_expr(s)
        out = self._emit(node)
        if neg:
            out = f"o['neg']({out})"
        return out

    def _tokenize(self, s: str) -> list[str]:
        toks: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
                j = i
                while j < n and (s[j].isdigit() or s[j] == "."):
                    j += 1
                toks.append(s[i:j])
                i = j
                continue
            if ch == "'":
                j = i + 1
                while j < n and s[j] != "'":
                    j += 1
                if j >= n:
                    raise DslParseError("unterminated string literal")
                toks.append(s[i + 1:j])
                i = j + 1
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                toks.append(s[i:j])
                i = j
                continue
            if ch in "+-*/^%()<>=!|&,.":
                toks.append(ch)
                i += 1
                continue
            raise DslParseError(f"unexpected char {ch!r}")
        return toks

    def _parse_expr(self, s: str):
        toks = self._tokenize(s)
        self._pos = 0
        node = self._parse_or(toks)
        if self._pos < len(toks):
            raise DslParseError(f"trailing tokens: {toks[self._pos:]}")
        return node

    def _peek(self, toks):
        return toks[self._pos] if self._pos < len(toks) else None

    def _parse_or(self, toks):
        left = self._parse_and(toks)
        while self._peek(toks) == "|":
            self._pos += 1
            left = ("or_", left, self._parse_and(toks))
        return left

    def _parse_and(self, toks):
        left = self._parse_cmp(toks)
        while self._peek(toks) == "&":
            self._pos += 1
            left = ("and_", left, self._parse_cmp(toks))
        return left

    def _parse_cmp(self, toks):
        left = self._parse_add(toks)
        while True:
            op = self._peek(toks)
            if op in ("<", ">", "<=", ">=", "==", "=", "!="):
                self._pos += 1
                right = self._parse_add(toks)
                opmap = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge",
                         "==": "eq", "=": "eq", "!=": "ne"}
                left = (opmap[op], left, right)
            else:
                break
        return left

    def _parse_add(self, toks):
        left = self._parse_mul(toks)
        while self._peek(toks) in ("+", "-"):
            op = self._peek(toks)
            self._pos += 1
            right = self._parse_mul(toks)
            left = ("add" if op == "+" else "subtract", left, right)
        return left

    def _parse_mul(self, toks):
        left = self._parse_power(toks)
        while self._peek(toks) in ("*", "/", "%"):
            op = self._peek(toks)
            self._pos += 1
            right = self._parse_power(toks)
            left = {"*": "multiply", "/": "divide", "%": "mod"}[op], left, right
        return left

    def _parse_power(self, toks):
        left = self._parse_unary(toks)
        if self._peek(toks) == "^":
            self._pos += 1
            return ("power", left, self._parse_power(toks))
        return left

    def _parse_unary(self, toks):
        tok = self._peek(toks)
        if tok == "-":
            self._pos += 1
            return ("neg", self._parse_unary(toks))
        if tok == "+":
            self._pos += 1
            return self._parse_unary(toks)
        if tok == "!":
            self._pos += 1
            return ("not_", self._parse_unary(toks))
        if tok == "|":
            self._pos += 1
            inner = self._parse_add(toks)
            if self._peek(toks) != "|":
                raise DslParseError("missing closing |")
            self._pos += 1
            return ("abs", inner)
        return self._parse_primary(toks)

    def _parse_primary(self, toks):
        tok = self._peek(toks)
        if tok is None:
            raise DslParseError("unexpected end")
        if tok == "(":
            self._pos += 1
            node = self._parse_or(toks)
            if self._peek(toks) != ")":
                raise DslParseError("missing )")
            self._pos += 1
            return node
        if tok.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", tok or ""):
            self._pos += 1
            return ("lit", float(tok))
        if re.fullmatch(_IDENT, tok):
            self._pos += 1
            if self._peek(toks) == "(":
                self._pos += 1
                args = []
                if self._peek(toks) != ")":
                    while True:
                        args.append(self._parse_or(toks))
                        if self._peek(toks) == ",":
                            self._pos += 1
                            continue
                        break
                if self._peek(toks) != ")":
                    raise DslParseError("missing ) in call")
                self._pos += 1
                return ("call", tok, tuple(args))
            if self._peek(toks) == ".":
                self._pos += 1
                m = self._peek(toks)
                if m is None or not re.fullmatch(_IDENT, m):
                    raise DslParseError(f"bad attribute chain on {tok!r}")
                self._pos += 1
                if self._peek(toks) != "(":
                    raise DslParseError(f"expected ( after .{m}")
                self._pos += 1
                args = [("col", tok)]
                if self._peek(toks) != ")":
                    while True:
                        args.append(self._parse_or(toks))
                        if self._peek(toks) == ",":
                            self._pos += 1
                            continue
                        break
                if self._peek(toks) != ")":
                    raise DslParseError("missing ) in attr call")
                self._pos += 1
                return ("call", m, tuple(args))
            if tok in COL_MAP or tok in _KNOWN_FIELDS:
                return ("col", tok)
            if tok in ("drawdown", "volatility"):
                return ("call", tok, ())
            if tok in RECIPE_OPS:
                return ("call", tok, (("lit", 20.0),) if tok in ("MA20", "SMA20", "EMA20", "MA30", "SMA10", "SMA90", "SMA60", "SMA5", "SMA21", "EMA50", "EMA200", "EMA5", "MA_volume") else ())
            if tok in ("epsilon", "pi", "e"):
                return ("lit", {"epsilon": 1e-6, "pi": np.pi, "e": np.e}[tok])
            raise DslParseError(f"unknown identifier {tok!r}")
        raise DslParseError(f"unexpected token {tok!r}")

    def _emit(self, node) -> str:
        kind = node[0]
        if kind == "lit":
            v = node[1]
            return str(int(v)) if float(v).is_integer() else repr(float(v))
        if kind == "col":
            name = node[1]
            fe_col = COL_MAP.get(name, name)
            if fe_col == "range":
                return f"o['subtract'](col('high'), col('low'))"
            if fe_col == "lag":
                return f"o['ts_delay'](col('close'), 1)"
            if fe_col == "factor":
                return f"o['subtract'](o['divide'](col('close'), o['ts_delay'](col('close'))), 1)"
            return f"col('{fe_col}')"
        if kind == "neg":
            return f"o['neg']({self._emit(node[1])})"
        if kind == "abs":
            return f"o['abs']({self._emit(node[1])})"
        if kind in ("add", "subtract", "multiply", "divide", "mod", "power"):
            return f"o['{kind}']({self._emit(node[1])}, {self._emit(node[2])})"
        if kind in ("lt", "gt", "le", "ge", "eq", "ne", "and_", "or_", "not_"):
            if len(node) == 3:
                return f"o['{kind}']({self._emit(node[1])}, {self._emit(node[2])})"
            return f"o['{kind}']({self._emit(node[1])})"
        if kind == "call":
            return self._emit_call(node[1], node[2])
        raise DslParseError(f"cannot emit {kind}")

    def _emit_call(self, fn: str, args) -> str:
        n = len(args)
        if fn in ("C", "V", "O", "H", "L") and n == 1:
            base = {"C": "close", "V": "Volume", "O": "open", "H": "high", "L": "low"}[fn]
            k = args[0]
            if isinstance(k, tuple) and k and k[0] == "neg" and k[1][0] == "lit":
                return f"o['ts_delay'](col('{base}'), {abs(int(k[1][1]))})"
            return f"o['ts_delay'](col('{base}'), {self._emit(k)})"
        if fn in ("ewm_mean", "ts_ema", "ewm_std", "ewm_var", "ts_ewm_std", "ts_ewm_var") and n == 1:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                canon = "ts_ema" if fn in ("ewm_mean", "ts_ema") else ("ts_ewm_std" if "std" in fn else "ts_ewm_var")
                return f"o['{canon}'](col('close'), {int(a0[1])})"
        if fn in ("ewm_mean", "ts_ema", "ewm_std", "ewm_var", "ts_ewm_std", "ts_ewm_var") and n == 2:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                canon = "ts_ema" if fn in ("ewm_mean", "ts_ema") else ("ts_ewm_std" if "std" in fn else "ts_ewm_var")
                return f"o['{canon}']({self._emit(args[1])}, {int(a0[1])})"
        if fn in ("ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_median",
                  "ts_skew", "ts_zscore", "ts_max", "ts_min", "ts_kurt",
                  "ts_mad", "ts_quantile", "ts_product", "ts_argmax",
                  "ts_argmin", "ts_decay_linear") and n == 1:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                return f"o['{_fe_op(fn)}'](col('close'), {int(a0[1])})"
        if fn in ("MA_volume", "SMA_10", "SMA20", "SMA_20", "MA20", "MA30", "SMA_5", "SMA_21", "SMA_10") and n == 1:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                series = "Volume" if "vol" in fn.lower() else "close"
                return f"o['ts_mean'](col('{series}'), {int(a0[1])})"
        if fn in ("MA_volume", "SMA_10", "SMA20", "SMA_20", "MA20", "MA30") and n == 2:
            a1 = args[1]
            if isinstance(a1, tuple) and a1 and a1[0] == "lit":
                series = "Volume" if "vol" in fn.lower() else "close"
                return f"o['ts_mean'](col('{series}'), {int(a1[1])})"
        if fn in ("down_vol", "up_vol", "downside_vol", "upside_vol", "down_vol_60", "up_vol_60", "down_vol_30", "up_vol_30") and n == 1:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                w = int(a0[1])
                neg = "-" if fn.startswith("down") else ""
                return f"o['ts_std'](o['maximum']({neg}o['ts_pct'](col('close'), 1), 0), {w})"
        if fn in ("ts_atr", "ATR", "ATR90", "ATR20", "ATR14", "ATR21", "ATR_20", "ATR_14", "ATR_21", "ATR5", "true_range") and n == 1:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                w = int(a0[1])
                return f"o['ts_mean'](o['true_range'](col('high'), col('low'), col('close')), {w})"
        if fn == "rank_corr" and n == 3:
            return f"o['rank_corr']({self._emit(args[0])}, {self._emit(args[1])}, {self._emit(args[2])})"
        if fn in RECIPE_OPS:
            tpl = RECIPE_OPS[fn]
            if n == 0:
                if fn in ("MA20", "SMA20", "EMA20", "MA30", "SMA10", "SMA90", "SMA60", "SMA5", "SMA21", "EMA50", "EMA200", "EMA5", "MA_volume"):
                    w = {"MA20": 20, "SMA20": 20, "EMA20": 20, "MA30": 30, "SMA10": 10, "SMA90": 90, "SMA60": 60, "SMA5": 5, "SMA21": 21, "EMA50": 50, "EMA200": 200, "EMA5": 5, "MA_volume": 20}[fn]
                    op = "ts_ema" if fn.startswith("EMA") else "ts_mean"
                    return f"o['{op}'](col('close'), {w})"
                if fn in ("EWMA_up_vol", "EWMA_down_vol"):
                    return "o['ts_ema'](col('close'), 20)"
                if fn in ("ATR20", "ATR14", "ATR21", "ATR_20", "ATR_14", "ATR_21", "ATR5", "ATR"):
                    w = {"ATR20": 20, "ATR14": 14, "ATR21": 21, "ATR_20": 20, "ATR_14": 14, "ATR_21": 21, "ATR5": 5, "ATR": 20}[fn]
                    return f"o['ts_mean'](o['true_range'](col('high'), col('low'), col('close')), {w})"
                if fn in ("down_vol", "up_vol", "downside_vol", "upside_vol", "down_vol_60", "up_vol_60", "down_vol_30", "up_vol_30"):
                    w = 60 if fn.endswith("_60") else (30 if fn.endswith("_30") else 20)
                    neg = "-" if fn.startswith("down") else ""
                    return f"o['ts_std'](o['maximum']({neg}o['ts_pct'](col('close'), 1), 0), {w})"
                if fn == "volatility":
                    return "o['ts_std'](o['ts_pct'](col('close'), 1), 20)"
                if fn == "drawdown":
                    return "o['ts_max_drawdown'](col('close'), 20)"
                return tpl.format(x="col('close')", w="20")
            binds = {}
            if n >= 1:
                binds["x"] = self._emit(args[0])
                binds["p"] = self._emit(args[0])
            if n >= 2:
                binds["w"] = self._emit(args[1])
                binds["v"] = self._emit(args[1])
            if n >= 3:
                binds["h"] = self._emit(args[0])
                binds["l"] = self._emit(args[1])
                binds["c"] = self._emit(args[2])
            if n >= 4:
                binds["w"] = self._emit(args[3])
            binds.setdefault("w", "20")
            if "h" not in binds:
                binds["h"] = self._emit(args[0]) if n >= 1 else "col('high')"
                binds["l"] = self._emit(args[1]) if n >= 2 else "col('low')"
                binds["c"] = self._emit(args[2]) if n >= 3 else "col('close')"
            return tpl.format(**binds)
        canon = _fe_op(fn)
        if canon is None:
            raise DslParseError(f"unknown op {fn!r}")
        arg_str = ", ".join(self._emit(a) for a in args)
        if canon in ("ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_median",
                     "ts_skew", "ts_zscore", "ts_max", "ts_min", "ts_kurt",
                     "ts_mad", "ts_quantile", "ts_product", "ts_argmax",
                     "ts_argmin", "ts_decay_linear") and n >= 2:
            a0 = args[0]
            if isinstance(a0, tuple) and a0 and a0[0] == "lit":
                a1 = args[1]
                if not (isinstance(a1, tuple) and a1 and a1[0] == "lit"):
                    args = (a1, a0) + tuple(args[2:])
                    arg_str = ", ".join(self._emit(a) for a in args)
        if fn in ("max", "min") and n == 2:
            a1 = args[1]
            if isinstance(a1, tuple) and a1 and a1[0] == "lit" and float(a1[1]) == 0.0:
                elem = "maximum" if fn == "max" else "minimum"
                return f"o['{elem}']({self._emit(args[0])}, {self._emit(args[1])})"
        if canon in ("log", "abs", "sign", "sqrt", "exp", "sigmoid", "tanh",
                     "signed_sqrt", "zscore", "rank") and n >= 2:
            if isinstance(args[1], tuple) and args[1] and args[1][0] == "lit":
                arg_str = self._emit(args[0])
        return f"o['{canon}']({arg_str})"


def _wrapped_in_outer_paren(s: str) -> bool:
    s = s.strip()
    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


_RECIPE_OP_NAMES = set()
for _k, _v in RECIPE_OPS.items():
    _RECIPE_OP_NAMES |= set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)", _v))
_RECIPE_OP_NAMES -= {"h", "l", "c", "x", "w", "p", "v", "y", "d"}
_RECIPE_OP_NAMES |= set(DIRECT_OPS.values()) | {"true_range", "maximum", "minimum"}


def _wrap_bare_ops(code: str) -> str:
    pattern = re.compile(r"(?<!['\w])([A-Za-z_][A-Za-z0-9_]*)\(")

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name in _RECIPE_OP_NAMES or name in DIRECT_OPS or name in OP_ALIAS:
            return f"o['{name}']("
        return m.group(0)
    return pattern.sub(repl, code)


def _strip_outer_neg(code: str) -> str:
    if code.startswith("o['neg'](") and code.endswith(")"):
        depth = 0
        for i, ch in enumerate(code):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return code[9:i] + code[i + 1:]
    return code


def _is_single_table_convertible(fe_code: str) -> bool:
    for m in re.finditer(r"col\('([^']+)'\)", fe_code):
        c = m.group(1)
        if c in ADJ_SINGLE_TABLE_COLS:
            continue
        if c in MULTI_TABLE_OR_INTERMEDIATE:
            return False
    return True


def convert_one(dsl: str, *, strip_flip: bool = False) -> tuple[bool, str, str]:
    ok, reason = _is_convertible_dsl(dsl)
    if not ok:
        return False, "", reason
    if "EWMA_up_vol" in dsl or "EWMA_down_vol" in dsl or "vol_down" in dsl or "vol_up" in dsl:
        return False, "", "requires-multi-table-or-intermediate"
    try:
        comp = _DslCompiler()
        code = comp.compile(dsl)
        code = _wrap_bare_ops(code)
        if not _is_single_table_convertible(code):
            return False, "", "requires-multi-table-or-intermediate"
        if strip_flip:
            code = _strip_outer_neg(code)
        return True, code, ""
    except DslParseError as exc:
        return False, "", f"parse-error:{exc}"
    except Exception as exc:
        return False, "", f"error:{type(exc).__name__}:{exc}"


def update_json(verbose: bool = True) -> dict:
    data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
    stats = {"total": len(data), "fe_ok": 0, "fe_fail": 0,
             "fail_reasons": {}, "converted_examples": []}
    changed = 0
    for item in data:
        dsl = item.get("dsl", "") or ""
        status = item.get("status", "")
        if status == "evoalpha_week_new":
            item["can_use_factor_engine"] = True
            item["fe_formula"] = dsl
            stats["fe_ok"] += 1
            continue
        if status == "custom" or "MYDSL" in dsl:
            item["can_use_factor_engine"] = False
            stats["fe_fail"] += 1
            stats["fail_reasons"]["MYDSL"] = stats["fail_reasons"].get("MYDSL", 0) + 1
            continue
        ok, code, reason = convert_one(dsl, strip_flip=bool(item.get("is_flipped")))
        if ok:
            item["can_use_factor_engine"] = True
            item["fe_formula"] = code
            stats["fe_ok"] += 1
            if verbose and len(stats["converted_examples"]) < 20:
                stats["converted_examples"].append({"name": item.get("factor_name"),
                                                    "dsl": dsl[:100], "fe": code[:140]})
        else:
            item["can_use_factor_engine"] = False
            stats["fe_fail"] += 1
            key = reason.split(":")[0] if ":" in reason else reason
            stats["fail_reasons"][key] = stats["fail_reasons"].get(key, 0) + 1
        changed += 1
    LQTP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    stats["changed"] = changed
    return stats


def _engine_env():
    import factor_engine.cleaned_operators  # noqa: F401
    from factor_engine.api.columns import col  # noqa: F401
    from factor_engine.api.factor import Factor
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.backend.factory import build_backend
    from factor_engine.storage.factory import build_data_source
    from factor_engine.runtime.engine import FactorEngine

    ops_needed = set(DIRECT_OPS.values()) | set(OP_ALIAS.values()) | {
        "neg", "add", "subtract", "multiply", "divide", "mod", "lt", "gt",
        "le", "ge", "eq", "ne", "and_", "or_", "not_", "abs", "where",
        "expanding_max", "col",
    }
    for v in RECIPE_OPS.values():
        ops_needed |= set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", v))
    data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
    for item in data:
        code = item.get("fe_formula", "") or ""
        ops_needed |= set(re.findall(r"o\['([A-Za-z_][A-Za-z0-9_]*)'\]", code))
    o = {nm: make_cleaned_call_factory(nm) for nm in sorted(ops_needed)}
    o["col"] = col
    # 验证窗口可用 env 收窄（默认 2018-06..2019-03 全市场；smoke 用 FE_SMOKE_START/END 加速）
    import os as _os
    _start = _os.environ.get("FE_SMOKE_START", "2018-06-01")
    _end = _os.environ.get("FE_SMOKE_END", "2019-03-01")
    ds = build_data_source({
        "type": "data_access", "dataset": "ashare_stock_daily_adj",
        "start_date": _start, "end_date": _end,
    })
    eng = FactorEngine(build_backend("pandas"), ds, run_mode="research")
    return o, eng, Factor


def _run_factor(o, eng, Factor, fe_code: str, name: str):
    ns = {"o": o, "col": o["col"], "Factor": Factor}
    exec(f"_f = Factor(name={name!r}, expr={fe_code})", ns)
    f = ns["_f"]
    out = eng.run(f, market="ashare")
    return out["result"].unstack()


def verify_smoke(sample_n: int = 30, n_syms: int = 3, n_days: int = 5) -> dict:
    data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
    convertible = []
    for i, it in enumerate(data):
        if it.get("status") == "evoalpha_week_new":
            continue
        ok, code, reason = convert_one(it.get("dsl", "") or "", strip_flip=bool(it.get("is_flipped")))
        if ok:
            convertible.append((i, it, code))
    rng = np.random.default_rng(7)
    if len(convertible) <= sample_n:
        picks = convertible
    else:
        picks = [convertible[i] for i in rng.choice(len(convertible), sample_n, replace=False)]
    o, eng, Factor = _engine_env()
    ref_syms = ["000001.SZ", "000002.SZ", "000004.SZ"]
    result = {"tested": 0, "passed": 0, "failed": [], "corr": []}
    for idx, (i, item, fe_code) in enumerate(picks):
        name = item.get("factor_name", f"f{i}")
        try:
            mat = _run_factor(o, eng, Factor, fe_code, name)
            tail = mat.iloc[-n_days:]
            ref_path = FV_DIR / f"{name.replace('factor_', '', 1)}.parquet"
            if not ref_path.exists():
                ref_path = FV_DIR / f"{name}.parquet"
            if not ref_path.exists():
                result["failed"].append({"name": name, "err": "no-ref-matrix"})
                continue
            ref = pd.read_parquet(ref_path)
            ref_tail = ref.reindex(index=tail.index)
            corrs = []
            for sym in ref_syms:
                if sym not in tail.columns or sym not in ref_tail.columns:
                    continue
                a = tail[sym].to_numpy(dtype=float)
                b = ref_tail[sym].to_numpy(dtype=float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    continue
                c = float(np.corrcoef(a[m], b[m])[0, 1])
                corrs.append(c)
            if corrs:
                avg_c = float(np.mean(corrs))
                result["corr"].append(avg_c)
                if abs(avg_c) > 0.95:
                    result["passed"] += 1
                else:
                    result["failed"].append({"name": name, "corr": round(avg_c, 3)})
            else:
                result["failed"].append({"name": name, "err": "no-common-data"})
            result["tested"] += 1
            _log(f"  verify[{idx+1}/{len(picks)}] {name[:40]:42s} corr={avg_c if corrs else float('nan'):.3f}")
        except Exception as exc:
            result["failed"].append({"name": name, "err": f"{type(exc).__name__}: {str(exc)[:120]}"})
            result["tested"] += 1
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    _init_fe_ops()
    _init_known_fields()
    _log(f"FE ops={len(_FE_OPS)} fields={len(_KNOWN_FIELDS)}")

    if args.dry or args.smoke:
        data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
        stats = {"fe_ok": 0, "fe_fail": 0, "fail_reasons": {}}
        for item in data:
            if item.get("status") == "evoalpha_week_new":
                stats["fe_ok"] += 1
                continue
            ok, code, reason = convert_one(item.get("dsl", "") or "", strip_flip=bool(item.get("is_flipped")))
            if ok:
                stats["fe_ok"] += 1
            else:
                stats["fe_fail"] += 1
                key = reason.split(":")[0] if ":" in reason else reason
                stats["fail_reasons"][key] = stats["fail_reasons"].get(key, 0) + 1
        _log(f"DRY 统计: 可转 {stats['fe_ok']}/{len(data)} 不可转 {stats['fe_fail']}")
        _log("无法转换原因: " + json.dumps(stats["fail_reasons"], ensure_ascii=False))

    if args.convert and not args.dry:
        stats = update_json()
        _log(f"UPDATE 完成: 可转 {stats['fe_ok']}/{stats['total']} 不可转 {stats['fe_fail']} changed={stats['changed']}")
        _log("无法转换原因: " + json.dumps(stats["fail_reasons"], ensure_ascii=False))

    if args.smoke:
        result = verify_smoke(args.sample)
        _log(f"SMOKE 结果: tested={result['tested']} passed={result['passed']} failures={len(result['failed'])}")
        for f in result["failed"][:15]:
            _log(f"  FAIL {f}")
        if result["corr"]:
            _log(f"  corr mean={np.mean(result['corr']):.3f} min={np.min(result['corr']):.3f}")


if __name__ == "__main__":
    main()
