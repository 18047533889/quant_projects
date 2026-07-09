# -*- coding: utf-8 -*-
"""算子执行策略（Operator Policy）：企业级语义的标准化描述。

与 ``OperatorMetadata``（文档/catalog）互补：
- metadata：人类可读、LLM 提示词
- policy：机器可读、lookback 推断、PIT 审计、lineage hash
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

Scope = Literal["ts", "cs", "elementwise", "aggregate", "hypothesis", "unknown"]
NanPolicy = Literal["propagate", "ignore", "zero", "ffill_only"]

# Tier-1 核心 canonical（不含 alias；alias 见 TIER1_ALIASES）
TIER1_CANONICALS: frozenset[str] = frozenset({
    "ts_delay", "ts_delta", "ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_corr",
    "ts_sharpe", "ts_autocorr",
    "ts_min", "ts_max", "ts_ema", "ts_beta", "ts_decay_linear", "WMA",
    "ewm_mean", "ewm_corr",
    "rank", "zscore", "winsorize", "neutralize", "quantile", "scale",
    "normalize", "standardize", "group_rank", "group_neutralize",
    "group_winsorize", "group_zscore", "group_mean", "group_decay_linear",
    "fillna_const", "fillna_interpolate", "ffill", "bfill",
    "cs_regression", "cs_resid", "cum_prod", "cum_delta", "cum_first",
    "expanding_rank", "rank_corr", "add", "subtract", "multiply", "divide",
    "ts_topk_sum", "vp_weighted_price", "real_turnover_rate", "micro_realized_vol",
    "micro_spread", "micro_amihud_hf", "micro_mid_return", "micro_bipower_var",
    "micro_jump_indicator", "micro_trade_imbalance", "micro_vpin", "micro_kyle_lambda",
    "hump_decay", "col", "corr_test",
})

# Tier-1 DSL 别名 → canonical（CI / policy 校验前先 resolve）
TIER1_ALIASES: dict[str, str] = {
    "SMA": "ts_mean",
    "EMA": "ts_ema",
    "decay_linear": "ts_decay_linear",
}

# production ``auto`` backend 下允许走 Polars 的 canonical 白名单
POLARS_PRODUCTION_SAFE: frozenset[str] = frozenset({
    "ts_mean",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_delta",
    "ts_delay",
    "delay",
    "add",
    "subtract",
    "multiply",
    "divide",
    "abs",
    "log",
    "clip",
    "neg",
    "exp",
    "sqrt",
    "sign",
    "ts_sharpe",
    "ts_autocorr",
    "rank",
    "zscore",
    "ts_std",
})


def resolve_tier1_canonical(name: str) -> str:
    """Tier-1 校验 / cost 门禁：alias → canonical。"""
    from cleaned_operators.registry import OperatorRegistry

    if name in TIER1_ALIASES:
        return TIER1_ALIASES[name]
    return OperatorRegistry._aliases.get(name, name)


def tier1_policy_keys() -> frozenset[str]:
    """Tier-1 在 _EXPLICIT_POLICIES 中必须存在的键（含 alias 名）。"""
    keys = set(TIER1_CANONICALS)
    keys.update(TIER1_ALIASES.keys())
    for alias, canon in TIER1_ALIASES.items():
        keys.add(canon)
    return frozenset(keys)

# 不做 Polars 移植且通常非 PIT 安全（FFT/矩阵/随机/CDF-PDF 等）
INTENTIONALLY_PANDAS_ONLY: frozenset[str] = frozenset(
    {
        "constant",
        "shuffle",
        "fft",
        "ifft",
        "convolve",
        "filter_bandpass",
        "filter_highpass",
        "filter_lowpass",
        "filter_notch",
        "wavelet",
        "wavelet_denoise",
        "mat_add",
        "mat_subtract",
        "mat_multiply",
        "mat_transpose",
        "mat_inverse",
        "mat_determinant",
        "mat_rank",
        "eig",
        "svd",
        "pca",
        "qr_decompose",
        "lu_decompose",
        "rand_exp",
        "rand_lognormal",
        "rand_normal",
        "rand_poisson",
        "rand_uniform",
        "cdf_chi2",
        "cdf_f",
        "cdf_normal",
        "cdf_t",
        "pdf_chi2",
        "pdf_f",
        "pdf_normal",
        "pdf_t",
    }
)

PIT_UNSAFE_CANONICALS: frozenset[str] = INTENTIONALLY_PANDAS_ONLY | frozenset(
    {"Lead", "next", "bfill", "fillna_interpolate", "shuffle"}
)

# 显式声明的核心算子策略（Tier-1 + 特殊语义；其余由 infer_operator_policy 推断）
_EXPLICIT_POLICIES: dict[str, dict[str, Any]] = {
    "ts_delay": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_delta": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_pct": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_log_return": {"scope": "ts", "lag": 1, "pit_safe": True, "min_periods": 2},
    "ts_sharpe": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_autocorr": {"scope": "ts", "pit_safe": True, "min_periods": 3, "lag": 1},
    "intraday_vwap_deviation": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "operating_margin": {"scope": "elementwise", "pit_safe": True},
    "current_ratio": {"scope": "elementwise", "pit_safe": True},
    "quick_ratio": {"scope": "elementwise", "pit_safe": True},
    "debt_to_equity": {"scope": "elementwise", "pit_safe": True},
    "delay": {"scope": "ts", "lag": 1, "pit_safe": True},
    "ts_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_std": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_rank": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_min": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_max": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_ema": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "SMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "EMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "WMA": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ewm_mean": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ewm_corr": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_beta": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "rolling_beta": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "idio_vol": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "idio_skew": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "downside_beta": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "tail_beta": {"scope": "ts", "pit_safe": True, "min_periods": 3},
    "residual_momentum_capm": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "coskewness_to_market": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "open_gap": {"scope": "ts", "pit_safe": True, "lag": 1},
    "close_gap": {"scope": "elementwise", "pit_safe": True},
    "cs_mad": {"scope": "cs", "pit_safe": True},
    "cs_mad_zscore": {"scope": "cs", "pit_safe": True},
    "cs_quantile": {"scope": "cs", "pit_safe": True},
    "ADXR": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON_up": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "AROON_down": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "RSI_WILDER": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ATR_WILDER": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "group_decay_linear": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "hump_decay": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_topk_sum": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quantile": {"scope": "cs", "pit_safe": True},
    "scale": {"scope": "cs", "pit_safe": True},
    "normalize": {"scope": "cs", "pit_safe": True},
    "standardize": {"scope": "cs", "pit_safe": True},
    "col": {"scope": "elementwise", "lookback_window": 0, "pit_safe": True},
    "Lead": {"scope": "ts", "lag": -1, "pit_safe": False},
    "next": {"scope": "ts", "lag": -1, "pit_safe": False},
    "bfill": {"scope": "elementwise", "pit_safe": False, "nan_policy": "ffill_only"},
    "ffill": {"scope": "elementwise", "pit_safe": True, "nan_policy": "ffill_only"},
    "fillna_const": {"scope": "elementwise", "pit_safe": True},
    "fillna_interpolate": {"scope": "elementwise", "pit_safe": False},
    "rank": {"scope": "cs", "pit_safe": True},
    "rank_pct": {"scope": "cs", "pit_safe": True},
    "cs_pct_rank": {"scope": "cs", "pit_safe": True},
    "cs_rank_01": {"scope": "cs", "pit_safe": True},
    "c_percentile": {"scope": "cs", "pit_safe": True},
    "c_mean": {"scope": "cs", "pit_safe": True},
    "c_std": {"scope": "cs", "pit_safe": True},
    "c_sum": {"scope": "cs", "pit_safe": True},
    "c_count": {"scope": "cs", "pit_safe": True},
    "protected_div": {"scope": "elementwise", "pit_safe": True},
    "protected_log": {"scope": "elementwise", "pit_safe": True},
    "protected_sqrt": {"scope": "elementwise", "pit_safe": True},
    "vwap": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "volatility": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "dropna": {"scope": "elementwise", "pit_safe": True},
    "zscore": {"scope": "cs", "pit_safe": True},
    "winsorize": {"scope": "cs", "pit_safe": True},
    "neutralize": {"scope": "cs", "pit_safe": True},
    "group_rank": {"scope": "cs", "pit_safe": True},
    "group_neutralize": {"scope": "cs", "pit_safe": True},
    "group_winsorize": {"scope": "cs", "pit_safe": True},
    "group_zscore": {"scope": "cs", "pit_safe": True},
    "group_mean": {"scope": "cs", "pit_safe": True},
    "cs_regression": {"scope": "cs", "pit_safe": True},
    "cs_resid": {"scope": "cs", "pit_safe": True},
    "add": {"scope": "elementwise", "pit_safe": True},
    "subtract": {"scope": "elementwise", "pit_safe": True},
    "multiply": {"scope": "elementwise", "pit_safe": True},
    "divide": {"scope": "elementwise", "pit_safe": True},
    "cum_prod": {"scope": "ts", "pit_safe": True, "includes_current_bar": True},
    "cum_delta": {"scope": "ts", "pit_safe": True},
    "cum_first": {"scope": "ts", "pit_safe": True},
    "expanding_rank": {"scope": "ts", "pit_safe": True},
    "rank_corr": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "vp_weighted_price": {"scope": "ts", "pit_safe": True},
    "real_turnover_rate": {"scope": "ts", "pit_safe": True},
    "micro_realized_vol": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_spread": {"scope": "elementwise", "pit_safe": True},
    "micro_amihud_hf": {
        "scope": "ts",
        "pit_safe": True,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_mid_return": {
        "scope": "ts",
        "pit_safe": True,
        "lag": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_bipower_var": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_jump_indicator": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_trade_imbalance": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_vpin": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 1,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "micro_kyle_lambda": {
        "scope": "ts",
        "pit_safe": True,
        "min_periods": 2,
        "session_aware": True,
        "reset_at_session_boundary": True,
    },
    "shuffle": {"scope": "unknown", "pit_safe": False},
    "avg": {"scope": "aggregate", "pit_safe": True},
    "corr_test": {"scope": "hypothesis", "pit_safe": True},
    "abs": {"scope": "elementwise", "pit_safe": True},
    "log": {"scope": "elementwise", "pit_safe": True},
    "clip": {"scope": "elementwise", "pit_safe": True},
    "neg": {"scope": "elementwise", "pit_safe": True},
    "exp": {"scope": "elementwise", "pit_safe": True},
    "sqrt": {"scope": "elementwise", "pit_safe": True},
    "sign": {"scope": "elementwise", "pit_safe": True},
    "where": {"scope": "elementwise", "pit_safe": True},
    "if_else": {"scope": "elementwise", "pit_safe": True},
    "coalesce": {"scope": "elementwise", "pit_safe": True},
    "cs_demean": {"scope": "cs", "pit_safe": True},
    "ts_var": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ts_median": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "quarter": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "ttm": {"scope": "ts", "pit_safe": True, "min_periods": 1},
    "yoy": {"scope": "ts", "pit_safe": True, "min_periods": 5},
    "avg2": {"scope": "ts", "pit_safe": True, "min_periods": 2},
    "ts_zscore": {"scope": "ts", "pit_safe": True, "min_periods": 1},
}


@dataclass
class OperatorPolicy:
    """算子执行语义（企业级 schema 子集）。"""

    scope: Scope = "unknown"
    lookback_window: int | None = None
    min_periods: int | None = None
    lag: int = 0
    pit_safe: bool = True
    nan_policy: NanPolicy = "propagate"
    includes_current_bar: bool = True
    calendar: str = "bar"  # 本引擎 bar = 每标的连续行，非自然日
    session_aware: bool = False
    reset_at_session_boundary: bool = False
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_operator_policy(op: Any, *, canonical: str | None = None) -> OperatorPolicy:
    """从算子实例 metadata/category/tags 推断默认 policy。"""
    canon = canonical or getattr(getattr(op, "metadata", None), "name", "") or ""
    if canon in _EXPLICIT_POLICIES:
        return OperatorPolicy(**{**{"scope": "unknown", "pit_safe": True}, **_EXPLICIT_POLICIES[canon]})

    meta = getattr(op, "metadata", None)
    category = (getattr(meta, "category", "") or "").lower()
    tags = [str(t).lower() for t in (getattr(meta, "tags", None) or [])]
    name = (getattr(meta, "name", "") or canon).lower()

    scope: Scope = "unknown"
    if category in ("cross_sectional",) or "cross_section" in tags:
        scope = "cs"
    elif category in ("group_neutralization",):
        scope = "cs"
    elif category in (
        "time_series",
        "shift_diff_cum",
        "technical_signal",
        "price_volume",
        "intraday_microstructure",
        "signal",
    ):
        scope = "ts"
    elif category in ("statistics",):
        if any(k in name for k in ("test", "corr_test", "granger", "ttest", "adf", "kpss")):
            scope = "hypothesis"
        else:
            scope = "aggregate"
    elif category in ("math", "elementwise_math", "data_handling"):
        scope = "elementwise"

    pit_safe = "pit_safe" in tags or "causal" in tags
    if canon in PIT_UNSAFE_CANONICALS:
        pit_safe = False
    elif name in ("lead", "next"):
        pit_safe = False
    elif name == "shuffle":
        pit_safe = False

    if scope == "ts" and any(k in name for k in ("mean", "std", "sum", "corr", "rank", "decay")):
        lookback: int | None = None  # 运行时由 window 参数决定
        min_periods = 1
    elif scope == "cs":
        lookback = 0
        min_periods = 1
    elif scope == "aggregate":
        lookback = None  # expanding
        min_periods = 1
    else:
        lookback = None
        min_periods = None

    return OperatorPolicy(
        scope=scope,
        lookback_window=lookback,
        min_periods=min_periods,
        pit_safe=pit_safe,
        tags=list(getattr(meta, "tags", None) or []),
    )


def compute_operator_catalog_hash(*, backend: str = "pandas_numpy") -> str:
    """对所有已实现算子的 canonical + policy 做确定性 hash，用于 lineage。"""
    from cleaned_operators.registry import OperatorRegistry

    entries: list[dict[str, Any]] = []
    for canon in sorted(OperatorRegistry._operators.keys()):
        op = OperatorRegistry.get(canon, backend=backend)
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canon)
        entries.append(
            {
                "canonical": canon,
                "category": getattr(op.metadata, "category", ""),
                "policy": policy.to_dict(),
            }
        )
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_lookback(
    analysis_lookback: int,
    *,
    factor_freq: str | None = None,
    source_bar_freq: str | None = None,
    lag_buffer: int = 1,
    extra: int = 5,
) -> int:
    """将 IR 分析 lookback 转为数据加载历史缓冲 bar 数。

    当因子频率与数据源 bar 频率不一致时（如因子 ``1d``、源 ``5m``），
    按 ``bars_per_day`` 比例换算 warmup 长度。
    """
    base = max(0, int(analysis_lookback))
    bars = base + max(0, int(lag_buffer)) + max(0, int(extra))
    if factor_freq and source_bar_freq and factor_freq != source_bar_freq:
        f_bpd = bars_per_day(factor_freq)
        s_bpd = bars_per_day(source_bar_freq)
        if f_bpd > 0 and s_bpd > 0 and f_bpd != s_bpd:
            # lookback 按因子频率解释，换算为数据源 bar 数（如 20 日 × 78 个 5m bar）
            bars = int(max(bars, round(bars * s_bpd / f_bpd)))
    return bars


def bars_to_calendar_trading_days(bars: int, bar_freq: str | None) -> int:
    """将 bar 数换算为交易日数（用于增量/扩窗的 calendar offset）。"""
    count = max(0, int(bars))
    if count <= 0:
        return 0
    s_bpd = bars_per_day(bar_freq)
    if s_bpd > 1:
        return max(1, (count + s_bpd - 1) // s_bpd)
    return count


def infer_source_bar_freq(data_source: Any, *, fallback: str | None = "1d") -> str:
    """从 DataSource 读取 bar 频率；无则回退 factor 或日频。"""
    for attr in ("bar_freq", "freq"):
        value = getattr(data_source, attr, None)
        if value:
            return str(value)
    inner = getattr(data_source, "inner", None) or getattr(data_source, "_inner", None)
    if inner is not None:
        return infer_source_bar_freq(inner, fallback=fallback)
    return str(fallback or "1d")


# 常见 bar 频率 → 每交易日 bar 数（按市场 regular session）
# US: 6.5h RTH；CN: 4h（240min）；HK: 5.5h（330min）
_MARKET_BARS_PER_DAY: dict[str, dict[str, dict[str, int]]] = {
    "US": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "60m": 7,
        },
    },
    "CN": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 240, "5m": 48, "15m": 16, "30m": 8, "1h": 4, "60m": 4,
        },
    },
    "HK": {
        "regular": {
            "1d": 1, "1D": 1, "d": 1,
            "1m": 330, "5m": 66, "15m": 22, "30m": 11, "1h": 6, "60m": 6,
        },
    },
}

# 向后兼容：默认美股 regular session
_BARS_PER_DAY: dict[str, int] = dict(_MARKET_BARS_PER_DAY["US"]["regular"])


def bars_per_day(
    freq: str | None,
    *,
    market: str = "US",
    session: str = "regular",
) -> int:
    """解析频率字符串为每交易日 bar 数；未知频率默认 1（日频）。

    ``market`` 支持 ``US`` / ``CN`` / ``HK``；``session`` 目前仅 ``regular``。
    """
    if not freq:
        return 1
    text = str(freq).strip()
    mkt = str(market or "US").strip().upper()
    sess = str(session or "regular").strip().lower()
    table = _MARKET_BARS_PER_DAY.get(mkt, _MARKET_BARS_PER_DAY["US"]).get(
        sess, _MARKET_BARS_PER_DAY["US"]["regular"]
    )
    if text in table:
        return table[text]
    lowered = text.lower()
    if lowered in table:
        return table[lowered]
    if lowered.endswith("d"):
        return 1
    return 1


def normalize_bars_market(market: str | None) -> str:
    """``infer_market`` / universe 市场码 → ``bars_per_day`` 市场键（US/CN/HK）。"""
    if not market:
        return "US"
    key = str(market).strip().lower()
    if key in {"ashare", "a_share", "cn", "china", "sse", "szse"}:
        return "CN"
    if key in {"hk", "hongkong", "hkg"}:
        return "HK"
    if key in {"us", "usa", "nyse", "nasdaq"}:
        return "US"
    upper = str(market).strip().upper()
    if upper in _MARKET_BARS_PER_DAY:
        return upper
    return "US"


def bar_freq_to_timedelta(freq: str | None) -> pd.Timedelta:
    """将 bar 频率字符串转为 Timedelta（用于日内 tick 级扩窗）。"""
    if not freq:
        return pd.Timedelta(days=1)
    text = str(freq).strip().lower()
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "60m": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "d": pd.Timedelta(days=1),
    }
    if text in mapping:
        return mapping[text]
    if text.endswith("m") and text[:-1].isdigit():
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("h") and text[:-1].isdigit():
        return pd.Timedelta(hours=int(text[:-1]))
    return pd.Timedelta(days=1)
