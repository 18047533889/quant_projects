# -*- coding: utf-8 -*-
"""重复算子合并 + 业界标准命名（WorldQuant / pandas 惯例）。"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

CANONICAL_RENAMES: dict[str, str] = {
    # 时序滚动：统一 ts_ 前缀
    "m_var": "ts_var",
    "m_median": "ts_median",
    "m_mad": "ts_mad",
    "m_beta": "ts_beta",
    "m_top_n_avg": "ts_top_n_avg",
    "m_top_n_std": "ts_top_n_std",
    "m_bottom_n_avg": "ts_bottom_n_avg",
    "m_bottom_n_sum": "ts_bottom_n_sum",
    "decay_linear": "ts_decay_linear",
    "ema": "ts_ema",
    "ratios": "ts_ratio",
    # 元素级：numpy/pandas 惯例 clip
    "cap": "clip",
    # 分组：业界常用 group_neutralize
    "group_demean": "group_neutralize",
    "cum_standardize": "expanding_zscore",
}

# 别名 → 保留的 canonical
DEDUPE_ALIASES: dict[str, str] = {
    # --- 时序滚动 ---
    "SMA": "ts_mean",
    "move": "ts_mean",
    "window_mean": "ts_mean",
    "running_mean": "ts_mean",
    "window_sum": "ts_sum",
    "window_max": "ts_max",
    "window_min": "ts_min",
    "window_std": "ts_std",
    "deltas": "ts_delta",
    "ts_decay": "ts_decay_linear",
    "Sum": "ts_sum",
    "Kurt": "ts_kurt",
    "Skew": "ts_skew",
    "Corr": "ts_corr",
    "Cov": "ts_cov",
    "Covariance": "ts_cov",
    "m_zscore": "ts_zscore",
    "returns": "ts_pct",
    "m_pct_change": "ts_pct",
    "pct_change": "ts_pct",
    "m_var": "ts_var",
    "m_median": "ts_median",
    "m_mad": "ts_mad",
    "m_beta": "ts_beta",
    "m_top_n_avg": "ts_top_n_avg",
    "m_top_n_std": "ts_top_n_std",
    "m_bottom_n_avg": "ts_bottom_n_avg",
    "m_bottom_n_sum": "ts_bottom_n_sum",
    "m_top_n_sum": "ts_topk_sum",
    "Percentile": "ts_quantile",
    "percentile": "ts_quantile",
    "running_std": "ts_std",
    "running_sum": "ts_sum",
    "cum_standardize": "expanding_zscore",
    "DECAY_LINEAR": "ts_decay_linear",
    "TS_DECAY_LINEAR": "ts_decay_linear",
    "decay_linear": "ts_decay_linear",
    "EMA": "ts_ema",
    "ema": "ts_ema",
    "ratios": "ts_ratio",
    # --- 回归统计量 ---
    "Var": "ts_var",
    "Mad": "ts_mad",
    "Median": "ts_median",
    "Beta": "beta",
    "Intercept": "intercept",
    "Residual": "residual",
    "R2": "r_squared",
    "var": "ts_var",
    "mad": "ts_mad",
    "median": "ts_median",
    "corr": "ts_corr",
    "cov": "ts_cov",
    "kurt": "ts_kurt",
    "skew": "ts_skew",
    "sum": "ts_sum",
    # --- 元素 / 信号 ---
    "negate": "neg",
    "reciprocal": "inv",
    "if_else": "where",
    "clamp": "clip",
    "cap": "clip",
    "CLIP": "clip",
    "clip": "clip",
    "dft": "fft",
    "idft": "ifft",
    # --- 截面 / 分组 ---
    "standardize": "zscore",
    "panel_rank": "rank",
    "panel_zscore": "zscore",
    "panel_standardize": "zscore",

    "industry_neutralize": "group_neutralize",
    "neutralize": "group_neutralize",
    "c_neutralize": "group_neutralize",
    "panel_neutralize": "group_neutralize",
    "group_demean": "group_neutralize",
    "NEUTRALIZE": "group_neutralize",
    "industry_size_neutralize": "group_neutralize",
    "size_industry_neutralize": "group_neutralize",
    "INDUSTRY_NEUTRAL": "group_neutralize",
    "INDUSTRY_NEUTRALIZE": "group_neutralize",
    "IND_NEUTRALIZE": "group_neutralize",
    # --- 累计 / 扩展 ---
    "mean_agg": "avg",
    "cum_avg": "expanding_mean",
    "cum_rank": "expanding_rank",
    "cum_std": "expanding_std",
    "cum_last": "ffill",
    "last": "ffill",
    "last_not_null": "ffill",
    "first": "first_not_null",
    # --- 时序别名（已有） ---
    "Mean": "ts_mean",
    "TS_MEAN": "ts_mean",
    "m_avg": "ts_mean",
    "mean": "ts_mean",
    "Max": "ts_max",
    "TS_MAX": "ts_max",
    "m_max": "ts_max",
    "Min": "ts_min",
    "TS_MIN": "ts_min",
    "m_min": "ts_min",
    "Std": "ts_std",
    "TS_STD": "ts_std",
    "m_std": "ts_std",
    "std": "ts_std",
    "TS_SUM": "ts_sum",
    "m_sum": "ts_sum",
    "Delta": "ts_delta",
    "Diff": "ts_delta",
    "TS_DELTA": "ts_delta",
    "TS_PCT": "ts_pct",
    "TS_KURT": "ts_kurt",
    "TS_SKEW": "ts_skew",
    "TS_CORR": "ts_corr",
    "TS_COV": "ts_cov",
    "m_percentile": "ts_quantile",
    "m_argmax": "ts_argmax",
    "m_argmin": "ts_argmin",
    "TS_TOPK_SUM": "ts_topk_sum",
}

# 注销的冗余 canonical（实现类仍可在源码中，但不再占主键）
REMOVED_CANONICALS: tuple[str, ...] = (
    "SMA",
    "move",
    "window_mean",
    "running_mean",
    "window_sum",
    "window_max",
    "window_min",
    "window_std",
    "deltas",
    "ts_decay",
    "Sum",
    "Kurt",
    "Skew",
    "Corr",
    "Cov",
    "Covariance",
    "Var",
    "Mad",
    "Median",
    "m_zscore",
    "Beta",
    "Intercept",
    "Residual",
    "R2",
    "negate",
    "reciprocal",
    "if_else",
    "clamp",
    "dft",
    "idft",
    "standardize",
    "panel_rank",
    "panel_zscore",
    "panel_standardize",
    "industry_neutralize",
    "neutralize",
    "panel_neutralize",
    "mean_agg",
    "cum_avg",
    "cum_rank",
    "cum_std",
    "cum_last",
    "last",
    "last_not_null",
    "first",
    "returns",
    "Percentile",
    "running_std",
    "running_sum",
)


_DEDUPE_APPLIED = False


def apply_operator_deduplication() -> None:
    """在全部算子注册与基础别名加载后调用（``load_all()`` 末尾）。"""
    global _DEDUPE_APPLIED
    if _DEDUPE_APPLIED:
        return
    for old, new in CANONICAL_RENAMES.items():
        OperatorRegistry.rename_canonical(old, new)

    for alias, canonical in DEDUPE_ALIASES.items():
        if OperatorRegistry.get(canonical) is None:
            raise RuntimeError(f"dedupe target missing: {canonical!r} (alias {alias!r})")
        OperatorRegistry.register_alias(alias, canonical)

    for name in REMOVED_CANONICALS:
        OperatorRegistry.unregister(name)
    _DEDUPE_APPLIED = True
