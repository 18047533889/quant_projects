# -*- coding: utf-8 -*-
"""重复算子合并 + LQTP / 业界标准命名。

本模块在 ``load_all()`` 末尾调用 ``apply_operator_deduplication()``，
将历史方言名重定向到 canonical 并注销冗余主键。

常量
----
- ``CANONICAL_RENAMES``：旧 canonical → 新 canonical 重命名表；
- ``DEDUPE_ALIASES``：别名 → 保留 canonical 映射；
- ``REMOVED_CANONICALS``：去重后注销的冗余 canonical 列表。

命名原则
--------
时序算子保留 ``ts_*`` canonical，元素算子保留现有生产契约中的名字。
较短的 LQTP 名称作为 DSL 别名，不反向改写 canonical。这样 registry、
policy、SQL/Polars evidence 和持久化 manifest 始终引用同一个稳定主键。
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

CANONICAL_RENAMES: dict[str, str] = {
    # 时序滚动：历史 m_* → ts_*
    "m_var": "ts_var",
    "m_median": "ts_median",
    "m_mad": "ts_mad",
    "m_beta": "ts_beta",
    "m_top_n_avg": "ts_top_n_avg",
    "m_top_n_std": "ts_top_n_std",
    "m_bottom_n_avg": "ts_bottom_n_avg",
    "m_bottom_n_sum": "ts_bottom_n_sum",
    "ratios": "ts_ratio",
    # 回归算子统一输出名称
    "ts_regression": "ts_regression_slope",
    # 分组 / 扩展
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
    "ma": "ts_mean",
    "window_sum": "ts_sum",
    "sum_n": "ts_sum",
    "window_max": "ts_max",
    "window_min": "ts_min",
    "window_std": "ts_std",
    "std_n": "ts_std",
    "deltas": "ts_delta",
    "delta": "ts_delta",
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
    "ewm_mean": "ts_ema",
    "ratios": "ts_ratio",
    # --- 回归统计量 ---
    "Var": "ts_var",
    # NEW-037: an alias is only legal when the maths is EQUIVALENT.  ``Mad`` /
    # ``mad`` are the single-centre mean-absolute-deviation (``mean(|x-mean|)``);
    # ``ts_mad`` is a DIFFERENT legacy double-rolling estimator (rolling median
    # -> abs -> rolling mean).  Aliasing Mad->ts_mad silently redefined the
    # statistic, so the alias now points at the mathematically equal
    # ``ts_mean_abs_deviation``.
    "Mad": "ts_mean_abs_deviation",
    "Median": "ts_median",
    "Beta": "beta",
    "Intercept": "intercept",
    "Residual": "residual",
    "R2": "r_squared",
    "var": "ts_var",
    "mad": "ts_mean_abs_deviation",
    "median": "ts_median",
    "corr": "ts_corr",
    "cov": "ts_cov",
    "kurt": "ts_kurt",
    "skew": "ts_skew",
    "sum": "ts_sum",
    "ts_regression": "ts_regression_slope",
    "TS_REGRESSION_SLOPE": "ts_regression_slope",
    # --- 元素 / 信号 ---
    "negate": "neg",
    "inv": "inverse",
    "reciprocal": "inverse",
    "if_else": "where",
    "fmax": "maximum",
    "fmin": "minimum",
    "sqr": "square",
    "cumulative_max": "expanding_max",
    "cumulative_mean": "expanding_mean",
    "cumulative_min": "expanding_min",
    "clamp": "clip",
    "cap": "clip",
    "CLIP": "clip",
    "dft": "fft",
    "idft": "ifft",
    "safe_div": "safe_div_null",
    # --- delay ---
    "DELAY": "ts_delay",
    "Delay": "ts_delay",
    "Ref": "ts_delay",
    "delay": "ts_delay",
    "m_delay": "ts_delay",
    "shift": "ts_delay",
    # --- 截面 / 分组 ---
    "standardize": "zscore",
    "panel_rank": "rank",
    "panel_zscore": "zscore",
    "panel_standardize": "zscore",
    "industry_neutralize": "group_neutralize",
    "industry_neutral": "group_neutralize",
    "ind_neutralize": "group_neutralize",
    "neutralize": "group_neutralize",
    "c_neutralize": "group_neutralize",
    "panel_neutralize": "group_neutralize",
    "group_demean": "group_neutralize",
    "NEUTRALIZE": "group_neutralize",
    "INDUSTRY_NEUTRAL": "group_neutralize",
    "INDUSTRY_NEUTRALIZE": "group_neutralize",
    "IND_NEUTRALIZE": "group_neutralize",
    "market_cap_neutralize": "size_neutralize",
    "cap_neutralize": "size_neutralize",
    "size_industry_neutralize": "industry_size_neutralize",
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
    # Removed from the factor runtime: future-looking, misleading, random,
    # non-shape-preserving, or full-sample operators.
    "Lead",
    "next",
    "bfill",
    "causal_bfill",
    "fillna_interpolate",
    "interpolate",
    "norm",
    "norm_l1",
    "norm_linf",
    "rand_exp",
    "rand_lognormal",
    "rand_normal",
    "rand_poisson",
    "rand_uniform",
    "sample",
    "shuffle",
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
    "inv",
    "reciprocal",
    "fmax",
    "fmin",
    "sqr",
    "cumulative_max",
    "cumulative_mean",
    "cumulative_min",
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
    # LQTP 对齐：与 ema 重复的独立实现
    "ewm_mean",
)


_DEDUPE_APPLIED = False


def apply_operator_deduplication() -> None:
    """在全部算子注册与基础别名加载后执行去重合并。

    依次执行：canonical 重命名 → 别名登记 → 冗余 canonical 注销。
    幂等设计，重复调用仅第一次生效。

    返回:
        None

    异常:
        RuntimeError: 去重目标 canonical 无 runtime 实现时。
    """
    global _DEDUPE_APPLIED
    if _DEDUPE_APPLIED:
        return
    for old, new in CANONICAL_RENAMES.items():
        OperatorRegistry.rename_canonical(old, new)

    # Remove retired canonical spellings before registering them as aliases;
    # alias/canonical collisions are always errors in the Registry.
    for name in REMOVED_CANONICALS:
        OperatorRegistry.unregister(name)

    for alias, canonical in DEDUPE_ALIASES.items():
        if alias == canonical:
            raise RuntimeError(
                f"dedupe self-loop: {alias!r} aliases itself (R16-075 — a "
                "canonical->canonical edge has no value and pollutes the alias graph)"
            )
        if canonical not in OperatorRegistry._operators:
            raise RuntimeError(f"dedupe target missing: {canonical!r} (alias {alias!r})")
        OperatorRegistry.register_alias(alias, canonical)

    _DEDUPE_APPLIED = True
