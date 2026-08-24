# -*- coding: utf-8 -*-
"""
DSL 别名 → canonical 映射（**须在全部算子类注册之后**加载）。

作用
----
让挖掘侧习惯的写法（``ts_rsi``、``CS_RANK``、``DECAY_LINEAR``）与 registry 内
canonical 名（``RSI``、``rank``、``decay_linear``）指向同一 ``Operator`` 实例。

维护
----
- 新增 DSL 别名：在本文件 ``register_alias``，并确保 canonical 已在对应 ``*.py`` 实现；
- 技术指标 ``ts_*`` → 大写 canonical 见下方「§ 技术指标」段；
- 重复算子合并见 [`_dedupe.py`](_dedupe.py)（别名重定向 + 注销冗余 canonical）。

本模块无函数定义，import 时产生副作用（向 ``OperatorRegistry`` 登记别名）。
"""
from dataclasses import dataclass

from factor_engine.cleaned_operators.registry import OperatorRegistry


# ---------------------------------------------------------------------------
# R40 #147: 版本化 migration table —— deprecated alias → canonical 的完整凭证。
#
# ``register_compat_alias`` 只在 registry catalog 上记录
# ``compat_aliases[alias] = {migration_reason, deprecated_since, removal_version}``，
# 但没有版本化的迁移语义表（old → new → 生效 DSL 版本 → 语义等价 → 迁移规则 →
# 弃用时间 → 移除时间）。本表补齐该语义，供 mining / 审计 / 公式迁移统一消费；
# ``test_migration_table_completeness`` 校验所有 deprecated alias 都在表中。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorMigrationRecord:
    """一条 deprecated alias → canonical 的版本化迁移记录（R40 #147）。"""

    old_canonical: str
    new_canonical: str
    effective_dsl_version: str
    semantic_equivalent: bool
    migration_rule: str
    deprecated_since: str
    remove_after: str


MIGRATION_TABLE: tuple[OperatorMigrationRecord, ...] = (
    # R40 #146: fin_mad 是 fin_mean_abs_deviation 的兼容名（mean(|x-mean|)，非
    # median-based MAD）。从 canonical 降为 registry alias，mining 只搜 canonical。
    OperatorMigrationRecord(
        old_canonical="fin_mad",
        new_canonical="fin_mean_abs_deviation",
        effective_dsl_version="0.11.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="0.11.0",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_matrix_profile_motif_distance",
        new_canonical="ts_matrix_profile_discord_score",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="2026-08",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_weighted_semivariance_sqrt",
        new_canonical="ts_weighted_downside_deviation",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="0.11.0",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_garch_vol_forecast",
        new_canonical="ts_garch_next_vol_forecast",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="2026-08",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_har_rv_next_forecast",
        new_canonical="ts_har_rv_next_vol_forecast",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="2026-08",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_har_rv_forecast",
        new_canonical="ts_har_rv_next_vol_forecast",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="2026-08",
        remove_after="1.0",
    ),
    OperatorMigrationRecord(
        old_canonical="ts_har_rv_innovation_z",
        new_canonical="ts_har_rv_forecast_error_z",
        effective_dsl_version="1.0",
        semantic_equivalent=True,
        migration_rule="rename",
        deprecated_since="2026-08",
        remove_after="1.0",
    ),
)

# ---------------------------------------------------------------------------
# § 统计 / 回归（statistics_regression.py）
# ---------------------------------------------------------------------------
OperatorRegistry.register_alias("acf", "ACF")
OperatorRegistry.register_alias("mode", "Mode")
OperatorRegistry.register_alias("percentile", "ts_quantile")
# ---------------------------------------------------------------------------
# § 技术指标（technical_signal.py）
# factor_engine DSL 名 → cleaned canonical；实现见 technical_signal.py §1
# ---------------------------------------------------------------------------
OperatorRegistry.register_alias("ts_macd", "MACD")
OperatorRegistry.register_alias("ts_rsi", "RSI")
OperatorRegistry.register_alias("ts_rsi_wilder", "RSI_WILDER")
OperatorRegistry.register_alias("ts_adx", "ADX")
OperatorRegistry.register_alias("ts_adxr", "ADXR")
OperatorRegistry.register_alias("ts_aroon", "AROON")
OperatorRegistry.register_alias("ts_atr", "ATR")
OperatorRegistry.register_alias("ts_atr_wilder", "ATR_WILDER")
OperatorRegistry.register_alias("ts_bbands", "BollingerBands")
OperatorRegistry.register_alias("ts_cci", "CCI")
OperatorRegistry.register_alias("ts_mom", "MOM")
OperatorRegistry.register_alias("ts_obv", "OBV")
OperatorRegistry.register_alias("ts_roc", "ROC")
OperatorRegistry.register_alias("ts_stoch", "StochasticK")
OperatorRegistry.register_alias("ts_stochf", "StochasticD")
OperatorRegistry.register_alias("ts_trix", "TRIX")
OperatorRegistry.register_alias("ts_willr", "WilliamsR")
OperatorRegistry.register_alias("ts_wma", "WMA")
OperatorRegistry.register_alias("wma", "WMA")

# ---------------------------------------------------------------------------
# § 元素数学 / 清洗 / 截面 / 时序 / 基本面 等（其余模块）
# ---------------------------------------------------------------------------
OperatorRegistry.register_alias("ABS", "abs")
OperatorRegistry.register_alias("ts_return", "ts_pct")
OperatorRegistry.register_alias("COALESCE", "coalesce")
OperatorRegistry.register_alias("CS_DEMEAN", "cs_demean")
OperatorRegistry.register_alias("DECAY_LINEAR", "ts_decay_linear")
OperatorRegistry.register_alias("TS_DECAY_LINEAR", "ts_decay_linear")
OperatorRegistry.register_alias("FillForward", "ffill")
OperatorRegistry.register_alias("fillna_forward", "ffill")
OperatorRegistry.register_alias("FillNA", "fillna")
OperatorRegistry.register_alias("IS_NAN", "is_nan")
OperatorRegistry.register_alias("IS_NULL", "is_null")
OperatorRegistry.register_alias("is_null", "is_null")
OperatorRegistry.register_alias("IS_INFINITE", "is_infinite")
OperatorRegistry.register_alias("log_clip_domain", "protected_log")
OperatorRegistry.register_alias("IS_NOT_NULL", "is_not_null")
OperatorRegistry.register_alias("is_not_null", "is_not_null")
OperatorRegistry.register_alias("LOG", "log")
OperatorRegistry.register_alias("ln", "log")
OperatorRegistry.register_alias("NAN_TO_NUM", "nan_to_num")
OperatorRegistry.register_alias("POWER", "power")
OperatorRegistry.register_alias("QUARTER", "quarter")
OperatorRegistry.register_alias("quarter_from_cumulative", "quarter_from_cumulative")
OperatorRegistry.register_alias("ttm_from_quarterly", "ttm_from_quarterly")
OperatorRegistry.register_alias("ttm_from_cumulative", "ttm_from_cumulative")
OperatorRegistry.register_alias("yoy_by_period", "yoy_by_period")
OperatorRegistry.register_alias("CS_RANK", "rank")
OperatorRegistry.register_alias("RANK", "rank")
OperatorRegistry.register_alias("c_rank", "rank")
OperatorRegistry.register_alias("cs_rank", "rank")
OperatorRegistry.register_alias("RANKCORR", "rank_corr")
OperatorRegistry.register_alias("RANK_CORR", "rank_corr")
OperatorRegistry.register_alias("rankcorr", "rank_corr")
# log_returns 为独立 canonical，见 price_volume.py
OperatorRegistry.register_alias("FP_BETA", "rolling_beta_to_market")
OperatorRegistry.register_alias("ROLLING_BETA_TO_MARKET", "rolling_beta_to_market")
OperatorRegistry.register_alias("fp_beta", "rolling_beta_to_market")
OperatorRegistry.register_alias("ROLLING_BETA", "rolling_beta")
OperatorRegistry.register_alias("ts_rolling_beta", "rolling_beta")
OperatorRegistry.register_alias("ROUND", "round")
OperatorRegistry.register_alias("SCALE", "scale")
OperatorRegistry.register_alias("c_scale", "scale")
OperatorRegistry.register_alias("SIGN", "sign")
OperatorRegistry.register_alias("CAP_NEUTRALIZE", "size_neutralize")
OperatorRegistry.register_alias("MARKET_CAP_NEUTRALIZE", "size_neutralize")
OperatorRegistry.register_alias("SIZE_NEUTRALIZE", "size_neutralize")
OperatorRegistry.register_alias("SQRT", "sqrt")
OperatorRegistry.register_alias("ts_arg_max", "ts_argmax")
OperatorRegistry.register_alias("ts_arg_min", "ts_argmin")
OperatorRegistry.register_alias("TS_CORR", "ts_corr")
OperatorRegistry.register_alias("correlation", "ts_corr")
OperatorRegistry.register_alias("m_cor", "ts_corr")
OperatorRegistry.register_alias("ts_correlation", "ts_corr")
OperatorRegistry.register_alias("TS_COV", "ts_cov")
OperatorRegistry.register_alias("m_cov", "ts_cov")
OperatorRegistry.register_alias("ts_covariance", "ts_cov")
OperatorRegistry.register_alias("DELAY", "ts_delay")
OperatorRegistry.register_alias("Delay", "ts_delay")
OperatorRegistry.register_alias("Ref", "ts_delay")
OperatorRegistry.register_alias("delay", "ts_delay")
OperatorRegistry.register_alias("m_delay", "ts_delay")
OperatorRegistry.register_alias("shift", "ts_delay")
OperatorRegistry.register_alias("Delta", "ts_delta")
OperatorRegistry.register_alias("Diff", "ts_delta")
OperatorRegistry.register_alias("TS_DELTA", "ts_delta")
OperatorRegistry.register_alias("delta", "ts_delta")
OperatorRegistry.register_alias("CLIP", "clip")
OperatorRegistry.register_alias("cap", "clip")
OperatorRegistry.register_alias("clamp", "clip")
OperatorRegistry.register_alias("ma", "ts_mean")
OperatorRegistry.register_alias("sum_n", "ts_sum")
OperatorRegistry.register_alias("std_n", "ts_std")
OperatorRegistry.register_alias("industry_neutral", "group_neutralize")
OperatorRegistry.register_alias("ind_neutralize", "group_neutralize")
# ``group_rank_linear_weighted_value`` / ``event_compounded_return`` 别名在其
# canonical 定义模块内注册（group.py / relation/ops.py），因为这些 canonical 在
# _aliases 导入之后才被加载。
OperatorRegistry.register_alias("market_cap_neutralize", "size_neutralize")
OperatorRegistry.register_alias("cap_neutralize", "size_neutralize")
# LQTP-aligned primary names are applied in _dedupe.py (ts_ema→ema, clip→cap, …).
# Keep pre-dedupe aliases pointed at pre-rename canonicals so rename_canonical can retarget them.
OperatorRegistry.register_alias("m_pct_change", "ts_pct")
OperatorRegistry.register_alias("m_percentile", "ts_quantile")
OperatorRegistry.register_alias("m_argmax", "ts_argmax")
OperatorRegistry.register_alias("m_argmin", "ts_argmin")
OperatorRegistry.register_alias("m_top_n_sum", "ts_topk_sum")
OperatorRegistry.register_alias("TS_TOPK_SUM", "ts_topk_sum")
OperatorRegistry.register_alias("TS_KURT", "ts_kurt")
OperatorRegistry.register_alias("Max", "ts_max")
OperatorRegistry.register_alias("TS_MAX", "ts_max")
OperatorRegistry.register_alias("m_max", "ts_max")
OperatorRegistry.register_alias("max", "flex_max")
OperatorRegistry.register_alias("Mean", "ts_mean")
OperatorRegistry.register_alias("TS_MEAN", "ts_mean")
OperatorRegistry.register_alias("m_avg", "ts_mean")
OperatorRegistry.register_alias("mean", "ts_mean")
OperatorRegistry.register_alias("Min", "ts_min")
OperatorRegistry.register_alias("TS_MIN", "ts_min")
OperatorRegistry.register_alias("m_min", "ts_min")
OperatorRegistry.register_alias("min", "flex_min")
OperatorRegistry.register_alias("TS_PCT", "ts_pct")
OperatorRegistry.register_alias("TS_QUANTILE", "ts_quantile")
OperatorRegistry.register_alias("TS_RANK", "ts_rank")
OperatorRegistry.register_alias("m_rank", "ts_rank")
OperatorRegistry.register_alias("TS_REGRESSION_SLOPE", "ts_regression")
# ts_regression_slope is the post-dedupe canonical; do not point it back to ts_regression.

OperatorRegistry.register_alias("TS_SKEW", "ts_skew")
OperatorRegistry.register_alias("Std", "ts_std")
OperatorRegistry.register_alias("TS_STD", "ts_std")
OperatorRegistry.register_alias("m_std", "ts_std")
OperatorRegistry.register_alias("std", "ts_std")
OperatorRegistry.register_alias("ts_std_dev", "ts_std")
OperatorRegistry.register_alias("ts_stddev", "ts_std")
OperatorRegistry.register_alias("TS_SUM", "ts_sum")
OperatorRegistry.register_alias("m_sum", "ts_sum")
OperatorRegistry.register_alias("TTM", "ttm")
OperatorRegistry.register_alias("IIF", "where")
OperatorRegistry.register_alias("WHERE", "where")
OperatorRegistry.register_alias("if", "where")
OperatorRegistry.register_alias("iif", "where")
OperatorRegistry.register_alias("WINSORIZE", "winsorize")
OperatorRegistry.register_alias("YOY", "yoy")
OperatorRegistry.register_alias("CS_ZSCORE", "zscore")
OperatorRegistry.register_alias("ZSCORE", "zscore")
OperatorRegistry.register_alias("c_zscore", "zscore")
OperatorRegistry.register_alias("cs_zscore", "zscore")

# intraday_return 是 open_close_return 的挖掘侧别名（close/open - 1）。
OperatorRegistry.register_alias("intraday_return", "open_close_return")

# ---------------------------------------------------------------------------
# Review #4 R4-22 / R4-101 (governance verification — no behavioural change).
# ---------------------------------------------------------------------------
# R4-22 (legacy fiscal names are compat, not a second implementation):
#   * ``period_lag`` is registered by ``common/daily_panel`` as a legacy
#     implementation, but ``load_all()`` pops its pandas/polars backends and
#     ``fiscal_strict.register()`` (then the overhaul audit layer) re-registers
#     the SAME canonical with the strict ``(x, period_id, periods,
#     revision_policy)`` signature.  At runtime there is exactly ONE
#     implementation; the daily-panel legacy name never survives as a second
#     implementation.  Confirmed via ``OperatorRegistry.backends_for("period_lag")``.
#   * ``fin_mad`` / ``fin_mean_abs_deviation`` (P1-41 naming): the fundamental
#     module registers BOTH names pointing at the SAME kernel function object
#     (source ``fundamental_transforms_v2``), so the "old name" is a compat
#     name, not a second divergent implementation.  They remain separate
#     canonicals by design in ``fundamental/transforms_v2._SPECS``; making
#     ``fin_mad`` a true registry alias would require unregistering the
#     canonical there (a ``_dedupe`` concern, out of this file's scope).
# R4-101 (no blanket alias bypass): every alias in this module (and in the
#   curated dicts of ``alpha_language_aliases`` / ``layer_governance``) points
#   at one explicit canonical.  No loop over a family auto-registers aliases;
#   ``OperatorRegistry.finalize()`` still hard-fails on dangling / non-flattened
#   aliases.

# dedupe 在 load_all() 全部模块加载后执行，见 cleaned_operators.__init__.load_all
