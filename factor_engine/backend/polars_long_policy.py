# -*- coding: utf-8
"""Polars long-table 能力分层与 strict fallback 策略。"""
from __future__ import annotations

import os

from factor_engine.planner.logical_plan import PlanNode

from .polars_registry_bridge import registry_op_long_capable

# 纯 Polars Expr（无 Python rolling_map / map_groups / pandas kernel）
POLARS_LONG_NATIVE: frozenset[str] = frozenset(
    {
        'abs',
        'add',
        'and_',
        'cs_count',
        'cs_mean',
        'cs_quantile',
        'cs_std',
        'cs_sum',
        'clip',
        'ceil',
        'coalesce',
        'column',
        'count',
        'cs_demean',
        'cs_mad',
        'cs_mad_zscore',
        'cs_pct_rank',
        'cs_regression',
        'cs_resid',
        'size_neutralize',
        'industry_size_neutralize',
        'cum_delta',
        'cum_max',
        'cum_min',
        'cum_prod',
        'cum_sum',
        'ts_delay',
        'div_or_default',
        'div_or_null',
        'divide',
        'eq',
        'ewm_std',
        'ewm_var',
        'exp',
        'expanding_mean',
        'expanding_sum',
        'ffill',
        'fillna',
        'fillna_const',
        'floor',
        'ge',
        'group_count',
        'group_decay_linear',
        'group_max',
        'group_mean',
        'group_min',
        'group_neutralize',
        'group_normalize',
        'group_percentile',
        'group_rank',
        'group_std',
        'group_sum',
        'group_winsorize',
        'group_zscore',
        'gt',
        'if_else',
        'inverse',
        'is_finite',
        'is_infinite',
        'is_nan',
        'is_not_null',
        'is_null',
        'le',
        'literal',
        'log',
        'log_abs',
        'log_fill_invalid',
        'log_returns',
        'ts_log_return',
        'lt',
        'materialized_series',
        'maximum',
        'minimum',
        'multiply',
        'nan_to_num',
        'ne',
        'neg',
        'normalize',
        'not_',
        'or_',
        'plan_ref',
        'power',
        'protected_div',
        'protected_log',
        'protected_sqrt',
        'rank',
        'rank_pct',
        'safe_div_null',
        'scale',
        'sign',
        'signed_log',
        'signed_sqrt',
        'sqrt',
        'subtract',
        'ts_autocorr',
        'ts_beta',
        'ts_corr',
        'ts_cov',
        'ts_delta',
        'ts_max',
        'ts_mean',
        'ts_median',
        'ts_min',
        'ts_pct',
        'ts_rank',
        'ts_ratio',
        'ts_sharpe',
        'ts_std',
        'ts_sum',
        'ts_var',
        'ts_zscore',
        'volatility',
        'vwap',
        'where',
        'winsorize',
        'zscore',
        # 2026-09-07 fin_* elementwise algebraic family (pure Expr, no period
        # walk): ratio / abs-ratio / sum-diff-ratio over up to seven operands.
        'fin_common_size',
        'fin_cash_conversion',
        'fin_acquisition_cash_intensity',
        'fin_borrowing_intensity',
        'fin_capex_intensity',
        'fin_goodwill_intensity',
        'fin_debt_repayment_intensity',
        'fin_contract_asset_intensity',
        'fin_contract_liability_intensity',
        'fin_oci_to_equity',
        'fin_interest_coverage_proxy',
        'fin_discontinued_operation_ratio',
        'fin_minority_profit_share',
        'fin_fair_value_income_dependence',
        'fin_investment_income_dependence',
        'fin_other_earnings_dependence',
        'fin_rd_capitalization_ratio',
        'fin_expectation_dispersion',
        'fin_cash_burn_runway',
        'fin_accrual_ratio',
        'fin_cash_earnings_gap',
        'fin_impairment_intensity',
        'fin_lease_intensity',
        'fin_rd_total_intensity',
        'fin_contract_asset_liability_gap',
        'fin_lease_asset_liability_gap',
        'fin_deferred_tax_gap',
        'fin_comprehensive_income_gap',
        'fin_roe_cash_gap',
        'fin_debt_service_coverage_proxy',
        'fin_actual_expectation_divergence',
        'fin_surprise',
        'fin_net_borrowing_cashflow',
        'fin_financing_gap',
        'fin_core_earnings_ratio',
        'fin_noncore_income_ratio',
        'cs_physical_panel_coverage',
        'group_rank_weighted_value',
        # wave2 csg (2026-09-07): polars native branches in
        # backend/polars_expr_emitter.py (cs_shrink / weighted zscore),
        # 三方 parity 见 tests/backend_parity/test_csg_wave2_parity.py.
        'cs_shrink_to_group_mean',
        'group_weighted_zscore',
        # wave2 fin/valuation (2026-09-07): 纯元素级比值（_FIN_ELEMENTWISE_OPS
        # 表复用，NumpyKernels/ops_v2 参考）。
        'free_float_turnover',
        'real_turnover_rate',
        'true_turnover_rate',
        'market_cap_free_cap_gap',
        'price_spread_deviation',
        # wave3 valuation/shareholder (2026-09-08): 纯元素级族（_FIN_ELEMENTWISE_OPS
        # 表复用）+ shift 分支（holder_pledge_change / circulating_cap_ratio_change）。
        'a_share_cap_ratio',
        'free_float_ratio',
        'holder_pledge_ratio',
        'holder_freeze_ratio',
        'holder_locked_share_ratio',
        'holder_float_concentration_gap',
        'holder_pledge_change',
        'val1_relative_valuation_gap',
        'valuation_pe_ttm_lyr_gap',
        'valuation_pcf_definition_gap',
        'valuation_pe_gap_signed_log',
        'valuation_pcf_gap_signed_log',
        'valuation_pe_gap_positive',
        'valuation_pcf_gap_positive',
        'circulating_cap_ratio_change',
        # wave2 ts (2026-09-07): overnight/intraday decomposition family —
        # polars native branches (safe_ret decomp + trailing stats),
        # 三方 parity 见 tests/backend_parity/test_ts_wave2_parity.py.
        'ts_overnight_intraday_cov',
        'ts_overnight_intraday_spread',
        'ts_overnight_intraday_sign_agreement',
        'ts_opening_mispricing_score',
        'cs_universe_coverage',
        # wave3 flow/momentum/quality window family (2026-09-08): trailing-window
        # statistics over the wave1_orderflow / wave1_cs_momentum /
        # wave1_earnings pandas references — polars native branches in
        # backend/polars_expr_emitter.py (_FLOWMOM_WINDOW_OPS block), 三方 parity
        # 见 tests/backend_parity/test_flowmom_wave3_parity.py.
        'ofi_volume_imbalance',
        'ofi_abs_imbalance_trend',
        'ofi_dominant_direction',
        'ofi_imbalance_agreement',
        'ofi_imbalance_cv',
        'ofi_volume_flow_regime',
        'ofi_zero_flow_balance',
        'm1_momentum_stability',
        'm1_volume_adjusted_momentum',
        'sv_net_flow_direction',
        'sv_own_flow_fraction',
        'sv_signed_volume_volatility',
        'sv_self_relative_change',
        'aq1_cash_flow_volatility',
        'aq1_cash_conversion_strength',
        'aq1_working_capital_accrual',
        # wave3d ashare limit family (2026-09-08): 涨跌停触碰/炸板/开板布尔
        # （_FIN_ELEMENTWISE_OPS 表复用）+ 滚动计数/密度/量比/连板 polars
        # native 分支，pandas 权威 ashare/limit_ops.py + ashare/state_machine.py，
        # 三方 parity 见 tests/backend_parity/test_ashare_wave3_parity.py.
        'ashare_limit_up_touch',
        'ashare_limit_down_touch',
        'ashare_open_at_upper_limit',
        'ashare_limit_failed',
        'ashare_limit_open_failed',
        'ashare_limit_touch_count',
        'ashare_failed_limit_count',
        'ashare_limit_asymmetry',
        'ashare_limit_event_density',
        'ashare_limit_up_volume_ratio',
        'ashare_limit_down_volume_ratio',
        'ashare_limit_up_streak',
        # wave3e cs/group family (2026-09-08): 固定边界分桶 / EB 收缩 / 组内
        # ex-self 加权均值 / 组有效成员诊断计数 / 多帧偏离指数 — polars native
        # 分支（backend/polars_expr_emitter.py wave3e 块），pandas 权威
        # cs_batch1 / group_ext / group_spectrum / overhaul.daily /
        # cross_section.peer_ops，三方 parity 见
        # tests/backend_parity/test_csgrp_wave3_parity.py.
        'cs_bucket_fixed',
        'cs_empirical_bayes_shrinkage',
        'group_ex_self_weighted_mean',
        'group_feature_valid_member_count',
        'group_peer_deviation_index',
        # wave3f ts2 (2026-09-08): prior-extreme / range / consolidation /
        # liquidity-beta family — polars native branches (backend/
        # polars_expr_emitter.py wave3f 块), pandas 权威 technical_extensions /
        # structure_patterns_v2 / peer_ops。ts_days_since_high/low 保持
        # registry bridge（argmax 位置语义无法纯 over 表达, DEFER）。
        # 三方 parity 见 tests/backend_parity/test_ts2_wave3_parity.py.
        'ts_prev_high',
        'ts_prev_low',
        'ts_distance_to_high',
        'ts_distance_to_low',
        'ts_breakout_high',
        'ts_breakdown_low',
        'ts_new_high',
        'ts_new_low',
        'ts_channel_position',
        'ts_range_expansion',
        'ts_consolidation_width',
        'ts_market_liquidity_beta',
        'ts_industry_liquidity_beta',
        # Tech / candle / misc family (2026-09 native branches in
        # backend/polars_expr_emitter.py — pure rolling/ewm shifted-window,
        # no Python UDF).  Sequential-recursion kernels (FisherTransform / QQE /
        # RSX) stay on the registry bridge (cap path), NOT native.
        "ALMA",
        "CoppockCurve",
        "ElderRay",
        "atr_acceleration",
        "atr_pct",
        "atr_percentile",
        "atr_short_long_ratio",
        "atr_zscore",
        "candle_body_strength",
        "candle_pattern_count",
        "candle_range_pct",
        "candle_wick_balance",
    }
)

POLARS_LONG_STATEFUL: frozenset[str] = frozenset(
    {
        'Supertrend',
        'SupertrendDirection',
        'PSAR',
        'ADX',
        'ADXR',
        'ATR_WILDER',
        'KAMA',
        'MACD',
        'MACD_hist',
        'MACD_line',
        'MACD_signal',
        'RSI_WILDER',
        'TRIX',
        'ts_ema',
        'vpmacd',
        'vpmacd_signal',
        # candlestick engine (multi-bar pattern state) + cdl_* 嵌套 prior-trend
        # 上下文（pandas 参考嵌入 LAG 序列，polars 走 registry native 复刻）。
        "candlestick_pattern",
        "cdl_hammer",
        "cdl_hanging_man",
    }
)

# 算法语义未冻结 / 非标准实现：仍可走 long path，但不得标为 native production
POLARS_LONG_NONSTANDARD_ALG: frozenset[str] = frozenset(
    {
        "ts_mad",
        "ts_regression_slope",
    }
)

# rolling_map + NumPy/pandas callback（LazyFrame 内仍含 Python UDF）
POLARS_LONG_PYTHON_ROLLING: frozenset[str] = frozenset(
    {
        "expanding_std",
        "ts_decay_linear",
        "WMA",
        "ts_time_slope",
        "Slope",
        "ts_skew",
        "ts_quantile",
        "ts_argmax",
        "ts_argmin",
        "ts_product",
        "ts_median_abs_deviation",
        "ts_mean_abs_deviation",
        # wave3 ts path/risk family (2026-09-08): rolling_map branches in
        # backend/polars_expr_emitter.py over the certified pandas kernels
        # (trailing-run / pairwise loops — not expressible as pure over Expr,
        # hence python_rolling and NOT native).  三方 parity 见
        # tests/backend_parity/test_ts_wave3_parity.py.
        "ts_monotonicity",
        "ts_turning_point_ratio",
        "ts_endpoint_deviation",
        "ts_vol_shift_score",
        "ts_time_under_water",
        "ts_current_drawdown_duration",
        "ts_recovery_fraction",
        # wave3 flowmom (2026-09-08): COMPACTED finite-slice kernels over the
        # wave1 pandas references — sign-reversal rate / lag-1 autocorr /
        # compound window product Π(1+r) (can go negative when r < -1, so
        # log-space is inexact).  rolling_map, NOT native.
        "ofi_reversal_rate",
        "ofi_imbalance_persistence",
        "m1_momentum_strength",
        "m1_momentum_speed_change",
    }
)

# map_groups / rolling_map(pandas) / 截面 Python 分箱等
POLARS_LONG_MAP_GROUPS: frozenset[str] = frozenset(
    {
        # alias names plus resolved canonicals (ewm_corr -> ts_ewm_corr)
        "ewm_corr",
        "ewm_cov",
        "ts_ewm_corr",
        "ts_ewm_cov",
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "expanding_rank",
        "causal_linear_extrapolate",
        "quantile",
        # wave3 flowmom (2026-09-08): ΔWC pairs over the COMPACTED ok slice —
        # pair membership depends on the consuming window start; needs
        # map_groups, NOT native.
        "aq1_accrual_stability",
        "aq1_accrual_ratio_dispersion",
    }
)

# R30 §2: bfill/causal_bfill were physically removed (``tombstones`` is the
# single authority).  No tier classifies them any more; the emitter's raise
# guards remain as migration errors for stale hand-built plans only.
POLARS_LONG_BLOCKED_CAUSAL: frozenset[str] = frozenset()

POLARS_LONG_PASSTHROUGH: frozenset[str] = frozenset()

POLARS_LONG_COMPATIBLE: frozenset[str] = (
    POLARS_LONG_NATIVE
    | POLARS_LONG_NONSTANDARD_ALG
    | POLARS_LONG_STATEFUL
    | POLARS_LONG_PYTHON_ROLLING
    | POLARS_LONG_MAP_GROUPS
    | POLARS_LONG_PASSTHROUGH
)

# 向后兼容旧名
POLARS_EXPR_CAPABLE = POLARS_LONG_COMPATIBLE

_POLARS_LONG_CAPABLE_CACHE: frozenset[str] | None = None


def get_polars_long_capable() -> frozenset[str]:
    """``POLARS_LONG_COMPATIBLE`` ∪ Registry polars bridge。"""
    global _POLARS_LONG_CAPABLE_CACHE
    if _POLARS_LONG_CAPABLE_CACHE is not None:
        return _POLARS_LONG_CAPABLE_CACHE
    from factor_engine.cleaned_operators import load_all

    load_all()
    from .polars_registry_bridge import polars_registry_long_capable

    _POLARS_LONG_CAPABLE_CACHE = frozenset(
        POLARS_LONG_COMPATIBLE | polars_registry_long_capable()
    )
    return _POLARS_LONG_CAPABLE_CACHE


POLARS_LONG_CAPABLE: frozenset[str] = POLARS_LONG_COMPATIBLE


def _resolve(op: str) -> str:
    """将算子别名解析为 canonical 名称。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._aliases.get(op, op)


def classify_plan_op(op: str) -> str:
    """返回 ``native`` | ``python_rolling`` | ``map_groups`` | ``passthrough`` | ``registry`` | ``other`` | ``meta``。"""
    from factor_engine.backend.production_fastpath_tiers import resolve_polars_native_canonical

    canon = resolve_polars_native_canonical(_resolve(op))
    if canon in {"column", "literal", "materialized_series", "plan_ref"}:
        return "meta"
    if canon in POLARS_LONG_BLOCKED_CAUSAL:
        return "blocked_causal"
    if canon in POLARS_LONG_NONSTANDARD_ALG:
        return "nonstandard_alg"
    if canon in POLARS_LONG_STATEFUL:
        return "stateful"
    if canon in POLARS_LONG_PASSTHROUGH:
        return "passthrough"
    if canon in POLARS_LONG_NATIVE:
        return "native"
    if canon in POLARS_LONG_PYTHON_ROLLING:
        return "python_rolling"
    if canon in POLARS_LONG_MAP_GROUPS:
        return "map_groups"
    if registry_op_long_capable(canon):
        return "registry"
    return "other"


def infer_polars_long_tier(op: str) -> str:
    """算子级 long-table 能力 tier。"""
    kind = classify_plan_op(op)
    if kind == "meta":
        return "meta"
    if kind == "blocked_causal":
        return "blocked_causal"
    if kind in {"native", "nonstandard_alg", "stateful", "python_rolling", "map_groups", "passthrough", "registry"}:
        return kind
    return "unsupported"


def collect_plan_op_stats(plan: PlanNode) -> dict[str, list[str]]:
    """遍历逻辑计划，按 PolarsLong tier 分类算子。

    参数:
        plan: 逻辑计划根节点。

    返回:
        含 ``polars_long_native_ops`` 等各 tier 算子列表的字典。
    """
    native: set[str] = set()
    nonstandard: set[str] = set()
    stateful: set[str] = set()
    python_rolling: set[str] = set()
    map_groups: set[str] = set()
    passthrough: set[str] = set()
    blocked_causal: set[str] = set()
    registry: set[str] = set()
    other: set[str] = set()

    def _walk(node: PlanNode) -> None:
        """递归遍历计划并按 tier 分类算子。"""
        canon = _resolve(node.op)
        kind = classify_plan_op(canon)
        if kind == "native" and canon not in {"column", "literal", "materialized_series", "plan_ref"}:
            native.add(canon)
        elif kind == "nonstandard_alg":
            nonstandard.add(canon)
        elif kind == "stateful":
            stateful.add(canon)
        elif kind == "python_rolling":
            python_rolling.add(canon)
        elif kind == "map_groups":
            map_groups.add(canon)
        elif kind == "passthrough":
            passthrough.add(canon)
        elif kind == "blocked_causal":
            blocked_causal.add(canon)
        elif kind == "registry":
            registry.add(canon)
        elif kind == "other" and canon not in {
            "column",
            "literal",
            "materialized_series",
            "plan_ref",
        }:
            other.add(canon)
        for child in node.inputs:
            _walk(child)

    _walk(plan)
    return {
        "polars_long_native_ops": sorted(native),
        "polars_long_nonstandard_alg_ops": sorted(nonstandard),
        "polars_long_stateful_ops": sorted(stateful),
        "polars_long_python_rolling_ops": sorted(python_rolling),
        "polars_long_map_group_ops": sorted(map_groups),
        "polars_long_passthrough_ops": sorted(passthrough),
        "polars_long_blocked_causal_ops": sorted(blocked_causal),
        "polars_long_registry_ops": sorted(registry),
        "polars_long_other_ops": sorted(other),
    }


def long_path_telemetry_flags(op_stats: dict[str, list[str]]) -> dict[str, bool]:
    """从 ``collect_plan_op_stats`` 推断 runtime 布尔标记。"""
    has_map = bool(op_stats.get("polars_long_map_group_ops"))
    has_registry = bool(op_stats.get("polars_long_registry_ops"))
    has_other = bool(op_stats.get("polars_long_other_ops"))
    has_passthrough = bool(op_stats.get("polars_long_passthrough_ops"))
    has_python_rolling = bool(op_stats.get("polars_long_python_rolling_ops"))
    has_native = bool(op_stats.get("polars_long_native_ops"))
    fast_native = has_native and not (
        has_map or has_registry or has_other or has_passthrough or has_python_rolling
    )
    return {
        "used_polars_long_path": True,
        "used_polars_long_native": fast_native,
        "used_polars_long_python_rolling": has_python_rolling,
        "used_polars_long_map_groups": has_map,
        "used_polars_long_registry": has_registry,
        "used_polars_long_passthrough": has_passthrough,
    }


def strict_polars_long_fallback(ctx=None) -> bool:
    """production 或 ``FACTOR_ENGINE_STRICT_POLARS_LONG=1`` 时禁止 silent fallback。

    R7-246: the ExecutionContext's own ``run_mode`` is authoritative FIRST — a
    caller running with ``ctx.run_mode="production"`` must be strict even when a
    global environment flag says otherwise (and vice versa).  Resolution order:
    1. ``ctx.run_mode`` (the caller's own mode, most specific);
    2. ``FACTOR_ENGINE_STRICT_POLARS_LONG`` env override (explicit user intent);
    3. ``ctx.runtime_stats["run_mode"]`` (legacy embedded stats);
    4. global ``is_production_mode()``.
    """
    ctx_mode = getattr(ctx, "run_mode", None) if ctx is not None else None
    if ctx_mode is not None:
        return str(ctx_mode).lower() == "production"
    # R9-P0-012 (production is ALWAYS strict): the env "off" branch below must
    # only ever apply to a NON-production context.  The old order let
    # ``FACTOR_ENGINE_STRICT_POLARS_LONG=0`` (or an env-var typo) make a
    # production process non-strict when the caller passed no ctx.  An env var
    # may make RESEARCH stricter (``=1``) but can never loosen production.
    from factor_engine.runtime.production_policy import is_production_mode

    if is_production_mode():
        return True
    raw = os.environ.get("FACTOR_ENGINE_STRICT_POLARS_LONG", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    runtime = getattr(ctx, "runtime_stats", None) if ctx is not None else None
    if isinstance(runtime, dict):
        mode = runtime.get("run_mode")
        if mode is not None:
            return str(mode).lower() == "production"

    return is_production_mode()


class PolarsLongStrictError(RuntimeError):
    """Strict 模式下 long native 失败，禁止 fallback。"""


class UnsupportedCausalOperatorError(PolarsLongStrictError):
    """bfill / causal_bfill 等在 fast path 禁止 silent no-op。"""


class PolarsLongNativeRequiredError(PolarsLongStrictError):
    """``FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE=1`` 时 plan 含非 native 算子。"""


def require_polars_long_native_only(ctx=None) -> bool:
    """仅允许 ``POLARS_LONG_NATIVE`` + meta；禁止 map_groups / registry / passthrough。"""
    raw = os.environ.get("FACTOR_ENGINE_POLARS_LONG_REQUIRE_NATIVE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def plan_contains_blocked_causal(plan) -> bool:
    """计划树是否含 ``bfill`` / ``causal_bfill``。"""
    from factor_engine.planner.logical_plan import PlanNode

    if not isinstance(plan, PlanNode):
        return False
    canon = _resolve(str(plan.op))
    if canon in POLARS_LONG_BLOCKED_CAUSAL:
        return True
    return any(plan_contains_blocked_causal(c) for c in plan.inputs)


def assert_no_blocked_causal_plan(plan, *, backend: str) -> None:
    """fast path 遇 blocked causal 算子时立即失败。"""
    if plan_contains_blocked_causal(plan):
        raise UnsupportedCausalOperatorError(
            f"{backend} 不支持 bfill/causal_bfill（因果占位，非传统 backward fill）；请改用 ffill 或 pandas 研究路径"
        )


def assert_native_only_plan(op_stats: dict[str, list[str]], ctx=None) -> None:
    """REQUIRE_NATIVE 模式下，plan 不得含 map_groups / registry / passthrough。"""
    if not require_polars_long_native_only(ctx):
        return
    blocked: list[str] = []
    for key, label in (
        ("polars_long_map_group_ops", "map_groups"),
        ("polars_long_python_rolling_ops", "python_rolling"),
        ("polars_long_registry_ops", "registry"),
        ("polars_long_passthrough_ops", "passthrough"),
        ("polars_long_other_ops", "other"),
    ):
        ops = op_stats.get(key) or []
        if ops:
            blocked.append(f"{label}: {','.join(ops)}")
    if blocked:
        raise PolarsLongNativeRequiredError(
            "polars_long require_native: plan uses non-native ops — " + "; ".join(blocked)
        )
