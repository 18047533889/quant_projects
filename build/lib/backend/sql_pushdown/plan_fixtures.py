# -*- coding: utf-8
"""SQL capability / emitter 测试用最小 PlanNode 构造。

为 registry 中每个 canonical 生成可编译的最小逻辑计划，供能力探测与
emitter 回归测试复用。
"""
from __future__ import annotations

from planner.logical_plan import PlanNode


def column(name: str) -> PlanNode:
    """构造列引用节点 ``column(name)``。"""
    return PlanNode(op="column", attrs={"name": name})


def literal(value) -> PlanNode:
    """构造字面量节点 ``literal(value)``。"""
    return PlanNode(op="literal", attrs={"value": value})


def _minimal_plan_raw(op: str) -> PlanNode:
    """为 registry 中指定 canonical 构造可编译的最小逻辑计划。"""
    close, volume, industry = column("close"), column("volume"), column("industry")
    high, low = column("high"), column("low")
    open_ = column("open")
    # 4-input OHLC candle ops
    if op in {
        "candle_body_ratio",
        "candle_upper_shadow_ratio",
        "candle_lower_shadow_ratio",
        "candle_body_position",
        "candle_rejection_upper",
        "candle_rejection_lower",
        "candle_range_atr",
        "candle_gap_atr",
    }:
        if op in {"candle_range_atr", "candle_gap_atr"}:
            return PlanNode(op=op, inputs=[open_, high, low, close, literal(14.0)], attrs={"atr_window": 14})
        return PlanNode(op=op, inputs=[open_, high, low, close], attrs={})
    if op == "candle_upper_shadow":
        return PlanNode(op=op, inputs=[open_, high, close], attrs={})
    if op == "candle_lower_shadow":
        return PlanNode(op=op, inputs=[open_, low, close], attrs={})
    if op in {"candle_close_location", "candle_close_strength"}:
        return PlanNode(op=op, inputs=[high, low, close], attrs={})
    if op.startswith("cdl_"):
        return PlanNode(op=op, inputs=[open_, high, low, close], attrs={})
    if op in {"candle_body_zscore", "candle_body_percentile"}:
        return PlanNode(op=op, inputs=[open_, close, literal(10.0)], attrs={"window": 10})
    if op in {"candle_range_zscore", "candle_range_percentile"}:
        return PlanNode(op=op, inputs=[high, low, literal(10.0)], attrs={"window": 10})
    if op in {"candle_upper_shadow_zscore", "candle_lower_shadow_zscore"}:
        cols = [open_, high, close] if op == "candle_upper_shadow_zscore" else [open_, low, close]
        return PlanNode(op=op, inputs=cols + [literal(10.0)], attrs={"window": 10})
    if op == "abs_return_volume_corr":
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op == "ichimoku_tenkan":
        return PlanNode(op=op, inputs=[high, low, literal(9.0)], attrs={"tenkan_window": 9})
    if op == "ichimoku_kijun":
        return PlanNode(op=op, inputs=[high, low, literal(26.0)], attrs={"kijun_window": 26})
    if op == "ichimoku_senkou_a":
        return PlanNode(op=op, inputs=[high, low, literal(9.0), literal(26.0)], attrs={"tenkan_window": 9, "kijun_window": 26})
    if op == "ichimoku_senkou_b":
        return PlanNode(op=op, inputs=[high, low, literal(52.0)], attrs={"senkou_b_window": 52})
    if op == "ichimoku_cloud_width":
        return PlanNode(op=op, inputs=[high, low, literal(9.0), literal(26.0), literal(52.0)], attrs={"tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52})
    if op == "ichimoku_cloud_position":
        return PlanNode(op=op, inputs=[high, low, close, literal(9.0), literal(26.0), literal(52.0)], attrs={"tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52})
    if op == "efficiency_ratio":
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op == "choppiness_index":
        return PlanNode(op=op, inputs=[high, low, close, literal(20.0)], attrs={"window": 20})
    if op == "coskewness_to_market":
        return PlanNode(op=op, inputs=[close, volume, literal(60.0)], attrs={"window": 60})
    if op in {"ts_valid_count", "ts_coverage_ratio", "ts_abs_concentration"}:
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(1.0)], attrs={"window": 20, "min_periods": 1})
    if op == "ts_abs_entropy":
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(1.0), literal(1.0)], attrs={"window": 20, "normalize": 1, "min_periods": 1})
    if op in {"ts_downside_deviation", "ts_upside_deviation"}:
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(0.0), literal(2.0)], attrs={"window": 20, "target": 0.0, "min_periods": 2})
    if op == "ts_impulse_return":
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op == "ts_impulse_strength":
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(20.0)], attrs={"window": 20, "vol_window": 20})
    if op == "ts_impulse_volume":
        return PlanNode(op=op, inputs=[volume, literal(20.0), literal(20.0)], attrs={"window": 20, "baseline_window": 20})
    if op in {"ts_prev_high", "ts_prev_low", "ts_distance_to_high", "ts_distance_to_low",
              "ts_breakout_high", "ts_breakdown_low", "ts_channel_position", "ts_new_high", "ts_new_low"}:
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op in {"ts_argmax_age", "ts_argmin_age", "ts_argmax_index_from_oldest", "ts_argmin_index_from_oldest"}:
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(1.0)], attrs={"window": 20, "min_periods": 1})
    if op == "ts_staleness":
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op in {"ts_days_since_high", "ts_days_since_low"}:
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op in {"cs_valid_count", "cs_coverage_ratio", "cs_fill_mean", "cs_fill_median",
              "cs_impute_mean", "cs_impute_median", "cs_residual_percentile"}:
        return PlanNode(op=op, inputs=[close], attrs={})
    if op in {"cs_weighted_mean", "cs_weighted_demean", "cs_weighted_zscore"}:
        return PlanNode(op=op, inputs=[close, volume], attrs={})
    if op in {"candle_body", "candle_abs_body", "candle_gap", "candle_gap_pct", "candle_direction"}:
        return PlanNode(op=op, inputs=[open_, close], attrs={})
    if op in {"candle_range", "candle_overlap_ratio", "candle_inside_ratio"}:
        return PlanNode(op=op, inputs=[high, low], attrs={})
    if op in {"open_close_return", "overnight_return", "open_to_vwap_return", "vwap_to_close_return"}:
        return PlanNode(op=op, inputs=[open_, close], attrs={})
    if op in {"limit_up_close", "limit_down_close"}:
        return PlanNode(op=op, inputs=[close, high, literal(0.005)], attrs={"tick_tolerance": 0.005})
    if op == "true_range":
        return PlanNode(op=op, inputs=[high, low, close], attrs={})
    if op in {"parkinson_vol", "high_low_spread_proxy"}:
        return PlanNode(op=op, inputs=[high, low, literal(20.0)], attrs={"window": 20})
    if op == "overnight_volatility":
        return PlanNode(op=op, inputs=[open_, close, literal(20.0)], attrs={"window": 20})
    if op == "intraday_volatility":
        return PlanNode(op=op, inputs=[open_, close, literal(20.0)], attrs={"window": 20})
    if op == "range_volatility":
        return PlanNode(op=op, inputs=[high, low, close, literal(20.0)], attrs={"window": 20})
    if op == "ulcer_index":
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op in {"garman_klass_vol", "rogers_satchell_vol", "yang_zhang_vol"}:
        return PlanNode(op=op, inputs=[open_, high, low, close, literal(20.0)], attrs={"window": 20})
    # volume / turnover rolling ops
    if op == "volume_to_range":
        return PlanNode(op=op, inputs=[volume, high, low, literal(20.0)], attrs={"window": 20})
    if op in {"average_volume", "average_turnover", "relative_volume",
              "volume_zscore", "turnover_zscore", "volume_shock", "turnover_shock",
              "volume_momentum", "turnover_momentum", "volume_volatility", "turnover_volatility",
              "volume_autocorr", "turnover_autocorr", "abnormal_volume", "abnormal_turnover",
              "zero_return_ratio", "roll_spread_proxy"}:
        return PlanNode(op=op, inputs=[volume, literal(20.0), literal(1.0)], attrs={"window": 20, "lag": 1})
    if op in {"volume_acceleration", "turnover_acceleration"}:
        return PlanNode(op=op, inputs=[volume, literal(5.0), literal(20.0)], attrs={"short_window": 5, "long_window": 20})
    if op in {"up_volume_ratio", "down_volume_ratio", "signed_volume_imbalance", "up_down_volume_ratio",
              "volume_weighted_return", "volume_weighted_momentum", "return_volume_corr",
              "return_volume_beta", "return_turnover_beta", "price_volume_divergence",
              "price_turnover_divergence", "turnover_adjusted_volatility"}:
        return PlanNode(op=op, inputs=[close, volume, literal(20.0), literal(20.0)], attrs={"window": 20})
    if op in {"rolling_vwap", "vwap_deviation", "rolling_obv", "rolling_pvt"}:
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op == "signed_volume":
        return PlanNode(op=op, inputs=[close, volume], attrs={})
    if op == "signed_dollar_volume":
        return PlanNode(op=op, inputs=[close, close, volume], attrs={})
    if op in {"dollar_volume", "adv"}:
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op == "dollar_volume_zscore":
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op == "amihud_illiquidity":
        return PlanNode(op=op, inputs=[close, close, volume, literal(20.0)], attrs={"window": 20})
    if op == "price_impact":
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op == "return_per_turnover":
        return PlanNode(op=op, inputs=[close, volume], attrs={})
    if op == "corwin_schultz_spread":
        return PlanNode(op=op, inputs=[high, low, literal(20.0)], attrs={"window": 20})
    if op in {"bounded_nvi", "bounded_pvi"}:
        return PlanNode(op=op, inputs=[close, volume, literal(20.0)], attrs={"window": 20})
    if op in {"donchian_upper", "donchian_lower"}:
        return PlanNode(op=op, inputs=[high if op == "donchian_upper" else low, literal(20.0)], attrs={"window": 20})
    if op == "donchian_mid":
        return PlanNode(op=op, inputs=[high, low, literal(20.0)], attrs={"window": 20})
    if op == "donchian_position":
        return PlanNode(op=op, inputs=[close, high, low, literal(20.0)], attrs={"window": 20})
    if op in {"bollinger_pct_b", "bollinger_width"}:
        return PlanNode(op=op, inputs=[close, literal(20.0), literal(2.0)], attrs={"window": 20, "std_dev": 2.0})
    if op in {"MACD_line", "MACD_signal", "MACD_hist"}:
        return PlanNode(op=op, inputs=[close, literal(12.0), literal(26.0), literal(9.0)], attrs={"fast": 12, "slow": 26, "signal": 9})
    if op in {"DEMA", "TEMA"}:
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"window": 20})
    if op in {"PPO", "PVO"}:
        return PlanNode(op=op, inputs=[close, literal(12.0), literal(26.0)], attrs={"fast_window": 12, "slow_window": 26})
    if op in {"PPO_signal", "PPO_hist", "PVO_signal", "PVO_hist"}:
        return PlanNode(op=op, inputs=[close, literal(12.0), literal(26.0), literal(9.0)], attrs={"fast_window": 12, "slow_window": 26, "signal_window": 9})
    if op in {"TSI", "TSI_signal"}:
        return PlanNode(op=op, inputs=[close, literal(25.0), literal(13.0), literal(9.0)], attrs={"long_window": 25, "short_window": 13, "signal_window": 9})
    if op in {"CMO", "RSI_WILDER", "ForceIndex"}:
        if op == "ForceIndex":
            return PlanNode(op=op, inputs=[close, volume, literal(13.0)], attrs={"window": 13})
        return PlanNode(op=op, inputs=[close, literal(14.0)], attrs={"window": 14})
    if op in {"VortexPlus", "VortexMinus", "DMI_plus", "DMI_minus", "DX", "ADX"}:
        return PlanNode(op=op, inputs=[high, low, close, literal(14.0)], attrs={"window": 14})
    if op in {"KeltnerMid"}:
        return PlanNode(op=op, inputs=[close, literal(20.0)], attrs={"ema_window": 20})
    if op in {"KeltnerUpper", "KeltnerLower", "KeltnerPosition"}:
        return PlanNode(op=op, inputs=[high, low, close, literal(20.0), literal(14.0), literal(2.0)], attrs={"ema_window": 20, "atr_window": 14, "multiplier": 2.0})
    if op == "UltimateOscillator":
        return PlanNode(op=op, inputs=[high, low, close, literal(7.0), literal(14.0), literal(28.0), literal(4.0), literal(2.0), literal(1.0)], attrs={"short_window": 7, "medium_window": 14, "long_window": 28, "short_weight": 4.0, "medium_weight": 2.0, "long_weight": 1.0})
    if op in {"ADL", "CMF", "MFI"}:
        return PlanNode(op=op, inputs=[high, low, close, volume, literal(14.0)], attrs={"window": 14})
    if op == "ChaikinOscillator":
        return PlanNode(op=op, inputs=[high, low, close, volume, literal(3.0), literal(10.0), literal(20.0)], attrs={"fast_window": 3, "slow_window": 10, "adl_window": 20})
    if op == "EaseOfMovement":
        return PlanNode(op=op, inputs=[high, low, volume, literal(14.0), literal(1.0)], attrs={"window": 14, "volume_scale": 1.0})
    if op == "atan2":
        return PlanNode(op=op, inputs=[close, volume], attrs={})
    if op in {"lerp"}:
        return PlanNode(op=op, inputs=[close, volume, literal(0.5)], attrs={"fraction": 0.5})
    if op in {"round", "signed_power"}:
        return PlanNode(op=op, inputs=[close, literal(2.0)], attrs={"decimals": 2, "c": 2})
    if op == "NATR":
        return PlanNode(op=op, inputs=[high, low, close], attrs={"window": 14})
    if op in {
        "add",
        "subtract",
        "multiply",
        "divide",
        "gt",
        "lt",
        "eq",
        "ge",
        "le",
        "ne",
        "and_",
        "or_",
        "ts_corr",
        "ts_cov",
        "ts_beta",
        "ewm_corr",
        "ewm_cov",
        "ts_ewm_corr",
        "ts_ewm_cov",
        "protected_div",
        "div_or_default",
        "safe_div_null",
        "fin_ratio",
        "float_share_ratio",
        "free_float_share_ratio",
        "benchmark_relative_price",
        "holder_concentration",
        "benchmark_excess_return",
        "ashare_limit_distance",
        "ts_regression_slope",
        "Slope",
        "power",
        "rolling_beta",
        "maximum",
        "minimum",
    }:
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 3, "window": 3, "span": 3})
    if op in {"cs_resid", "cs_regression", "size_neutralize"}:
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 3})
    if op == "industry_size_neutralize":
        return PlanNode(op=op, inputs=[close, industry, volume], attrs={})
    if op in {"ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if"}:
        return PlanNode(
            op=op,
            inputs=[close, volume],
            attrs={"window": 3, "min_periods": 1 if op != "ts_std_if" else 2},
        )
    if op == "cs_multi_resid":
        return PlanNode(
            op=op,
            inputs=[close, volume, high],
            attrs={"min_obs": 4, "add_intercept": True},
        )
    if op == "cs_wls_resid":
        return PlanNode(
            op=op,
            inputs=[close, volume, high],
            attrs={"min_obs": 4, "add_intercept": True},
        )
    if op == "period_lag":
        return PlanNode(op=op, inputs=[close, industry], attrs={"periods": 1})
    if op in {"period_change", "period_average", "period_cagr", "ttm_from_quarterly", "yoy_by_period"}:
        return PlanNode(op=op, inputs=[close, industry], attrs={"periods": 4})
    if op in {"quarter_from_cumulative", "ttm_from_cumulative"}:
        return PlanNode(op=op, inputs=[close, industry, high], attrs={})
    if op == "ts_regression_tstat":
        return PlanNode(
            op=op,
            inputs=[close, volume],
            attrs={"window": 5, "min_periods": 3, "add_intercept": True},
        )
    if op == "ts_partial_corr":
        return PlanNode(
            op=op,
            inputs=[close, volume, high],
            attrs={"window": 5, "min_periods": 3},
        )
    # Multi-input SQL-pushdown ops that need more than the generic fallback's
    # single frame (the emitter branches require 2-3 compiled child layers).
    if op in {"ts_crossing_speed", "ts_crossing_acceleration"}:
        return PlanNode(op=op, inputs=[close, volume], attrs={"window": 20})
    if op == "ts_cov_if":
        return PlanNode(op=op, inputs=[close, volume, high], attrs={"window": 20, "min_periods": 2})
    if op == "ts_value_at_argextreme":
        return PlanNode(op=op, inputs=[close, volume], attrs={"window": 20, "mode": "max", "include_current": False})
    if op == "ts_weighted_standardized_moment":
        return PlanNode(op=op, inputs=[close, volume], attrs={"window": 20, "order": 3})
    if op == "ts_abdi_ranaldo_spread":
        return PlanNode(op=op, inputs=[close, high, low], attrs={"window": 20})
    if op == "cs_weighted_percentile_rank":
        return PlanNode(op=op, inputs=[close, volume], attrs={})
    if op == "group_topk_mean":
        return PlanNode(op=op, inputs=[close, volume, industry], attrs={"k": 3, "exclude_self": True})
    if op == "group_weighted_mean":
        return PlanNode(op=op, inputs=[close, industry, volume], attrs={})
    if op.startswith("group_"):
        return PlanNode(op=op, inputs=[close, industry], attrs={"d": 3, "window": 3, "p": 0.5})
    if op in {"where", "if_else", "coalesce"}:
        return PlanNode(op=op, inputs=[close, volume, literal(0.0)], attrs={})
    if op == "clip":
        return PlanNode(op=op, inputs=[close], attrs={"min": 0.0, "max": 1.0})
    if op in {"winsorize"}:
        return PlanNode(op=op, inputs=[close], attrs={"lower": 0.01, "upper": 0.99, "a": 0.05})
    if op in {"rank_pct", "cs_pct_rank", "log_returns", "ts_log_return"}:
        return PlanNode(op=op, inputs=[close], attrs={})
    if op in {"cs_quantile", "c_percentile"}:
        return PlanNode(op=op, inputs=[close], attrs={"p": 0.5})
    if op == "volatility":
        return PlanNode(op=op, inputs=[close], attrs={"d": 20, "window": 20})
    if op == "vwap":
        return PlanNode(op=op, inputs=[close, volume], attrs={"d": 20, "window": 20})
    if op in {"fillna", "fillna_const", "nan_to_num", "bfill", "ffill"}:
        return PlanNode(op=op, inputs=[close], attrs={"value": 0, "limit": 1})
    if op == "ts_quantile":
        return PlanNode(op=op, inputs=[close], attrs={"d": 3, "q": 0.5})
    if op in {"ts_sharpe", "ts_autocorr"}:
        return PlanNode(op=op, inputs=[close], attrs={"d": 20, "window": 20})
    if op in {"ts_ema", "ewm_std", "ewm_var", "ts_ewm_std", "ts_ewm_var", "WMA", "ts_decay_linear"}:
        return PlanNode(op=op, inputs=[close], attrs={"span": 5, "window": 5, "d": 5})
    if op == "ATR_WILDER":
        return PlanNode(op=op, inputs=[high, low, close], attrs={"d": 14, "window": 14})
    if op == "RSI_WILDER":
        return PlanNode(op=op, inputs=[close], attrs={"d": 14, "window": 14})
    if op == "not_":
        return PlanNode(op=op, inputs=[close], attrs={})
    if op == "index_weight":
        return PlanNode(op=op, inputs=[close], attrs={"normalize": True})
    return PlanNode(op=op, inputs=[close], attrs={"d": 3, "window": 3, "span": 3})


def minimal_plan(op: str) -> PlanNode:
    """Build a fixture whose production parameters use canonical spellings only."""
    plan = _minimal_plan_raw(op)
    from backend.parameter_aliases import normalize_parameter_aliases
    from backend.production_signature import PRODUCTION_SIGNATURES

    signature = PRODUCTION_SIGNATURES.get(op)
    if signature is None:
        return plan
    attrs = dict(plan.attrs)
    if op == "clip":
        if "min" in attrs:
            attrs["lo"] = attrs.pop("min")
        if "max" in attrs:
            attrs["hi"] = attrs.pop("max")
    attrs = normalize_parameter_aliases(op, attrs)
    allowed = {constraint.name for constraint in signature.params}
    attrs = {name: value for name, value in attrs.items() if name in allowed}
    return PlanNode(op=plan.op, inputs=plan.inputs, attrs=attrs, node_id=plan.node_id)
