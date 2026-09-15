from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()


def _panels(rows: int = 48):
    index = pd.date_range("2026-01-01", periods=rows)
    close = pd.DataFrame({"A": np.linspace(10.0, 20.0, rows)}, index=index)
    high = close + 1.0
    low = close - 1.0
    volume = pd.DataFrame({"A": np.linspace(100.0, 200.0, rows)}, index=index)
    return high, low, close, volume


@pytest.mark.parametrize(
    ("name", "panels", "scalars"),
    [
        ("DMI_plus", ("high", "low", "close"), ("window",)),
        ("DMI_minus", ("high", "low", "close"), ("window",)),
        ("DX", ("high", "low", "close"), ("window",)),
        ("CMO", ("close",), ("window",)),
        ("DEMA", ("x",), ("window",)),
        ("KAMA", ("close",), ("er_window", "fast_window", "slow_window")),
        ("CMF", ("high", "low", "close", "volume"), ("window",)),
        (
            "ChaikinOscillator",
            ("high", "low", "close", "volume"),
            ("fast_window", "slow_window", "adl_window"),
        ),
    ],
)
def test_priority_operator_declares_authoritative_panel_scalar_topology(name, panels, scalars):
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    assert op.metadata.panel_params == panels
    assert op.metadata.scalar_params == scalars
    assert set(op.metadata.param_specs) == set(scalars)


def test_priority_kernels_execute_real_values_and_reject_invalid_parameters():
    high, low, close, volume = _panels()
    calls = [
        ("DMI_plus", (high, low, close), {"window": 5}),
        ("DMI_minus", (high, low, close), {"window": 5}),
        ("DX", (high, low, close), {"window": 5}),
        ("CMO", (close,), {"window": 5}),
        ("DEMA", (close,), {"window": 5}),
        ("KAMA", (close,), {"er_window": 5, "fast_window": 2, "slow_window": 10}),
        ("CMF", (high, low, close, volume), {"window": 5}),
        (
            "ChaikinOscillator",
            (high, low, close, volume),
            {"fast_window": 2, "slow_window": 5, "adl_window": 5},
        ),
    ]
    outputs = [
        OperatorRegistry.get(name, backend="pandas_numpy", mode="research").calculate(
            *panels, **parameters
        )
        for name, panels, parameters in calls
    ]
    assert all(output.shape == close.shape for output in outputs)
    assert all(np.isfinite(output.to_numpy()).any() for output in outputs)
    with pytest.raises(ValueError):
        OperatorRegistry.get("CMO", backend="pandas_numpy", mode="research").calculate(close, window=1)
    with pytest.raises(ValueError):
        OperatorRegistry.get("KAMA", backend="pandas_numpy", mode="research").calculate(
            close, er_window=5, fast_window=10, slow_window=2
        )
    with pytest.raises(ValueError):
        OperatorRegistry.get("ChaikinOscillator", backend="pandas_numpy", mode="research").calculate(
            high, low, close, volume, fast_window=5, slow_window=2, adl_window=5
        )


def test_atr_wilder_contract_and_execution():
    high, low, close, _ = _panels()
    op = OperatorRegistry.get("ATR_WILDER", backend="pandas_numpy", mode="research")
    assert op.metadata.panel_params == ("high", "low", "close")
    assert op.metadata.scalar_params == ("window",)
    result = op.calculate(high, low, close, window=5)
    assert result.shape == close.shape
    assert np.isfinite(result.to_numpy()).any()
    with pytest.raises(ValueError):
        op.calculate(high, low, close, window=0)

_REMAINING_DECLARED_TECHNICAL = (
    'EaseOfMovement',
    'KeltnerLower',
    'KeltnerMid',
    'KeltnerPosition',
    'KeltnerUpper',
    'NATR',
    'PPO',
    'PPO_hist',
    'PPO_signal',
    'PSAR',
    'PVO',
    'PVO_hist',
    'PVO_signal',
    'Supertrend',
    'SupertrendDirection',
    'TEMA',
    'TSI',
    'TSI_signal',
    'UltimateOscillator',
    'VortexMinus',
    'VortexPlus',
    'atr_acceleration',
    'atr_pct',
    'atr_percentile',
    'atr_short_long_ratio',
    'atr_zscore',
    'bvc_imbalance_ma',
    'bvc_sign_pct',
    'candle_body_strength',
    'candle_pattern_count',
    'candle_range_pct',
    'candle_wick_balance',
    'chikou_distance_pct',
    'consolidation_pct',
    'consolidation_range_pct',
    'dema_distance_pct',
    'donchian_breakout_down',
    'donchian_breakout_up',
    'donchian_channel_position',
    'donchian_width_pct',
    'ema_crossover',
    'ema_distance_pct',
    'ichimoku_cloud_position',
    'ichimoku_cloud_width',
    'ichimoku_kijun',
    'ichimoku_senkou_a',
    'ichimoku_senkou_b',
    'ichimoku_tenkan',
    'kama_distance_pct',
    'keltner_breakout_strength',
    'keltner_compression',
    'keltner_width_pct',
    'ma_slope_pct',
    'psar_days_since_flip',
    'psar_direction',
    'psar_distance_pct',
    'psar_flip',
    'reg_forecast_error_pct',
    'reg_r2_trailing',
    'reg_residual_zscore',
    'reg_slope_tstat',
    'senkou_span_causal_pct',
    'sma_distance_pct',
    'spectral_energy_ratio',
    'spectral_trend_share',
    'sr_distance_pct',
    'sr_touch_count',
    'supertrend_days_since_flip',
    'supertrend_direction',
    'supertrend_distance_pct',
    'supertrend_flip',
    'tema_distance_pct',
    'tenkan_kijun_cross',
    'true_range_surprise',
    'true_range_zscore',
    'ts_confirmed_pivot_high',
    'ts_confirmed_pivot_low',
    'turnover_autocorr',
    'volume_autocorr',
    'vpin_pct',
    'vwap_distance_pct',
    'vwap_premium_pct',
    'vwap_slope_pct',
    'wavelet_detail_energy_ratio',
    'zero_return_ratio',
)


def _rich_inputs(rows: int = 128):
    index = pd.date_range("2026-03-01", periods=rows)
    t = np.arange(rows, dtype=float)
    close_values = 100.0 + 0.08 * t + 3.0 * np.sin(t / 3.2) + 1.2 * np.sin(t / 9.0)
    open_values = close_values + 0.45 * np.sin(t / 2.1)
    high_values = np.maximum(open_values, close_values) + 1.0 + 0.15 * np.cos(t / 4.0)
    low_values = np.minimum(open_values, close_values) - 1.0 - 0.15 * np.sin(t / 4.0)
    volume_values = 1000.0 + 7.0 * t + 180.0 * (1.0 + np.sin(t / 5.0))
    values = {
        "open": open_values,
        "high": high_values,
        "low": low_values,
        "close": close_values,
        "x": close_values,
        "price": close_values,
        "volume": volume_values,
        "turnover": volume_values * close_values,
        "dollar_volume": volume_values * close_values,
        "ret": np.r_[0.0, np.diff(close_values) / close_values[:-1]],
        "event": np.where((t.astype(int) % 5) == 0, 1.0, 0.0),
        "state": ((t.astype(int) // 3) % 4).astype(float),
    }
    return {
        name: pd.DataFrame({"A": data}, index=index)
        for name, data in values.items()
    }


def _r6_scalar_value(name: str):
    exact = {
        "acceleration": 0.02,
        "maximum": 0.2,
        "multiplier": 2.0,
        "short_weight": 4.0,
        "medium_weight": 2.0,
        "long_weight": 1.0,
        "tol": 0.03,
        "epsilon": 1e-12,
        "volume_scale": 1.0,
        "std_dev": 2.0,
        "lag": 1,
        "top_k": 2,
        "trend_bins": 2,
        "buckets": 4,
        "left_window": 2,
        "right_window": 2,
        "history_window": 48,
        "n": 2,
        "points": 2,
        "min_spacing": 2,
        "max_spacing": 24,
        "tolerance": 0.10,
        "min_depth": 0.01,
        "shoulder_tolerance": 0.10,
        "head_min_prominence": 0.01,
        "max_neckline_slope": 0.20,
        "slope_threshold": 0.01,
        "parallel_tolerance": 0.05,
        "min_impulse": 0.01,
        "max_retracement": 0.80,
        "max_width": 0.25,
        "volume_decay_threshold": 0.0,
        "halflife": 4.0,
        "max_cap": 5,
    }
    if name in exact:
        return exact[name]
    if name in {"fast_window", "short_window", "signal_window", "tenkan_window"}:
        return 4
    if name in {"er_window", "atr_window", "ema_window", "kijun_window"}:
        return 8
    if name in {"slow_window", "medium_window"}:
        return 12
    if name in {"long_window", "score_window", "senkou_b_window"}:
        return 24
    if name == "window" or name.endswith("_window"):
        return 16
    raise AssertionError(f"R6 needs an explicit non-panel fixture for scalar {name!r}")


@pytest.mark.parametrize("name", _REMAINING_DECLARED_TECHNICAL)
def test_remaining_declared_technical_contracts_execute_causally(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert metadata.scalar_params
    assert set(metadata.scalar_params) == set(metadata.param_specs)

    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name

    prefix_rows = 80
    prefix = op.calculate(
        *(panel.iloc[:prefix_rows] for panel in args),
        **kwargs,
    )
    pd.testing.assert_frame_equal(
        result.iloc[:prefix_rows],
        prefix,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )

    if name == "KeltnerMid":
        expected = panels["close"].ewm(span=kwargs["ema_window"], adjust=False).mean()
        expected.iloc[: kwargs["ema_window"] - 1] = np.nan
        pd.testing.assert_frame_equal(result, expected)

_STRUCTURE_CONTRACTS = (
    "ts_last_pivot_high",
    "ts_last_pivot_low",
    "ts_pivot_high_age",
    "ts_pivot_low_age",
    "ts_resistance_level",
    "ts_support_level",
    "ts_resistance_slope",
    "ts_support_slope",
    "ts_distance_to_resistance",
    "ts_distance_to_support",
    "ts_resistance_break",
    "ts_support_break",
)


@pytest.mark.parametrize("name", _STRUCTURE_CONTRACTS)
def test_bounded_structure_contracts_execute_with_scalar_history_and_points(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert "points" not in metadata.panel_params
    assert metadata.panel_params in (("high",), ("low",), ("close", "high"), ("close", "low"))
    assert metadata.scalar_params[:3] == ("left_window", "right_window", "history_window")
    assert set(metadata.scalar_params) == set(metadata.param_specs)

    args = [panels[param] for param in metadata.panel_params]
    kwargs = {"left_window": 2, "right_window": 2, "history_window": 48}
    if "points" in metadata.scalar_params:
        kwargs["points"] = 2
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name

    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(result.iloc[:80], prefix)

    with pytest.raises(ValueError):
        op.calculate(*args, **{**kwargs, "history_window": 0})
    if "points" in kwargs:
        with pytest.raises(ValueError):
            op.calculate(*args, **{**kwargs, "points": 1})


def test_structure_distance_matches_independently_called_level():
    panels = _rich_inputs()
    common = {"left_window": 2, "right_window": 2, "history_window": 48, "points": 2}
    level = OperatorRegistry.get(
        "ts_resistance_level", backend="pandas_numpy", mode="research"
    ).calculate(panels["high"], **common)
    distance = OperatorRegistry.get(
        "ts_distance_to_resistance", backend="pandas_numpy", mode="research"
    ).calculate(panels["close"], panels["high"], **common)
    expected = panels["close"] / level.replace(0.0, np.nan) - 1.0
    pd.testing.assert_frame_equal(distance, expected)

_LIQUIDITY_NEXT = (
    "ForceIndex", "abnormal_turnover", "abnormal_volume", "adv",
    "amihud_illiquidity", "average_turnover", "bounded_nvi", "bounded_pvi",
    "corwin_schultz_spread", "down_volume_ratio", "high_low_spread_proxy",
    "price_impact", "price_turnover_divergence", "price_volume_divergence",
    "return_per_turnover", "return_turnover_beta", "return_volume_beta",
    "roll_spread_proxy", "rolling_adl_flow", "signed_volume_imbalance",
    "turnover_acceleration", "turnover_adjusted_volatility", "turnover_shock",
    "turnover_volatility", "up_down_volume_ratio", "up_volume_ratio",
    "volume_acceleration", "volume_price_range_density", "volume_shock",
    "volume_volatility", "volume_weighted_momentum", "volume_weighted_return",
)


@pytest.mark.parametrize("name", _LIQUIDITY_NEXT)
def test_remaining_liquidity_contracts_execute_causally(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs)
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:80], prefix, check_exact=False, rtol=1e-10, atol=1e-10
    )

_TECHNICAL_EXTENSIONS_NEXT = (
    "MFI",
    "abs_return_volume_corr",
    "bollinger_pct_b",
    "bollinger_width",
    "candle_abs_body",
    "candle_body",
    "candle_body_ratio",
    "candle_close_location",
    "candle_direction",
    "candle_gap",
    "candle_gap_pct",
    "candle_lower_shadow",
    "candle_lower_shadow_ratio",
    "candle_range",
    "candle_range_atr",
    "candle_upper_shadow",
    "candle_upper_shadow_ratio",
    "cdl_doji",
    "cdl_engulfing",
    "cdl_hammer",
    "cdl_inside_bar",
    "cdl_inverted_hammer",
    "cdl_marubozu",
    "cdl_outside_bar",
    "cdl_shooting_star",
    "cdl_spinning_top",
    "choppiness_index",
    "dollar_volume",
    "dollar_volume_zscore",
    "donchian_lower",
    "donchian_mid",
    "donchian_position",
    "donchian_upper",
    "efficiency_ratio",
    "garman_klass_vol",
    "intraday_volatility",
    "overnight_volatility",
    "parkinson_vol",
    "range_volatility",
    "relative_volume",
    "return_volume_corr",
    "rogers_satchell_vol",
    "rolling_obv",
    "rolling_pvt",
    "rolling_vwap",
    "signed_dollar_volume",
    "signed_volume",
    "ts_breakdown_low",
    "ts_breakout_high",
    "ts_channel_position",
    "ts_days_since_high",
    "ts_days_since_low",
    "ts_distance_to_high",
    "ts_distance_to_low",
    "ts_new_high",
    "ts_new_low",
    "ts_prev_high",
    "ts_prev_low",
    "ts_range_expansion",
    "turnover_momentum",
    "turnover_zscore",
    "ulcer_index",
    "volume_momentum",
    "volume_zscore",
    "vwap_deviation",
    "yang_zhang_vol",
)

@pytest.mark.parametrize("name", _TECHNICAL_EXTENSIONS_NEXT)
def test_remaining_technical_extensions_execute_causally(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs or {})
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:80], prefix, check_exact=False, rtol=1e-10, atol=1e-10
    )

_CANDLE_GEOMETRY_NEXT = (
    "candle_body_zscore", "candle_range_zscore", "candle_upper_shadow_zscore",
    "candle_lower_shadow_zscore", "candle_body_percentile",
    "candle_range_percentile", "candle_gap_atr", "candle_body_position",
    "candle_overlap_ratio", "candle_inside_ratio", "candle_close_strength",
    "candle_rejection_upper", "candle_rejection_lower",
)


@pytest.mark.parametrize("name", _CANDLE_GEOMETRY_NEXT)
def test_candle_geometry_contracts_execute_causally(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs or {})
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(result.iloc[:80], prefix)

_STRUCTURE_PATTERNS_NEXT = (
    "pattern_ascending_triangle",
    "pattern_bear_flag",
    "pattern_broadening",
    "pattern_bull_flag",
    "pattern_descending_triangle",
    "pattern_double_bottom",
    "pattern_double_top",
    "pattern_falling_channel",
    "pattern_falling_wedge",
    "pattern_head_shoulders",
    "pattern_inverse_head_shoulders",
    "pattern_rectangle",
    "pattern_rising_channel",
    "pattern_rising_wedge",
    "pattern_sym_triangle",
    "ts_channel_width",
    "ts_channel_width_atr",
    "ts_channel_width_pct",
    "ts_channel_width_slope",
    "ts_consolidation_slope",
    "ts_consolidation_volume_decay",
    "ts_consolidation_width",
    "ts_impulse_return",
    "ts_impulse_strength",
    "ts_impulse_volume",
    "ts_line_convergence",
    "ts_line_parallelism",
    "ts_nth_pivot_high",
    "ts_nth_pivot_high_age",
    "ts_nth_pivot_low",
    "ts_nth_pivot_low_age",
    "ts_pattern_symmetry",
    "ts_pivot_high_count",
    "ts_pivot_high_spacing",
    "ts_pivot_low_count",
    "ts_pivot_low_spacing",
    "ts_resistance_fit_r2",
    "ts_support_fit_r2",
    "ts_swing_amplitude",
    "ts_swing_amplitude_atr",
    "ts_swing_amplitude_pct",
    "ts_swing_duration",
    "ts_swing_velocity",
)

@pytest.mark.parametrize("name", _STRUCTURE_PATTERNS_NEXT)
def test_structure_pattern_contracts_execute_causally_and_validate(name):
    panels = _rich_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs or {})
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    for param, spec in metadata.param_specs.items():
        if spec.dtype is int and spec.min is not None:
            kwargs[param] = max(kwargs[param], int(spec.min))
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:80], prefix, check_exact=False, rtol=1e-10, atol=1e-10
    )
    invalid_param = "points" if "points" in kwargs else next(
        param for param in metadata.scalar_params if metadata.param_specs[param].dtype is int
    )
    with pytest.raises(ValueError):
        op.calculate(*args, **{**kwargs, invalid_param: 0})

_EVENT_STATE_DERIVATIONS = (
    "event_streak", "event_cluster_duration", "event_decay_window",
    "signed_event_rate", "positive_event_rate", "negative_event_rate",
    "positive_event_age", "negative_event_age", "signed_event_decay",
    "event_direction_imbalance", "event_flip_density", "category_age",
    "category_frequency", "category_transition_rate",
    "category_transition_surprise", "event_direction_persistence", "state_age",
    "state_persistence", "state_transition_count", "state_transition_rate",
    "state_flip_density", "state_episode_age", "state_flip_age",
    "state_episode_age_lower_bound", "state_episode_age_capped",
    "state_episode_censored_flag", "category_age_lower_bound",
)
_BOOL_EVENT_NAMES = {"event_streak", "event_cluster_duration", "event_decay_window"}
_SIGNED_EVENT_NAMES = {
    "signed_event_rate", "positive_event_rate", "negative_event_rate",
    "positive_event_age", "negative_event_age", "signed_event_decay",
    "event_direction_imbalance", "event_flip_density",
    "event_direction_persistence",
}


@pytest.mark.parametrize("name", _EVENT_STATE_DERIVATIONS)
def test_event_state_derivation_contracts_and_temporal_semantics(name):
    panels = _rich_inputs()
    if name in _SIGNED_EVENT_NAMES:
        t = np.arange(len(panels["event"]))
        signed = np.select([t % 7 == 0, t % 5 == 0], [-1.0, 1.0], default=0.0)
        panels["event"] = pd.DataFrame(
            {"A": signed}, index=panels["event"].index
        )
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params in (("event",), ("state",))
    assert set(metadata.scalar_params) == set(metadata.param_specs)
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _r6_scalar_value(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["event"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(result.iloc[:80], prefix)

    with pytest.raises(ValueError):
        op.calculate(*args, **{**kwargs, "window": 1})
    bad = args[0].copy()
    bad.iloc[20, 0] = 0.5 if name in _BOOL_EVENT_NAMES | _SIGNED_EVENT_NAMES else np.inf
    with pytest.raises(ValueError):
        op.calculate(bad, **kwargs)

_WAVE1_LIQUIDITY = (
    "lf1_herfindahl_volume", "lf1_volume_concentration",
    "lf1_turnover_distribution_skew", "lf1_amihud_own",
    "lf1_volume_amplification", "lf1_volume_amihud_ratio",
    "lf1_mean_variance_turnover", "lf1_market_share_turnover",
    "lf1_own_volume_share", "tod_open_midday_ratio",
    "tod_close_midday_ratio", "tod_edge_activity",
    "tod_intraday_range_position", "tod_overnight_activity_ratio",
    "lf1_ret_volume_ratio", "lf1_liquidity_decay_base",
)
_WAVE1_VOLREGIME = (
    "vr1_range_everage", "vr1_parkinson_close_scale",
    "vr1_rogers_satchell", "vr1_garman_klass_ext",
    "vr1_range_to_close_eff", "vr1_ewma_range_vol", "vv1_vol_of_vol",
    "vv1_regime_change_ratio", "vv1_vol_level_score", "vv1_vol_acceleration",
    "hfl_variance_ratio", "vv1_long_short_vol_beta", "hfl_hurst_ratio",
    "vv1_fractional_share", "vv1_dispersion_vol", "vv1_downside_vol_share",
)


def _wave1_inputs(rows=128, cols=12):
    index = pd.date_range("2026-07-01", periods=rows)
    t = np.arange(rows, dtype=float)[:, None]
    offsets = np.arange(cols, dtype=float)[None, :]
    close = 80.0 + 0.06 * t + 2.5 * np.sin(t / 4.0 + offsets / 7.0) + offsets
    open_ = close + 0.3 * np.sin(t / 3.0 + offsets)
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1000.0 + 9.0 * t + 35.0 * offsets + 120.0 * (1 + np.sin(t / 6.0))
    ret = np.vstack([np.zeros((1, cols)), np.diff(close, axis=0) / close[:-1]])
    frames = {
        "open": open_, "high": high, "low": low, "close": close,
        "close_return": ret, "ret": ret, "volume": volume,
        "turnover": volume * close, "amount": volume * close,
        "x": close, "overnight_mark": np.abs(ret) * 0.4,
        "intraday_mark": np.abs((close - open_) / open_),
        "shock_mark": (np.arange(rows)[:, None] % 13 == 0).astype(float)
            * np.ones((1, cols)),
        "signed_volume": volume * np.where((t.astype(int) + offsets.astype(int)) % 3, 1.0, -1.0),
        "market_signed_volume": volume * np.where(t.astype(int) % 4, 1.0, -1.0),
        "market_ret": np.repeat(ret.mean(axis=1, keepdims=True), cols, axis=1),
        "amihud": np.abs(ret) / (volume * close),
    }
    columns = [f"S{i:02d}" for i in range(cols)]
    return {
        name: pd.DataFrame(values, index=index, columns=columns)
        for name, values in frames.items()
    }


def _wave1_scalar(name):
    return {
        "min_breadth": 5, "top_share": 0.2, "window": 40,
        "min_periods": 5, "ba_window": 20, "open_rows": 10,
        "middle_rows": 20, "close_rows": 10, "decay_scale": 5.0,
        "short_window": 10, "fast_window": 10, "slow_window": 30, "long_window": 30, "history_window": 40,
        "vol_window": 10, "accel_window": 5, "band": 0.5,
        "k": 2, "max_agg": 8, "threshold": 0.2,
        "half_life": 5.0, "corr_window": 10, "penalty": 1.0,
    }[name]


@pytest.mark.parametrize("name", _WAVE1_LIQUIDITY + _WAVE1_VOLREGIME)
def test_wave1_liquidity_and_volregime_contracts(name):
    panels = _wave1_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs)
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _wave1_scalar(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["close"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:80], prefix, check_exact=False, rtol=1e-10, atol=1e-10
    )
    invalid = metadata.scalar_params[0]
    spec = metadata.param_specs[invalid]
    bad_value = (spec.min - 1) if spec.min is not None else np.nan
    with pytest.raises(ValueError):
        op.calculate(*args, **{**kwargs, invalid: bad_value})

_WAVE1_ORDERFLOW = (
    "ofi_volume_imbalance", "ofi_imbalance_persistence",
    "ofi_abs_imbalance_trend", "ofi_imbalance_cv", "sv_self_relative_change",
    "sv_own_flow_fraction", "sv_signed_volume_volatility",
    "sv_net_flow_direction", "ofi_dominant_direction",
    "ofi_imbalance_agreement", "ofi_zero_flow_balance", "ofi_reversal_rate",
    "ofi_volume_flow_regime", "sv_signed_beta_market",
    "sv_signed_shock_persistence",
)
_WAVE1_CS_MOMENTUM = (
    "m1_ranked_momentum", "m1_volume_adjusted_momentum",
    "m1_momentum_strength", "m1_momentum_stability",
    "m1_momentum_speed_change", "m1_cs_momentum_dispersion",
    "m1_rank_momentum_gap", "m1_corr_momentum", "m1_momentum_regime",
    "vax_liquidity_adjusted_return", "vax_liquidity_penalty_exposure",
    "vax_ret_per_liquidity_unit",
)


@pytest.mark.parametrize("name", _WAVE1_ORDERFLOW + _WAVE1_CS_MOMENTUM)
def test_wave1_orderflow_and_cs_momentum_contracts(name):
    panels = _wave1_inputs()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    metadata = op.metadata
    assert metadata.panel_params
    assert set(metadata.scalar_params) == set(metadata.param_specs)
    args = [panels[param] for param in metadata.panel_params]
    kwargs = {param: _wave1_scalar(param) for param in metadata.scalar_params}
    result = op.calculate(*args, **kwargs)
    assert result.shape == panels["ret"].shape
    assert np.isfinite(result.to_numpy()).any(), name
    prefix = op.calculate(*(panel.iloc[:80] for panel in args), **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:80], prefix, check_exact=False, rtol=1e-10, atol=1e-10
    )
    invalid = metadata.scalar_params[0]
    spec = metadata.param_specs[invalid]
    with pytest.raises(ValueError):
        op.calculate(*args, **{**kwargs, invalid: spec.min - 1})

def test_hfl_variance_ratio_keyword_and_mixed_binding_equivalence():
    ret = _wave1_inputs()["ret"]
    op = OperatorRegistry.get(
        "hfl_variance_ratio", backend="pandas_numpy", mode="research"
    )
    keyword = op.calculate(ret, k=2, window=40)
    reverse_keyword = op.calculate(ret, window=40, k=2)
    mixed = op.calculate(ret, 2, window=40)
    pd.testing.assert_frame_equal(keyword, reverse_keyword)
    pd.testing.assert_frame_equal(keyword, mixed)
    assert np.isfinite(keyword.to_numpy()).any()
    with pytest.raises(ValueError):
        op.calculate(ret, k=1, window=40)

@pytest.mark.parametrize(
    ("name", "parameter", "role"),
    [
        ("ts_mcginley_dynamic", "power", "estimator_resolution"),
        ("ts_vidya", "smooth", "estimator_resolution"),
        ("ts_one_euro_filter", "beta", "estimator_resolution"),
        ("ts_nlms_filter", "mu", "estimator_resolution"),
        ("ts_nlms_filter", "eps", "numerical"),
        ("ts_rls_filter", "lambda_", "estimator_resolution"),
        ("ts_rls_filter", "delta", "numerical"),
    ],
)
def test_adaptive_filter_scalar_roles_match_semantics(name, parameter, role):
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    assert op.metadata.param_specs[parameter].param_role.value == role
