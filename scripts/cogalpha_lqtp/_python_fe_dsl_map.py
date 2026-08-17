EXTRA_PYTHON_FE_DSL: dict[str, str] = {
    "factor_abnormality_asymmetry": (
        "-( abs(safe_div(log(volume + 1) - ts_mean(log(volume + 1), 20), ts_std(log(volume + 1), 20) + 1e-8)) * safe_div((high - low), close) * where(ts_ema(where(close >= open, volume, 0), 10) > 0, safe_div(ts_ema(where(close >= open, volume, 0), 10), ts_ema(where(not_(close >= open), volume, 0), 10)), 1.0) )"
    ),
    "factor_adaptive_vol_directional_smooth_range10": (
        "-( ts_ema(safe_div((high - low), close), 10) * (sigmoid(5 * (safe_div(volume, ts_mean(volume, 25)) - 0.85)) * ts_ema(safe_div(volume, ts_mean(volume, 25)), 10) + (1 - sigmoid(5 * (safe_div(volume, ts_mean(volume, 25)) - 0.85))) * ts_ema(safe_div(volume, ts_mean(volume, 25)), 30)) * ts_ema(safe_div(close - open, close), 3) )"
    ),
    "factor_asym_vol_attenuate": (
        "where( safe_div(volume, ts_ema(volume, 20)) < 0.6, log(cap(safe_div( safe_div(ts_sum(where(ts_pct(close, 1) < 0, (high - low), 0), 20), ts_sum(where(ts_pct(close, 1) < 0, 1, 0), 20) + 1e-8), safe_div(ts_sum(where(ts_pct(close, 1) >= 0, (high - low), 0), 20), ts_sum(where(ts_pct(close, 1) >= 0, 1, 0), 20) + 1e-8) ), 1e-6, 1e6)) * (1 + 0.3 * clip(safe_div(volume - ts_ema(volume, 20), ts_ema(volume, 20)), -1, 1)) * 0.7, log(cap(safe_div( safe_div(ts_sum(where(ts_pct(close, 1) < 0, (high - low), 0), 20), ts_sum(where(ts_pct(close, 1) < 0, 1, 0), 20) + 1e-8), safe_div(ts_sum(where(ts_pct(close, 1) >= 0, (high - low), 0), 20), ts_sum(where(ts_pct(close, 1) >= 0, 1, 0), 20) + 1e-8) ), 1e-6, 1e6)) * (1 + 0.3 * clip(safe_div(volume - ts_ema(volume, 20), ts_ema(volume, 20)), -1, 1)) )"
    ),
    "factor_asym_vol_down_up_volatility_gated": (
        "where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), log(safe_div( ts_ema(where(ts_pct(close, 1) < 0, pow(ts_pct(close, 1), 2), 0), 20), ts_ema(where(ts_pct(close, 1) > 0, pow(ts_pct(close, 1), 2), 0), 20) + 1e-8 )), 0)"
    ),
    "factor_asym_vol_volume_cont_30": (
        "log(safe_div( ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 30), ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 30) + 1e-8 )) * safe_div(volume, ts_mean(volume, 20))"
    ),
    "factor_crash_vol_spike_volregime_boost": (
        "-( safe_div(ts_mean(log(safe_div(high, low)), 5), ts_mean(log(safe_div(high, low)), 50) + 1e-8) * (1 + where(safe_div(volume, ts_ema(volume, 20)) > 1.5, 1, 0) * 0.8) )"
    ),
    "factor_crossover_hybrid_magnitude": (
        "-( (0.5 * ts_ema(abs(ts_pct(close, 1)), 21) + 0.5 * ts_rank(abs(ts_pct(close, 1)), 21)) * ts_rank(safe_div((minimum(open, close) - low) - (high - maximum(open, close)), (minimum(open, close) - low) + (high - maximum(open, close)) + 1e-10), 20) * safe_div(volume, ts_mean(volume, 20)) )"
    ),
    "factor_directional_vol_smooth_range_mut": (
        "ts_ema(safe_div((high - low), close), 8) * (sigmoid(5 * (safe_div(volume, ts_mean(volume, 25)) - 0.85)) * ts_ema(safe_div(volume, ts_mean(volume, 25)), 10) + (1 - sigmoid(5 * (safe_div(volume, ts_mean(volume, 25)) - 0.85))) * ts_ema(safe_div(volume, ts_mean(volume, 25)), 30)) * safe_div(close - open, close)"
    ),
    "factor_directional_vol_volbase25_smooth_dir_range10": (
        "-( ts_ema(safe_div((high - low), close), 10) * ts_ema(safe_div(volume, ts_mean(volume, 25)), 15) * ts_ema(safe_div(close - open, close), 3) )"
    ),
    "factor_downside_vol_asymmetry_smoothed": (
        "log(ts_mean( safe_div( sqrt(ts_mean(pow(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 2), 20)), sqrt(ts_mean(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 20) + 1e-8) + 1e-10 ), 5) + 1e-10)"
    ),
    "factor_drawdown_atr_20_normalized": (
        "-safe_div(safe_div(close, ts_max(close, 60)) - 1, ATR_WILDER(high, low, close, 20))"
    ),
    "factor_drawdown_atr_normalized_20": (
        "-safe_div(safe_div(close, ts_max(close, 60)) - 1, ATR_WILDER(high, low, close, 20))"
    ),
    "factor_drawdown_atr_normalized_v2": (
        "-safe_div(safe_div(close, ts_max(close, 60)) - 1, ATR_WILDER(high, low, close, 60))"
    ),
    "factor_drawdown_atr_recovery": (
        "-safe_div(safe_div(close, ts_max(close, 60)) - 1, ATR_WILDER(high, low, close, 20))"
    ),
    "factor_drawdown_vol_persistence": (
        "-( safe_div(safe_div(close, ts_max(close, 60)) - 1, ts_std(ts_pct(close, 1), 20) + 1e-8) * cap(safe_div(ATR_WILDER(high, low, close, 40), ATR_WILDER(high, low, close, 10)), 0.5, 2.0) )"
    ),
    "factor_ema_vol_asymmetry_rank": (
        "-( ts_ema(pow(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 2), 20) - ts_ema(pow(where(ts_pct(close, 1) < 0, -ts_pct(close, 1), 0), 2), 20) )"
    ),
    "factor_fear_adjusted_dollar_pressure_short_ema": (
        "ts_rank(safe_div( ts_ema((close - open) * volume, 10) * (1 + safe_div(delay(close, 1) - close, delay(close, 1))), ATR_WILDER(high, low, close, 21) + 1e-8 ), 252) * 2 - 1"
    ),
    "factor_herding_vol_gated_momentum": (
        "-( safe_div( (ts_rank(ts_pct(close, 1), 20) - 0.5) * (ts_rank(ts_pct(volume, 1), 20) - 0.5), 1 + ts_std(ts_pct(close, 1), 20) ) * (1 + 0.5 * ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 20)) * (1 + 0.5 * safe_div( ts_ema(ts_pct(close, 1), 20) - ts_mean(ts_ema(ts_pct(close, 1), 20), 40), ts_std(ts_ema(ts_pct(close, 1), 20), 40) + 1e-8 )) )"
    ),
    "factor_herding_volume_momentum_continuous": (
        "-( ts_sum(ts_pct(close, 1) * safe_div(volume, ts_mean(delay(volume, 1), 20)), 20) )"
    ),
    "factor_intraday_adx_momentum": (
        "-( ts_ema(ts_sum(safe_div(close, open) - 1, 5), 3) * safe_div(ADX(high, low, close, 14), 100) )"
    ),
    "factor_intraday_pressure_asym_diff": (
        "-(safe_div(close - low, high - low + 1e-8) - 0.5) * maximum( safe_div( ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) - ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) + ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20) + 1e-8 ), 0)"
    ),
    "factor_intraday_pressure_ema": (
        "-ts_ema(safe_div(close - open, close) * volume, 10)"
    ),
    "factor_intraday_volume_tanh": (
        "-( tanh(ts_ema((safe_div(close, open) - 1) * safe_div(volume, ts_mean(volume, 20)), 20)) )"
    ),
    "factor_intraday_volume_tanh_short": (
        "-( tanh(ts_ema((safe_div(close, open) - 1) * safe_div(volume, ts_mean(volume, 20)), 10)) )"
    ),
    "factor_vol_weighted_pct_range_clip_ema10": (
        "ts_ema(ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)) "
        "* (clip(safe_div((high - low) - ts_mean((high - low), 10), ts_std((high - low), 10)), 0, 2) + 1.0), 10)"
    ),
    "factor_crossover_herding_stress_span12_volrank_clip3": (
        "clip(ts_ema(ts_pct(close, 1) * ts_rank(volume, 8) * safe_div(high - low, close), 12), -3, 3)"
    ),
    "factor_mut_cross_vwpc_z10_clip2_v3": (
        "ts_ema(ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)) "
        "* (1 + clip(safe_div((high - low) - ts_mean((high - low), 10), ts_std((high - low), 10)), -2, 2)), 10)"
    ),
    "factor_mut_cross_vwpc_z10_clip2_span8_v1": (
        "ts_ema(ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)) "
        "* (1 + clip(safe_div((high - low) - ts_mean((high - low), 10), ts_std((high - low), 10)), -2, 2)), 8)"
    ),
    "factor_volume_weighted_price_change": (
        "ts_ema(ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)) "
        "* (1 + clip(safe_div((high - low) - ts_mean((high - low), 20), ts_std((high - low), 20)), -3, 3)), 10)"
    ),
    "factor_volume_confirmed_momentum_clipped": (
        "clip((safe_div(close, delay(close, 5)) - 1) * safe_div(volume, ts_mean(volume, 20)), -3, 3)"
    ),
    "factor_reversal_intraday_5d_ema": (
        "-(ts_ema(safe_div(close, open) - 1, 5))"
    ),
    "factor_lag_response_vol_tool_adaptive": (
        "-( where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), ts_ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_mean(volume, 20))) * safe_div(ATR_WILDER(high, low, close, 20), close), 10), ts_ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_mean(volume, 20))) * safe_div(ATR_WILDER(high, low, close, 20), close), 5) ) )"
    ),
    "factor_lag_vol_volatility_adjusted": (
        "-( ts_ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_ema(volume, 20))) * safe_div(ATR_WILDER(high, low, close, 20), close), 5) )"
    ),
    "factor_lag_vol_volatility_adjusted_gate": (
        "-( where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), ts_ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_ema(volume, 20))) * safe_div(ATR_WILDER(high, low, close, 20), close), 10), ts_ema((safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_ema(volume, 20))) * safe_div(ATR_WILDER(high, low, close, 20), close), 5) ) )"
    ),
    "factor_lag_volatility_regime_adapted": (
        "-( ts_ema( (safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_ema(volume, 20))) * where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), safe_div(ATR_WILDER(high, low, close, 10), close), safe_div(ATR_WILDER(high, low, close, 20), close)), 5) )"
    ),
    "factor_liquidity_range_gated_momentum_v5": (
        "-( ts_pct(close, 5) * ts_rank((high - low) * safe_div(volume, ts_median(volume, 20)), 20) * tanh(safe_div(ATR_WILDER(high, low, close, 14), close)) )"
    ),
    "factor_liquidity_range_gated_momentum_v6": (
        "-( ts_pct(close, 5) * ts_rank((high - low) * safe_div(volume, ts_median(volume, 20)), 20) * tanh(safe_div(ATR_WILDER(high, low, close, 20), close)) )"
    ),
    "factor_momentum_volratio": (
        "-( (safe_div(close, delay(close, 5)) - 1) * ts_ema(where(close > delay(close, 1), 1, 0), 20) * exp(-safe_div(ts_std(ts_pct(close, 1), 20), ts_std(ts_pct(close, 1), 60) + 1e-8)) )"
    ),
    "factor_mut_cro_asym_vol_ratio_med_15_minp3_vol10": (
        "-ts_ema( safe_div( ts_median(where(close > open, (high - low), 0), 15), ts_median(where(close <= open, (high - low), 0), 15) + 1e-8 ) * safe_div(volume, ts_median(volume, 10)), 3)"
    ),
    "factor_mut_cro_asym_vol_ratio_med_20_minp3_vol10_ema2": (
        "-ts_ema( safe_div( ts_median(where(close > open, (high - low), 0), 20), ts_median(where(close <= open, (high - low), 0), 20) + 1e-8 ) * safe_div(volume, ts_median(volume, 10)), 2)"
    ),
    "factor_mut_hc004_volpctrank_s12_clip2_5_volwin6": (
        "-( cap(ts_ema(ts_pct(close, 1) * ts_rank(volume, 6) * safe_div((high - low), close), 12), -2.5, 2.5) )"
    ),
    "factor_pressure_compression_simplified_v3": (
        "-( safe_div((close - open) * volume, ts_mean(abs((close - open) * volume), 21) + 1e-8) * tanh(safe_div(volume, ts_mean(volume, 21))) * tanh(safe_div(ATR_WILDER(high, low, close, 21), close)) )"
    ),
    "factor_range_asym_log_short": (
        "-( log(cap( safe_div( safe_div(ts_sum(where(close > delay(close, 1), (high - low), 0), 10), ts_sum(where(close > delay(close, 1), 1, 0), 10) + 1e-8), cap(safe_div(ts_sum(where(close <= delay(close, 1), (high - low), 0), 10), ts_sum(where(close <= delay(close, 1), 1, 0), 10) + 1e-8), 1e-12, 1e12) ), 1e-12, 1e12)) )"
    ),
    "factor_range_asym_log_vol": (
        "-( log(safe_div( safe_div(ts_sum(where(close > delay(close, 1), (high - low), 0), 20), ts_sum(where(close > delay(close, 1), 1, 0), 20) + 1e-12) + 1e-12, safe_div(ts_sum(where(close <= delay(close, 1), (high - low), 0), 20), ts_sum(where(close <= delay(close, 1), 1, 0), 20) + 1e-12) + 1e-12 )) * safe_div(ts_mean(volume, 20), ts_max(ts_mean(volume, 20), 20) + 1e-8) )"
    ),
    "factor_range_asym_vol_smooth": (
        "-( ts_ema( safe_div( safe_div(ts_sum(where(close > delay(close, 1), (high - low), 0), 20), ts_sum(where(close > delay(close, 1), 1, 0), 20) + 1e-8) - safe_div(ts_sum(where(close <= delay(close, 1), (high - low), 0), 20), ts_sum(where(close <= delay(close, 1), 1, 0), 20) + 1e-8), safe_div(ts_sum(where(close > delay(close, 1), (high - low), 0), 20), ts_sum(where(close > delay(close, 1), 1, 0), 20) + 1e-8) + safe_div(ts_sum(where(close <= delay(close, 1), (high - low), 0), 20), ts_sum(where(close <= delay(close, 1), 1, 0), 20) + 1e-8) + 1e-10 ) * safe_div(ts_mean(volume, 20), ts_max(ts_mean(volume, 20), 20) + 1e-8), 5) )"
    ),
    "factor_resvol_volume_momentum": (
        "-( where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), ts_pct(close, 20), ts_pct(close, 5)) * safe_div(volume, ts_ema(volume, 20)) )"
    ),
    "factor_reversal_volume_volregime": (
        "(0.5 - ts_ema(safe_div(close - low, high - low + 1e-8), 3)) * ts_rank(safe_div(volume, ts_mean(volume, 21)), 63) * where(safe_div(ts_mean((high - low), 5), ts_mean((high - low), 20) + 1e-8) > 1.0, 1.0, 0.5)"
    ),
    "factor_rsi_reversal_resvol_volume": (
        "(50 - RSI_WILDER(close, 14)) * where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), 1.5, 0.5) * safe_div(volume, ts_ema(volume, 20))"
    ),
    "factor_rsi_volume_confirmed_reversal": (
        "(50 - RSI_WILDER(close, 14)) * safe_div(volume, ts_ema(volume, 20))"
    ),
    "factor_simple_convex_momentum_log_volume": (
        "-( sign(safe_div(close, ts_ema(close, 10)) - 1) * pow(abs(safe_div(close, ts_ema(close, 10)) - 1), 2) * log(1 + cap(safe_div(volume, ts_ema(volume, 20)), 0, 1e6)) )"
    ),
    "factor_smoothed_atr_ratio_gated": (
        "-( where((ts_std(ts_pct(close, 1), 20) > ts_quantile(ts_std(ts_pct(close, 1), 20), 60, 0.5)), ts_ema(safe_div(ATR_WILDER(high, low, close, 20), ATR_WILDER(high, low, close, 60) + 1e-8), 5), 0) )"
    ),
    "factor_squared_range_close_ema_cuberoot": (
        "-( sign(ts_ema(pow((high - low), 2) / close, 10)) * pow(abs(ts_ema(pow((high - low), 2) / close, 10)), 1/3) )"
    ),
    "factor_squared_range_volume_rank_smoothed_no_cuberoot": (
        "-( ts_ema(pow((high - low), 2) / close * ts_rank(volume, 20), 10) )"
    ),
    "factor_stable_symmetry_reinforced": (
        "-( ts_ema(abs(ts_pct(close, 1)), 21) * ts_rank(safe_div((minimum(open, close) - low) - (high - maximum(open, close)), (minimum(open, close) - low) + (high - maximum(open, close)) + 1e-10), 20) * safe_div(volume, ts_mean(volume, 20)) )"
    ),
    "factor_tanh_thrust_dynamic_smooth": (
        "-( sigmoid(2 * (safe_div(volume, ts_ema(volume, 20)) - 1)) * ts_ema(tanh(ts_pct(close, 5)) * tanh(ts_pct(volume, 5)), 5) + (1 - sigmoid(2 * (safe_div(volume, ts_ema(volume, 20)) - 1))) * ts_ema(tanh(ts_pct(close, 5)) * tanh(ts_pct(volume, 5)), 20) )"
    ),
    "factor_vol_gated_herding_crossover": (
        "-( cap( safe_div(safe_div(close, delay(close, 5)) - 1, 1 + abs(ts_zscore(safe_div(ATR_WILDER(high, low, close, 20), close), 20))) * (1 + tanh(ts_ema(ts_pct(close, 1), 7) * 5)) * (1 + tanh(ts_corr(ts_pct(close, 1), ts_pct(volume, 1), 30) * tanh(safe_div(ts_pct(close, 1) - ts_mean(ts_pct(close, 1), 40), ts_std(ts_pct(close, 1), 40) + 1e-8)))) * (1 + tanh(ts_rank(volume, 20) * safe_div(ts_mean(volume, 5), ts_mean(volume, 20) + 1e-8))), -5, 5) )"
    ),
    "factor_vol_gated_momentum_v2": (
        "-( (safe_div(close, delay(close, 21)) - 1) * safe_div(1, 1 + safe_div(ATR_WILDER(high, low, close, 21), close)) * safe_div(volume, ts_mean(volume, 21)) )"
    ),
    "factor_vol_lag_ret_smoothed": (
        "-( ts_ema( (safe_div(close, delay(close, 5)) - 1) * log(safe_div(volume, ts_ema(volume, 20))) * (1 + safe_div(ATR_WILDER(high, low, close, 20), close)), 5) )"
    ),
    "factor_vol_price_log_range_gate_clipped_10day_mut_003": (
        "-( ts_ema( ts_pct(close, 1) * log(1 + safe_div(volume, ts_mean(volume, 5))) * (clip(safe_div((high - low) - ts_mean((high - low), 10), ts_std((high - low), 10) + 1e-8), 0, 2) + 1), 12) )"
    ),
    "factor_vol_price_log_range_gate_clipped_boost_5day_mut_001": (
        "-( ts_ema( ts_pct(close, 1) * log(1 + safe_div(volume, ts_mean(volume, 5))) * (1 + where(safe_div((high - low) - ts_mean((high - low), 5), ts_std((high - low), 5) + 1e-8) > 0, clip(safe_div((high - low) - ts_mean((high - low), 5), ts_std((high - low), 5) + 1e-8), 0, 2), 0)), 12) )"
    ),
    "factor_vol_regime_trend_smoothed": (
        "-( ts_ema( (safe_div(close, ts_mean(close, 20)) - 1) * safe_div(volume, ts_mean(volume, 20)) * (safe_div((high - low), ts_median((high - low), 20)) - 1), 5) )"
    ),
    "factor_volatility_asymmetry_spike_adj_ema5": (
        "-( ts_ema(sign(close - open) * log(safe_div(high, low)), 5) * (1 - 0.5 * ts_mean(where(safe_div(volume, ts_ema(volume, 20)) > 1.5, 1, 0), 20)) )"
    ),
    "factor_volatility_compression": (
        "ts_mean(1 - safe_div(ATR_WILDER(high, low, close, 5), ATR_WILDER(high, low, close, 60) + 1e-8), 3)"
    ),
    "factor_volume_abnormality_liquidity_gated": (
        "-( where((volume > ts_median(volume, 20)), ts_std(safe_div(volume, ts_median(volume, 20)), 15), ts_std(safe_div(volume, ts_median(volume, 20)), 15) * 0.7) )"
    ),
    "factor_volume_adaptive_momentum": (
        "-( ts_rank(safe_div(volume, ts_ema(volume, 20)), 63) * (safe_div(close, delay(close, 5)) - 1) + (1 - ts_rank(safe_div(volume, ts_ema(volume, 20)), 63)) * (safe_div(close, delay(close, 21)) - 1) )"
    ),
    "factor_volume_adjusted_divergence_ema10": (
        "(ts_ema(safe_div(open, delay(close, 1)) - 1, 10) - ts_ema(safe_div(close, open) - 1, 10)) * safe_div(volume, ts_ema(volume, 20))"
    ),
    "factor_volume_confirmed_ewm": (
        "-( ts_ema(ts_pct(close, 1) * (safe_div(volume, ts_mean(volume, 10)) - 1), 5) )"
    ),
    "factor_volume_confirmed_up_capture": (
        "-( safe_div( ts_sum(where(ts_pct(close, 1) > 0, ts_pct(close, 1) * volume, 0), 20), ts_sum(abs(ts_pct(close, 1)) * volume, 20) + 1e-8 ) * clip(1 + 0.5 * (safe_div(volume, ts_mean(volume, 20)) - 1), 0.5, 1.5) )"
    ),
    "factor_volume_regime_intraday_momentum": (
        "-( where(where(safe_div(volume, ts_ema(volume, 20)) < 0.6, 1, 0), ts_ema((safe_div(close, open) - 1) * volume, 10) * 0.5, ts_ema((safe_div(close, open) - 1) * volume, 10)) )"
    ),
    "factor_volume_surge_persistence": (
        "-( ts_ema(where(volume > 2 * ts_median(volume, 20), 1, 0), 10) )"
    ),
    "factor_volume_weighted_range_skew": (
        "-( tanh((safe_div(ts_ema(volume * (high - low), 10), ts_ema(volume * (high - low), 60) + 1e-8) - 1) * 10) )"
    ),
    "factor_vwap_adjusted_range_tanh_mutated_v2": (
        "-( tanh( safe_div(ATR_WILDER(high, low, close, 14), rolling_vwap(close, volume, 14)) * safe_div(volume, ts_ema(volume, 30)) ) )"
    ),
    "factor_vwap_adjusted_range_tanh_smooth": (
        "-( tanh( safe_div(ATR_WILDER(high, low, close, 20), rolling_vwap(close, volume, 30)) * safe_div(volume, ts_ema(volume, 30)) ) )"
    ),
    # --- previously python_only (hand-written FE DSL) ---
    "factor_lag_vol_robust_volratio": (
        "ts_ema((safe_div(close, delay(close, 5)) - 1) "
        "* safe_div((high - low) - ts_median((high - low), 20), "
        "ts_mean(abs((high - low) - ts_median((high - low), 20)), 20) + 1e-8) "
        "* safe_div(volume, ts_ema(volume, 20)), 10)"
    ),
    "factor_lag_vol_ratio_smoothed_v2": (
        "ts_ema((safe_div(close, delay(close, 5)) - 1) "
        "* safe_div((high - low) - ts_median((high - low), 30), "
        "ts_mean(abs((high - low) - ts_median((high - low), 30)), 30) + 1e-8) "
        "* safe_div(volume, ts_ema(volume, 20)), 15)"
    ),
    "factor_intraday_reversal_vol_intensity": (
        "-(ts_ema(safe_div(close, open) - 1, 5) "
        "* clip(safe_div(ts_std(ts_pct(close, 1), 20), ts_std(ts_pct(close, 1), 60) + 1e-8), 0.5, 2.0))"
    ),
    "factor_vol_regime_breakout_continuous": (
        "ts_ema((safe_div(close - low, high - low + 1e-8) - 0.5) "
        "* safe_div(high - low, ts_median((high - low), 20) + 1e-8), 5)"
    ),
    "factor_vol_gated_herding_mutated": (
        "clip(((safe_div(close, delay(close, 5)) - 1) / (1 + abs(clip(safe_div("
        "safe_div(ATR_WILDER(high, low, close, 20), close) - ts_mean(safe_div(ATR_WILDER(high, low, close, 20), close), 20), "
        "ts_std(safe_div(ATR_WILDER(high, low, close, 20), close), 20) + 1e-8), -2, 2))) "
        "* (1 + tanh(ts_ema(ts_pct(close, 1), 7) * 5))) "
        "* (1 + tanh(sign(ts_pct(close, 1)) * tanh(safe_div(volume - ts_mean(volume, 20), ts_mean(volume, 20) + 1e-8) * 3))) "
        "* (1 + tanh(ts_rank(volume, 12) * safe_div(ts_mean(volume, 3), ts_mean(volume, 10) + 1e-8))), -5, 5)"
    ),
    "factor_volume_gated_momentum_mutated_v3": (
        "clip(((safe_div(close, delay(close, 5)) - 1) / (1 + abs(clip(safe_div("
        "ts_std(safe_div(high - low, close), 10) - ts_mean(ts_std(safe_div(high - low, close), 10), 20), "
        "ts_std(ts_std(safe_div(high - low, close), 10), 20) + 1e-8), -2, 2))) "
        "* (1 + clip(ts_ema(ts_pct(close, 1), 10) * 3, -1, 1))) "
        "* (1 + tanh(ts_rank(volume, 20) * safe_div(ts_mean(volume, 5), ts_mean(volume, 20) + 1e-8))), -5, 5)"
    ),
    "factor_volume_adjusted_breakout_with_vol_filter_v2": (
        "ts_sum(where((close > delay(ts_max(high, 20), 1)) "
        "& (safe_div(volume, ts_mean(volume, 20)) > 1) "
        "& (log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) + 1e-12), 1e-12, 1e18)) "
        "< ts_median(log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) + 1e-12), 1e-12, 1e18)), 20)), "
        "ts_pct(close, 10) * safe_div(volume, ts_mean(volume, 20)), 0), 5)"
    ),
    "factor_volume_adjusted_breakout_v3": (
        "ts_sum(where((close > delay(ts_max(high, 10), 1)) "
        "& (safe_div(volume, ts_mean(volume, 10)) > 1) "
        "& (log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) + 1e-12), 1e-12, 1e18)) "
        "< ts_median(log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 20) + 1e-12), 1e-12, 1e18)), 20)), "
        "ts_pct(close, 5) * safe_div(volume, ts_mean(volume, 10)), 0), 5)"
    ),
    "factor_volume_adjusted_breakout_window_001": (
        "ts_sum(where((close > delay(ts_max(high, 20), 1)) "
        "& (safe_div(volume, ts_mean(volume, 20)) > 1) "
        "& (log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 10), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 10) + 1e-12), 1e-12, 1e18)) "
        "< ts_median(log(cap(safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 10), "
        "ts_std(where(ts_pct(close, 1) > 0, ts_pct(close, 1), 0), 10) + 1e-12), 1e-12, 1e18)), 20)), "
        "ts_pct(close, 1) * safe_div(volume, ts_mean(volume, 20)), 0), 5)"
    ),
    "factor_volume_stress_adjusted_momentum_robust": (
        "ts_pct(close, 10) * clip(safe_div(volume - ts_median(volume, 20), "
        "ts_median(abs(volume - ts_median(volume, 20)), 20) * 1.4826 + 1e-8), -3, 3) "
        "* (1 - safe_div(ts_std(where(ts_pct(close, 1) < 0, ts_pct(close, 1), 0), 20), "
        "ts_std(ts_pct(close, 1), 20) + 1e-8))"
    ),
    "factor_volume_persistence_momentum_robust": (
        "ts_pct(close, 10) * clip(safe_div(volume - ts_median(volume, 20), "
        "ts_median(abs(volume - ts_median(volume, 20)), 20) + 1e-8), -3, 3) "
        "* safe_div(ts_sum(where(ts_pct(close, 1) > 0, 1, 0), 10), 10)"
    ),
    "factor_simplified_robust_volume_momentum": (
        "ts_pct(close, 10) * clip(safe_div(volume - ts_median(volume, 20), "
        "ts_median(abs(volume - ts_median(volume, 20)), 20) + 1e-8), -3, 3)"
    ),
    "factor_dollar_pressure_range_zscore": (
        "safe_div(safe_div((close - (high + low + close) / 3) * volume, ts_mean(high - low, 20) + 1e-8) "
        "- ts_mean(safe_div((close - (high + low + close) / 3) * volume, ts_mean(high - low, 20) + 1e-8), 20), "
        "ts_std(safe_div((close - (high + low + close) / 3) * volume, ts_mean(high - low, 20) + 1e-8), 20) + 1e-8)"
    ),
    "factor_asym_vol_gated": (
        "where(safe_div(volume, ts_ema(volume, 20)) > 1.5, "
        "log(cap(safe_div(safe_div(ts_sum(where(ts_pct(close, 1) < 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) < 0, 1, 0), 20) + 1e-8), "
        "safe_div(ts_sum(where(ts_pct(close, 1) >= 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) >= 0, 1, 0), 20) + 1e-8)), 1e-6, 1e6)) * 1.5, "
        "where(safe_div(volume, ts_ema(volume, 20)) < 0.6, "
        "log(cap(safe_div(safe_div(ts_sum(where(ts_pct(close, 1) < 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) < 0, 1, 0), 20) + 1e-8), "
        "safe_div(ts_sum(where(ts_pct(close, 1) >= 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) >= 0, 1, 0), 20) + 1e-8)), 1e-6, 1e6)) * 0.5, "
        "log(cap(safe_div(safe_div(ts_sum(where(ts_pct(close, 1) < 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) < 0, 1, 0), 20) + 1e-8), "
        "safe_div(ts_sum(where(ts_pct(close, 1) >= 0, high - low, 0), 20), "
        "ts_sum(where(ts_pct(close, 1) >= 0, 1, 0), 20) + 1e-8)), 1e-6, 1e6))))"
    ),
    "factor_lagret_volratio_gated_smooth": (
        "where(ts_std(ts_pct(close, 1), 20) > ts_median(ts_std(ts_pct(close, 1), 20), 60), "
        "ts_ema(delay(ts_pct(close, 5), 1) * log(cap(safe_div(volume, ts_ema(volume, 15)), 1e-12, 1e18)), 10), "
        "ts_ema(delay(ts_pct(close, 5), 1) * log(cap(safe_div(volume, ts_ema(volume, 15)), 1e-12, 1e18)), 5))"
    ),
    # --- lookahead-deferred (causal ts_rank / fixed forms) ---
    "factor_adaptive_vol_volume_asym": (
        "log(safe_div(ts_ema(where(close >= open, volume, 0), 10), "
        "ts_ema(where(close < open, volume, 0), 10) + 1e-10) + 1e-10) * ts_rank(high - low, 252)"
    ),
    "factor_drawdown_vol_simple": (
        "ts_rank(1 - safe_div(close, ts_max(close, 252)), 252) "
        "* ts_rank(safe_div(volume, ts_mean(volume, 20)), 252)"
    ),
    "factor_volatility_adjusted_dollar_pressure": (
        "ts_rank(safe_div(ts_ema((close - open) * volume, 21), ATR_WILDER(high, low, close, 21) + 1e-8), 252) * 2 - 1"
    ),
    "factor_pressure_ema_mutation": (
        "ts_rank(safe_div(ts_ema((close - open) * volume, 21) "
        "* (1 + safe_div(delay(close, 1) - close, delay(close, 1))), "
        "ATR_WILDER(high, low, close, 21) + 1e-8), 252) * 2 - 1"
    ),
    "factor_simplified_fear_pressure": (
        "ts_rank(safe_div(ts_mean((close - open) * volume, 21) "
        "* (1 + safe_div(delay(close, 1) - close, delay(close, 1))), "
        "ATR_WILDER(high, low, close, 21) + 1e-8), 252) * 2 - 1"
    ),
    "factor_herding_pressure": (
        "ts_corr(ts_rank(ts_pct(close, 1), 252), ts_rank(ts_pct(volume, 1), 252), 20) "
        "* ts_rank(safe_div(ts_ema((close - open) * volume, 5), ts_std((close - open) * volume, 20) + 1e-10), 252)"
    ),
    "factor_stability_crash_guard": (
        "safe_div(ts_ema(ts_pct(close, 1), 21), ts_ema(abs(ts_pct(close, 1)), 21) + 1e-8) "
        "* safe_div(volume, ts_mean(volume, 21)) "
        "* (1 - ts_rank(-safe_div("
        "ts_mean(pow(ts_pct(close, 1), 3), 20) - 3 * ts_mean(ts_pct(close, 1), 20) * ts_mean(pow(ts_pct(close, 1), 2), 20) "
        "+ 2 * pow(ts_mean(ts_pct(close, 1), 20), 3), "
        "pow(ts_std(ts_pct(close, 1), 20), 3) + 1e-8), 252))"
    ),
    # --- weekly python-9 hand DSL ---
    "cand_vw_ew_spread_responsive": (
        "safe_div(ema(ts_pct(close,1)*volume,19),ema(volume,19))-ema(ts_pct(close,1),19)"
    ),
    "cand_overnight_intraday_divergence_simplified": (
        "ema(safe_div(close,open)-1,5)-ema(safe_div(open,delay(close,1))-1,5)"
    ),
    "cand_fusion_upcap_volregime": (
        "safe_div(ts_sum(volume*where(ts_pct(close,1)>0,1,0),20),ts_sum(volume,20))*(1+(safe_div(volume,ema(volume,20))-1)*0.5)"
    ),
    "cand_gap_reversal_intensity": (
        "ts_sum((-where(safe_div(open,delay(close,1))-1<0,safe_div(open,delay(close,1))-1,0))*where(safe_div(close,open)-1>0,safe_div(close,open)-1,0),5)"
    ),
    "cand_gpdev_volregime_ema10": (
        "ema((safe_div(close,sqrt(high*low))-1)*(safe_div(volume,ema(volume,20))-1),10)"
    ),
    "cand_gap_vol_cluster_adaptive_w30": (
        "(2*sigmoid(2*(((safe_div(close,open)-1)-(safe_div(open,delay(close,1))-1))*safe_div(volume,ema(volume,20))*10))-1)*(1+0.5*(2*sigmoid(2*((-ts_corr(power(ts_pct(close,1),2),delay(power(ts_pct(close,1),2),1),30))*2))-1))"
    ),
    "cand_reversal_volregime_overnight_csz": (
        "(-ts_pct(close,3)*safe_div(ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),5),ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),20)))*(1/(1+safe_div(abs(safe_div(open,delay(close,1))-1),ts_std(safe_div(open,delay(close,1))-1,20))))"
    ),
    "cand_vol_expansion_atr_ratio": (
        "sign(close-open)*sigmoid(5*(safe_div(ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),5),ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),60))-1))"
    ),
    "cand_geometric_deviation_vol_asym_tilt": (
        "ema(sign(((2*log(clip(close,1e-12,1e18))-log(clip(high,1e-12,1e18))-log(clip(low,1e-12,1e18)))/3))*power(((2*log(clip(close,1e-12,1e18))-log(clip(high,1e-12,1e18))-log(clip(low,1e-12,1e18)))/3),2),5)*(1+where(safe_div(ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20))>0,safe_div(ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)),0))"
    ),
    "factor_vw_ew_spread_responsive": (
        "safe_div(ema(ts_pct(close,1)*volume,19),ema(volume,19))-ema(ts_pct(close,1),19)"
    ),
    "factor_overnight_intraday_divergence_simplified": (
        "ema(safe_div(close,open)-1,5)-ema(safe_div(open,delay(close,1))-1,5)"
    ),
    "factor_fusion_upcap_volregime": (
        "safe_div(ts_sum(volume*where(ts_pct(close,1)>0,1,0),20),ts_sum(volume,20))*(1+(safe_div(volume,ema(volume,20))-1)*0.5)"
    ),
    "factor_gap_reversal_intensity": (
        "ts_sum((-where(safe_div(open,delay(close,1))-1<0,safe_div(open,delay(close,1))-1,0))*where(safe_div(close,open)-1>0,safe_div(close,open)-1,0),5)"
    ),
    "factor_gpdev_volregime_ema10": (
        "ema((safe_div(close,sqrt(high*low))-1)*(safe_div(volume,ema(volume,20))-1),10)"
    ),
    "factor_gap_vol_cluster_adaptive_w30": (
        "(2*sigmoid(2*(((safe_div(close,open)-1)-(safe_div(open,delay(close,1))-1))*safe_div(volume,ema(volume,20))*10))-1)*(1+0.5*(2*sigmoid(2*((-ts_corr(power(ts_pct(close,1),2),delay(power(ts_pct(close,1),2),1),30))*2))-1))"
    ),
    "factor_reversal_volregime_overnight_csz": (
        "(-ts_pct(close,3)*safe_div(ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),5),ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),20)))*(1/(1+safe_div(abs(safe_div(open,delay(close,1))-1),ts_std(safe_div(open,delay(close,1))-1,20))))"
    ),
    "factor_vol_expansion_atr_ratio": (
        "sign(close-open)*sigmoid(5*(safe_div(ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),5),ts_mean(where(high-low>where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1))),high-low,where(abs(high-delay(close,1))>abs(low-delay(close,1)),abs(high-delay(close,1)),abs(low-delay(close,1)))),60))-1))"
    ),
    "factor_geometric_deviation_vol_asym_tilt": (
        "ema(sign(((2*log(clip(close,1e-12,1e18))-log(clip(high,1e-12,1e18))-log(clip(low,1e-12,1e18)))/3))*power(((2*log(clip(close,1e-12,1e18))-log(clip(high,1e-12,1e18))-log(clip(low,1e-12,1e18)))/3),2),5)*(1+where(safe_div(ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20))>0,safe_div(ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)-ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20),ts_std(where(ts_pct(close,1)>0,ts_pct(close,1),0),20)+ts_std(where(ts_pct(close,1)<0,ts_pct(close,1),0),20)),0))"
    ),
}
