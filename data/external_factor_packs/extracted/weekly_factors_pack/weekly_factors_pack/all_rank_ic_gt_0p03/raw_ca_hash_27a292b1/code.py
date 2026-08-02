def factor_vol_price_log_range_gate_clipped_boost_5day_mut_001(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    high = df_copy['high']
    low = df_copy['low']
    pct_change = close.pct_change()
    vol_ma = volume.rolling(5, min_periods=5).mean().replace(0, np.nan)
    vol_ratio = volume / vol_ma
    vol_weight = np.log1p(vol_ratio)
    raw = pct_change * vol_weight
    range_ = high - low
    range_mean = range_.rolling(5, min_periods=5).mean()
    range_std = range_.rolling(5, min_periods=5).std(ddof=1).replace(0, np.nan)
    range_zscore = (range_ - range_mean) / range_std
    boost = np.where(range_zscore > 0, np.minimum(range_zscore, 2.0), 0)
    raw_weighted = raw * (1 + boost)
    raw_weighted = raw_weighted.replace([np.inf, -np.inf], np.nan)
    ema = raw_weighted.ewm(span=12, min_periods=12).mean()
    df_copy['factor_vol_price_log_range_gate_clipped_boost_5day_mut_001'] = ema
    return df_copy['factor_vol_price_log_range_gate_clipped_boost_5day_mut_001']
