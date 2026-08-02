def factor_vol_gated_momentum_volume_weighted(df):
    df_copy = df.copy()
    close = df_copy['close']
    high = df_copy['high']
    low = df_copy['low']
    volume = df_copy['volume']
    # 5-day return
    ret5 = close / close.shift(5) - 1
    # 14-day ATR
    atr14 = talib.ATR(high.astype(float).values, low.astype(float).values, close.astype(float).values, timeperiod=14)
    atr14 = pd.Series(atr14, index=df_copy.index)
    # Normalized volatility
    norm_vol = atr14 / close
    # Time-series rank percentile (continuous gate)
    vol_rank = norm_vol.rolling(20, min_periods=20).rank(pct=True)
    # Weight: 0.5 (low vol) to 1.0 (high vol)
    weight = 0.5 + 0.5 * vol_rank
    gated_ret = ret5 * weight
    # Volume factor: relative volume (20-day SMA)
    vol_ma = volume.rolling(20, min_periods=20).mean()
    rel_vol = volume / vol_ma.replace(0, np.nan)
    # Combine: gated momentum amplified by volume conviction
    factor = gated_ret * rel_vol
    df_copy['factor_vol_gated_momentum_volume_weighted'] = factor
    return df_copy['factor_vol_gated_momentum_volume_weighted']
