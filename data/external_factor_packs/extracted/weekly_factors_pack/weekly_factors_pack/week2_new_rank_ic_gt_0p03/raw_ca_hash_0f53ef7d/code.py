def factor_vol_gated_momentum_v3(df):
    df_copy = df.copy()
    close = df_copy['close']
    # 5-day return
    ret5 = close / close.shift(5) - 1
    # 20-day rolling standard deviation of daily returns (volatility)
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(20, min_periods=20).std(ddof=1)
    # Time-series rank percentile
    vol_rank = vol.rolling(20, min_periods=20).rank(pct=True)
    # Inverse volatility weighting: low vol gets higher weight (1.0), high vol gets lower (0.5)
    weight = 1.5 - 0.5 * vol_rank
    gated_ret = ret5 * weight
    df_copy['factor_vol_gated_momentum_v3'] = gated_ret
    return df_copy['factor_vol_gated_momentum_v3']
