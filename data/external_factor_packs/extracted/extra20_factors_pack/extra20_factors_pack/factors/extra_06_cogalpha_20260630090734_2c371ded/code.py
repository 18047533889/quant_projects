def factor_mut_cross_vwpc_z10_clip2_v3(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    price_change = close.pct_change()
    vol_ratio = volume / volume.rolling(20, min_periods=20).mean()
    range_ = df_copy['high'] - df_copy['low']
    range_z = (range_ - range_.rolling(10, min_periods=10).mean()) / range_.rolling(10, min_periods=10).std(ddof=1)
    raw = price_change * vol_ratio * (1 + range_z.clip(-2, 2))
    smoothed = raw.ewm(span=10, min_periods=10).mean()
    df_copy['factor_mut_cross_vwpc_z10_clip2_v3'] = smoothed
    return df_copy['factor_mut_cross_vwpc_z10_clip2_v3']
