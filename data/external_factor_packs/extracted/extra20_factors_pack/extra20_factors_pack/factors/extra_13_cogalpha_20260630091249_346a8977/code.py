def factor_mut_cross_vwpc_z10_clip2_span8_v1(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    price_change = close.pct_change()
    vol_ratio = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    range_ = df_copy['high'] - df_copy['low']
    range_mean = range_.rolling(10, min_periods=10).mean()
    range_std = range_.rolling(10, min_periods=10).std(ddof=1)
    range_z = (range_ - range_mean) / range_std.replace(0, np.nan)
    raw = price_change * vol_ratio * (1 + range_z.clip(-2, 2))
    smoothed = raw.ewm(span=8, min_periods=8).mean()
    df_copy['factor_mut_cross_vwpc_z10_clip2_span8_v1'] = smoothed
    return df_copy['factor_mut_cross_vwpc_z10_clip2_span8_v1']
