def factor_volume_weighted_price_change(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    price_change = close.pct_change()
    vol_ratio = volume / volume.rolling(20, min_periods=20).mean()
    range_ = df_copy['high'] - df_copy['low']
    range_z = (range_ - range_.rolling(20, min_periods=20).mean()) / range_.rolling(20, min_periods=20).std(ddof=1)
    raw = price_change * vol_ratio * (1 + range_z.clip(-3, 3))
    smoothed = raw.ewm(span=10, min_periods=10).mean()
    df_copy['factor_volume_weighted_price_change'] = smoothed
    return df_copy['factor_volume_weighted_price_change']
