def factor_volume_weighted_price_change(df):
    df_copy = df.copy()
    close = df_copy['close']
    volume = df_copy['volume']
    pct_change = close.pct_change()
    vol_ma = volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    vol_ratio = volume / vol_ma
    raw = pct_change * vol_ratio
    raw = raw.replace([np.inf, -np.inf], np.nan)
    ema = raw.ewm(span=5, min_periods=5).mean()
    df_copy['factor_volume_weighted_price_change'] = ema
    return df_copy['factor_volume_weighted_price_change']
