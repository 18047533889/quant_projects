def factor_vol_regime_trend_smoothed(df):
    df_copy = df.copy()
    price_ratio = df_copy['close'] / df_copy['close'].rolling(20, min_periods=20).mean() - 1
    volume_surge = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    range_ = df_copy['high'] - df_copy['low']
    median_range = range_.rolling(20, min_periods=20).median()
    vol_dev = range_ / median_range - 1
    signal = price_ratio * volume_surge * vol_dev
    smoothed = signal.ewm(span=5, adjust=False).mean()
    df_copy['factor_vol_regime_trend_smoothed'] = smoothed.fillna(0.0)
    return df_copy['factor_vol_regime_trend_smoothed']
