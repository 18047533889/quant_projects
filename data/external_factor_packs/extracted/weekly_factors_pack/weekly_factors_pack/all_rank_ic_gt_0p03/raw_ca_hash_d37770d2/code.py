def factor_volume_confirmed_momentum(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    signal = ret * vol_ratio
    smoothed = signal.ewm(span=10, min_periods=10, adjust=False).mean()
    df_copy['factor_volume_confirmed_momentum'] = smoothed
    return df_copy['factor_volume_confirmed_momentum']
