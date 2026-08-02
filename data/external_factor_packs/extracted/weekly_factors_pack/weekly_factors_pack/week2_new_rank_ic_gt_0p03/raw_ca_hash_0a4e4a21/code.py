def factor_volume_confirmed_momentum_tanh_gated(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    base = ret * vol_ratio
    smoothed_base = base.ewm(span=10, min_periods=10, adjust=False).mean()
    range_ = df_copy['high'] - df_copy['low']
    range_ = range_.replace(0, np.nan)
    intraday_pos = (df_copy['close'] - df_copy['low']) / range_
    raw_factor = smoothed_base * intraday_pos
    factor = np.tanh(raw_factor)
    df_copy['factor_volume_confirmed_momentum_tanh_gated'] = factor
    return df_copy['factor_volume_confirmed_momentum_tanh_gated']
