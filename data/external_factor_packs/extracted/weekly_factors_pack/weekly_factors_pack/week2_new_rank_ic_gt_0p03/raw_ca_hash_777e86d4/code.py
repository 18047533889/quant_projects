def factor_volume_confirmed_momentum_binary_gate(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    base = ret * vol_ratio
    smoothed_base = base.ewm(span=10, min_periods=10, adjust=False).mean()
    range_ = df_copy['high'] - df_copy['low']
    range_ = range_.replace(0, np.nan)
    mid_point = (df_copy['high'] + df_copy['low']) / 2
    binary_gate = np.where(df_copy['close'] >= mid_point, 1, -1)
    factor = smoothed_base * binary_gate
    df_copy['factor_volume_confirmed_momentum_binary_gate'] = factor
    return df_copy['factor_volume_confirmed_momentum_binary_gate']
