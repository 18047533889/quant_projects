def factor_herding_momentum_stress(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(5, min_periods=5).mean()
    stress_range = (df_copy['high'] - df_copy['low']) / df_copy['close'].replace(0, np.nan)
    raw = ret * vol_ratio * stress_range
    smoothed = raw.ewm(span=5, min_periods=5, adjust=False).mean()
    factor = smoothed.clip(-3, 3)
    factor.name = 'factor_herding_momentum_stress'
    return factor
