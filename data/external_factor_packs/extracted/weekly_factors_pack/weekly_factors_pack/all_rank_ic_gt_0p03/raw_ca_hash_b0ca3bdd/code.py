def factor_crossover_herding_stress_span12_volrank_clip3(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_rank = df_copy['volume'].rolling(8, min_periods=8).rank(pct=True)
    stress_range = (df_copy['high'] - df_copy['low']) / df_copy['close'].replace(0, np.nan)
    raw = ret * vol_rank * stress_range
    smoothed = raw.ewm(span=12, min_periods=12, adjust=False).mean()
    factor = smoothed.clip(-3, 3)
    factor.name = 'factor_crossover_herding_stress_span12_volrank_clip3'
    return factor
