def factor_intraday_reversal_crossover_v1(df):
    df_copy = df.copy()
    intraday_pos = (2*df_copy['close'] - df_copy['high'] - df_copy['low']) / (df_copy['high'] - df_copy['low']).replace(0, np.nan)
    avg_range = (df_copy['high'] - df_copy['low']).rolling(5).mean()
    avg_close = df_copy['close'].rolling(5).mean()
    raw_factor = -intraday_pos * avg_range / avg_close.replace(0, np.nan)
    smoothed = raw_factor.rolling(3, min_periods=3).mean()
    smoothed.name = 'factor_intraday_reversal_crossover_v1'
    return smoothed
