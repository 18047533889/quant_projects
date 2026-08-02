def factor_mut_hc004_volpctrank_s12_clip2_5_volwin6(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol_rank = df_copy['volume'].rolling(6, min_periods=6).rank(pct=True)
    stress_range = (df_copy['high'] - df_copy['low']) / df_copy['close'].replace(0, np.nan)
    raw = ret * vol_rank * stress_range
    smoothed = raw.ewm(span=12, min_periods=12, adjust=False).mean()
    factor = smoothed.clip(-2.5, 2.5)
    factor.name = 'factor_mut_hc004_volpctrank_s12_clip2_5_volwin6'
    return factor
