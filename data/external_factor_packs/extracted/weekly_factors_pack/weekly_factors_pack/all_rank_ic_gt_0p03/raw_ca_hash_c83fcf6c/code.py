def factor_vol_regime_breakout_continuous(df):
    df_copy = df.copy()
    range_ = df_copy['high'] - df_copy['low']
    range_med = range_.rolling(20).median()
    pos = (df_copy['close'] - df_copy['low']) / range_.replace(0, np.nan)
    vol_ratio = range_ / range_med.replace(0, np.nan)
    factor = (pos - 0.5) * vol_ratio
    factor = factor.ewm(span=5, min_periods=5).mean()
    factor.name = 'factor_vol_regime_breakout_continuous'
    return factor
