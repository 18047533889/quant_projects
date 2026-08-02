def factor_adaptive_momentum_gate_v2(df):
    df_copy = df.copy()
    ret = df_copy['close'].pct_change()
    vol = ret.rolling(30, min_periods=30).std()
    momentum_5 = df_copy['close'] / df_copy['close'].shift(5) - 1
    factor = momentum_5 / vol.replace(0, np.nan)
    df_copy['factor_adaptive_momentum_gate_v2'] = factor
    return df_copy['factor_adaptive_momentum_gate_v2']
