def factor_volume_confirmed_momentum_clipped(df):
    df_copy = df.copy()
    momentum = df_copy["close"] / df_copy["close"].shift(5) - 1
    vol_avg = df_copy["volume"].rolling(20, min_periods=20).mean()
    vol_ratio = df_copy["volume"] / vol_avg.replace(0, np.nan)
    factor = momentum * vol_ratio
    factor = factor.clip(-3, 3)
    factor.name = "factor_volume_confirmed_momentum_clipped"
    return factor
