def factor_volume_adjusted_momentum(df):
    df_copy = df.copy()
    ret = df_copy["close"].pct_change()
    vol_ratio = df_copy["volume"] / df_copy["volume"].rolling(20, min_periods=20).mean().replace(0, np.nan)
    raw_signal = ret * vol_ratio
    signal = raw_signal.ewm(span=5, adjust=False).mean()
    df_copy["factor_volume_adjusted_momentum"] = signal
    return df_copy["factor_volume_adjusted_momentum"]
