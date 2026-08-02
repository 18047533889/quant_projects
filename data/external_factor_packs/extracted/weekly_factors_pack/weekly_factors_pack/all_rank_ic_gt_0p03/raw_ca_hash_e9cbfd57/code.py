def factor_intraday_volume_spike(df):
    df_copy = df.copy()
    intday_ret = df_copy["close"] / df_copy["open"] - 1.0
    vol_mean = df_copy["volume"].rolling(20, min_periods=20).mean()
    vol_surge = df_copy["volume"] / vol_mean.replace(0, np.nan)
    raw = intday_ret * vol_surge
    df_copy["factor_intraday_volume_spike"] = raw.ewm(span=10, adjust=False).mean()
    return df_copy["factor_intraday_volume_spike"]
