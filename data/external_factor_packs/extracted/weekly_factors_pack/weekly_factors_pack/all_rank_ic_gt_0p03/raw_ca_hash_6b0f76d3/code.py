def factor_intraday_return_volume_surge(df):
    df_copy = df.copy()
    intday_ret = df_copy["close"] / df_copy["open"] - 1.0
    vol_max = df_copy["volume"].rolling(20, min_periods=20).max()
    vol_surge = df_copy["volume"] / vol_max.replace(0, np.nan)
    raw = intday_ret * vol_surge
    df_copy["factor_intraday_return_volume_surge"] = raw.ewm(span=20, adjust=False).mean()
    return df_copy["factor_intraday_return_volume_surge"]
