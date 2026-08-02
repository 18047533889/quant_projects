def factor_intraday_volume_tanh(df):
    df_copy = df.copy()
    close = df_copy["close"]
    open_ = df_copy["open"]
    volume = df_copy["volume"]
    w = 20
    intra_ret = close / open_.replace(0, np.nan) - 1
    vol_sma = volume.rolling(w, min_periods=w).mean()
    vol_surge = volume / vol_sma.replace(0, np.nan)
    raw = intra_ret * vol_surge
    factor = raw.ewm(span=w, adjust=False).mean()
    factor = np.tanh(factor)
    df_copy["factor_intraday_volume_tanh"] = factor.fillna(0)
    return df_copy["factor_intraday_volume_tanh"]
