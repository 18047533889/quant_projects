def factor_intraday_volume_tanh_short(df):
    df_copy = df.copy()
    close = df_copy["close"]
    open_ = df_copy["open"]
    volume = df_copy["volume"]
    w_sma = 20
    w_ewm = 10
    intra_ret = close / open_.replace(0, np.nan) - 1
    vol_sma = volume.rolling(w_sma, min_periods=w_sma).mean()
    vol_surge = volume / vol_sma.replace(0, np.nan)
    raw = intra_ret * vol_surge
    factor = raw.ewm(span=w_ewm, adjust=False).mean()
    factor = np.tanh(factor)
    df_copy["factor_intraday_volume_tanh_short"] = factor.fillna(0)
    return df_copy["factor_intraday_volume_tanh_short"]
