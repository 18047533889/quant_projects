def factor_vol_surge_regime_gated(df):
    df_copy = df.copy()
    intday_ret = df_copy["close"] / df_copy["open"] - 1.0
    vol_max = df_copy["volume"].rolling(20, min_periods=20).max()
    vol_surge = df_copy["volume"] / vol_max.replace(0, np.nan)
    raw = intday_ret * vol_surge
    range_ = df_copy["high"] - df_copy["low"]
    range_med = range_.rolling(20).median()
    high_vol = range_ > range_med
    factor = raw * high_vol
    factor = factor.ewm(span=20, adjust=False).mean()
    factor.name = "factor_vol_surge_regime_gated"
    return factor
