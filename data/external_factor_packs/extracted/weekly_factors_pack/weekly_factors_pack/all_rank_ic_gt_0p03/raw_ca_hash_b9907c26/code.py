def factor_pressure_tanh(df):
    df_copy = df.copy()
    range_ = df_copy["high"] - df_copy["low"]
    safe_range = range_.replace(0, np.nan)
    intra_pressure = (df_copy["close"] - df_copy["low"]) / safe_range * df_copy["volume"]
    rolling_mean = intra_pressure.rolling(20, min_periods=20).mean()
    rolling_std = intra_pressure.rolling(20, min_periods=20).std(ddof=1)
    z = (intra_pressure - rolling_mean) / rolling_std.replace(0, np.nan)
    factor = np.tanh(z)
    factor = factor.fillna(0)
    df_copy["factor_pressure_tanh"] = factor
    return df_copy["factor_pressure_tanh"]
