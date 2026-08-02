def factor_herding_simple(df):
    df_copy = df.copy()
    range_ = df_copy["high"] - df_copy["low"]
    intraday_direction = (df_copy["close"] - df_copy["open"]) / range_.replace(0.0, np.nan)
    vol_median = df_copy["volume"].rolling(10, min_periods=10).median()
    vol_ratio = df_copy["volume"] / vol_median.replace(0.0, np.nan)
    herding_signal = intraday_direction * vol_ratio
    factor = herding_signal.ewm(span=5, min_periods=5, adjust=False).mean()
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    factor.name = "factor_herding_simple"
    return factor
