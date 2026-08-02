def factor_volume_pressure_imbalance(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['open'] - 1
    vol_med = df_copy['volume'].rolling(20, min_periods=20).median().replace(0, np.nan)
    vol_ratio = df_copy['volume'] / vol_med
    raw = ret * vol_ratio
    smooth = raw.ewm(halflife=5, min_periods=20).mean()
    return smooth.rename('factor_volume_pressure_imbalance')
