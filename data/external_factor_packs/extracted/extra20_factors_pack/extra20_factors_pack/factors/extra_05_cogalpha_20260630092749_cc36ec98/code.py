def factor_volume_pressure_ewm_smooth(df):
    df_copy = df.copy()
    ret = df_copy['close'] / df_copy['open'] - 1
    vol_med = df_copy['volume'].rolling(10, min_periods=10).median().replace(0, np.nan)
    vol_ratio = df_copy['volume'] / vol_med
    vol_ratio_clipped = vol_ratio.clip(lower=0.3, upper=2.0)
    raw = ret * vol_ratio_clipped
    smooth = raw.ewm(halflife=7, min_periods=7).mean()
    return smooth.rename('factor_volume_pressure_ewm_smooth')
