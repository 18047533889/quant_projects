def factor_imbalance_volume_ewm_mut(df):
    df_copy = df.copy()
    midpoint = (df_copy['high'] + df_copy['low']) / 2
    range_ = df_copy['high'] - df_copy['low']
    imbalance = (df_copy['close'] - midpoint) / range_.replace(0, np.nan)
    vol_ma = df_copy['volume'].rolling(10, min_periods=10).mean()
    vol_dev = df_copy['volume'] / vol_ma.replace(0, np.nan) - 1
    weighted = imbalance * df_copy['volume'] * (1 + vol_dev)
    df_copy['factor_imbalance_volume_ewm_mut'] = weighted.ewm(span=5, min_periods=3).mean()
    return df_copy['factor_imbalance_volume_ewm_mut']
