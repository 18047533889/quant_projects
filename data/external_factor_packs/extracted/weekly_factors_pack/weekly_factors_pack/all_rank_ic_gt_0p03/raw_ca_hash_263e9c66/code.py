def factor_imbalance_volume_ratio_ewm(df):
    df_copy = df.copy()
    midpoint = (df_copy['high'] + df_copy['low']) / 2
    range_ = df_copy['high'] - df_copy['low']
    imbalance = (df_copy['close'] - midpoint) / range_.replace(0, np.nan)
    vol_ma = df_copy['volume'].rolling(20, min_periods=20).mean()
    vol_ratio = df_copy['volume'] / vol_ma.replace(0, np.nan)
    weighted = imbalance * vol_ratio
    df_copy['factor_imbalance_volume_ratio_ewm'] = weighted.ewm(span=5, min_periods=3).mean()
    return df_copy['factor_imbalance_volume_ratio_ewm']
