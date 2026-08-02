def factor_stress_range_volume_surge(df):
    df_copy = df.copy()
    stress_range = (df_copy['high'] - df_copy['low']) / df_copy['close'].replace(0, np.nan)
    vol_ma = df_copy['volume'].rolling(20, min_periods=20).mean().replace(0, np.nan)
    vol_surge = df_copy['volume'] / vol_ma
    factor = stress_range * vol_surge
    df_copy['factor_stress_range_volume_surge'] = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_copy['factor_stress_range_volume_surge']
