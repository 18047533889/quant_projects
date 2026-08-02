def factor_directional_vol_adaptive_crossover(df):
    df_copy = df.copy()
    range_pct = (df_copy['high'] - df_copy['low']) / df_copy['close']
    smooth_range = range_pct.ewm(span=5).mean()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(20, min_periods=20).mean()
    smooth_vol_ratio_fast = vol_ratio.ewm(span=10).mean()
    smooth_vol_ratio_slow = vol_ratio.ewm(span=30).mean()
    smooth_vol_ratio = np.where(vol_ratio > 0.9, smooth_vol_ratio_fast, smooth_vol_ratio_slow)
    direction = (df_copy['close'] - df_copy['open']) / df_copy['close']
    df_copy['factor_directional_vol_adaptive_crossover'] = smooth_range * smooth_vol_ratio * direction
    return df_copy['factor_directional_vol_adaptive_crossover']
