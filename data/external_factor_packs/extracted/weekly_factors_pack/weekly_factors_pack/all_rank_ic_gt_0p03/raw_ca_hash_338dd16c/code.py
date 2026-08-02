def factor_directional_vol_smooth_range_mut(df):
    df_copy = df.copy()
    range_pct = (df_copy['high'] - df_copy['low']) / df_copy['close']
    smooth_range = range_pct.ewm(span=8).mean()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(25, min_periods=25).mean()
    fast = vol_ratio.ewm(span=10).mean()
    slow = vol_ratio.ewm(span=30).mean()
    weight = 1 / (1 + np.exp(-5 * (vol_ratio - 0.85)))
    smooth_vol_ratio = weight * fast + (1 - weight) * slow
    direction = (df_copy['close'] - df_copy['open']) / df_copy['close']
    df_copy['factor_directional_vol_smooth_range_mut'] = smooth_range * smooth_vol_ratio * direction
    return df_copy['factor_directional_vol_smooth_range_mut']
