def factor_directional_vol_volbase25_smooth_dir_range10(df):
    df_copy = df.copy()
    range_pct = (df_copy['high'] - df_copy['low']) / df_copy['close']
    smooth_range = range_pct.ewm(span=10, min_periods=10).mean()
    vol_ratio = df_copy['volume'] / df_copy['volume'].rolling(25, min_periods=25).mean()
    smooth_vol = vol_ratio.ewm(span=15, min_periods=15).mean()
    direction = (df_copy['close'] - df_copy['open']) / df_copy['close']
    smooth_direction = direction.ewm(span=3, min_periods=3).mean()
    df_copy['factor_directional_vol_volbase25_smooth_dir_range10'] = smooth_range * smooth_vol * smooth_direction
    return df_copy['factor_directional_vol_volbase25_smooth_dir_range10']
