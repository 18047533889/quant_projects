def factor_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5(df):
    df_copy = df.copy()
    high = df_copy['high']
    low = df_copy['low']
    close = df_copy['close']
    open_ = df_copy['open']
    volume = df_copy['volume']

    # Asymmetry ratio using median range (window 20, min_periods=3)
    up_day = (close > open_).astype(float)
    down_day = (close <= open_).astype(float)
    range_ = high - low
    up_range = range_.where(up_day == 1, np.nan)
    down_range = range_.where(down_day == 1, np.nan)
    up_med = up_range.rolling(20, min_periods=3).median()
    down_med = down_range.rolling(20, min_periods=3).median()
    up_med = up_med.fillna(1.0)
    down_med = down_med.fillna(1.0)
    asym_ratio = up_med / down_med.replace(0.0, np.nan)
    asym_ratio = asym_ratio.fillna(1.0)

    # Volume ratio using median (window 6, min_periods=6)
    vol_med6 = volume.rolling(6, min_periods=6).median()
    vol_ratio = volume / vol_med6.replace(0.0, np.nan)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Combine with negative sign
    factor = -1.0 * asym_ratio * vol_ratio
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)

    # EMA5 smoothing (increased from 3 to reduce noise)
    factor_smoothed = factor.ewm(span=5, min_periods=5, adjust=False).mean()
    factor_smoothed = factor_smoothed.replace([np.inf, -np.inf], np.nan).fillna(0)

    df_copy['factor_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5'] = factor_smoothed
    return df_copy['factor_mut_cro_asym_vol_ratio_med_20_minp3_vol6_ema5']
