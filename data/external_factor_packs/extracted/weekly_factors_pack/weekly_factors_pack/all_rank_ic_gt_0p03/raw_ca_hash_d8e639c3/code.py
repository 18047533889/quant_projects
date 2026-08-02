def factor_range_position_volume_gated_smoothed_mut(df):
    df_copy = df.copy()
    low_ = df_copy["low"]
    high_ = df_copy["high"]
    close_ = df_copy["close"]
    volume_ = df_copy["volume"]
    range_ = high_ - low_
    pos = (close_ - low_) / range_.replace(0.0, np.nan)
    pos = pos - 0.5
    smooth_pos = pos.ewm(span=5, min_periods=5, adjust=False).mean()
    vol_20 = volume_.rolling(20, min_periods=20).mean()
    vol_ratio = volume_ / vol_20.replace(0.0, np.nan)
    factor = smooth_pos * vol_ratio
    df_copy["factor_range_position_volume_gated_smoothed_mut"] = factor
    return df_copy["factor_range_position_volume_gated_smoothed_mut"]
