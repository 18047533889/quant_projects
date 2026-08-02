def factor_stress_volume_cont_weight_cont_dir_w20_clip2_5_w40_ewm15_stressclip25(df):
    df_copy = df.copy()
    high = df_copy["high"]
    low = df_copy["low"]
    close = df_copy["close"]
    open_ = df_copy["open"]
    volume = df_copy["volume"]
    range_ = high - low
    mean_range = range_.rolling(40, min_periods=40).mean()
    std_range = range_.rolling(40, min_periods=40).std(ddof=1)
    z = (range_ - mean_range) / std_range.replace(0, np.nan)
    stress_ewm = z.clip(lower=0).ewm(span=15, adjust=False).mean()
    stress_ewm = stress_ewm.clip(upper=2.5)
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_weight = volume / vol_mean.replace(0, np.nan)
    vol_weight = vol_weight.clip(upper=2)
    cont_dir = (close - open_) / range_.replace(0, np.nan)
    cont_dir = cont_dir.clip(lower=-2.5, upper=2.5)
    factor = stress_ewm * vol_weight * cont_dir
    df_copy["factor_stress_volume_cont_weight_cont_dir_w20_clip2_5_w40_ewm15_stressclip25"] = factor
    return df_copy["factor_stress_volume_cont_weight_cont_dir_w20_clip2_5_w40_ewm15_stressclip25"]
