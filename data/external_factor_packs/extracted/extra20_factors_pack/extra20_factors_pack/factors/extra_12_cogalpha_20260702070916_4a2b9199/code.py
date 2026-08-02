def factor_intraday_reversal_vol_intensity(df):
    """Intraday reversal scaled by relative volatility intensity."""
    df_copy = df.copy()
    overnight, intraday = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    intraday_ema = intraday.ewm(span=5, min_periods=5).mean()
    rv_short = df_copy["close"].pct_change().rolling(20).std()
    rv_long = df_copy["close"].pct_change().rolling(60).std()
    vol_ratio = rv_short / rv_long.replace(0, np.nan)
    vol_ratio = vol_ratio.clip(0.5, 2.0)
    result = -intraday_ema * vol_ratio
    result = result.fillna(0)
    df_copy["factor_intraday_reversal_vol_intensity"] = result
    return df_copy["factor_intraday_reversal_vol_intensity"]
