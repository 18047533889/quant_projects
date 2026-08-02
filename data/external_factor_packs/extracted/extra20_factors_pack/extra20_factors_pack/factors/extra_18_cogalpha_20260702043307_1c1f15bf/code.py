def factor_reversal_intraday_5d_ema(df):
    """Intraday return reversal: negative of 5-period EMA of intraday returns."""
    df_copy = df.copy()
    overnight_ret, intraday_ret = alpha_tools.decompose_overnight_intraday(df_copy["close"], df_copy["open"])
    ema_intraday = intraday_ret.ewm(span=5, min_periods=5).mean()
    df_copy["factor_reversal_intraday_5d_ema"] = -ema_intraday
    return df_copy["factor_reversal_intraday_5d_ema"]
