#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 extend_all_456 的 _AlphaTools / _MinuteTools 补齐因子 code 依赖但缺失的工具方法，
使 backfill_28_factors.py 的 B 类（HASCODE 日线）因子可以执行。
用法：from patch_alpha_tools import patch_tools; patch_tools()
"""
import numpy as np
import pandas as pd


def patch_tools():
    import extend_all_456 as E

    # ---- _AlphaTools 补: volume_momentum_ratio / intraday_close_location / decompose_return_volatility_direction ----
    if not hasattr(E.alpha_tools, "volume_momentum_ratio"):
        def volume_momentum_ratio(volume, long_window=20, short_window=5):
            lv = pd.Series(volume).rolling(long_window, min_periods=5).mean().replace(0, np.nan)
            if short_window == 1:
                sv = pd.Series(volume)
            else:
                sv = pd.Series(volume).rolling(short_window, min_periods=1).mean()
            return (sv / lv).fillna(1.0)
        E.alpha_tools.volume_momentum_ratio = staticmethod(volume_momentum_ratio)

    if not hasattr(E.alpha_tools, "intraday_close_location"):
        def intraday_close_location(df, window=20):
            hi = df["high"]; lo = df["low"]; cl = df["close"]
            rng = (hi - lo).replace(0, np.nan)
            return ((cl - lo) / rng).clip(0.0, 1.0)
        E.alpha_tools.intraday_close_location = staticmethod(intraday_close_location)

    if not hasattr(E.alpha_tools, "decompose_return_volatility_direction"):
        def decompose_return_volatility_direction(close, window=20, min_periods=10):
            ret = pd.Series(close).pct_change().fillna(0)
            pos = ret.clip(lower=0).abs()
            neg = ret.clip(upper=0).abs()
            upside_vol = pos.rolling(window, min_periods=min_periods).mean()
            downside_vol = neg.rolling(window, min_periods=min_periods).mean()
            return upside_vol, downside_vol
        E.alpha_tools.decompose_return_volatility_direction = staticmethod(decompose_return_volatility_direction)

    if not hasattr(E.alpha_tools, "directional_efficiency"):
        def directional_efficiency(close, high=None, low=None, window=14):
            # 兼容 DataFrame（df 整体）/ Series / ndarray
            if isinstance(close, pd.DataFrame):
                s = close["close"] if "close" in close.columns else close.iloc[:, 0]
            elif isinstance(close, pd.Series):
                s = close
            else:
                s = pd.Series(close)
            return (s - s.shift(window)).abs() / s.diff().abs().rolling(window, min_periods=1).sum().replace(0, np.nan)
        E.alpha_tools.directional_efficiency = staticmethod(directional_efficiency)

    if not hasattr(E.alpha_tools, "average_true_range"):
        def average_true_range(df, window=14):
            return E.talib.ATR(df["high"], df["low"], df["close"], timeperiod=window)
        E.alpha_tools.average_true_range = staticmethod(average_true_range)

    # ---- _MinuteTools 补: pulse_start / amount_weighted_mean ----
    if not hasattr(E.minute_tools, "pulse_start"):
        def pulse_start(vol, window=5):
            s = pd.Series(vol)
            return (s > s.rolling(window, min_periods=1).mean() * 2).astype(float)
        E.minute_tools.pulse_start = staticmethod(pulse_start)

    if not hasattr(E.minute_tools, "amount_weighted_mean"):
        def amount_weighted_mean(values, amounts, trading_days):
            df_local = pd.DataFrame({"value": pd.Series(values), "amount": pd.Series(amounts), "date": trading_days})
            df_local = df_local.dropna(subset=["amount", "value"])
            if df_local.empty:
                return pd.Series(dtype=float)
            df_local = df_local.reset_index(drop=True)

            def w_mean(group):
                amt = group["amount"].values
                val = group["value"].values
                total = np.nansum(amt)
                if total == 0 or np.isnan(total):
                    return np.nan
                return np.nansum(amt * val) / total

            result = df_local.groupby("date", sort=True).apply(w_mean, include_groups=False)
            result.index = pd.to_datetime(result.index)
            return result
        E.minute_tools.amount_weighted_mean = staticmethod(amount_weighted_mean)

    print("[patch] _AlphaTools/_MinuteTools 已补齐")
    return E