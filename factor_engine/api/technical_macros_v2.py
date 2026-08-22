# -*- coding: utf-8 -*-
"""DSL lowering for composite technical names.

The Pandas classes remain semantic-reference/oracle implementations, while the
normal DSL path expands these names to reusable primitives for CSE and backend
routing.  This removes duplicate *physical* kernels without breaking public API.
"""
from __future__ import annotations
from typing import Any,Callable,Mapping


def augment_technical_macros(base:Mapping[str,Callable[...,Any]])->dict[str,Callable[...,Any]]:
    out=dict(base)
    def f(name):
        if name not in out:raise KeyError(name)
        return out[name]
    add,sub,mul,div=f("add"),f("subtract"),f("multiply"),f("divide")
    maximum,minimum=f("maximum"),f("minimum")
    ts_max,ts_min,ts_mean,ts_std,ts_sum,ts_delay,ts_ema=f("ts_max"),f("ts_min"),f("ts_mean"),f("ts_std"),f("ts_sum"),f("ts_delay"),f("ts_ema")
    true_range=f("true_range")

    def prev_high(x,window):return ts_delay(ts_max(x,window),1)
    def prev_low(x,window):return ts_delay(ts_min(x,window),1)
    out["ts_prev_high"]=prev_high;out["ts_prev_low"]=prev_low
    out["ts_distance_to_high"]=lambda x,window:sub(div(x,prev_high(x,window)),1.0)
    out["ts_distance_to_low"]=lambda x,window:sub(div(x,prev_low(x,window)),1.0)
    # Positive-part macros use maximum with scalar zero.
    out["ts_breakout_high"]=lambda x,window:maximum(sub(div(x,prev_high(x,window)),1.0),0.0)
    out["ts_breakdown_low"]=lambda x,window:maximum(sub(div(prev_low(x,window),x),1.0),0.0)
    out["ts_new_high"]=lambda x,window:f("gt")(x,prev_high(x,window))
    out["ts_new_low"]=lambda x,window:f("lt")(x,prev_low(x,window))
    out["ts_channel_position"]=lambda x,window:div(sub(x,prev_low(x,window)),sub(prev_high(x,window),prev_low(x,window)))

    out["donchian_upper"]=lambda high,window:ts_delay(ts_max(high,window),1)
    out["donchian_lower"]=lambda low,window:ts_delay(ts_min(low,window),1)
    out["donchian_mid"]=lambda high,low,window:mul(add(out["donchian_upper"](high,window),out["donchian_lower"](low,window)),0.5)
    out["donchian_position"]=lambda high,low,close,window:div(sub(close,out["donchian_lower"](low,window)),sub(out["donchian_upper"](high,window),out["donchian_lower"](low,window)))

    out["bollinger_pct_b"]=lambda close,window,std_dev:div(sub(close,sub(ts_mean(close,window),mul(ts_std(close,window),std_dev))),mul(ts_std(close,window),mul(std_dev,2.0)))
    out["bollinger_width"]=lambda close,window,std_dev:div(mul(ts_std(close,window),mul(std_dev,2.0)),ts_mean(close,window))

    def natr(high,low,close,window):return mul(div(f("ATR_WILDER")(high,low,close,window),close),100.0)
    out["NATR"]=natr
    def ppo(x,fast_window,slow_window):
        fast=ts_ema(x,fast_window);slow=ts_ema(x,slow_window);return mul(div(sub(fast,slow),slow),100.0)
    out["PPO"]=lambda close,fast_window,slow_window:ppo(close,fast_window,slow_window)
    out["PPO_signal"]=lambda close,fast_window,slow_window,signal_window:ts_ema(ppo(close,fast_window,slow_window),signal_window)
    out["PPO_hist"]=lambda close,fast_window,slow_window,signal_window:sub(ppo(close,fast_window,slow_window),ts_ema(ppo(close,fast_window,slow_window),signal_window))
    out["PVO"]=lambda volume,fast_window,slow_window:ppo(volume,fast_window,slow_window)
    out["PVO_signal"]=lambda volume,fast_window,slow_window,signal_window:ts_ema(ppo(volume,fast_window,slow_window),signal_window)
    out["PVO_hist"]=lambda volume,fast_window,slow_window,signal_window:sub(ppo(volume,fast_window,slow_window),ts_ema(ppo(volume,fast_window,slow_window),signal_window))

    out["KeltnerMid"]=lambda close,ema_window:ts_ema(close,ema_window)
    out["KeltnerUpper"]=lambda high,low,close,ema_window,atr_window,multiplier:add(ts_ema(close,ema_window),mul(f("ATR_WILDER")(high,low,close,atr_window),multiplier))
    out["KeltnerLower"]=lambda high,low,close,ema_window,atr_window,multiplier:sub(ts_ema(close,ema_window),mul(f("ATR_WILDER")(high,low,close,atr_window),multiplier))
    out["KeltnerPosition"]=lambda high,low,close,ema_window,atr_window,multiplier:div(sub(close,out["KeltnerLower"](high,low,close,ema_window,atr_window,multiplier)),sub(out["KeltnerUpper"](high,low,close,ema_window,atr_window,multiplier),out["KeltnerLower"](high,low,close,ema_window,atr_window,multiplier)))

    out["DEMA"]=lambda x,window:sub(mul(ts_ema(x,window),2.0),ts_ema(ts_ema(x,window),window))
    out["TEMA"]=lambda x,window:add(sub(mul(ts_ema(x,window),3.0),mul(ts_ema(ts_ema(x,window),window),3.0)),ts_ema(ts_ema(ts_ema(x,window),window),window))

    out["ichimoku_tenkan"]=lambda high,low,tenkan_window:mul(add(ts_max(high,tenkan_window),ts_min(low,tenkan_window)),0.5)
    out["ichimoku_kijun"]=lambda high,low,kijun_window:mul(add(ts_max(high,kijun_window),ts_min(low,kijun_window)),0.5)
    out["ichimoku_senkou_a"]=lambda high,low,tenkan_window,kijun_window:mul(add(out["ichimoku_tenkan"](high,low,tenkan_window),out["ichimoku_kijun"](high,low,kijun_window)),0.5)
    out["ichimoku_senkou_b"]=lambda high,low,senkou_b_window:mul(add(ts_max(high,senkou_b_window),ts_min(low,senkou_b_window)),0.5)
    out["ichimoku_cloud_width"]=lambda high,low,tenkan_window,kijun_window,senkou_b_window:f("abs")(sub(out["ichimoku_senkou_a"](high,low,tenkan_window,kijun_window),out["ichimoku_senkou_b"](high,low,senkou_b_window)))
    out["ichimoku_cloud_position"]=lambda high,low,close,tenkan_window,kijun_window,senkou_b_window:div(sub(close,minimum(out["ichimoku_senkou_a"](high,low,tenkan_window,kijun_window),out["ichimoku_senkou_b"](high,low,senkou_b_window))),sub(maximum(out["ichimoku_senkou_a"](high,low,tenkan_window,kijun_window),out["ichimoku_senkou_b"](high,low,senkou_b_window)),minimum(out["ichimoku_senkou_a"](high,low,tenkan_window,kijun_window),out["ichimoku_senkou_b"](high,low,senkou_b_window))))
    return out
