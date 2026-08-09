# -*- coding: utf-8 -*-
"""Parameterized production-oriented technical indicators.

No conventional period is hard coded into the public contract: every lookback,
smoothing period and multiplier is explicit.  Recursive indicators are tagged
``stateful`` and require full replay/checkpoint governance in production.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator

_EPS=1e-12

def _pi(v,name,minimum=1):
    if isinstance(v,bool): raise ValueError(f"{name} must be integer")
    v=int(v)
    if v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v

def _pf(v,name,minimum=None):
    v=float(v)
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    if minimum is not None and v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v

def _ema(x,span): return x.ewm(span=_pi(span,"span"),adjust=False,min_periods=_pi(span,"span")).mean()
def _wilder(x,window): return x.ewm(alpha=1.0/_pi(window,"window"),adjust=False,min_periods=_pi(window,"window")).mean()
def _tr(high,low,close):
    prev=close.shift(1)
    arr=np.maximum.reduce([(high-low).to_numpy(float),(high-prev).abs().to_numpy(float),(low-prev).abs().to_numpy(float)])
    return pd.DataFrame(arr,index=high.index,columns=high.columns)

def _register(name,params,fn,desc,*,tags=(),param_specs=None,relational_specs=None,expected_old_source=""):
    meta=OperatorMetadata(name=name,category="technical_signal",description=desc,param_names=list(params),return_type="series",
        tags=["pit_safe","causal","production_extension",*tags], param_specs=dict(param_specs or {}),
        relational_specs=list(relational_specs or []))
    def _calculate_series(self,*args,**kwargs): return fn(*args,**kwargs)
    cls=type(f"TechnicalV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="technical_signal",business_category="technical",canonical=name,
        source="technical_indicators_v2",backend="pandas_numpy",status="production",
        replace=bool(expected_old_source),
        replacement_reason=("overrides composite fastpath layer (round-7 P0 chain pinning)" if expected_old_source else ""),
        expected_old_source=expected_old_source)(cls)


def _dmi(high,low,close,window):
    w=_pi(window,"window",2); up=high.diff(); down=-low.diff()
    # A missing bar previously collapsed to ``0.0`` ("valid zero directional
    # movement") and leaked into the Wilder smoothing; it must stay NaN
    # ("cannot judge") — review batch-2 missing semantics.
    valid=up.notna() & down.notna()
    plus=up.where((up>down)&(up>0),0.0).where(valid)
    minus=down.where((down>up)&(down>0),0.0).where(valid)
    atr=_wilder(_tr(high,low,close),w)
    p=100.0*_wilder(plus,w)/atr.replace(0,np.nan); m=100.0*_wilder(minus,w)/atr.replace(0,np.nan)
    return p,m

def DMI_plus(high,low,close,window): return _dmi(high,low,close,window)[0]
def DMI_minus(high,low,close,window): return _dmi(high,low,close,window)[1]
def DX(high,low,close,window):
    p,m=_dmi(high,low,close,window); return 100.0*(p-m).abs()/(p+m).replace(0,np.nan)
def NATR(high,low,close,window):
    # R5-38: close is a PositivePrice input — a non-positive close is bad data,
    # not something to be laundered by ``abs()`` (a negative price would have
    # produced a negative/meaningless "normalized" ATR).  Keep the numerator,
    # mask the normalization to strict-positive so bad bars become NaN.
    return 100.0*_wilder(_tr(high,low,close),window)/close.where(close>0.0)

def PPO(close,fast_window,slow_window):
    f=_pi(fast_window,"fast_window"); s=_pi(slow_window,"slow_window")
    if f>=s: raise ValueError("fast_window must be < slow_window")
    ef,es=_ema(close,f),_ema(close,s); return 100.0*(ef-es)/es.replace(0,np.nan)
def PPO_signal(close,fast_window,slow_window,signal_window): return _ema(PPO(close,fast_window,slow_window),signal_window)
def PPO_hist(close,fast_window,slow_window,signal_window): return PPO(close,fast_window,slow_window)-PPO_signal(close,fast_window,slow_window,signal_window)

def PVO(volume,fast_window,slow_window): return PPO(volume,fast_window,slow_window)
def PVO_signal(volume,fast_window,slow_window,signal_window): return _ema(PVO(volume,fast_window,slow_window),signal_window)
def PVO_hist(volume,fast_window,slow_window,signal_window): return PVO(volume,fast_window,slow_window)-PVO_signal(volume,fast_window,slow_window,signal_window)

def CMO(close,window):
    w=_pi(window,"window",2); d=close.diff(); up=d.clip(lower=0).rolling(w,min_periods=w).sum(); dn=(-d.clip(upper=0)).rolling(w,min_periods=w).sum()
    return 100.0*(up-dn)/(up+dn).replace(0,np.nan)

def VortexPlus(high,low,close,window):
    w=_pi(window,"window",2); vm=(high-low.shift(1)).abs().rolling(w,min_periods=w).sum(); tr=_tr(high,low,close).rolling(w,min_periods=w).sum(); return vm/tr.replace(0,np.nan)
def VortexMinus(high,low,close,window):
    w=_pi(window,"window",2); vm=(low-high.shift(1)).abs().rolling(w,min_periods=w).sum(); tr=_tr(high,low,close).rolling(w,min_periods=w).sum(); return vm/tr.replace(0,np.nan)

def _keltner_multiplier(v):
    # R5-36: multiplier=0 degenerates KeltnerUpper/Lower into KeltnerMid and
    # makes KeltnerPosition divide by zero — a false search node that renames
    # the band to a different canonical.  The multiplier is a strict positive.
    x=_pf(v,"multiplier",0.0)
    if x<=0: raise ValueError("multiplier must be > 0 (Keltner multiplier=0 degenerates to the mid band)")
    return x
def KeltnerMid(close,ema_window): return _ema(close,ema_window)
def KeltnerUpper(high,low,close,ema_window,atr_window,multiplier): return KeltnerMid(close,ema_window)+_keltner_multiplier(multiplier)*_wilder(_tr(high,low,close),atr_window)
def KeltnerLower(high,low,close,ema_window,atr_window,multiplier): return KeltnerMid(close,ema_window)-_keltner_multiplier(multiplier)*_wilder(_tr(high,low,close),atr_window)
def KeltnerPosition(high,low,close,ema_window,atr_window,multiplier):
    u=KeltnerUpper(high,low,close,ema_window,atr_window,multiplier); l=KeltnerLower(high,low,close,ema_window,atr_window,multiplier)
    return (close-l)/(u-l).replace(0,np.nan)

def TSI(close,long_window,short_window):
    lw=_pi(long_window,"long_window",2); sw=_pi(short_window,"short_window",2); m=close.diff(); num=_ema(_ema(m,lw),sw); den=_ema(_ema(m.abs(),lw),sw)
    return 100.0*num/den.replace(0,np.nan)
def TSI_signal(close,long_window,short_window,signal_window): return _ema(TSI(close,long_window,short_window),signal_window)

def UltimateOscillator(high,low,close,short_window,medium_window,long_window,short_weight=4.0,medium_weight=2.0,long_weight=1.0):
    s,m,l=_pi(short_window,"short_window",2),_pi(medium_window,"medium_window",2),_pi(long_window,"long_window",2)
    if not (s<m<l): raise ValueError("require short_window < medium_window < long_window")
    pc=close.shift(1); minl=pd.DataFrame(np.minimum(low.to_numpy(float),pc.to_numpy(float)),index=low.index,columns=low.columns); maxh=pd.DataFrame(np.maximum(high.to_numpy(float),pc.to_numpy(float)),index=high.index,columns=high.columns)
    bp=close-minl; tr=maxh-minl
    def avg(w): return bp.rolling(w,min_periods=w).sum()/tr.rolling(w,min_periods=w).sum().replace(0,np.nan)
    ws,wm,wl=_pf(short_weight,"short_weight",0),_pf(medium_weight,"medium_weight",0),_pf(long_weight,"long_weight",0)
    if wl<=0: raise ValueError("long_weight must be > 0")
    # R5-35: the formula divides by (ws+wm+wl), so (4,2,1) == (40,20,10) — the
    # three weights are ONE ratio dimension (2 free parameters).  Canonicalize
    # to ``w3 = 1`` so the effective search space collapses to the two ratios
    # ws/wl and wm/wl instead of a false 3-D hypercube of duplicate nodes.
    ws, wm, wl = ws/wl, wm/wl, 1.0
    den=ws+wm+wl
    if den<=0: raise ValueError("oscillator weights must sum to > 0")
    return 100.0*(ws*avg(s)+wm*avg(m)+wl*avg(l))/den

def DEMA(x,window):
    e1=_ema(x,window); e2=_ema(e1,window); return 2.0*e1-e2
def TEMA(x,window):
    e1=_ema(x,window); e2=_ema(e1,window); e3=_ema(e2,window); return 3.0*e1-3.0*e2+e3

def ichimoku_tenkan(high,low,tenkan_window):
    w=_pi(tenkan_window,"tenkan_window",2); return (high.rolling(w,min_periods=w).max()+low.rolling(w,min_periods=w).min())/2.0
def ichimoku_kijun(high,low,kijun_window):
    w=_pi(kijun_window,"kijun_window",2); return (high.rolling(w,min_periods=w).max()+low.rolling(w,min_periods=w).min())/2.0
def ichimoku_senkou_a(high,low,tenkan_window,kijun_window): return (ichimoku_tenkan(high,low,tenkan_window)+ichimoku_kijun(high,low,kijun_window))/2.0
def ichimoku_senkou_b(high,low,senkou_b_window):
    w=_pi(senkou_b_window,"senkou_b_window",2); return (high.rolling(w,min_periods=w).max()+low.rolling(w,min_periods=w).min())/2.0
def ichimoku_cloud_width(high,low,tenkan_window,kijun_window,senkou_b_window): return (ichimoku_senkou_a(high,low,tenkan_window,kijun_window)-ichimoku_senkou_b(high,low,senkou_b_window)).abs()
def ichimoku_cloud_position(high,low,close,tenkan_window,kijun_window,senkou_b_window):
    a=ichimoku_senkou_a(high,low,tenkan_window,kijun_window); b=ichimoku_senkou_b(high,low,senkou_b_window); lo=pd.DataFrame(np.minimum(a,b),index=a.index,columns=a.columns); hi=pd.DataFrame(np.maximum(a,b),index=a.index,columns=a.columns)
    return (close-lo)/(hi-lo).replace(0,np.nan)

def KAMA(close,er_window,fast_window,slow_window):
    er=_pi(er_window,"er_window",2); fast=_pi(fast_window,"fast_window"); slow=_pi(slow_window,"slow_window")
    if fast>=slow: raise ValueError("fast_window must be < slow_window")
    change=(close-close.shift(er)).abs(); vol=close.diff().abs().rolling(er,min_periods=er).sum(); efficiency=change/vol.replace(0,np.nan)
    fast_sc=2.0/(fast+1.0); slow_sc=2.0/(slow+1.0); sc=(efficiency*(fast_sc-slow_sc)+slow_sc)**2
    arr=close.to_numpy(float); alpha=sc.to_numpy(float); out=np.full_like(arr,np.nan,float)
    for c in range(arr.shape[1]):
        # break + rewarm (single production missing-state policy): a NaN price
        # INVALIDATES the recursive state.  KAMA then emits NaN until
        # ``er_window`` consecutive finite prices re-accumulate and the ER is
        # re-derived over that contiguous history — no frozen flat line between
        # the re-seed and the ER window re-filling (round-11 P0).
        last=np.nan
        contiguous=0
        for t in range(arr.shape[0]):
            if not np.isfinite(arr[t,c]):
                last=np.nan
                contiguous=0
                continue
            contiguous+=1
            if contiguous<er:
                # Not enough contiguous history to re-derive the ER yet.
                continue
            if not np.isfinite(last): last=arr[t,c]
            elif np.isfinite(alpha[t,c]): last=last+alpha[t,c]*(arr[t,c]-last)
            out[t,c]=last
    return pd.DataFrame(out,index=close.index,columns=close.columns)

def Supertrend(high,low,close,atr_window,multiplier):
    w=_pi(atr_window,"atr_window",2); mult=_pf(multiplier,"multiplier",0)
    if mult<=0: raise ValueError("multiplier must be > 0 (Supertrend multiplier=0 collapses upper/lower to the midpoint)")
    atr=_wilder(_tr(high,low,close),w); mid=(high+low)/2.0; basic_u=mid+mult*atr; basic_l=mid-mult*atr
    rows,cols=close.shape; out=np.full((rows,cols),np.nan)
    hh=high.to_numpy(float); ll=low.to_numpy(float); cu=basic_u.to_numpy(float); cl=basic_l.to_numpy(float); cv=close.to_numpy(float)
    final_u=cu.copy(); final_l=cl.copy(); trend=np.ones((rows,cols),dtype=int)
    for c in range(cols):
        for t in range(1,rows):
            # break + rewarm: a bar is state-valid only when high/low/close AND
            # the derived ATR-based bands are ALL finite.  A single non-finite
            # component (e.g. high/NaN + low/NaN with a finite close) must NOT
            # keep the stale trend alive — it invalidates the state and re-warms
            # over the next contiguous valid segment (round-11 P0).
            if not (np.isfinite(hh[t,c]) and np.isfinite(ll[t,c]) and np.isfinite(cv[t,c])
                    and np.isfinite(cu[t,c]) and np.isfinite(cl[t,c])):
                trend[t,c] = 0  # sentinel: state broken, next valid bar re-seeds
                continue
            if trend[t-1,c] == 0:
                # Re-seed after a gap from the current bar's own valid bands.
                trend[t,c] = 1
                final_u[t,c] = cu[t,c]
                final_l[t,c] = cl[t,c]
                out[t,c] = final_l[t,c]
                continue
            if np.isfinite(final_u[t-1,c]) and (cu[t,c]>=final_u[t-1,c] and cv[t-1,c]<=final_u[t-1,c]): final_u[t,c]=final_u[t-1,c]
            if np.isfinite(final_l[t-1,c]) and (cl[t,c]<=final_l[t-1,c] and cv[t-1,c]>=final_l[t-1,c]): final_l[t,c]=final_l[t-1,c]
            if trend[t-1,c]>0 and cv[t,c]<final_l[t,c]: trend[t,c]=-1
            elif trend[t-1,c]<0 and cv[t,c]>final_u[t,c]: trend[t,c]=1
            else: trend[t,c]=trend[t-1,c]
            out[t,c]=final_l[t,c] if trend[t,c]>0 else final_u[t,c]
    return pd.DataFrame(out,index=close.index,columns=close.columns)
def SupertrendDirection(high,low,close,atr_window,multiplier):
    st=Supertrend(high,low,close,atr_window,multiplier); return pd.DataFrame(np.where(close>=st,1.0,-1.0),index=close.index,columns=close.columns).where(st.notna())

def PSAR(high,low,acceleration,maximum):
    af0=_pf(acceleration,"acceleration",0); afmax=_pf(maximum,"maximum",0)
    if af0<=0 or afmax<af0: raise ValueError("require 0 < acceleration <= maximum")
    h,l=high.to_numpy(float),low.to_numpy(float); rows,cols=h.shape; out=np.full((rows,cols),np.nan)
    for c in range(cols):
        # R5-37: seed from the FIRST jointly-valid (h,l) bar, never from row 0
        # (a missing first row would poison ``sar``/``ep`` and every following
        # state).  The t-1/t-2 references use a *valid-bar* history — after a
        # gap / re-seed the history is reset, so SAR never reads raw rows from
        # inside a suspension (half-carry / half-raw-index mixing).
        # break + rewarm (single production missing-state policy): a missing bar
        # invalidates the state; the next jointly-valid bar re-seeds.
        bull=True; sar=None; ep=None; af=af0; live=False; seeded_once=False
        prev1_h=prev1_l=prev2_h=prev2_l=None
        for t in range(rows):
            hv, lv = h[t,c], l[t,c]
            if not (np.isfinite(hv) and np.isfinite(lv)):
                live = False
                continue
            if not live:
                # Seed / re-seed from this jointly-valid bar and reset the
                # valid-bar history (a gap breaks the sequence; the pre-gap
                # bars must not resurface as t-1/t-2 references).  The FIRST
                # seed does not emit (matches the legacy row-0=NaN contract);
                # a mid-series re-seed emits like the legacy interrupt path.
                bull=True; sar=lv; ep=hv; af=af0; live=True
                prev1_h, prev1_l = hv, lv
                prev2_h, prev2_l = None, None
                if seeded_once:
                    out[t,c]=sar
                seeded_once = True
                continue
            sar=sar+af*(ep-sar)
            if bull:
                if prev2_l is not None: sar=min(sar,prev1_l,prev2_l)
                else: sar=min(sar,prev1_l)
                if lv<sar: bull=False; sar=ep; ep=lv; af=af0
                elif hv>ep: ep=hv; af=min(af+af0,afmax)
            else:
                if prev2_h is not None: sar=max(sar,prev1_h,prev2_h)
                else: sar=max(sar,prev1_h)
                if hv>sar: bull=True; sar=ep; ep=hv; af=af0
                elif lv<ep: ep=lv; af=min(af+af0,afmax)
            prev2_h, prev2_l = prev1_h, prev1_l
            prev1_h, prev1_l = hv, lv
            out[t,c]=sar
    return pd.DataFrame(out,index=high.index,columns=high.columns)

_SPECS=[
("DMI_plus",["high","low","close","window"],DMI_plus,"Wilder positive directional indicator."),
("DMI_minus",["high","low","close","window"],DMI_minus,"Wilder negative directional indicator."),
("DX",["high","low","close","window"],DX,"Directional movement index before ADX smoothing."),
("NATR",["high","low","close","window"],NATR,"Normalized Wilder ATR as percent of close."),
("PPO",["close","fast_window","slow_window"],PPO,"Percentage price oscillator."),
("PPO_signal",["close","fast_window","slow_window","signal_window"],PPO_signal,"PPO signal line."),
("PPO_hist",["close","fast_window","slow_window","signal_window"],PPO_hist,"PPO histogram."),
("PVO",["volume","fast_window","slow_window"],PVO,"Percentage volume oscillator."),
("PVO_signal",["volume","fast_window","slow_window","signal_window"],PVO_signal,"PVO signal line."),
("PVO_hist",["volume","fast_window","slow_window","signal_window"],PVO_hist,"PVO histogram."),
("CMO",["close","window"],CMO,"Chande momentum oscillator."),
("VortexPlus",["high","low","close","window"],VortexPlus,"Positive Vortex indicator."),
("VortexMinus",["high","low","close","window"],VortexMinus,"Negative Vortex indicator."),
("KeltnerMid",["close","ema_window"],KeltnerMid,"Keltner EMA midline."),
("KeltnerUpper",["high","low","close","ema_window","atr_window","multiplier"],KeltnerUpper,"Keltner upper envelope."),
("KeltnerLower",["high","low","close","ema_window","atr_window","multiplier"],KeltnerLower,"Keltner lower envelope."),
("KeltnerPosition",["high","low","close","ema_window","atr_window","multiplier"],KeltnerPosition,"Close position inside Keltner channel."),
("TSI",["close","long_window","short_window"],TSI,"True Strength Index with explicit smoothing windows."),
("TSI_signal",["close","long_window","short_window","signal_window"],TSI_signal,"Signal line of TSI."),
("UltimateOscillator",["high","low","close","short_window","medium_window","long_window","short_weight","medium_weight","long_weight"],UltimateOscillator,"Ultimate oscillator with explicit windows and weights."),
("DEMA",["x","window"],DEMA,"Double exponential moving average."),
("TEMA",["x","window"],TEMA,"Triple exponential moving average."),
("ichimoku_tenkan",["high","low","tenkan_window"],ichimoku_tenkan,"Raw Tenkan value available at current timestamp."),
("ichimoku_kijun",["high","low","kijun_window"],ichimoku_kijun,"Raw Kijun value available at current timestamp."),
("ichimoku_senkou_a",["high","low","tenkan_window","kijun_window"],ichimoku_senkou_a,"Raw Senkou A without chart-forward timestamp shifting."),
("ichimoku_senkou_b",["high","low","senkou_b_window"],ichimoku_senkou_b,"Raw Senkou B without chart-forward timestamp shifting."),
("ichimoku_cloud_width",["high","low","tenkan_window","kijun_window","senkou_b_window"],ichimoku_cloud_width,"Current raw Ichimoku cloud width."),
("ichimoku_cloud_position",["high","low","close","tenkan_window","kijun_window","senkou_b_window"],ichimoku_cloud_position,"Close position inside raw Ichimoku cloud."),
("KAMA",["close","er_window","fast_window","slow_window"],KAMA,"Kaufman adaptive moving average; recursive/stateful."),
("Supertrend",["high","low","close","atr_window","multiplier"],Supertrend,"Recursive Supertrend level."),
("SupertrendDirection",["high","low","close","atr_window","multiplier"],SupertrendDirection,"Recursive Supertrend direction (+1/-1)."),
("PSAR",["high","low","acceleration","maximum"],PSAR,"Parabolic SAR; recursive/stateful."),
]
# Recursive EMA / Wilder families: every output depends on the full history
# through ``ewm(adjust=False)`` (``_ema``/``_wilder``), so a
# ``start_date + finite warmup`` run is NOT bit-exact with a full-history run.
# They must be governed ``stateful`` + ``full_replay`` exactly like
# KAMA/Supertrend/PSAR (review P0-05).  Bounded rolling-sum indicators (CMO,
# Vortex, UltimateOscillator, ichimoku) stay unbounded-state-free.
_RECURSIVE_EWM = {
    "DMI_plus", "DMI_minus", "DX", "NATR",
    "PPO", "PPO_signal", "PPO_hist", "PVO", "PVO_signal", "PVO_hist",
    "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition",
    "TSI", "TSI_signal", "DEMA", "TEMA",
}
# R5-35/36 + round-11 P0 (search-space contracts): EVERY parameter of the
# technical indicators below gets a formal ``ParamSpec`` — never a silent
# ``_pi`` float truncation.  A window declared ``dtype=int`` rejects 20.1/20.5/
# 20.9 at the call boundary (they all used to truncate to the SAME 20, a fake
# search space); a multiplier/acceleration is a strict positive float.
_WIN_GE2 = ParamSpec(dtype=int, min=2)   # windows whose kernel uses _pi(w, name, 2)
_WIN_GE1 = ParamSpec(dtype=int, min=1)   # windows whose kernel uses _pi(w, name)
_POS_FLOAT = ParamSpec(dtype=float, min=1e-9)  # strict positive (finite) float


def _rel(expression, message):
    return RelationalParamSpec(expression=expression, message=message)


_PARAM_SPECS = {
    "DMI_plus": {"window": _WIN_GE2},
    "DMI_minus": {"window": _WIN_GE2},
    "DX": {"window": _WIN_GE2},
    "NATR": {"window": _WIN_GE2},
    "PPO": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
    "PPO_signal": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1, "signal_window": _WIN_GE1},
    "PPO_hist": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1, "signal_window": _WIN_GE1},
    "PVO": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
    "PVO_signal": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1, "signal_window": _WIN_GE1},
    "PVO_hist": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1, "signal_window": _WIN_GE1},
    "CMO": {"window": _WIN_GE2},
    "VortexPlus": {"window": _WIN_GE2},
    "VortexMinus": {"window": _WIN_GE2},
    "KeltnerMid": {"ema_window": _WIN_GE1},
    "KeltnerUpper": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT},
    "KeltnerLower": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT},
    "KeltnerPosition": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT},
    "TSI": {"long_window": _WIN_GE2, "short_window": _WIN_GE2},
    "TSI_signal": {"long_window": _WIN_GE2, "short_window": _WIN_GE2, "signal_window": _WIN_GE1},
    "UltimateOscillator": {
        "short_window": _WIN_GE2,
        "medium_window": _WIN_GE2,
        "long_window": _WIN_GE2,
        # R5-35: the weight triple is ONE ratio dimension (kernel normalizes to
        # ``w3=1``), so the individual weights must not be independently mined.
        "short_weight": ParamSpec(dtype=float, searchable=False),
        "medium_weight": ParamSpec(dtype=float, searchable=False),
        "long_weight": ParamSpec(dtype=float, searchable=False),
    },
    "DEMA": {"window": _WIN_GE1},
    "TEMA": {"window": _WIN_GE1},
    "ichimoku_tenkan": {"tenkan_window": _WIN_GE2},
    "ichimoku_kijun": {"kijun_window": _WIN_GE2},
    "ichimoku_senkou_a": {"tenkan_window": _WIN_GE2, "kijun_window": _WIN_GE2},
    "ichimoku_senkou_b": {"senkou_b_window": _WIN_GE2},
    "ichimoku_cloud_width": {"tenkan_window": _WIN_GE2, "kijun_window": _WIN_GE2, "senkou_b_window": _WIN_GE2},
    "ichimoku_cloud_position": {"tenkan_window": _WIN_GE2, "kijun_window": _WIN_GE2, "senkou_b_window": _WIN_GE2},
    "KAMA": {"er_window": _WIN_GE2, "fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
    "Supertrend": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "SupertrendDirection": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "PSAR": {"acceleration": _POS_FLOAT, "maximum": _POS_FLOAT},
}

# R6-24 relational feasibility constraints (round-11 P0): TSI's dual-EMA window
# swap (long < short) creates near-duplicate search nodes and is rejected before
# budget is spent; PPO/PVO/KAMA fast/slow and UltimateOscillator's window chain
# are declared the same way.  PSAR requires 0 < acceleration <= maximum.
_PPO_REL = [
    _rel("fast_window < slow_window", "fast_window must be < slow_window (fast={fast_window}, slow={slow_window})")
]
_TSI_REL = [
    _rel("long_window > short_window", "long_window must be > short_window (long={long_window}, short={short_window})")
]
_UO_REL = [
    _rel("short_window < medium_window", "require short_window < medium_window (short={short_window}, medium={medium_window})"),
    _rel("medium_window < long_window", "require medium_window < long_window (medium={medium_window}, long={long_window})"),
]
_PSAR_REL = [
    _rel("acceleration <= maximum", "require 0 < acceleration <= maximum (acceleration={acceleration}, maximum={maximum})")
]
_RELATIONAL_SPECS = {
    "PPO": _PPO_REL,
    "PPO_signal": _PPO_REL,
    "PPO_hist": _PPO_REL,
    "PVO": _PPO_REL,
    "PVO_signal": _PPO_REL,
    "PVO_hist": _PPO_REL,
    "KAMA": _PPO_REL,
    "TSI": _TSI_REL,
    "TSI_signal": _TSI_REL,
    "UltimateOscillator": _UO_REL,
    "PSAR": _PSAR_REL,
}
for _name,_params,_fn,_desc in _SPECS:
    _tags=("stateful","full_replay") if _name in _RECURSIVE_EWM or _name in {"KAMA","Supertrend","SupertrendDirection","PSAR"} else ()
    # KAMA overrides the composite_fastpath_primitives implementation; pin the
    # exact source it replaces so the chain is independent of import order
    # (round-7 P0).  Other spec entries register fresh — no pin needed.
    _pin = "composite_fastpath_primitives" if _name == "KAMA" else ""
    _register(_name,_params,_fn,_desc,tags=_tags,param_specs=_PARAM_SPECS.get(_name),
              relational_specs=_RELATIONAL_SPECS.get(_name),expected_old_source=_pin)
