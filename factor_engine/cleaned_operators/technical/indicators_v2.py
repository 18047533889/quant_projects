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
from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator

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

def atr_pct(high,low,close,window):
    """Wilder ATR as a fraction of close (dimensionless; NATR / 100)."""
    w=_pi(window,"window",2)
    atr=_wilder(_tr(high,low,close),w)
    return atr/close.where(close>0.0)

def atr_zscore(high,low,close,window,score_window):
    """Rolling z-score of ATR / close; both windows use strictly trailing data."""
    ratio=atr_pct(high,low,close,window)
    w=_pi(score_window,"score_window",2)
    mean=ratio.rolling(w,min_periods=w).mean(); std=ratio.rolling(w,min_periods=w).std(ddof=1)
    return (ratio-mean)/std.replace(0,np.nan)

def atr_percentile(high,low,close,window,score_window):
    """Trailing-window percentile rank (0..1) of ATR / close."""
    ratio=atr_pct(high,low,close,window)
    w=_pi(score_window,"score_window",2)
    rank=ratio.rolling(w,min_periods=w).rank(pct=True)
    # pandas percentile rank includes the current observation by definition of
    # the trailing window; it is PIT-safe because the window ends at t.
    return rank

def true_range_pct(high,low,close):
    """Raw true range as a fraction of the previous close."""
    return _tr(high,low,close)/close.shift(1).where(close.shift(1)>0.0)

def true_range_surprise(high,low,close,window):
    """Current true-range ratio vs its own trailing mean, minus 1."""
    ratio=true_range_pct(high,low,close)
    w=_pi(window,"window",2)
    return ratio/ratio.rolling(w,min_periods=w).mean().replace(0,np.nan)-1.0

def true_range_zscore(high,low,close,window):
    """Trailing z-score of the true-range ratio."""
    ratio=true_range_pct(high,low,close)
    w=_pi(window,"window",2)
    mean=ratio.rolling(w,min_periods=w).mean(); std=ratio.rolling(w,min_periods=w).std(ddof=1)
    return (ratio-mean)/std.replace(0,np.nan)

def atr_short_long_ratio(high,low,close,short_window,long_window):
    """Short-window ATR ratio divided by long-window ATR ratio (dimensionless)."""
    s=_pi(short_window,"short_window",2); l=_pi(long_window,"long_window",2)
    if s>=l: raise ValueError("short_window must be < long_window")
    return atr_pct(high,low,close,s)/atr_pct(high,low,close,l).replace(0,np.nan)

def atr_acceleration(high,low,close,window):
    """First difference of ATR / close."""
    return atr_pct(high,low,close,window).diff()

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

# R20: Keltner DirectUse rehabilitation — raw KeltnerMid/Upper/Lower levels stay
# INTERMEDIATE (price-scale); these are the causal, dimensionless canonicals.
def keltner_width_pct(high,low,close,ema_window,atr_window,multiplier):
    """Keltner band width (Upper-Lower) as a fraction of the mid line."""
    u=KeltnerUpper(high,low,close,ema_window,atr_window,multiplier); l=KeltnerLower(high,low,close,ema_window,atr_window,multiplier)
    m=KeltnerMid(close,ema_window)
    return (u-l)/m.where(m>0.0)
def keltner_compression(high,low,close,ema_window,atr_window,multiplier,score_window):
    """Trailing-window percentile rank (0..1) of the Keltner width ratio; low = compressed."""
    width=keltner_width_pct(high,low,close,ema_window,atr_window,multiplier)
    w=_pi(score_window,"score_window",2)
    # Trailing rank ends at t (PIT-safe): the current bar is its own reference.
    return width.rolling(w,min_periods=w).rank(pct=True)
def keltner_breakout_strength(high,low,close,ema_window,atr_window,multiplier):
    """Signed breakout distance from the band, normalized by band width."""
    u=KeltnerUpper(high,low,close,ema_window,atr_window,multiplier); l=KeltnerLower(high,low,close,ema_window,atr_window,multiplier)
    span=(u-l).replace(0,np.nan)
    # Above the upper band -> positive; below the lower band -> negative;
    # inside -> 0.  A NaN band makes the comparison NaN (cannot judge), so the
    # bar stays NaN instead of collapsing to a false 0.
    above=(close-u).where(close>u,0.0).where(close.notna()&u.notna(),np.nan)
    below=(close-l).where(close<l,0.0).where(close.notna()&l.notna(),np.nan)
    return (above+below)/span

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
            # R30 §21 (P0-016): the ER needs ``er_window + 1`` contiguous finite
            # prices — ``close - close.shift(er)`` pairs bar ``t`` with bar
            # ``t-er``, and the ``diff().rolling(er).sum()`` volatility needs
            # ``er`` finite diffs (which exist only once ``er+1`` prices are
            # present).  Seeding at exactly ``er`` points emitted a KAMA reading
            # whose ER / alpha were not yet fully defined.
            if contiguous < er + 1:
                # Not enough contiguous history to fully re-derive the ER yet
                # (need er+1 prices).  No output until the ER is well-defined.
                continue
            if not np.isfinite(last):
                # First legal KAMA seed after a warmup / gap: the CURRENT close,
                # the same rule first-time and after every break.
                last = arr[t, c]
            elif np.isfinite(alpha[t, c]):
                last = last + alpha[t, c] * (arr[t, c] - last)
            out[t, c] = last
    return pd.DataFrame(out, index=close.index, columns=close.columns)

def Supertrend(high,low,close,atr_window,multiplier):
    w=_pi(atr_window,"atr_window",2); mult=_pf(multiplier,"multiplier",0)
    if mult<=0: raise ValueError("multiplier must be > 0 (Supertrend multiplier=0 collapses upper/lower to the midpoint)")
    atr=_wilder(_tr(high,low,close),w); mid=(high+low)/2.0; basic_u=mid+mult*atr; basic_l=mid-mult*atr
    rows,cols=close.shape; out=np.full((rows,cols),np.nan)
    hh=high.to_numpy(float); ll=low.to_numpy(float); cu=basic_u.to_numpy(float); cl=basic_l.to_numpy(float); cv=close.to_numpy(float)
    final_u=cu.copy(); final_l=cl.copy(); trend=np.ones((rows,cols),dtype=int)
    for c in range(cols):
        # R30 §22: ``post_gap`` distinguishes "just broke after a gap" from "still
        # warming up / still waiting for direction".  The FIRST valid bar after a
        # break enters UNKNOWN (trend=0, post_gap=True); the NEXT valid bar
        # re-asserts direction from the price/band relationship — it must NOT
        # re-enter UNKNOWN (that would deadlock on ``trend[t-1]==0`` forever).
        post_gap = False
        for t in range(1,rows):
            # break + rewarm: a bar is state-valid only when high/low/close AND
            # the derived ATR-based bands are ALL finite.  A single non-finite
            # component (e.g. high/NaN + low/NaN with a finite close) must NOT
            # keep the stale trend alive — it invalidates the state and re-warms
            # over the next contiguous valid segment (round-11 P0).
            if not (np.isfinite(hh[t,c]) and np.isfinite(ll[t,c]) and np.isfinite(cv[t,c])
                    and np.isfinite(cu[t,c]) and np.isfinite(cl[t,c])):
                trend[t,c] = 0  # sentinel: state broken, next valid bar re-seeds
                post_gap = True
                continue
            if post_gap:
                # R30 §22 (P0-017): re-seed after a gap enters UNKNOWN, NOT a
                # hardcoded bullish restart.  A gap destroys the trend context, so
                # the next valid bar alone cannot assert a direction — emitting the
                # lower band here would manufacture a systematic long bias out of a
                # data hole.  This first valid bar publishes UNKNOWN and clears the
                # flag; the following valid bar re-asserts direction below.
                trend[t,c] = 0
                final_u[t,c] = cu[t,c]
                final_l[t,c] = cl[t,c]
                post_gap = False
                out[t,c] = np.nan  # UNKNOWN after a gap — never a fake direction
                continue
            if np.isfinite(final_u[t-1,c]) and (cu[t,c]>=final_u[t-1,c] and cv[t-1,c]<=final_u[t-1,c]): final_u[t,c]=final_u[t-1,c]
            if np.isfinite(final_l[t-1,c]) and (cl[t,c]<=final_l[t-1,c] and cv[t-1,c]>=final_l[t-1,c]): final_l[t,c]=final_l[t-1,c]
            if trend[t-1,c] == 0:
                # R30 §22: after a gap / warmup the trend is UNKNOWN.  The next
                # valid bar re-asserts direction from its own position RELATIVE TO
                # THE BAND MIDLINE — the standard Supertrend initialisation that
                # does not pre-judge direction.  Closing above the midpoint starts
                # a long bias, below starts a short bias; neither is manufactured
                # by the gap itself.
                mid_t = 0.5 * (cu[t,c] + cl[t,c])
                if cv[t,c] >= mid_t:
                    trend[t,c] = 1
                else:
                    trend[t,c] = -1
            elif trend[t-1,c] > 0 and cv[t,c] < final_l[t,c]:
                trend[t,c] = -1
            elif trend[t-1,c] < 0 and cv[t,c] > final_u[t,c]:
                trend[t,c] = 1
            else:
                trend[t,c] = trend[t-1,c]
            if trend[t,c] > 0:
                out[t,c] = final_l[t,c]
            elif trend[t,c] < 0:
                out[t,c] = final_u[t,c]
            else:
                out[t,c] = np.nan
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
                # R30 §23 (P0-018): seed / re-seed does NOT hardcode a bullish
                # restart.  A data gap destroys the trend context; forcing
                # ``bull=True; sar=low`` after every gap manufactures a long bias
                # out of missing data.  After a gap the state is UNKNOWN: no SAR is
                # emitted until TWO consecutive valid bars re-assert a direction
                # from their own high/low movement (first-time seed also uses this
                # rule).  ``seeded_once`` distinguishes the very first seed (which
                # the legacy contract leaves NaN) from a mid-series re-seed (which
                # must not emit a direction either).
                bull=None; sar=None; ep=None; af=af0; live=True
                prev1_h, prev1_l = hv, lv
                prev2_h, prev2_l = None, None
                # Wait for a second valid bar to determine initial direction.
                seeded_once = True
                continue
            if bull is None:
                # Two valid bars now present: the direction is read from their
                # own movement — rising high/low -> uptrend (bull), falling ->
                # downtrend (bear).  No direction, no SAR.
                if hv > prev1_h and lv > prev1_l:
                    bull = True
                    sar = prev1_l
                    ep = max(hv, prev1_h)
                elif hv < prev1_h and lv < prev1_l:
                    bull = False
                    sar = prev1_h
                    ep = min(lv, prev1_l)
                else:
                    # Indeterminate two-bar move: keep waiting for direction.
                    prev2_h, prev2_l = prev1_h, prev1_l
                    prev1_h, prev1_l = hv, lv
                    continue
                prev2_h, prev2_l = prev1_h, prev1_l
                prev1_h, prev1_l = hv, lv
                out[t,c]=sar
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

def _psar_state(high,low,close,acceleration,maximum):
    """PSAR level + per-bar regime sign for the R20 DirectUse derivatives.

    ``direction`` is +1 when close is above the PSAR (bull / rising regime,
    SAR trailing under price) and -1 otherwise (bear regime, SAR over price),
    masked NaN wherever PSAR itself is NaN (warmup / re-seed).  Strictly
    trailing: both values are read from the same recursive state machine as
    :func:`PSAR` with no future displacement.
    """
    sar=PSAR(high,low,acceleration,maximum)
    c=close.to_numpy(float); s=sar.to_numpy(float)
    sign=np.full(s.shape,np.nan)
    for j in range(s.shape[1]):
        for t in range(s.shape[0]):
            if np.isfinite(s[t,j]) and np.isfinite(c[t,j]):
                sign[t,j]=1.0 if c[t,j]>s[t,j] else -1.0
    direction=pd.DataFrame(sign,index=sar.index,columns=sar.columns)
    return sar, direction

def psar_direction(high,low,close,acceleration,maximum):
    return _psar_state(high,low,close,acceleration,maximum)[1]

def psar_distance_pct(high,low,close,acceleration,maximum):
    sar=_psar_state(high,low,close,acceleration,maximum)[0]
    pos=close.where(close>0)
    return (pos-sar).div(pos)

def psar_flip(high,low,close,acceleration,maximum):
    sar,direction=_psar_state(high,low,close,acceleration,maximum)
    prev=direction.shift(1)
    flip=(direction-prev).where(direction.notna()&prev.notna())
    flip=flip.where(direction.notna()).fillna(0.0).where(direction.notna())
    return pd.DataFrame(np.sign(flip.to_numpy(float)),index=sar.index,columns=sar.columns)

def psar_days_since_flip(high,low,close,acceleration,maximum):
    sar,direction=_psar_state(high,low,close,acceleration,maximum)
    flip=psar_flip(high,low,close,acceleration,maximum).to_numpy(float)
    out=np.full(flip.shape,np.nan)
    for j in range(flip.shape[1]):
        since=np.nan
        for t in range(flip.shape[0]):
            if np.isnan(flip[t,j]):
                continue
            if flip[t,j]!=0.0:
                since=0.0
            elif since==since:
                since+=1.0
            out[t,j]=since
    return pd.DataFrame(out,index=sar.index,columns=sar.columns)

# --- Supertrend DirectUse family (R20-SUPERTREND-DIRECTUSE) ---
# Raw ``Supertrend`` level stays INTERMEDIATE (price-scale, recursive state).
# These four are the causal, dimensionless Direct Alpha canonicals, all read
# from the SAME recursive Supertrend state machine (Wilder ATR + band
# ratchet/flip) with no future displacement — the Supertrend math is never
# reimplemented here.
def _supertrend_state(high,low,close,atr_window,multiplier):
    """Supertrend level + per-bar regime sign for the R20 DirectUse derivatives.

    ``direction`` is +1 when close is at/above the Supertrend line (bull
    regime, line trailing under price) and -1 below, masked NaN wherever the
    Supertrend itself is NaN (warmup / post-gap UNKNOWN).  The sign contract
    matches the registered ``SupertrendDirection`` canonical exactly."""
    st=Supertrend(high,low,close,atr_window,multiplier)
    c=close.to_numpy(float); s=st.to_numpy(float)
    sign=np.full(s.shape,np.nan)
    for j in range(s.shape[1]):
        for t in range(s.shape[0]):
            if np.isfinite(s[t,j]) and np.isfinite(c[t,j]):
                sign[t,j]=1.0 if c[t,j]>=s[t,j] else -1.0
    direction=pd.DataFrame(sign,index=st.index,columns=st.columns)
    return st, direction

def supertrend_direction(high,low,close,atr_window,multiplier):
    return _supertrend_state(high,low,close,atr_window,multiplier)[1]

def supertrend_distance_pct(high,low,close,atr_window,multiplier):
    """(close - Supertrend line) / close; strict-positive close masked (bad
    close -> NaN, never an abs-flipped or sign-laundered ratio)."""
    st=_supertrend_state(high,low,close,atr_window,multiplier)[0]
    pos=close.where(close>0)
    return (pos-st).div(pos)

def supertrend_flip(high,low,close,atr_window,multiplier):
    """+1 on up-flip, -1 on down-flip, 0 hold; NaN when either side of the
    direction series is NaN (same both-sides-finite contract as psar_flip)."""
    st,direction=_supertrend_state(high,low,close,atr_window,multiplier)
    prev=direction.shift(1)
    flip=(direction-prev).where(direction.notna()&prev.notna())
    flip=flip.where(direction.notna()).fillna(0.0).where(direction.notna())
    return pd.DataFrame(np.sign(flip.to_numpy(float)),index=st.index,columns=st.columns)

def supertrend_days_since_flip(high,low,close,atr_window,multiplier):
    """Bars since the last Supertrend regime flip; NaN before the first flip
    (mirrors psar_days_since_flip exactly)."""
    st,_direction=_supertrend_state(high,low,close,atr_window,multiplier)
    flip=supertrend_flip(high,low,close,atr_window,multiplier).to_numpy(float)
    out=np.full(flip.shape,np.nan)
    for j in range(flip.shape[1]):
        since=np.nan
        for t in range(flip.shape[0]):
            if np.isnan(flip[t,j]):
                continue
            if flip[t,j]!=0.0:
                since=0.0
            elif since==since:
                since+=1.0
            out[t,j]=since
    return pd.DataFrame(out,index=st.index,columns=st.columns)

# --- Donchian channel family (R20-DONCHIAN-DIRECTUSE) ---
# Trailing rolling window ENDS at t (inclusive): upper = rolling-max high,
# lower = rolling-min low, min_periods=window.  Rolling-only => bounded state,
# NOT added to _RECURSIVE_EWM (no full-replay governance needed).
def _donchian(high,low,window):
    w=_pi(window,"window",2)
    return high.rolling(w,min_periods=w).max(), low.rolling(w,min_periods=w).min()

def donchian_width_pct(high,low,close,window):
    """(upper - lower) / close; strict-positive close masked (bad close -> NaN)."""
    u,l=_donchian(high,low,window)
    pos=close.where(close>0.0)
    return (u-l)/pos

def donchian_channel_position(high,low,close,window):
    """(close - lower) / (upper - lower); NaN when width==0 or either band NaN.

    Inclusive-t trailing window — distinct from the legacy ``donchian_position``
    canonical (production_technical_extensions), which is prior-window
    (t-1-ending) with param order [close, high, low, window]."""
    u,l=_donchian(high,low,window)
    return (close-l)/(u-l).replace(0,np.nan)

def donchian_breakout_up(high,low,close,window):
    """Breakout magnitude above the PRIOR Donchian channel: close / max(high
    over the previous ``window`` bars ending at t-1) - 1, emitted ONLY when the
    current close makes a fresh N-bar high (close >= prior max); 0.0 on other
    post-warmup bars, NaN during warmup or on bad (non-positive) close.

    Causal semantic (documented choice): the channel a bar-t breakout breaks
    THROUGH must be the one that existed BEFORE t — a window including bar t
    would contain the very high being broken, clipping every breakout to <= 0
    (close <= high_t <= inclusive max).  So the reference window ends at t-1
    (``rolling(w, min_periods=w).max().shift(1)``), strictly trailing and
    PIT-safe; the event itself is fully observable at t.  Note: a bar closing
    exactly AT the prior extreme (close == prev max) emits 0.0, which is
    indistinguishable from the no-breakout 0.0 sentinel by design."""
    u,l=_donchian(high,low,window)
    prev_u=u.shift(1)
    pos=close.where(close>0.0)
    ret=pos/prev_u-1.0
    fresh=(close>=prev_u)&(close>0.0)&prev_u.notna()
    # A non-positive close is bad data (R5-38): NaN, never a laundered 0.0.
    valid=prev_u.notna()&close.notna()&(close>0.0)
    return ret.where(fresh).where(valid).fillna(0.0).where(valid)

def donchian_breakout_down(high,low,close,window):
    """Mirror of donchian_breakout_up: close / min(low over the previous
    ``window`` bars ending at t-1) - 1 (a negative fraction), emitted ONLY when
    the current close makes a fresh N-bar low (close <= prior min); 0.0 on
    other post-warmup bars, NaN during warmup or on bad close."""
    u,l=_donchian(high,low,window)
    prev_l=l.shift(1)
    pos=close.where(close>0.0)
    ret=pos/prev_l-1.0
    fresh=(close<=prev_l)&(close>0.0)&prev_l.notna()
    valid=prev_l.notna()&close.notna()&(close>0.0)
    return ret.where(fresh).where(valid).fillna(0.0).where(valid)

# --- Channel / consolidation family (R20-CHANNEL-CONSOLIDATION-DIRECTUSE) ---
# Dimensionless Direct Alpha canonicals over trailing rolling windows that END
# at t (inclusive, PIT-safe, min_periods=window).  Rolling-only => bounded
# state, NOT added to _RECURSIVE_EWM.  Duplicate-audit before landing:
# * channel_width_pct / channel_position are NOT the same math as
#   donchian_width_pct / donchian_channel_position (those ARE the same
#   inclusive-t Donchian bands) — SKIPPED as duplicates, not re-registered
#   (see evidence/r2/R20-CHANNEL-CONSOLIDATION-DIRECTUSE.yaml).
# * ts_channel_width_pct (structure_patterns_v2) is regression-line distance
#   based (pivot-fitted lines, abs()-close), mathematically distinct.
# * ts_consolidation_width is PRICE-SCALE (high.max - low.min, not normalized)
#   — the dimensionless close-only tightness measures below are new.
# * No existing canonical computes rolling_std(close)/rolling_mean(close) or
#   the close-only range fraction; consolidation_pct /
#   consolidation_range_pct are genuinely new.
def _channel_pos_close(close):
    """Strict-positive close mask (R5-38): close<=0 or NaN -> NaN, never abs()."""
    return close.where(close > 0.0)

def consolidation_pct(close, window):
    """rolling_std(close, w) / rolling_mean(close, w) — coefficient of
    variation of the close level over the trailing w bars ending at t.
    Dimensionless tightness: LOW = price consolidating, HIGH = dispersing.
    Strict-positive rolling mean masked (bad/flat-zero mean -> NaN).

    Fail-closed window kill: min_periods=w means a single bad close inside
    the window NaNs every trailing window covering it (w consecutive NaN
    outputs) — stricter than excluding the bar from an over-valid cohort,
    deliberately mirroring the candle_/vwap_ rolling-family contract."""
    w = _pi(window, "window", 2)
    pos = _channel_pos_close(close)
    mean = pos.rolling(w, min_periods=w).mean()
    std = pos.rolling(w, min_periods=w).std(ddof=1)
    return std / mean.where(mean > 0.0)

def consolidation_range_pct(close, window):
    """(rolling_max(close,w) - rolling_min(close,w)) / close — close-only
    range tightness over the trailing w bars ending at t, dimensionless.
    LOW = tight close-only range (consolidation).  Strict-positive close
    masked (bad close -> NaN); scale-invariant across price levels.
    Fail-closed window kill: same min_periods=w contract as
    consolidation_pct — a bad close NaNs its whole covering window."""
    w = _pi(window, "window", 2)
    pos = _channel_pos_close(close)
    hi = pos.rolling(w, min_periods=w).max()
    lo = pos.rolling(w, min_periods=w).min()
    return (hi - lo) / pos

# --- MA distance / slope / crossover family (R20-MA-DISTANCE-SLOPE) ---
# Dimensionless Direct Alpha canonicals over EXISTING MA machinery (EMA /
# DEMA / TEMA helpers above, registered KAMA).  These are NOT new MAs: every
# one is a causal transform of an already-registered MA level, normalized to
# strict-positive close so a bad (non-positive) close bar -> NaN, never a
# laundered value.  All MAs are computed at t (inclusive); no shift, no
# future data.
def _ma_distance(close, ma):
    """(close - MA) / close with strict-positive close masking."""
    pos = close.where(close > 0.0)
    return (pos - ma) / pos

def ema_distance_pct(close, window):
    """(close - EMA(close, window)) / close; dimensionless, causal."""
    return _ma_distance(close, _ema(close, window))

def sma_distance_pct(close, window):
    """(close - rolling SMA(close, window)) / close; window ends at t (PIT)."""
    w = _pi(window, "window", 2)
    return _ma_distance(close, close.rolling(w, min_periods=w).mean())

def dema_distance_pct(close, window):
    """(close - DEMA(close, window)) / close; DEMA = 2*EMA - EMA(EMA)."""
    return _ma_distance(close, DEMA(close, window))

def tema_distance_pct(close, window):
    """(close - TEMA(close, window)) / close; TEMA = 3EMA - 3EMA(EMA) + EMA(EMA(EMA))."""
    return _ma_distance(close, TEMA(close, window))

def kama_distance_pct(close, er_window, fast_window, slow_window):
    """(close - KAMA) / close.  Reuses the REGISTERED pandas_numpy KAMA
    canonical (indicators_v2.KAMA, which pins/replaces the composite_fastpath
    KAMA per the round-7 chain) — the KAMA math itself is never reimplemented
    here.  Param contract mirrors the registered KAMA spec exactly."""
    return _ma_distance(close, KAMA(close, er_window, fast_window, slow_window))

def ma_slope_pct(close, window):
    """EMA(close, window).diff() / close — a dimensionless per-bar MA slope.
    The price-unit MA difference is normalized by the strict-positive close,
    so it is comparable across names/prices without carrying price scale."""
    return _ema(close, window).diff() / close.where(close > 0.0)

def ema_crossover(close, fast_window, slow_window):
    """sign((fast EMA - slow EMA) / close), bounded {-1, 0, +1}: +1 when the
    fast EMA is above the slow EMA, -1 below, 0 exactly on a dead tie, NaN
    while either EMA (or close) is NaN.  Semantic choice: normalizing by close
    only localizes the sign (sign(x/c) == sign(x) for c>0), so the bounded
    ±1 regime is clean and dimensionless; the magnitude lives in
    ema_distance_pct/PPO.  fast_window < slow_window enforced (same relational
    contract as PPO)."""
    f = _pi(fast_window, "fast_window"); s = _pi(slow_window, "slow_window")
    if f >= s: raise ValueError("fast_window must be < slow_window")
    diff = (_ema(close, f) - _ema(close, s)) / close.where(close > 0.0)
    return pd.DataFrame(np.sign(diff.to_numpy(float)), index=diff.index, columns=diff.columns)

# --- Ichimoku DirectUse family (R20-ICHIMOKU-DIRECTUSE) ---
# CAUSAL, dimensionless Ichimoku canonicals.  The classic 26-bar FORWARD
# displacement of Senkou A/B is DISPLAY-ONLY: honouring it would read future
# bars, so every window here is trailing and ENDS at t (min_periods=window,
# rolling-only bounded state => NOT in _RECURSIVE_EWM).  The classic Chikou
# span (close plotted 26 bars BACK, i.e. compared against bars 26 ahead) is
# likewise future-looking in its standard usage and is deliberately NOT used;
# the equilibrium-line distance is computed directly at t instead.
# Tenkan/Kijun levels are reused from the registered canonicals above — the
# midpoint math is never reimplemented here.
def _ichimoku_pos_close(close):
    """Strict-positive close (R5-38): a non-positive close is bad data -> NaN."""
    return close.where(close > 0.0)

def tenkan_kijun_cross(high, low, close, tenkan_window, kijun_window):
    """sign((Tenkan - Kijun) / close), bounded {-1, 0, +1}: +1 when the fast
    Tenkan line is above the slow Kijun line, -1 below, 0 exactly on a dead
    tie, NaN during warmup or on a bad (non-positive / missing) close.
    Normalizing by close only localizes the sign (sign(x/c) == sign(x) for
    c > 0), so the bounded +/-1 regime stays clean and dimensionless.  NO
    future shift: both lines are trailing windows ending at t.  tenkan_window
    < kijun_window enforced (same relational contract family as PPO)."""
    t = _pi(tenkan_window, "tenkan_window", 2); k = _pi(kijun_window, "kijun_window", 2)
    if t >= k: raise ValueError("tenkan_window must be < kijun_window")
    diff = (ichimoku_tenkan(high, low, t) - ichimoku_kijun(high, low, k)) / _ichimoku_pos_close(close)
    return pd.DataFrame(np.sign(diff.to_numpy(float)), index=diff.index, columns=diff.columns)

def chikou_distance_pct(high, low, close, kijun_window):
    """(close - Kijun) / close — dimensionless distance of price from the
    Kijun equilibrium line, computed CAUSALLY at t.  The classic Chikou span
    shifts close 26 bars back so that "today's close vs the cloud 26 bars
    ahead" can be judged — that reads future bars and is future-looking; this
    canonical keeps the same economic content (price vs equilibrium) with NO
    shift: strict-positive close masked (bad close -> NaN, never abs())."""
    kijun = ichimoku_kijun(high, low, kijun_window)
    pos = _ichimoku_pos_close(close)
    return (pos - kijun) / pos

def senkou_span_causal_pct(high, low, close, tenkan_window, kijun_window):
    """((Tenkan_t + Kijun_t) / 2 - close) / close — the CAUSAL "cloud mid vs
    price" gap.  The classic Senkou A is displaced 26 bars forward for chart
    display; that displacement is display-only and future-looking, so this
    canonical uses the un-shifted mid ((Tenkan_t + Kijun_t)/2, i.e. the same
    formula as ichimoku_senkou_a) against the CURRENT close.  Strict-positive
    close masked (bad close -> NaN, never abs()).  tenkan_window <
    kijun_window enforced (swap creates a duplicate search node)."""
    t = _pi(tenkan_window, "tenkan_window", 2); k = _pi(kijun_window, "kijun_window", 2)
    if t >= k: raise ValueError("tenkan_window must be < kijun_window")
    # Scoped-review P2: delegate to the registered senkou_a formula rather than
    # re-averaging tenkan/kijun here (same math today, no drift vector).
    mid = ichimoku_senkou_a(high, low, t, k)
    pos = _ichimoku_pos_close(close)
    return (mid - pos) / pos

# --- Rolling VWAP DirectUse family (R20-VWAP-DIRECTUSE) ---
# Causal, dimensionless volume-weighted-cost canonicals.  The trailing window
# ENDS at t (min_periods=window, rolling-only bounded state => NOT in
# _RECURSIVE_EWM).  The kernel mirrors the registered price-scale
# ``rolling_vwap`` (price_volume/technical_extensions) but adds the STRICT
# valid-volume cohort contract this slice requires: a bar contributes only
# when price AND volume are present AND volume > 0; a window whose valid
# volume sum is <= 0 -> NaN (bad volume never launders into a price).
def _rolling_vwap_strict(price, volume, window):
    """Trailing rolling VWAP = rolling(w).sum(price*volume)/rolling(w).sum(volume).

    Window ends at t, min_periods=w (every bar in the window must be valid).
    ``volume <= 0`` or NaN on a bar excludes that bar from BOTH sums; if the
    valid volume sum is <= 0 the result is NaN (never 0/0 -> fake price)."""
    w = _pi(window, "window", 2)
    valid = price.notna() & volume.notna() & (volume > 0.0)
    num = (price * volume).where(valid).rolling(w, min_periods=w).sum()
    den = volume.where(valid).rolling(w, min_periods=w).sum()
    return num / den.where(den > 0.0)

def vwap_distance_pct(close, volume, window):
    """(close - rolling VWAP(close, volume, window)) / close — dimensionless
    distance of price from its volume-weighted cost basis.  Strict-positive
    close masked AND valid-volume masked: a non-positive close or a bad
    (<= 0 / missing) volume bar -> NaN, never a laundered value."""
    pos = close.where(close > 0.0)
    return (pos - _rolling_vwap_strict(close, volume, window)) / pos

def vwap_slope_pct(close, volume, window):
    """rolling VWAP.diff() / close — a dimensionless per-bar slope of the
    volume-weighted cost basis (price-unit difference normalized by the
    strict-positive close, comparable across names without price scale)."""
    return _rolling_vwap_strict(close, volume, window).diff() / close.where(close > 0.0)

def vwap_premium_pct(high, low, close, volume, window):
    """(close - HLC3 rolling VWAP) / close — the classic typical-price VWAP
    variant ((high+low+close)/3 weighted by volume), distinct from
    vwap_distance_pct's close-only kernel: it measures price premium over the
    INTRADAY-average cost basis rather than the close-print cost basis.
    Same strict-positive close + valid-volume masking contract."""
    tp = (high + low + close) / 3.0
    pos = close.where(close > 0.0)
    return (pos - _rolling_vwap_strict(tp, volume, window)) / pos

# --- Candlestick DirectUse family (R20-CANDLESTICK-DIRECTUSE) ---
# AGGREGATE per-bar candlestick-pattern STRENGTH over a trailing window (w bars
# ending at t, min_periods=w, rolling-only bounded state => NOT in
# _RECURSIVE_EWM): these are NOT one-off binary cdl_* detectors.  Per-bar
# geometry (body / upper wick / lower wick) is normalized by the bar range so
# every kernel is dimensionless and comparable across names/prices/scales.
# Bad bar handling (never laundered): a bar is structurally valid only when
# high > low (STRICT: a zero range cannot normalize the geometry), close > 0
# and the OHLC nesting holds (high >= max(open,close), low <= min(open,close));
# anything else (high < low, non-positive close, zero range, missing fields) is
# NaN on that bar and excluded from the trailing mean (skipna), so a deformed
# bar can never contribute a fabricated 0 / ±1 geometry term.  This mirrors the
# spirit of ``_validate_ohlc`` in price_volume/candle_geometry_v2.py, inlined
# here as a local helper because the registered canonicals there are
# prior-window z-scores / percentiles with a different
# (not aggregate-strength) contract.
def _candle_valid(open_, high, low, close):
    """Structural OHLC validity mask for aggregate candle geometry.

    True only when all four fields are present, close > 0, high > low (a
    strict inequality: a zero range cannot normalize the geometry) AND
    high >= max(open, close) / low <= min(open, close).  Fails closed on NaN."""
    valid = (
        open_.notna() & high.notna() & low.notna() & close.notna()
        & (close > 0.0) & (high > low)
        & (high >= pd.DataFrame(np.maximum(open_.to_numpy(float), close.to_numpy(float)),
                                index=high.index, columns=high.columns))
        & (low <= pd.DataFrame(np.minimum(open_.to_numpy(float), close.to_numpy(float)),
                               index=high.index, columns=high.columns))
    )
    return valid

def _candle_geometry(open_, high, low, close):
    """(signed_body_ratio, wick_balance, range_pct) per bar, NaN on bad bars.

    signed_body_ratio = sign(close-open) * |close-open| / (high-low)
    wick_balance      = (lower_wick - upper_wick) / (high-low)
    range_pct         = (high-low) / close
    All three are dimensionless; body and wick terms are in [-1, 1]."""
    valid = _candle_valid(open_, high, low, close)
    rng = (high - low).where(valid)                      # > 0 on valid bars
    upper = high - pd.DataFrame(np.maximum(open_.to_numpy(float), close.to_numpy(float)),
                                index=high.index, columns=high.columns)
    lower = pd.DataFrame(np.minimum(open_.to_numpy(float), close.to_numpy(float)),
                         index=high.index, columns=high.columns) - low
    sign = np.sign((close - open_).to_numpy(float))
    body = pd.DataFrame(sign * (close - open_).abs().to_numpy(float) / rng.to_numpy(float),
                        index=high.index, columns=high.columns)
    wick = ((lower - upper) / rng.replace(0, np.nan)).where(valid)
    rpct = (rng / close.where(close > 0.0)).where(valid)
    return body.where(valid), wick, rpct

def candle_body_strength(open_, high, low, close, window):
    """Mean over the trailing ``window`` bars of sign(close-open) * body/range
    — aggregate directional candle strength in [-1, 1]: +1 = every bar a full
    marubozu up-candle, -1 = all-down, ~0 = doji-heavy / directionless bars.
    min_periods=window means a bad bar NaNs every window covering it (w
    consecutive NaN outputs) — stricter than plain skipna exclusion, and
    deliberately different from candle_pattern_count's over-valid-fraction
    contract; both are fail-closed, the asymmetry is intentional."""
    w = _pi(window, "window", 2)
    body, _wick, _rpct = _candle_geometry(open_, high, low, close)
    return body.rolling(w, min_periods=w).mean()

def candle_wick_balance(open_, high, low, close, window):
    """Mean over the trailing ``window`` bars of (lower_wick - upper_wick)/range
    — buying-vs-selling pressure proxy in [-1, 1]: persistent lower (upper)
    wicks mean dip-buyers (distribution) rejecting the extreme each bar."""
    w = _pi(window, "window", 2)
    _body, wick, _rpct = _candle_geometry(open_, high, low, close)
    return wick.rolling(w, min_periods=w).mean()

def candle_range_pct(open_, high, low, close, window):
    """Mean over the trailing ``window`` bars of (high-low)/close — bar-range
    volatility normalized by price (dimensionless); distinct from ATR-family
    kernels: plain per-bar range over close, no Wilder smoothing, no gaps."""
    w = _pi(window, "window", 2)
    _body, _wick, rpct = _candle_geometry(open_, high, low, close)
    return rpct.rolling(w, min_periods=w).mean()

def candle_pattern_count(open_, high, low, close, window):
    """Fraction of VALID bars in the trailing ``window`` classified as a
    classic single-bar indecision / one-sided-rejection pattern: |body/range|
    < 0.1 (doji), OR >= 70% of the range in one wick (hammer / inverted hammer
    / shooting-star / marubozu-wick shapes) — i.e. one of body or wicks
    dominates while the other two are small.  Clean, bounded [0, 1], computed
    per bar from the same valid-masked geometry; bad bars are excluded from
    BOTH numerator and denominator, never counted as non-patterns."""
    w = _pi(window, "window", 2)
    valid = _candle_valid(open_, high, low, close)
    rng = (high - low).where(valid)
    upper = (high - pd.DataFrame(np.maximum(open_.to_numpy(float), close.to_numpy(float)),
                                 index=high.index, columns=high.columns)).where(valid)
    lower = (pd.DataFrame(np.minimum(open_.to_numpy(float), close.to_numpy(float)),
                          index=high.index, columns=high.columns) - low).where(valid)
    body = (close - open_).abs().where(valid)
    doji = body / rng < 0.1
    dom_upper = upper / rng >= 0.7
    dom_lower = lower / rng >= 0.7
    # A dominant wick bar is a genuine pattern only if the OPPOSITE wick is
    # small (the rejection shape), so full-range one-sided tails qualify while
    # an ordinary wide-spread bar does not.
    pattern = doji | (dom_upper & (lower / rng <= 0.15)) | (dom_lower & (upper / rng <= 0.15))
    # Exclusion semantics: a bad bar drops out of BOTH numerator and
    # denominator (fraction over VALID bars only).  min_periods=1 lets a
    # partially-valid window still emit; the warmup mask below restores the
    # strict trailing-window contract (no output before w ROWS exist).
    num = pattern.astype(float).where(valid).rolling(w, min_periods=1).sum()
    den = valid.astype(float).rolling(w, min_periods=1).sum()
    rows = (open_.notna() & high.notna() & low.notna() & close.notna()).astype(float).rolling(w, min_periods=w).sum()
    return (num / den.where(den > 0.0)).where(rows >= float(w))

# --- Support/Resistance DirectUse family (R20-SR-DIRECTUSE) ---
# Causal, dimensionless support/resistance canonicals built on TYPICAL-PRICE
# pivot levels.  ONE documented pivot definition for the whole family: the
# prior-window pivot at t is the rolling(w) MEDIAN of the typical price
# (high+low+close)/3 over the ``window`` bars ending at t-1 (strictly trailing
# — a level a bar-t decision can act against must be formed BEFORE t).
# Reuse note: the registered ts_support_level / ts_resistance_level /
# ts_last_pivot_* canonicals are pivot-CLUSTER regression levels with a
# left/right confirmation-window contract (a confirmed pivot is only known
# right_window bars LATER) — different semantics and param shape; the Donchian
# family is high/low EXTREMES, not typical-price medians.  Nothing suitable to
# reuse, so the median pivot is computed here.  Rolling-only bounded state =>
# NOT in _RECURSIVE_EWM (no full-replay governance).
# Bad-bar contract (never laundered): a bar contributes to / is judged against
# a level only when its high/low/close are present; a non-positive close is
# bad data (R5-38) -> NaN; min_periods=w keeps the strict warmup contract.
def _sr_prior_pivot(high, low, close, window):
    """Prior-window typical-price pivot: rolling(w).median of (h+l+c)/3 over
    the bars ending at t-1 (``shift(1)``), min_periods=w.  Bars with a
    non-positive or missing close are bad data (R5-38) and contribute a NaN
    typical price — a pivot window covering one NaNs out (fail-closed), it
    never silently contaminates the median."""
    w = _pi(window, "window", 2)
    tp = ((high + low + close) / 3.0).where(
        high.notna() & low.notna() & close.notna() & (close > 0.0))
    return tp.rolling(w, min_periods=w).median().shift(1)

def sr_distance_pct(high, low, close, window):
    """(close - prior typical-price pivot) / close — signed dimensionless
    distance of price from the prior-window median typical-price level
    (rolling(w).median of (h+l+c)/3 over bars t-w..t-1).  Strict-positive
    close masked (bad close -> NaN, never abs()); NaN during warmup (the
    prior pivot needs w+1 rows)."""
    pivot = _sr_prior_pivot(high, low, close, window)
    pos = close.where(close > 0.0)
    return (pos - pivot) / pos

def sr_touch_count(high, low, close, window, tol):
    """Fraction of the last ``window`` bars whose high OR low came within
    ``tol`` (a dimensionless relative tolerance) of the prior pivot level —
    a dimensionless "level importance" measure in [0, 1].

    touch is a TIME-COINCIDENT event: window bar ``i`` in t-w+1..t
    "touches" when |high_i - pivot_i| <= tol * pivot_i OR
    |low_i - pivot_i| <= tol * pivot_i, where pivot_i is the prior-window
    median typical price AT BAR i'S OWN ROW (formed from bars i-w..i-1, the
    same level definition the whole SR family judges against).  Bar i is
    never compared against the level formed later at t — that would leak
    t's information backwards.  ``tol`` is a strict positive float
    (_POS_FLOAT).  Unjudgeable bars (missing high/low, non-positive close,
    or no comparable level at their own row) drop out of BOTH numerator and
    denominator — never counted as non-touches; a non-positive close on the
    JUDGED bar t is NaN (R5-38).  Warmup-edge rows t in [w, 2w) emit a
    fraction over the partial (<w) judgeable cohort — a documented
    boundary artifact of the over-valid-fraction contract."""
    w = _pi(window, "window", 2)
    t = _pf(tol, "tol")
    if t <= 0: raise ValueError("tol must be > 0")
    pivot = _sr_prior_pivot(high, low, close, window)
    # band_i = tol * pivot_i; pivot/band are DataFrames aligned with the OHLC
    # panel (typical price is computed columnwise), so plain elementwise
    # DataFrame arithmetic compares bar i against ITS OWN row's level.
    band = (t * pivot).where(pivot > 0.0)
    touch_h = (high - pivot).abs() <= band
    touch_l = (low - pivot).abs() <= band
    # A bar is judgeable when both extremes AND its own comparable level exist.
    valid = high.notna() & low.notna() & band.notna()
    touch = (touch_h | touch_l) & valid
    # Fraction over judgeable bars (bad bars drop out of BOTH numerator and
    # denominator — never counted as non-touches).  den > 0 also enforces the
    # warmup: the earliest judgeable bar needs w prior rows for its own pivot,
    # so no output before row w.
    num = touch.astype(float).rolling(w, min_periods=1).sum()
    den = valid.astype(float).rolling(w, min_periods=1).sum()
    frac = num / den.where(den > 0.0)
    # The judged bar t itself needs present extremes and a strict-positive
    # close (R5-38); interior bad bars stay excluded from num/den instead.
    return frac.where(high.notna() & low.notna() & close.notna() & (close > 0.0))

# --- Spectral / wavelet excess family (R20-SPECTRAL-EXCESS-DIRECTUSE) ---
# Dimensionless energy-concentration canonicals.  Duplicate audit: the existing
# frequency_layer provides *filters* (ts_spectral_lowpass_trailing, denoise
# ts_wavelet_shrinkage_trailing) that return a filtered PRICE series; no
# existing canonical returns a dimensionless energy RATIO.  These are new math.
#
# Contract shared by the family:
# * trailing window ENDS at t, min_periods=window (a single NaN inside the
#   window makes the output NaN via min_periods — fail-closed, NaNs are never
#   zero-filled before the transform, which would fabricate a flat segment);
# * DC rejection is explicit: the window is de-meaned (trailing mean subtracted)
#   BEFORE numpy rfft — bin 0 is exactly zero by construction, and the total
#   energy denominator EXCLUDES DC.  No separate rolling-mean detrend step;
# * strict-positive close masking (R5-38): close<=0 or NaN -> NaN;
# * outputs are ratios of non-negative energies -> [0,1] by construction;
#   zero total energy (perfectly constant window) -> 0.0 (documented, tested).
def _spectral_energy(close, window, reducer):
    """Shared trailing-window spectral pipeline.  ``reducer(a)`` maps the RAW
    (non-de-meaned) window numpy array to a scalar ratio — the rfft-based
    reducers de-mean and compute the power spectrum themselves via
    _rfft_power_excluding_dc, the Haar reducer de-means and transforms
    directly; nothing outside the reducer assumes a spectrum.  Window ends at
    t, min_periods=window (any NaN inside the window -> NaN output,
    fail-closed)."""
    w = _pi(window, "window", 4)
    pos = _channel_pos_close(close)
    return pos.rolling(w, min_periods=w).apply(
        lambda a: reducer(a), raw=True
    )

def _rfft_power_excluding_dc(a):
    """Non-DC rfft power (squared |X_k|) of the de-meaned window; the total
    is the window variance energy — no DC leakage, no zero-fill of NaN."""
    x = a - a.mean()
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    return power[1:]  # bin 0 (DC) is 0 by construction after de-meaning

def spectral_energy_ratio(close, window, top_k=2):
    """Fraction of the trailing window's non-DC spectral energy carried by
    its ``top_k`` largest rfft bins.  Dimensionless in [0,1]: HIGH = a single
    frequency (or a few) dominates — trend/cycle regime; LOW = flat noise
    spectrum.  ``top_k`` is a fixed structural constant of the measure (NOT a
    look-ahead-tuned parameter); window ends at t, min_periods=window (any
    NaN inside the window -> NaN output, fail-closed).  Constant window
    (zero total energy) -> 0.0 (no concentration to measure)."""
    def _ratio(a, k=_pi(top_k, "top_k", 1)):
        power = _rfft_power_excluding_dc(a)
        total = power.sum()
        if total <= _EPS:
            return 0.0
        k = min(k, power.size)
        return float(np.sort(power)[::-1][:k].sum() / total)
    return _spectral_energy(close, window, _ratio)

def spectral_trend_share(close, window, trend_bins=2):
    """Fraction of the trailing window's non-DC spectral energy in its
    ``trend_bins`` LOWEST-frequency bins (bin 1 = the slowest full-window
    cycle; DC is excluded via explicit de-meaning before rfft — the window
    mean carries no oscillatory energy).  Dimensionless in [0,1]: HIGH =
    energy concentrated at slow frequencies (trend-dominated), LOW =
    high-frequency noise dominates.  Window ends at t, min_periods=window,
    NaN-in-window -> NaN; constant window -> 0.0."""
    def _share(a, b=_pi(trend_bins, "trend_bins", 1)):
        power = _rfft_power_excluding_dc(a)
        total = power.sum()
        if total <= _EPS:
            return 0.0
        b = min(b, power.size)
        return float(power[:b].sum() / total)
    return _spectral_energy(close, window, _share)

def _pow2(v, name):
    # min=4 (not 2): the whole spectral family rejects windows < 4 in
    # _spectral_energy, so _pow2 must reject 2 with the power-of-2 message
    # rather than letting it pass and die later on the ">= 4" message.
    v = _pi(v, name, 4)
    if v & (v - 1):
        raise ValueError(f"{name} must be a power of 2 (got {v})")
    return v

def wavelet_detail_energy_ratio(close, window):
    """Fraction of the de-meaned trailing window's energy in the Haar DETAIL
    (high-frequency) coefficients — pure numpy, NO new dependency.  Standard
    adjacent-pair multilevel Haar: level-j details pair neighbouring samples
    ((a-b)/sqrt(2)), the approximation carries the smooth component forward.
    A FULL decomposition of a de-meaned window leaves only the single DC
    scaling coefficient (exactly 0), so "all detail levels / total" would be
    identically 1 — meaningless.  Instead the detail sum runs over levels
    1..J-1 and the COARSEST dyadic detail (the single half-vs-half
    coefficient) is grouped with the zero DC into the trend subspace; the
    split point is structural (determined solely by the window length), not
    a tuned parameter.  Dimensionless in [0,1]: LOW = smooth/trend-dominated,
    HIGH = choppy/noise-dominated.  ``window`` MUST be a power of 2 (enforced
    by ValueError at the call boundary); window ends at t, min_periods=window,
    any NaN inside the window -> NaN (fail-closed, no zero-fill); constant
    window (zero total energy) -> 0.0.  Complements spectral_trend_share: the
    Haar split is a dyadic time-frequency partition, not an rfft bin ratio."""
    def _detail_ratio(a):
        x = a - a.mean()
        total = float(np.dot(x, x))
        if total <= _EPS:
            return 0.0
        detail_energy = 0.0
        n = x.size  # power of 2, >= 2
        y = x
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        # stop one level early: the last detail (n==2, half-vs-half) is trend
        while n >= 4:
            even, odd = y[0::2], y[1::2]
            diff = even - odd
            detail_energy += 0.5 * float(np.dot(diff, diff))
            y = (even + odd) * inv_sqrt2
            n = y.size
        return float(min(detail_energy / total, 1.0))
    w = _pow2(window, "window")
    return _spectral_energy(close, w, _detail_ratio)

# --- Causal dimensionless regression family (R20-REGRESSION-CAUSAL-DIRECTUSE) ---
# Duplicate audit (survey before implementing — documented):
# * ts_regression_* (overhaul/regression.py, EXTENDED_ONLY) is a general
#   two-frame (y, x) rolling OLS with pairwise-finite NaN handling and
#   min_periods semantics; ts_trend_tstat is its time-trend t-stat twin.
#   Here the target is a *technical-signal* canonical family with the strict
#   family contract of this module: close-only, single trailing window,
#   min_periods=window (a single NaN inside the window -> NaN, fail-closed,
#   never pairwise-finite-reconnect), strict-positive close masking (R5-38),
#   dimensionless output, and a *prior-window* (target-excluding) forecast
#   variant.  Math summary:
#   - reg_slope_tstat / reg_r2_trailing are the dimensionless time-trend
#     t-stat / R^2 of the trailing window (math shared with ts_trend_tstat /
#     ts_regression(y=t) r2, but the landed canonical enforces this module's
#     fail-closed masking + promotion surface — no duplicate canonical name
#     is registered).
#   - reg_forecast_error_pct is genuinely new on this surface: the forecast
#     target bar t is EXCLUDED from the fit window (fit on t-w..t-1), then
#     the one-step-ahead error is normalized by close_t -> dimensionless.
#   - reg_residual_zscore: contemporaneous fitted value at t (fit window
#     INCLUDES t — that is NOT look-ahead) z-scored by the trailing residual
#     std of the same window.
# * ts_mean_reversion_half_life / ts_mean_reversion_ou_approx_half_life
#   (ts_model/ar_meanrev.py, experimental, spread/residual input semantic
#   with a trending-input warning) already implement BOTH the exact discrete
#   AR(1) half-life ln(0.5)/ln(phi) and the OU approximation -ln(2)/b.
#   Landing a mean_reversion_half_life here would be a near-exact duplicate
#   (same estimator, same 0<phi<1 NaN guard) -> SKIPPED; rationale recorded
#   in evidence/r2/R20-REGRESSION-CAUSAL-DIRECTUSE.yaml.
# Family contract (all four): trailing windows END at t, min_periods=window,
# no future shift anywhere; strict-positive close masking (R5-38, close<=0 ->
# NaN, never abs()-laundered); NaN anywhere inside a window -> NaN (fail-closed,
# no zero-fill); degenerate denominators (zero time variance, zero residual
# std) -> NaN, never 0.  All outputs dimensionless (error normalized by close,
# t-stat by the slope SE, R^2 and z-score by construction).
def _reg_fit_window(a):
    """OLS of the de-meaned trailing window on de-meaned time — closed form.

    Returns (slope, intercept, resid_std_ddof1, r2, slope_se) or None for a
    degenerate window (zero time variance is impossible for min_periods>=2 —
    the guard stays fail-closed anyway)."""
    n = a.size
    if n < 3 or not np.isfinite(a).all():
        return None
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    s_tt = float(np.dot(tc, tc))
    if s_tt <= _EPS:
        return None
    slope = float(np.dot(tc, a - a.mean()) / s_tt)
    intercept = float(a.mean() - slope * t.mean())
    resid = a - (intercept + slope * t)
    ss_res = float(np.dot(resid, resid))
    centered = a - a.mean()
    ss_tot = float(np.dot(centered, centered))
    r2 = np.nan if ss_tot <= _EPS else 1.0 - ss_res / ss_tot
    # residual SE with n-2 dof (the fit consumes 2: slope + intercept);
    # SE(slope) = s / sqrt(sum((t-tbar)^2)).  n >= 3 here (n < 3 returned
    # None above), so n - 2 >= 1 always.
    resid_std = float(np.sqrt(ss_res / (n - 2)))
    slope_se = np.nan if not np.isfinite(resid_std) else resid_std / np.sqrt(s_tt)
    return slope, intercept, resid_std, r2, slope_se

def reg_forecast_error_pct(close, window):
    """(close_t - OLS forecast of close_t) / close_t — one-step-ahead
    prediction error of a trailing linear time-trend regression, normalized
    by close (dimensionless surprise).  CAUSALITY CONTRACT: the forecast
    target bar t is EXCLUDED from the fit window — the regression is fit on
    bars t-window..t-1 ONLY (strictly prior data), then the fitted line is
    extrapolated one step to t.  Window >= 4 (>= 3 fit bars — the OLS fit
    needs n >= 3 for a nonzero residual dof — plus the target bar; window=3
    leaves only 2 fit bars and is all-NaN by construction, so the spec and
    the runtime guard both reject it up front);
    min_periods=window so any NaN inside the fit window OR a NaN/bad target
    close -> NaN (fail-closed).  Strict-positive close masking (R5-38): a
    non-positive close anywhere in the window or at t -> NaN (never
    abs()-laundered).  Degenerate fit (zero time variance) -> NaN."""
    w = _pi(window, "window", 4)
    pos = _channel_pos_close(close)

    def _fcst_err(a):
        y, x_t = a[:-1], a[-1]
        fit = _reg_fit_window(y)
        if fit is None:
            return np.nan
        slope, intercept, _, _, _ = fit
        return float((x_t - (intercept + slope * float(y.size))) / x_t)

    return pos.rolling(w, min_periods=w).apply(_fcst_err, raw=True)

def reg_slope_tstat(close, window):
    """slope / SE(slope) of the trailing-window OLS of close on time — trend
    strength normalized by its own uncertainty (dimensionless).  Window ends
    at t and INCLUDES t (contemporaneous trend estimate, not a forecast);
    min_periods=window, NaN-in-window -> NaN; strict-positive close masking
    (R5-38); window >= 3 (2 degrees of freedom are consumed by the fit);
    zero residual variance or zero time variance -> NaN (fail-closed, never
    0)."""
    w = _pi(window, "window", 3)
    pos = _channel_pos_close(close)

    def _tstat(a):
        fit = _reg_fit_window(a)
        if fit is None:
            return np.nan
        slope, _, resid_std, _, slope_se = fit
        if not (np.isfinite(slope_se) and slope_se > 0.0):
            return np.nan
        return float(slope / slope_se)

    return pos.rolling(w, min_periods=w).apply(_tstat, raw=True)

def reg_r2_trailing(close, window):
    """R^2 of the trailing-window OLS of close on time — [0,1] trend-linearity
    measure (dimensionless by construction).  Window ends at t and includes t
    (contemporaneous fit, not look-ahead); min_periods=window, NaN-in-window
    -> NaN; strict-positive close masking (R5-38); zero total variation
    (constant window) -> NaN (degenerate denominator, never a fabricated
    1.0/0.0).  Distinct from the general ts_regression_r2: close-only,
    fail-closed masking, and promoted on the technical surface."""
    w = _pi(window, "window", 3)
    pos = _channel_pos_close(close)

    def _r2(a):
        fit = _reg_fit_window(a)
        if fit is None:
            return np.nan
        return fit[3]

    return pos.rolling(w, min_periods=w).apply(_r2, raw=True)

def reg_residual_zscore(close, window):
    """(close_t - fitted_t) / trailing residual std — contemporaneous
    detrended-level z-score.  CAUSALITY NOTE (documented, not look-ahead):
    the fitted value at t comes from the OLS fit over the window ENDING at t
    — the fit includes the current bar, which makes the output
    contemporaneous, not future-looking; no data after t enters anywhere.
    The z-score denominator is the ddof=1 residual std of the SAME window;
    zero residual std (perfect line) -> NaN (degenerate, never 0);
    min_periods=window, NaN-in-window -> NaN; strict-positive close masking
    (R5-38).  Dimensionless by construction."""
    w = _pi(window, "window", 3)
    pos = _channel_pos_close(close)

    def _z(a):
        fit = _reg_fit_window(a)
        if fit is None:
            return np.nan
        _, intercept, resid_std, _, _ = fit
        if not (np.isfinite(resid_std) and resid_std > 0.0):
            return np.nan
        fitted_t = intercept + float(a.size - 1) * fit[0]
        return float((a[-1] - fitted_t) / resid_std)

    return pos.rolling(w, min_periods=w).apply(_z, raw=True)

# --- Causal intraday BVC / VPIN family (R20-BVC-VPIN-DIRECTUSE) ---
# Duplicate audit (survey of cleaned_operators/ BEFORE implementing):
# * micro_vpin (microstructure/ops.py, experimental "legacy_proxy"): rolling
#   sum(|r|*volume)/rolling sum(volume) — a volume-weighted |return| INTENSITY
#   proxy, explicitly documented as NOT strict VPIN (no equal-volume buckets,
#   no buy/sell classification) -> different estimator, untouched.  The landed
#   vpin_pct is the first genuine equal-volume-bucket VPIN on this surface.
# * micro_bvc_vpin (microstructure/, P2/research-only, kept out of production
#   by production_hardening): panel-level flow_impact pipeline — different
#   input/output shape and research status -> not a duplicate canonical; the
#   causal replacement already maps it to intraday_bvc_imbalance.
# * intraday_bvc_imbalance (microstructure/flow_impact.py, promoted in the
#   chip-flow pack): minute-PANEL per-(date,symbol) daily aggregate using the
#   2*Phi(z)-1 CDF flow with a rolling volatility scale and lock-marker
#   neutralization -> different estimator (probabilistic, not sign), different
#   shape (panel -> one scalar per day).  The landed bvc_sign_pct is a
#   series-level SIGN-weighted trailing-window canonical -> not a duplicate.
# * micro_trade_imbalance (microstructure/ops.py, experimental): session-aware
#   rolling sum(sign(r)*volume)/rolling sum(volume) with min_periods != window
#   and NO ParamSpec -> near-duplicate in spirit, different contract (session
#   resets + partial windows, no fail-closed masking, no promotion surface).
#   Documented; not skipped here (different estimator contract & surface).
# * signed_volume_imbalance (price_volume/liquidity_v2.py): up-volume share
#   minus down-volume share — SHARE-based (per-bar direction counted equally),
#   not volume-magnitude-weighted -> different estimator.
# Family contract: trailing windows END at t, min_periods=window, any
# NaN/invalid bar inside the window -> NaN (fail-closed, never zero-filled or
# silently reconnected); strict-positive close AND strict-positive volume
# masking (R5-38: non-positive volume is bad data, masked to NaN, never
# abs()-laundered); zero-volume window -> NaN; dimensionless outputs
# ([-1,1] / [0,1] / [-1,1]-scaled); bvc_sign_pct and vpin_pct are rolling-only
# bounded state (NOT in _RECURSIVE_EWM); bvc_imbalance_ma uses EMAs -> IS in
# _RECURSIVE_EWM (stateful/full_replay governance).
def _bvc_valid(close, volume):
    """Strict valid-price/valid-volume mask for the BVC family.

    A bar is classifiable only when BOTH close_t and close_{t-1} are
    strict-positive finite (a formed, sign-able price change) and volume_t is
    strict-positive finite (zero/negative volume is bad data — masked to NaN,
    never abs()-laundered; R5-38 applied to the volume input)."""
    prev = close.shift(1)
    return (
        close.notna() & (close > 0.0)
        & prev.notna() & (prev > 0.0)
        & volume.notna() & (volume > 0.0)
    )

def _bvc_flow_series(close, volume):
    """sign(close_t - close_{t-1}) * volume_t on the strict-valid cohort."""
    valid = _bvc_valid(close, volume)
    diff = close - close.shift(1)
    return np.sign(diff).where(valid) * volume.where(valid), valid

def bvc_sign_pct(close, volume, window):
    """sum(sign(Δclose)*volume) / sum(volume) over the trailing window.

    Signed Bulk-Volume-Classification flow as a FRACTION of total volume —
    dimensionless in [-1, 1]: +1 = every share bought, -1 = every share sold,
    0 = perfectly balanced.  Rolling-only (bounded state); window ends at t,
    min_periods=window; strict-positive close + strict-positive volume masked
    (bad bar -> NaN, fail-closed); zero-volume window -> NaN."""
    w = _pi(window, "window", 2)
    flow, valid = _bvc_flow_series(close, volume)
    num = flow.rolling(w, min_periods=w).sum()
    den = volume.where(valid).rolling(w, min_periods=w).sum()
    return num / den.where(den > 0.0)

def _vpin_window(a_c, a_v, buckets):
    """Equal-volume-bucket VPIN of one trailing window.

    ``a_c`` / ``a_v`` are the aligned close / volume slices (length == window,
    pre-validated finite & positive).  Bars are concatenated into ``buckets``
    equal-volume buckets (a bar straddling a boundary is split proportionally
    by volume); per-bucket signed flow is the BVC sign flow apportioned with
    the split; VPIN = mean |bucket imbalance| / bucket volume, which for equal
    buckets equals sum_b |OF_b| / total volume — dimensionless in [0, 1]."""
    flow = np.sign(np.diff(a_c)) * a_v[1:]
    vol = a_v[1:]
    total_v = float(vol.sum())
    if not (total_v > 0.0):
        return np.nan
    b = max(2, int(buckets))
    target = total_v / b
    capacity = target
    bucket_flow = 0.0
    abs_of = 0.0
    for m in range(vol.shape[0]):
        vm = float(vol[m])
        ofm = float(flow[m])
        if vm <= 0.0:
            continue
        # the flow of a split bar is apportioned against the bar's OWN volume
        # (ofm * take/vm), never against the shrinking remainder — the split
        # shares must sum back to the bar's original flow exactly.
        v_left = vm
        while v_left > 0.0:
            take = min(v_left, capacity)
            if take <= 0.0:
                break  # defensive: no progress, avoid an infinite loop
            bucket_flow += ofm * (take / vm)
            v_left -= take
            capacity -= take
            if capacity <= 0.0:
                abs_of += abs(bucket_flow)
                bucket_flow = 0.0
                capacity = target
    if capacity < target:  # close the final partial bucket
        abs_of += abs(bucket_flow)
    return float(abs_of / total_v)

def vpin_pct(close, volume, window, buckets):
    """Volume-Synchronized (equal-volume-bucket) VPIN — trailing window of
    bars bucketed into ``buckets`` equal-volume buckets, VPIN =
    mean per-bucket |bulk-direction imbalance| / bucket volume (==
    sum_b |OF_b| / total volume), dimensionless in [0, 1]; HIGH = toxic
    order flow (one-sided buckets), LOW = balanced.  CAUSAL: only the
    trailing window ending at t enters the bucketing.  Rolling-only
    (bounded state); min_periods=window; strict-positive close + volume
    masked (bad bar -> NaN, fail-closed); zero-volume window -> NaN;
    buckets >= 2 (a single bucket is the whole window — degenerate)."""
    w = _pi(window, "window", 2)
    b = _pi(buckets, "buckets", 2)
    # Positional numpy kernel: close and volume are bucketed TOGETHER over the
    # same trailing window; a window is emitted only when all w closes AND all
    # w volumes are finite and strict-positive (fail-closed — a NaN or
    # non-positive volume inside the window must NaN the output, never shrink
    # the buckets or zero-fill the flow).
    c_arr = close.to_numpy(float)
    v_arr = volume.to_numpy(float)
    out = np.full(c_arr.shape, np.nan, dtype=float)
    for j in range(c_arr.shape[1]):
        for t in range(c_arr.shape[0]):
            if t < w - 1:
                continue
            cs = c_arr[t - w + 1 : t + 1, j]
            vs = v_arr[t - w + 1 : t + 1, j]
            if not (np.isfinite(cs).all() and np.isfinite(vs).all()):
                continue
            if np.any(cs <= 0.0) or np.any(vs <= 0.0):
                continue
            if float(vs.sum()) <= 0.0:
                continue
            out[t, j] = _vpin_window(cs, vs, b)
    return pd.DataFrame(out, index=close.index, columns=close.columns)

def bvc_imbalance_ma(close, volume, fast_window, slow_window):
    """(EMA_fast(flow) - EMA_slow(flow)) / EMA_slow(volume) — fast/slow EMA
    spread of the BVC signed flow normalized by a slow EMA of total volume.
    Dimensionless momentum of the signed-flow intensity; EMA-based
    (recursive/full-history dependence -> ``stateful`` + ``full_replay``
    governance via _RECURSIVE_EWM).  fast_window < slow_window (relational
    spec — the swap is a duplicate search node); strict-positive close +
    strict-positive volume masking (bad bar -> NaN on the flow and volume
    cohorts); EMA(volume) <= 0 (e.g. warmup all-NaN cohort) -> NaN."""
    f = _pi(fast_window, "fast_window", 1)
    s = _pi(slow_window, "slow_window", 1)
    if f >= s:
        raise ValueError("fast_window must be < slow_window")
    flow, valid = _bvc_flow_series(close, volume)
    num = _ema(flow, f) - _ema(flow, s)
    den = _ema(volume.where(valid), s).where(lambda x: x > 0.0)
    return num / den

_SPECS=[
("DMI_plus",["high","low","close","window"],DMI_plus,"Wilder positive directional indicator."),
("DMI_minus",["high","low","close","window"],DMI_minus,"Wilder negative directional indicator."),
("DX",["high","low","close","window"],DX,"Directional movement index before ADX smoothing."),
("NATR",["high","low","close","window"],NATR,"Normalized Wilder ATR as percent of close."),
("atr_pct",["high","low","close","window"],atr_pct,"Wilder ATR as a fraction of close."),
("atr_zscore",["high","low","close","window","score_window"],atr_zscore,"Rolling z-score of ATR / close."),
("atr_percentile",["high","low","close","window","score_window"],atr_percentile,"Trailing percentile rank of ATR / close."),
("true_range_pct",["high","low","close"],true_range_pct,"True range as a fraction of previous close."),
("true_range_surprise",["high","low","close","window"],true_range_surprise,"True-range ratio vs trailing mean, minus 1."),
("true_range_zscore",["high","low","close","window"],true_range_zscore,"Trailing z-score of the true-range ratio."),
("atr_short_long_ratio",["high","low","close","short_window","long_window"],atr_short_long_ratio,"Short ATR ratio over long ATR ratio."),
("atr_acceleration",["high","low","close","window"],atr_acceleration,"First difference of ATR / close."),
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
("keltner_width_pct",["high","low","close","ema_window","atr_window","multiplier"],keltner_width_pct,"Keltner band width as a fraction of the mid line (dimensionless)."),
("keltner_compression",["high","low","close","ema_window","atr_window","multiplier","score_window"],keltner_compression,"Trailing percentile rank of the Keltner width ratio; low = compressed."),
("keltner_breakout_strength",["high","low","close","ema_window","atr_window","multiplier"],keltner_breakout_strength,"Signed breakout distance from the Keltner band, normalized by band width."),
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
("psar_direction",["high","low","close","acceleration","maximum"],psar_direction,"PSAR regime sign (+1 bull / -1 bear); recursive/stateful-derived, dimensionless."),
("psar_distance_pct",["high","low","close","acceleration","maximum"],psar_distance_pct,"(close - PSAR) / close; dimensionless, strict-positive close masked."),
("psar_flip",["high","low","close","acceleration","maximum"],psar_flip,"PSAR regime flip (+1 up / -1 down / 0 hold); recursive/stateful-derived."),
("psar_days_since_flip",["high","low","close","acceleration","maximum"],psar_days_since_flip,"Bars since the last PSAR regime flip; recursive/stateful-derived."),
("supertrend_direction",["high","low","close","atr_window","multiplier"],supertrend_direction,"Supertrend regime sign (+1 at/above the line / -1 below); recursive/stateful-derived, dimensionless."),
("supertrend_distance_pct",["high","low","close","atr_window","multiplier"],supertrend_distance_pct,"(close - Supertrend line) / close; dimensionless, strict-positive close masked."),
("supertrend_flip",["high","low","close","atr_window","multiplier"],supertrend_flip,"Supertrend regime flip (+1 up / -1 down / 0 hold); recursive/stateful-derived."),
("supertrend_days_since_flip",["high","low","close","atr_window","multiplier"],supertrend_days_since_flip,"Bars since the last Supertrend regime flip; recursive/stateful-derived."),
("donchian_width_pct",["high","low","close","window"],donchian_width_pct,"Donchian channel width as a fraction of close (dimensionless, strict-positive close masked)."),
("donchian_channel_position",["high","low","close","window"],donchian_channel_position,"Close position inside the trailing Donchian channel [0,1] when within bands."),
("donchian_breakout_up",["high","low","close","window"],donchian_breakout_up,"Fresh N-bar-high breakout magnitude (close/rolling-max high - 1); 0 unless close makes a new high; window ends at t."),
("donchian_breakout_down",["high","low","close","window"],donchian_breakout_down,"Fresh N-bar-low breakdown magnitude (rolling-min low/close - 1); 0 unless close makes a new low; window ends at t."),
("ema_distance_pct",["close","window"],ema_distance_pct,"(close - EMA) / close; dimensionless, strict-positive close masked."),
("sma_distance_pct",["close","window"],sma_distance_pct,"(close - rolling SMA) / close; dimensionless, window ends at t."),
("dema_distance_pct",["close","window"],dema_distance_pct,"(close - DEMA) / close; DEMA = 2*EMA - EMA(EMA); dimensionless."),
("tema_distance_pct",["close","window"],tema_distance_pct,"(close - TEMA) / close; TEMA = 3EMA - 3EMA(EMA) + EMA(EMA(EMA)); dimensionless."),
("kama_distance_pct",["close","er_window","fast_window","slow_window"],kama_distance_pct,"(close - registered KAMA) / close; dimensionless; reuses the canonical KAMA implementation."),
("ma_slope_pct",["close","window"],ma_slope_pct,"EMA.diff() / close; dimensionless per-bar MA slope, strict-positive close masked."),
("ema_crossover",["close","fast_window","slow_window"],ema_crossover,"sign((fast EMA - slow EMA) / close) bounded {-1,0,+1}; fast_window < slow_window."),
("tenkan_kijun_cross",["high","low","close","tenkan_window","kijun_window"],tenkan_kijun_cross,"sign((Tenkan - Kijun) / close) bounded {-1,0,+1}; trailing windows end at t; NO future shift (Senkou displacement is display-only)."),
("chikou_distance_pct",["high","low","close","kijun_window"],chikou_distance_pct,"(close - Kijun) / close; causal equilibrium-line distance (classic Chikou span-shift is future-looking and NOT used)."),
("senkou_span_causal_pct",["high","low","close","tenkan_window","kijun_window"],senkou_span_causal_pct,"((Tenkan_t + Kijun_t)/2 - close) / close; causal cloud-mid vs price gap, no shift(26)."),
("vwap_distance_pct",["close","volume","window"],vwap_distance_pct,"(close - rolling VWAP) / close; dimensionless, strict-positive close + valid-volume masked, window ends at t."),
("vwap_slope_pct",["close","volume","window"],vwap_slope_pct,"rolling VWAP.diff() / close; dimensionless per-bar VWAP slope, strict-positive close masked."),
("vwap_premium_pct",["high","low","close","volume","window"],vwap_premium_pct,"(close - HLC3 rolling VWAP) / close; typical-price VWAP premium, strict-positive close + valid-volume masked."),
("candle_body_strength",["open","high","low","close","window"],candle_body_strength,"Mean signed body/range over the trailing window; aggregate directional candle strength in [-1,1], bad bars excluded."),
("candle_wick_balance",["open","high","low","close","window"],candle_wick_balance,"Mean (lower_wick - upper_wick)/range over the trailing window; buying-vs-selling pressure proxy in [-1,1]."),
("candle_range_pct",["open","high","low","close","window"],candle_range_pct,"Mean (high-low)/close over the trailing window; dimensionless bar-range volatility, no Wilder smoothing."),
("candle_pattern_count",["open","high","low","close","window"],candle_pattern_count,"Fraction of valid trailing-window bars classified as doji / one-sided-rejection patterns; bounded [0,1], bad bars excluded."),
("sr_distance_pct",["high","low","close","window"],sr_distance_pct,"(close - prior-window median typical-price pivot) / close; dimensionless, prior window ends at t-1, strict-positive close masked."),
("sr_touch_count",["high","low","close","window","tol"],sr_touch_count,"Fraction of the trailing window whose high/low came within tol (relative) of the prior typical-price pivot; bounded [0,1], level-importance measure."),
("consolidation_pct",["close","window"],consolidation_pct,"rolling_std(close,w)/rolling_mean(close,w) — coefficient of variation, dimensionless tightness; low = consolidation, window ends at t, strict-positive masked."),
("consolidation_range_pct",["close","window"],consolidation_range_pct,"(rolling_max(close,w)-rolling_min(close,w))/close — close-only range tightness, dimensionless; low = tight consolidation, window ends at t."),
("spectral_energy_ratio",["close","window","top_k"],spectral_energy_ratio,"Top-k rfft-bin share of the de-meaned trailing window's non-DC spectral energy; dimensionless [0,1], high = single-frequency dominance (trend/cycle), low = noise."),
("spectral_trend_share",["close","window","trend_bins"],spectral_trend_share,"Lowest-frequency-bin share of the de-meaned trailing window's non-DC rfft energy; dimensionless [0,1], high = trend-dominated, low = noise-dominated."),
("wavelet_detail_energy_ratio",["close","window"],wavelet_detail_energy_ratio,"Haar multilevel detail-energy share of the de-meaned trailing window (power-of-2 window); dimensionless [0,1], high = choppy, low = smooth/trend."),
("reg_forecast_error_pct",["close","window"],reg_forecast_error_pct,"One-step-ahead OLS time-trend forecast error normalized by close; fit window t-w..t-1 EXCLUDES the target bar; dimensionless, strict-positive close masked."),
("reg_slope_tstat",["close","window"],reg_slope_tstat,"Trailing OLS time-trend slope / SE(slope); trend strength in units of its own uncertainty, dimensionless, fail-closed masking."),
("reg_r2_trailing",["close","window"],reg_r2_trailing,"Trailing OLS time-trend R^2 in [0,1]; linearity of the trend, dimensionless, constant window -> NaN."),
("reg_residual_zscore",["close","window"],reg_residual_zscore,"Contemporaneous residual / trailing residual std from the OLS fit whose window ends at t (includes t — contemporaneous, not look-ahead); dimensionless."),
("bvc_sign_pct",["close","volume","window"],bvc_sign_pct,"sum(sign(dclose)*volume)/sum(volume) over the trailing window; signed BVC flow fraction in [-1,1], strict-positive close+volume masked, fail-closed."),
("vpin_pct",["close","volume","window","buckets"],vpin_pct,"Equal-volume-bucket VPIN (mean |bucket flow|/bucket volume) over the trailing window; order-flow toxicity in [0,1], fail-closed masking."),
("bvc_imbalance_ma",["close","volume","fast_window","slow_window"],bvc_imbalance_ma,"(EMA_fast - EMA_slow) of sign(dclose)*volume flow divided by EMA_slow(volume); dimensionless signed-flow momentum, EMA-based (stateful/full_replay)."),
]
# Recursive EMA / Wilder families: every output depends on the full history
# through ``ewm(adjust=False)`` (``_ema``/``_wilder``), so a
# ``start_date + finite warmup`` run is NOT bit-exact with a full-history run.
# They must be governed ``stateful`` + ``full_replay`` exactly like
# KAMA/Supertrend/PSAR (review P0-05).  Bounded rolling-sum indicators (CMO,
# Vortex, UltimateOscillator, ichimoku) stay unbounded-state-free.
_RECURSIVE_EWM = {
    "DMI_plus", "DMI_minus", "DX", "NATR",
    "atr_pct", "atr_zscore", "atr_percentile",
    "atr_short_long_ratio", "atr_acceleration",
    "PPO", "PPO_signal", "PPO_hist", "PVO", "PVO_signal", "PVO_hist",
    "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition",
    "keltner_width_pct", "keltner_compression", "keltner_breakout_strength",
    "TSI", "TSI_signal", "DEMA", "TEMA",
    # R20-MA-DISTANCE-SLOPE: EMA/DEMA/TEMA/KAMA-recursive distance/slope/crossover
    # (sma_distance_pct is rolling-only, bounded state — NOT in this set).
    "ema_distance_pct", "dema_distance_pct", "tema_distance_pct",
    "ma_slope_pct", "ema_crossover",
    # R20-SUPERTREND-DIRECTUSE: all four derivatives sit on the recursive
    # Supertrend state machine (Wilder ATR full-history dependence).
    "supertrend_direction", "supertrend_distance_pct",
    "supertrend_flip", "supertrend_days_since_flip",
    # R20-BVC-VPIN-DIRECTUSE: only the EMA-spread member is recursive
    # (bvc_sign_pct / vpin_pct are rolling-only bounded state).
    "bvc_imbalance_ma",
}
# R5-35/36 + round-11 P0 (search-space contracts): EVERY parameter of the
# technical indicators below gets a formal ``ParamSpec`` — never a silent
# ``_pi`` float truncation.  A window declared ``dtype=int`` rejects 20.1/20.5/
# 20.9 at the call boundary (they all used to truncate to the SAME 20, a fake
# search space); a multiplier/acceleration is a strict positive float.
# R30 §24 (P1-025): production scalar parameters carry an explicit ParamRole.
# Window / horizon knobs are HORIZON (searchable on a coarse grid); a
# multiplier / band-width float is a STATE_THRESHOLD (a real rule dimension,
# not an estimator epsilon).  Shared helpers here fix the whole technical
# family at once instead of leaving 300+ scalars to the ECONOMIC fallback.
_WIN_GE2 = ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON)
_WIN_GE4 = ParamSpec(dtype=int, min=4, param_role=ParamRole.HORIZON)
_WIN_GE1 = ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON)
# R20-REGRESSION-CAUSAL-DIRECTUSE: OLS slope+intercept consume 2 dof.
_WIN_GE3 = ParamSpec(dtype=int, min=3, param_role=ParamRole.HORIZON)
_POS_FLOAT = ParamSpec(dtype=float, min=1e-9, param_role=ParamRole.STATE_THRESHOLD)


def _rel(expression, message):
    return RelationalParamSpec(expression=expression, message=message)


_PARAM_SPECS = {
    "DMI_plus": {"window": _WIN_GE2},
    "DMI_minus": {"window": _WIN_GE2},
    "DX": {"window": _WIN_GE2},
    "NATR": {"window": _WIN_GE2},
    "atr_pct": {"window": _WIN_GE2},
    "atr_zscore": {"window": _WIN_GE2, "score_window": _WIN_GE2},
    "atr_percentile": {"window": _WIN_GE2, "score_window": _WIN_GE2},
    "true_range_surprise": {"window": _WIN_GE2},
    "true_range_zscore": {"window": _WIN_GE2},
    "atr_short_long_ratio": {"short_window": _WIN_GE2, "long_window": _WIN_GE2},
    "atr_acceleration": {"window": _WIN_GE2},
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
    "keltner_width_pct": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT},
    "keltner_compression": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT, "score_window": _WIN_GE2},
    "keltner_breakout_strength": {"ema_window": _WIN_GE1, "atr_window": _WIN_GE1, "multiplier": _POS_FLOAT},
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
    "psar_direction": {"acceleration": _POS_FLOAT, "maximum": _POS_FLOAT},
    "psar_distance_pct": {"acceleration": _POS_FLOAT, "maximum": _POS_FLOAT},
    "psar_flip": {"acceleration": _POS_FLOAT, "maximum": _POS_FLOAT},
    "psar_days_since_flip": {"acceleration": _POS_FLOAT, "maximum": _POS_FLOAT},
    # R20-SUPERTREND-DIRECTUSE: same param contract as Supertrend itself.
    "supertrend_direction": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "supertrend_distance_pct": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "supertrend_flip": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "supertrend_days_since_flip": {"atr_window": _WIN_GE2, "multiplier": _POS_FLOAT},
    "donchian_width_pct": {"window": _WIN_GE2},
    "donchian_channel_position": {"window": _WIN_GE2},
    "donchian_breakout_up": {"window": _WIN_GE2},
    "donchian_breakout_down": {"window": _WIN_GE2},
    # R20-MA-DISTANCE-SLOPE: windows follow the family precedent — EMA-family
    # knobs are _WIN_GE1 (matches DEMA/TEMA/KAMA fast/slow), rolling SMA needs
    # a real window (>= 2) to be distinct from the raw close.
    "ema_distance_pct": {"window": _WIN_GE1},
    "sma_distance_pct": {"window": _WIN_GE2},
    "dema_distance_pct": {"window": _WIN_GE1},
    "tema_distance_pct": {"window": _WIN_GE1},
    "kama_distance_pct": {"er_window": _WIN_GE2, "fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
    "ma_slope_pct": {"window": _WIN_GE1},
    "ema_crossover": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
    # R20-ICHIMOKU-DIRECTUSE: trailing midpoint windows need >= 2 bars to be
    # distinct from the raw bar (matches the ichimoku_* family precedent).
    "tenkan_kijun_cross": {"tenkan_window": _WIN_GE2, "kijun_window": _WIN_GE2},
    "chikou_distance_pct": {"kijun_window": _WIN_GE2},
    "senkou_span_causal_pct": {"tenkan_window": _WIN_GE2, "kijun_window": _WIN_GE2},
    # R20-VWAP-DIRECTUSE: rolling VWAP needs a real window (>= 2, same as the
    # registered rolling_vwap contract in price_volume/technical_extensions).
    "vwap_distance_pct": {"window": _WIN_GE2},
    "vwap_slope_pct": {"window": _WIN_GE2},
    "vwap_premium_pct": {"window": _WIN_GE2},
    # R20-CANDLESTICK-DIRECTUSE: aggregate trailing window (>= 2, same contract
    # as the vwap_/donchian_ rolling families).
    "candle_body_strength": {"window": _WIN_GE2},
    "candle_wick_balance": {"window": _WIN_GE2},
    "candle_range_pct": {"window": _WIN_GE2},
    "candle_pattern_count": {"window": _WIN_GE2},
    # R20-SR-DIRECTUSE: pivot window needs >= 2 bars (same contract as the
    # vwap_/donchian_/candle_ rolling families); tol is a strict positive
    # relative tolerance (STATE_THRESHOLD).
    "sr_distance_pct": {"window": _WIN_GE2},
    "sr_touch_count": {"window": _WIN_GE2, "tol": _POS_FLOAT},
    # R20-CHANNEL-CONSOLIDATION-DIRECTUSE: single trailing window (>= 2 bars
    # to be distinct from the raw close), same contract as the vwap_/donchian_/
    # candle_ rolling families.  No relational spec (single window knob).
    "consolidation_pct": {"window": _WIN_GE2},
    "consolidation_range_pct": {"window": _WIN_GE2},
    # R20-SPECTRAL-EXCESS-DIRECTUSE: rolling-only trailing rfft/Haar windows
    # (NOT recursive) — fail-closed NaN-in-window via min_periods=window.
    # top_k / trend_bins are structural constants of the measure, not tuned
    # look-ahead knobs; searchable=False keeps them out of the alpha grammar.
    # window min=4 matches the runtime _pi(window,"window",4) guard in
    # _spectral_energy — the rfft/Haar decomposition is meaningless below 4
    # samples, so the spec prunes those search nodes up front (review P1).
    "spectral_energy_ratio": {
        "window": _WIN_GE4,
        "top_k": ParamSpec(dtype=int, min=1, searchable=False, param_role=ParamRole.STATE_THRESHOLD),
    },
    "spectral_trend_share": {
        "window": _WIN_GE4,
        "trend_bins": ParamSpec(dtype=int, min=1, searchable=False, param_role=ParamRole.STATE_THRESHOLD),
    },
    # Power-of-2 is enforced at the call boundary (``_pow2`` ValueError);
    # ParamSpec has no power-of-2 validator, so the spec keeps min=4 (the
    # family runtime minimum — prunes window=2/3 search nodes up front).
    "wavelet_detail_energy_ratio": {"window": _WIN_GE4},
    # R20-REGRESSION-CAUSAL-DIRECTUSE: the OLS family consumes 2 degrees of
    # freedom (slope + intercept), so window >= 3 (matches the runtime
    # _pi(window,"window",3) guard); single window knob per canonical — no
    # relational spec (no ordered window pair).
    # forecast error's effective minimum is 4: the fit excludes the target
    # bar, so window=3 leaves a 2-bar fit (all-NaN by construction) — the
    # spec prunes that search node up front (review P1).
    "reg_forecast_error_pct": {"window": _WIN_GE4},
    "reg_slope_tstat": {"window": _WIN_GE3},
    "reg_r2_trailing": {"window": _WIN_GE3},
    "reg_residual_zscore": {"window": _WIN_GE3},
    # R20-BVC-VPIN-DIRECTUSE: windows follow the rolling-family contract
    # (>= 2 bars — the flow needs one formed price change); buckets >= 2
    # (a single bucket is the whole window, degenerate); the EMA-spread
    # member follows the PPO/KAMA fast/slow contract (>= 1 each + the
    # fast<slow relational spec below).
    "bvc_sign_pct": {"window": _WIN_GE2},
    "vpin_pct": {"window": _WIN_GE2, "buckets": _WIN_GE2},
    "bvc_imbalance_ma": {"fast_window": _WIN_GE1, "slow_window": _WIN_GE1},
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
_ATR_SHORT_LONG_REL = [
    _rel("short_window < long_window", "require short_window < long_window (short={short_window}, long={long_window})")
]
_ICHI_REL = [
    _rel("tenkan_window < kijun_window", "require tenkan_window < kijun_window (tenkan={tenkan_window}, kijun={kijun_window})")
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
    "psar_direction": _PSAR_REL,
    "psar_distance_pct": _PSAR_REL,
    "psar_flip": _PSAR_REL,
    "psar_days_since_flip": _PSAR_REL,
    "atr_short_long_ratio": _ATR_SHORT_LONG_REL,
    # R20-MA-DISTANCE-SLOPE: crossover fast/slow swap creates duplicate search
    # nodes; same relational contract family as PPO/KAMA.
    "ema_crossover": _PPO_REL,
    # Scoped-review F1 (P1): kama_distance_pct delegates to KAMA, whose own
    # ValueError guard fires only at calculate time; the relational spec prunes
    # the fast>=slow search nodes up front, mirroring "KAMA": _PPO_REL.
    "kama_distance_pct": _PPO_REL,
    # R20-ICHIMOKU-DIRECTUSE: Tenkan/Kijun window swap creates a duplicate
    # search node (the classic Tenkan is by definition the faster line); same
    # relational contract family as PPO/KAMA/ema_crossover.
    "tenkan_kijun_cross": _ICHI_REL,
    "senkou_span_causal_pct": _ICHI_REL,
    # R20-BVC-VPIN-DIRECTUSE: fast/slow EMA window swap is a duplicate search
    # node — same relational contract family as PPO/KAMA/ema_crossover.
    "bvc_imbalance_ma": _PPO_REL,
}
for _name,_params,_fn,_desc in _SPECS:
    _tags=("stateful","full_replay") if _name in _RECURSIVE_EWM or _name in {"KAMA","Supertrend","SupertrendDirection","PSAR","psar_direction","psar_distance_pct","psar_flip","psar_days_since_flip","kama_distance_pct"} else ()
    # KAMA overrides the composite_fastpath_primitives implementation; pin the
    # exact source it replaces so the chain is independent of import order
    # (round-7 P0).  Other spec entries register fresh — no pin needed.
    _pin = "composite_fastpath_primitives" if _name == "KAMA" else ""
    _register(_name,_params,_fn,_desc,tags=_tags,param_specs=_PARAM_SPECS.get(_name),
              relational_specs=_RELATIONAL_SPECS.get(_name),expected_old_source=_pin)
