# -*- coding: utf-8 -*-
"""Causal bounded liquidity, price-impact and price/volume interaction operators."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry

_EPS=1e-12

def _pi(v,name,minimum=1):
    if isinstance(v,bool): raise ValueError(f"{name} must be integer")
    v=int(v)
    if v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v

def _register(name,params,fn,desc):
    # ChaikinOscillator / ForceIndex smooth via ``ewm(adjust=False)`` (infinite
    # recursion) and therefore require full-history replay — review P0-05.
    _tags=("stateful","full_replay") if name in {"ChaikinOscillator","ForceIndex"} else ()
    meta=OperatorMetadata(name=name,category="price_volume_extension",description=desc,param_names=list(params),return_type="series",tags=["pit_safe","causal","bounded_history","production_extension",*_tags])
    def _calculate_series(self,*args,**kwargs): return fn(*args,**kwargs)
    cls=type(f"LiquidityV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="price_volume_extension",business_category="price_volume",canonical=name,source="liquidity_v2",backend="pandas_numpy",status="production")(cls)

def average_volume(volume,window):
    w=_pi(window,"window"); return volume.rolling(w,min_periods=w).mean()
def average_turnover(turnover,window): return average_volume(turnover,window)
def adv(close,volume,window):
    w=_pi(window,"window"); return (close.abs()*volume).rolling(w,min_periods=w).mean()
def abnormal_volume(volume,window):
    base=volume.shift(1).rolling(_pi(window,"window"),min_periods=_pi(window,"window")).mean(); return volume/base.replace(0,np.nan)-1.0
def abnormal_turnover(turnover,window): return abnormal_volume(turnover,window)
def volume_volatility(volume,window):
    w=_pi(window,"window",2); return volume.pct_change(fill_method=None).rolling(w,min_periods=w).std()
def turnover_volatility(turnover,window): return volume_volatility(turnover,window)
def volume_autocorr(volume,window,lag):
    w=_pi(window,"window",3); l=_pi(lag,"lag"); return volume.rolling(w,min_periods=w).corr(volume.shift(l))
def turnover_autocorr(turnover,window,lag): return volume_autocorr(turnover,window,lag)
def amihud_illiquidity(ret,close,volume,window):
    w=_pi(window,"window"); dv=(close.abs()*volume).replace(0,np.nan); raw=ret.abs()/dv; return raw.rolling(w,min_periods=w).mean()
def price_impact(ret,dollar_volume,window):
    w=_pi(window,"window"); return (ret.abs()/dollar_volume.replace(0,np.nan)).rolling(w,min_periods=w).mean()
def return_per_turnover(ret,turnover): return ret/turnover.replace(0,np.nan)
def _zscore(x,window):
    w=_pi(window,"window",2); mean=x.shift(1).rolling(w,min_periods=w).mean(); sd=x.shift(1).rolling(w,min_periods=w).std(); return (x-mean)/sd.replace(0,np.nan)
def volume_shock(volume,window): return _zscore(volume,window)
def turnover_shock(turnover,window): return _zscore(turnover,window)
def volume_acceleration(volume,short_window,long_window):
    s,l=_pi(short_window,"short_window"),_pi(long_window,"long_window");
    if s>=l: raise ValueError("short_window must be < long_window")
    return volume.rolling(s,min_periods=s).mean()/volume.rolling(l,min_periods=l).mean().replace(0,np.nan)-1.0
def turnover_acceleration(turnover,short_window,long_window): return volume_acceleration(turnover,short_window,long_window)
def up_volume_ratio(ret,volume,window):
    w=_pi(window,"window")
    # A bar whose return is missing/unknown must NOT be counted as "not up"
    # volume: the old ``where(ret>0, 0.0)`` mapped ``NaN -> False -> 0.0``,
    # silently deflating the up share (unknown != zero volume / zero direction).
    # Mask invalid rows to NaN so a window containing any missing input emits
    # NaN, and only a fully-known window with zero up-volume emits 0 — R4-29.
    valid=ret.notna()&volume.notna()
    up=volume.where((ret>0)&valid,0.0).where(valid)
    total=volume.abs().where(valid)
    return up.rolling(w,min_periods=w).sum()/total.rolling(w,min_periods=w).sum().replace(0,np.nan)
def down_volume_ratio(ret,volume,window):
    w=_pi(window,"window")
    # Same unknown != zero discipline as up_volume_ratio (R4-29).
    valid=ret.notna()&volume.notna()
    dn=volume.where((ret<0)&valid,0.0).where(valid)
    total=volume.abs().where(valid)
    return dn.rolling(w,min_periods=w).sum()/total.rolling(w,min_periods=w).sum().replace(0,np.nan)
def signed_volume_imbalance(ret,volume,window): return up_volume_ratio(ret,volume,window)-down_volume_ratio(ret,volume,window)
def up_down_volume_ratio(ret,volume,window): return up_volume_ratio(ret,volume,window)/down_volume_ratio(ret,volume,window).replace(0,np.nan)
def volume_weighted_return(ret,volume,window):
    w=_pi(window,"window")
    # Cohort-consistent numerator/denominator: a bar missing either input must
    # not enter one side and not the other (a missing ``ret`` with a valid
    # ``volume`` previously diluted the mean toward 0) — review §4 / P1.
    valid=ret.notna()&volume.notna(); rv=ret.where(valid); vv=volume.where(valid)
    num=(rv*vv).rolling(w,min_periods=w).sum(); den=vv.abs().rolling(w,min_periods=w).sum()
    return num/den.replace(0,np.nan)
def volume_weighted_momentum(close,volume,window): return volume_weighted_return(close.pct_change(fill_method=None),volume,window)
def price_volume_divergence(close,volume,price_window,volume_window):
    pw,vw=_pi(price_window,"price_window"),_pi(volume_window,"volume_window"); pr=close/close.shift(pw)-1.0; vr=volume/volume.shift(vw).replace(0,np.nan)-1.0; return pr-vr
def price_turnover_divergence(close,turnover,price_window,turnover_window): return price_volume_divergence(close,turnover,price_window,turnover_window)
def return_volume_beta(ret,volume,window):
    w=_pi(window,"window",3); vc=volume.pct_change(fill_method=None); cov=ret.rolling(w,min_periods=w).cov(vc); var=vc.rolling(w,min_periods=w).var(); return cov/var.replace(0,np.nan)
def return_turnover_beta(ret,turnover,window): return return_volume_beta(ret,turnover,window)
def _mf_multiplier(high,low,close): return ((close-low)-(high-close))/(high-low).replace(0,np.nan)
def ADL(high,low,close,volume,window):
    w=_pi(window,"window"); flow=_mf_multiplier(high,low,close)*volume; return flow.rolling(w,min_periods=w).sum()
def ChaikinOscillator(high,low,close,volume,fast_window,slow_window,adl_window):
    f,s=_pi(fast_window,"fast_window"),_pi(slow_window,"slow_window");
    if f>=s: raise ValueError("fast_window must be < slow_window")
    adl=ADL(high,low,close,volume,_pi(adl_window,"adl_window")); return adl.ewm(span=f,adjust=False,min_periods=f).mean()-adl.ewm(span=s,adjust=False,min_periods=s).mean()
def ForceIndex(close,volume,window):
    w=_pi(window,"window"); raw=close.diff()*volume; return raw.ewm(span=w,adjust=False,min_periods=w).mean()
def EaseOfMovement(high,low,volume,window,volume_scale=1.0):
    w=_pi(window,"window",2); midpoint=(high+low)/2.0; distance=midpoint.diff(); box=(high-low)/(volume.replace(0,np.nan)/float(volume_scale)); raw=distance*box; return raw.rolling(w,min_periods=w).mean()
def bounded_nvi(close,volume,window):
    w=_pi(window,"window",2)
    # A missing volume / missing volume-growth bar must not become a silent
    # "no-volume day" (the old ``where(cond, 0.0)`` mapped NaN comparisons to
    # False -> 0.0) — that injected fabricated zero-return bars into the index.
    r=close.pct_change(fill_method=None)
    cond=(volume<volume.shift(1)); valid=cond.notna()&r.notna()
    r=r.where(cond,0.0).where(valid)
    return np.exp(np.log1p(r.clip(lower=-0.999999)).rolling(w,min_periods=w).sum())-1.0
def bounded_pvi(close,volume,window):
    w=_pi(window,"window",2)
    r=close.pct_change(fill_method=None)
    cond=(volume>volume.shift(1)); valid=cond.notna()&r.notna()
    r=r.where(cond,0.0).where(valid)
    return np.exp(np.log1p(r.clip(lower=-0.999999)).rolling(w,min_periods=w).sum())-1.0
def zero_return_ratio(ret,window,epsilon=1e-12):
    w=_pi(window,"window")
    # Missing returns must be NaN (cannot judge), not counted as "non-zero":
    # ``NaN <= eps`` is False and previously inflated the ratio with a 0.
    valid=ret.notna(); ratio=ret.abs().le(float(epsilon)).astype(float).where(valid)
    return ratio.rolling(w,min_periods=w).mean()
def roll_spread_proxy(ret,window):
    w=_pi(window,"window",3); cov=ret.rolling(w,min_periods=w).cov(ret.shift(1)); return 2.0*np.sqrt((-cov).clip(lower=0.0))
def corwin_schultz_spread(high,low,window):
    w=_pi(window,"window",2); h=high.replace(0,np.nan); l=low.replace(0,np.nan); loghl=np.log(h/l)
    beta=loghl.pow(2)+loghl.shift(1).pow(2); high2=pd.DataFrame(np.maximum(h.to_numpy(float),h.shift(1).to_numpy(float)),index=h.index,columns=h.columns); low2=pd.DataFrame(np.minimum(l.to_numpy(float),l.shift(1).to_numpy(float)),index=l.index,columns=l.columns); gamma=np.log(high2/low2.replace(0,np.nan)).pow(2)
    den=3.0-2.0*np.sqrt(2.0); alpha=(np.sqrt(2.0*beta)-np.sqrt(beta))/den-np.sqrt(gamma/den); alpha=alpha.clip(lower=0.0); spread=2.0*(np.exp(alpha)-1.0)/(1.0+np.exp(alpha)); return spread.rolling(w,min_periods=w).mean()
def high_low_spread_proxy(high,low,window):
    w=_pi(window,"window"); return np.log(high/low.replace(0,np.nan)).rolling(w,min_periods=w).mean()
def turnover_adjusted_volatility(ret,turnover,window):
    w=_pi(window,"window",2); vol=ret.rolling(w,min_periods=w).std(); act=turnover.rolling(w,min_periods=w).mean(); return vol/act.replace(0,np.nan)
def volume_to_range(volume,high,low,window):
    w=_pi(window,"window"); raw=volume/(high-low).replace(0,np.nan); return raw.rolling(w,min_periods=w).mean()

_SPECS=[
("average_volume",["volume","window"],average_volume,"Rolling average volume."),("average_turnover",["turnover","window"],average_turnover,"Rolling average turnover/activity."),("adv",["close","volume","window"],adv,"Average dollar volume."),
("abnormal_volume",["volume","window"],abnormal_volume,"Current volume versus prior rolling mean."),("abnormal_turnover",["turnover","window"],abnormal_turnover,"Current turnover versus prior rolling mean."),("volume_volatility",["volume","window"],volume_volatility,"Volatility of volume growth."),("turnover_volatility",["turnover","window"],turnover_volatility,"Volatility of turnover growth."),
("volume_autocorr",["volume","window","lag"],volume_autocorr,"Rolling volume autocorrelation."),("turnover_autocorr",["turnover","window","lag"],turnover_autocorr,"Rolling turnover autocorrelation."),("amihud_illiquidity",["ret","close","volume","window"],amihud_illiquidity,"Rolling Amihud illiquidity proxy."),("price_impact",["ret","dollar_volume","window"],price_impact,"Absolute return per dollar-volume price-impact proxy."),("return_per_turnover",["ret","turnover"],return_per_turnover,"Return per unit turnover."),
("volume_shock",["volume","window"],volume_shock,"Prior-window volume z-score."),("turnover_shock",["turnover","window"],turnover_shock,"Prior-window turnover z-score."),("volume_acceleration",["volume","short_window","long_window"],volume_acceleration,"Short/long average volume acceleration."),("turnover_acceleration",["turnover","short_window","long_window"],turnover_acceleration,"Short/long turnover acceleration."),
("up_volume_ratio",["ret","volume","window"],up_volume_ratio,"Share of recent volume occurring on positive-return bars."),("down_volume_ratio",["ret","volume","window"],down_volume_ratio,"Share of recent volume occurring on negative-return bars."),("signed_volume_imbalance",["ret","volume","window"],signed_volume_imbalance,"Up-volume share minus down-volume share."),("up_down_volume_ratio",["ret","volume","window"],up_down_volume_ratio,"Up-volume to down-volume ratio."),("volume_weighted_return",["ret","volume","window"],volume_weighted_return,"Rolling volume-weighted return."),("volume_weighted_momentum",["close","volume","window"],volume_weighted_momentum,"Rolling volume-weighted close return."),
("price_volume_divergence",["close","volume","price_window","volume_window"],price_volume_divergence,"Price momentum minus volume momentum."),("price_turnover_divergence",["close","turnover","price_window","turnover_window"],price_turnover_divergence,"Price momentum minus turnover momentum."),("return_volume_beta",["ret","volume","window"],return_volume_beta,"Rolling beta of return to volume growth."),("return_turnover_beta",["ret","turnover","window"],return_turnover_beta,"Rolling beta of return to turnover growth."),
("ADL",["high","low","close","volume","window"],ADL,"Bounded accumulation/distribution flow over an explicit window."),("ChaikinOscillator",["high","low","close","volume","fast_window","slow_window","adl_window"],ChaikinOscillator,"Chaikin oscillator over bounded ADL."),("ForceIndex",["close","volume","window"],ForceIndex,"EMA-smoothed price-change times volume."),("EaseOfMovement",["high","low","volume","window","volume_scale"],EaseOfMovement,"Ease-of-Movement with explicit smoothing and volume scale."),("bounded_nvi",["close","volume","window"],bounded_nvi,"Bounded Negative Volume Index return over explicit history."),("bounded_pvi",["close","volume","window"],bounded_pvi,"Bounded Positive Volume Index return over explicit history."),
("zero_return_ratio",["ret","window","epsilon"],zero_return_ratio,"Fraction of near-zero returns in recent window."),("roll_spread_proxy",["ret","window"],roll_spread_proxy,"Roll implied-spread proxy from negative first-order return covariance."),("corwin_schultz_spread",["high","low","window"],corwin_schultz_spread,"Corwin-Schultz high-low spread proxy."),("high_low_spread_proxy",["high","low","window"],high_low_spread_proxy,"Rolling log high-low spread proxy."),("turnover_adjusted_volatility",["ret","turnover","window"],turnover_adjusted_volatility,"Return volatility scaled by trading activity."),("volume_to_range",["volume","high","low","window"],volume_to_range,"Rolling volume per unit intraday range."),
]
for _name,_params,_fn,_desc in _SPECS:_register(_name,_params,_fn,_desc)

# P1-86: EaseOfMovement's ``volume_scale`` is a pure unit-conversion constant —
# it multiplies the whole output uniformly and leaves cross-sectional ordering
# invariant, so treating it as an alpha-search dimension only manufactures
# linearly-scaled duplicates.  Tag it ``unit_conversion_only`` so a dead-parameter
# audit exempts it (its output-sensitivity is a fixed constant, by design).
_em_op = OperatorRegistry.get("EaseOfMovement", "pandas_numpy")
if _em_op is not None and "unit_conversion_only" not in (_em_op.metadata.tags or ()):
    _em_op.metadata.tags = [*(_em_op.metadata.tags or ()), "unit_conversion_only"]
