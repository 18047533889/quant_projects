# -*- coding: utf-8 -*-
"""Continuous, parameterized candlestick geometry features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator

_EPS=1e-12

def _pi(v,name,minimum=1):
    if isinstance(v,bool):raise ValueError(f"{name} must be integer")
    v=int(v)
    if v<minimum:raise ValueError(f"{name} must be >= {minimum}")
    return v

def _body(o,c):return (c-o).abs()
def _range(h,l):return h-l
def _upper(o,h,c):return h-pd.DataFrame(np.maximum(o.to_numpy(float),c.to_numpy(float)),index=o.index,columns=o.columns)
def _lower(o,l,c):return pd.DataFrame(np.minimum(o.to_numpy(float),c.to_numpy(float)),index=o.index,columns=o.columns)-l
def _z(x,w):
    w=_pi(w,"window",2);m=x.shift(1).rolling(w,min_periods=w).mean();s=x.shift(1).rolling(w,min_periods=w).std();return (x-m)/s.replace(0,np.nan)
def _pct_rank(x,w):
    w=_pi(w,"window",2)
    return x.rolling(w,min_periods=w).apply(lambda a:float((np.sum(a[:-1]<a[-1])+0.5*np.sum(a[:-1]==a[-1]))/max(1,len(a)-1)),raw=True)
def _tr(h,l,c):
    pc=c.shift(1);arr=np.maximum.reduce([(h-l).to_numpy(float),(h-pc).abs().to_numpy(float),(l-pc).abs().to_numpy(float)]);return pd.DataFrame(arr,index=h.index,columns=h.columns)
def _validate_ohlc(open=None,high=None,low=None,close=None):
    """Unified OHLC structural-validity mask (round-3 audit item 15).

    Returns a boolean DataFrame (True = structurally valid bar) where every
    provided OHLC field is strictly positive and the bar satisfies
    ``High >= max(Open, Close) >= min(Open, Close) >= Low``.  Fields omitted as
    ``None`` are not checked (operators that only consume a subset of OHLC keep
    their own axis contract).  Any NaN / missing field fails closed to False.

    Operators then ``.where(valid)`` so an invalid bar emits NaN instead of a
    negative shadow / out-of-range strength / rejection fraction outside
    [0,1] / a fabricated multi-candle pattern.
    """
    base = open if open is not None else (high if high is not None else (low if low is not None else close))
    n = len(base)
    idx = base.index
    cols = base.columns
    valid = np.ones((n, len(cols)), dtype=bool)

    def _arr(x):
        return None if x is None else np.asarray(x.to_numpy(float), dtype=float).reshape(n, len(cols))

    o, h, l, c = _arr(open), _arr(high), _arr(low), _arr(close)
    if o is not None:
        valid &= o > 0.0
    if h is not None:
        valid &= h > 0.0
    if l is not None:
        valid &= l > 0.0
    if c is not None:
        valid &= c > 0.0
    if h is not None and l is not None:
        valid &= h >= l
    if o is not None and c is not None and h is not None:
        valid &= h >= np.maximum(o, c)
    if o is not None and c is not None and l is not None:
        valid &= l <= np.minimum(o, c)
    # subset operators without ``open`` (e.g. candle_close_strength) still must
    # not emit close-strength outside [-1,1]: the close must sit inside the
    # daily range.  Redundant for full-OHLC operators (already implied by the
    # two body checks above).
    if o is None and h is not None and l is not None and c is not None:
        valid &= (c >= l) & (c <= h)
    return pd.DataFrame(valid, index=idx, columns=cols)
def _register(name,params,fn,desc):
    # round-3 cross-cutting: window/atr_window are lookback-horizon scale
    # dimensions (the engine convention for ``window`` in ts_mean & friends is
    # ParamRole.HORIZON -> searched at full resolution, NOT an estimator knob).
    specs={p:ParamSpec(dtype=int,min=2,param_role=ParamRole.HORIZON)
           for p in params if p in ("window","atr_window")}
    meta=OperatorMetadata(name=name,category="candle_pattern",description=desc,param_names=list(params),return_type="series",param_specs=specs,tags=["pit_safe","causal","bounded_history","continuous_candle","production_extension"])
    def _calculate_series(self,*args,**kwargs):return fn(*args,**kwargs)
    cls=type(f"CandleGeometryV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="candle_pattern",business_category="technical_extension",canonical=name,source="candle_geometry_v2",backend="pandas_numpy",status="production")(cls)

def candle_body_zscore(open,close,window):
    return _z(_body(open,close),window).where(_validate_ohlc(open,None,None,close))
def candle_range_zscore(high,low,window):
    return _z(_range(high,low),window).where(_validate_ohlc(None,high,low,None))
def candle_upper_shadow_zscore(open,high,close,window):
    return _z(_upper(open,high,close),window).where(_validate_ohlc(open,high,None,close))
def candle_lower_shadow_zscore(open,low,close,window):
    return _z(_lower(open,low,close),window).where(_validate_ohlc(open,None,low,close))
def candle_body_percentile(open,close,window):
    return _pct_rank(_body(open,close),window).where(_validate_ohlc(open,None,None,close))
def candle_range_percentile(high,low,window):
    return _pct_rank(_range(high,low),window).where(_validate_ohlc(None,high,low,None))
def candle_gap_atr(open,high,low,close,atr_window):
    w=_pi(atr_window,"atr_window",2)
    # P1-13: unify "ATR" with the engine's Wilder definition (indicators_v2
    # ``_wilder``) instead of a plain rolling mean of true range — one ATR
    # object across the engine.
    atr=_tr(high,low,close).ewm(alpha=1.0/w,adjust=False,min_periods=w).mean()
    # round-3 audit item 16: the denominator must be ATR AS OF t-1 (excludes
    # today's High/Low/Close); otherwise today's late-session range
    # retroactively weakens the morning gap.
    atr_prev=atr.shift(1)
    valid=_validate_ohlc(open,high,low,close) & _validate_ohlc(open,high,low,close).shift(1)
    return ((open-close.shift(1))/atr_prev.replace(0,np.nan)).where(valid)
def candle_body_position(open,high,low,close):
    return ((((open+close)/2.0)-low)/(high-low).replace(0,np.nan)).where(_validate_ohlc(open,high,low,close))
def candle_overlap_ratio(high,low):
    overlap=(pd.DataFrame(np.minimum(high.to_numpy(float),high.shift(1).to_numpy(float)),index=high.index,columns=high.columns)-pd.DataFrame(np.maximum(low.to_numpy(float),low.shift(1).to_numpy(float)),index=low.index,columns=low.columns)).clip(lower=0)
    union=pd.DataFrame(np.maximum(high.to_numpy(float),high.shift(1).to_numpy(float)),index=high.index,columns=high.columns)-pd.DataFrame(np.minimum(low.to_numpy(float),low.shift(1).to_numpy(float)),index=low.index,columns=low.columns)
    valid=_validate_ohlc(None,high,low,None) & _validate_ohlc(None,high,low,None).shift(1)
    return (overlap/union.replace(0,np.nan)).where(valid)
def candle_inside_ratio(high,low):
    prev=(high.shift(1)-low.shift(1)).replace(0,np.nan);current=high-low
    inside=(high<=high.shift(1))&(low>=low.shift(1))
    outside=(high>=high.shift(1))&(low<=low.shift(1))  # engulfing containment
    # P1-12: a shifted-up / shifted-down / gap bar is NEITHER inside NOR
    # outside containment; the old code dumped every non-inside bar into the
    # >1 "outside" region, fabricating a containment relationship.  Only true
    # containment states get a value; ambiguous bars are NaN.
    in_v=(current/prev).where(inside)                  # (0,1]
    out_v=(1.0+prev/current).where(outside & ~inside)  # (1,2]
    valid=_validate_ohlc(None,high,low,None) & _validate_ohlc(None,high,low,None).shift(1)
    return in_v.fillna(out_v).where((prev.notna() & current.notna() & (inside|outside)) & valid)
def candle_close_strength(high,low,close):
    return (2.0*(close-low)/(high-low).replace(0,np.nan)-1.0).where(_validate_ohlc(None,high,low,close))
def candle_rejection_upper(open,high,low,close):
    return (_upper(open,high,close)/(high-low).replace(0,np.nan)).where(_validate_ohlc(open,high,low,close))
def candle_rejection_lower(open,high,low,close):
    return (_lower(open,low,close)/(high-low).replace(0,np.nan)).where(_validate_ohlc(open,high,low,close))

_NAMES=[]
for _name,_params,_fn,_desc in [
("candle_body_zscore",["open","close","window"],candle_body_zscore,"Body size z-score versus prior candles."),
("candle_range_zscore",["high","low","window"],candle_range_zscore,"Range z-score versus prior candles."),
("candle_upper_shadow_zscore",["open","high","close","window"],candle_upper_shadow_zscore,"Upper-shadow z-score versus prior candles."),
("candle_lower_shadow_zscore",["open","low","close","window"],candle_lower_shadow_zscore,"Lower-shadow z-score versus prior candles."),
("candle_body_percentile",["open","close","window"],candle_body_percentile,"Body-size percentile within an explicit window."),
("candle_range_percentile",["high","low","window"],candle_range_percentile,"Range percentile within an explicit window."),
("candle_gap_atr",["open","high","low","close","atr_window"],candle_gap_atr,"Opening gap normalized by rolling ATR."),
("candle_body_position",["open","high","low","close"],candle_body_position,"Body midpoint location inside daily range."),
("candle_overlap_ratio",["high","low"],candle_overlap_ratio,"Current/previous candle range-overlap ratio."),
("candle_inside_ratio",["high","low"],candle_inside_ratio,"Continuous inside/outside range-size encoding."),
("candle_close_strength",["high","low","close"],candle_close_strength,"Close location mapped to [-1,1]."),
("candle_rejection_upper",["open","high","low","close"],candle_rejection_upper,"Upper-wick rejection fraction."),
("candle_rejection_lower",["open","high","low","close"],candle_rejection_lower,"Lower-wick rejection fraction."),
]:
    _register(_name,_params,_fn,_desc);_NAMES.append(_name)

import cleaned_operators.operator_surface as _surface
_surface.extend_extended_only(set(_NAMES))
