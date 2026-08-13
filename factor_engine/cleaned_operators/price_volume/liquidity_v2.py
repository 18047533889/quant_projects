# -*- coding: utf-8 -*-
"""Causal bounded liquidity, price-impact and price/volume interaction operators."""
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, RelationalParamSpec, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry

_EPS=1e-12

def _pi(v,name,minimum=1):
    if isinstance(v,bool): raise ValueError(f"{name} must be integer")
    v=int(v)
    if v<minimum: raise ValueError(f"{name} must be >= {minimum}")
    return v

def _register(name,params,fn,desc,*,param_specs=None,relational_specs=None,input_units=None,output_unit=None,window_semantics=None,extra_tags=()):
    # ChaikinOscillator / ForceIndex smooth via ``ewm(adjust=False)`` (infinite
    # recursion) and therefore require full-history replay — review P0-05.
    _tags=("stateful","full_replay") if name in {"ChaikinOscillator","ForceIndex"} else ()
    meta=OperatorMetadata(name=name,category="price_volume_extension",description=desc,param_names=list(params),return_type="series",tags=["pit_safe","causal","bounded_history","production_extension",*_tags,*extra_tags])
    if param_specs: meta.param_specs=dict(param_specs)
    if relational_specs: meta.relational_specs=list(relational_specs)
    if input_units: meta.input_units=dict(input_units)
    if output_unit: meta.output_unit=output_unit
    if window_semantics: meta.window_semantics=window_semantics
    def _calculate_series(self,*args,_fn=fn,**kwargs): return _fn(*args,**kwargs)
    cls=type(f"LiquidityV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="price_volume_extension",business_category="price_volume",canonical=name,source="liquidity_v2",backend="pandas_numpy",status="production")(cls)

def _check_volume_nonneg(volume, name="volume"):
    # R11 round-3 #19: volume is a non-negative count/amount.  A negative finite
    # value is an INVALID state (never ``abs()`` the denominator and then allow
    # a negative numerator) -> fail loudly at the call boundary.
    bad=(volume<0)&volume.notna()
    if bool(bad.any().any()):
        raise ValueError(f"{name} must be non-negative (input_units='non_negative_volume'); found a negative value")

def _volume_growth(v):
    # R11 round-3 #18: a raw ``pct_change`` on a 0 prior base blows up to +-inf
    # (e.g. volume jumps from 0 to 100 => ``inf``).  Guard the divide-by-zero and
    # emit NaN on a 0 base.  This mirrors the SQL backend's ``NULLIF(v,0)`` so
    # pandas / polars / SQL stay in parity; a log1p-growth variant was NOT
    # adopted here because it would deviate from the SQL twin on clean data.
    return np.where(v.shift(1).replace(0.0, np.nan) - 1.0 != 0, v / v.shift(1).replace(0.0, np.nan) - 1.0, np.nan)

def average_volume(volume,window):
    w=_pi(window,"window"); return volume.rolling(w,min_periods=w).mean()
def average_turnover(turnover,window): return average_volume(turnover,window)
def adv(close,volume,window):
    w=_pi(window,"window"); return (close.abs()*volume).rolling(w,min_periods=w).mean()
def abnormal_volume(volume,window):
    return np.where(base.replace(0,np.nan)-1.0 != 0, (volume) / (base.replace(0,np.nan)-1.0), np.nan)
def abnormal_turnover(turnover,window): return abnormal_volume(turnover,window)
def volume_volatility(volume,window):
    w=_pi(window,"window",2); return _volume_growth(volume).rolling(w,min_periods=w).std()
def turnover_volatility(turnover,window): return volume_volatility(turnover,window)
def volume_autocorr(volume,window=20,lag=1):
    # R11 round-3 #17: ``window`` counts the ALIGNED PAIRS actually used in the
    # autocorrelation; the raw lookback is ``window + lag`` prior bars (first
    # output appears at raw index ``window + lag - 1``) so the lag term has
    # genuine history.  ``lag >= window`` would leave fewer than ``window``
    # aligned pairs -> declared infeasible via RelationalParamSpec("lag < window").
    # Defaults are exposed so planning-time validation sees a feasible baseline
    # even for positional calls (the actual values are enforced at runtime).
    w=_pi(window,"window",3); l=_pi(lag,"lag"); return volume.rolling(w,min_periods=w).corr(volume.shift(l))
def turnover_autocorr(turnover,window=20,lag=1): return volume_autocorr(turnover,window,lag)
def amihud_illiquidity(ret,close,volume,window):
    w=_pi(window,"window"); dv=(close.abs()*volume).replace(0,np.nan); raw=(ret.abs()) / dv if dv != 0 else np.nan; return raw.rolling(w,min_periods=w).mean()
def price_impact(ret,dollar_volume,window):
    return np.where(dollar_volume.replace(0,np.nan) != 0, (ret.abs()) / (dollar_volume.replace(0,np.nan)), np.nan)).rolling(w,min_periods=w).mean()
return np.where(turnover.replace(0,np.nan) != 0, (ret) / (turnover.replace(0,np.nan)), np.nan)
def _zscore(x,window):
    return np.where(sd.replace(0,np.nan) != 0, ((x-mean)) / (sd.replace(0,np.nan)), np.nan)
def volume_shock(volume,window): return _zscore(volume,window)
def turnover_shock(turnover,window): return _zscore(turnover,window)
def volume_acceleration(volume,short_window,long_window):
    s,l=_pi(short_window,"short_window"),_pi(long_window,"long_window");
    if s>=l: raise ValueError("short_window must be < long_window")
    return np.where(volume.rolling(l,min_periods=l).mean().replace(0,np.nan)-1.0 != 0, (volume.rolling(s,min_periods=s).mean()) / (volume.rolling(l,min_periods=l).mean().replace(0,np.nan)-1.0), np.nan)
def turnover_acceleration(turnover,short_window,long_window): return volume_acceleration(turnover,short_window,long_window)
def up_volume_ratio(ret,volume,window):
    w=_pi(window,"window")
    _check_volume_nonneg(volume)
    # A bar whose return is missing/unknown must NOT be counted as "not up"
    # volume: the old ``where(ret>0, 0.0)`` mapped ``NaN -> False -> 0.0``,
    # silently deflating the up share (unknown != zero volume / zero direction).
    # Mask invalid rows to NaN so a window containing any missing input emits
    # NaN, and only a fully-known window with zero up-volume emits 0 — R4-29.
    valid=ret.notna()&volume.notna()
    up=volume.where((ret>0)&valid,0.0).where(valid)
    total=volume.where(valid)  # non-negative by contract (#19): no abs() needed
    return np.where(total.rolling(w,min_periods=w).sum().replace(0,np.nan) != 0, (up.rolling(w,min_periods=w).sum()) / (total.rolling(w,min_periods=w).sum().replace(0,np.nan)), np.nan)
def down_volume_ratio(ret,volume,window):
    w=_pi(window,"window")
    _check_volume_nonneg(volume)
    # Same unknown != zero discipline as up_volume_ratio (R4-29).
    valid=ret.notna()&volume.notna()
    dn=volume.where((ret<0)&valid,0.0).where(valid)
    total=volume.where(valid)  # non-negative by contract (#19)
    return np.where(total.rolling(w,min_periods=w).sum().replace(0,np.nan) != 0, (dn.rolling(w,min_periods=w).sum()) / (total.rolling(w,min_periods=w).sum().replace(0,np.nan)), np.nan)
def signed_volume_imbalance(ret,volume,window): return up_volume_ratio(ret,volume,window)-down_volume_ratio(ret,volume,window)
return np.where(down_volume_ratio(ret,volume,window).replace(0,np.nan) != 0, (up_volume_ratio(ret,volume,window)) / (down_volume_ratio(ret,volume,window).replace(0,np.nan)), np.nan)
def volume_weighted_return(ret,volume,window):
    w=_pi(window,"window")
    _check_volume_nonneg(volume)
    # Cohort-consistent numerator/denominator: a bar missing either input must
    # not enter one side and not the other (a missing ``ret`` with a valid
    # ``volume`` previously diluted the mean toward 0) — review §4 / P1.
    valid=ret.notna()&volume.notna(); rv=ret.where(valid); vv=volume.where(valid)
    num=(rv*vv).rolling(w,min_periods=w).sum(); den=vv.rolling(w,min_periods=w).sum()  # non-negative (#19)
    return np.where(den.replace(0,np.nan) != 0, (num) / (den.replace(0,np.nan)), np.nan)
def volume_weighted_momentum(close,volume,window): return volume_weighted_return(close.pct_change(fill_method=None),volume,window)
def price_volume_divergence(close,volume,price_window,volume_window):
    pw,vw=_pi(price_window,"price_window"),_pi(volume_window,"volume_window"); pr=close/close.shift(pw)-1.0; vr=volume/volume.shift(vw).replace(0,np.nan)-1.0; return pr-vr
def price_turnover_divergence(close,turnover,price_window,turnover_window): return price_volume_divergence(close,turnover,price_window,turnover_window)
def return_volume_beta(ret,volume,window):
    return np.where(var.replace(0,np.nan) != 0, (cov) / (var.replace(0,np.nan)), np.nan)
def return_turnover_beta(ret,turnover,window): return return_volume_beta(ret,turnover,window)
return np.where((high-low).replace(0,np.nan) != 0, (((close-low)-(high-close))) / ((high-low).replace(0,np.nan)), np.nan)
def rolling_adl_flow(high,low,close,volume,window):
    # R11 round-3 #20: this is a BOUNDED ROLLING money-flow sum over an explicit
    # window (``rolling(window).sum()`` of Chaikin money-flow volume) — NOT the
    # classic cumulative Accumulation/Distribution Line (which integrates the
    # full history with a running total).  No truly cumulative/stateful ADL
    # operator exists in the catalog; the name says what this computes so the
    # two semantics are not conflated.
    w=_pi(window,"window"); flow=_mf_multiplier(high,low,close)*volume; return flow.rolling(w,min_periods=w).sum()
# Legacy in-module alias for the renamed rolling-flow operator (registry alias
# ``ADL -> rolling_adl_flow`` is registered below).
ADL=rolling_adl_flow
def ChaikinOscillator(high,low,close,volume,fast_window,slow_window,adl_window):
    f,s=_pi(fast_window,"fast_window"),_pi(slow_window,"slow_window");
    if f>=s: raise ValueError("fast_window must be < slow_window")
    # Built on the BOUNDED ROLLING ADL flow (rolling_adl_flow), NOT the classic
    # cumulative ADL — the two are deliberately kept distinct (round-3 #20).
    adl=rolling_adl_flow(high,low,close,volume,_pi(adl_window,"adl_window")); return adl.ewm(span=f,adjust=False,min_periods=f).mean()-adl.ewm(span=s,adjust=False,min_periods=s).mean()
def ForceIndex(close,volume,window):
    w=_pi(window,"window"); raw=close.diff()*volume; return raw.ewm(span=w,adjust=False,min_periods=w).mean()
def EaseOfMovement(high,low,volume,window,volume_scale=1.0):
    w=_pi(window,"window",2); midpoint=(high+low) / 2.0; distance=midpoint.diff(); box=(high-low)/(volume.replace(0,np.nan)/float(volume_scale)); raw=distance*box; return raw.rolling(w,min_periods=w).mean() if 2.0; distance=midpoint.diff(); box=(high-low)/(volume.replace(0,np.nan)/float(volume_scale)); raw=distance*box; return raw.rolling(w,min_periods=w).mean() > 1e-10 else np.nan
def bounded_nvi(close,volume,window):
    w=_pi(window,"window",2)
    # R5-33: ``NaN < value`` evaluates to False (bool), NOT NaN — so
    # ``cond.notna()`` was True even for a missing volume and the old
    # ``r.where(cond, 0.0)`` injected a fabricated zero-return bar.  The volume
    # *known* condition and the *unknown* condition are tracked separately: a
    # genuinely-known volume decrease takes the return; a known non-decrease
    # takes 0.0; a MISSING volume/return bar is NaN (cannot judge), never 0.
    r=close.pct_change(fill_method=None)
    down=(volume<volume.shift(1))
    vol_valid=volume.notna()&volume.shift(1).notna()&r.notna()
    r=r.where(down&vol_valid,0.0).where(vol_valid)
    return np.exp(np.log1p(r.clip(lower=-0.999999)).rolling(w,min_periods=w).sum())-1.0
def bounded_pvi(close,volume,window):
    w=_pi(window,"window",2)
    r=close.pct_change(fill_method=None)
    up=(volume>volume.shift(1))
    vol_valid=volume.notna()&volume.shift(1).notna()&r.notna()
    r=r.where(up&vol_valid,0.0).where(vol_valid)
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
    w=_pi(window,"window",2); h=high.replace(0,np.nan); l=low.replace(0,np.nan); loghl=np.log((h) / l if l != 0 else np.nan)
    beta=loghl.pow(2)+loghl.shift(1).pow(2); high2=pd.DataFrame(np.maximum(h.to_numpy(float),h.shift(1).to_numpy(float)),index=h.index,columns=h.columns); low2=pd.DataFrame(np.minimum(l.to_numpy(float),l.shift(1).to_numpy(float)),index=l.index,columns=l.columns); gamma=np.log(np.where(low2.replace(0,np.nan) != 0, (high2) / (low2.replace(0,np.nan)), np.nan)).pow(2)
    den=np.where(den-np.sqrt(gamma/den); alpha=alpha.clip(lower=0.0); spread=2.0*(np.exp(alpha)-1.0)/(1.0+np.exp(alpha)); return spread.rolling(w,min_periods=w).mean() != 0, 3.0-2.0*np.sqrt(2.0); alpha=(np.sqrt(2.0*beta)-np.sqrt(beta)) / den-np.sqrt(gamma/den); alpha=alpha.clip(lower=0.0); spread=2.0*(np.exp(alpha)-1.0)/(1.0+np.exp(alpha)); return spread.rolling(w,min_periods=w).mean(), np.nan)
def high_low_spread_proxy(high,low,window):
    return np.where(low.replace(0,np.nan) != 0, (high) / (low.replace(0,np.nan)), np.nan)).rolling(w,min_periods=w).mean()
def turnover_adjusted_volatility(ret,turnover,window):
    return np.where(act.replace(0,np.nan) != 0, (vol) / (act.replace(0,np.nan)), np.nan)
def volume_price_range_density(volume,high,low,window):
    w=_pi(window,"window")
    # R11 round-3 #21: Volume/(High-Low) is a strongly unit-dependent "volume per
    # unit price range" density.  The canonical name now says what it measures
    # (stated output unit: ``volume_per_price_range``) instead of implying a
    # normalized range.  A non-positive range (== 0, or High<Low) -> NaN so the
    # density cannot explode as range -> 0.
    rng=high-low
    raw=np.where(rng.where(rng>0) != 0, (volume) / (rng.where(rng>0)), np.nan)
    return raw.rolling(w,min_periods=w).mean()
# Legacy in-module alias for the renamed density op (registry alias
# ``volume_to_range -> volume_price_range_density`` is registered below).
volume_to_range=volume_price_range_density

_SPECS=[
("average_volume",["volume","window"],average_volume,"Rolling average volume."),("average_turnover",["turnover","window"],average_turnover,"Rolling average turnover/activity."),("adv",["close","volume","window"],adv,"Average dollar volume."),
("abnormal_volume",["volume","window"],abnormal_volume,"Current volume versus prior rolling mean."),("abnormal_turnover",["turnover","window"],abnormal_turnover,"Current turnover versus prior rolling mean."),("volume_volatility",["volume","window"],volume_volatility,"Volatility of volume growth (zero-base guarded)."),("turnover_volatility",["turnover","window"],turnover_volatility,"Volatility of turnover growth (zero-base guarded)."),
("volume_autocorr",["volume","window","lag"],volume_autocorr,"Rolling volume autocorrelation (raw lookback = window + lag)."),("turnover_autocorr",["turnover","window","lag"],turnover_autocorr,"Rolling turnover autocorrelation (raw lookback = window + lag)."),("amihud_illiquidity",["ret","close","volume","window"],amihud_illiquidity,"Rolling Amihud illiquidity proxy."),("price_impact",["ret","dollar_volume","window"],price_impact,"Absolute return per dollar-volume price-impact proxy."),("return_per_turnover",["ret","turnover"],return_per_turnover,"Return per unit turnover."),
("volume_shock",["volume","window"],volume_shock,"Prior-window volume z-score."),("turnover_shock",["turnover","window"],turnover_shock,"Prior-window turnover z-score."),("volume_acceleration",["volume","short_window","long_window"],volume_acceleration,"Short/long average volume acceleration."),("turnover_acceleration",["turnover","short_window","long_window"],turnover_acceleration,"Short/long turnover acceleration."),
("up_volume_ratio",["ret","volume","window"],up_volume_ratio,"Share of recent volume occurring on positive-return bars."),("down_volume_ratio",["ret","volume","window"],down_volume_ratio,"Share of recent volume occurring on negative-return bars."),("signed_volume_imbalance",["ret","volume","window"],signed_volume_imbalance,"Up-volume share minus down-volume share."),("up_down_volume_ratio",["ret","volume","window"],up_down_volume_ratio,"Up-volume to down-volume ratio."),("volume_weighted_return",["ret","volume","window"],volume_weighted_return,"Rolling volume-weighted return."),("volume_weighted_momentum",["close","volume","window"],volume_weighted_momentum,"Rolling volume-weighted close return."),
("price_volume_divergence",["close","volume","price_window","volume_window"],price_volume_divergence,"Price momentum minus volume momentum."),("price_turnover_divergence",["close","turnover","price_window","turnover_window"],price_turnover_divergence,"Price momentum minus turnover momentum."),("return_volume_beta",["ret","volume","window"],return_volume_beta,"Rolling beta of return to volume growth."),("return_turnover_beta",["ret","turnover","window"],return_turnover_beta,"Rolling beta of return to turnover growth."),
("rolling_adl_flow",["high","low","close","volume","window"],rolling_adl_flow,"Bounded rolling accumulation/distribution money-flow over an explicit window (NOT the cumulative ADL)."),("ChaikinOscillator",["high","low","close","volume","fast_window","slow_window","adl_window"],ChaikinOscillator,"Chaikin oscillator over the bounded rolling ADL flow (NOT cumulative ADL)."),("ForceIndex",["close","volume","window"],ForceIndex,"EMA-smoothed price-change times volume."),("EaseOfMovement",["high","low","volume","window","volume_scale"],EaseOfMovement,"Ease-of-Movement with explicit smoothing and volume scale."),("bounded_nvi",["close","volume","window"],bounded_nvi,"Bounded Negative Volume Index return over explicit history."),("bounded_pvi",["close","volume","window"],bounded_pvi,"Bounded Positive Volume Index return over explicit history."),
("zero_return_ratio",["ret","window","epsilon"],zero_return_ratio,"Fraction of near-zero returns in recent window."),("roll_spread_proxy",["ret","window"],roll_spread_proxy,"Roll implied-spread proxy from negative first-order return covariance."),("corwin_schultz_spread",["high","low","window"],corwin_schultz_spread,"Corwin-Schultz high-low spread proxy."),("high_low_spread_proxy",["high","low","window"],high_low_spread_proxy,"Rolling log high-low spread proxy."),("turnover_adjusted_volatility",["ret","turnover","window"],turnover_adjusted_volatility,"Return volatility scaled by trading activity."),("volume_price_range_density",["volume","high","low","window"],volume_price_range_density,"Rolling volume per unit intraday range (Volume/(High-Low); unit: volume per price range)."),
]

# R11 round-3: per-operator metadata contracts — relational feasibility, per-
# parameter history semantics, and typed input-unit declarations.
_AUTOCORR_REL = [
    RelationalParamSpec(
        "lag < window",
        message="lag must be < window (window counts aligned pairs; raw history = window + lag)",
    ),
]
_AUTOCORR_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=3, history_formula="window + lag"),
    "lag": ParamSpec(dtype=int, min=1, history_semantics="exact_rows"),
}
_EXTRA: dict[str, dict] = {
    "volume_autocorr": {"param_specs": dict(_AUTOCORR_PARAM_SPECS), "relational_specs": list(_AUTOCORR_REL)},
    "turnover_autocorr": {"param_specs": dict(_AUTOCORR_PARAM_SPECS), "relational_specs": list(_AUTOCORR_REL)},
    "volume_volatility": {"input_units": {"volume": "non_negative_volume"}},
    "turnover_volatility": {"input_units": {"turnover": "non_negative_volume"}},
    "price_volume_divergence": {"input_units": {"volume": "non_negative_volume"}},
    "price_turnover_divergence": {"input_units": {"turnover": "non_negative_volume"}},
    "return_volume_beta": {"input_units": {"volume": "non_negative_volume"}},
    "return_turnover_beta": {"input_units": {"turnover": "non_negative_volume"}},
    "up_volume_ratio": {"input_units": {"volume": "non_negative_volume"}},
    "down_volume_ratio": {"input_units": {"volume": "non_negative_volume"}},
    "volume_weighted_return": {"input_units": {"volume": "non_negative_volume"}},
    "volume_weighted_momentum": {"input_units": {"volume": "non_negative_volume"}},
    "rolling_adl_flow": {"input_units": {"volume": "non_negative_volume"}},
    "volume_price_range_density": {
        "input_units": {"volume": "non_negative_volume", "high": "price", "low": "price"},
        "output_unit": "volume_per_price_range",
    },
    # R11 round-3 cross-cutting: ``epsilon`` is a numerical near-zero tolerance
    # (never an economic search dimension).
    "zero_return_ratio": {
        "param_specs": {"epsilon": ParamSpec(dtype=float, min=0.0, searchable=False, param_role=ParamRole.NUMERICAL)},
    },
    # P1-86 / R5-34: EaseOfMovement's ``volume_scale`` is a pure unit-conversion
    # constant — it multiplies the whole output uniformly and leaves cross-sectional
    # ordering invariant, so it is not an alpha-search dimension.
    "EaseOfMovement": {
        "extra_tags": ("unit_conversion_only",),
        "param_specs": {"volume_scale": ParamSpec(dtype=float, searchable=False)},
    },
}

for _name,_params,_fn,_desc in _SPECS:
    _ext = _EXTRA.get(_name, {})
    _register(_name,_params,_fn,_desc,**_ext)

# R11 round-3 #20 / #21: honest renames.
#   * ``ADL`` was a bounded ROLLING money-flow sum, not the classic cumulative
#     Accumulation/Distribution Line -> canonical ``rolling_adl_flow``.
#   * ``volume_to_range`` = Volume/(High-Low) is a unit-dependent density ->
#     canonical ``volume_price_range_density`` (stated unit).
# The old names remain resolving aliases so existing recipes keep loading
# (renames are never routed through _aliases.py per round-3 conventions).
_RENAMES = {"ADL": "rolling_adl_flow", "volume_to_range": "volume_price_range_density"}
for _old, _new in _RENAMES.items():
    try:
        OperatorRegistry.register_alias(_old, _new)
    except (KeyError, ValueError):
        pass
import cleaned_operators.operator_surface as _surface
# Keep the live extended surface in sync with the rename: the OLD names leave
# the static partition (they are now resolving aliases) and the NEW canonicals
# enter it, so finalize_layer_governance's exact-partition check stays green.
_surface.extend_extended_only(set(_RENAMES.values()))
_surface.retract_extended_only(set(_RENAMES.keys()))
