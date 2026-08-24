# -*- coding: utf-8 -*-
"""Static production call validation for Extended/Pandas-reference operators."""
from __future__ import annotations
import math
from typing import Any
from factor_engine.planner.logical_plan import PlanNode

_PANEL_NAMES=frozenset({
"x","y","z","a","b","w","g","left","right","numerator","denominator","ret","returns","benchmark_ret","market_ret","benchmark","market","open","high","low","close","price","volume","amount","vwap","turnover","weight","weights","condition","group","industry","sector","fiscal_quarter","period_id","quarter","revision_id","decision_time","available_time","available_at","exposure","exposures","control","controls","factor","target","mask","event","sort_col","float_shares","flow","balance","earnings","cashflow","assets","working_capital","base","dollar_volume","scale_base",
})
_WINDOW_NAMES=frozenset({
"window","d","n","span","period","periods","lookback","max_lookback","fast","slow","fast_period","slow_period","signal_span","signal_period","left_window","right_window","history_window","fast_window","slow_window","signal_window","short_window","medium_window","long_window","ema_window","atr_window","tenkan_window","kijun_window","senkou_b_window","er_window","vol_window","baseline_window","price_window","volume_window","turnover_window","adl_window","impulse_window","flag_window","growth_periods","compare_periods","window_periods","average_periods","periods_per_year","short_periods","long_periods","max_periods","history_days","bar_minutes","minutes","lag","window_days","max_days","body_window","shadow_window",
})
_MACD=frozenset({"MACD_line","MACD_signal","MACD_hist"})
_QUANTILE_Q=frozenset({"ts_quantile","cs_quantile","group_percentile","ts_tail_mean","tail_beta","lqtp_historical_cvar"})
_TOPK=frozenset({"ts_topk_sum","ts_topk_mean","ts_topk_std","ts_bottomk_sum","ts_bottomk_mean","ts_bottomk_std"})
_STRUCTURE_WITH_HISTORY=frozenset({
"ts_last_pivot_high","ts_last_pivot_low","ts_pivot_high_age","ts_pivot_low_age","ts_resistance_level","ts_support_level","ts_resistance_slope","ts_support_slope","ts_distance_to_resistance","ts_distance_to_support","ts_resistance_break","ts_support_break","ts_nth_pivot_high","ts_nth_pivot_low","ts_nth_pivot_high_age","ts_nth_pivot_low_age","ts_pivot_high_count","ts_pivot_low_count","ts_pivot_high_spacing","ts_pivot_low_spacing","ts_swing_amplitude","ts_swing_amplitude_pct","ts_swing_duration","ts_swing_velocity","ts_swing_amplitude_atr","ts_channel_width","ts_channel_width_pct","ts_channel_width_atr","ts_channel_width_slope","ts_line_convergence","ts_line_parallelism","ts_resistance_fit_r2","ts_support_fit_r2","ts_pattern_symmetry","pattern_double_top","pattern_double_bottom","pattern_head_shoulders","pattern_inverse_head_shoulders","pattern_sym_triangle","pattern_ascending_triangle","pattern_descending_triangle","pattern_rising_wedge","pattern_falling_wedge","pattern_rectangle","pattern_rising_channel","pattern_falling_channel","pattern_broadening",
})
_CANDLE_PATTERNS=frozenset({
"2_crows","3_inside","3_outside","3_line_strike","3_stars_south","abandoned_baby","advance_block","belt_hold","breakaway","closing_marubozu","counterattack","doji_star","evening_doji_star","high_wave","homing_pigeon","identical_3_crows","in_neck","on_neck","thrusting","ladder_bottom","long_legged_doji","long_line","short_line","matching_low","mat_hold","rickshaw_man","rise_fall_3_methods","separating_lines","stalled_pattern","stick_sandwich","takuri","tasuki_gap","tristar","unique_3_river","upside_gap_2_crows","xside_gap_3_methods",
})

def _declared_panel_params(canonical: str) -> frozenset[str]:
    """R7-312: read the operator's DECLARED panel contract instead of guessing
    from a hardcoded name list.  ``OperatorMetadata.panel_params`` /
    ``scalar_params`` (R7-224) are authoritative when present."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        meta = getattr(op, "metadata", None)
        declared = getattr(meta, "panel_params", None)
        if declared:
            return frozenset(str(x) for x in declared)
        panel_arity = getattr(meta, "panel_arity", None)
        if panel_arity is not None:
            names = [str(x) for x in (getattr(meta, "param_names", None) or [])]
            return frozenset(names[: int(panel_arity)])
        declared_scalar = getattr(meta, "scalar_params", None)
        if declared_scalar:
            names = [str(x) for x in (getattr(meta, "param_names", None) or [])]
            scalar = frozenset(str(x) for x in declared_scalar)
            return frozenset(n for n in names if n not in scalar)
    except Exception:
        pass
    return frozenset()


def _declared_int_params(canonical: str) -> frozenset[str]:
    """R7-314: parameters declared ``ParamSpec(dtype=int)`` on the operator —
    the authoritative source for the positive-integer gate.  Empty = the
    operator declares no ParamSpec, so the legacy name heuristic applies."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        meta = getattr(op, "metadata", None)
        specs = getattr(meta, "param_specs", None) or {}
        return frozenset(
            name for name, spec in specs.items()
            if spec is not None and getattr(spec, "dtype", None) is int
        )
    except Exception:
        return frozenset()


def _int_spec_min(canonical: str, name: str) -> int:
    """R7-314: the declared ``ParamSpec.min`` for an int param (default 1 when
    unset — the static gate keeps its historical conservative default)."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        meta = getattr(op, "metadata", None)
        spec = (getattr(meta, "param_specs", None) or {}).get(name)
        if spec is not None and getattr(spec, "min", None) is not None:
            return int(spec.min)
    except Exception:
        pass
    return 1


def _is_panel(canonical:str,name:str)->bool:
    # R7-312: a DECLARED panel contract wins over the legacy name heuristic —
    # a new optional panel input must not be silently treated as a tuning
    # scalar because its name is absent from the hardcoded list.
    declared = _declared_panel_params(canonical)
    if declared:
        return name in declared
    if name in _PANEL_NAMES:return True
    if name=="signal":return canonical not in _MACD
    if name in {"value","values","fallback"}:return canonical!="fillna_const"
    return False
def _finite(v):
    # R7-313: the static validator must not accept a numeric STRING that the
    # runtime ParamSpec binder rejects.  ``float("20")`` succeeds here but a
    # declared-int parameter receiving the string ``"20"`` is a contract
    # violation at runtime ("must be an integer, not str").  Strings are
    # non-finite for static purposes, so static and runtime agree.
    if isinstance(v,bool):return False
    if isinstance(v,str):return False
    try:return math.isfinite(float(v))
    except (TypeError,ValueError):return False
def _integer(v):
    if not _finite(v):return None
    x=float(v);return int(x) if int(x)==x else None
def _call_values(node,canonical):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    names=[str(x) for x in (OperatorRegistry._catalog.get(canonical,{}).get("param_names") or []) if str(x)!="..."];values={};dynamic=[]
    for i,name in enumerate(names):
        if _is_panel(canonical,name):continue
        if i>=len(node.inputs):continue
        child=node.inputs[i]
        if getattr(child,"op",None)=="literal":values[name]=(getattr(child,"attrs",None) or {}).get("value")
        else:dynamic.append(name)
    for name,value in dict(node.attrs or {}).items():
        if not _is_panel(canonical,str(name)):values[str(name)]=value
    return values,sorted(set(dynamic))
def _posint(canonical,values,name,minimum=1):
    if name not in values:return None
    v=_integer(values[name])
    return None if v is not None and v>=minimum else f"{canonical}: {name} must be an integer >= {minimum}"
def _first(values,*names):
    for n in names:
        if n in values:return values[n]
    return None
def _ordered(canonical,values,names):
    vals=[]
    for name in names:
        if name not in values:return None
        v=_integer(values[name])
        if v is None:return f"{canonical}: {name} must be an integer"
        vals.append(v)
    return f"{canonical}: require {' < '.join(names)}" if any(a>=b for a,b in zip(vals,vals[1:])) else None
def _range(canonical,values,name,lo=None,hi=None,strict_lo=False,strict_hi=False):
    if name not in values:return None
    if not _finite(values[name]):return f"{canonical}: {name} must be finite"
    x=float(values[name])
    if lo is not None and (x<=lo if strict_lo else x<lo):return f"{canonical}: {name} below allowed range"
    if hi is not None and (x>=hi if strict_hi else x>hi):return f"{canonical}: {name} above allowed range"
    return None

def validate_pandas_first_call(canonical:str,node:PlanNode)->tuple[bool,str]:
    from factor_engine.cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS
    if canonical not in PANDAS_FIRST_PRODUCTION_CANONICALS:return True,""
    values,dynamic=_call_values(node,canonical)
    if dynamic:return False,f"{canonical}: production tuning parameter(s) must be literal: {', '.join(dynamic)}"
    # R7-314: a declared ParamSpec(dtype=int) drives the window/positive-int
    # gate.  The legacy ``_WINDOW_NAMES`` heuristic applies only to operators
    # that declare NO ParamSpec (undeclared legacy ops).
    declared_int = _declared_int_params(canonical)
    if declared_int:
        for name in declared_int:
            if name in values:
                err=_posint(canonical,values,name, _int_spec_min(canonical, name))
                if err:return False,err
    else:
        for name in _WINDOW_NAMES:
            err=_posint(canonical,values,name)
            if err:return False,err
        for name in ("min_periods","min_obs","run","buckets","top","k","points"):
            err=_posint(canonical,values,name)
            if err:return False,err
    for names in (("fast_window","slow_window"),("fast_period","slow_period"),("fast","slow"),("short_periods","long_periods")):
        err=_ordered(canonical,values,names)
        if err:return False,err
    if canonical=="UltimateOscillator":
        err=_ordered(canonical,values,("short_window","medium_window","long_window"))
        if err:return False,err
        weights=[values.get(x) for x in ("short_weight","medium_weight","long_weight") if x in values]
        if weights and (any(not _finite(x) or float(x)<0 for x in weights) or sum(float(x) for x in weights)<=0):return False,"UltimateOscillator: weights must be finite non-negative and sum to > 0"
    if canonical=="KAMA":
        err=_ordered(canonical,values,("fast_window","slow_window"))
        if err:return False,err
    if canonical in _STRUCTURE_WITH_HISTORY:
        h=_integer(values.get("history_window")) if "history_window" in values else None;p=_integer(values.get("points")) if "points" in values else None;n=_integer(values.get("n")) if "n" in values else None
        if h is not None and p is not None and p>h:return False,f"{canonical}: points must be <= history_window"
        if h is not None and n is not None and n>h:return False,f"{canonical}: n must be <= history_window"
    if "min_spacing" in values and "max_spacing" in values:
        lo,hi=_integer(values["min_spacing"]),_integer(values["max_spacing"])
        if lo is None or hi is None or lo>hi:return False,f"{canonical}: require min_spacing <= max_spacing"
    for name in ("q","quantile","p","fraction"):
        if canonical in _QUANTILE_Q and name in values:
            err=_range(canonical,values,name,0.0,1.0)
            if err:return False,err
    if canonical in _TOPK:
        w=_integer(_first(values,"window","d"));k=_integer(values.get("k")) if "k" in values else None
        if w is not None and k is not None and k>w:return False,f"{canonical}: k must be <= window"
    if canonical=="ts_moment" and "k" in values:
        k=_integer(values["k"])
        if k is None or not 1<=k<=8:return False,"ts_moment: production supports integer moment order 1..8"
    if canonical=="ts_sma_cn":
        n,m=_integer(values.get("n")),_integer(values.get("m"))
        if n is not None and m is not None and not 0<m<=n:return False,"ts_sma_cn: require 0 < m <= n"
    if canonical=="digital_count":
        d,run=_integer(values.get("d")),_integer(values.get("run"))
        if d is not None and run is not None and run>d:return False,"digital_count: run must be <= d"
    if canonical in _MACD:
        f=_integer(_first(values,"fast","fast_period"));s=_integer(_first(values,"slow","slow_period"))
        if f is not None and s is not None and f>=s:return False,f"{canonical}: fast period must be < slow period"
    if canonical=="PSAR" and "acceleration" in values and "maximum" in values:
        a,m=values["acceleration"],values["maximum"]
        if not (_finite(a) and _finite(m) and 0<float(a)<=float(m)):return False,"PSAR: require 0 < acceleration <= maximum"
    if canonical=="candlestick_pattern":
        pattern=str(values.get("pattern") or "").strip().lower().removeprefix("cdl_")
        if pattern not in _CANDLE_PATTERNS:return False,f"candlestick_pattern: unsupported pattern={pattern!r}"
        err=_range(canonical,values,"penetration",0.0,1.0)
        if err:return False,err
    for name in ("tolerance","min_depth","shoulder_tolerance","head_min_prominence","max_neckline_slope","slope_threshold","parallel_tolerance","min_impulse","max_retracement","max_width","multiplier","volume_scale","volume_decay_threshold","threshold","hump"):
        if name in values:
            err=_range(canonical,values,name,0.0)
            if err:return False,err
    if "min_coverage" in values:
        err=_range(canonical,values,"min_coverage",0.0,1.0,strict_lo=True)
        if err:return False,err
    if "ddof" in values and _integer(values["ddof"]) not in {0,1}:return False,f"{canonical}: ddof must be 0 or 1"
    if "std_dev" in values and (not _finite(values["std_dev"]) or float(values["std_dev"])<=0):return False,f"{canonical}: std_dev must be > 0"
    if "add_intercept" in values and not isinstance(values["add_intercept"],bool):return False,f"{canonical}: add_intercept must be boolean"
    if canonical=="clip" and "lo" in values and "hi" in values and not (_finite(values["lo"]) and _finite(values["hi"]) and float(values["lo"])<=float(values["hi"])):return False,"clip: require finite lo <= hi"
    if canonical in {"winsorize","group_winsorize"}:
        lo=values.get("lo",values.get("lower"));hi=values.get("hi",values.get("upper"))
        if lo is not None and hi is not None and not (_finite(lo) and _finite(hi) and 0<=float(lo)<float(hi)<=1):return False,f"{canonical}: invalid winsor bounds"
    for name,value in values.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool) and not _finite(value):return False,f"{canonical}: {name} must be finite"
    return True,""

def check_pandas_first_plan_signatures(plan:Any)->list[str]:
    from factor_engine.cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    errors=[]
    def walk(node):
        op=str(getattr(node,"op","") or "")
        if op:
            canonical=OperatorRegistry._aliases.get(op,op)
            if canonical in PANDAS_FIRST_PRODUCTION_CANONICALS:
                ok,msg=validate_pandas_first_call(canonical,node)
                if not ok:errors.append(msg)
        for child in getattr(node,"inputs",[]) or []:walk(child)
    walk(plan);return errors
