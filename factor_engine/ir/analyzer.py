"""表达式 → IR：Analyzer 把 ``Expr`` 树降为可执行的 ``IRNode``。

技术结构、财报期和递归指标必须显式声明隐藏 lookback，避免 production
warmup 只看到表面参数而低估历史读取量。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from expr.base import Expr
from expr.cleaned_call import CleanedCall
from expr.column import ColumnRef
from expr.literal import Literal
from ir.nodes import IRNode

_LAG_PARAM_NAMES={
    "ts_delay":("n","d","lag","periods","window"),"prev":(),
    "ts_delta":("n","d","lag","periods","window"),
    "ts_pct":("d","n","lag","periods","window"),
    "ts_log_return":("d","n","lag","periods","window"),"ts_ratio":(),
    "MOM":("window","d","n"),"ROC":("window","d","n"),
}
_FIXED_LAGS={"prev":1,"ts_ratio":1,"candle_gap":1,"candle_gap_pct":1,"cdl_engulfing":1,"cdl_inside_bar":1,"cdl_outside_bar":1}
_WINDOW_PARAM_NAMES=(
    "window","d","span","period","periods","lookback","max_lookback","fast","slow","fast_period","slow_period","signal_span","signal_period",
    "fast_window","slow_window","signal_window","short_window","medium_window","long_window","ema_window","atr_window","tenkan_window","kijun_window","senkou_b_window","er_window","vol_window","baseline_window","price_window","volume_window","turnover_window","adl_window","impulse_window","flag_window","max_periods","window_days","max_days",
)
_WINDOW_PLUS_ONE_CANONICALS=frozenset({
    "ts_prev_high","ts_prev_low","ts_distance_to_high","ts_distance_to_low","ts_breakout_high","ts_breakdown_low","ts_new_high","ts_new_low","ts_channel_position","ts_days_since_high","ts_days_since_low","ts_range_expansion","donchian_upper","donchian_lower","donchian_mid","donchian_position","relative_volume","volume_zscore","dollar_volume_zscore","volume_momentum","turnover_momentum","turnover_zscore","rolling_obv","rolling_pvt","MFI","choppiness_index","yang_zhang_vol","overnight_volatility","candle_range_atr","abnormal_volume","abnormal_turnover","volume_shock","turnover_shock",
})
_PIVOT_CONFIRM_CANONICALS=frozenset({"ts_confirmed_pivot_high","ts_confirmed_pivot_low"})
_BOUNDED_STRUCTURE_CANONICALS=frozenset({
    "ts_last_pivot_high","ts_last_pivot_low","ts_pivot_high_age","ts_pivot_low_age","ts_resistance_level","ts_support_level","ts_resistance_slope","ts_support_slope","ts_distance_to_resistance","ts_distance_to_support","ts_resistance_break","ts_support_break",
})
_STRUCTURE_PREFIXES=("ts_nth_pivot_","ts_pivot_","ts_swing_","ts_channel_","ts_line_","ts_resistance_fit_","ts_support_fit_","ts_pattern_")
_STRUCTURE_PATTERNS=frozenset({
    "pattern_double_top","pattern_double_bottom","pattern_head_shoulders","pattern_inverse_head_shoulders","pattern_sym_triangle","pattern_ascending_triangle","pattern_descending_triangle","pattern_rising_wedge","pattern_falling_wedge","pattern_rectangle","pattern_rising_channel","pattern_falling_channel","pattern_broadening",
})
_FLAG_PATTERNS=frozenset({"pattern_bull_flag","pattern_bear_flag"})
_FIN_DAILY_WINDOW_CANONICALS=frozenset({"fin_revision_count","fin_revision_magnitude","fin_restated_flag","fin_days_since_update","fin_staleness"})


def _positive_int(value:Any)->int|None:
    if isinstance(value,bool):return None
    try:parsed=int(value)
    except (TypeError,ValueError):return None
    return parsed if parsed>0 else None

def _literal_value(expr:Expr)->Any|None:return expr.value if isinstance(expr,Literal) else None

def _operator_param_values(node:CleanedCall,op_impl:Any)->dict[str,Any]:
    values=dict(node.kwargs_dict());param_names=tuple(getattr(getattr(op_impl,"metadata",None),"param_names",()) or ())
    for index,arg in enumerate(node.args):
        if index>=len(param_names):break
        literal=_literal_value(arg)
        if literal is not None or isinstance(arg,Literal):values.setdefault(param_names[index],literal)
    return values

def _report_period_count(canon:str,p:dict[str,Any])->int:
    def pos(name,default=0):return _positive_int(p.get(name)) or default
    if canon=="fin_growth_change":return pos("growth_periods",4)+pos("compare_periods",1)
    if canon in {"fin_growth_volatility","fin_growth_stability","fin_growth_persistence"}:return pos("growth_periods",1)+pos("window_periods",8)
    if canon=="fin_trend_acceleration":return max(pos("short_periods",4),pos("long_periods",8))
    if canon=="fin_turnover":return pos("average_periods",2)
    if canon in {"fin_positive_streak","fin_negative_streak"}:return pos("max_periods",8)
    if canon in {"fin_yoy","fin_ttm"}:return pos("periods_per_year",4)
    return max(1,pos("periods"),pos("window_periods"),pos("average_periods"),pos("long_periods"),pos("max_periods"))
def _financial_lookback(canon:str,p:dict[str,Any])->int:
    if not canon.startswith("fin_") or canon in _FIN_DAILY_WINDOW_CANONICALS:return 0
    rows_per_period=_positive_int(os.environ.get("FACTOR_ENGINE_REPORT_PERIOD_LOOKBACK_ROWS","80")) or 80
    return rows_per_period*_report_period_count(canon,p)


def _operator_lookback_increment(canon:str,node:CleanedCall,op_impl:Any,policy:Any|None)->int:
    params=_operator_param_values(node,op_impl);increment=_financial_lookback(canon,params)
    if canon in _FIXED_LAGS:increment=max(increment,_FIXED_LAGS[canon])
    lag_names=_LAG_PARAM_NAMES.get(canon)
    if lag_names is not None:
        for name in lag_names:
            value=_positive_int(params.get(name))
            if value is not None:increment=max(increment,value)
        if policy is not None:
            lag=_positive_int(getattr(policy,"lag",None))
            if lag is not None:increment=max(increment,lag)
        return increment
    left=_positive_int(params.get("left_window")) or 0;right=_positive_int(params.get("right_window")) or 0;history=_positive_int(params.get("history_window")) or 0
    if canon in _PIVOT_CONFIRM_CANONICALS:increment=max(increment,left+right)
    if canon in _BOUNDED_STRUCTURE_CANONICALS or canon in _STRUCTURE_PATTERNS or canon.startswith(_STRUCTURE_PREFIXES):
        if history:increment=max(increment,max(0,history-1)+left+right)
    for name in _WINDOW_PARAM_NAMES:
        value=_positive_int(params.get(name))
        if value is not None:
            increment=max(increment,value-1)
            if canon in _WINDOW_PLUS_ONE_CANONICALS:increment=max(increment,value)
    if canon=="StochasticD":
        w=_positive_int(params.get("window"));increment=max(increment,(w+1) if w else 0)
    if canon=="ulcer_index":
        w=_positive_int(params.get("window"));increment=max(increment,2*(w-1)) if w else increment
    if canon in {"MACD","MACD_line","MACD_signal","MACD_hist"}:
        fast=_positive_int(params.get("fast")) or 0;slow=_positive_int(params.get("slow")) or 0;signal=_positive_int(params.get("signal")) or 0;increment=max(increment,max(fast,slow)-1+max(signal-1,0))
    if canon in {"PPO","PPO_signal","PPO_hist","PVO","PVO_signal","PVO_hist"}:
        slow=_positive_int(params.get("slow_window")) or 0;signal=_positive_int(params.get("signal_window")) or 0;increment=max(increment,max(0,slow-1)+max(0,signal-1))
    if canon in {"TSI","TSI_signal"}:
        lw=_positive_int(params.get("long_window")) or 0;sw=_positive_int(params.get("short_window")) or 0;sg=_positive_int(params.get("signal_window")) or 0;increment=max(increment,max(0,lw-1)+max(0,sw-1)+max(0,sg-1))
    if canon in {"DEMA","TEMA"}:
        w=_positive_int(params.get("window")) or 0;increment=max(increment,(3 if canon=="TEMA" else 2)*max(0,w-1))
    if canon.startswith("ichimoku_"):
        for name in ("tenkan_window","kijun_window","senkou_b_window"):
            v=_positive_int(params.get(name));increment=max(increment,max(0,(v or 1)-1))
    if canon in _FLAG_PATTERNS:
        iw=_positive_int(params.get("impulse_window")) or 0;fw=_positive_int(params.get("flag_window")) or 0;increment=max(increment,iw+fw)
    if canon=="ts_impulse_volume":
        w=_positive_int(params.get("window")) or 0;b=_positive_int(params.get("baseline_window")) or 0;increment=max(increment,w+b)
    if canon=="ts_channel_width_slope" and history:
        w=_positive_int(params.get("window")) or 1;increment=max(increment,max(0,history-1)+left+right+max(0,w-1))
    if policy is not None:
        lag=_positive_int(getattr(policy,"lag",None));lookback_window=_positive_int(getattr(policy,"lookback_window",None));min_periods=_positive_int(getattr(policy,"min_periods",None))
        if lag is not None:increment=max(increment,lag)
        if lookback_window is not None:increment=max(increment,lookback_window-1)
        if min_periods is not None:increment=max(increment,min_periods-1)
    return increment


def _legacy_single_window_lookback(expr:Expr)->int|None:
    def visit(node:Expr)->int|None:
        if isinstance(node,(ColumnRef,Literal)):return 0
        if not isinstance(node,CleanedCall):return None
        from backend.cleaned_bridge import ensure_cleaned_loaded
        from cleaned_operators.registry import OperatorRegistry
        ensure_cleaned_loaded();canon=OperatorRegistry.resolve_canonical_strict(node.op);op_impl=OperatorRegistry.get(canon)
        if op_impl is None:return None
        from cleaned_operators.operator_policy import infer_operator_policy
        increment=_operator_lookback_increment(canon,node,op_impl,infer_operator_policy(op_impl,canonical=canon))
        expr_children=[arg for arg in node.args if isinstance(arg,Expr)];series_children=[arg for arg in expr_children if not isinstance(arg,Literal)]
        if increment>0:
            if len(series_children)!=1 or not isinstance(series_children[0],ColumnRef):return None
            params=_operator_param_values(node,op_impl)
            for name in _WINDOW_PARAM_NAMES:
                window=_positive_int(params.get(name))
                if window is not None:return max(window,increment)
            return None
        child_windows=[visit(child) for child in series_children]
        if any(window is None for window in child_windows):return None
        return max((window or 0 for window in child_windows),default=0)
    result=visit(expr);return result if result and result>0 else None

@dataclass
class AnalysisResult:
    ir:IRNode;lookback:int;has_ts_op:bool;has_cs_op:bool;referenced_columns:set[str]

class Analyzer:
    def lower(self,expr:Expr)->AnalysisResult:
        cols:set[str]=set();has_ts=False;has_cs=False
        def visit(node:Expr)->tuple[IRNode,int]:
            nonlocal has_ts,has_cs
            if isinstance(node,ColumnRef):cols.add(node.name);return IRNode(op="column",attrs={"name":node.name}),0
            if isinstance(node,Literal):return IRNode(op="literal",attrs={"value":node.value}),0
            if isinstance(node,CleanedCall):
                from backend.cleaned_bridge import ensure_cleaned_loaded
                from cleaned_operators.registry import OperatorRegistry
                ensure_cleaned_loaded()
                if node.op in {"bfill","causal_bfill"}:
                    from backend.polars_long_policy import UnsupportedCausalOperatorError
                    raise UnsupportedCausalOperatorError(f"{node.op} is disabled: backward-looking fill is not point-in-time safe")
                canon=OperatorRegistry.resolve_canonical_strict(node.op);op_impl=OperatorRegistry.get(canon)
                if op_impl is not None:
                    cat=getattr(op_impl.metadata,"category","") or ""
                    if cat in ("time_series","shift_diff_cum","technical_signal","price_volume","price_volume_extension","ohlc_volatility","candle_pattern","intraday_microstructure","signal","price_structure","chart_pattern"):has_ts=True
                    if cat in ("cross_sectional","group_neutralization"):has_cs=True
                if canon.startswith("ts_") or canon in {"SMA","EMA","WMA","delay","decay_linear"}:has_ts=True
                if canon in {"rank","zscore","scale","normalize","winsorize","quantile","neutralize"} or canon.startswith("group_"):has_cs=True
                visited=[visit(arg) for arg in node.args];inputs=tuple(x[0] for x in visited);deepest=max((x[1] for x in visited),default=0);attrs={}
                for key,value in node.kwargs_dict().items():
                    if isinstance(value,Expr):
                        lowered,kw_lb=visit(value)
                        if lowered.op!="literal":raise NotImplementedError(f"cleaned op {node.op!r} kwargs must be literals, got {key!r}")
                        attrs[key]=lowered.attrs["value"];deepest=max(deepest,kw_lb)
                    else:attrs[key]=value
                from backend.parameter_aliases import normalize_parameter_aliases
                attrs=normalize_parameter_aliases(canon,attrs);policy=None
                if op_impl is not None:
                    from cleaned_operators.operator_policy import infer_operator_policy
                    policy=infer_operator_policy(op_impl,canonical=canon)
                return IRNode(op=canon,inputs=inputs,attrs=attrs),deepest+_operator_lookback_increment(canon,node,op_impl,policy)
            raise NotImplementedError(f"Unsupported expr: {type(node).__name__}")
        ir,lookback=visit(expr);legacy=_legacy_single_window_lookback(expr)
        if legacy is not None:lookback=max(lookback,legacy)
        return AnalysisResult(ir=ir,lookback=lookback,has_ts_op=has_ts,has_cs_op=has_cs,referenced_columns=cols)
