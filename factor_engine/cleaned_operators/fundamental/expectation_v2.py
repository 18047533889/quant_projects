# -*- coding: utf-8 -*-
"""Generic PIT-aware actual/expectation and estimate-revision operators."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata,SeriesOperator,register_operator
from cleaned_operators.fundamental.transforms_v2 import _pos_int,_values,_walk_periods


def _register(name,params,fn,desc):
    meta=OperatorMetadata(name=name,category="fundamental_period",description=desc,param_names=list(params),return_type="series",tags=["fundamental","expectation","pit_safe","causal","production_extension"])
    def calc(self,*a,**k):return fn(*a,**k)
    cls=type(f"ExpectationV2_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":calc,"__module__":__name__})
    register_operator(name=name,category="fundamental_period",business_category="fundamental",canonical=name,source="fundamental_expectation_v2",backend="pandas_numpy",status="production")(cls)
def _safe(num,den):return num/den.replace(0,np.nan)
def fin_surprise(actual,expected,scale_base):return _safe(actual-expected,scale_base.abs())
def fin_surprise_zscore(actual,expected,scale_base,window_days):
    w=_pos_int(window_days,"window_days",2);s=fin_surprise(actual,expected,scale_base);m=s.shift(1).rolling(w,min_periods=w).mean();sd=s.shift(1).rolling(w,min_periods=w).std();return (s-m)/sd.replace(0,np.nan)
def fin_expectation_revision(expected,target_period_id):
    pid=target_period_id.reindex(index=expected.index,columns=expected.columns);same=pid.eq(pid.shift(1))&pid.notna();return (expected-expected.shift(1)).where(same,0.0)
def fin_expectation_revision_pct(expected,target_period_id):
    pid=target_period_id.reindex(index=expected.index,columns=expected.columns);same=pid.eq(pid.shift(1))&pid.notna();return (_safe(expected,expected.shift(1))-1).where(same,0.0)
def fin_expectation_revision_speed(expected,target_period_id,window_days):
    w=_pos_int(window_days,"window_days",2);return fin_expectation_revision_pct(expected,target_period_id).rolling(w,min_periods=1).sum()
def fin_expectation_dispersion(expected_std,expected_mean):return _safe(expected_std.abs(),expected_mean.abs())
def fin_actual_expectation_divergence(actual,expected,scale_base):return fin_surprise(actual,expected,scale_base)
def _beat_miss_streak(actual,expected,period_id,max_periods,beat):
    n=_pos_int(max_periods,"max_periods",2);diff=actual-expected
    def calc(o,v,c):
        vals=_values(o,v,c,n);count=0
        for z in vals[::-1]:
            ok=z>0 if beat else z<0
            if ok:count+=1
            else:break
        return float(count)
    return _walk_periods(diff,period_id,calc)
def fin_beat_streak(actual,expected,period_id,max_periods):return _beat_miss_streak(actual,expected,period_id,max_periods,True)
def fin_miss_streak(actual,expected,period_id,max_periods):return _beat_miss_streak(actual,expected,period_id,max_periods,False)

_NAMES=[]
for n,p,f,d in [
("fin_surprise",["actual","expected","scale_base"],fin_surprise,"Scaled actual-minus-expectation surprise."),
("fin_surprise_zscore",["actual","expected","scale_base","window_days"],fin_surprise_zscore,"Prior-window z-score of realized surprise."),
("fin_expectation_revision",["expected","target_period_id"],fin_expectation_revision,"Same-target-period change in analyst expectation."),
("fin_expectation_revision_pct",["expected","target_period_id"],fin_expectation_revision_pct,"Same-target-period percent expectation revision."),
("fin_expectation_revision_speed",["expected","target_period_id","window_days"],fin_expectation_revision_speed,"Bounded cumulative expectation revision speed."),
("fin_expectation_dispersion",["expected_std","expected_mean"],fin_expectation_dispersion,"Consensus dispersion scaled by absolute consensus mean."),
("fin_actual_expectation_divergence",["actual","expected","scale_base"],fin_actual_expectation_divergence,"Generic actual/expectation divergence."),
("fin_beat_streak",["actual","expected","period_id","max_periods"],fin_beat_streak,"Bounded consecutive positive surprise streak."),
("fin_miss_streak",["actual","expected","period_id","max_periods"],fin_miss_streak,"Bounded consecutive negative surprise streak."),
]:_register(n,p,f,d);_NAMES.append(n)
import cleaned_operators.operator_surface as _surface
_surface.EXTENDED_ONLY_CANONICALS=frozenset(set(_surface.EXTENDED_ONLY_CANONICALS)|set(_NAMES))
_surface._FUNDAMENTAL_V2_CANONICALS=frozenset(set(_surface._FUNDAMENTAL_V2_CANONICALS)|set(_NAMES))
