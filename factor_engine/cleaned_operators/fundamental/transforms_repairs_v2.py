# -*- coding: utf-8 -*-
"""Production repairs/extensions for fundamental-v2 transforms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import _pos_int, _values, _walk_periods, fin_ratio


def _register(name,params,fn,desc):
    meta=OperatorMetadata(name=name,category="fundamental_period",description=desc,param_names=list(params),return_type="series",tags=["fundamental","period_aware","pit_safe","causal","bounded_history","production_repair"])
    def _calculate_series(self,*args,**kwargs):return fn(*args,**kwargs)
    cls=type(f"FundamentalRepair_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="fundamental_period",business_category="fundamental",canonical=name,source="fundamental_transforms_repairs_v2",backend="pandas_numpy",status="production")(cls)


def _streak(x,period_id,max_periods,positive):
    n=_pos_int(max_periods,"max_periods",2)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n),dtype=float)
        if len(vals)<2:return 0.0
        count=0
        for z in np.diff(vals)[::-1]:
            condition=(z>0) if positive else (z<0)
            if condition:count+=1
            else:break
        return float(count)
    return _walk_periods(x,period_id,calc)
def fin_positive_streak(x,period_id,max_periods=8):return _streak(x,period_id,max_periods,True)
def fin_negative_streak(x,period_id,max_periods=8):return _streak(x,period_id,max_periods,False)
def fin_cash_earnings_gap(earnings,cashflow,scale_base):return fin_ratio(earnings-cashflow,scale_base.abs())


def _revision_event(x,period_id):
    pid=period_id.reindex(index=x.index,columns=x.columns)
    same=pid.eq(pid.shift(1)) & pid.notna()
    changed=(x-x.shift(1)).abs().gt(0) & x.notna() & x.shift(1).notna()
    return same & changed

def fin_revision_delta(x,period_id):return (x-x.shift(1)).where(_revision_event(x,period_id),0.0)
def fin_revision_pct(x,period_id):return ((x/x.shift(1).replace(0,np.nan))-1.0).where(_revision_event(x,period_id),0.0)
def fin_revision_direction(x,period_id):return np.sign(fin_revision_delta(x,period_id))
def fin_revision_count(x,period_id,window_days=252):
    w=_pos_int(window_days,"window_days");return _revision_event(x,period_id).astype(float).rolling(w,min_periods=1).sum()
def fin_revision_magnitude(x,period_id,window_days=252):
    w=_pos_int(window_days,"window_days");return fin_revision_pct(x,period_id).abs().rolling(w,min_periods=1).sum()
def fin_restated_flag(x,period_id,window_days=252):return fin_revision_count(x,period_id,window_days).gt(0).astype(float)
def fin_days_since_update(x,period_id,max_days=504):
    cap=_pos_int(max_days,"max_days")
    pid=period_id.reindex(index=x.index,columns=x.columns)
    update=pid.ne(pid.shift(1))|x.ne(x.shift(1)); out=pd.DataFrame(np.nan,index=x.index,columns=x.columns,dtype=float)
    for col in x.columns:
        age=cap
        arr=[]
        for i,flag in enumerate(update[col].fillna(False).to_numpy(bool)):
            if i==0 or flag:age=0
            else:age=min(cap,age+1)
            arr.append(float(age))
        out[col]=arr
    return out
def fin_staleness(x,period_id,max_days=504):return fin_days_since_update(x,period_id,max_days)

for _name,_params,_fn,_desc in [
("fin_positive_streak",["x","period_id","max_periods"],fin_positive_streak,"Bounded streak of positive report-to-report changes."),
("fin_negative_streak",["x","period_id","max_periods"],fin_negative_streak,"Bounded streak of negative report-to-report changes."),
("fin_cash_earnings_gap",["earnings","cashflow","scale_base"],fin_cash_earnings_gap,"Scaled earnings-minus-cashflow gap."),
("fin_revision_delta",["x","period_id"],fin_revision_delta,"Value change while the visible report period is unchanged."),
("fin_revision_pct",["x","period_id"],fin_revision_pct,"Percent revision while the visible report period is unchanged."),
("fin_revision_direction",["x","period_id"],fin_revision_direction,"Sign of the latest same-period revision."),
("fin_revision_count",["x","period_id","window_days"],fin_revision_count,"Count of visible revisions in a bounded trading-day window."),
("fin_revision_magnitude",["x","period_id","window_days"],fin_revision_magnitude,"Absolute revision magnitude accumulated over a bounded window."),
("fin_restated_flag",["x","period_id","window_days"],fin_restated_flag,"Whether a same-period revision occurred in the bounded window."),
("fin_days_since_update",["x","period_id","max_days"],fin_days_since_update,"Bounded trading days since report-period or value update."),
("fin_staleness",["x","period_id","max_days"],fin_staleness,"Bounded accounting-data staleness in trading days."),
]:_register(_name,_params,_fn,_desc)
