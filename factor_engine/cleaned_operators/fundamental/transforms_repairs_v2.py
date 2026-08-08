# -*- coding: utf-8 -*-
"""Production repairs/extensions for fundamental-v2 transforms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fundamental.transforms_v2 import (
    _pos_int,
    _streak_span,
    _walk_periods,
    fin_ratio,
)


def _register(name,params,fn,desc):
    meta=OperatorMetadata(name=name,category="fundamental_period",description=desc,param_names=list(params),return_type="series",tags=["fundamental","period_aware","pit_safe","causal","bounded_history","production_repair"])
    def _calculate_series(self,*args,**kwargs):return fn(*args,**kwargs)
    cls=type(f"FundamentalRepair_{name}",(SeriesOperator,),{"metadata":meta,"_calculate_series":_calculate_series,"__module__":__name__})
    register_operator(name=name,category="fundamental_period",business_category="fundamental",canonical=name,source="fundamental_transforms_repairs_v2",backend="pandas_numpy",status="production")(cls)


def _streak(x,period_id,max_periods,positive):
    # A streak counts report-to-report changes only between *adjacent* fiscal
    # periods; a skipped report ends the streak (review R4-23).
    n=_pos_int(max_periods,"max_periods",2)
    return _walk_periods(
        x, period_id, lambda o, v, c: _streak_span(o, v, c, positive=positive, max_periods=n)
    )
def fin_positive_streak(x,period_id,max_periods=8):return _streak(x,period_id,max_periods,True)
def fin_negative_streak(x,period_id,max_periods=8):return _streak(x,period_id,max_periods,False)
def fin_cash_earnings_gap(earnings,cashflow,scale_base):return fin_ratio(earnings-cashflow,scale_base.abs())


def _revision_masks(x, period_id):
    """Complete-data masks for revision detection.

    A revision can only be asserted when the current value, prior value, current
    period id and prior period id are ALL present.  Any missing input (x or
    period_id NaN, or a prior observation missing because of a data gap) makes
    the row undetermined -> the caller must emit NaN, never a guessed 0
    (review R4-26).  The very first row of the panel is a valid baseline: there
    is no prior disclosure to revise, so it is complete and reads as "no
    revision" (0), matching the pandas/polars parity contract.
    """
    pid = period_id.reindex(index=x.index, columns=x.columns)
    prev_x = x.shift(1)
    prev_pid = pid.shift(1)
    first_row = pd.DataFrame(False, index=x.index, columns=x.columns)
    first_row.iloc[0] = True
    prev_ok = prev_x.notna() & prev_pid.notna()
    complete = x.notna() & pid.notna() & (prev_ok | first_row)
    same = pid.eq(prev_pid) & prev_ok
    delta = x - prev_x
    changed = delta.abs().gt(0) & prev_ok
    return complete, same, changed, delta


def _revision_event(x, period_id):
    complete, same, changed, _ = _revision_masks(x, period_id)
    return complete & same & changed


def _revision_complete(x, period_id):
    pid = period_id.reindex(index=x.index, columns=x.columns)
    prev_ok = x.shift(1).notna() & pid.shift(1).notna()
    first_row = pd.DataFrame(False, index=x.index, columns=x.columns)
    first_row.iloc[0] = True
    return x.notna() & pid.notna() & (prev_ok | first_row)


def fin_revision_delta(x, period_id):
    complete, same, changed, delta = _revision_masks(x, period_id)
    revision = complete & same & changed
    # 0 only when the data is complete and a same-period revision is confirmed
    # absent; any missing input -> NaN (review R4-26).
    out = delta.where(revision, 0.0)
    return out.where(complete, np.nan)


def fin_revision_pct(x, period_id):
    complete, same, changed, _ = _revision_masks(x, period_id)
    revision = complete & same & changed
    prev_x = x.shift(1)
    denom = prev_x.where(prev_x.ne(0))
    pct = (x / denom - 1.0).replace([np.inf, -np.inf], np.nan)
    out = pct.where(revision, 0.0)
    return out.where(complete, np.nan)


def fin_revision_direction(x, period_id):
    return np.sign(fin_revision_delta(x, period_id))


def fin_revision_count(x, period_id, window_days=252):
    w = _pos_int(window_days, "window_days")
    count = _revision_event(x, period_id).astype(float).rolling(w, min_periods=1).sum()
    return count.where(_revision_complete(x, period_id), np.nan)


def fin_revision_magnitude(x, period_id, window_days=252):
    w = _pos_int(window_days, "window_days")
    magnitude = fin_revision_pct(x, period_id).abs().rolling(w, min_periods=1).sum()
    return magnitude.where(_revision_complete(x, period_id), np.nan)


def fin_restated_flag(x, period_id, window_days=252):
    count = fin_revision_count(x, period_id, window_days)
    flag = count.gt(0).astype(float)
    return flag.where(count.notna(), np.nan)


def fin_days_since_update(x, period_id, max_days=504):
    """Bounded trading days since the last confirmed report-period/value update.

    Observed-clock semantics (review R4-27): the age advances only across
    *complete*, confirmed no-update observations.  A missing current row (value
    or period id unavailable) emits NaN instead of blindly ageing, and a
    resumed observation after an unobservable gap is treated as a fresh update
    boundary (age 0) — we cannot confirm the age across the gap.
    """
    cap = _pos_int(max_days, "max_days")
    pid = period_id.reindex(index=x.index, columns=x.columns)
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        xv = x[col].to_numpy(dtype=float)
        pv = pid[col].to_numpy()
        age = None
        last_x = None
        last_pid = None
        arr = []
        for i in range(len(x)):
            complete = bool(np.isfinite(xv[i]) and not pd.isna(pv[i]))
            if not complete:
                # Cannot observe an update event today -> age is unknown.
                arr.append(np.nan)
                age = None
                continue
            if last_x is None:
                # First complete observation: the value just became visible.
                arr.append(0.0)
                age = 0
                last_x, last_pid = xv[i], pv[i]
                continue
            update_event = bool(last_pid != pv[i] or last_x != xv[i])
            if update_event:
                arr.append(0.0)
                age = 0
            elif age is not None:
                # Only advance across complete, confirmed no-update observations
                # (observed clock).
                age = min(cap, age + 1)
                arr.append(float(age))
            else:
                # Observed-clock resume after a gap: treat as a fresh update.
                arr.append(0.0)
                age = 0
            last_x, last_pid = xv[i], pv[i]
        out[col] = arr
    return out


def fin_staleness(x, period_id, max_days=504):
    return fin_days_since_update(x, period_id, max_days)

_EXTRA=set()
for _name,_params,_fn,_desc in [
("fin_positive_streak",["x","period_id","max_periods"],fin_positive_streak,"Bounded streak of positive report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
("fin_negative_streak",["x","period_id","max_periods"],fin_negative_streak,"Bounded streak of negative report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
("fin_cash_earnings_gap",["earnings","cashflow","scale_base"],fin_cash_earnings_gap,"Scaled earnings-minus-cashflow gap."),
("fin_revision_delta",["x","period_id"],fin_revision_delta,"Value change while the visible report period is unchanged; NaN when any required input is missing, 0 only on confirmed no-revision (review R4-26)."),
("fin_revision_pct",["x","period_id"],fin_revision_pct,"Percent revision while the visible report period is unchanged; NaN on missing inputs, 0 only on confirmed no-revision (review R4-26)."),
("fin_revision_direction",["x","period_id"],fin_revision_direction,"Sign of the latest same-period revision; NaN on missing inputs (review R4-26)."),
("fin_revision_count",["x","period_id","window_days"],fin_revision_count,"Count of visible revisions in a bounded trading-day window; NaN where the current row is incomplete (review R4-26)."),
("fin_revision_magnitude",["x","period_id","window_days"],fin_revision_magnitude,"Absolute revision magnitude accumulated over a bounded window; NaN where the current row is incomplete (review R4-26)."),
("fin_restated_flag",["x","period_id","window_days"],fin_restated_flag,"Whether a same-period revision occurred in the bounded window; NaN where the count is undetermined (review R4-26)."),
("fin_days_since_update",["x","period_id","max_days"],fin_days_since_update,"Observed-clock trading days since report-period or value update; missing rows emit NaN instead of blindly ageing (review R4-27)."),
("fin_staleness",["x","period_id","max_days"],fin_staleness,"Observed-clock accounting-data staleness in trading days; missing rows emit NaN (review R4-27)."),
]:
    _register(_name,_params,_fn,_desc);_EXTRA.add(_name)

# Surface extension is performed during bootstrap before production tiers and DSL
# allowlists are consumed.
import cleaned_operators.operator_surface as _surface
_surface._FUNDAMENTAL_V2_CANONICALS=frozenset(set(_surface._FUNDAMENTAL_V2_CANONICALS)|_EXTRA)
_surface.EXTENDED_ONLY_CANONICALS=frozenset(set(_surface.EXTENDED_ONLY_CANONICALS)|_EXTRA)
