# -*- coding: utf-8 -*-
"""Production repairs/extensions for fundamental-v2 transforms."""
from __future__ import annotations
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
    _pos_int,
    _streak_span,
    _walk_periods,
    fin_ratio,
)


def _register(name,params,fn,desc,tags=()):
    meta=OperatorMetadata(name=name,category="fundamental_period",description=desc,param_names=list(params),return_type="series",tags=["fundamental","period_aware","pit_safe","causal","bounded_history","production_repair",*tags])
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

    R23-039: period_id is strict-aligned to x (never silently reindexed) — a
    period-id panel on a different date/instrument grid is a caller bug.
    """
    from factor_engine.cleaned_operators.alignment import align_panel_inputs

    x, period_id = align_panel_inputs(x, period_id, names=("x", "period_id"))
    pid = period_id
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
    from factor_engine.cleaned_operators.alignment import align_panel_inputs

    x, period_id = align_panel_inputs(x, period_id, names=("x", "period_id"))
    pid = period_id
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


_REVISION_COVERAGE_THRESHOLD = 0.8


def _revision_coverage_gate(x, period_id, window_days: int, coverage_threshold: float):
    """Three-state revision observation plus a known-coverage gate.

    Three-state (round-7 P0): a daily row is a revision (1), a confirmed
    no-revision (0), or undetermined (NaN) when any input needed to assert a
    revision is missing (a gap in the value or period-id panel).  The rolling
    window statistic is only emitted when the fraction of *known* rows in the
    window is at least ``coverage_threshold`` — otherwise the running sum would
    silently treat missing history as "no revision" (rolling ``sum`` skips NaN).
    """
    complete = _revision_complete(x, period_id)
    known = complete.astype(float)  # 1.0 known, 0.0 undetermined
    known_sum = known.rolling(window_days, min_periods=1).sum()
    coverage = known_sum / float(window_days)
    return complete, coverage


def fin_revision_count(x, period_id, window_days=252, coverage_threshold=0.8):
    """Count of visible revisions in a bounded trading-day window.

    Three-state observation (round-7 P0): missing historical input is NOT "no
    revision" — the count is emitted only when ``known_coverage`` (fraction of
    window rows with complete revision data) reaches ``coverage_threshold``,
    else NaN.
    """
    w = _pos_int(window_days, "window_days")
    threshold = float(coverage_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("coverage_threshold must be in [0, 1]")
    complete, coverage = _revision_coverage_gate(x, period_id, w, threshold)
    # Three-state revision series: 1.0 where a revision is observed, 0.0 where a
    # no-revision is confirmed, NaN where the row is undetermined.
    revision = _revision_event(x, period_id).astype(float).where(complete, np.nan)
    count = revision.rolling(w, min_periods=1).sum()
    return count.where(coverage >= threshold, np.nan).where(complete, np.nan)


def fin_revision_magnitude(x, period_id, window_days=252, coverage_threshold=0.8):
    """Absolute revision magnitude accumulated over a bounded window.

    The magnitude of a same-period revision is accumulated only over rows with
    complete revision data; a window whose known coverage is below
    ``coverage_threshold`` emits NaN rather than treating missing history as a
    zero-magnitude "no revision" (round-7 P0).
    """
    w = _pos_int(window_days, "window_days")
    threshold = float(coverage_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("coverage_threshold must be in [0, 1]")
    complete, coverage = _revision_coverage_gate(x, period_id, w, threshold)
    # fin_revision_pct already emits NaN for undetermined rows, 0 for confirmed
    # no-revision, and the signed revision magnitude for a revision.
    magnitude = fin_revision_pct(x, period_id).abs()
    mag_sum = magnitude.rolling(w, min_periods=1).sum()
    return mag_sum.where(coverage >= threshold, np.nan).where(complete, np.nan)


def fin_restated_flag(x, period_id, window_days=252, coverage_threshold=0.8):
    """Whether a same-period revision occurred in the bounded window.

    Inherits the three-state / known-coverage gate of ``fin_revision_count``:
    NaN where the count is undetermined (round-7 P0).
    """
    count = fin_revision_count(x, period_id, window_days, coverage_threshold)
    flag = count.gt(0).astype(float)
    return flag.where(count.notna(), np.nan)


def fin_days_since_update(x, period_id, max_days=504):
    """Bounded trading days since the last confirmed report-period/value update.

    Observed-clock semantics (review R4-27): the age advances only across
    *complete*, confirmed no-update observations.  A missing current row (value
    or period id unavailable) emits NaN instead of blindly ageing.

    Round-7 P1 gap recovery: ``last_confirmed_economic_update_time`` is tracked
    separately from ``last_observed_time``.  A value that resumes after a
    provider gap and is IDENTICAL to the last confirmed state is a provider-
    outage recovery, NOT an economic disclosure update — the age is NOT reset to
    0; the elapsed unobservable days are added to the confirmed age.  Only an
    observed change in value or report period is an economic update and resets
    the age.

    R23-094..097 / R23-298 (left censoring): the FIRST complete observation is
    NOT a new report event — when a backtest sample starts mid-history the
    report may have been published weeks earlier, so age=0 would under-report
    staleness.  Without a knowledge_time / PubDate the true publication time
    before the sample start is unknown -> the first observation is
    ``left_censored`` and emits NaN.  The age clock only starts at the first
    genuinely OBSERVED update (a value/period change vs the retained confirmed
    state); rows between the censored anchor and the first observed update
    stay NaN.
    """
    from factor_engine.cleaned_operators.alignment import align_panel_inputs

    cap = _pos_int(max_days, "max_days")
    x, period_id = align_panel_inputs(x, period_id, names=("x", "period_id"))
    pid = period_id
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        xv = x[col].to_numpy(dtype=float)
        pv = pid[col].to_numpy()
        age = None       # trading days since the last OBSERVED economic update
        last_x = None    # last confirmed value (retained across gaps)
        last_pid = None  # last confirmed period id (retained across gaps)
        gap_days = 0     # unobservable trading days currently in a provider gap
        arr = []
        for i in range(len(x)):
            complete = bool(np.isfinite(xv[i]) and not pd.isna(pv[i]))
            if not complete:
                # Cannot observe an update event today -> age is unknown.  The
                # confirmed state is retained across the gap so a resumed value
                # identical to the pre-gap value is not mistaken for an update.
                arr.append(np.nan)
                gap_days += 1
                continue
            if last_x is None:
                # R23-298 left censor: first complete observation anchors the
                # confirmed state but is NOT a proven new report — the age is
                # unknown (the report may be weeks old), so NaN until a truly
                # observed update.
                arr.append(np.nan)
                gap_days = 0
                last_x, last_pid = xv[i], pv[i]
                continue
            update_event = bool(last_pid != pv[i] or last_x != xv[i])
            if update_event:
                # A real economic disclosure update: reset the confirmed age.
                arr.append(0.0)
                age = 0
                gap_days = 0
            else:
                # Confirmed no economic update.  A value identical to the last
                # confirmed state after a provider gap is NOT an update — the age
                # continues (confirmed age + elapsed unobservable days + today).
                # Between the censored anchor and the first observed update the
                # age stays NaN (still left-censored).
                if age is None:
                    arr.append(np.nan)
                else:
                    age = min(cap, age + gap_days + 1)
                    arr.append(float(age))
                gap_days = 0
            last_x, last_pid = xv[i], pv[i]
        out[col] = arr
    return out


def fin_staleness(x, period_id, max_days=504):
    return fin_days_since_update(x, period_id, max_days)

_EXTRA=set()
# Non-revision repair operators (streaks / cash-earnings gap) are ordinary
# PIT-safe period operators.
for _name,_params,_fn,_desc in [
("fin_positive_streak",["x","period_id","max_periods"],fin_positive_streak,"Bounded streak of positive report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
("fin_negative_streak",["x","period_id","max_periods"],fin_negative_streak,"Bounded streak of negative report-to-report changes between adjacent fiscal periods; a skipped report ends the streak (review R4-23)."),
("fin_cash_earnings_gap",["earnings","cashflow","scale_base"],fin_cash_earnings_gap,"Scaled earnings-minus-cashflow gap."),
]:
    _register(_name,_params,_fn,_desc);_EXTRA.add(_name)

# R23-297: the revision family infers same-period revisions from DAILY PANEL
# state change (value changed while the visible report period is unchanged).
# That is only a valid revision proof when the source provides true historical
# vintage state (a RevisionEventSource / bitemporal source); a re-synced COS
# final-value history cannot recover revision events.  The operators therefore
# DECLARE ``requires:RevisionEventSource`` and ``revision_vintage_pit_certified:false``
# on their contract so production mining/admission can block them until the
# source proves immutable vintage history — never claim ``pit_safe`` alone.
_REVISION_SOURCE_TAGS = ("requires:RevisionEventSource", "revision_vintage_pit_certified:false")
for _name,_params,_fn,_desc in [
("fin_revision_delta",["x","period_id"],fin_revision_delta,"Value change while the visible report period is unchanged; NaN when any required input is missing, 0 only on confirmed no-revision (review R4-26)."),
("fin_revision_pct",["x","period_id"],fin_revision_pct,"Percent revision while the visible report period is unchanged; NaN on missing inputs, 0 only on confirmed no-revision (review R4-26)."),
("fin_revision_direction",["x","period_id"],fin_revision_direction,"Sign of the latest same-period revision; NaN on missing inputs (review R4-26)."),
("fin_revision_count",["x","period_id","window_days","coverage_threshold"],fin_revision_count,"Count of visible revisions in a bounded trading-day window; three-state observation with a known-coverage gate — a window below coverage_threshold emits NaN instead of treating missing history as no-revision (round-7 P0)."),
("fin_revision_magnitude",["x","period_id","window_days","coverage_threshold"],fin_revision_magnitude,"Absolute revision magnitude accumulated over a bounded window; three-state observation with a known-coverage gate (round-7 P0)."),
("fin_restated_flag",["x","period_id","window_days","coverage_threshold"],fin_restated_flag,"Whether a same-period revision occurred in the bounded window; NaN where the count is undetermined by the known-coverage gate (round-7 P0)."),
("fin_days_since_update",["x","period_id","max_days"],fin_days_since_update,"Observed-clock trading days since report-period or value update; missing rows emit NaN instead of blindly ageing, a provider-gap recovery identical to the pre-gap value does NOT reset the confirmed-update age (round-7 P1), and the FIRST sample observation is left-censored (NaN) rather than age 0 (R23-298)."),
("fin_staleness",["x","period_id","max_days"],fin_staleness,"Observed-clock accounting-data staleness in trading days; missing rows emit NaN (review R4-27), first observation left-censored (R23-298)."),
]:
    _register(_name,_params,_fn,_desc,tags=_REVISION_SOURCE_TAGS);_EXTRA.add(_name)

# Surface extension is performed during bootstrap before production tiers and DSL
# allowlists are consumed.
import factor_engine.cleaned_operators.operator_surface as _surface
_surface._FUNDAMENTAL_V2_CANONICALS=frozenset(set(_surface._FUNDAMENTAL_V2_CANONICALS)|_EXTRA)
_surface.extend_extended_only(set(_EXTRA))
