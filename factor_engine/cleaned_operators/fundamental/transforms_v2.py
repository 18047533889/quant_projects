# -*- coding: utf-8 -*-
"""Generic PIT-safe fundamental transformation operators.

These operators are intentionally field-agnostic.  They operate on a daily
as-of panel ``x`` together with a daily-aligned ``period_id`` panel identifying
the report period currently visible at each decision timestamp.  Calculations
advance only when a new report period becomes visible; a later revision of an
already-visible period updates results only from the revision timestamp onward.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_strict import period_ordinal

_EPS = 1e-12


def _pos_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= _EPS:
        return np.nan
    return float(a / b)


def _period_key(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value) if isinstance(value, (str, np.datetime64)) else value
    except Exception:
        return value


def _period_insert(order: list[object], key: object) -> None:
    """Insert ``key`` into ``order`` keeping fiscal-ordinal sorted order.

    Report periods must advance by fiscal-quarter ordinal (``year*4 + quarter``),
    not by first-appearance order.  A late-disclosed revision or a back-filled
    older period (e.g. a restated 2025Q2 arriving after 2025Q3) must not reorder
    the sequence that lag / TTM / growth operators walk — that reordering corrupts
    every lag, trend and growth factor (audit §4.1).
    """
    target = period_ordinal(key)
    if target is None:
        # Unparseable period ids keep first-seen order at the end.
        order.append(key)
        return
    for position, existing in enumerate(order):
        existing_ord = period_ordinal(existing)
        if existing_ord is not None and existing_ord > target:
            order.insert(position, key)
            return
    order.append(key)


def _walk_periods(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    fn: Callable[[list[object], OrderedDict, object], float],
) -> pd.DataFrame:
    period_id = period_id.reindex(index=x.index, columns=x.columns)
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        order: list[object] = []
        visible: OrderedDict[object, float] = OrderedDict()
        xv = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        pv = period_id[col].to_numpy()
        arr = np.full(len(x), np.nan, dtype=float)
        for i, (value, raw_period) in enumerate(zip(xv, pv)):
            key = _period_key(raw_period)
            if key is not None and np.isfinite(value):
                if key not in visible:
                    _period_insert(order, key)
                visible[key] = float(value)
            if key is None or key not in visible:
                continue
            try:
                arr[i] = fn(order, visible, key)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                arr[i] = np.nan
        out[col] = arr
    return out


def _values(
    order: list[object],
    visible: OrderedDict,
    current: object,
    count: int | None = None,
    require_consecutive: bool = False,
):
    """Values of the most recent ``count`` visible report periods ending at
    ``current`` (or the full visible history when ``count`` is None).

    ``require_consecutive=True`` fails closed on a skipped fiscal period: the
    selected window's ordinals must be contiguous (each adjacent pair differs by
    exactly one fiscal period) and ordinal-parseable.  Otherwise a missing
    report (e.g. a vendor gap at 2025Q2) would silently substitute a
    non-adjacent quarter — acceptable for "historical distribution" summaries
    but wrong for sums/averages whose math requires adjacent periods (review
    P0-03).
    """
    try:
        pos = order.index(current)
    except ValueError:
        return []
    keys = order[: pos + 1]
    if count is not None:
        keys = keys[-int(count):]
    if require_consecutive and len(keys) > 1:
        ords = [period_ordinal(k) for k in keys]
        if any(o is None for o in ords):
            return []
        for prev_o, o in zip(ords, ords[1:]):
            if o != prev_o + 1:
                return []
    vals = [float(visible[k]) for k in keys if k in visible and np.isfinite(visible[k])]
    return vals


def _lag_value(order, visible, current, periods: int):
    """Exact ordinal lag: the value whose fiscal ordinal is ``current - periods``.

    Position-based lookup breaks when a quarter is missing (a skipped report period
    must yield NaN, not the value of a non-adjacent quarter) — audit §4.2.
    """
    periods = _pos_int(periods, "periods")
    target = period_ordinal(current)
    if target is None:
        return np.nan
    target -= periods
    for key in reversed(order):
        if period_ordinal(key) == target:
            return float(visible.get(key, np.nan))
    return np.nan


def _register(name: str, params: Iterable[str], fn, description: str, *, tags=()):
    metadata = OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["fundamental", "period_aware", "pit_safe", "causal", "production_extension", *tags],
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"FundamentalV2_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fundamental_period",
        business_category="fundamental",
        canonical=name,
        source="fundamental_transforms_v2",
        backend="pandas_numpy",
        status="production",
    )(cls)


def fin_lag(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: _lag_value(o, v, c, p))


def fin_diff(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: float(v[c]) - _lag_value(o, v, c, p))


def fin_pct_change(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    return _walk_periods(x, period_id, lambda o, v, c: _safe_div(float(v[c]), _lag_value(o, v, c, p)) - 1.0)


def fin_log_change(x, period_id, periods=1):
    p = _pos_int(periods, "periods")
    def calc(o, v, c):
        old = _lag_value(o, v, c, p); cur = float(v[c])
        return float(np.log(cur / old)) if cur > 0 and old > 0 else np.nan
    return _walk_periods(x, period_id, calc)


def fin_qoq(x, period_id):
    return fin_pct_change(x, period_id, 1)


def fin_yoy(x, period_id, periods_per_year=4):
    return fin_pct_change(x, period_id, _pos_int(periods_per_year, "periods_per_year"))


def fin_ttm(x, period_id, periods_per_year=4):
    n = _pos_int(periods_per_year, "periods_per_year")
    def calc(o, v, c):
        # TTM is a sum of *adjacent* fiscal periods: a skipped report must not
        # pull in a non-adjacent quarter (review P0-03).
        vals = _values(o, v, c, n, require_consecutive=True)
        return float(np.sum(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_average_balance(x, period_id, periods=2):
    n = _pos_int(periods, "periods")
    def calc(o, v, c):
        # Average balance over a contiguous window of fiscal periods (review
        # P0-03): a skipped report period fails closed instead of substituting
        # a non-adjacent report.
        vals = _values(o, v, c, n, require_consecutive=True)
        return float(np.mean(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_growth(x, period_id, periods=1):
    return fin_pct_change(x, period_id, periods)


def fin_cagr(x, period_id, periods=4, periods_per_year=4):
    p = _pos_int(periods, "periods")
    ppy = _pos_int(periods_per_year, "periods_per_year")
    def calc(o, v, c):
        old = _lag_value(o, v, c, p); cur = float(v[c])
        if cur <= 0 or old <= 0:
            return np.nan
        return float((cur / old) ** (ppy / p) - 1.0)
    return _walk_periods(x, period_id, calc)


def fin_growth_acceleration(x, period_id, short_periods=1, long_periods=4):
    s = _pos_int(short_periods, "short_periods")
    l = _pos_int(long_periods, "long_periods")
    if s >= l:
        raise ValueError("short_periods must be < long_periods")
    short = fin_pct_change(x, period_id, s)
    long = fin_pct_change(x, period_id, l)
    return short - long


def fin_growth_change(x, period_id, growth_periods=4, compare_periods=1):
    g = fin_pct_change(x, period_id, _pos_int(growth_periods, "growth_periods"))
    return g - fin_lag(g, period_id, _pos_int(compare_periods, "compare_periods"))


def _rolling_period_stat(x, period_id, periods: int, reducer):
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        return float(reducer(vals)) if len(vals) == n else np.nan
    return _walk_periods(x, period_id, calc)


def fin_std(x, period_id, periods=8):
    return _rolling_period_stat(x, period_id, periods, lambda a: np.std(a, ddof=1))


def fin_mad(x, period_id, periods=8):
    return _rolling_period_stat(x, period_id, periods, lambda a: np.mean(np.abs(a - np.mean(a))))


def fin_cv(x, period_id, periods=8):
    def reducer(a):
        mean = float(np.mean(a)); sd = float(np.std(a, ddof=1))
        return sd / abs(mean) if abs(mean) > _EPS else np.nan
    return _rolling_period_stat(x, period_id, periods, reducer)


def fin_stability(x, period_id, periods=8):
    cv = fin_cv(x, period_id, periods)
    return 1.0 / (1.0 + cv.abs())


def fin_range(x, period_id, periods=8):
    return _rolling_period_stat(x, period_id, periods, lambda a: np.max(a) - np.min(a))


def fin_zscore_history(x, period_id, periods=8):
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        sd = float(np.std(vals, ddof=1))
        # Zero variance -> the z-score is undefined; 0 would masquerade as
        # "exactly at the historical mean".  Return NaN and let a dedicated
        # constant-flag operator surface the degenerate case.
        return (float(vals[-1]) - float(np.mean(vals))) / sd if sd > _EPS else np.nan
    return _walk_periods(x, period_id, calc)


def fin_percentile_history(x, period_id, periods=8):
    n = _pos_int(periods, "periods", 2)
    def calc(o, v, c):
        vals = np.asarray(_values(o, v, c, n), dtype=float)
        if len(vals) != n:
            return np.nan
        return float((np.sum(vals < vals[-1]) + 0.5 * np.sum(vals == vals[-1])) / len(vals))
    return _walk_periods(x, period_id, calc)


def _trend_stat(x, period_id, periods, which: str):
    n = _pos_int(periods, "periods", 3)
    def calc(o, v, c):
        y = np.asarray(_values(o, v, c, n), dtype=float)
        if len(y) != n:
            return np.nan
        xx = np.arange(n, dtype=float)
        xb, yb = float(xx.mean()), float(y.mean())
        den = float(np.sum((xx-xb)**2))
        if den <= _EPS:
            return np.nan
        slope = float(np.sum((xx-xb)*(y-yb))/den)
        if which == "slope":
            return slope
        fitted = yb + slope*(xx-xb)
        resid = y-fitted
        ss_res = float(np.sum(resid**2)); ss_tot=float(np.sum((y-yb)**2))
        if which == "r2":
            # Constant series has no identifiable trend; a perfect "fit" of 1.0
            # would be meaningless, so return NaN instead.
            return 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan
        if n <= 2:
            return np.nan
        mse = ss_res/(n-2)
        se = float(np.sqrt(mse/den)) if mse >= 0 else np.nan
        return slope/se if np.isfinite(se) and se > _EPS else np.nan
    return _walk_periods(x, period_id, calc)


def fin_trend_slope(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "slope")
def fin_trend_r2(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "r2")
def fin_trend_tstat(x, period_id, periods=8): return _trend_stat(x, period_id, periods, "tstat")


def fin_trend_acceleration(x, period_id, short_periods=4, long_periods=8):
    s=_pos_int(short_periods,"short_periods",3); l=_pos_int(long_periods,"long_periods",3)
    if s >= l: raise ValueError("short_periods must be < long_periods")
    return fin_trend_slope(x,period_id,s)-fin_trend_slope(x,period_id,l)


def fin_monotonicity(x, period_id, periods=8):
    n=_pos_int(periods,"periods",2)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n),dtype=float)
        if len(vals)!=n:return np.nan
        d=np.diff(vals)
        return float((np.sum(d>0)-np.sum(d<0))/max(1,len(d)))
    return _walk_periods(x,period_id,calc)


def _streak(x, period_id, positive: bool):
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,None),dtype=float)
        if len(vals)<2:return 0.0
        d=np.diff(vals); count=0
        for z in d[::-1]:
            if (z>0) if positive else (z<0): count+=1
            else: break
        return float(count)
    return _walk_periods(x,period_id,calc)


def fin_positive_streak(x,period_id): return _streak(x,period_id,True)
def fin_negative_streak(x,period_id): return _streak(x,period_id,False)


def fin_sign_change_count(x,period_id,periods=8):
    n=_pos_int(periods,"periods",3)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n),dtype=float)
        if len(vals)!=n:return np.nan
        signs=np.sign(np.diff(vals)); signs=signs[signs!=0]
        return float(np.sum(signs[1:]!=signs[:-1])) if len(signs)>1 else 0.0
    return _walk_periods(x,period_id,calc)


def fin_growth_volatility(x,period_id,growth_periods=1,window_periods=8):
    g=fin_pct_change(x,period_id,_pos_int(growth_periods,"growth_periods"))
    return fin_std(g,period_id,_pos_int(window_periods,"window_periods",2))


def fin_growth_stability(x,period_id,growth_periods=1,window_periods=8):
    vol=fin_growth_volatility(x,period_id,growth_periods,window_periods)
    return 1.0/(1.0+vol.abs())


def fin_growth_persistence(x,period_id,growth_periods=1,window_periods=8):
    g=fin_pct_change(x,period_id,_pos_int(growth_periods,"growth_periods"))
    n=_pos_int(window_periods,"window_periods",2)
    def calc(o,v,c):
        vals=np.asarray(_values(o,v,c,n),dtype=float)
        return float(np.mean(vals>0)) if len(vals)==n else np.nan
    return _walk_periods(g,period_id,calc)


def fin_ratio(numerator,denominator):
    den=denominator.replace(0,np.nan)
    return (numerator/den).replace([np.inf,-np.inf],np.nan)


def fin_common_size(x,base): return fin_ratio(x,base)


def fin_turnover(flow,balance,period_id,average_periods=2):
    avg=fin_average_balance(balance,period_id,_pos_int(average_periods,"average_periods"))
    return fin_ratio(flow,avg)


def fin_divergence(x,y,period_id,periods=4):
    p=_pos_int(periods,"periods")
    return fin_pct_change(x,period_id,p)-fin_pct_change(y,period_id,p)


def fin_cash_earnings_gap(earnings,cashflow,scale):
    return fin_ratio(earnings-cashflow,scale.abs())


def fin_accrual_ratio(earnings,cashflow,assets):
    return fin_ratio(earnings-cashflow,assets.abs())


def fin_cash_conversion(cashflow,earnings): return fin_ratio(cashflow,earnings)


def fin_working_capital_change(working_capital,period_id,periods=1):
    return fin_diff(working_capital,period_id,_pos_int(periods,"periods"))


_SPECS = [
    ("fin_lag",["x","period_id","periods"],fin_lag,"Lag by visible reporting periods, never by trading days."),
    ("fin_diff",["x","period_id","periods"],fin_diff,"Difference versus a prior visible report period."),
    ("fin_pct_change",["x","period_id","periods"],fin_pct_change,"Percent change versus a prior visible report period."),
    ("fin_log_change",["x","period_id","periods"],fin_log_change,"Log change versus a prior visible report period."),
    ("fin_qoq",["x","period_id"],fin_qoq,"Quarter-over-quarter/one-report-period growth."),
    ("fin_yoy",["x","period_id","periods_per_year"],fin_yoy,"Year-over-year growth with configurable periods per year."),
    ("fin_ttm",["x","period_id","periods_per_year"],fin_ttm,"Rolling sum over visible report periods."),
    ("fin_average_balance",["x","period_id","periods"],fin_average_balance,"Average balance over visible report periods."),
    ("fin_growth",["x","period_id","periods"],fin_growth,"Generic reporting-period growth."),
    ("fin_cagr",["x","period_id","periods","periods_per_year"],fin_cagr,"Reporting-period CAGR with configurable annualization."),
    ("fin_growth_acceleration",["x","period_id","short_periods","long_periods"],fin_growth_acceleration,"Short-horizon minus long-horizon fundamental growth."),
    ("fin_growth_change",["x","period_id","growth_periods","compare_periods"],fin_growth_change,"Change in a reporting-period growth rate."),
    ("fin_growth_volatility",["x","period_id","growth_periods","window_periods"],fin_growth_volatility,"Volatility of period growth across visible reports."),
    ("fin_growth_stability",["x","period_id","growth_periods","window_periods"],fin_growth_stability,"Inverse growth-volatility stability score."),
    ("fin_growth_persistence",["x","period_id","growth_periods","window_periods"],fin_growth_persistence,"Fraction of recent visible reports with positive growth."),
    ("fin_std",["x","period_id","periods"],fin_std,"Historical reporting-period standard deviation."),
    ("fin_mad",["x","period_id","periods"],fin_mad,"Historical reporting-period mean absolute deviation."),
    ("fin_cv",["x","period_id","periods"],fin_cv,"Historical reporting-period coefficient of variation."),
    ("fin_stability",["x","period_id","periods"],fin_stability,"Generic bounded fundamental stability score."),
    ("fin_range",["x","period_id","periods"],fin_range,"Historical reporting-period range."),
    ("fin_zscore_history",["x","period_id","periods"],fin_zscore_history,"Current value z-score versus own visible reporting history."),
    ("fin_percentile_history",["x","period_id","periods"],fin_percentile_history,"Current value percentile versus own visible reporting history."),
    ("fin_trend_slope",["x","period_id","periods"],fin_trend_slope,"OLS slope across recent visible report periods."),
    ("fin_trend_r2",["x","period_id","periods"],fin_trend_r2,"OLS trend R-squared across recent visible report periods."),
    ("fin_trend_tstat",["x","period_id","periods"],fin_trend_tstat,"OLS trend slope t-statistic across recent visible report periods."),
    ("fin_trend_acceleration",["x","period_id","short_periods","long_periods"],fin_trend_acceleration,"Difference between short and long reporting-period slopes."),
    ("fin_monotonicity",["x","period_id","periods"],fin_monotonicity,"Signed monotonicity score of recent reporting-period changes."),
    ("fin_positive_streak",["x","period_id"],fin_positive_streak,"Current streak of positive report-to-report changes."),
    ("fin_negative_streak",["x","period_id"],fin_negative_streak,"Current streak of negative report-to-report changes."),
    ("fin_sign_change_count",["x","period_id","periods"],fin_sign_change_count,"Number of growth-direction reversals in recent reports."),
    ("fin_ratio",["numerator","denominator"],fin_ratio,"Generic finite ratio; domain ratios should be recipes over this primitive."),
    ("fin_common_size",["x","base"],fin_common_size,"Common-size accounting transform x/base."),
    ("fin_turnover",["flow","balance","period_id","average_periods"],fin_turnover,"Flow divided by average reporting-period balance."),
    ("fin_divergence",["x","y","period_id","periods"],fin_divergence,"Difference between two reporting-period growth rates."),
    ("fin_cash_earnings_gap",["earnings","cashflow","scale"],fin_cash_earnings_gap,"Scaled earnings-minus-cash-flow gap."),
    ("fin_accrual_ratio",["earnings","cashflow","assets"],fin_accrual_ratio,"Accrual proxy (earnings-cash flow)/assets."),
    ("fin_cash_conversion",["cashflow","earnings"],fin_cash_conversion,"Cash flow divided by earnings."),
    ("fin_working_capital_change",["working_capital","period_id","periods"],fin_working_capital_change,"Reporting-period change in working capital."),
]

for _name,_params,_fn,_desc in _SPECS:
    _register(_name,_params,_fn,_desc)
