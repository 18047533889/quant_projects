# -*- coding: utf-8 -*-
"""Activity-clock primitives (2026-08-08 Gemini round).

A generic activity clock rescales any non-negative daily ``activity``
(volume / turnover / abs return / amount) by its trailing median and counts
how many trading days are needed to consume a fixed ``budget``.  This one
kernel serves both volume-time and volatility-time factor construction:

* ``ts_activity_clock_lagged_value`` — ``x`` at the activity-clock lag ``k*``
  (the row at which the accumulated scaled activity first reaches ``budget``).
  ``x - ts_activity_clock_lagged_value(x, activity, ...)`` is an activity-clock
  momentum recipe.
* ``ts_activity_clock_age``           — the lag ``k*`` itself.
* ``ts_max_drawdown_activity_cost``   — share of window activity consumed by the
  max-drawdown peak→trough segment (turnover-clock drawdown cost).

All kernels are strict-PIT (scale uses strictly prior rows), deterministic and
NaN fail-closed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry

_EPS = 1e-12


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        out.append(frame)
    return tuple(out)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _activity_clock_kernel(
    activity: np.ndarray,
    scale_window: int,
    max_lookback: int,
    budget: float,
) -> np.ndarray:
    """Return per-row k* (rows back to consume ``budget`` units of scaled act).

    Each row's activity is rescaled by the trailing median of *strictly prior*
    rows (median of ``[s-scale_window, s-1]``).  Budget is applied on the
    normalized scale (NOT by rescaling the raw activity, which would cancel
    under the median normalization).  k* is the smallest back-distance whose
    cumulative scaled activity reaches ``budget``; NaN when the budget is not
    consumed within ``max_lookback`` rows.
    """
    rows = activity.shape[0]
    scaled = np.full(rows, np.nan, dtype=float)
    for s in range(rows):
        if s < 1:
            continue
        lo = max(0, s - scale_window)
        past = activity[lo:s]
        past = past[np.isfinite(past)]
        if past.size == 0:
            continue
        scale = float(np.median(past))
        if not np.isfinite(scale) or scale <= _EPS:
            continue
        a = activity[s]
        if np.isfinite(a) and a >= 0.0:
            scaled[s] = a / scale
    k = np.full(rows, np.nan, dtype=float)
    for r in range(rows):
        cum = 0.0
        for back in range(max_lookback + 1):
            s = r - back
            if s < 0:
                break
            a = scaled[s]
            if np.isnan(a):
                continue
            cum += a
            if cum >= budget:
                k[r] = float(back)
                break
    return k


def _ts_activity_clock_lagged_value(
    x: pd.DataFrame,
    activity: pd.DataFrame,
    budget: float = 1.0,
    scale_window: int = 20,
    max_lookback: int = 60,
) -> pd.DataFrame:
    x, activity = _align(x, activity)
    sw = max(2, int(scale_window))
    ml = max(1, int(max_lookback))
    budget_f = float(budget)
    if budget_f <= _EPS:
        raise ValueError("budget must be > 0")
    xv = x.to_numpy(dtype=float)
    av = activity.to_numpy(dtype=float)
    rows, cols = xv.shape
    k = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        k[:, c] = _activity_clock_kernel(av[:, c], sw, ml, budget_f)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            back = k[r, c]
            if np.isnan(back):
                continue
            s = int(r - back)
            if s < 0 or not np.isfinite(xv[s, c]):
                continue
            out[r, c] = float(xv[s, c])
    return _frame_like(x, out)


def _ts_activity_clock_age(
    activity: pd.DataFrame,
    budget: float = 1.0,
    scale_window: int = 20,
    max_lookback: int = 60,
) -> pd.DataFrame:
    activity = activity.copy()
    sw = max(2, int(scale_window))
    ml = max(1, int(max_lookback))
    budget_f = float(budget)
    if budget_f <= _EPS:
        raise ValueError("budget must be > 0")
    av = activity.to_numpy(dtype=float)
    rows, cols = av.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _activity_clock_kernel(av[:, c], sw, ml, budget_f)
    return _frame_like(activity, out)


def _ts_max_drawdown_activity_cost(
    x: pd.DataFrame,
    activity: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    x, activity = _align(x, activity)
    w = int(window)
    if w < 2:
        raise ValueError("ts_max_drawdown_activity_cost requires window >= 2")
    xv = x.to_numpy(dtype=float)
    av = activity.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            seg = xv[start : r + 1, c]
            act = av[start : r + 1, c]
            if not np.all(np.isfinite(seg)) or not np.all(np.isfinite(act)):
                continue
            total_act = float(act.sum())
            if total_act <= _EPS:
                continue
            running_max = np.maximum.accumulate(seg)
            dd = seg / running_max - 1.0
            trough = int(np.argmin(dd))
            if dd[trough] >= -1e-12:
                continue  # no drawdown within the window
            peak = int(np.argmax(seg[: trough + 1]))
            seg_act = float(act[peak : trough + 1].sum())
            out[r, c] = float(seg_act / total_act)
    return _frame_like(x, out)


# ---------------------------------------------------------------------------
# Registration (pandas + polars)
# ---------------------------------------------------------------------------
_DAILY_CANONICALS: tuple[str, ...] = (
    "ts_activity_clock_lagged_value",
    "ts_activity_clock_age",
    "ts_max_drawdown_activity_cost",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_activity_clock_lagged_value": _ts_activity_clock_lagged_value,
    "ts_activity_clock_age": _ts_activity_clock_age,
    "ts_max_drawdown_activity_cost": _ts_max_drawdown_activity_cost,
}

_PARAMS: dict[str, list[str]] = {
    "ts_activity_clock_lagged_value": ["x", "activity", "budget", "scale_window", "max_lookback"],
    "ts_activity_clock_age": ["activity", "budget", "scale_window", "max_lookback"],
    "ts_max_drawdown_activity_cost": ["x", "activity", "window"],
}

_CATEGORIES: dict[str, str] = {
    "ts_activity_clock_lagged_value": "time_series_event",
    "ts_activity_clock_age": "time_series_event",
    "ts_max_drawdown_activity_cost": "time_series_risk",
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]

        class _PandasOp(PandasOperator):
            metadata = PandasMetadata(
                name=canonical,
                category=category,
                description=canonical,
                examples=[],
                param_names=params,
                return_type="series",
                tags=["daily", "panel", "pit_safe", "causal", "deterministic",
                      f"signature:{','.join(params)}->series",
                      "domain:activity_clock", "unit:same_as:target", "cost:4"],
            )

            def calculate(self, *args, _fn=fn, **kwargs):
                return _fn(*args, **kwargs)

        OperatorRegistry.register(
            _PandasOp(), canonical=canonical, backend="pandas_numpy",
            source="activity_clock", backend_explicit=True,
        )

        class _PolarsOp(PolarsSeriesOperator):
            metadata = PolarsMetadata(name=canonical, category="activity_clock", param_names=[])

            def _calculate_series(self, *frames, _fn=fn, **params):
                import polars as pl  # noqa: F401
                pdfs = [f.select([c for c in f.columns if c not in _SKIP]).to_pandas() for f in frames]
                out = _fn(*pdfs, **params)
                base = frames[0]
                cols = [c for c in base.columns if c not in _SKIP]
                return base.with_columns(
                    [pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols]
                )

        OperatorRegistry.register(
            _PolarsOp(), canonical=canonical, backend="polars",
            source="activity_clock_polars", backend_explicit=True,
        )

    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_DAILY_CANONICALS)
    )


_register()
