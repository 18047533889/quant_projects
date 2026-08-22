# -*- coding: utf-8 -*-
"""Activity-clock primitives (2026-08-08 Gemini round).

A generic activity clock rescales any non-negative daily ``activity``
(volume / turnover / abs return / amount) by its trailing median and counts
how many trading days are needed to consume a fixed ``budget``.  This one
kernel serves both volume-time and volatility-time factor construction:

* ``ts_activity_clock_lagged_value`` — ``x`` at the activity-clock lag ``k*``
  (the row at which the accumulated scaled activity first reaches ``budget``),
  CURRENT-INCLUSIVE (``include_current=True`` default).  ``x -
  ts_activity_clock_lagged_value(x, activity, ...)`` is an activity-clock
  momentum recipe.
* ``ts_activity_clock_lagged_value_prior`` — same lagged value with the current
  bar ALWAYS excluded (P0-O #79): a single high-activity bar can never consume
  the whole budget at ``k*=0`` and manufacture a zero momentum on the shock
  day.  This is the canonical to build activity-clock momentum from.
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

from cleaned_operators.base import ParamSpec
from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.closure import SameAxisError
from cleaned_operators.registry import OperatorRegistry

_EPS = 1e-12


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """SameAxis align of multi-panel inputs.

    P0-O (78): a silent ``frame.reindex(...)`` would bridge a date-shifted
    panel (``x_t`` vs ``activity_{t+1}``) or a differently-ordered instrument
    axis back onto the base — manufacturing a momentum/age series that the data
    never actually contained.  Fail closed instead: the panels must share the
    EXACT same row index and instrument columns, else ``SameAxisError``.
    """
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for position, frame in enumerate(frames[1:], start=1):
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise SameAxisError(
                f"activity-clock input {position} is not aligned with input 0: "
                "row index or instrument columns differ (no silent reindex; "
                "date-shifted / different-stock panels are rejected — P0-O #78)"
            )
        out.append(frame)
    return tuple(out)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _activity_clock_kernel(
    activity: np.ndarray,
    scale_window: int,
    max_lookback: int,
    budget: float,
    include_current: bool = True,
) -> np.ndarray:
    """Return per-row k* (rows back to consume ``budget`` units of scaled act).

    Each row's activity is rescaled by the trailing median of *strictly prior*
    rows (median of ``[s-scale_window, s-1]``).  Budget is applied on the
    normalized scale (NOT by rescaling the raw activity, which would cancel
    under the median normalization).  k* is the smallest back-distance whose
    cumulative scaled activity reaches ``budget``; NaN when the budget is not
    consumed within ``max_lookback`` rows.

    Missing-value policy (P0-006): an UNKNOWN activity value inside the budget
    path does not consume market time — it fail-closes the row (``NaN``), it is
    never skipped.  Real ``activity == 0`` consumes nothing but is not blocking.
    Negative activity is an *invalid* state (not a zero) and invalidates any
    window that contains it.  The scale estimate requires at least
    ``min_scale_periods = max(5, scale_window//2)`` finite prior observations.
    """
    rows = activity.shape[0]
    min_scale = max(5, int(scale_window) // 2)
    scaled = np.full(rows, np.nan, dtype=float)
    for s in range(rows):
        if s < 1:
            continue
        lo = max(0, s - scale_window)
        past = activity[lo:s]
        finite_past = past[np.isfinite(past)]
        # negative activity invalidates the scale window (not silently dropped)
        if np.any(finite_past < 0.0) or finite_past.size < min_scale:
            continue
        scale = float(np.median(finite_past))
        if not np.isfinite(scale) or scale <= _EPS:
            continue
        a = activity[s]
        if not np.isfinite(a) or a < 0.0:
            continue
        scaled[s] = a / scale
    k = np.full(rows, np.nan, dtype=float)
    start = 0 if include_current else 1
    for r in range(rows):
        cum = 0.0
        for back in range(start, max_lookback + 1):
            s = r - back
            if s < 0:
                break
            a = scaled[s]
            # Fail-closed: any unknown activity inside [r-back, r] invalidates.
            if np.isnan(a):
                break
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
    include_current: bool = True,
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
        k[:, c] = _activity_clock_kernel(av[:, c], sw, ml, budget_f, bool(include_current))
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


def _ts_activity_clock_lagged_value_prior(
    x: pd.DataFrame,
    activity: pd.DataFrame,
    budget: float = 1.0,
    scale_window: int = 20,
    max_lookback: int = 60,
) -> pd.DataFrame:
    """``x`` at the activity-clock lag with the CURRENT bar always excluded.

    P0-O #79: the current-inclusive variant returns ``k* = 0`` on a
    high-activity shock day (a single bar consumes the whole budget), so
    ``x - lagged_value = x_t - x_t = 0`` — momentum reads zero on the very day
    it should spike.  This canonical starts the budget path one bar back
    (``include_current=False`` fixed), so the lagged value is always a STRICTLY
    PRIOR observable ``x`` and an activity-clock momentum built on it is never
    identically zero on the shock bar.
    """
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
        k[:, c] = _activity_clock_kernel(av[:, c], sw, ml, budget_f, include_current=False)
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
    include_current: bool = True,
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
        out[:, c] = _activity_clock_kernel(av[:, c], sw, ml, budget_f, bool(include_current))
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
            # Typed contract (P0-007): ``x`` is a positive level (price /
            # wealth index / positive fundamental level).  ``seg/running_max-1``
            # is only a drawdown on a positive scale; a non-positive or
            # negative window is out of contract -> fail closed.
            if np.any(seg <= 0.0):
                continue
            # activity is non-negative by contract; a negative value is an
            # invalid state (never "cheap" activity), fail closed.
            if np.any(act < 0.0):
                continue
            total_act = float(act.sum())
            if total_act <= _EPS:
                continue
            running_max = np.maximum.accumulate(seg)
            dd = seg / running_max - 1.0
            trough = int(np.argmin(dd))
            if dd[trough] >= -1e-12:
                # No drawdown within the window is a VALID state: cost is 0,
                # not NaN (the row is fully observed and flat / monotone up).
                out[r, c] = 0.0
                continue
            # R4-71: with a flat (equal) peak plateau ``np.argmax`` returns the
            # FIRST maximum, which can be far earlier than the actual peak the
            # drawdown departed from.  Take the MOST RECENT maximum instead so a
            # sideways-then-drawdown episode is measured from the last peak.
            pre = seg[: trough + 1]
            peak = int(len(pre) - 1 - int(np.argmax(pre[::-1])))
            seg_act = float(act[peak : trough + 1].sum())
            out[r, c] = float(seg_act / total_act)
    return _frame_like(x, out)


# ---------------------------------------------------------------------------
# Registration (pandas + polars)
# ---------------------------------------------------------------------------
_DAILY_CANONICALS: tuple[str, ...] = (
    "ts_activity_clock_lagged_value",
    "ts_activity_clock_lagged_value_prior",
    "ts_activity_clock_age",
    "ts_max_drawdown_activity_cost",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_activity_clock_lagged_value": _ts_activity_clock_lagged_value,
    "ts_activity_clock_lagged_value_prior": _ts_activity_clock_lagged_value_prior,
    "ts_activity_clock_age": _ts_activity_clock_age,
    "ts_max_drawdown_activity_cost": _ts_max_drawdown_activity_cost,
}

_PARAMS: dict[str, list[str]] = {
    "ts_activity_clock_lagged_value": ["x", "activity", "budget", "scale_window", "max_lookback", "include_current"],
    "ts_activity_clock_lagged_value_prior": ["x", "activity", "budget", "scale_window", "max_lookback"],
    "ts_activity_clock_age": ["activity", "budget", "scale_window", "max_lookback", "include_current"],
    "ts_max_drawdown_activity_cost": ["x", "activity", "window"],
}

_CATEGORIES: dict[str, str] = {
    "ts_activity_clock_lagged_value": "time_series_event",
    "ts_activity_clock_lagged_value_prior": "time_series_event",
    "ts_activity_clock_age": "time_series_event",
    "ts_max_drawdown_activity_cost": "time_series_risk",
}

# Output unit per operator (R4-70).  ``same_as:target`` is only correct for the
# lagged-value operator (returns ``x`` at the activity-clock lag).  The age
# operator returns a back-distance in bars / activity-time units, and the
# max-drawdown cost is a dimensionless share of window activity (a ratio).
_UNITS: dict[str, str] = {
    "ts_activity_clock_lagged_value": "same_as:target",
    "ts_activity_clock_lagged_value_prior": "same_as:target",
    "ts_activity_clock_age": "bars",
    "ts_max_drawdown_activity_cost": "ratio",
}

# R4-69: ``scale_window < 5`` is a dead parameter region.  The kernel's internal
# scale estimate requires ``min_scale = max(5, scale_window//2)`` finite prior
# observations, so a scale_window of 2/3/4 can never satisfy its own sample
# floor and the operator degrades to a permanent NaN.  Expose that floor as a
# ParamSpec so ``validate_operator_call`` rejects it and the search grammar
# excludes the dead range.
# R26-118/119: the activity-clock history is NESTED, not max.  Each historical
# ``scaled_activity[s]`` needs ``scale_window`` PRIOR rows for its rolling
# median, and the k*-lookback then searches up to ``max_lookback`` more rows —
# so the raw input history the planner must prefetch is
# ``scale_window + max_lookback`` (plus the exact boundary).  Declared on
# ``scale_window`` so the compound formula drives prefetch / warmup / cache.
_PARAM_SPECS: dict[str, dict[str, ParamSpec]] = {
    "ts_activity_clock_lagged_value": {
        "scale_window": ParamSpec(
            dtype=int, min=5, history_formula="scale_window + max_lookback"
        ),
        "include_current": ParamSpec(dtype=bool, choices=(True, False)),
    },
    "ts_activity_clock_lagged_value_prior": {
        "scale_window": ParamSpec(
            dtype=int, min=5, history_formula="scale_window + max_lookback"
        ),
    },
    "ts_activity_clock_age": {
        "scale_window": ParamSpec(
            dtype=int, min=5, history_formula="scale_window + max_lookback"
        ),
        "include_current": ParamSpec(dtype=bool, choices=(True, False)),
    },
    "ts_max_drawdown_activity_cost": {},
}

# R4-98: explicit input unit contracts (enforced as documentation metadata and
# consumed by the unit-aware catalog consumers).
_INPUT_UNITS: dict[str, dict[str, str]] = {
    "ts_activity_clock_lagged_value": {
        "x": "target_value",
        "activity": "non_negative_activity",
    },
    "ts_activity_clock_lagged_value_prior": {
        "x": "target_value",
        "activity": "non_negative_activity",
    },
    "ts_activity_clock_age": {
        "activity": "non_negative_activity",
    },
    "ts_max_drawdown_activity_cost": {
        "x": "positive_level",
        "activity": "non_negative_activity",
    },
}

# R4-95: meaning of the window-like parameters.
#   - scale_window: trailing *finite-observation* window for the scale estimate
#     (gaps do not invalidate; a minimum number of finite observations is
#     required, and negative activity invalidates).
#   - max_drawdown window: trailing contiguous window (the whole segment and
#     activity series must be finite for the drawdown episode).
_WINDOW_SEMANTICS: dict[str, str] = {
    "ts_activity_clock_lagged_value": "finite_observations",
    "ts_activity_clock_lagged_value_prior": "finite_observations",
    "ts_activity_clock_age": "finite_observations",
    "ts_max_drawdown_activity_cost": "trailing_contiguous",
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata
    from cleaned_operators.base import validate_operator_call

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
                      "domain:activity_clock", f"unit:{_UNITS[canonical]}", "cost:4"],
                param_specs=_PARAM_SPECS[canonical],
                input_units=_INPUT_UNITS.get(canonical),
                window_semantics=_WINDOW_SEMANTICS.get(canonical),
            )

            _HANDLES_CALL_CONTRACT = True  # R5-02: routes through validate_operator_call

            def calculate(self, *args, _fn=fn, **kwargs):
                # R4-02: this module registered ``calculate`` directly, bypassing
                # the central integer / panel-axis / param validation.  Route it
                # through the registry-level logical-call validator.
                processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
                return _fn(*processed_args, **processed_kwargs)

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

    _surface.extend_extended_only(set(_DAILY_CANONICALS))

    # P0-O #79 back-compat spelling: the prior-only lagged-value canonical is
    # also reachable under the un-prefixed audit name ``activity_clock_lagged_value_prior``.
    try:
        OperatorRegistry.register_alias(
            "activity_clock_lagged_value_prior", "ts_activity_clock_lagged_value_prior"
        )
    except (KeyError, ValueError):
        pass


_register()
