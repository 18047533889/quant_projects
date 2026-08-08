# -*- coding: utf-8 -*-
"""Native Polars backends for return decomposition, state/event transitions,
elementwise complex helpers and the max-buildup kernel.

Per-column NumPy kernels over polars column arrays; results wrapped into
``pl.DataFrame`` without constructing pandas DataFrames.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, ScalarOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _safe_ratio_1d(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------
# return decomposition (elementwise)
# ---------------------------------------------------------------------------


def _ratio_op(a, b, name):
    cols = _cols(a, b)
    rows = a.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _safe_ratio_1d(a[c].to_numpy(), b[c].to_numpy()) - 1.0
    return _make(a, cols, out)


def open_close_return(open_px, close):
    return _ratio_op(close, open_px, "open_close_return")


def open_to_vwap_return(open_px, vwap):
    return _ratio_op(vwap, open_px, "open_to_vwap_return")


def overnight_return(open_px, pre_close):
    return _ratio_op(open_px, pre_close, "overnight_return")


def vwap_to_close_return(vwap, close):
    return _ratio_op(close, vwap, "vwap_to_close_return")


# ---------------------------------------------------------------------------
# elementwise helpers
# ---------------------------------------------------------------------------


def arg(x):
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        values = x[c].to_numpy()
        result = np.angle(values)
        result[~np.isfinite(result)] = np.nan
        out[:, i] = result
    return _make(x, cols, out)


def atan2(y, x):
    cols = _cols(y, x)
    rows = y.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = np.arctan2(y[c].to_numpy(), x[c].to_numpy())
    return _make(y, cols, out)


def constant_scalar(c=0.0, **kwargs):
    return float(c)


# ---------------------------------------------------------------------------
# ts_max_buildup
# ---------------------------------------------------------------------------


def ts_max_buildup(x, d):
    window = _pi(d, "d")
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        series = x[c].to_numpy()
        for end in range(rows):
            start = max(0, end - window + 1)
            segment = series[start : end + 1]
            current_max = -np.inf
            count = 0
            has_finite = False
            for value in segment:
                if not np.isfinite(value):
                    continue
                has_finite = True
                if value >= current_max:
                    current_max = value
                    count += 1
            if has_finite:
                out[end, i] = float(count)
    return _make(x, cols, out)


# ---------------------------------------------------------------------------
# state / event transitions
# ---------------------------------------------------------------------------


def _truth(cv: np.ndarray) -> np.ndarray:
    return np.isfinite(cv) & (cv != 0)


def ts_transition_count(condition, window=20, missing_policy="break"):
    w = _pi(window, "window")
    policy = str(missing_policy).lower()
    if policy not in ("break", "carry"):
        raise ValueError("missing_policy must be 'break' or 'carry'")
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        valid = np.isfinite(cv)
        for row in range(rows):
            start = max(0, row - w + 1)
            if not valid[row]:
                continue
            transitions = 0
            prev = None
            for t in range(start, row + 1):
                if not valid[t]:
                    if policy == "carry":
                        continue  # missing row transparent: carry last state
                    prev = None  # break: missing resets continuity
                    continue
                current = bool(truth[t])
                if prev is not None and current != prev:
                    transitions += 1
                prev = current
            out[row, i] = float(transitions)
    return _make(condition, cols, out)


def ts_time_since_change(condition, max_lookback=None, missing_policy="break"):
    limit = None if max_lookback is None else int(max_lookback)
    policy = str(missing_policy).lower()
    if policy not in ("break", "carry"):
        raise ValueError("missing_policy must be 'break' or 'carry'")
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        last_change = -1
        prev = None
        for row in range(rows):
            if not np.isfinite(cv[row]):
                if policy == "carry":
                    # missing row is a continuation: distance keeps growing
                    if last_change >= 0:
                        distance = row - last_change
                        if limit is None or distance < limit:
                            out[row, i] = float(distance)
                    continue
                # break: missing emits NaN and resets continuity
                last_change = -1
                prev = None
                continue
            current = bool(truth[row])
            if prev is not None and current != prev:
                last_change = row
            prev = current
            if last_change >= 0:
                distance = row - last_change
                if limit is None or distance < limit:
                    out[row, i] = float(distance)
    return _make(condition, cols, out)


def _event_positions(truth: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero(truth[start:end])


def ts_event_spacing_mean(condition, window=60, min_events=2):
    w = _pi(window, "window")
    min_e = max(2, int(min_events))
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        for row in range(rows):
            start = max(0, row - w + 1)
            positions = _event_positions(truth, start, row + 1)
            if positions.size < min_e:
                continue
            gaps = np.diff(positions)
            out[row, i] = float(np.mean(gaps)) if gaps.size else np.nan
    return _make(condition, cols, out)


def ts_event_spacing_cv(condition, window=60, min_events=3):
    w = _pi(window, "window")
    min_e = max(3, int(min_events))
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        for row in range(rows):
            start = max(0, row - w + 1)
            positions = _event_positions(truth, start, row + 1)
            if positions.size < min_e:
                continue
            gaps = np.diff(positions)
            if gaps.size and float(np.mean(gaps)) > 0:
                out[row, i] = float(np.std(gaps) / np.mean(gaps))
    return _make(condition, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("open_close_return", ("open", "close"), open_close_return, "close / open - 1."),
    ("open_to_vwap_return", ("open", "vwap"), open_to_vwap_return, "vwap / open - 1."),
    ("overnight_return", ("open", "pre_close"), overnight_return, "open / pre_close - 1."),
    ("vwap_to_close_return", ("vwap", "close"), vwap_to_close_return, "close / vwap - 1."),
    ("ts_max_buildup", ("x", "d"), ts_max_buildup, "Rolling count of running-max updates."),
    ("ts_transition_count", ("condition", "window"), ts_transition_count, "Count of state transitions in a window."),
    ("ts_time_since_change", ("condition", "max_lookback"), ts_time_since_change, "Rows since the latest state change."),
    ("ts_event_spacing_mean", ("condition", "window", "min_events"), ts_event_spacing_mean, "Mean event spacing in a window."),
    ("ts_event_spacing_cv", ("condition", "window", "min_events"), ts_event_spacing_cv, "CV of event spacing in a window."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="time_series_event",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsStateEvent_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="time_series_event",
        business_category="time_series_event",
        canonical=name,
        source="polars_state_event",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)


def _register_constant() -> None:
    metadata = OperatorMetadata(
        name="constant",
        category="math",
        description="Return a constant value.",
        param_names=["c"],
        return_type="scalar",
        tags=["math", "utility", "scalar", "polars", "native"],
    )

    def _calculate_scalar(self, c=0.0, **kwargs):
        return float(c)

    cls = type(
        "PolarsStateEvent_constant",
        (ScalarOperator,),
        {"metadata": metadata, "_calculate_scalar": _calculate_scalar, "__module__": __name__},
    )
    register_operator(
        name="constant",
        category="math",
        business_category="elementwise_math",
        canonical="constant",
        source="polars_state_event",
        backend="polars",
        status="production",
    )(cls)


_register_constant()
