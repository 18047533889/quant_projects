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

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, ScalarOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    """R16-058: STRICT integer gate — never ``int(value)`` truncation.

    ``5.9`` / ``True`` / ``NaN`` must fail BEFORE reaching the kernel, exactly
    as the pandas binder rejects them.  The polars backend only receives
    binder-validated parameters.
    """
    from factor_engine.cleaned_operators.base import strict_int_param

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    result = strict_int_param(value, name)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _cols(*frames: pl.DataFrame) -> list[str]:
    """R16-059: EXACT axis contract — no silent intersection.

    A column mismatch across multi-input frames silently shrank the universe
    (a missing stock just disappeared).  Without an explicit BroadcastSpec the
    exact axis/order is REQUIRED: a missing / extra / reordered column in any
    frame is a hard error.
    """
    if not frames:
        return []
    base_cols = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        frame_cols = [c for c in frame.columns if c not in _SKIP]
        if frame_cols != base_cols:
            raise ValueError(
                "polars_state_event: multi-input frames must share the EXACT "
                f"column axis (R16-059).  base={base_cols[:5]}... got={frame_cols[:5]}... "
                "— a missing/extra/reordered column would silently shrink the "
                "universe; broadcast requires an explicit BroadcastSpec"
            )
    return base_cols


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


_LEGAL_PRICE_BASIS = frozenset({"raw", "adjusted", "pre_close", "factor_adjusted", "none", None})


def _validate_price_basis(**kwargs) -> None:
    """R16-060: backend-independent price_basis domain gate.

    The pandas path enforces the price_basis domain in its metadata; the native
    Polars path must enforce the SAME gate so an illegal basis cannot bypass the
    typed-IR contract on this backend.
    """
    basis = kwargs.get("price_basis")
    if isinstance(basis, str):
        if basis.strip().lower() not in _LEGAL_PRICE_BASIS:
            raise ValueError(
                f"price_basis={basis!r} is not a legal basis {sorted(_LEGAL_PRICE_BASIS - {None})} "
                "(R16-060 — domain gate is backend-independent)"
            )


def _ratio_op(a, b, name):
    cols = _cols(a, b)
    rows = a.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _safe_ratio_1d(a[c].to_numpy(), b[c].to_numpy()) - 1.0
    return _make(a, cols, out)


def open_close_return(open_px, close, **kwargs):
    _validate_price_basis(**kwargs)
    return _ratio_op(close, open_px, "open_close_return")


def open_to_vwap_return(open_px, vwap, **kwargs):
    _validate_price_basis(**kwargs)
    return _ratio_op(vwap, open_px, "open_to_vwap_return")


def overnight_return(open_px, pre_close, **kwargs):
    _validate_price_basis(**kwargs)
    return _ratio_op(open_px, pre_close, "overnight_return")


def vwap_to_close_return(vwap, close, **kwargs):
    _validate_price_basis(**kwargs)
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
    """R16-057: STRICT ConditionBool/EventBool truth — legal set {0, 1, NaN}.

    The old ``np.isfinite(cv) & (cv != 0)`` read ``-1`` / ``0.2`` / ``2`` as
    True — inconsistent with the pandas strict tri-state validator.  Any finite
    value outside {0, 1} is a DATA ERROR shared by every backend, so it raises
    here instead of silently becoming a truthy.
    """
    missing = cv != cv
    valid = (cv == 0.0) | (cv == 1.0)
    bad = ~missing & ~valid
    if np.any(bad):
        bad_values = sorted({str(v) for v in np.unique(cv[bad]).tolist()})[:10]
        raise ValueError(
            f"ConditionBool/EventBool input must be {{0, 1, NaN}}; found values "
            f"outside the legal set: {bad_values} (R16-057 — ±Inf / -1 / 0.2 must "
            "never be read as a truthy)"
        )
    return cv == 1.0


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


def ts_time_since_change(condition, max_lookback=None, missing_policy="break", initial_semantics="since_transition"):
    limit = None if max_lookback is None else _pi(max_lookback, "max_lookback", minimum=1)
    policy = str(missing_policy).lower()
    if policy not in ("break", "carry"):
        raise ValueError("missing_policy must be 'break' or 'carry'")
    sem = str(initial_semantics).lower()
    if sem not in ("since_transition", "state_age"):
        raise ValueError("initial_semantics must be 'since_transition' or 'state_age'")
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        last_change = -1
        prev = None
        started = False
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
                started = False
                continue
            current = bool(truth[row])
            if prev is not None and current != prev:
                last_change = row
            prev = current
            if not started:
                started = True
                # #146: ``state_age`` starts the clock at the first VALID state;
                # ``since_transition`` waits for the first transition.
                if sem == "state_age" and last_change < 0:
                    last_change = row
            if last_change >= 0:
                distance = row - last_change
                if limit is None or distance < limit:
                    out[row, i] = float(distance)
    return _make(condition, cols, out)


def _event_positions(truth: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero(truth[start:end])


def _gaps_censored(valid_col: np.ndarray, positions: np.ndarray, start: int) -> np.ndarray | None:
    """#145 mirror: inter-event gaps, censored (None) when an interval crosses
    an UNKNOWN (NaN) row."""
    if positions.size < 2:
        return None
    gaps = np.diff(positions).astype(float)
    for idx in range(positions.size - 1):
        a = start + int(positions[idx])
        b = start + int(positions[idx + 1])
        if np.any(~valid_col[a + 1 : b]):
            return None  # interval crosses an unknown row -> censored
    return gaps


def ts_event_spacing_mean(condition, window=60, min_events=2):
    w = _pi(window, "window")
    # R16-058: strict int gate — never ``max(2, int(min_events))`` silent clamp.
    min_e = _pi(min_events, "min_events", minimum=2)
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        valid = np.isfinite(cv)
        for row in range(rows):
            start = max(0, row - w + 1)
            positions = _event_positions(truth, start, row + 1)
            if positions.size < min_e:
                continue
            gaps = _gaps_censored(valid, positions, start)
            if gaps is None:
                continue  # censored: an interval crossed an unknown row
            out[row, i] = float(np.mean(gaps)) if gaps.size else np.nan
    return _make(condition, cols, out)


def ts_event_spacing_cv(condition, window=60, min_events=3):
    w = _pi(window, "window")
    # R16-058: strict int gate — never ``max(3, int(min_events))`` silent clamp.
    min_e = _pi(min_events, "min_events", minimum=3)
    cols = _cols(condition)
    rows = condition.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        cv = condition[c].to_numpy()
        truth = _truth(cv)
        valid = np.isfinite(cv)
        for row in range(rows):
            start = max(0, row - w + 1)
            positions = _event_positions(truth, start, row + 1)
            if positions.size < min_e:
                continue
            gaps = _gaps_censored(valid, positions, start)
            if gaps is None:
                continue  # censored: an interval crossed an unknown row
            if gaps.size and float(np.mean(gaps)) > 0:
                out[row, i] = float(np.std(gaps) / np.mean(gaps))
    return _make(condition, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    # R11 P0-60: ``price_basis`` is a declared trailing scalar on the pandas
    # reference (the price-basis runtime gate).  The native polars impls do not
    # run the basis gate but must declare the same arity so positional calls
    # against the reference stay valid (R4-100); the kwarg is ignored.
    ("open_close_return", ("open", "close", "price_basis"), open_close_return, "close / open - 1."),
    ("open_to_vwap_return", ("open", "vwap", "price_basis"), open_to_vwap_return, "vwap / open - 1."),
    ("overnight_return", ("open", "pre_close", "price_basis"), overnight_return, "open / pre_close - 1."),
    ("vwap_to_close_return", ("vwap", "close", "price_basis"), vwap_to_close_return, "close / vwap - 1."),
    ("ts_max_buildup", ("x", "d"), ts_max_buildup, "Rolling count of running-max updates."),
    ("ts_transition_count", ("condition", "window", "missing_policy"), ts_transition_count, "Count of state transitions in a window."),
    ("ts_time_since_change", ("condition", "max_lookback", "missing_policy", "initial_semantics"), ts_time_since_change, "Rows since the latest state change (since_transition or state_age)."),
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
