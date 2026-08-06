# -*- coding: utf-8 -*-
"""Native Polars backends for strict fiscal signal primitives.

These operators are per-column fiscal-ordinal state machines (the pandas
references build a ``FiscalEventView`` that maps each report period to its
monotonic ordinal, applies revision policy and returns an ordered history).  The
same semantics are reproduced per column with an independent NumPy kernel over
polars column arrays.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_strict import period_ordinal

_SKIP = frozenset({"date", "stock_code"})
_EPS = 1e-12


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fiscal_map_1d(
    xv: np.ndarray,
    pv: list,
    fn: Callable[[list], float],
    require_consecutive: bool,
) -> np.ndarray:
    """Per-column fiscal state machine returning fn(history) at each row."""
    n = len(xv)
    order: list[int] = []
    visible: dict[int, float] = {}
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        ordinal = period_ordinal(pv[t]) if pv[t] is not None else None
        value = xv[t]
        if ordinal is not None and np.isfinite(value):
            if ordinal not in visible:
                order.append(ordinal)
                order.sort()
            visible[ordinal] = float(value)
        # history = ordered (ordinal, value) for visible ordinals at/before current
        current = period_ordinal(pv[t]) if pv[t] is not None else None
        if current is None:
            continue
        keys = [o for o in order if o <= current]
        if not keys:
            continue
        history = [(o, visible[o]) for o in keys if o in visible]
        if require_consecutive and history:
            # ordinals must form a consecutive run ending at current
            for i in range(1, len(history)):
                if history[i][0] != history[i - 1][0] + 1:
                    history = []
                    break
            if history and history[-1][0] != current:
                history = []
        if not history:
            continue
        out[t] = fn(history)
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def fiscal_true_streak(condition, period_id, require_consecutive=True, revision_policy="latest_available"):
    cols = _cols(condition, period_id)
    rows = condition.height
    out = np.zeros((rows, len(cols)), dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = condition[c].to_numpy()
        pv = _pv_list(period_id, c)

        def _calc(history):
            streak = 0
            for _, value in reversed(history):
                if _finite(value) and value != 0:
                    streak += 1
                else:
                    break
            return float(streak)

        arr = _fiscal_map_1d(xv, pv, _calc, rc)
        out[:, i] = np.where(np.isnan(arr), 0.0, arr)
    return _make(condition, cols, out)


def _pv_list(frame: pl.DataFrame, c: str) -> list:
    return frame[c].to_list()


def _signal_consistency(history, periods, min_periods):
    window = history[-periods:]
    usable = [np.sign(value) for _, value in window if _finite(value) and value != 0]
    if len(usable) < min_periods:
        return np.nan
    current = usable[-1]
    return float(sum(sign == current for sign in usable) / len(usable))


def fiscal_sign_consistency(signal, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available"):
    periods = _pi(periods, "periods")
    min_periods = _pi(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")
    cols = _cols(signal, period_id)
    rows = signal.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = signal[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_1d(xv, pv, lambda h: _signal_consistency(h, periods, min_periods), rc)
    return _make(signal, cols, out)


def _reversal_calc(history, periods, min_pairs):
    history = history[-periods:]
    if len(history) < min_pairs + 1:
        return np.nan
    numerator = denominator = 0.0
    pairs = 0
    for (_, prev_val), (_, cur_val) in zip(history, history[1:]):
        if not (_finite(prev_val) and _finite(cur_val)) or prev_val == 0 or cur_val == 0:
            continue
        pairs += 1
        denominator += abs(prev_val)
        if np.sign(prev_val) != np.sign(cur_val):
            numerator += min(abs(cur_val), abs(prev_val))
    return numerator / denominator if pairs >= min_pairs and denominator > _EPS else np.nan


def fiscal_reversal_ratio(x, period_id, periods=8, min_pairs=3, require_consecutive=True, revision_policy="latest_available"):
    periods = _pi(periods, "periods")
    min_pairs = _pi(min_pairs, "min_pairs")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = x[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_1d(xv, pv, lambda h: _reversal_calc(h, periods, min_pairs), rc)
    return _make(x, cols, out)


def _autocorr_calc(history, periods, lag, min_pairs):
    values = np.asarray([v for _, v in history[-periods:]], dtype=float)
    if len(values) <= lag:
        return np.nan
    a, b = values[lag:], values[:-lag]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < min_pairs or np.std(a[valid]) <= _EPS or np.std(b[valid]) <= _EPS:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def fiscal_autocorr(x, period_id, periods=12, lag=1, min_pairs=3, require_consecutive=True, revision_policy="latest_available"):
    periods = _pi(periods, "periods")
    lag = _pi(lag, "lag")
    min_pairs = _pi(min_pairs, "min_pairs")
    if periods < lag + min_pairs:
        raise ValueError("periods must be at least lag + min_pairs")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = x[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_1d(xv, pv, lambda h: _autocorr_calc(h, periods, lag, min_pairs), rc)
    return _make(x, cols, out)


def _surprise_calc(history, seasonal_lag, lookback, min_history):
    if len(history) <= seasonal_lag:
        return np.nan
    current_ord, current = history[-1]
    by_ord = dict(history)
    previous = by_ord.get(current_ord - seasonal_lag)
    if previous is None:
        return np.nan
    surprises = []
    for ordinal, value in history[:-1]:
        prior = by_ord.get(ordinal - seasonal_lag)
        if prior is not None and _finite(value) and _finite(prior):
            surprises.append(value - prior)
    surprises = surprises[-lookback:]
    if len(surprises) < min_history:
        return np.nan
    std = float(np.std(np.asarray(surprises), ddof=1)) if len(surprises) > 1 else np.nan
    if np.isfinite(std) and std > _EPS:
        return float((current - previous) / std)
    return np.nan


def fiscal_standardized_surprise(x, period_id, seasonal_lag=4, lookback_periods=8, min_history=4, require_consecutive=True, revision_policy="latest_available"):
    seasonal_lag = _pi(seasonal_lag, "seasonal_lag")
    lookback = _pi(lookback_periods, "lookback_periods")
    min_history = _pi(min_history, "min_history")
    cols = _cols(x, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = x[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_1d(xv, pv, lambda h: _surprise_calc(h, seasonal_lag, lookback, min_history), rc)
    return _make(x, cols, out)


def _fiscal_map_2d(
    xv: np.ndarray,
    yv: np.ndarray,
    pv: list,
    fn: Callable[[list, list], float],
    require_consecutive: bool,
) -> np.ndarray:
    n = len(xv)
    order: list[int] = []
    visible_x: dict[int, float] = {}
    visible_y: dict[int, float] = {}
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        ordinal = period_ordinal(pv[t]) if pv[t] is not None else None
        if ordinal is not None and np.isfinite(xv[t]):
            if ordinal not in visible_x:
                order.append(ordinal)
                order.sort()
            visible_x[ordinal] = float(xv[t])
        if ordinal is not None and np.isfinite(yv[t]):
            visible_y[ordinal] = float(yv[t])
        current = period_ordinal(pv[t]) if pv[t] is not None else None
        if current is None:
            continue
        keys = [o for o in order if o <= current]
        hx = [(o, visible_x[o]) for o in keys if o in visible_x]
        hy = [(o, visible_y[o]) for o in keys if o in visible_x and o in visible_y]
        if require_consecutive and hx:
            for i in range(1, len(hx)):
                if hx[i][0] != hx[i - 1][0] + 1:
                    hx = []
                    hy = []
                    break
            if hx and hx[-1][0] != current:
                hx = []
                hy = []
        if not hx:
            continue
        out[t] = fn(hx, hy)
    return out


def _sign_agreement_calc(hx, hy, periods, min_periods):
    dx = dict(hx)
    dy = dict(hy)
    pairs = [(dx[o], dy[o]) for o in sorted(set(dx) & set(dy))[-periods:]]
    pairs = [(a, b) for a, b in pairs if _finite(a) and _finite(b) and a != 0 and b != 0]
    if len(pairs) >= min_periods:
        return float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))
    return np.nan


def fiscal_sign_agreement(signal_x, signal_y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available"):
    periods = _pi(periods, "periods")
    min_periods = _pi(min_periods, "min_periods")
    cols = _cols(signal_x, signal_y, period_id)
    rows = signal_x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = signal_x[c].to_numpy()
        yv = signal_y[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_2d(xv, yv, pv, lambda hx, hy: _sign_agreement_calc(hx, hy, periods, min_periods), rc)
    return _make(signal_x, cols, out)


def _change_direction_calc(hx, hy, periods, min_periods):
    if len(hx) < 2:
        return np.nan
    hx = hx[-periods:]
    dx = dict(hx)
    dy = dict(hy)
    pairs = []
    for i, (ordinal, _) in enumerate(hx):
        if i == 0:
            continue
        prev = hx[i - 1][0]
        a = dx.get(ordinal, np.nan) - dx.get(prev, np.nan)
        b = dy.get(ordinal, np.nan) - dy.get(prev, np.nan)
        if _finite(a) and _finite(b) and a != 0 and b != 0:
            pairs.append((a, b))
    if len(pairs) >= min_periods:
        return float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))
    return np.nan


def fiscal_change_direction_agreement(x, y, period_id, periods=8, min_periods=3, require_consecutive=True, revision_policy="latest_available"):
    periods = _pi(periods, "periods")
    min_periods = _pi(min_periods, "min_periods")
    cols = _cols(x, y, period_id)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    rc = bool(require_consecutive)
    for i, c in enumerate(cols):
        xv = x[c].to_numpy()
        yv = y[c].to_numpy()
        pv = _pv_list(period_id, c)
        out[:, i] = _fiscal_map_2d(xv, yv, pv, lambda hx, hy: _change_direction_calc(hx, hy, periods, min_periods), rc)
    return _make(x, cols, out)


_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("fiscal_true_streak", ("condition", "period_id", "require_consecutive", "revision_policy"), fiscal_true_streak, "Consecutive non-zero fiscal signals."),
    ("fiscal_sign_consistency", ("signal", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"), fiscal_sign_consistency, "Share of recent same-sign fiscal signals."),
    ("fiscal_reversal_ratio", ("x", "period_id", "periods", "min_pairs", "require_consecutive", "revision_policy"), fiscal_reversal_ratio, "Reversal ratio of fiscal period changes."),
    ("fiscal_autocorr", ("x", "period_id", "periods", "lag", "min_pairs", "require_consecutive", "revision_policy"), fiscal_autocorr, "Fiscal-series autocorrelation."),
    ("fiscal_standardized_surprise", ("x", "period_id", "seasonal_lag", "lookback_periods", "min_history", "require_consecutive", "revision_policy"), fiscal_standardized_surprise, "Seasonally-lagged surprise z-score."),
    ("fiscal_sign_agreement", ("signal_x", "signal_y", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"), fiscal_sign_agreement, "Share of same-sign fiscal signals."),
    ("fiscal_change_direction_agreement", ("x", "y", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"), fiscal_change_direction_agreement, "Share of same-direction fiscal changes."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="fiscal_strict",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["fiscal", "period_aware", "pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsFiscalV2_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="fiscal_strict",
        business_category="fiscal",
        canonical=name,
        source="polars_fiscal_v2",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
