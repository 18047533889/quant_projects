# -*- coding: utf-8 -*-
"""R68 batch11: genuine-Polars backends for 45 pandas-delegate canonicals.

Same protocol as ``r68_native_batch4/5/6/9``: numpy authority kernels called on
``pl -> numpy`` column panels, plus numpy ports for the kernels whose authority
is a pandas-Series pipeline (EMA/Wilder recursive smoothing uses the exact
pandas ``ewm(adjust=False, ignore_na=False)`` gap semantics: after a NaN gap of
``d`` rows the update renormalizes as
``y = (alpha*x + (1-alpha)**d * y) / (alpha + (1-alpha)**d)`` — verified
empirically against the installed pandas).  **No pandas DataFrame is
constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``, no
``iterrows``).
"""
from __future__ import annotations

import copy
import hashlib
import math
from collections import OrderedDict, deque
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch11"

_SKIP = frozenset({
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "__fe_instrument__",
})

_EPS = 1e-12


def _ncols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _SKIP]


def _panel(value: Any) -> np.ndarray:
    if isinstance(value, pl.Series):
        value = value.to_frame()
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"expected a polars panel, got {type(value)!r}")
    cols = _ncols(value)
    if not cols:
        raise ValueError("panel has no instrument columns")
    out = np.empty((value.height, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        out[:, j] = value[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True)
    return out


def _panel_raw(value: Any) -> np.ndarray:
    """pl wide panel -> (rows, n_instruments) object columns (no float cast)."""
    if isinstance(value, pl.Series):
        value = value.to_frame()
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"expected a polars panel, got {type(value)!r}")
    cols = _ncols(value)
    if not cols:
        raise ValueError("panel has no instrument columns")
    out = np.empty((value.height, len(cols)), dtype=object)
    for j, c in enumerate(cols):
        out[:, j] = value[c].to_numpy(allow_copy=True)
    return out


def _panel_native(value: Any) -> np.ndarray:
    """pl wide panel -> 2D numpy preserving Date/Datetime columns as datetime64."""
    if isinstance(value, pl.Series):
        value = value.to_frame()
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"expected a polars panel, got {type(value)!r}")
    cols = _ncols(value)
    if not cols:
        raise ValueError("panel has no instrument columns")
    if all(isinstance(value[c].dtype, (pl.Date, pl.Datetime)) for c in cols):
        out = np.empty((value.height, len(cols)), dtype="datetime64[ns]")
        for j, c in enumerate(cols):
            out[:, j] = value[c].to_numpy(allow_copy=True).astype("datetime64[ns]")
        return out
    out = np.empty((value.height, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        out[:, j] = value[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True).astype(float)
    return out


def _rebuild(base: pl.DataFrame, arr: np.ndarray) -> pl.DataFrame:
    cols = _ncols(base)
    if arr.shape != (base.height, len(cols)):
        from factor_engine.backend.operator_errors import OperatorShapeError
        raise OperatorShapeError(
            f"r68 native result shape {arr.shape} does not match panel {(base.height, len(cols))}"
        )
    return base.with_columns([
        pl.Series(c, np.ascontiguousarray(arr[:, j]), dtype=pl.Float64)
        for j, c in enumerate(cols)
    ])


def _daily_panel(base: pl.DataFrame, day_ts: np.ndarray, vals: dict[str, list[float]]) -> pl.DataFrame:
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(vals[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _time_numpy(frame: pl.DataFrame, session_tz: Any) -> np.ndarray:
    tcol = None
    for c in ("__fe_time__", "date", "timestamp", "trade_date", "datetime"):
        if c in frame.columns:
            tcol = c
            break
    if tcol is None:
        return None
    s = frame[tcol]
    if isinstance(s.dtype, pl.Datetime) and s.dtype.time_zone is not None:
        from factor_engine.cleaned_operators.intraday._core import _SESSION_TZ
        tz = session_tz if session_tz is not None else _SESSION_TZ
        s = s.dt.convert_time_zone(str(tz)).dt.replace_time_zone(None)
    return s.to_numpy(allow_copy=True).astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# shared numpy micro-kernels (pandas-semantics ports)
# ---------------------------------------------------------------------------
def _pi(v, name, minimum=1):
    if isinstance(v, bool):
        raise ValueError(f"{name} must be integer")
    v = int(v)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _np_shift(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if n < a.shape[0]:
        out[n:] = a[: a.shape[0] - n]
    return out


def _np_wma(col: np.ndarray, w: int) -> np.ndarray:
    """Weighted moving average, weights 1..w newest-largest; full finite window."""
    n = col.size
    out = np.full(n, np.nan)
    weights = np.arange(1, w + 1, dtype=float)
    wsum = weights.sum()
    for t in range(w - 1, n):
        seg = col[t - w + 1: t + 1]
        if np.any(~np.isfinite(seg)):
            continue
        out[t] = float(np.dot(seg, weights) / wsum)
    return out


def _np_ewm_panel(a: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """pandas ``ewm(alpha=..., adjust=False, min_periods=...)`` port.

    ignore_na=False (default) gap semantics, verified against the installed
    pandas: after ``d`` rows since the last valid observation the update is
    ``y = (alpha*x + (1-alpha)**d * y) / (alpha + (1-alpha)**d)``; NaN rows
    carry the running mean and the min_periods gate counts valid observations.
    """
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan)
    one_m = 1.0 - alpha
    for j in range(cols):
        col = a[:, j]
        m = np.nan
        cnt = 0
        last = -1
        for t in range(rows):
            v = col[t]
            if np.isfinite(v):
                if cnt == 0:
                    m = float(v)
                else:
                    w = one_m ** (t - last)
                    m = (alpha * float(v) + w * m) / (alpha + w)
                cnt += 1
                last = t
            if cnt >= min_periods:
                out[t, j] = m
    return out


def _np_rolling_fullwindow(a: np.ndarray, w: int, fn) -> np.ndarray:
    """Trailing window of exactly ``w`` rows ending at t; NaN output when the
    window contains any NaN or during warmup (pandas rolling min_periods=w
    contract for a w-wide window)."""
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(a[:, j])
        for t in range(w - 1, rows):
            seg = col[t - w + 1: t + 1]
            if np.any(~np.isfinite(seg)):
                continue
            out[t, j] = fn(seg)
    return out


def _np_reg_fit(a: np.ndarray):
    """OLS of the window on time — authority port of indicators_v2._reg_fit_window."""
    n = a.size
    if n < 3 or not np.isfinite(a).all():
        return None
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    s_tt = float(np.dot(tc, tc))
    if s_tt <= _EPS:
        return None
    slope = float(np.dot(tc, a - a.mean()) / s_tt)
    intercept = float(a.mean() - slope * t.mean())
    resid = a - (intercept + slope * t)
    ss_res = float(np.dot(resid, resid))
    centered = a - a.mean()
    ss_tot = float(np.dot(centered, centered))
    r2 = np.nan if ss_tot <= _EPS else 1.0 - ss_res / ss_tot
    resid_std = float(np.sqrt(ss_res / (n - 2)))
    slope_se = np.nan if not np.isfinite(resid_std) else resid_std / np.sqrt(s_tt)
    return slope, intercept, resid_std, r2, slope_se


def _round_rule(x: float, rounding: str) -> int:
    if rounding == "floor":
        return int(np.floor(x))
    return int(round(x))


def _walk_periods_np(xv_col: np.ndarray, pv_col: np.ndarray, fn) -> np.ndarray:
    """NumPy mirror of transforms_v2._walk_periods (single column)."""
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        _default_require_parseable, _period_insert, _period_key,
    )
    order: list = []
    visible: OrderedDict = OrderedDict()
    require_parseable = _default_require_parseable()
    arr = np.full(len(xv_col), np.nan, dtype=float)
    for i in range(len(xv_col)):
        value = xv_col[i]
        key = _period_key(pv_col[i])
        if key is not None and np.isfinite(value):
            if key not in visible:
                _period_insert(order, key, require_parseable=require_parseable)
            visible[key] = float(value)
        if key is None or key not in visible:
            continue
        try:
            arr[i] = fn(order, visible, key)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            arr[i] = np.nan
    return arr


def _finite_v(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive_v(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


class _FiscalEventView:
    """NumPy port of fiscal_event_ops.FiscalEventView."""

    def __init__(self, events: list, ordinals: np.ndarray):
        self.events = events
        self.ordinals = ordinals

    @classmethod
    def from_arrays(cls, values: np.ndarray, raw_periods: np.ndarray, policy: str) -> "_FiscalEventView":
        from factor_engine.cleaned_operators.fiscal_strict import period_ordinal
        ordinal_values = np.full(raw_periods.shape, np.nan, dtype=float)
        for row in range(raw_periods.shape[0]):
            for col in range(raw_periods.shape[1]):
                ordinal = period_ordinal(raw_periods[row, col])
                if ordinal is not None:
                    ordinal_values[row, col] = ordinal
        snapshots: list = []
        state: list = [dict() for _ in range(values.shape[1])]
        first_seen: list = [set() for _ in range(values.shape[1])]
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ordinal = ordinal_values[row, col]
                value = values[row, col]
                if not np.isfinite(ordinal) or not _finite_v(value):
                    continue
                key = int(ordinal)
                if policy == "latest_available" or key not in first_seen[col]:
                    state[col][key] = float(value)
                first_seen[col].add(key)
            snapshots.append([dict(sorted(column.items())) for column in state])
        return cls(snapshots, ordinal_values)

    def history(self, row: int, col: int, *, require_consecutive: bool) -> list:
        current = self.ordinals[row, col]
        if not np.isfinite(current):
            return []
        history = list(self.events[row][col].items())
        if not history:
            return []
        history = [(int(k), float(v)) for k, v in history if k <= int(current) and _finite_v(v)]
        history.sort()
        if require_consecutive and history:
            contiguous: list = [history[-1]]
            for item in reversed(history[:-1]):
                if contiguous[0][0] - item[0] != 1:
                    break
                contiguous.insert(0, item)
            history = contiguous
        return history


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _k_CoppockCurve(b: dict) -> pl.DataFrame:
    close = _panel(b["close"])
    roc1 = max(1, int(b.get("roc1", 14)))
    roc2 = max(1, int(b.get("roc2", 11)))
    wma_window = max(1, int(b.get("wma_window", 10)))
    roc_mode = str(b.get("roc_mode", "pct"))
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(cols):
            c = np.ascontiguousarray(close[:, j])
            if roc_mode == "pct":
                r1 = c / _np_shift(c, roc1) - 1.0
                r2 = c / _np_shift(c, roc2) - 1.0
            else:
                r1 = np.log(c / _np_shift(c, roc1))
                r2 = np.log(c / _np_shift(c, roc2))
            out[:, j] = _np_wma(r1 + r2, wma_window)
    return _rebuild(b["close"], out)


def _k_reg_slope_tstat(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 20), "window", 3)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(pos[:, j])
        for t in range(w - 1, rows):
            fit = _np_reg_fit(col[t - w + 1: t + 1])
            if fit is None:
                continue
            slope, _intercept, _resid_std, _r2, slope_se = fit
            if not (np.isfinite(slope_se) and slope_se > 0.0):
                continue
            out[t, j] = float(slope / slope_se)
    return _rebuild(b["close"], out)


def _k_bvc_imbalance_ma(b: dict) -> pl.DataFrame:
    f = _pi(b.get("fast_window", 5), "fast_window", 1)
    s = _pi(b.get("slow_window", 40), "slow_window", 1)
    if f >= s:
        raise ValueError("fast_window must be < slow_window")
    close = _panel(b["close"])
    volume = _panel(b["volume"])
    prev = _np_shift(close, 1)
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(close) & (close > 0.0)
            & np.isfinite(prev) & (prev > 0.0)
            & np.isfinite(volume) & (volume > 0.0)
        )
        diff = close - prev
        sign = np.sign(diff)
    flow = np.where(valid, sign, np.nan) * np.where(valid, volume, np.nan)
    vol_m = np.where(valid, volume, np.nan)
    num = _np_ewm_panel(flow, 2.0 / (f + 1.0), f) - _np_ewm_panel(flow, 2.0 / (s + 1.0), s)
    den = _np_ewm_panel(vol_m, 2.0 / (s + 1.0), s)
    den = np.where(den > 0.0, den, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return _rebuild(b["close"], out)


def _k_keltner_breakout_strength(b: dict) -> pl.DataFrame:
    ema_w = _pi(b.get("ema_window", 20), "ema_window", 1)
    atr_w = _pi(b.get("atr_window", 20), "atr_window", 1)
    mult = float(b.get("multiplier", 1.0))
    if mult <= 0:
        raise ValueError("multiplier must be > 0 (Keltner multiplier=0 degenerates to the mid band)")
    high = _panel(b["high"])
    low = _panel(b["low"])
    close = _panel(b["close"])
    prev = _np_shift(close, 1)
    with np.errstate(invalid="ignore"):
        tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    mid = _np_ewm_panel(close, 2.0 / (ema_w + 1.0), ema_w)
    atr = _np_ewm_panel(tr, 1.0 / atr_w, atr_w)
    u = mid + mult * atr
    lo = mid - mult * atr
    span = np.where((u - lo) == 0.0, np.nan, u - lo)
    with np.errstate(invalid="ignore"):
        above = np.where(close > u, close - u, 0.0)
        below = np.where(close < lo, close - lo, 0.0)
        above = np.where(np.isfinite(close) & np.isfinite(u), above, np.nan)
        below = np.where(np.isfinite(close) & np.isfinite(lo), below, np.nan)
        out = (above + below) / span
    return _rebuild(b["high"], out)


def _k_group_feature(b: dict, which: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.group_spectrum import (
        _group_spectrum_series, _GROUP_SCHEMA_VERSION_DEFAULT,
        _BREADTH_STABILITY_WINDOW, _REL_GAP_THRESHOLD,
    )
    f1 = _panel(b["f1"])
    feats = np.stack([f1, _panel(b["f2"]), _panel(b["f3"])], axis=2)
    group = _panel_raw(b["group"])
    kw = dict(
        group_schema_version=b.get("group_schema_version", _GROUP_SCHEMA_VERSION_DEFAULT),
        breadth_window=b.get("breadth_window", _BREADTH_STABILITY_WINDOW),
    )
    if which == "second_mode_localization":
        kw["eigen_gap"] = float(b.get("eigen_gap", _REL_GAP_THRESHOLD))
    return _rebuild(b["f1"], _group_spectrum_series(feats, group, which, **kw))


def _k_day_diff(b: dict, left_key: str, right_key: str) -> pl.DataFrame:
    a = _panel_native(b[left_key])
    bb = _panel_native(b[right_key])
    if a.shape != bb.shape:
        from factor_engine.backend.operator_errors import OperatorShapeError
        raise OperatorShapeError(f"{left_key} {a.shape} vs {right_key} {bb.shape} not aligned")
    out = np.full(a.shape, np.nan, dtype=float)
    try:
        ad = a.astype("datetime64[ns]")
        bd = bb.astype("datetime64[ns]")
        delta = (bd - ad) / np.timedelta64(1, "D")
        valid = ~np.isnat(ad) & ~np.isnat(bd)
        out[valid] = np.asarray(delta[valid], dtype=float)
    except (ValueError, TypeError):
        out = bb.astype(float) - a.astype(float)
    return _rebuild(b[left_key], out)


def _k_fin_announcement_lag(b: dict) -> pl.DataFrame:
    return _k_day_diff(b, "period_end_date", "pub_date")


def _k_calendar_day_diff(b: dict) -> pl.DataFrame:
    return _k_day_diff(b, "date1", "date2")


def _k_intra_supply_absorption_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.event_response import (
        _supply_absorption_day, DataDegeneracy,
    )
    event = str(b.get("event", "all")).lower()
    if event not in {"up_impulse", "down_impulse", "all"}:
        raise ValueError(f"unknown event {event!r}")
    horizon = b.get("horizon", 30)
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    output = str(b.get("output", "absorption")).lower()
    if output not in {"absorption", "price_per_amount", "volume_no_drop", "downside_resilience"}:
        raise ValueError(f"unknown output {output!r}")
    price_f, volume_f, amount_f = b["price"], b["volume"], b["amount"]
    price = _panel(price_f)
    volume = _panel(volume_f)
    amount = _panel(amount_f)
    times = _time_numpy(price_f, b.get("session_tz"))
    day_keys = times.astype("datetime64[D]")
    uniq_days = np.unique(day_keys)
    names = _ncols(price_f)
    out_days: list[np.datetime64] = []
    vals: dict[str, list[float]] = {c: [] for c in names}
    for d in uniq_days:
        sel = day_keys == d
        out_days.append(np.datetime64(d, "ns"))
        for j, c in enumerate(names):
            p_d = price[sel, j]
            v_d = volume[sel, j]
            a_d = amount[sel, j]
            t_d = times[sel]
            if int(np.sum(np.isfinite(p_d))) < 2:
                vals[c].append(np.nan)
                continue
            try:
                vals[c].append(float(
                    _supply_absorption_day(p_d, v_d, a_d, t_d, event, int(horizon), output)))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                vals[c].append(np.nan)
    return _daily_panel(price_f, np.asarray(out_days), vals)


def _k_fiscal_pair_direction_agreement(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_event_ops import _POLICIES
    policy = str(b.get("revision_policy", "latest_available")).lower()
    if policy not in _POLICIES:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    periods = _positive_v(b.get("periods", 8), "periods")
    min_periods = _positive_v(b.get("min_periods", 3), "min_periods")
    require_consecutive = bool(b.get("require_consecutive", True))
    xv = _panel(b["signal_x"])
    yv = _panel(b["signal_y"])
    pid = _panel_raw(b["period_id"])
    viewx = _FiscalEventView.from_arrays(xv, pid, policy)
    viewy = _FiscalEventView.from_arrays(yv, pid, policy)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan)
    for row in range(rows):
        for col in range(cols):
            hx = dict(viewx.history(row, col, require_consecutive=require_consecutive))
            hy = dict(viewy.history(row, col, require_consecutive=require_consecutive))
            pairs = [(hx[o], hy[o]) for o in sorted(set(hx) & set(hy))][-periods:]
            pairs = [(a_, b_) for a_, b_ in pairs if _finite_v(a_) and _finite_v(b_) and a_ != 0 and b_ != 0]
            if len(pairs) >= min_periods:
                out[row, col] = float(np.mean([np.sign(a_) == np.sign(b_) for a_, b_ in pairs]))
    return _rebuild(b["signal_x"], out)


def _k_group_distribution_js_divergence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.parameter_validation import strict_integer
    x = _panel(b["x"])
    gv = _panel_raw(b["group"])
    nb = strict_integer(b.get("bins", 10), "bins", minimum=2)
    mg = strict_integer(b.get("min_group_size", 5), "min_group_size", minimum=1)
    exclude = bool(b.get("exclude_group_from_reference", True))
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    def _ext_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        full_edges = np.concatenate(([-np.inf], np.asarray(edges, dtype=float), [np.inf]))
        return np.histogram(values, bins=full_edges)[0].astype(float)

    def _js(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        kl = 0.0
        for a_, b_ in ((p, m), (q, m)):
            for pi, mi in zip(a_, b_):
                if pi <= 0.0:
                    continue
                kl += pi * (np.log(pi + _EPS) - np.log(mi + _EPS))
        return float(0.5 * kl)

    for r in range(rows):
        xr = x[r]
        g_row = gv[r]
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            positions.setdefault(lab, []).append(i)
        market_mask = np.isfinite(xr)
        if int(market_mask.sum()) < 2:
            continue
        market = xr[market_mask]
        if exclude:
            for lab, members in positions.items():
                gvals = xr[members]
                gvals = gvals[np.isfinite(gvals)]
                if gvals.size < mg:
                    continue
                other = np.ones(cols, dtype=bool)
                other[members] = False
                ref = xr[other & market_mask]
                if ref.size < 2.0 * nb:
                    continue
                ref_edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, nb + 1)))
                if ref_edges.size < 2:
                    continue
                g_bin = _ext_hist(gvals, ref_edges)
                if g_bin.sum() <= 0:
                    continue
                gp = g_bin / g_bin.sum()
                r_bin = _ext_hist(ref, ref_edges)
                rp = r_bin / r_bin.sum()
                val = _js(gp, rp)
                for i in members:
                    out[r, i] = val
        else:
            if market.size < 2.0 * nb:
                continue
            edges = np.unique(np.quantile(market, np.linspace(0.0, 1.0, nb + 1)))
            if int(edges.size - 1) < 1:
                continue
            market_bin = _ext_hist(market, edges)
            market_p = market_bin / market_bin.sum()
            for lab, members in positions.items():
                gvals = xr[members]
                gvals = gvals[np.isfinite(gvals)]
                if gvals.size < mg:
                    continue
                g_bin = _ext_hist(gvals, edges)
                if g_bin.sum() <= 0:
                    continue
                gp = g_bin / g_bin.sum()
                val = _js(gp, market_p)
                for i in members:
                    out[r, i] = val
    return _rebuild(b["x"], out)


def _k_consolidation_pct(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 20), "window", 2)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(pos[:, j])
        for t in range(w - 1, rows):
            seg = col[t - w + 1: t + 1]
            if np.any(~np.isfinite(seg)):
                continue
            mean = float(seg.mean())
            if not (mean > 0.0):
                continue
            std = float(seg.std(ddof=1))
            out[t, j] = std / mean
    return _rebuild(b["close"], out)


def _k_ts_poly2(b: dict, z: bool) -> pl.DataFrame:
    from factor_engine.cleaned_operators._numpy_kernels import (
        ts_poly2_forecast_error_, ts_poly2_forecast_error_z_,
    )
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    d = strict_int(b.get("d", 20), "d", minimum=3)
    x = _panel(b["x"])
    fn = ts_poly2_forecast_error_z_ if z else ts_poly2_forecast_error_
    out = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        out[:, j] = fn(x[:, j], d)
    return _rebuild(b["x"], out)


def _k_ts_poly2_forecast_error(b: dict) -> pl.DataFrame:
    return _k_ts_poly2(b, False)


def _k_ts_poly2_forecast_error_z(b: dict) -> pl.DataFrame:
    return _k_ts_poly2(b, True)


def _k_ts_multifractal_spectrum_width(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.multifractal import (
        _spectrum_width_series, _check_window,
    )
    w = _check_window(b.get("window", 120))
    return _rebuild(b["x"], _spectrum_width_series(_panel(b["x"]), w))


def _k_relation_topk_sum(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.ops import (
        _MISSING_SEMANTIC_CHOICES, HolderRankMissingSemantic,
    )
    panels = []
    for i in range(1, 11):
        v = b.get(f"s{i}")
        if isinstance(v, (pl.DataFrame, pl.Series)):
            panels.append(v)
    missing_semantic = str(b.get("missing_semantic", "outside_top_k"))
    if len(panels) < 2:
        raise ValueError("relation_topk_sum requires at least two ranked panels")
    if missing_semantic not in _MISSING_SEMANTIC_CHOICES:
        raise ValueError(
            f"missing_semantic must be one of {_MISSING_SEMANTIC_CHOICES!r}; "
            f"got {missing_semantic!r}"
        )
    stacked = np.stack([_panel(p) for p in panels], axis=0)
    if HolderRankMissingSemantic.permits_zero(missing_semantic):
        values = np.nan_to_num(stacked, nan=0.0)
    else:
        values = stacked
    total = np.nansum(values, axis=0)
    total = np.where(np.isfinite(values).sum(axis=0) > 0, total, np.nan)
    return _rebuild(panels[0], total)


def _k_transition_count(b: dict, forward: bool) -> pl.DataFrame:
    w = int(b.get("window", 60))
    missing_policy = str(b.get("missing_policy", "break"))
    if missing_policy not in {"break", "false"}:
        raise ValueError("missing_policy must be 'break' or 'false'")
    mv = _panel(b["member"])
    rows, cols = mv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            segment = mv[start: row + 1, col]
            valid = np.isfinite(segment)
            if valid.sum() < 2:
                continue
            if missing_policy == "break":
                current = segment[1:] != 0
                previous = segment[:-1] != 0
                pair_valid = valid[1:] & valid[:-1]
                entry = pair_valid & current & ~previous
                exit_ = pair_valid & ~current & previous
            else:
                truth = valid & (segment != 0)
                entry = truth[1:] & ~truth[:-1]
                exit_ = ~truth[1:] & truth[:-1]
            out[row, col] = float(np.sum(entry) if forward else np.sum(exit_))
    return _rebuild(b["member"], out)


def _k_relation_entry_count(b: dict) -> pl.DataFrame:
    return _k_transition_count(b, forward=True)


def _k_relation_exit_count(b: dict) -> pl.DataFrame:
    return _k_transition_count(b, forward=False)


def _k_report_change_breadth(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_events import _make_change_z
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    p = strict_int(b.get("periods", 1), "periods", minimum=1)
    eps_v = float(b.get("eps", 0.5))
    if not np.isfinite(eps_v) or eps_v < 0.0:
        raise ValueError("eps must be finite and >= 0")
    calc = _make_change_z(p)
    f1 = _panel(b["f1"])
    pid = _panel_raw(b["period_id"])
    zs = []
    for fv in (f1, _panel(b["f2"]), _panel(b["f3"])):
        z = np.full(fv.shape, np.nan, dtype=float)
        for j in range(fv.shape[1]):
            z[:, j] = _walk_periods_np(fv[:, j], pid[:, j], calc)
        zs.append(z)
    a = np.stack(zs)
    K = a.shape[0]
    pos = np.sum(a > eps_v, axis=0)
    neg = np.sum(a < -eps_v, axis=0)
    breadth = (pos - neg) / K
    breadth[np.isnan(a).any(axis=0)] = np.nan
    return _rebuild(b["f1"], breadth)


def _k_group_feature_second_mode_localization(b: dict) -> pl.DataFrame:
    return _k_group_feature(b, "second_mode_localization")


def _k_group_feature_mode_share(b: dict) -> pl.DataFrame:
    return _k_group_feature(b, "mode_share")


def _k_group_feature_effective_rank(b: dict) -> pl.DataFrame:
    return _k_group_feature(b, "effective_rank")


def _k_group_feature_spectral_gap(b: dict) -> pl.DataFrame:
    return _k_group_feature(b, "spectral_gap")


def _k_donchian_channels(b: dict):
    w = _pi(b.get("window", 20), "window", 2)
    high = _panel(b["high"])
    low = _panel(b["low"])
    u = _np_rolling_fullwindow(high, w, lambda seg: float(seg.max()))
    l = _np_rolling_fullwindow(low, w, lambda seg: float(seg.min()))
    return u, l


def _k_donchian_breakout_up(b: dict) -> pl.DataFrame:
    u, _l = _k_donchian_channels(b)
    prev_u = _np_shift(u, 1)
    close = _panel(b["close"])
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        for t in range(rows):
            pu = prev_u[t, j]
            cv = close[t, j]
            if not (np.isfinite(pu) and np.isfinite(cv) and cv > 0.0):
                continue
            out[t, j] = (cv / pu - 1.0) if cv >= pu else 0.0
    return _rebuild(b["high"], out)


def _k_donchian_breakout_down(b: dict) -> pl.DataFrame:
    _u, l = _k_donchian_channels(b)
    prev_l = _np_shift(l, 1)
    close = _panel(b["close"])
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        for t in range(rows):
            pl_ = prev_l[t, j]
            cv = close[t, j]
            if not (np.isfinite(pl_) and np.isfinite(cv) and cv > 0.0):
                continue
            out[t, j] = (cv / pl_ - 1.0) if cv <= pl_ else 0.0
    return _rebuild(b["high"], out)


def _k_donchian_width_pct(b: dict) -> pl.DataFrame:
    u, l = _k_donchian_channels(b)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (u - l) / pos
    return _rebuild(b["high"], out)


def _k_chikou_distance_pct(b: dict) -> pl.DataFrame:
    w = _pi(b.get("kijun_window", 26), "kijun_window", 2)
    high = _panel(b["high"])
    low = _panel(b["low"])
    close = _panel(b["close"])
    hi = _np_rolling_fullwindow(high, w, lambda seg: float(seg.max()))
    lo = _np_rolling_fullwindow(low, w, lambda seg: float(seg.min()))
    kijun = (hi + lo) / 2.0
    pos = np.where(close > 0.0, close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (pos - kijun) / pos
    return _rebuild(b["high"], out)


def _k_event_hawkes_branching_ratio_proxy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.event_response import (
        _hawkes_branching_ratio_series, _strict_history_window, strict_int,
    )
    w = _strict_history_window(b.get("window", 120))
    L = strict_int(b.get("max_lag", 10), "max_lag", lower=1)
    me = strict_int(b.get("min_events", 5), "min_events", lower=2)
    ev = _panel(b["event"])
    rows, cols = ev.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _hawkes_branching_ratio_series(ev[:, c], w, L, me)
    return _rebuild(b["event"], out)


def _fin_divergence_kernel(a_key: str) -> Callable[[dict], pl.DataFrame]:
    def _k(b: dict) -> pl.DataFrame:
        from factor_engine.cleaned_operators.fiscal_strict import (
            GROWTH_FORBIDDEN_FLOW_TYPES, flow_types as _flow_types,
        )
        from factor_engine.cleaned_operators.fundamental.transforms_v2 import _lag_value
        flow_type = b.get("flow_type")
        if flow_type is not None:
            types = _flow_types(flow_type, 2)
            for slot_label, t in zip(("a", "b"), types):
                if t in GROWTH_FORBIDDEN_FLOW_TYPES:
                    raise ValueError(
                        f"fin_divergence slot {slot_label}: growth over a "
                        f"CumulativeYTDFlow input is not a period growth rate; "
                        "convert with fin_quarter_from_cumulative first."
                    )

        def _pct1(order, visible, current):
            cur = float(visible[current])
            old = _lag_value(order, visible, current, 1)
            if not np.isfinite(old) or abs(old) <= _EPS:
                return np.nan
            return float((cur - old) / abs(old))

        a = _panel(b[a_key])
        bb = _panel(b["operating_revenue"])
        pid = _panel_raw(b["period_id"])
        rows, cols = a.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for j in range(cols):
            out[:, j] = (
                _walk_periods_np(a[:, j], pid[:, j], _pct1)
                - _walk_periods_np(bb[:, j], pid[:, j], _pct1)
            )
        return _rebuild(b[a_key], out)
    return _k


def _k_true_range_surprise(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 2), "window", 2)
    high = _panel(b["high"])
    low = _panel(b["low"])
    close = _panel(b["close"])
    prev = _np_shift(close, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
        prev_pos = np.where(np.isfinite(prev) & (prev > 0.0), prev, np.nan)
        ratio = tr / prev_pos
        mean = _np_rolling_fullwindow(ratio, w, lambda seg: float(seg.mean()))
        mean = np.where(mean == 0.0, np.nan, mean)
        out = ratio / mean - 1.0
    return _rebuild(b["high"], out)


def _k_spectral_energy_ratio(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 20), "window", 4)
    k = _pi(b.get("top_k", 2), "top_k", 1)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(pos[:, j])
        for t in range(w - 1, rows):
            seg = col[t - w + 1: t + 1]
            if np.any(~np.isfinite(seg)):
                continue
            x = seg - seg.mean()
            spec = np.fft.rfft(x)
            power = spec.real ** 2 + spec.imag ** 2
            power = power[1:]
            total = power.sum()
            if total <= _EPS:
                out[t, j] = 0.0
                continue
            kk = min(k, power.size)
            out[t, j] = float(np.sort(power)[::-1][:kk].sum() / total)
    return _rebuild(b["close"], out)


def _k_reg_forecast_error_pct(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 20), "window", 4)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)
    rows, cols = close.shape
    out = np.full_like(close, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(pos[:, j])
        for t in range(w - 1, rows):
            seg = col[t - w + 1: t + 1]
            y = seg[:-1]
            x_t = seg[-1]
            fit = _np_reg_fit(y)
            if fit is None:
                continue
            slope, intercept = fit[0], fit[1]
            out[t, j] = float((x_t - (intercept + slope * float(y.size))) / x_t)
    return _rebuild(b["close"], out)


def _k_report_revision_magnitude(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.report_timing import (
        _revision_magnitude_series, _validate_indicator,
    )
    x = _panel(b["x"])
    pv = _panel(b["prev_x"])
    cp = _panel_raw(b["current_period_id"])
    pp = _panel_raw(b["prev_period_id"])
    w = int(b.get("window", 8))
    mp = int(b.get("min_periods", 3))
    rev = b.get("revision_event")
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        re_c = None
        if isinstance(rev, (pl.DataFrame, pl.Series)):
            re_c = _panel(rev)[:, c]
            _validate_indicator(re_c, "report_revision_magnitude", "revision_event")
        out[:, c] = _revision_magnitude_series(
            x[:, c], pv[:, c], cp[:, c], pp[:, c], w, mp, re_c)
    return _rebuild(b["x"], out)


def _candle_geometry(b: dict):
    o = _panel(b["open"])
    h = _panel(b["high"])
    l = _panel(b["low"])
    c = _panel(b["close"])
    with np.errstate(invalid="ignore"):
        valid = (
            np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
            & (c > 0.0) & (h > l)
            & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c))
        )
        rng = np.where(valid, h - l, np.nan)
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        sign = np.sign(c - o)
        body = np.where(valid, sign * np.abs(c - o) / rng, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            wick = np.where(valid, (lower - upper) / rng, np.nan)
    return o, h, l, c, valid, rng, body, wick


def _k_candle_body_strength(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 2), "window", 2)
    o, _h, _l, _c, _valid, _rng, body, _wick = _candle_geometry(b)
    out = _np_rolling_fullwindow(body, w, lambda seg: float(seg.mean()))
    return _rebuild(b["open"], out)


def _k_candle_wick_balance(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 2), "window", 2)
    o, _h, _l, _c, _valid, _rng, _body, wick = _candle_geometry(b)
    out = _np_rolling_fullwindow(wick, w, lambda seg: float(seg.mean()))
    return _rebuild(b["open"], out)


def _k_candle_pattern_count(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 2), "window", 2)
    o, h, l, c, valid, rng, _body, _wick = _candle_geometry(b)
    upper = np.where(valid, h - np.maximum(o, c), np.nan)
    lower = np.where(valid, np.minimum(o, c) - l, np.nan)
    body_abs = np.where(valid, np.abs(c - o), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        doji = body_abs / rng < 0.1
        dom_upper = upper / rng >= 0.7
        dom_lower = lower / rng >= 0.7
        small_lower = lower / rng <= 0.15
        small_upper = upper / rng <= 0.15
    pattern = doji | (dom_upper & small_lower) | (dom_lower & small_upper)
    pat_f = np.where(valid, pattern.astype(float), np.nan)
    valid_f = valid.astype(float)
    all_present = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)).astype(float)
    rows_n, cols_n = o.shape
    num = np.full((rows_n, cols_n), np.nan, dtype=float)
    den = np.full((rows_n, cols_n), np.nan, dtype=float)
    rows_cnt = np.full((rows_n, cols_n), np.nan, dtype=float)
    for j in range(cols_n):
        pat_col = np.ascontiguousarray(pat_f[:, j])
        val_col = np.ascontiguousarray(valid_f[:, j])
        all_col = np.ascontiguousarray(all_present[:, j])
        for t in range(w - 1, rows_n):
            seg_pat = pat_col[t - w + 1: t + 1]
            fin = np.isfinite(seg_pat)
            if fin.any():
                num[t, j] = float(seg_pat[fin].sum())
            den[t, j] = float(val_col[t - w + 1: t + 1].sum())
            rows_cnt[t, j] = float(all_col[t - w + 1: t + 1].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = num / np.where(den > 0.0, den, np.nan)
    out = np.where(rows_cnt >= float(w), frac, np.nan)
    return _rebuild(b["open"], out)


def _k_ts_transfer_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_information import (
        _te_feasibility, _rolling_apply_2d_pair, _transfer_entropy_window,
    )
    w = int(b.get("window", 60))
    nb = int(b.get("bins", 3))
    lg = int(b.get("lag", 1))
    ratio = float(b.get("min_cells_ratio", 1.0))
    if not (2 <= nb <= 8):
        raise ValueError("ts_transfer_entropy requires 2 <= bins <= 8")
    if lg < 1:
        raise ValueError("ts_transfer_entropy requires lag >= 1")
    ok, reason, mt = _te_feasibility(
        window=w, bins=nb, lag=lg, min_cells_ratio=ratio,
        min_transitions=b.get("min_transitions"),
    )
    if not ok:
        raise ValueError(
            f"ts_transfer_entropy {reason}; raise window or lower bins "
            "(default window=60 supports bins<=4)"
        )
    target = _panel(b["target"])
    source = _panel(b["source"])
    return _rebuild(b["target"], _rolling_apply_2d_pair(
        target, source, w,
        lambda a, bb: _transfer_entropy_window(a, bb, nb, lg, mt, ratio)))


def _k_ts_ssa_prior_reconstruction_error(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.hankel import (
        _check_hankel_params, _ssa_prior_reconstruction_error_series,
    )
    w, e, k, mcf = _check_hankel_params(
        b.get("window", 60), b.get("embedding_dim", 15),
        b.get("n_components", 3), b.get("min_contiguous_fraction", 0.8),
    )
    return _rebuild(b["x"], _ssa_prior_reconstruction_error_series(
        _panel(b["x"]), w, e, k, "strict_contiguous", mcf))


def _k_HMA(b: dict) -> pl.DataFrame:
    x = _panel(b["x"])
    w = max(2, int(b.get("window", 16)))
    rounding = str(b.get("rounding", "floor"))
    n2 = max(1, _round_rule(w / 2.0, rounding))
    ns = max(1, _round_rule(np.sqrt(w), rounding))
    rows, cols = x.shape
    out = np.full_like(x, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(x[:, j])
        inner = 2.0 * _np_wma(col, n2) - _np_wma(col, w)
        out[:, j] = _np_wma(inner, ns)
    return _rebuild(b["x"], out)


def _k_ALMA(b: dict) -> pl.DataFrame:
    x = _panel(b["x"])
    w = max(2, int(b.get("window", 10)))
    offset = float(b.get("offset", 0.85))
    sigma = float(b.get("sigma", 6.0))
    m = offset * (w - 1)
    s = w / max(sigma, 1e-6)
    weights = np.array(
        [np.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(w)], dtype=float)
    total = weights.sum()
    if total <= _EPS:
        weights = np.ones(w, dtype=float) / w
    else:
        weights = weights / total
    rows, cols = x.shape
    out = np.full_like(x, np.nan)
    for j in range(cols):
        col = np.ascontiguousarray(x[:, j])
        for t in range(w - 1, rows):
            seg = col[t - w + 1: t + 1]
            if np.any(~np.isfinite(seg)):
                continue
            out[t, j] = float(np.dot(seg, weights))
    return _rebuild(b["x"], out)


def _quantile_stat(yv: np.ndarray, xv: np.ndarray, w: int, mp: int, q: float,
                   stat: str, lag: int) -> np.ndarray:
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        pinball_quantile_fit,
    )
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        ycol, xcol = yv[:, c], xv[:, c]
        for row in range(rows):
            fit_end = row - lag
            if fit_end < 0:
                continue
            start = max(0, fit_end - w + 1)
            seg_y = ycol[start: fit_end + 1]
            seg_x = xcol[start: fit_end + 1]
            valid = np.isfinite(seg_y) & np.isfinite(seg_x)
            if valid.sum() < mp:
                continue
            vy = seg_y[valid]
            vx = seg_x[valid]
            if np.std(vx) <= 0.0:
                continue
            design = np.column_stack([np.ones(vy.size), vx])
            beta = pinball_quantile_fit(design, vy, q)
            if beta is None:
                continue
            if stat == "coeff":
                out[row, c] = float(beta[1])
            else:  # resid
                with np.errstate(over="ignore", invalid="ignore"):
                    if np.isfinite(ycol[row]) and np.isfinite(xcol[row]):
                        terms = np.array([1.0, xcol[row]])
                        pred_now = float(np.dot(terms, beta))
                        resid = float(ycol[row] - pred_now)
                        if np.isfinite(pred_now) and np.isfinite(resid):
                            out[row, c] = resid
    return out


def _k_ts_quantile_regression_coeff(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        validate_multi_configured_history,
    )
    y, x = b["y"], b["x"]
    quantile = float(b.get("q", 0.5))
    if not (0.0 < quantile < 1.0):
        raise ValueError("q must be in (0, 1)")
    w = int(b.get("window", 60))
    mp_raw = int(b.get("min_periods", 10))
    validate_multi_configured_history(w, mp_raw, 1, True, fit_lag=0)
    mp = max(mp_raw, 10)
    return _rebuild(y, _quantile_stat(_panel(y), _panel(x), w, mp, quantile, "coeff", 0))


def _k_ts_quantile_regression_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        validate_multi_configured_history,
    )
    y, x = b["y"], b["x"]
    quantile = float(b.get("q", 0.5))
    if not (0.0 < quantile < 1.0):
        raise ValueError("q must be in (0, 1)")
    w = int(b.get("window", 60))
    mp_raw = int(b.get("min_periods", 10))
    validate_multi_configured_history(w, mp_raw, 1, True, fit_lag=0)
    mp = max(mp_raw, 10)
    return _rebuild(y, _quantile_stat(_panel(y), _panel(x), w, mp, quantile, "resid", 0))


def _k_ts_quantile_beta_spread_prior(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        validate_multi_configured_history,
    )
    y, x = b["y"], b["x"]
    qh, ql = float(b.get("q_high", 0.9)), float(b.get("q_low", 0.1))
    if not (0.0 < ql < qh < 1.0):
        raise ValueError("q_low < q_high must hold in (0, 1)")
    w = int(b.get("window", 60))
    mp_raw = int(b.get("min_periods", 10))
    validate_multi_configured_history(w, mp_raw, 1, True, fit_lag=1)
    mp = max(mp_raw, 10)
    yv, xv = _panel(y), _panel(x)
    return _rebuild(y, _quantile_stat(yv, xv, w, mp, qh, "coeff", 1)
                    - _quantile_stat(yv, xv, w, mp, ql, "coeff", 1))


def _k_group_multi_level_rank_consistency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.peer_ops import _within_group_rank
    x = _panel(b["x"])
    g1 = _panel_raw(b["group1"])
    g2 = _panel_raw(b["group2"])
    g3 = _panel_raw(b["group3"])
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            ranks = np.array([
                _within_group_rank(x[row], g1[row], g1[row, col], col),
                _within_group_rank(x[row], g2[row], g2[row, col], col),
                _within_group_rank(x[row], g3[row], g3[row, col], col),
            ])
            if np.any(~np.isfinite(ranks)):
                continue
            sd = float(np.std(ranks))
            out[row, col] = float(np.clip(1.0 - sd * (3.0 / np.sqrt(2.0)), 0.0, 1.0))
    return _rebuild(b["x"], out)


def _state_flip_age_chunk(chunk: np.ndarray) -> float:
    if np.any(np.isnan(chunk)):
        return np.nan
    compressed = chunk[np.concatenate(([True], chunk[1:] != chunk[:-1]))]
    if compressed.size < 2:
        return np.nan
    last_flip_idx = None
    for i in range(compressed.size - 1, 0, -1):
        if compressed[i] != compressed[i - 1]:
            last_flip_idx = i
            break
    if last_flip_idx is None:
        return np.nan
    orig_idx = 0
    comp_count = 0
    for i in range(chunk.size):
        if i == 0 or chunk[i] != chunk[i - 1]:
            if comp_count == last_flip_idx:
                orig_idx = i
                break
            comp_count += 1
    return float(chunk.size - 1 - orig_idx)


def _k_state_flip_age(b: dict) -> pl.DataFrame:
    w = _pi(b.get("window", 2), "window", 2)
    arr = _panel(b["state"])
    if not bool(np.all(np.isnan(arr) | np.isfinite(arr))):
        raise ValueError(
            "state panel must contain finite state codes or NaN; ±Inf is "
            "out-of-domain"
        )
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(w - 1, rows):
            out[r, c] = _state_flip_age_chunk(col[r - w + 1: r + 1])
    return _rebuild(b["state"], out)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "CoppockCurve": _k_CoppockCurve,
    "reg_slope_tstat": _k_reg_slope_tstat,
    "bvc_imbalance_ma": _k_bvc_imbalance_ma,
    "keltner_breakout_strength": _k_keltner_breakout_strength,
    "group_feature_mode_share": _k_group_feature_mode_share,
    "group_feature_effective_rank": _k_group_feature_effective_rank,
    "group_feature_spectral_gap": _k_group_feature_spectral_gap,
    "fin_announcement_lag": _k_fin_announcement_lag,
    "intra_supply_absorption_score": _k_intra_supply_absorption_score,
    "fiscal_pair_direction_agreement": _k_fiscal_pair_direction_agreement,
    "group_distribution_js_divergence": _k_group_distribution_js_divergence,
    "consolidation_pct": _k_consolidation_pct,
    "ts_poly2_forecast_error": _k_ts_poly2_forecast_error,
    "ts_multifractal_spectrum_width": _k_ts_multifractal_spectrum_width,
    "relation_topk_sum": _k_relation_topk_sum,
    "relation_entry_count": _k_relation_entry_count,
    "relation_exit_count": _k_relation_exit_count,
    "report_change_breadth": _k_report_change_breadth,
    "group_feature_second_mode_localization": _k_group_feature_second_mode_localization,
    "ts_poly2_forecast_error_z": _k_ts_poly2_forecast_error_z,
    "donchian_breakout_up": _k_donchian_breakout_up,
    "donchian_breakout_down": _k_donchian_breakout_down,
    "calendar_day_diff": _k_calendar_day_diff,
    "event_hawkes_branching_ratio_proxy": _k_event_hawkes_branching_ratio_proxy,
    "donchian_width_pct": _k_donchian_width_pct,
    "chikou_distance_pct": _k_chikou_distance_pct,
    "fin_receivable_sales_divergence": _fin_divergence_kernel("account_receivable"),
    "fin_inventory_sales_divergence": _fin_divergence_kernel("inventories"),
    "fin_cash_sales_divergence": _fin_divergence_kernel("goods_sale_cash"),
    "true_range_surprise": _k_true_range_surprise,
    "spectral_energy_ratio": _k_spectral_energy_ratio,
    "reg_forecast_error_pct": _k_reg_forecast_error_pct,
    "report_revision_magnitude": _k_report_revision_magnitude,
    "candle_body_strength": _k_candle_body_strength,
    "candle_wick_balance": _k_candle_wick_balance,
    "candle_pattern_count": _k_candle_pattern_count,
    "ts_transfer_entropy": _k_ts_transfer_entropy,
    "ts_ssa_prior_reconstruction_error": _k_ts_ssa_prior_reconstruction_error,
    "HMA": _k_HMA,
    "ALMA": _k_ALMA,
    "ts_quantile_regression_coeff": _k_ts_quantile_regression_coeff,
    "ts_quantile_regression_resid": _k_ts_quantile_regression_resid,
    "ts_quantile_beta_spread_prior": _k_ts_quantile_beta_spread_prior,
    "group_multi_level_rank_consistency": _k_group_multi_level_rank_consistency,
    "state_flip_age": _k_state_flip_age,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch9)
# ---------------------------------------------------------------------------
class _R68NativeOperator(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, canonical: str, metadata, kernel: Callable[[dict], pl.DataFrame]):
        self._canonical = canonical
        self.metadata = metadata
        self._kernel_fn = kernel

    def calculate(self, *args, **kwargs):
        args, kwargs = self._prepare_call(args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return self._kernel_fn(bound)

    # Same binding contract under direct kernel-style invocation.
    _calculate_series = calculate


def register_r68_native_batch11() -> list[str]:
    from factor_engine.cleaned_operators import (
        record_backend_replacement_after, replace_backend,
    )
    from factor_engine.backend.polars_backend_kind import (
        PolarsImplementationKind, canonical_polars_kind,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    registered: list[str] = []
    source_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    for canonical, kernel in _KERNELS.items():
        ref = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        if ref is None:
            continue  # not a registry canonical in this environment
        current_meta = (
            ((OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {})
            .get("polars") or {}
        )
        if current_meta.get("source") == _SOURCE:
            continue
        current = OperatorRegistry.get(canonical, "polars", mode="any")
        if current is not None:
            cur_kind = getattr(getattr(current, "_physical_spec", None), "execution_kind", None)
            try:
                if cur_kind is None:
                    cur_kind = canonical_polars_kind(canonical)
            except Exception:
                cur_kind = None
            if cur_kind is PolarsImplementationKind.POLARS_NATIVE:
                continue  # first native registrant wins
        metadata = copy.deepcopy(ref.metadata)
        op = _R68NativeOperator(canonical, metadata, kernel)
        parameter_hash = hashlib.sha256(
            repr((metadata.param_names, getattr(metadata, "param_specs", None))).encode()
        ).hexdigest()
        op._physical_spec = PhysicalImplementationSpec(
            canonical=canonical, backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            supports_lazy=False, supports_streaming=False,
            materializes_full_panel=True,
            supports_nulls=True, supports_nan=True, supports_inf=True,
            implementation_source_hash=source_hash,
            emitter_identity=f"{_SOURCE}:pl.Expr/numpy:v1",
            kernel_identity=f"{_SOURCE}._KERNELS:{canonical}",
            parameter_domain_hash=parameter_hash,
            semantic_contract_hash=hashlib.sha256(
                (canonical + ":pandas-authority-parity:r68b11").encode()
            ).hexdigest(),
            notes=(
                "R68 batch11 genuine Polars backend: numpy authority kernels over "
                "pl->numpy columns; no pandas conversion, no pandas-delegate UDF."
            ),
        )
        migration = replace_backend(
            canonical, "polars",
            reason="R68 replace pandas-delegate UDF with genuine Polars implementation",
            source=_SOURCE,
        )
        OperatorRegistry.register(op, canonical=canonical, backend="polars", source=_SOURCE)
        record_backend_replacement_after(migration, canonical, "polars", source=_SOURCE)
        registered.append(canonical)
    return registered


__all__ = ["register_r68_native_batch11"]

register_r68_native_batch11()
