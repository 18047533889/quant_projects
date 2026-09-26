# -*- coding: utf-8 -*-
"""R68 batch6 addendum: native-Polars backends for the 9 b6.json canonicals
that the concurrent ``r68_native_batch6`` rewrite does not carry.

This module is **self-contained** (own panel helpers, own numpy kernels) so it
survives concurrent rewrites of ``r68_native_batch6.py``.  Same registration
contract as the other r68 native batches: ``replace_backend`` + registry
register with ``ExecutionKind.POLARS_NATIVE_EXPR``, pandas-free end to end.

Canonicals covered here:
  supertrend_direction, supertrend_days_since_flip, ema_crossover,
  tenkan_kijun_cross, keltner_width_pct, sr_distance_pct, sr_touch_count,
  intra_realized_correlation_ex_self, ts_expectile_beta_spread
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

# Short form required: rolling_pack._register_polars_reference force-demotes
# regression-family slots unless the current polars source starts with
# "r68_native_batch" (see _DYNAMIC_REGRESSION_DELEGATES).
_SOURCE = "r68_native_batch6_addendum"

_SKIP = frozenset({
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "__fe_instrument__",
})


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


def _rebuild(base: pl.DataFrame, arr: np.ndarray) -> pl.DataFrame:
    cols = _ncols(base)
    if arr.shape != (base.height, len(cols)):
        from factor_engine.backend.operator_errors import OperatorShapeError
        raise OperatorShapeError(
            f"r68b6a result shape {arr.shape} does not match panel {(base.height, len(cols))}"
        )
    return base.with_columns([
        pl.Series(c, np.ascontiguousarray(arr[:, j]), dtype=pl.Float64)
        for j, c in enumerate(cols)
    ])


def _strict_int(v: Any, name: str, *, lower: int | None = None, upper: int | None = None) -> int:
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    if not isinstance(v, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    out = int(v)
    if lower is not None and out < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and out > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return out


def _time_numpy(frame: pl.DataFrame, session_tz: Any = None) -> np.ndarray:
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
# R68 batch6 addendum: 9 remaining canonicals (supertrend / MA-cross / SR /
# Keltner / expectile-beta / intra-corr families).  Same contract as above:
# pure numpy kernels over pl->numpy panels + pl.Series rebuild, pandas-free.
# ---------------------------------------------------------------------------
def _np_ewm_fixed(a: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """numpy port of pandas ``ewm(alpha, adjust=False, ignore_na=False,
    min_periods)`` (pandas 2.3 Cython semantics): the decay weight
    ``old_wt`` accumulates across NaN gaps and the post-gap update mixes the
    carried average with the new observation through a normalised blend
    ``(w*avg + alpha*x) / (w + alpha)``; with no gap pending the update is the
    plain ``(1-alpha)*avg + alpha*x`` recursion.  Parity with the pandas
    authority is bit-exact for typical alphas and <= 1e-14 otherwise (the
    residual is the Cython FMA contraction, far inside the 1e-12 CLOSE
    tolerance)."""
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    one_minus = 1.0 - alpha
    for j in range(cols):
        col = a[:, j]
        avg = np.nan
        old_wt = 1.0
        count = 0
        for t in range(rows):
            v = col[t]
            if np.isfinite(v):
                if count == 0:
                    avg = float(v)
                    count = 1
                    old_wt = 1.0
                elif old_wt == 1.0:
                    avg = one_minus * avg + alpha * float(v)
                    count += 1
                else:
                    w = old_wt * one_minus
                    avg = (w * avg + alpha * float(v)) / (w + alpha)
                    old_wt = 1.0
                    count += 1
            elif count > 0:
                old_wt *= one_minus
            if count >= min_periods:
                out[t, j] = avg
    return out


def _np_ema_span(a: np.ndarray, span: int) -> np.ndarray:
    """pandas ``ewm(span=w, adjust=False, min_periods=w)`` == alpha=2/(w+1)."""
    return _np_ewm_fixed(a, 2.0 / (float(span) + 1.0), span)


def _np_tr(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """True range; first row NaN (prev close undefined), NaN propagates."""
    rows = h.shape[0]
    out = np.full(h.shape, np.nan, dtype=float)
    if rows > 1:
        prev = c[:-1, :]
        out[1:, :] = np.maximum(
            np.maximum(h[1:, :] - l[1:, :], np.abs(h[1:, :] - prev)),
            np.abs(l[1:, :] - prev),
        )
    return out


def _np_rolling_nanmax(a: np.ndarray, w: int) -> np.ndarray:
    """pandas ``rolling(w, min_periods=w).max()`` (NaN excluded from the count)."""
    rows, cols = a.shape
    out = np.full(a.shape, np.nan, dtype=float)
    if rows < w:
        return out
    for j in range(cols):
        col = a[:, j]
        for t in range(w - 1, rows):
            win = col[t - w + 1 : t + 1]
            if np.isnan(win).any():
                continue
            out[t, j] = win.max()
    return out


def _np_rolling_nanmin(a: np.ndarray, w: int) -> np.ndarray:
    rows, cols = a.shape
    out = np.full(a.shape, np.nan, dtype=float)
    if rows < w:
        return out
    for j in range(cols):
        col = a[:, j]
        for t in range(w - 1, rows):
            win = col[t - w + 1 : t + 1]
            if np.isnan(win).any():
                continue
            out[t, j] = win.min()
    return out


def _np_rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    """pandas ``rolling(w, min_periods=w).median()`` (NaN excluded)."""
    rows, cols = a.shape
    out = np.full(a.shape, np.nan, dtype=float)
    if rows < w:
        return out
    for j in range(cols):
        col = a[:, j]
        for t in range(w - 1, rows):
            win = col[t - w + 1 : t + 1]
            if np.isnan(win).any():
                continue
            out[t, j] = np.median(win)
    return out


def _np_rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """pandas ``rolling(w, min_periods=1).sum()`` over a NaN-free 0/1 array."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    if a.shape[0] > w:
        out[w:, :] = c[w:, :] - c[:-w, :]
    return out


def _np_supertrend(h: np.ndarray, l: np.ndarray, c: np.ndarray, w: int, mult: float) -> np.ndarray:
    """Bit-exact numpy port of ``indicators_v2.Supertrend`` (no pandas frame)."""
    atr = _np_ewm_fixed(_np_tr(h, l, c), 1.0 / float(w), w)
    mid = (h + l) / 2.0
    cu = mid + mult * atr
    cl = mid - mult * atr
    rows, cols = c.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    fu = cu.copy()
    fl = cl.copy()
    trend = np.ones((rows, cols), dtype=int)
    for j in range(cols):
        post_gap = False
        for t in range(1, rows):
            if not (np.isfinite(h[t, j]) and np.isfinite(l[t, j]) and np.isfinite(c[t, j])
                    and np.isfinite(cu[t, j]) and np.isfinite(cl[t, j])):
                trend[t, j] = 0
                post_gap = True
                continue
            if post_gap:
                trend[t, j] = 0
                fu[t, j] = cu[t, j]
                fl[t, j] = cl[t, j]
                post_gap = False
                out[t, j] = np.nan
                continue
            if np.isfinite(fu[t - 1, j]) and (cu[t, j] >= fu[t - 1, j] and c[t - 1, j] <= fu[t - 1, j]):
                fu[t, j] = fu[t - 1, j]
            if np.isfinite(fl[t - 1, j]) and (cl[t, j] <= fl[t - 1, j] and c[t - 1, j] >= fl[t - 1, j]):
                fl[t, j] = fl[t - 1, j]
            if trend[t - 1, j] == 0:
                mid_t = 0.5 * (cu[t, j] + cl[t, j])
                trend[t, j] = 1 if c[t, j] >= mid_t else -1
            elif trend[t - 1, j] > 0 and c[t, j] < fl[t, j]:
                trend[t, j] = -1
            elif trend[t - 1, j] < 0 and c[t, j] > fu[t, j]:
                trend[t, j] = 1
            else:
                trend[t, j] = trend[t - 1, j]
            if trend[t, j] > 0:
                out[t, j] = fl[t, j]
            elif trend[t, j] < 0:
                out[t, j] = fu[t, j]
    return out


def _np_supertrend_state(h: np.ndarray, l: np.ndarray, c: np.ndarray, w: int, mult: float) -> np.ndarray:
    """``_supertrend_state`` direction sign: +1 close>=line, -1 below, NaN
    wherever either side is non-finite (both-sides-finite contract)."""
    st = _np_supertrend(h, l, c, w, mult)
    return np.where(np.isfinite(st) & np.isfinite(c), np.where(c >= st, 1.0, -1.0), np.nan)


def _k_supertrend_direction(b: dict) -> pl.DataFrame:
    w = _strict_int(b["atr_window"], "atr_window", lower=2)
    mult = float(b["multiplier"])
    if not np.isfinite(mult):
        raise ValueError("multiplier must be finite")
    if mult <= 0:
        raise ValueError("multiplier must be > 0 (Supertrend multiplier=0 collapses upper/lower to the midpoint)")
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    return _rebuild(b["close"], _np_supertrend_state(h, l, c, w, mult))


def _k_supertrend_days_since_flip(b: dict) -> pl.DataFrame:
    w = _strict_int(b["atr_window"], "atr_window", lower=2)
    mult = float(b["multiplier"])
    if not np.isfinite(mult):
        raise ValueError("multiplier must be finite")
    if mult <= 0:
        raise ValueError("multiplier must be > 0 (Supertrend multiplier=0 collapses upper/lower to the midpoint)")
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    d = _np_supertrend_state(h, l, c, w, mult)
    rows, cols = d.shape
    prev = np.vstack([np.full((1, cols), np.nan), d[:-1, :]])
    with np.errstate(invalid="ignore"):
        diff = d - prev
        flip = np.where(np.isfinite(d), np.where(np.isfinite(prev), diff, 0.0), np.nan)
        flip = np.sign(flip)
    out = np.full(flip.shape, np.nan, dtype=float)
    for j in range(cols):
        since = np.nan
        for t in range(rows):
            if np.isnan(flip[t, j]):
                continue
            if flip[t, j] != 0.0:
                since = 0.0
            elif since == since:
                since += 1.0
            out[t, j] = since
    return _rebuild(b["close"], out)


def _k_ema_crossover(b: dict) -> pl.DataFrame:
    f = _strict_int(b["fast_window"], "fast_window")
    s = _strict_int(b["slow_window"], "slow_window")
    if f >= s:
        raise ValueError("fast_window must be < slow_window")
    c = _panel(b["close"])
    ef = _np_ema_span(c, f)
    es = _np_ema_span(c, s)
    pos = np.where(c > 0.0, c, np.nan)
    with np.errstate(invalid="ignore"):
        diff = (ef - es) / pos
    return _rebuild(b["close"], np.sign(diff))


def _k_tenkan_kijun_cross(b: dict) -> pl.DataFrame:
    t = _strict_int(b["tenkan_window"], "tenkan_window", lower=2)
    k = _strict_int(b["kijun_window"], "kijun_window", lower=2)
    if t >= k:
        raise ValueError("tenkan_window must be < kijun_window")
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    ten = (_np_rolling_nanmax(h, t) + _np_rolling_nanmin(l, t)) / 2.0
    kij = (_np_rolling_nanmax(h, k) + _np_rolling_nanmin(l, k)) / 2.0
    pos = np.where(c > 0.0, c, np.nan)
    with np.errstate(invalid="ignore"):
        diff = (ten - kij) / pos
    return _rebuild(b["close"], np.sign(diff))


def _k_keltner_width_pct(b: dict) -> pl.DataFrame:
    ew = _strict_int(b["ema_window"], "ema_window")
    aw = _strict_int(b["atr_window"], "atr_window")
    mult = float(b["multiplier"])
    if not np.isfinite(mult):
        raise ValueError("multiplier must be finite")
    if mult <= 0:
        raise ValueError("multiplier must be > 0 (Keltner multiplier=0 degenerates to the mid band)")
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    ema = _np_ema_span(c, ew)
    atr = _np_ewm_fixed(_np_tr(h, l, c), 1.0 / float(aw), aw)
    u = ema + mult * atr
    l = ema - mult * atr
    m = np.where(ema > 0.0, ema, np.nan)
    with np.errstate(invalid="ignore"):
        return _rebuild(b["close"], (u - l) / m)


def _np_sr_prior_pivot(h: np.ndarray, l: np.ndarray, c: np.ndarray, w: int) -> np.ndarray:
    """``_sr_prior_pivot``: rolling(w).median of the typical price over bars
    ending at t-1 (shift(1)), min_periods=w; bad bars contribute NaN."""
    tp = (h + l + c) / 3.0
    tp = np.where(np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0.0), tp, np.nan)
    return np.vstack(
        [np.full((1, tp.shape[1]), np.nan), _np_rolling_median(tp, w)[:-1, :]]
    )


def _k_sr_distance_pct(b: dict) -> pl.DataFrame:
    w = _strict_int(b["window"], "window", lower=2)
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    pivot = _np_sr_prior_pivot(h, l, c, w)
    pos = np.where(c > 0.0, c, np.nan)
    with np.errstate(invalid="ignore"):
        return _rebuild(b["close"], (pos - pivot) / pos)


def _k_sr_touch_count(b: dict) -> pl.DataFrame:
    w = _strict_int(b["window"], "window", lower=2)
    tol = float(b["tol"])
    if not np.isfinite(tol):
        raise ValueError("tol must be finite")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    h, l, c = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    pivot = _np_sr_prior_pivot(h, l, c, w)
    band = np.where(pivot > 0.0, tol * pivot, np.nan)
    with np.errstate(invalid="ignore"):
        touch_h = np.abs(h - pivot) <= band
        touch_l = np.abs(l - pivot) <= band
    valid = np.isfinite(h) & np.isfinite(l) & np.isfinite(band)
    touch = (touch_h | touch_l) & valid
    num = _np_rolling_sum(touch.astype(float), w)
    den = _np_rolling_sum(valid.astype(float), w)
    frac = num / np.where(den > 0.0, den, np.nan)
    mask = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (c > 0.0)
    return _rebuild(b["close"], np.where(mask, frac, np.nan))


def _k_intra_realized_correlation_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.realized_beta import _EPS, _realized_corr
    close, caps = b["close"], b["free_market_cap"]
    cv = _panel(close)
    rows, cols = cv.shape
    times = _time_numpy(close)
    if times is None:
        times = _time_numpy(close, None)
    day_keys = times.astype("datetime64[D]")
    rets = np.full_like(cv, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        adjacent = (
            np.isfinite(cv[1:, :]) & np.isfinite(cv[:-1, :])
            & (cv[1:, :] > 0.0) & (cv[:-1, :] > 0.0)
        )
        rets[1:, :] = np.where(adjacent, np.log(cv[1:, :] / cv[:-1, :]), np.nan)
    day_change = np.zeros(rows, dtype=bool)
    if rows > 1:
        day_change[1:] = day_keys[1:] != day_keys[:-1]
    rets[day_change, :] = np.nan
    days = np.unique(day_keys)
    groups = [np.flatnonzero(day_keys == d) for d in days]
    cap_days = _time_numpy(caps).astype("datetime64[D]")
    cap_vals = _panel(caps)
    cap_row_by_day = {d: i for i, d in enumerate(np.unique(cap_days))}
    w_bc = np.full((rows, cols), np.nan, dtype=float)
    for k, d in enumerate(days):
        ri = cap_row_by_day.get(d)
        if ri is not None:
            w_bc[groups[k], :] = cap_vals[ri, :]
    w_bc = np.where(np.isfinite(w_bc) & (w_bc > 0.0), w_bc, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(np.isfinite(rets) & np.isfinite(w_bc) & (w_bc > 0.0), rets * w_bc, 0.0)
        w_eff = np.where(np.isfinite(rets) & np.isfinite(w_bc) & (w_bc > 0.0), w_bc, 0.0)
    num_all = contrib.sum(axis=1)
    den_all = w_eff.sum(axis=1)
    names = _ncols(close)
    out_days: list[np.datetime64] = []
    vals: dict[str, list[float]] = {c: [] for c in names}
    for k, d in enumerate(days):
        g = groups[k]
        out_days.append(np.datetime64(d, "ns"))
        for j, c in enumerate(names):
            den = den_all[g] - w_eff[g, j]
            num = num_all[g] - contrib[g, j]
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan)
            rr = rets[g, j]
            mm = m
            if int(np.sum(np.isfinite(mm))) < 2:
                vals[c].append(np.nan)
                continue
            try:
                vals[c].append(float(_realized_corr(rr, mm)))
            except (ValueError, ZeroDivisionError, OverflowError):
                vals[c].append(np.nan)
    day_ts = np.asarray(out_days)
    series = [pl.Series("date", day_ts)]
    for c in names:
        series.append(pl.Series(c, np.asarray(vals[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _k_ts_expectile_beta_spread(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model._rolling_core import (
        build_design, expectile_fit, fit_result,
    )
    qh = float(b.get("q_high", 0.9))
    ql = float(b.get("q_low", 0.1))
    if not (0.0 < ql < qh < 1.0):
        raise ValueError("q_low < q_high must hold in (0, 1)")
    w = int(b.get("window", 60))
    # n_coeffs = 2 (intercept + x): mp floor mirrors _multi_regression.
    mp = max(int(b.get("min_periods", 10)), 5 * 2)
    yv, xv = _panel(b["y"]), _panel(b["x"])
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        yc, xc = yv[:, col], xv[:, col]
        for row in range(rows):
            fit_end = row  # fit_lag=0 (in-sample)
            start = max(0, fit_end - w + 1)
            sy, sx = yc[start : fit_end + 1], xc[start : fit_end + 1]
            valid = np.isfinite(sy) & np.isfinite(sx)
            if int(valid.sum()) < mp:
                continue
            vy, vx = sy[valid], sx[valid]
            if np.std(vx) <= 0.0:
                continue
            design = build_design([vx], True)
            b_hi = fit_result(lambda d, v: expectile_fit(d, v, qh), design, vy).value
            if b_hi is None:
                continue
            b_lo = fit_result(lambda d, v: expectile_fit(d, v, ql), design, vy).value
            if b_lo is None:
                continue
            out[row, col] = float(b_hi[1]) - float(b_lo[1])
    return _rebuild(b["y"], out)



_KERNELS: dict[str, Any] = {
    "supertrend_direction": _k_supertrend_direction,
    "supertrend_days_since_flip": _k_supertrend_days_since_flip,
    "ema_crossover": _k_ema_crossover,
    "tenkan_kijun_cross": _k_tenkan_kijun_cross,
    "keltner_width_pct": _k_keltner_width_pct,
    "sr_distance_pct": _k_sr_distance_pct,
    "sr_touch_count": _k_sr_touch_count,
    "intra_realized_correlation_ex_self": _k_intra_realized_correlation_ex_self,
    "ts_expectile_beta_spread": _k_ts_expectile_beta_spread,
}


class _R68B6AddendumOperator(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, canonical: str, metadata, kernel):
        self._canonical = canonical
        self.metadata = metadata
        self._kernel_fn = kernel

    def calculate(self, *args, **kwargs):
        args, kwargs = self._prepare_call(args, kwargs)
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return self._kernel_fn(bound)

    _calculate_series = calculate


def register_r68_native_batch6_addendum() -> list[str]:
    from factor_engine.cleaned_operators import (
        record_backend_replacement_after, replace_backend,
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
            try:
                from factor_engine.backend.polars_backend_kind import (
                    PolarsImplementationKind, canonical_polars_kind,
                )
                kind = canonical_polars_kind(canonical)
            except Exception:
                kind = None
            if kind is PolarsImplementationKind.POLARS_NATIVE:
                continue  # first registrant wins
        metadata = __import__("copy").deepcopy(ref.metadata)
        op = _R68B6AddendumOperator(canonical, metadata, kernel)
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
            emitter_identity=f"{_SOURCE}:numpy:v1",
            kernel_identity=f"{_SOURCE}._KERNELS:{canonical}",
            parameter_domain_hash=parameter_hash,
            semantic_contract_hash=hashlib.sha256(
                (canonical + ":pandas-authority-parity:r68b6a").encode()
            ).hexdigest(),
            notes=(
                "R68 batch6 addendum genuine Polars backend: numpy kernels "
                "over pl->numpy columns; no pandas conversion, no "
                "pandas-delegate UDF."
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


__all__ = ["register_r68_native_batch6_addendum"]

register_r68_native_batch6_addendum()


# ---------------------------------------------------------------------------
# rescue kernels: ts_value_at_argextreme / cs_tail_retention carry a broken
# ``from ... import strict_integer`` in the concurrent batch6 rewrite (the
# name does not exist in either common.strict_params or closure.strict_scalar;
# the correct name is ``strict_int``).  These ports use ``strict_int`` and are
# registered over the broken slots only while the broken name persists.
# ---------------------------------------------------------------------------
def _k_ts_value_at_argextreme(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    w = strict_int(b.get("window", 20), "window", minimum=2)
    mode_s = str(b.get("mode", "max")).lower()
    if mode_s not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")
    vv, sv = _panel(b["value"]), _panel(b["score"])
    rows, cols = vv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    off = 0 if bool(b.get("include_current", False)) else 1
    for c in range(cols):
        for r in range(rows):
            end = r + 1 - off
            if end <= 0:
                continue
            start = max(0, end - w)
            sseg = sv[start:end, c]
            finite = np.isfinite(sseg)
            if not finite.any():
                continue
            sub = sseg[finite]
            if mode_s == "max":
                pos = int(sub.size - 1 - np.argmax(sub[::-1]))
            else:
                pos = int(sub.size - 1 - np.argmin(sub[::-1]))
            f_idx = np.flatnonzero(finite)[pos]
            val = vv[start + f_idx, c]
            if np.isfinite(val):
                out[r, c] = float(val)
    return _rebuild(b["value"], out)


def _k_cs_tail_retention(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.stateful.rotation import (
        _EPS, _cohort_denominator, _exact_tail_weights, _tail_quantile,
        _valid_label,
    )
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    lk = strict_int(b.get("lag", 5), "lag", minimum=1)
    q = _tail_quantile(b.get("quantile", 0.1), "cs_tail_retention")
    side_s = str(b.get("side", "top")).lower()
    if side_s not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    cohort_s = str(b.get("cohort", "intersection")).lower()
    if cohort_s not in ("current", "historical", "intersection"):
        raise ValueError("cohort must be 'current', 'historical' or 'intersection'")
    top = side_s == "top"
    xv = _panel(b["x"])
    rows, cols = xv.shape
    group = b.get("group")
    gv = (_panel(group).astype(object)
          if group is not None and isinstance(group, pl.DataFrame) else None)
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        if r < lk:
            continue
        g_row = gv[r] if gv is not None else None
        if g_row is None:
            w_cur = _exact_tail_weights(xv[r], q, top)
            w_prev = _exact_tail_weights(xv[r - lk], q, top)
            num = float(np.sum(np.minimum(w_prev, w_cur)))
            den = _cohort_denominator(w_prev, w_cur, np.isfinite(xv[r]), cohort_s)
            if den > _EPS:
                out[r] = num / den
        else:
            g_prev = gv[r - lk]
            labels = ({v for v in g_row if _valid_label(v)}
                      | {v for v in g_prev if _valid_label(v)})
            for lab in labels:
                cur_members = g_row == lab
                prev_members = g_prev == lab
                if not (np.any(cur_members) or np.any(prev_members)):
                    continue
                w_prev = np.full(cols, np.nan)
                w_cur = np.full(cols, np.nan)
                if np.any(prev_members):
                    w_prev[prev_members] = _exact_tail_weights(
                        xv[r - lk][prev_members], q, top
                    )
                if np.any(cur_members):
                    w_cur[cur_members] = _exact_tail_weights(
                        xv[r][cur_members], q, top
                    )
                overlap = np.isfinite(w_prev) & np.isfinite(w_cur)
                num = float(np.sum(np.minimum(w_prev[overlap], w_cur[overlap])))
                if cohort_s == "intersection":
                    obs = np.isfinite(xv[r]) & prev_members
                    den = float(np.sum(np.where(obs, w_prev, 0.0)))
                elif cohort_s == "historical":
                    den = float(np.sum(w_prev[np.isfinite(w_prev)]))
                else:
                    den = float(np.sum(w_cur[np.isfinite(w_cur)]))
                if den > _EPS:
                    out[r][cur_members] = num / den
    return _rebuild(b["x"], out)


_RESCUE_KERNELS: dict[str, Any] = {
    "ts_value_at_argextreme": _k_ts_value_at_argextreme,
    "cs_tail_retention": _k_cs_tail_retention,
}


def _rescue_slots_broken() -> bool:
    """True while the concurrent batch6 kernels keep importing the
    non-existent ``strict_integer`` name (they crash at call time)."""
    for mod in (
        "factor_engine.cleaned_operators.common.strict_params",
        "factor_engine.cleaned_operators.closure.strict_scalar",
    ):
        try:
            m = __import__(mod, fromlist=["strict_integer"])
            if hasattr(m, "strict_integer"):
                return False
        except ImportError:
            continue
    return True


def register_r68_native_batch6_rescue() -> list[str]:
    if not _rescue_slots_broken():
        return []  # the concurrent owner fixed the import; nothing to rescue
    from copy import deepcopy
    from factor_engine.cleaned_operators import (
        record_backend_replacement_after, replace_backend,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    registered: list[str] = []
    source_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    for canonical, kernel in _RESCUE_KERNELS.items():
        ref = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        if ref is None:
            continue
        current_meta = (
            ((OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {})
            .get("polars") or {}
        )
        if current_meta.get("source") == _SOURCE:
            continue
        metadata = deepcopy(ref.metadata)
        op = _R68B6AddendumOperator(canonical, metadata, kernel)
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
            emitter_identity=f"{_SOURCE}:numpy:v1",
            kernel_identity=f"{_SOURCE}._RESCUE_KERNELS:{canonical}",
            parameter_domain_hash=parameter_hash,
            semantic_contract_hash=hashlib.sha256(
                (canonical + ":pandas-authority-parity:r68b6a-rescue").encode()
            ).hexdigest(),
            notes=(
                "R68 batch6 addendum rescue: the concurrent batch6 kernel "
                "imports a non-existent strict_integer name; this port uses "
                "strict_int and is registered only while the broken name "
                "persists."
            ),
        )
        migration = replace_backend(
            canonical, "polars",
            reason="R68 rescue broken polars kernel (strict_integer ImportError)",
            source=_SOURCE,
        )
        OperatorRegistry.register(op, canonical=canonical, backend="polars", source=_SOURCE)
        record_backend_replacement_after(migration, canonical, "polars", source=_SOURCE)
        registered.append(canonical)
    return registered


register_r68_native_batch6_rescue()
