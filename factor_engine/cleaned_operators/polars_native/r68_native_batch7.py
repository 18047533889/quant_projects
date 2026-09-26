# -*- coding: utf-8 -*-
"""R68 batch7: genuine-Polars backends for the pandas-delegate canonicals.

Same protocol as ``r68_native_batch5/6``: module-level NumPy authority helpers
called directly on ``pl -> numpy`` column arrays; **no pandas DataFrame is
constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``, no
``iterrows``).  Daily-grain intraday kernels build their own daily panel
exactly like ``rolling_pack._pl_rebuild_intraday_result`` / batch5's
``_daily_out``.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch7"

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
            f"r68 native result shape {arr.shape} does not match panel {(base.height, len(cols))}"
        )
    return base.with_columns([
        pl.Series(c, np.ascontiguousarray(arr[:, j]), dtype=pl.Float64)
        for j, c in enumerate(cols)
    ])


def _time_numpy(frame: pl.DataFrame, session_tz: Any, *, convert_tz: bool = True) -> np.ndarray:
    tcol = None
    for c in ("__fe_time__", "date", "timestamp", "trade_date", "datetime"):
        if c in frame.columns:
            tcol = c
            break
    if tcol is None:
        return None
    s = frame[tcol]
    if convert_tz and isinstance(s.dtype, pl.Datetime) and s.dtype.time_zone is not None:
        from factor_engine.cleaned_operators.intraday._core import _SESSION_TZ
        tz = session_tz if session_tz is not None else _SESSION_TZ
        s = s.dt.convert_time_zone(str(tz)).dt.replace_time_zone(None)
    return s.to_numpy(allow_copy=True).astype("datetime64[ns]")


def _day_groups(times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    day = times.astype("datetime64[D]")
    days = np.unique(day)
    return day, [np.flatnonzero(day == d) for d in days]


def _daily_out(base: pl.DataFrame, times: np.ndarray, per_col: dict[str, list[float]]) -> pl.DataFrame:
    days = np.unique(times.astype("datetime64[D]"))
    day_ts = days.astype("datetime64[ns]")
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(per_col[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _daily_apply(
    frame: pl.DataFrame,
    session_tz: Any,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_finite: int | None = 2,
    catch_value_error: bool = False,
    convert_tz: bool = True,
) -> pl.DataFrame:
    """Per-(instrument, calendar-day) aggregation.

    ``min_finite=2`` mirrors ``intraday._core.daily_agg`` (P0-09);
    ``min_finite=None`` mirrors ``microstructure.intraday_agg._daily_agg_legacy``
    (only requires at least one finite value).  ``catch_value_error`` selects
    the legacy exception mapping (ValueError -> NaN).
    """
    from factor_engine.cleaned_operators.intraday._core import DataDegeneracy
    times = _time_numpy(frame, session_tz, convert_tz=convert_tz)
    xv = _panel(frame)
    day, row_groups = _day_groups(times)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(frame)):
        res: list[float] = []
        for idx in row_groups:
            v = xv[idx, j]
            t = times[idx].astype("datetime64[ns]")
            if min_finite is not None:
                if int(np.sum(np.isfinite(v))) < int(min_finite):
                    res.append(np.nan)
                    continue
            else:
                if not np.any(np.isfinite(v)):
                    res.append(np.nan)
                    continue
            excs: tuple[type[BaseException], ...] = (DataDegeneracy, ZeroDivisionError, OverflowError)
            if catch_value_error:
                excs = excs + (ValueError,)
            try:
                res.append(float(fn(v, t)))
            except excs:
                res.append(np.nan)
        per_col[c] = res
    return _daily_out(frame, times, per_col)


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


# ===========================================================================
# kernels
# ===========================================================================

# --- trivial / pure polars -------------------------------------------------
def _k_cs_coverage_ratio(b: dict) -> pl.DataFrame:
    arr = _panel(b["x"])
    count = np.sum(np.isfinite(arr), axis=1, keepdims=True)
    total = arr.shape[1]
    out = np.broadcast_to(count / float(total), arr.shape).astype(float)
    return _rebuild(b["x"], out)


def _k_index_weight_change(b: dict) -> pl.DataFrame:
    base = b["weight"]
    w = int(b.get("window", 20))
    cols = _ncols(base)
    return base.with_columns([pl.col(c) - pl.col(c).shift(w) for c in cols])


# --- rolling helpers reused from numpy authority modules --------------------
def _k_ts_interval_nesting_depth(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.interval_geometry import _nesting_depth_series
    mode = b.get("mode", "inside")
    if mode not in ("inside", "outside"):
        raise ValueError("mode must be 'inside' or 'outside'")
    out = _nesting_depth_series(_panel(b["low"]), _panel(b["high"]), mode)
    return _rebuild(b["low"], out)


def _k_ts_multiscale_permutation_entropy_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.state_geometry import (
        _effective_pattern_floor, _multiscale_feasible, _multiscale_slope_series,
    )
    window = b.get("window", 256)
    ord_ = int(b.get("order", 3))
    if not 2 <= ord_ <= 5:
        raise ValueError("ts_multiscale_permutation_entropy_slope requires order in [2, 5]")
    min_patterns = b.get("min_patterns", 5)
    if not _multiscale_feasible(window=window, order=ord_, min_patterns=min_patterns):
        raise ValueError(
            "ts_multiscale_permutation_entropy_slope window too small for "
            f"order: window={window}, order={ord_}, min_patterns={min_patterns} "
            f"(coarsest scale needs >= {_effective_pattern_floor(ord_, min_patterns)} "
            "ordinal patterns)"
        )
    xv = _panel(b["x"])
    out = np.empty_like(xv)
    for c in range(xv.shape[1]):
        out[:, c] = _multiscale_slope_series(xv[:, c], window, ord_, min_patterns)
    return _rebuild(b["x"], out)


def _k_intraday_volatility_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_vol_ext import _entropy, _rolling_returns
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 240))
    out = _rolling_returns(_panel(b["returns"]), w, _entropy)
    return _rebuild(b["returns"], out)


def _k_intraday_realized_semivariance_balance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_vol_ext import _rolling_returns, _semi_balance
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 240))
    out = _rolling_returns(_panel(b["returns"]), w, _semi_balance)
    return _rebuild(b["returns"], out)


def _k_ts_hodges_lehmann_location(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_scale import _vec_hodges_lehmann
    w = int(b.get("window", 60))
    if w < 2:
        raise ValueError("ts_hodges_lehmann_location requires window >= 2")
    mp = max(2, int(b.get("min_periods", 8)))
    xv = _panel(b["x"])
    out = np.full(xv.shape, np.nan, dtype=float)
    for c in range(xv.shape[1]):
        out[:, c] = _vec_hodges_lehmann(xv[:, c], w, mp)
    return _rebuild(b["x"], out)


def _rqa_out(b: dict, key: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rqa_ext import _rqa_series
    out = _rqa_series(
        _panel(b["x"]), b.get("window", 60), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), b.get("min_line", 4), b.get("min_periods", 10),
        key, b.get("theiler"),
    )
    return _rebuild(b["x"], out)


def _k_ts_recurrence_mean_diagonal_length(b: dict) -> pl.DataFrame:
    return _rqa_out(b, "mean_diagonal_length")


def _k_ts_recurrence_longest_vertical_length(b: dict) -> pl.DataFrame:
    return _rqa_out(b, "longest_vertical_length")


def _k_ts_autocorrelation_time_initial_positive_sequence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.memory_ext import (
        _autocorrelation_time_series_ips, _check_window,
    )
    w = _check_window(b.get("window", 120))
    ml = int(b.get("max_lag", 20))
    if ml < 1:
        raise ValueError("max_lag must be >= 1")
    if ml >= w:
        raise ValueError("max_lag must be < window")
    out = _autocorrelation_time_series_ips(_panel(b["x"]), w, ml)
    return _rebuild(b["x"], out)


def _k_ts_quantile_kurtosis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_tail import _r63_window_quantiles
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 60))
    outer = b.get("outer", (0.025, 0.975))
    inner = b.get("inner", (0.25, 0.75))
    try:
        o_lo, o_hi = float(outer[0]), float(outer[1])
        i_lo, i_hi = float(inner[0]), float(inner[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError("outer/inner must be 2-tuples of quantiles")
    if not (0.0 < o_lo < o_hi < 1.0 and 0.0 < i_lo < i_hi < 1.0):
        raise ValueError("outer/inner must be strictly ordered quantiles")
    if not (o_lo < i_lo < i_hi < o_hi):
        raise ValueError("outer quantiles must strictly enclose inner quantiles")
    mp = max(8, int(b.get("min_periods", 8)))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    Q, std, cnt = _r63_window_quantiles(xv, w, (i_hi, i_lo, o_hi, o_lo))
    inner_spread = Q[0] - Q[1]
    outer_spread = Q[2] - Q[3]
    ok = (cnt >= mp) & (std >= 1e-12) & np.isfinite(inner_spread) & (np.abs(inner_spread) >= 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = outer_spread / inner_spread
    return _rebuild(b["x"], np.where(ok, val, np.nan).reshape(rows, cols))


def _k_ts_dfa_hurst(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _dfa_hurst
    w = int(b.get("window", 250))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _dfa_hurst(xv[: r + 1, c], w, int(b.get("min_scale", 4)), int(b.get("max_scale", 32)))
    return _rebuild(b["x"], out)


def _k_ts_cusum_vol_break_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _cusum_vol_break
    w = int(b.get("window", 60))
    mp = int(b.get("min_periods", 10))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _cusum_vol_break(xv[: r + 1, c], w, mp)
    return _rebuild(b["x"], out)


def _k_ts_motif_recurrence_count(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.sequence_anomaly import _mp_stats
    xv = _panel(b["x"])
    m = b.get("m", 20)
    hist = b.get("history_window", 252)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _mp_stats(xv[: r + 1, c], m, "recurrence", hist)
    return _rebuild(b["x"], out)


def _k_ts_bicoherence_top_decile_excess(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.gemini_v2_common import trailing_contiguous_finite
    from factor_engine.cleaned_operators.research_spectral import _bicoherence_top_decile_excess
    if int(b.get("window", 120)) < 16:
        raise ValueError("ts_bicoherence_top_decile_excess requires window >= 16")
    w = int(b.get("window", 120))
    ns = int(b.get("n_segments", 4))
    nsurr = int(b.get("n_surrogates", 5))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = xv[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo: r + 1])
            if v.size < w:
                continue
            val = _bicoherence_top_decile_excess(v, ns, nsurr)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


def _k_ts_rolling_sr_gaussian_mean_shift_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.gemini_v2_common import trailing_contiguous_finite
    from factor_engine.cleaned_operators.research_spectral import _sr_gaussian
    if int(b.get("window", 120)) < 20:
        raise ValueError("ts_rolling_sr_gaussian_mean_shift_score requires window >= 20")
    side = b.get("side", "up")
    if side not in ("up", "down"):
        raise ValueError("side must be 'up' or 'down'")
    w = int(b.get("window", 120))
    sigma = float(b.get("shift_sigma", 1.0))
    bw = int(b.get("baseline_window", 40))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = xv[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo: r + 1])
            if v.size < bw + 8:
                continue
            val = _sr_gaussian(v, sigma, bw, side)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


def _k_ts_ks_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_distribution import map_two_window
    from factor_engine.cleaned_operators.rolling_pack import check_window, valid_values
    ws = check_window(b.get("recent_window", 20), name="recent_window")
    wl = check_window(b.get("old_window", 40), name="old_window")
    mp = max(3, int(b.get("min_periods", 5)))
    xv = _panel(b["x"])

    def _fn(recent: np.ndarray, old: np.ndarray) -> float:
        ra = np.sort(valid_values(recent))
        oa = np.sort(valid_values(old))
        if ra.size < mp or oa.size < mp:
            return np.nan
        combined = np.unique(np.concatenate([ra, oa]))
        ecdf_a = np.searchsorted(ra, combined, side="right") / ra.size
        ecdf_b = np.searchsorted(oa, combined, side="right") / oa.size
        return float(np.max(np.abs(ecdf_a - ecdf_b)))

    return _rebuild(b["x"], map_two_window(xv, ws, wl, _fn))


def _k_ts_chord_excursion_area(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_shape import _EPS
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 20))
    mp = max(2, int(b.get("min_periods", 2)))
    xv = _panel(b["x"])

    def _fn(chunk: np.ndarray) -> float:
        if chunk.size < w or not np.all(np.isfinite(chunk)):
            return np.nan
        v = chunk.astype(float)
        if v.size < mp:
            return np.nan
        j = np.arange(v.size, dtype=float)
        den = max(v.size - 1, 1)
        L = v[0] + (j / den) * (v[-1] - v[0])
        dev = v - L
        s_dev = float(np.sum(dev))
        s_abs = float(np.sum(np.abs(dev)))
        return s_dev / (s_abs + _EPS)

    return _rebuild(b["x"], map_rolling(xv, w, _fn))


def _k_ashare_limit_asymmetry(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    uv = _panel(b["up_event"])
    dv = _panel(b["down_event"])
    rows, cols = uv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        lo = max(0, r - w + 1)
        for c in range(cols):
            u_chunk = uv[lo: r + 1, c]
            d_chunk = dv[lo: r + 1, c]
            known = np.isfinite(u_chunk) | np.isfinite(d_chunk)
            known_count = int(known.sum())
            if known_count == 0:
                continue
            up_count = float(np.nansum(np.where(known & (u_chunk != 0), u_chunk, 0.0)))
            down_count = float(np.nansum(np.where(known & (d_chunk != 0), d_chunk, 0.0)))
            out[r, c] = (up_count - down_count) / known_count
    return _rebuild(b["up_event"], out)


def _k_ts_lag_of_peak_corr(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.base import strict_int_param
    w = strict_int_param(b.get("window", 20), "window", lower=3)
    ml = strict_int_param(b.get("max_lag", 5), "max_lag", lower=1)
    mp_raw = b.get("min_periods")
    mp = strict_int_param(mp_raw, "min_periods", lower=2) if mp_raw is not None else max(ml + 2, w // 2)
    xv = _panel(b["x"])
    yv = _panel(b["y"])
    if xv.shape != yv.shape:
        raise ValueError("ts_lag_of_peak_corr inputs must share identical shape")
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            lo = max(0, row - w + 1)
            best: tuple[float, int] | None = None
            for kk in range(ml + 1):
                s_lo = max(lo, kk)
                if s_lo > row:
                    continue
                xs = xv[s_lo: row + 1, col]
                ys = yv[s_lo - kk: row + 1 - kk, col]
                valid = np.isfinite(xs) & np.isfinite(ys)
                if int(valid.sum()) < mp:
                    continue
                a = xs[valid]
                bb = ys[valid]
                if a.size < 2 or np.std(a) <= 0.0 or np.std(bb) <= 0.0:
                    continue
                corr = float(np.corrcoef(a, bb)[0, 1])
                if best is None or abs(corr) > abs(best[0]):
                    best = (corr, kk)
            if best is not None:
                out[row, col] = best[1] / float(ml)
    return _rebuild(b["x"], out)


def _k_ts_hysteresis_age(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_state import (
        _DEFAULT_HYSTERESIS_MISSING_POLICY, HysteresisStateKernel,
    )
    hi = float(b.get("upper", 1.0))
    lo = float(b.get("lower", 0.0))
    if not (0.0 <= lo < hi):
        raise ValueError("require 0 <= lower < upper")
    cap_n = int(b.get("cap", 60))
    if cap_n < 1:
        raise ValueError("cap must be >= 1")
    missing_policy = b.get("missing_policy", _DEFAULT_HYSTERESIS_MISSING_POLICY)
    zv = _panel(b["z"])
    rows, cols = zv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        kernel = HysteresisStateKernel(hi, lo, missing_policy)
        for row in range(rows):
            val = zv[row, col]
            kernel.step(val, row)
            if not np.isfinite(val):
                continue
            st = kernel.current_state
            if st == 0:
                out[row, col] = 0.0
            else:
                out[row, col] = st * min(float(kernel.state_age), float(cap_n)) / float(cap_n)
    return _rebuild(b["z"], out)


def _k_ts_path_leadlag_area(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
    from factor_engine.cleaned_operators.ts_model.path_signature import _leadlag_area
    w = strict_int(b.get("window", 60), "window", lower=4)
    l = strict_int(b.get("lag", 1), "lag", lower=1)
    if l + 3 > w:
        raise ValueError("ts_path_leadlag_area requires lag + 3 <= window")
    xv = _panel(b["x"])
    yv = _panel(b["y"])
    if xv.shape != yv.shape:
        raise ValueError("path_signature inputs must share identical index and columns")
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _leadlag_area(xv[: row + 1, col], yv[: row + 1, col], w, l)
    return _rebuild(b["x"], out)


# --- GARCH family -----------------------------------------------------------
def _garch_out(b: dict, stat: str, asymmetric: bool) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import _GARCH_FIT_CACHE, _garch_path
    xv = _panel(b["x"])
    w = int(b.get("window", 120))
    _GARCH_FIT_CACHE.clear()
    try:
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            for r in range(rows):
                out[r, c] = _garch_path(xv[: r + 1, c], w, stat, asymmetric, 0.0)
    finally:
        _GARCH_FIT_CACHE.clear()
    return _rebuild(b["x"], out)


def _k_ts_garch_next_vol_forecast(b: dict) -> pl.DataFrame:
    return _garch_out(b, "forecast", False)


def _k_ts_garch_standardized_shock(b: dict) -> pl.DataFrame:
    return _garch_out(b, "shock", False)


def _k_ts_gjr_garch_vol_forecast(b: dict) -> pl.DataFrame:
    return _garch_out(b, "forecast", True)


# --- cross-section -----------------------------------------------------------
def _k_cs_residual_percentile(b: dict) -> pl.DataFrame:
    rv = _panel(b["resid"])
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        a = rv[row]
        valid = np.isfinite(a)
        if valid.sum() < 2:
            continue
        order = np.argsort(np.argsort(a[valid]))
        out[row, valid] = order / (valid.sum() - 1.0)
    return _rebuild(b["resid"], out)


def _rolling_pca_np(rv: np.ndarray, window: int, fn) -> np.ndarray:
    """numpy twin of panel_model._rolling_pca (fit_lag=1, warmup full)."""
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for row in range(rows):
        fit_end = row - 1
        if fit_end < 0:
            continue
        if fit_end < w - 1:
            continue
        start = max(0, fit_end - w + 1)
        X = rv[start: fit_end + 1]
        out[row] = fn(X, rv[row])
    return out


def _k_panel_rolling_pca_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_resid
    rv = _panel(b["ret"])
    out = _rolling_pca_np(rv, int(b.get("window", 120)),
                          lambda X, c: _pca_resid(X, c, int(b.get("n_components", 5))))
    return _rebuild(b["ret"], out)


def _k_panel_rolling_pca_resid_vol(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_resid
    rv = _panel(b["ret"])
    w = int(b.get("window", 120))
    resid = _rolling_pca_np(rv, w, lambda X, c: _pca_resid(X, c, int(b.get("n_components", 5))))
    rows, cols = resid.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = resid[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = col[lo: r + 1]
            fin = v[np.isfinite(v)]
            if fin.size < 10:
                continue
            out[r, c] = float(np.std(fin, ddof=1))
    return _rebuild(b["ret"], out)


# --- daily-grain intraday -----------------------------------------------------
def _k_intraday_realized_power_variation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import _realized_power_variation
    od = float(b.get("order", 4.0))
    sm = int(b.get("sampling", 1))
    if od <= 0.0:
        raise ValueError("intraday_realized_power_variation requires order > 0")
    if sm < 1:
        raise ValueError("intraday_realized_power_variation requires sampling >= 1")
    return _daily_apply(
        b["returns"], b.get("session_tz"),
        lambda v, t: _realized_power_variation(v, od, sm),
        min_finite=None, catch_value_error=True,
    )


def _k_intraday_subsampled_rv_dispersion(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import _subsampled_rv_dispersion
    sm = int(b.get("sampling", 5))
    if sm < 2:
        raise ValueError("intraday_subsampled_rv_dispersion requires sampling >= 2")
    return _daily_apply(
        b["returns"], b.get("session_tz"),
        lambda v, t: _subsampled_rv_dispersion(v, sm),
        min_finite=None, catch_value_error=True,
    )


def _intra_ute_out(b: dict, mode: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import minute_of_day
    from factor_engine.cleaned_operators.intraday.time_structure_v2 import (
        _EPS, _intraday_returns,
    )
    if mode == "high":
        edge = int(b.get("edge_minutes", 30))
        if edge < 1:
            raise ValueError("edge_minutes must be >= 1")

        def _fn(cv, times):
            r = _intraday_returns(cv, times)
            r2 = r * r
            total = float(np.nansum(r2))
            if total <= _EPS:
                return np.nan
            minutes = minute_of_day(times)
            finite_minutes = minutes[np.isfinite(r)]
            if finite_minutes.size < 4:
                return np.nan
            s_open, s_close = int(finite_minutes.min()), int(finite_minutes.max())
            eff_edge = min(edge, (s_close - s_open) // 2)
            if eff_edge < 1:
                return np.nan
            open_mask = (minutes >= s_open) & (minutes <= s_open + eff_edge)
            close_mask = (minutes >= s_close - eff_edge) & (minutes <= s_close)
            edge_rv = float(np.nansum(r2[open_mask])) + float(np.nansum(r2[close_mask]))
            return edge_rv / total

        return _daily_apply(b["close"], b.get("session_tz"), _fn, min_finite=2)
    ms, me = int(b.get("mid_start", 660)), int(b.get("mid_end", 810))
    if not 0 <= ms < me <= 1440:
        raise ValueError("require 0 <= mid_start < mid_end <= 1440")

    def _fn(cv, times):
        r = _intraday_returns(cv, times)
        r2 = r * r
        total = float(np.nansum(r2))
        if total <= _EPS:
            return np.nan
        minutes = minute_of_day(times)
        mid = (minutes >= ms) & (minutes <= me)
        return float(np.nansum(r2[mid])) / total

    return _daily_apply(b["close"], b.get("session_tz"), _fn, min_finite=2)


def _k_intra_ute_high(b: dict) -> pl.DataFrame:
    return _intra_ute_out(b, "high")


def _k_intra_ute_low(b: dict) -> pl.DataFrame:
    return _intra_ute_out(b, "low")


def _k_intra_bar_range_deviation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import minute_of_day
    high = b["high"]
    low = b["low"]
    times = _time_numpy(high, b.get("session_tz"))
    hv = _panel(high)
    lv = _panel(low)
    w = max(2, int(b.get("window", 20)))
    mp = max(2, w // 2)
    day = times.astype("datetime64[D]")
    days = np.unique(day)
    di = np.searchsorted(days, day)
    mods = minute_of_day(times.astype("datetime64[ns]"))
    slots = np.unique(mods)
    si = np.searchsorted(slots, mods)
    n_d, n_s = len(days), len(slots)
    flat = di * n_s + si
    cols = _ncols(high)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(cols):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((hv[:, j] > 0) & (lv[:, j] > 0)
                         & np.isfinite(hv[:, j]) & np.isfinite(lv[:, j]),
                         np.log(hv[:, j] / lv[:, j]), np.nan)
        fin = np.isfinite(r)
        cnt = np.bincount(flat, weights=fin.astype(float), minlength=n_d * n_s)
        ssum = np.bincount(flat, weights=np.where(fin, r, 0.0), minlength=n_d * n_s)
        cnt = cnt.reshape(n_d, n_s)
        ssum = ssum.reshape(n_d, n_s)
        with np.errstate(invalid="ignore"):
            mat = np.where(cnt > 0, ssum / np.where(cnt > 0, cnt, 1.0), np.nan)
        # trailing mean of the PAST w days per slot (causal, never today),
        # min_periods=mp on non-NaN entries (pandas shift(1).rolling(w).mean())
        csum = np.cumsum(np.where(np.isfinite(mat), mat, 0.0), axis=0)
        ccnt = np.cumsum(np.isfinite(mat).astype(float), axis=0)
        row_idx = np.arange(n_d)
        hi_idx = np.clip(row_idx, 0, n_d)          # exclusive end = i (today excluded)
        lo_idx = np.clip(row_idx - w, 0, n_d)      # inclusive start
        prev_hi = np.clip(hi_idx - 1, 0, n_d)
        prev_lo = np.clip(lo_idx - 1, 0, n_d)
        seg_cnt = ccnt[prev_hi, :] - np.where(lo_idx[:, None] > 0, ccnt[prev_lo, :], 0.0)
        seg_sum = csum[prev_hi, :] - np.where(lo_idx[:, None] > 0, csum[prev_lo, :], 0.0)
        seg_cnt = np.where(row_idx[:, None] > 0, seg_cnt, 0.0)
        seg_sum = np.where(row_idx[:, None] > 0, seg_sum, 0.0)
        seg_cnt = np.where(lo_idx[:, None] < hi_idx[:, None], seg_cnt, 0.0)
        with np.errstate(invalid="ignore"):
            hist = np.where(seg_cnt >= mp, seg_sum / np.where(seg_cnt > 0, seg_cnt, 1.0), np.nan)
        out_days: list[float] = []
        for i in range(n_d):
            a = mat[i]
            bb = hist[i]
            valid = np.isfinite(a) & np.isfinite(bb)
            if valid.sum() < 2:
                out_days.append(np.nan)
                continue
            va, vb = a[valid], bb[valid]
            out_days.append(float(np.mean(va - vb)))
        per_col[c] = out_days
    return _daily_out(high, times, per_col)


def _k_intraday_session_shape_novelty(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_session import (
        _minute_of_day, _resolve_official_grid, _shape_novelty_series,
    )
    x = b["x"]
    sid = b["session_id"]
    if isinstance(sid, pl.Series):
        sid = sid.to_frame()
    if not isinstance(sid, pl.DataFrame):
        raise TypeError("session_id must be a panel")
    # port of intraday_session._validate_session_ids (no pandas)
    sid_cols = _ncols(sid)
    for c in sid_cols:
        if sid.schema[c] == pl.Boolean:
            raise ValueError("session_id must be integral; got a boolean column")
    sid_arr = np.empty((sid.height, len(sid_cols)), dtype=float)
    for j, c in enumerate(sid_cols):
        sid_arr[:, j] = sid[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True)
    fin = sid_arr[np.isfinite(sid_arr)]
    if np.any(np.abs(fin - np.round(fin)) > 1e-9):
        raise ValueError("session_id must be integral (a non-integer float is not a valid SessionID)")
    index_ns = _time_numpy(x, None, convert_tz=False)
    dates = index_ns.astype("datetime64[D]").astype("int64")
    mods = _minute_of_day(index_ns)
    on_grid = index_ns.astype("datetime64[ns]").astype("int64") % 60_000_000_000 == 0
    expected, close_mod, slot_set = _resolve_official_grid(b.get("calendar"))
    arr = _shape_novelty_series(
        _panel(x), sid_arr, dates, mods,
        b.get("history_days", 20), b.get("min_history_sessions", 5),
        expected, close_mod, slot_set, on_grid,
    )
    return _rebuild(x, arr)


# --- fiscal (numpy port of FiscalEventView) -----------------------------------
def _period_objects(frame: pl.DataFrame) -> np.ndarray:
    cols = _ncols(frame)
    out = np.empty((frame.height, len(cols)), dtype=object)
    for j, c in enumerate(cols):
        out[:, j] = np.asarray(frame[c].to_list(), dtype=object)
    return out


def _fev_build(values2d: np.ndarray, period2d: np.ndarray, revision_policy: str):
    from factor_engine.cleaned_operators.fiscal_event_ops import _finite, _policy
    from factor_engine.cleaned_operators.fiscal_strict import period_ordinal
    policy = _policy(revision_policy)
    rows, cols = values2d.shape
    ordinals = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            o = period_ordinal(period2d[r, c])
            if o is not None:
                ordinals[r, c] = o
    state: list[dict[int, float]] = [dict() for _ in range(cols)]
    first_seen: list[set[int]] = [set() for _ in range(cols)]
    snapshots: list[list[dict[int, float]]] = []
    for r in range(rows):
        for c in range(cols):
            o = ordinals[r, c]
            v = values2d[r, c]
            if not np.isfinite(o) or not _finite(v):
                continue
            key = int(o)
            if policy == "latest_available" or key not in first_seen[c]:
                state[c][key] = float(v)
            first_seen[c].add(key)
        snapshots.append([dict(sorted(column.items())) for column in state])
    return snapshots, ordinals


def _fev_history(snapshots, ordinals, row, col, *, require_consecutive: bool):
    from factor_engine.cleaned_operators.fiscal_event_ops import _finite
    current = ordinals[row, col]
    if not np.isfinite(current):
        return []
    history = list(snapshots[row][col].items())
    if not history:
        return []
    history = [(int(k), float(v)) for k, v in history if k <= int(current) and _finite(v)]
    history.sort()
    if require_consecutive and history:
        contiguous: list[tuple[int, float]] = [history[-1]]
        for item in reversed(history[:-1]):
            if contiguous[0][0] - item[0] != 1:
                break
            contiguous.insert(0, item)
        history = contiguous
    return history


def _k_fiscal_asymmetric_elasticity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_event_ops import _positive
    cost2d = _panel(b["cost"])
    activity2d = _panel(b["activity"])
    period2d = _period_objects(b["period_id"])
    periods = _positive(b.get("periods", 12), "periods")
    minimum = _positive(b.get("min_obs_per_regime", 3), "min_obs_per_regime")
    mode = str(b.get("mode", "down_minus_up")).lower()
    if mode != "down_minus_up":
        raise ValueError("mode must be 'down_minus_up'")
    add_intercept = bool(b.get("add_intercept", True))
    require_consecutive = bool(b.get("require_consecutive", True))
    revision_policy = b.get("revision_policy", "latest_available")
    snapshots_c, ord_c = _fev_build(cost2d, period2d, revision_policy)
    snapshots_a, ord_a = _fev_build(activity2d, period2d, revision_policy)
    rows, cols = cost2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    def _finite4(v: float) -> bool:
        return np.isfinite(v)

    for row in range(rows):
        for col in range(cols):
            c_hist = dict(_fev_history(snapshots_c, ord_c, row, col,
                                       require_consecutive=require_consecutive))
            a_hist = dict(_fev_history(snapshots_a, ord_a, row, col,
                                       require_consecutive=require_consecutive))
            common = sorted(set(c_hist) & set(a_hist))[-periods:]
            changes = [
                (c_hist[o] - c_hist[p], a_hist[o] - a_hist[p])
                for p, o in zip(common, common[1:])
                if _finite4(c_hist[o]) and _finite4(c_hist[p])
                and _finite4(a_hist[o]) and _finite4(a_hist[p])
            ]
            up = [(dc, da) for dc, da in changes if da > 0]
            down = [(dc, da) for dc, da in changes if da < 0]
            if len(up) < minimum or len(down) < minimum:
                continue

            def slope(samples):
                yy = np.asarray([pair[0] for pair in samples])
                xx = np.asarray([pair[1] for pair in samples])
                design = np.column_stack([np.ones(len(xx)), xx]) if add_intercept else xx[:, None]
                return np.nan if np.linalg.matrix_rank(design) != design.shape[1] \
                    else float(np.linalg.lstsq(design, yy, rcond=None)[0][-1])

            beta_up, beta_down = slope(up), slope(down)
            if np.isfinite(beta_up) and np.isfinite(beta_down):
                out[row, col] = beta_down - beta_up
    return _rebuild(b["cost"], out)


def _k_fiscal_ar_resid_std(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_event_ops import _nonnegative, _positive
    x2d = _panel(b["x"])
    period2d = _period_objects(b["period_id"])
    periods = _positive(b.get("periods", 12), "periods")
    ar_lag = _positive(b.get("ar_lag", 1), "ar_lag")
    min_train = _positive(b.get("min_train", 6), "min_train")
    ddof = _nonnegative(b.get("ddof", 1), "ddof")
    if periods < ar_lag + min_train:
        raise ValueError("periods must be at least ar_lag + min_train")
    require_consecutive = bool(b.get("require_consecutive", True))
    revision_policy = b.get("revision_policy", "latest_available")
    snapshots, ordinals = _fev_build(x2d, period2d, revision_policy)
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        for col in range(cols):
            history = _fev_history(snapshots, ordinals, row, col,
                                   require_consecutive=require_consecutive)[-periods:]
            if len(history) < ar_lag + min_train + 1:
                continue
            values = np.asarray([value for _, value in history], dtype=float)
            train_y, train_x = values[ar_lag:-1], values[:-ar_lag - 1]
            valid = np.isfinite(train_y) & np.isfinite(train_x)
            if valid.sum() < min_train:
                continue
            design = np.column_stack([np.ones(valid.sum()), train_x[valid]])
            if np.linalg.matrix_rank(design) != design.shape[1] or valid.sum() - design.shape[1] <= ddof:
                continue
            coeff = np.linalg.lstsq(design, train_y[valid], rcond=None)[0]
            current_x, current_y = values[-ar_lag - 1], values[-1]
            if not (np.isfinite(current_x) and np.isfinite(current_y)):
                continue
            train_residual = train_y[valid] - design @ coeff
            current_residual = current_y - (coeff[0] + coeff[1] * current_x)
            residuals = np.append(train_residual, current_residual)
            if len(residuals) > ddof:
                out[row, col] = float(np.std(residuals, ddof=ddof))
    return _rebuild(b["x"], out)


# --- composition ---------------------------------------------------------------
def _k_composition_js_divergence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composition import _EPS, _check_zero_policy
    _check_zero_policy(b.get("zero_policy", "reject"))
    a_parts = [_panel(b[k]) for k in ("x1", "x2", "x3", "x4", "x5", "x6") if b.get(k) is not None]
    b_parts = [_panel(b[k]) for k in ("y1", "y2", "y3", "y4", "y5", "y6") if b.get(k) is not None]
    if len(a_parts) < 2 or len(b_parts) < 2 or len(a_parts) != len(b_parts):
        raise ValueError("composition_js_divergence requires equal-size compositions (2..6 parts each)")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(p) for p in (*a_parts, *b_parts)])
    valid = np.all(np.isfinite(logs), axis=0)
    na = len(a_parts)

    def _close(logs_: np.ndarray, i0: int, i1: int) -> np.ndarray:
        sub = logs_[i0:i1]
        w = np.exp(sub - sub.max(axis=0, keepdims=True))
        return w / w.sum(axis=0, keepdims=True)

    p = _close(logs, 0, na)
    q = _close(logs, na, na + len(b_parts))
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)), axis=0) + 0.5 * np.sum(
        q * (np.log(q + _EPS) - np.log(m + _EPS)), axis=0
    )
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = js[valid]
    return _rebuild(b["x1"], out)


# --- fundamental -----------------------------------------------------------------
def _growth_np(xv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """numpy twin of transforms_v2._walk_periods + accruals_scores._growth(p=1)."""
    from factor_engine.cleaned_operators.fundamental.accruals_scores import _pct_change
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        _default_require_parseable, _lag_value, _period_insert, _period_key,
    )
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    require_parseable = _default_require_parseable()
    for c in range(cols):
        order: list[object] = []
        visible: dict[object, float] = {}
        for i in range(rows):
            value = xv[i, c]
            key = _period_key(pv[i, c])
            if key is not None and np.isfinite(value):
                if key not in visible:
                    _period_insert(order, key, require_parseable=require_parseable)
                visible[key] = float(value)
            if key is None or key not in visible:
                continue
            try:
                out[i, c] = _pct_change(float(visible[key]), _lag_value(order, visible, key, 1))
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                out[i, c] = np.nan
    return out


def _k_fin_goodwill_risk_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fundamental.accruals_scores import _reject_ytd_growth
    goodwill = _panel(b["goodwill"])
    asset_imp = _panel(b["asset_impairment_loss"])
    credit_imp = _panel(b["credit_impairment_loss"])
    total_assets = _panel(b["total_assets"])
    pv = _period_objects(b["period_id"])
    flow_type = b.get("flow_type")
    _reject_ytd_growth("fin_growth", flow_type)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = np.where(total_assets == 0.0, np.nan, goodwill / np.where(total_assets == 0.0, 1.0, total_assets))
        imp_den = asset_imp + credit_imp
        imp = np.where(total_assets == 0.0, np.nan, imp_den / np.where(total_assets == 0.0, 1.0, total_assets))
    gw_growth = _growth_np(goodwill, pv)
    return _rebuild(b["goodwill"], intensity + gw_growth + imp)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ts_bicoherence_top_decile_excess": _k_ts_bicoherence_top_decile_excess,
    "cs_coverage_ratio": _k_cs_coverage_ratio,
    "ts_gjr_garch_vol_forecast": _k_ts_gjr_garch_vol_forecast,
    "ts_path_leadlag_area": _k_ts_path_leadlag_area,
    "ts_ks_shift": _k_ts_ks_shift,
    "ts_multiscale_permutation_entropy_slope": _k_ts_multiscale_permutation_entropy_slope,
    "ts_interval_nesting_depth": _k_ts_interval_nesting_depth,
    "composition_js_divergence": _k_composition_js_divergence,
    "index_weight_change": _k_index_weight_change,
    "ts_garch_next_vol_forecast": _k_ts_garch_next_vol_forecast,
    "cs_residual_percentile": _k_cs_residual_percentile,
    "intra_ute_high": _k_intra_ute_high,
    "intraday_session_shape_novelty": _k_intraday_session_shape_novelty,
    "fiscal_asymmetric_elasticity": _k_fiscal_asymmetric_elasticity,
    "panel_rolling_pca_resid": _k_panel_rolling_pca_resid,
    "panel_rolling_pca_resid_vol": _k_panel_rolling_pca_resid_vol,
    "intraday_realized_power_variation": _k_intraday_realized_power_variation,
    "intraday_volatility_entropy": _k_intraday_volatility_entropy,
    "ts_garch_standardized_shock": _k_ts_garch_standardized_shock,
    "ashare_limit_asymmetry": _k_ashare_limit_asymmetry,
    "ts_chord_excursion_area": _k_ts_chord_excursion_area,
    "ts_hodges_lehmann_location": _k_ts_hodges_lehmann_location,
    "fiscal_ar_resid_std": _k_fiscal_ar_resid_std,
    "ts_hysteresis_age": _k_ts_hysteresis_age,
    "intraday_subsampled_rv_dispersion": _k_intraday_subsampled_rv_dispersion,
    "intraday_realized_semivariance_balance": _k_intraday_realized_semivariance_balance,
    "ts_recurrence_mean_diagonal_length": _k_ts_recurrence_mean_diagonal_length,
    "ts_recurrence_longest_vertical_length": _k_ts_recurrence_longest_vertical_length,
    "ts_rolling_sr_gaussian_mean_shift_score": _k_ts_rolling_sr_gaussian_mean_shift_score,
    "intra_ute_low": _k_intra_ute_low,
    "ts_lag_of_peak_corr": _k_ts_lag_of_peak_corr,
    "ts_autocorrelation_time_initial_positive_sequence": _k_ts_autocorrelation_time_initial_positive_sequence,
    "ts_dfa_hurst": _k_ts_dfa_hurst,
    "ts_cusum_vol_break_score": _k_ts_cusum_vol_break_score,
    "ts_motif_recurrence_count": _k_ts_motif_recurrence_count,
    "fin_goodwill_risk_score": _k_fin_goodwill_risk_score,
    "ts_quantile_kurtosis": _k_ts_quantile_kurtosis,
    "intra_bar_range_deviation": _k_intra_bar_range_deviation,
}


# ---------------------------------------------------------------------------
# registration (R68 protocol, same as batch5/6)
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

    _calculate_series = calculate


def register_r68_native_batch7() -> list[str]:
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
            continue
        current_meta = (
            ((OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {})
            .get("polars") or {}
        )
        if current_meta.get("source") == _SOURCE:
            continue
        current = OperatorRegistry.get(canonical, "polars", mode="any")
        if current is not None:
            try:
                kind = canonical_polars_kind(canonical)
            except Exception:
                kind = None
            if kind is PolarsImplementationKind.POLARS_NATIVE:
                continue  # first registrant wins
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
                (canonical + ":pandas-authority-parity:r68b7").encode()
            ).hexdigest(),
            notes=(
                "R68 batch7 genuine Polars backend: numpy kernels over "
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


__all__ = ["register_r68_native_batch7"]

register_r68_native_batch7()
