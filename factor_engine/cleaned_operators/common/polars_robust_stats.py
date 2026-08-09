# -*- coding: utf-8 -*-
"""Native Polars backends for robust-statistics, direction-concentration,
downside-risk, conditional-time-series and quadratic-fit operators.

These are per-column rolling state machines whose pandas references are already
NumPy loops.  Each column is evaluated with the equivalent NumPy kernel over the
Polars column array and wrapped into a ``pl.DataFrame``; no pandas DataFrame is
constructed on the fast path.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


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


def _arr(frame: pl.DataFrame, c: str) -> np.ndarray:
    return frame[c].to_numpy()


# ---------------------------------------------------------------------------
# robust rolling kernels
# ---------------------------------------------------------------------------


def _rolling_1d(x: np.ndarray, w: int, fn, mp: int = 1) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - w + 1)
        out[t] = fn(x[lo : t + 1], mp)
    return out


def _quantile_range_fn(chunk, mp, lo, hi):
    valid = chunk[np.isfinite(chunk)]
    if valid.size < mp:
        return np.nan
    return float(np.quantile(valid, hi) - np.quantile(valid, lo))


def ts_quantile_range(x, window, q_low=0.25, q_high=0.75, min_periods=1):
    w = _pi(window, "window")
    lo = _pf(q_low, "q_low")
    hi = _pf(q_high, "q_high")
    if not (0.0 < lo < hi < 1.0):
        raise ValueError("ts_quantile_range requires 0 < q_low < q_high < 1")
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_1d(_arr(x, c), w, lambda ch, m: _quantile_range_fn(ch, m, lo, hi), mp)
    return _make(x, cols, out)


def ts_robust_zscore(x, window, center="median", scale="mad", clip=None):
    w = _pi(window, "window")
    center_name = str(center or "median").lower()
    scale_name = str(scale or "mad").lower()
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    bound = float(clip) if clip is not None else None
    for i, c in enumerate(cols):
        arr = _arr(x, c)
        for t in range(rows):
            lo = max(0, t - w + 1)
            chunk = arr[lo : t + 1]
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                continue
            center_value = float(np.median(valid)) if center_name == "median" else float(np.mean(valid))
            if scale_name == "std":
                spread = float(np.std(valid))
            else:
                spread = float(np.median(np.abs(valid - center_value))) * 1.4826
            if not np.isfinite(spread) or spread <= 0.0:
                continue
            value = (float(chunk[-1]) - center_value) / spread if np.isfinite(chunk[-1]) else np.nan
            if bound is not None and np.isfinite(value):
                value = max(-bound, min(bound, value))
            out[t, i] = value
    return _make(x, cols, out)


def _trimmed_mean_fn(chunk, mp, trim):
    valid = chunk[np.isfinite(chunk)]
    if valid.size < mp:
        return np.nan
    ordered = np.sort(valid)
    cut = int(np.floor(trim * ordered.size))
    if cut * 2 >= ordered.size:
        return float(np.mean(ordered))
    return float(np.mean(ordered[cut : ordered.size - cut]))


def ts_trimmed_mean(x, window, trim_ratio=0.1, min_periods=1):
    w = _pi(window, "window")
    trim = _pf(trim_ratio, "trim_ratio")
    if not (0.0 <= trim < 0.5):
        raise ValueError("ts_trimmed_mean requires 0 <= trim_ratio < 0.5")
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_1d(_arr(x, c), w, lambda ch, m: _trimmed_mean_fn(ch, m, trim), mp)
    return _make(x, cols, out)


def _abs_concentration_fn(chunk, mp):
    abs_values = np.abs(chunk[np.isfinite(chunk)])
    if abs_values.size < mp:
        return np.nan
    total = float(abs_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.nan
    shares = abs_values / total
    return float(np.sum(shares * shares))


def ts_abs_concentration(x, window, min_periods=1):
    w = _pi(window, "window")
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_1d(_arr(x, c), w, _abs_concentration_fn, mp)
    return _make(x, cols, out)


def _abs_entropy_fn(chunk, mp, normalize):
    abs_values = np.abs(chunk[np.isfinite(chunk)])
    if abs_values.size < mp:
        return np.nan
    total = float(abs_values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.nan
    shares = abs_values / total
    entropy = float(-np.sum(shares * np.log(shares + 1e-300)))
    if normalize and shares.size > 1:
        entropy = entropy / np.log(shares.size)
    return entropy


def ts_abs_entropy(x, window, normalize=True, min_periods=1):
    w = _pi(window, "window")
    norm = bool(normalize)
    mp = max(1, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_1d(_arr(x, c), w, lambda ch, m: _abs_entropy_fn(ch, m, norm), mp)
    return _make(x, cols, out)


def _deviation_fn(chunk, mp, target, upside):
    valid = chunk[np.isfinite(chunk)]
    if valid.size < mp:
        return np.nan
    diff = valid - target
    selected = np.maximum(diff, 0.0) if upside else np.minimum(diff, 0.0)
    return float(np.sqrt(np.mean(selected * selected)))


def _deviation_op(x, window, target, min_periods, upside, name):
    w = _pi(window, "window")
    tgt = _pf(target, "target", None)
    mp = max(2, int(min_periods))
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _rolling_1d(_arr(x, c), w, lambda ch, m: _deviation_fn(ch, m, tgt, upside), mp)
    return _make(x, cols, out)


def ts_downside_deviation(x, window, target=0.0, min_periods=2):
    return _deviation_op(x, window, target, min_periods, False, "ts_downside_deviation")


def ts_upside_deviation(x, window, target=0.0, min_periods=2):
    return _deviation_op(x, window, target, min_periods, True, "ts_upside_deviation")


def ts_current_drawdown_duration(x, window, **kwargs):
    w = _pi(window, "window")
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = _arr(x, c)
        for t in range(rows):
            start = max(0, t - w + 1)
            chunk = arr[start : t + 1]
            valid_mask = np.isfinite(chunk)
            if not valid_mask.any():
                continue
            running_peak = np.maximum.accumulate(np.where(valid_mask, chunk, -np.inf))
            streak = 0
            for back in range(len(chunk) - 1, -1, -1):
                if not valid_mask[back]:
                    streak = 0
                    continue
                if chunk[back] < running_peak[back]:
                    streak += 1
                else:
                    break
            out[t, i] = float(streak)
    return _make(x, cols, out)


def ts_time_under_water(x, window, **kwargs):
    w = _pi(window, "window")
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = _arr(x, c)
        for t in range(rows):
            start = max(0, t - w + 1)
            chunk = arr[start : t + 1]
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                continue
            running_peak = np.maximum.accumulate(np.where(np.isnan(chunk), -np.inf, chunk))
            under = np.sum((chunk < running_peak) & np.isfinite(chunk))
            out[t, i] = float(under) / float(valid.size)
    return _make(x, cols, out)


def _best_lag_corr(xv, yv, row, window, max_lag):
    best = 0.0
    for lag in range(0, max_lag + 1):
        end = row + 1 - lag
        start = max(0, end - window)
        if end - start < 2:
            continue
        xs = xv[start:end]
        ys = yv[start + lag : row + 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 2:
            continue
        if np.std(xs[valid]) > 0 and np.std(ys[valid]) > 0:
            value = abs(float(np.corrcoef(xs[valid], ys[valid])[0, 1]))
            best = max(best, value)
    return best


def ts_best_lag_corr(y, x, window, max_lag=5):
    w = _pi(window, "window")
    ml = max(0, int(max_lag))
    cols = _cols(x, y)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, yv = _arr(x, c), _arr(y, c)
        for t in range(rows):
            value = _best_lag_corr(xv, yv, t, w, ml)
            out[t, i] = value if value > 0 else np.nan
    return _make(y, cols, out)


def _price_delay(xv, row, window, max_lag):
    end = row + 1
    start = max(0, end - window)
    if end - start < 3:
        return np.nan
    restricted = xv[start:end]
    valid = np.isfinite(restricted)
    if valid.sum() < 3:
        return np.nan
    y_r = restricted[valid]
    x_r = restricted[valid]
    rss_restricted = float(np.sum((y_r - np.mean(y_r)) ** 2))
    errors = []
    for lag in range(0, max_lag + 1):
        lagged_stop = max(0, end - lag)
        lagged_start = max(0, lagged_stop - window)
        if lagged_stop <= lagged_start:
            continue
        lagged_x = xv[lagged_start:lagged_stop]
        lagged_y = xv[lagged_start + lag : end]
        if lagged_x.size != lagged_y.size:
            continue
        aligned = np.isfinite(lagged_x) & np.isfinite(lagged_y)
        if aligned.sum() < 3:
            continue
        residuals = lagged_y[aligned] - np.mean(lagged_y[aligned])
        errors.append(float(np.sum(residuals ** 2)))
    if not errors:
        return np.nan
    rss_full = min(errors)
    if rss_restricted <= 0.0 or rss_full <= 0.0:
        return np.nan
    return max(0.0, 1.0 - rss_full / rss_restricted)


def _price_delay_model(stock, bench, end, window, max_lag, min_periods):
    lo = max(0, end - window + 1 - max_lag)
    if lo + max_lag >= end:
        return np.nan
    ts = np.arange(max(lo + max_lag, end - window + 1), end + 1)
    if ts.size < max_lag + 2:
        return np.nan
    y = stock[ts]
    bench_values = bench[ts]
    lagged = np.column_stack([bench[ts - lag] for lag in range(max_lag + 1)])
    X_full = np.column_stack([np.ones(len(ts)), lagged])
    X_restricted = np.column_stack([np.ones(len(ts)), bench_values])
    valid = np.isfinite(y) & np.all(np.isfinite(X_full), axis=1)
    required = max(int(min_periods), max_lag + 2)
    if valid.sum() < required:
        return np.nan
    yv = y[valid]
    xr = X_restricted[valid]
    xf = X_full[valid]
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    if sst <= 0.0:
        return np.nan
    beta_r, *_ = np.linalg.lstsq(xr, yv, rcond=None)
    rss_r = float(np.sum((yv - xr @ beta_r) ** 2))
    r2_r = 1.0 - rss_r / sst
    beta_f, *_ = np.linalg.lstsq(xf, yv, rcond=None)
    rss_f = float(np.sum((yv - xf @ beta_f) ** 2))
    r2_f = 1.0 - rss_f / sst
    if r2_f <= 0.0:
        return np.nan
    return max(0.0, 1.0 - r2_r / r2_f)


def ts_price_delay(stock_return, benchmark_return, window, max_lag=5, min_periods=3):
    w = _pi(window, "window")
    ml = max(1, int(max_lag))
    mp = max(3, int(min_periods))
    cols = _cols(stock_return, benchmark_return)
    rows = stock_return.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        sv = _arr(stock_return, c)
        bv = _arr(benchmark_return, c)
        for t in range(rows):
            out[t, i] = _price_delay_model(sv, bv, t, w, ml, mp)
    return _make(stock_return, cols, out)


# ---------------------------------------------------------------------------
# conditional time-series (mask-based rolling)
# ---------------------------------------------------------------------------


def _assert_condition_bool(condition_frame: pl.DataFrame) -> None:
    """R11 #144: the condition input must be a ConditionBool.

    Mirrors ``conditional_ext._assert_condition_bool``: finite values outside
    {0, 1} are neither a probability nor a boolean — reject the call instead of
    silently treating them as "truthy".
    """
    for c in condition_frame.columns:
        if c in _SKIP:
            continue
        cv = condition_frame[c].to_numpy()
        finite = np.isfinite(cv)
        bad = finite & (cv != 0.0) & (cv != 1.0)
        if np.any(bad):
            raise ValueError(
                f"condition must be a ConditionBool (values in {{0, 1}} with NaN "
                f"as missing); found {int(bad.sum())} finite value(s) outside "
                "{{0, 1}} in column " + str(c)
            )


def _selected_mask(condition_frame: pl.DataFrame, c: str, *value_frames) -> np.ndarray:
    cv = condition_frame[c].to_numpy()
    mask = np.isfinite(cv) & (cv == 1.0)
    for frame in value_frames:
        mask = mask & np.isfinite(frame[c].to_numpy())
    return mask


def _min_max_if(x, condition, window, min_periods, op):
    _assert_condition_bool(condition)
    w = _pi(window, "window")
    mp = max(1, int(min_periods))
    cols = _cols(x, condition)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = _arr(x, c)
        mask = _selected_mask(condition, c, x)
        for t in range(rows):
            start = max(0, t - w + 1)
            selected = xv[start : t + 1][mask[start : t + 1]]
            if selected.size < mp:
                continue
            out[t, i] = float(np.min(selected)) if op == "min" else float(np.max(selected))
    return _make(x, cols, out)


def ts_max_if(x, condition, window, min_periods=1):
    return _min_max_if(x, condition, window, min_periods, "max")


def ts_min_if(x, condition, window, min_periods=1):
    return _min_max_if(x, condition, window, min_periods, "min")


def ts_quantile_if(x, condition, window, q=0.5, min_periods=1):
    _assert_condition_bool(condition)
    w = _pi(window, "window")
    quantile = _pf(q, "q")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("ts_quantile_if requires 0 <= q <= 1")
    mp = max(1, int(min_periods))
    cols = _cols(x, condition)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv = _arr(x, c)
        mask = _selected_mask(condition, c, x)
        for t in range(rows):
            start = max(0, t - w + 1)
            selected = xv[start : t + 1][mask[start : t + 1]]
            if selected.size < mp:
                continue
            out[t, i] = float(np.quantile(selected, quantile))
    return _make(x, cols, out)


def _pair_condition(x, y, condition, window, min_periods, kind):
    _assert_condition_bool(condition)
    w = _pi(window, "window")
    mp = max(2 if kind != "resid" else 3, int(min_periods))
    if kind == "resid":
        mp = max(3, int(min_periods))
    cols = _cols(x, y, condition)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        xv, yv = _arr(x, c), _arr(y, c)
        mask = _selected_mask(condition, c, x)
        for t in range(rows):
            start = max(0, t - w + 1)
            xs = xv[start : t + 1][mask[start : t + 1]]
            ys = yv[start : t + 1][mask[start : t + 1]]
            if kind == "resid":
                # R11 #143: the current row is EXCLUDED from the fit, so
                # ``mp`` must constrain the TRAINING observations (condition-true
                # rows strictly before the current row) — never the total window
                # count including the current row.  Mirrors conditional_ext.
                x_fit = xv[start:t][mask[start:t]]
                y_fit = yv[start:t][mask[start:t]]
                x_cur = xv[t]
                y_cur = yv[t]
                if (
                    x_fit.size >= mp
                    and bool(mask[t])
                    and np.isfinite(x_cur)
                    and np.isfinite(y_cur)
                    and x_fit.size >= 2
                    and np.std(x_fit) > 0
                ):
                    coeffs = np.polyfit(x_fit, y_fit, 1)
                    out[t, i] = float(y_cur - np.polyval(coeffs, x_cur))
                continue
            if xs.size < mp:
                continue
            if kind == "corr":
                if np.std(xs) > 0 and np.std(ys) > 0:
                    out[t, i] = float(np.corrcoef(xs, ys)[0, 1])
            elif kind == "beta":
                var_x = float(np.var(xs))
                if var_x > 0 and np.isfinite(var_x):
                    cov = float(np.mean((xs - np.mean(xs)) * (ys - np.mean(ys))))
                    out[t, i] = cov / var_x
    return _make(y if kind in ("beta", "resid") else x, cols, out)


def ts_corr_if(x, y, condition, window, min_periods=2):
    return _pair_condition(x, y, condition, window, min_periods, "corr")


def ts_beta_if(y, x, condition, window, min_periods=2):
    return _pair_condition(x, y, condition, window, min_periods, "beta")


def ts_regression_resid_if(y, x, condition, window, min_periods=3):
    return _pair_condition(x, y, condition, window, min_periods, "resid")


# ---------------------------------------------------------------------------
# quadratic fit
# ---------------------------------------------------------------------------


def ts_poly2_coeff(x, d):
    d_i = _pi(d, "d")
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = _arr(x, c)
        t_idx = np.arange(d_i, dtype=float)
        for t in range(d_i - 1, rows):
            y = arr[t - d_i + 1 : t + 1]
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                continue
            coeffs = np.polyfit(t_idx[valid], y[valid], 2)
            out[t, i] = coeffs[0]
    return _make(x, cols, out)


def ts_poly2_resid(y, x, d):
    d_i = _pi(d, "d")
    cols = _cols(y, x)
    rows = y.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        yv, xv = _arr(y, c), _arr(x, c)
        for t in range(d_i - 1, rows):
            yw = yv[t - d_i + 1 : t + 1]
            xw = xv[t - d_i + 1 : t + 1]
            valid = ~(np.isnan(yw) | np.isnan(xw))
            if valid.sum() < 3:
                continue
            coeffs = np.polyfit(xw[valid], yw[valid], 2)
            a, b, c = coeffs
            fitted = a + b * xw + c * xw ** 2
            out[t, i] = float(yw[-1] - fitted[-1])
    return _make(y, cols, out)


def lqtp_historical_cvar(x, window, q=0.05):
    w = _pi(window, "window")
    qf = _pf(q, "q")
    if not 0.0 < qf <= 1.0:
        raise ValueError("historical_cvar q must be in (0,1]")
    cols = _cols(x)
    rows = x.height
    out = np.full((rows, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        arr = _arr(x, c)
        for t in range(rows):
            lo = max(0, t - w + 1)
            vals = arr[lo : t + 1]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            cutoff = float(np.quantile(vals, qf))
            tail = vals[vals <= cutoff]
            if tail.size:
                out[t, i] = float(-np.mean(tail))
    return _make(x, cols, out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    ("ts_quantile_range", ("x", "window", "q_low", "q_high", "min_periods"), ts_quantile_range, "Rolling quantile range."),
    ("ts_robust_zscore", ("x", "window", "center", "scale", "clip"), ts_robust_zscore, "Robust z-score with median/MAD center and scale."),
    ("ts_trimmed_mean", ("x", "window", "trim_ratio", "min_periods"), ts_trimmed_mean, "Rolling trimmed mean."),
    ("ts_abs_concentration", ("x", "window", "min_periods"), ts_abs_concentration, "Rolling absolute-value concentration."),
    ("ts_abs_entropy", ("x", "window", "normalize", "min_periods"), ts_abs_entropy, "Rolling absolute-value entropy."),
    ("ts_downside_deviation", ("x", "window", "target", "min_periods"), ts_downside_deviation, "Downside deviation."),
    ("ts_upside_deviation", ("x", "window", "target", "min_periods"), ts_upside_deviation, "Upside deviation."),
    ("ts_current_drawdown_duration", ("x", "window"), ts_current_drawdown_duration, "Bars since the running peak."),
    ("ts_time_under_water", ("x", "window"), ts_time_under_water, "Bars below the running peak in the window."),
    ("ts_best_lag_corr", ("y", "x", "window", "max_lag"), ts_best_lag_corr, "Best absolute lagged correlation."),
    ("ts_price_delay", ("stock_return", "benchmark_return", "window", "max_lag", "min_periods"), ts_price_delay, "Hou-Moskowitz style price-delay proxy."),
    ("ts_max_if", ("x", "condition", "window", "min_periods"), ts_max_if, "Rolling max where condition holds."),
    ("ts_min_if", ("x", "condition", "window", "min_periods"), ts_min_if, "Rolling min where condition holds."),
    ("ts_quantile_if", ("x", "condition", "window", "q", "min_periods"), ts_quantile_if, "Rolling quantile where condition holds."),
    ("ts_corr_if", ("x", "y", "condition", "window", "min_periods"), ts_corr_if, "Rolling correlation where condition holds."),
    ("ts_beta_if", ("y", "x", "condition", "window", "min_periods"), ts_beta_if, "Rolling beta where condition holds."),
    ("ts_regression_resid_if", ("y", "x", "condition", "window", "min_periods"), ts_regression_resid_if, "Rolling regression residual where condition holds."),
    ("ts_poly2_coeff", ("x", "d"), ts_poly2_coeff, "Quadratic coefficient of rolling time fit."),
    ("ts_poly2_resid", ("y", "x", "d"), ts_poly2_resid, "Quadratic-fit residual at window end."),
    ("lqtp_historical_cvar", ("x", "window", "q"), lqtp_historical_cvar, "Historical CVaR."),
)


def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
    metadata = OperatorMetadata(
        name=name,
        category="robust_statistics",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe", "causal", "polars", "native"],
    )

    def _calculate_series(self, *args, **kwargs):
        return function(*args, **kwargs)

    cls = type(
        f"PolarsRobustStats_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="robust_statistics",
        business_category="robust_statistics",
        canonical=name,
        source="polars_robust_stats",
        backend="polars",
        status="production",
    )(cls)


for _name, _params, _function, _description in _SPECS:
    _register(_name, _params, _function, _description)
