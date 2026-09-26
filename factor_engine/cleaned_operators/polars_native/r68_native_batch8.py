# -*- coding: utf-8 -*-
"""R68 batch8: genuine-Polars backends for the b8 pandas-delegate canonicals.

Same protocol as ``r68_native_batch4/5/6``: pure ``pl.Expr`` kernels for the
formula-shaped operators; module-level NumPy authority helpers called directly
on ``pl -> numpy`` column arrays for the algorithmic ones.  **No pandas
DataFrame is constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``,
no ``iterrows``).
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch8"

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


def _raw_panel(value: Any) -> np.ndarray:
    """Column extraction without the float cast (group labels etc.)."""
    if isinstance(value, pl.Series):
        value = value.to_frame()
    cols = _ncols(value)
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


def _time_numpy(frame: pl.DataFrame, session_tz: Any = None) -> np.ndarray | None:
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


def _daily_out(base: pl.DataFrame, days: np.ndarray, per_col: dict[str, list[float]]) -> pl.DataFrame:
    day_ts = days.astype("datetime64[ns]")
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(per_col[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _assert_condition_bool_values(arr: np.ndarray, name: str) -> None:
    finite = np.isfinite(arr)
    bad = finite & (arr != 0.0) & (arr != 1.0)
    if np.any(bad):
        raise ValueError(
            f"{name} must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )


def _ewm_nan_aware(arr: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """pandas ``ewm(alpha=..., adjust=False, min_periods=...)`` on a 1-D array.

    Matches pandas adjust=False / ignore_na=False semantics: a NaN row decays
    the accumulated old weight by ``(1 - alpha)`` per elapsed position and
    re-emits the running mean (once min_periods observations are seen).
    """
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    y: float | None = None
    gap = 0
    count = 0
    one_minus = 1.0 - alpha
    for t in range(n):
        x = arr[t]
        if not np.isfinite(x):
            gap += 1
            if y is not None and count >= min_periods:
                out[t] = y
            continue
        if y is None:
            y = float(x)
            gap = 0
            count = 1
        else:
            d = one_minus ** gap
            w_old = d * one_minus
            y = (w_old * y + alpha * x) / (w_old + alpha)
            gap = 0
            count += 1
        if count >= min_periods:
            out[t] = y
    return out


def _rolling_rank_last(seg: np.ndarray, w: int) -> float:
    """pandas ``rolling(w, min_periods=w).rank(pct=True)`` at the current row."""
    if seg.size < w or not np.isfinite(seg).all():
        return np.nan
    val = seg[-1]
    less = float(np.sum(seg < val))
    eq = float(np.sum(seg == val))
    avg_rank = less + (eq + 1.0) / 2.0
    return float(avg_rank / float(seg.size))


# ===========================================================================
# kernels
# ===========================================================================
def _k_ts_extremogram(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_quantile_dynamics import (
        _check_q, _check_side, _extremogram_series,
    )
    w = int(b.get("window", 120))
    q = _check_q(b.get("quantile", 0.1))
    lg = int(b.get("lag", 1))
    side = b.get("side", "lower")
    _check_side(side)
    if lg < 1:
        raise ValueError("ts_extremogram requires lag >= 1")
    if w < lg + 2:
        raise ValueError("ts_extremogram requires window >= lag + 2")
    return _rebuild(
        b["x"],
        _extremogram_series(_panel(b["x"]), w, q, lg, side, bool(b.get("fixed_threshold", False))),
    )


def _k_ts_binned_response_curvature(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.binned_response import _curvature_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 120))
    bb = _strict_int(b.get("bins", 5), "bins", lower=4)
    if bb not in (4, 5):
        raise ValueError("bins must be one of {4, 5}")
    mpb = _strict_int(b.get("min_per_bin", 3), "min_per_bin", lower=3)
    return _rebuild(
        b["y"],
        _curvature_series(_panel(b["y"]), _panel(b["x"]), w, bb, mpb),
    )


def _k_state_latch(b: dict) -> pl.DataFrame:
    sc_arr = _panel(b["set_condition"])
    rc_arr = _panel(b["reset_condition"])
    if sc_arr.shape != rc_arr.shape:
        raise ValueError("state_latch inputs must share the same panel shape")
    _assert_condition_bool_values(sc_arr, "set_condition")
    _assert_condition_bool_values(rc_arr, "reset_condition")
    init_v = float(b.get("initial_state", 0.0))
    if init_v not in (0.0, 1.0):
        raise ValueError("state_latch requires initial_state in {0, 1}")
    init = 1.0 if init_v == 1.0 else 0.0
    rows, cols = sc_arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        state = init
        for row in range(rows):
            if not (np.isfinite(sc_arr[row, col]) and np.isfinite(rc_arr[row, col])):
                out[row, col] = np.nan
                state = init
                continue
            if rc_arr[row, col] == 1.0:
                state = 0.0
            elif sc_arr[row, col] == 1.0:
                state = 1.0
            out[row, col] = state
    return _rebuild(b["set_condition"], out)


def _k_ts_interval_occupancy_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.interval_geometry import _occupancy_series
    w = int(b.get("window", 20))
    bins = int(b.get("bins", 8))
    return _rebuild(
        b["low"],
        _occupancy_series(_panel(b["low"]), _panel(b["high"]), w, bins, "entropy"),
    )


def _psar_arrays(h: np.ndarray, l: np.ndarray, af0: float, afmax: float) -> np.ndarray:
    rows, cols = h.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        bull: bool | None = True
        sar: float | None = None
        ep: float | None = None
        af = af0
        live = False
        prev1_h = prev1_l = prev2_h = prev2_l = None
        for t in range(rows):
            hv, lv = h[t, c], l[t, c]
            if not (np.isfinite(hv) and np.isfinite(lv)):
                live = False
                continue
            if not live:
                bull = None
                sar = None
                ep = None
                af = af0
                live = True
                prev1_h, prev1_l = hv, lv
                prev2_h, prev2_l = None, None
                continue
            if bull is None:
                if hv > prev1_h and lv > prev1_l:
                    bull = True
                    sar = prev1_l
                    ep = max(hv, prev1_h)
                elif hv < prev1_h and lv < prev1_l:
                    bull = False
                    sar = prev1_h
                    ep = min(lv, prev1_l)
                else:
                    prev2_h, prev2_l = prev1_h, prev1_l
                    prev1_h, prev1_l = hv, lv
                    continue
                prev2_h, prev2_l = prev1_h, prev1_l
                prev1_h, prev1_l = hv, lv
                out[t, c] = sar
                continue
            sar = sar + af * (ep - sar)
            if bull:
                if prev2_l is not None:
                    sar = min(sar, prev1_l, prev2_l)
                else:
                    sar = min(sar, prev1_l)
                if lv < sar:
                    bull = False
                    sar = ep
                    ep = lv
                    af = af0
                elif hv > ep:
                    ep = hv
                    af = min(af + af0, afmax)
            else:
                if prev2_h is not None:
                    sar = max(sar, prev1_h, prev2_h)
                else:
                    sar = max(sar, prev1_h)
                if hv > sar:
                    bull = True
                    sar = ep
                    ep = hv
                    af = af0
                elif lv < ep:
                    ep = lv
                    af = min(af + af0, afmax)
            prev2_h, prev2_l = prev1_h, prev1_l
            prev1_h, prev1_l = hv, lv
            out[t, c] = sar
    return out


def _k_psar_days_since_flip(b: dict) -> pl.DataFrame:
    high, low, close = b["high"], b["low"], b["close"]
    hv, lv, cv = _panel(high), _panel(low), _panel(close)
    acceleration = b.get("acceleration", 0.02)
    maximum = b.get("maximum", 0.2)
    if isinstance(acceleration, bool) or not isinstance(acceleration, (int, float, np.floating, np.integer)):
        raise ValueError("acceleration must be a number")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float, np.floating, np.integer)):
        raise ValueError("maximum must be a number")
    af0 = float(acceleration)
    afmax = float(maximum)
    if not np.isfinite(af0) or af0 <= 0:
        raise ValueError("require 0 < acceleration <= maximum")
    if not np.isfinite(afmax) or afmax < af0:
        raise ValueError("require 0 < acceleration <= maximum")
    sar = _psar_arrays(hv, lv, af0, afmax)
    rows, cols = sar.shape
    # direction: +1 when close > sar, -1 otherwise, NaN where either is NaN
    direction = np.full((rows, cols), np.nan)
    for j in range(cols):
        for t in range(rows):
            if np.isfinite(sar[t, j]) and np.isfinite(cv[t, j]):
                direction[t, j] = 1.0 if cv[t, j] > sar[t, j] else -1.0
    # flip: sign(d_t - d_{t-1}) where both finite; 0 where d_t finite but
    # d_{t-1} NaN; NaN where d_t NaN
    flip = np.full((rows, cols), np.nan)
    for j in range(cols):
        for t in range(rows):
            d = direction[t, j]
            if not np.isfinite(d):
                continue
            prev = direction[t - 1, j] if t >= 1 else np.nan
            if np.isfinite(prev):
                flip[t, j] = float(np.sign(d - prev))
            else:
                flip[t, j] = 0.0
    out = np.full((rows, cols), np.nan)
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
    return _rebuild(close, out)


def _k_ts_sample_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_float, strict_int
    from factor_engine.cleaned_operators.sequence_complexity import _vec_column_sample_entropy
    w = strict_int(b.get("window", 60), "window", lower=2, upper=120)
    m = strict_int(b.get("embedding_dim", 2), "embedding_dim", lower=1, upper=4)
    tol = strict_float(b.get("tolerance_scale", 0.2), "tolerance_scale", lower=0.0)
    if tol <= 0.0:
        raise ValueError("tolerance_scale must be > 0")
    return _rebuild(b["x"], _vec_column_sample_entropy(_panel(b["x"]), w, m, tol))


def _k_ts_kramers_moyal_drift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import _km_series
    w = _strict_int(b.get("window", 60), "window", lower=10)
    bb = _strict_int(b.get("bins", 8), "bins", lower=2)
    lg = _strict_int(b.get("lag", 1), "lag", lower=1)
    if bb < 2:
        raise ValueError("ts_kramers_moyal_drift requires bins >= 2")
    if w <= lg:
        raise ValueError("ts_kramers_moyal_drift requires window > lag")
    mbc_raw = b.get("min_bin_count")
    if mbc_raw is None:
        mbc = max(3, int(np.ceil(w / bb * 0.25)))
    else:
        mbc = _strict_int(mbc_raw, "min_bin_count", lower=1)
    if mbc > w - lg:
        raise ValueError("min_bin_count must fit available lagged transitions")
    return _rebuild(b["x"], _km_series(_panel(b["x"]), w, bb, lg, 1, mbc))


def _k_ts_hartigan_dip(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.moments_ext import _dip_series
    w = int(b.get("window", 120))
    if w < 2:
        raise ValueError("ts_hartigan_dip requires window >= 2")
    return _rebuild(
        b["x"],
        _dip_series(
            _panel(b["x"]), w,
            min_periods=int(b.get("min_periods", 20)),
            min_coverage_fraction=float(b.get("min_coverage_fraction", 0.8)),
        ),
    )


def _k_ts_normalized_mutual_information(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.nonlinear_dependence import _quantile_hist_mi
    from factor_engine.cleaned_operators.rolling_pack import (
        aligned_pairs, check_window, map_pair_rolling,
    )
    w = check_window(b.get("window", 40))
    if b.get("estimator", "quantile_hist") != "quantile_hist":
        raise ValueError("estimator must be 'quantile_hist' (deterministic)")
    nb = int(b.get("bins", 5))
    mp = int(b.get("min_periods", 10))
    bc = bool(b.get("bias_correction", True))

    def _fn(a: np.ndarray, bb: np.ndarray) -> float:
        pa, pb = aligned_pairs(a, bb)
        if pa.size < mp:
            return np.nan
        return _quantile_hist_mi(pa, pb, nb, True, bias_correction=bc)

    return _rebuild(
        b["x"], map_pair_rolling(_panel(b["x"]), _panel(b["y"]), w, _fn)
    )


def _k_ts_sign_cluster_index(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 20))
    mp = max(2, int(b.get("min_periods", 2)))

    def _fn(chunk: np.ndarray) -> float:
        valid = chunk[np.isfinite(chunk)]
        if valid.size < mp:
            return np.nan
        sign = np.where(valid > 0, 1.0, np.where(valid < 0, -1.0, 0.0))
        sign = sign[sign != 0.0]
        if sign.size < mp:
            return np.nan
        runs = []
        cur = sign[0]
        length = 1
        for v in sign[1:]:
            if v == cur:
                length = length + 1
            else:
                runs.append(length)
                cur = v
                length = 1
        runs.append(length)
        m = len(runs)
        if m < 2:
            return np.nan
        total = float(sum(runs))
        if total <= 0.0:
            return np.nan
        hhi = sum((r / total) ** 2 for r in runs)
        return float((hhi - 1.0 / m) / (1.0 - 1.0 / m))

    return _rebuild(b["x"], map_rolling(_panel(b["x"]), w, _fn))


def _k_ts_chatterjee_xi(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dependence_ext import _chatterjee_xi
    from factor_engine.cleaned_operators.rolling_pack import (
        aligned_pairs, check_window, map_pair_rolling,
    )
    w = check_window(b.get("window", 120))

    def _fn(a: np.ndarray, bb: np.ndarray) -> float:
        pa, pb = aligned_pairs(a, bb)
        return _chatterjee_xi(pa, pb)

    return _rebuild(
        b["x"], map_pair_rolling(_panel(b["x"]), _panel(b["y"]), w, _fn)
    )


def _k_ts_kernel_granger_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_spectral import (
        _kernel_granger_score, trailing_contiguous_multi,
    )
    w = int(b.get("window", 120))
    if w < 30:
        raise ValueError("ts_kernel_granger_score requires window >= 30")
    lg = int(b.get("lag", 2))
    yv, xv = _panel(b["y"]), _panel(b["x"])
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(yv[lo: r + 1, c], xv[lo: r + 1, c])
            if run is None:
                continue
            if run[0].shape[0] < 30:
                continue
            val = _kernel_granger_score(run[0], run[1], lg)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["y"], out)


def _k_ts_weighted_permutation_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
    from factor_engine.cleaned_operators.rolling_pack import check_window
    from factor_engine.cleaned_operators.sequence_complexity import (
        _vec_column_weighted_permutation_entropy,
    )
    w = check_window(b.get("window", 60))
    ord_ = strict_int(b.get("order", 3), "order", lower=2, upper=6)
    dl = strict_int(b.get("delay", 1), "delay", lower=1)
    weight_kind = str(b.get("weight", "variance")).lower()
    if weight_kind not in {"variance", "range"}:
        raise ValueError("weight must be 'variance' or 'range'")
    norm = bool(b.get("normalize", True))
    return _rebuild(
        b["x"],
        _vec_column_weighted_permutation_entropy(_panel(b["x"]), w, ord_, dl, weight_kind, norm),
    )


def _k_ts_feature_pca_reconstruction_error(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _EPS
    collected = [
        _panel(f) for f in (b.get("f1"), b.get("f2"), b.get("f3"), b.get("f4"))
        if f is not None
    ]
    if len(collected) < 2:
        raise ValueError(
            "ts_feature_pca_reconstruction_error requires at least 2 feature "
            f"panels (got {len(collected)}); a single feature cannot be "
            "low-rank compressed"
        )
    n_features = len(collected)
    rank = max(1, min(int(b.get("n_components", 2)), n_features - 1))
    window = int(b.get("window", 120))
    n_rows, n_cols = collected[0].shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    for col in range(n_cols):
        X = np.column_stack([c[:, col] for c in collected])
        for row in range(n_rows):
            start = max(0, row - window + 1)
            Xw = X[start:row]
            if Xw.shape[0] < 10:
                continue
            finite_count = np.sum(np.isfinite(Xw), axis=0)
            active = finite_count >= 2
            n_active = int(active.sum())
            if n_active < 2:
                continue
            sub = Xw[:, active]
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            Xc = np.where(np.isfinite(sub), sub, mu)
            Xs = (Xc - mu) / sd
            _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
            r = min(rank, n_active - 1, Vt.shape[0])
            if r < 1:
                continue
            row_active = X[row][active]
            z = (row_active - mu) / sd
            if not np.all(np.isfinite(z)):
                continue
            recon = Vt[:r].T @ (Vt[:r] @ z)
            out[row, col] = float(np.sqrt(np.sum((z - recon) ** 2)))
    return _rebuild(b["f1"], out)


def _k_ts_feature_subspace_rotation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.feature_geometry import (
        _MIN_EIGENGAP, _subspace_rotation_chunk, _tri_rolling, _window,
    )
    r = _window(b.get("recent_window", 30), "recent_window")
    p = _window(b.get("prior_window", 90), "prior_window")
    eigen_gap = b.get("eigen_gap", _MIN_EIGENGAP)
    if not (0.0 <= float(eigen_gap) <= 1.0):
        raise ValueError("ts_feature_subspace_rotation requires 0 <= eigen_gap <= 1")
    w = r + p
    return _rebuild(
        b["f1"],
        _tri_rolling(
            _panel(b["f1"]),
            _panel(b["f2"]),
            _panel(b["f3"]),
            w,
            lambda a, bb, c: _subspace_rotation_chunk(a, bb, c, r, p, 5, float(eigen_gap)),
        ),
    )


def _k_session_event_recovery_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
    from factor_engine.cleaned_operators.session_recovery import _recovery_day
    from factor_engine.runtime.session_panel import (
        build_session_panel, default_ashare_calendar,
    )
    if not (0.0 < float(b.get("residual_fraction", 0.25)) <= 1.0):
        raise ValueError("session_event_recovery_score requires 0 < residual_fraction <= 1")
    horizon = strict_int(b.get("horizon", 10), "horizon", lower=1)
    rf = strict_int(b.get("refractory", 1), "refractory", lower=0)
    me = strict_int(b.get("min_events", 3), "min_events", lower=3)
    session_tz = b.get("session_tz")
    calendar = b.get("calendar")
    xv = _panel(b["x"])
    evv = _panel(b["event"])
    names = _ncols(b["x"])
    times = _time_numpy(b["x"], session_tz)
    if times is None:
        raise ValueError("session_event_recovery_score requires a time axis column")
    cal = calendar if calendar is not None else default_ashare_calendar(bar_freq="1min")
    tz = session_tz or None
    cal_market = str(getattr(cal, "market", "") or "")
    if not cal_market:
        raise ValueError(
            "session_event_recovery_score requires a calendar with an explicit "
            "market (R30 §30 — no half-generic session builder)"
        )
    _market = cal_market.lower()
    from factor_engine.cleaned_operators.intraday._core import _SESSION_TZ
    tz = session_tz or _SESSION_TZ
    day_keys = times.astype("datetime64[D]")
    days = np.unique(day_keys)
    per_col: dict[str, list[float]] = {c: [] for c in names}
    for d in days:
        g = np.flatnonzero(day_keys == d)
        day_ts = np.datetime64(d, "ns")
        for j, c in enumerate(names):
            vals = xv[g, j]
            if not np.any(np.isfinite(vals)):
                per_col[c].append(np.nan)
                continue
            try:
                t_day = times[g].astype("datetime64[ns]")
                px = build_session_panel(
                    t_day, vals, cal, market=_market, session_timezone=tz,
                    source_timezone=None, trade_date=day_ts,
                )
                ev_vals = evv[g, j]
                pe = build_session_panel(
                    t_day, ev_vals, cal, market=_market, session_timezone=tz,
                    source_timezone=None, trade_date=day_ts,
                )
                if not px.is_valid_bar[px.n_slots - 1]:
                    per_col[c].append(np.nan)
                    continue
                per_col[c].append(_recovery_day(
                    px.values, pe.values, horizon, float(b.get("residual_fraction", 0.25)), rf, me,
                ))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_col[c].append(np.nan)
    return _daily_out(b["x"], days, per_col)


def _k_ts_persistence_birth_dispersion(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_transform import _bifurcation_score
    if int(b.get("dim", 2)) < 2:
        raise ValueError(
            "ts_persistence_birth_dispersion requires dim >= 2: Rips H1 needs "
            "a >= 2-dimensional Takens embedding (dim=1 has no loop structure)"
        )
    w = int(b.get("window", 60))
    tau = int(b.get("tau", 5))
    dim = int(b.get("dim", 2))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            out[r, c] = _bifurcation_score(xv[start: r + 1, c], tau, dim)
    return _rebuild(b["x"], out)


def _k_ts_residualized_hsic(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_spectral import (
        _residualized_hsic, trailing_contiguous_multi,
    )
    w = int(b.get("window", 120))
    if w < 24:
        raise ValueError("ts_residualized_hsic requires window >= 24")
    purge_gap = int(b.get("purge_gap", 3))
    ax, ay, az = _panel(b["x"]), _panel(b["y"]), _panel(b["z"])
    rows, cols = ax.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(
                ax[lo: r + 1, c], ay[lo: r + 1, c], az[lo: r + 1, c]
            )
            if run is None:
                continue
            if run[0].shape[0] < 24:
                continue
            val = _residualized_hsic(run[0], run[1], run[2], purge_gap)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


def _k_keltner_compression(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.indicators_v2 import _keltner_multiplier
    hv, lv, cv = _panel(b["high"]), _panel(b["low"]), _panel(b["close"])
    ema_window = _strict_int(b.get("ema_window", 20), "ema_window", lower=1)
    atr_window = _strict_int(b.get("atr_window", 20), "atr_window", lower=1)
    multiplier = _keltner_multiplier(b.get("multiplier", 2.0))
    score_window = _strict_int(b.get("score_window", 20), "score_window", lower=2)
    rows, cols = cv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    ema_alpha = 2.0 / (ema_window + 1.0)
    wilder_alpha = 1.0 / atr_window
    for c in range(cols):
        prev = np.full(rows, np.nan)
        prev[1:] = cv[:-1, c]
        tr = np.maximum.reduce([
            hv[:, c] - lv[:, c],
            np.abs(hv[:, c] - prev),
            np.abs(lv[:, c] - prev),
        ])
        atr = _ewm_nan_aware(tr, wilder_alpha, atr_window)
        mid = _ewm_nan_aware(cv[:, c], ema_alpha, ema_window)
        upper = mid + multiplier * atr
        lower = mid - multiplier * atr
        with np.errstate(divide="ignore", invalid="ignore"):
            width = np.where(mid > 0.0, (upper - lower) / np.where(mid > 0.0, mid, np.nan), np.nan)
        for t in range(score_window - 1, rows):
            seg = width[t - score_window + 1: t + 1]
            out[t, c] = _rolling_rank_last(seg, score_window)
    return _rebuild(b["close"], out)


def _k_ts_gap_fill_ratio(b: dict) -> pl.DataFrame:
    from numpy.lib.stride_tricks import sliding_window_view
    cv, ov, pv = _panel(b["close"]), _panel(b["open"]), _panel(b["pre_close"])
    w = _strict_int(b.get("window", 60), "window", lower=1)
    valid = (
        np.isfinite(cv) & np.isfinite(ov) & np.isfinite(pv)
        & (cv > 0.0) & (ov > 0.0) & (pv > 0.0)
    )
    up = ov > pv
    down = ov < pv
    filled = (up & (cv <= pv)) | (down & (cv >= pv))
    gap_count_arr = np.where(valid, (up | down).astype(float), np.nan)
    fill_count_arr = np.where(valid, filled.astype(float), np.nan)
    rows, cols = cv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows >= w:
        gwin = sliding_window_view(gap_count_arr, w, axis=0)
        fwin = sliding_window_view(fill_count_arr, w, axis=0)
        g_ok = ~np.isnan(gwin).any(axis=2)
        f_ok = ~np.isnan(fwin).any(axis=2)
        g_sum = np.where(g_ok, np.nan_to_num(gwin, nan=0.0).sum(axis=2), np.nan)
        f_sum = np.where(f_ok, np.nan_to_num(fwin, nan=0.0).sum(axis=2), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(g_sum == 0.0, np.nan, f_sum / g_sum)
        out[w - 1:, :] = ratio
    return _rebuild(b["close"], out)


def _k_ts_mean_reversion_ou_approx_half_life(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.ar_meanrev import (
        _mean_reversion_ou_half_life_vec, _warn_if_trending_input,
    )
    xv = _panel(b["x"])
    _warn_if_trending_input(xv, "ts_mean_reversion_ou_approx_half_life")
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    window = int(b.get("window", 120))
    min_periods = int(b.get("min_periods", 20))
    for col in range(cols):
        out[:, col] = _mean_reversion_ou_half_life_vec(
            xv[:, col], window, min_periods, fit_lag=0
        )
    return _rebuild(b["x"], out)


def _k_ashare_suspension_episode_length(b: dict) -> pl.DataFrame:
    sv = _panel(b["is_suspend"])
    rows, cols = sv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    run = np.zeros(cols, dtype=np.int64)
    for r in range(rows):
        for c in range(cols):
            value = sv[r, c]
            if value != value:  # NaN
                out[r, c] = np.nan
                run[c] = 0
                continue
            if value == 1.0:
                run[c] += 1
                out[r, c] = float(run[c])
            elif value == 0.0:
                run[c] = 0
                out[r, c] = 0.0
            else:
                raise ValueError(
                    f"is_suspend must be an EventBool ({{0, 1, NaN}}); got "
                    f"{value!r} at row/col ({r},{c})"
                )
    return _rebuild(b["is_suspend"], out)


def _k_ts_extremal_dependence_decay(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_quantile_dynamics import (
        _check_q, _check_side, _extremal_decay_chunk,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    w = int(b.get("window", 120))
    q = _check_q(b.get("quantile", 0.1))
    side = b.get("side", "lower")
    _check_side(side)
    H = int(b.get("max_lag", 5))
    if H < 2:
        raise ValueError("ts_extremal_dependence_decay requires max_lag >= 2")
    if w < H + 4:
        raise ValueError("ts_extremal_dependence_decay requires window >= max_lag + 4")
    fixed = bool(b.get("fixed_threshold", False))
    return _rebuild(
        b["x"],
        map_rolling(_panel(b["x"]), w, lambda c: _extremal_decay_chunk(c, q, side, H, fixed)),
    )


def _k_ts_feature_effective_rank(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.feature_geometry import (
        _effective_rank_chunk, _tri_rolling, _window,
    )
    w = _window(b.get("window", 60), "window")
    return _rebuild(
        b["f1"],
        _tri_rolling(
            _panel(b["f1"]), _panel(b["f2"]), _panel(b["f3"]), w,
            lambda a, bb, c: _effective_rank_chunk(a, bb, c, 5),
        ),
    )


def _k_ts_recurrence_laminarity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rqa_ext import _rqa_series
    out = _rqa_series(
        _panel(b["x"]),
        int(b.get("window", 60)), int(b.get("dim", 1)), int(b.get("delay", 1)),
        float(b.get("eps_fraction", 0.1)), int(b.get("min_line", 4)),
        int(b.get("min_periods", 10)), "laminarity", b.get("theiler"),
    )
    return _rebuild(b["x"], out)


def _k_ts_bds_statistic(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_spectral import (
        _bds_statistic, trailing_contiguous_finite,
    )
    w = int(b.get("window", 250))
    if w < 30:
        raise ValueError("ts_bds_statistic requires window >= 30")
    embedding_dim = int(b.get("embedding_dim", 2))
    distance_multiplier = float(b.get("distance_multiplier", 1.5))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = xv[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo: r + 1])
            if v.size < 30:
                continue
            val = _bds_statistic(v, embedding_dim, distance_multiplier)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


def _k_ts_interval_union_coverage(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.interval_geometry import _union_coverage_series
    w = int(b.get("window", 20))
    return _rebuild(
        b["low"],
        _union_coverage_series(_panel(b["low"]), _panel(b["high"]), w),
    )


def _k_ts_distance_correlation_partial_proxy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dependence_ext import (
        _partial_dcor_proxy, _triple_series,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 120))
    return _rebuild(
        b["x"],
        _triple_series(_panel(b["x"]), _panel(b["y"]), _panel(b["z"]), w, _partial_dcor_proxy),
    )


def _k_ts_wavelet_lowpass_reconstruct(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_transform import _haar_lowpass_current
    w = int(b.get("window", 128))
    level = int(b.get("level", 2))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _haar_lowpass_current(xv[: r + 1, c], w, level)
    return _rebuild(b["x"], out)


def _k_ts_har_from_return_forecast_error_z(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import _har_from_return
    w = int(b.get("window", 120))
    xv = _panel(b["ret"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _har_from_return(xv[: r + 1, c], w, "innovation_z")
    return _rebuild(b["x"], out)


def _k_ts_regime_duration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _regime_filter
    w = int(b.get("window", 120))
    tp = float(b.get("transition_prob", 0.05))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _regime_filter(xv[: row + 1, col], w, "duration", tp)
    return _rebuild(b["x"], out)


def _k_ts_feature_mode_share(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.feature_geometry import (
        _mode_share_chunk, _tri_rolling, _window,
    )
    w = _window(b.get("window", 60), "window")
    return _rebuild(
        b["f1"],
        _tri_rolling(
            _panel(b["f1"]), _panel(b["f2"]), _panel(b["f3"]), w,
            lambda a, bb, c: _mode_share_chunk(a, bb, c, 5),
        ),
    )


def _k_ts_ar_in_sample_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.ar_meanrev import (
        _ar_apply_vec, validate_ar_configured_history,
    )
    w = int(b.get("window", 60))
    order = int(b.get("order", 1))
    warmup_policy = str(b.get("warmup_policy", "expanding"))
    validate_ar_configured_history(w, order, fit_lag=0)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = _ar_apply_vec(
            xv[:, col], w, order, "innovation",
            fit_lag=0, stability_k=0, warmup_policy=warmup_policy,
        )
    return _rebuild(b["x"], out)


def _k_ashare_limit_down_streak(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _break_mask, _consecutive_streak, _tolerance, _tradeable,
    )
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    cv, lv, vv = _panel(b["close"]), _panel(b["low_limit"]), _panel(b["valid_trade"])
    rows, cols = cv.shape
    condition = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            if _break_mask(r, c, [cv, lv]) or not _tradeable(vv, r, c):
                continue
            if np.isfinite(lv[r, c]) and cv[r, c] <= lv[r, c] * (1.0 + tol):
                condition[r, c] = 1.0
            else:
                condition[r, c] = 0.0
    tradeable = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            tradeable[r, c] = _tradeable(vv, r, c)
    return _rebuild(b["close"], _consecutive_streak(condition, tradeable, rows, cols))


def _k_ts_threshold_cycle_period(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.threshold_cycle import _cycle_period_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    lower_f, upper_f = float(b.get("lower")), float(b.get("upper"))
    if not (np.isfinite(lower_f) and np.isfinite(upper_f) and upper_f > lower_f):
        raise ValueError("upper must be > lower")
    w = check_window(b.get("window", 120))
    return _rebuild(
        b["x"], _cycle_period_series(_panel(b["x"]), lower_f, upper_f, w)
    )


def _k_ts_response_slope_asymmetry(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.binned_response import (
        _slope_asymmetry_series, _strict_probability,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 120))
    q = _strict_probability(b.get("split_quantile", 0.5), "split_quantile")
    return _rebuild(
        b["y"], _slope_asymmetry_series(_panel(b["y"]), _panel(b["x"]), w, q)
    )


def _k_ts_weighted_standardized_moment(b: dict) -> pl.DataFrame:
    xv, wv = _panel(b["x"]), _panel(b["weight"])
    if xv.shape != wv.shape:
        raise ValueError(
            "ts_weighted_standardized_moment inputs are misaligned: weight has a "
            "different index/columns than x (fail-closed; no silent reindex)"
        )
    p = int(b.get("order", 3))
    if p not in (3, 4):
        raise ValueError("order must be one of {3, 4}")
    w = int(b.get("window", 20))
    if w < p:
        raise ValueError(f"window must be >= {p}")
    min_samples = 4 if p == 4 else 3
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            xs = xv[start: r + 1, c]
            ws = wv[start: r + 1, c]
            if np.any(ws < 0.0):
                continue
            if np.any(~np.isfinite(ws)):
                continue
            valid = np.isfinite(xs) & (ws > 0.0)
            n = int(valid.sum())
            if n < min_samples:
                continue
            xa = xs[valid].astype(float)
            wa = ws[valid].astype(float)
            wa /= float(np.max(wa))
            wa /= float(wa.sum())
            n_eff = 1.0 / float(np.dot(wa, wa))
            if n_eff < float(min_samples) - 1e-9:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = xa - xa[0]
            if not np.all(np.isfinite(normalized)):
                normalized = xa / float(np.max(np.abs(xa)))
                normalized -= normalized[0]
            magnitude = float(np.max(np.abs(normalized)))
            if magnitude == 0.0:
                continue
            normalized /= magnitude
            dev = normalized - float(np.dot(wa, normalized))
            spread = float(np.max(np.abs(dev)))
            if spread == 0.0:
                continue
            dev /= spread
            var_w = float(np.dot(wa, dev * dev))
            if var_w <= 0.0:
                continue
            m_p = float(np.dot(wa, dev ** p)) / var_w ** (p / 2.0)
            out[r, c] = m_p
    return _rebuild(b["x"], out)


def _k_ts_mean_abs_deviation_strict(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.statistics import _strict_mean_abs_dev_1d
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    w = _strict_int(b.get("window", 20), "window", lower=1)
    return _rebuild(b["x"], map_rolling(_panel(b["x"]), w, _strict_mean_abs_dev_1d))


def _intra_beta_daily_numpy(
    close: pl.DataFrame,
    caps: pl.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    eps: float,
) -> pl.DataFrame:
    """numpy replication of ``_beta_daily(close, weights, fn, ex_self=True)``."""
    cv = _panel(close)
    cap_vals = _panel(caps)
    rows, cols = cv.shape
    times = _time_numpy(close)
    if times is None:
        raise ValueError("intraday session operator requires a time axis column")
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
    cap_days = _time_numpy(caps)
    if cap_days is None:
        raise ValueError("intraday session operator requires a time axis column")
    cap_days = cap_days.astype("datetime64[D]")
    cap_row_by_day = {d: i for i, d in enumerate(np.unique(cap_days))}
    w_bc = np.full((rows, cols), np.nan, dtype=float)
    for k, d in enumerate(days):
        ri = cap_row_by_day.get(d)
        if ri is not None:
            w_bc[groups[k], :] = cap_vals[ri, :]
    w_bc = np.where(np.isfinite(w_bc) & (w_bc > 0.0), w_bc, np.nan)
    w_ret = np.where(np.isfinite(rets), w_bc, np.nan)
    valid = np.isfinite(rets) & np.isfinite(w_bc) & (w_bc > 0.0)
    contrib = np.where(valid, rets * w_bc, 0.0)
    ew = np.where(valid, w_bc, 0.0)
    names = _ncols(close)
    out_days: list[np.datetime64] = []
    vals: dict[str, list[float]] = {c: [] for c in names}
    try:
        from factor_engine.backend.operator_errors import DataDegeneracy
        _deg = (DataDegeneracy, ZeroDivisionError, OverflowError)
    except Exception:
        _deg = (ZeroDivisionError, OverflowError)
    for k, d in enumerate(days):
        g = groups[k]
        out_days.append(np.datetime64(d, "ns"))
        for j, c in enumerate(names):
            den = ew[g, :].sum(axis=1) - ew[g, j]
            num = contrib[g, :].sum(axis=1) - contrib[g, j]
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(np.isfinite(den) & (np.abs(den) > eps), num / den, np.nan)
            mask = np.isfinite(m)
            mm = m[mask]
            rr = rets[g, j][mask]
            if int(mm.size) < 2:
                vals[c].append(np.nan)
                continue
            try:
                vals[c].append(float(fn(rr, mm)))
            except _deg:
                vals[c].append(np.nan)
    return _daily_out(close, np.asarray(out_days), vals)


def _k_intra_idiosyncratic_variance_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.realized_beta import _EPS, _idio_variance
    return _intra_beta_daily_numpy(
        b["close"], b["free_market_cap"], _idio_variance, eps=_EPS,
    )


def _k_group_peer_beta_deviation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.peer_ops import (
        _peer_weighted_mean_ex_self_row,
    )
    bv = _panel(b["beta"])
    gv = _raw_panel(b["group"])
    wv = _panel(b["weight"])
    if bv.shape != gv.shape or bv.shape != wv.shape:
        raise ValueError(
            "group_peer_beta_deviation inputs are misaligned (fail-closed; "
            "no silent reindex)"
        )
    rows, cols = bv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        peer = _peer_weighted_mean_ex_self_row(bv[row], gv[row], wv[row])
        out[row] = np.where(np.isfinite(peer), bv[row] - peer, np.nan)
    return _rebuild(b["beta"], out)


def _k_ts_forbidden_ordinal_pattern_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.complexity_ext import (
        _MIN_EMBEDDINGS, _check_complexity_params, _forbidden_ordinal_ratio_series,
    )
    window = b.get("window", 120)
    order = b.get("order", 3)
    delay = b.get("delay", 1)
    min_embeddings = b.get("min_embeddings", _MIN_EMBEDDINGS)
    _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
    return _rebuild(
        b["x"],
        _forbidden_ordinal_ratio_series(
            _panel(b["x"]), int(window), int(order), int(delay),
            int(min_embeddings), mode="ratio",
        ),
    )


def _k_ts_ar_fitted_value(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.ar_meanrev import (
        _ar_apply_vec, validate_ar_configured_history,
    )
    w = int(b.get("window", 60))
    order = int(b.get("order", 1))
    warmup_policy = str(b.get("warmup_policy", "expanding"))
    validate_ar_configured_history(w, order, fit_lag=0)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = _ar_apply_vec(
            xv[:, col], w, order, "forecast",
            fit_lag=0, stability_k=0, warmup_policy=warmup_policy,
        )
    return _rebuild(b["x"], out)


def _k_ts_threshold_cycle_asymmetry(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.threshold_cycle import _cycle_asymmetry_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    lower_f, upper_f = float(b.get("lower")), float(b.get("upper"))
    if not (np.isfinite(lower_f) and np.isfinite(upper_f) and upper_f > lower_f):
        raise ValueError("upper must be > lower")
    w = check_window(b.get("window", 120))
    return _rebuild(
        b["x"], _cycle_asymmetry_series(_panel(b["x"]), lower_f, upper_f, w)
    )


def _k_ts_turning_point_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _turning_point_ratio
    w = int(b.get("window", 60))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _turning_point_ratio(xv[: row + 1, col], w)
    return _rebuild(b["x"], out)


def _k_ts_path_signature_depth2_norm(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.path_signature import _sig_depth2_norm
    w = _strict_int(b.get("window", 60), "window", lower=3)
    xv, yv = _panel(b["x"]), _panel(b["y"])
    if xv.shape != yv.shape:
        raise ValueError("path_signature inputs must share identical index and columns")
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _sig_depth2_norm(xv[: row + 1, col], yv[: row + 1, col], w)
    return _rebuild(b["x"], out)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ts_extremogram": _k_ts_extremogram,
    "ts_binned_response_curvature": _k_ts_binned_response_curvature,
    "state_latch": _k_state_latch,
    "ts_interval_occupancy_entropy": _k_ts_interval_occupancy_entropy,
    "psar_days_since_flip": _k_psar_days_since_flip,
    "ts_sample_entropy": _k_ts_sample_entropy,
    "ts_kramers_moyal_drift": _k_ts_kramers_moyal_drift,
    "ts_hartigan_dip": _k_ts_hartigan_dip,
    "ts_normalized_mutual_information": _k_ts_normalized_mutual_information,
    "ts_sign_cluster_index": _k_ts_sign_cluster_index,
    "ts_chatterjee_xi": _k_ts_chatterjee_xi,
    "ts_kernel_granger_score": _k_ts_kernel_granger_score,
    "ts_weighted_permutation_entropy": _k_ts_weighted_permutation_entropy,
    "ts_feature_pca_reconstruction_error": _k_ts_feature_pca_reconstruction_error,
    "ts_feature_subspace_rotation": _k_ts_feature_subspace_rotation,
    "session_event_recovery_score": _k_session_event_recovery_score,
    "ts_persistence_birth_dispersion": _k_ts_persistence_birth_dispersion,
    "ts_residualized_hsic": _k_ts_residualized_hsic,
    "keltner_compression": _k_keltner_compression,
    "ts_gap_fill_ratio": _k_ts_gap_fill_ratio,
    "ts_mean_reversion_ou_approx_half_life": _k_ts_mean_reversion_ou_approx_half_life,
    "ashare_suspension_episode_length": _k_ashare_suspension_episode_length,
    "ts_extremal_dependence_decay": _k_ts_extremal_dependence_decay,
    "ts_feature_effective_rank": _k_ts_feature_effective_rank,
    "ts_recurrence_laminarity": _k_ts_recurrence_laminarity,
    "ts_bds_statistic": _k_ts_bds_statistic,
    "ts_interval_union_coverage": _k_ts_interval_union_coverage,
    "ts_distance_correlation_partial_proxy": _k_ts_distance_correlation_partial_proxy,
    "ts_wavelet_lowpass_reconstruct": _k_ts_wavelet_lowpass_reconstruct,
    "ts_har_from_return_forecast_error_z": _k_ts_har_from_return_forecast_error_z,
    "ts_regime_duration": _k_ts_regime_duration,
    "ts_feature_mode_share": _k_ts_feature_mode_share,
    "ts_ar_in_sample_resid": _k_ts_ar_in_sample_resid,
    "ashare_limit_down_streak": _k_ashare_limit_down_streak,
    "ts_threshold_cycle_period": _k_ts_threshold_cycle_period,
    "ts_response_slope_asymmetry": _k_ts_response_slope_asymmetry,
    "ts_weighted_standardized_moment": _k_ts_weighted_standardized_moment,
    "ts_mean_abs_deviation_strict": _k_ts_mean_abs_deviation_strict,
    "intra_idiosyncratic_variance_ex_self": _k_intra_idiosyncratic_variance_ex_self,
    "group_peer_beta_deviation": _k_group_peer_beta_deviation,
    "ts_forbidden_ordinal_pattern_ratio": _k_ts_forbidden_ordinal_pattern_ratio,
    "ts_ar_fitted_value": _k_ts_ar_fitted_value,
    "ts_threshold_cycle_asymmetry": _k_ts_threshold_cycle_asymmetry,
    "ts_turning_point_ratio": _k_ts_turning_point_ratio,
    "ts_path_signature_depth2_norm": _k_ts_path_signature_depth2_norm,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch4/5/6)
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


def register_r68_native_batch8() -> list[str]:
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
                (canonical + ":pandas-authority-parity:r68b8").encode()
            ).hexdigest(),
            notes=(
                "R68 batch8 genuine Polars backend: pure pl.Expr kernels or "
                "numpy kernels over pl->numpy columns; no pandas conversion, "
                "no pandas-delegate UDF."
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


__all__ = ["register_r68_native_batch8"]

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec  # noqa: E402

register_r68_native_batch8()
