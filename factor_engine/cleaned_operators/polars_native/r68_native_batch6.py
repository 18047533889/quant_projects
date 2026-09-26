# -*- coding: utf-8 -*-
"""R68 batch6: genuine-Polars backends for the next pandas-delegate canonicals.

Same protocol as batch3/4/5: module-level NumPy authority helpers called
directly on ``pl -> numpy`` column arrays; **no pandas DataFrame is constructed
anywhere**.
"""
from __future__ import annotations

import copy
import hashlib
from collections import deque
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch6"

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
        raise AttributeError(f"{type(value).__name__!r} object has no attribute 'to_numpy'")
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


def _safe_ret(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b == 0.0, np.nan, a / b) - 1.0


def _time_numpy(frame: pl.DataFrame, session_tz: Any) -> np.ndarray:
    """The panel's time column as naive datetime64[ns] (session wall clock)."""
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


def _day_groups(times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group rows by calendar day; returns (day_codes, list of row-index arrays)."""
    day = times.astype("datetime64[D]")
    days = np.unique(day)
    return day, [np.flatnonzero(day == d) for d in days]


def _daily_out(base: pl.DataFrame, times: np.ndarray, per_col: dict[str, list[float]]) -> pl.DataFrame:
    """Fresh daily panel (date column + instrument columns).

    Mirrors ``rolling_pack._pl_rebuild_intraday_result``: a panel with its own
    daily axis, never attached to the minute producer.
    """
    day, _rows = _day_groups(times)
    day_ts = day.astype("datetime64[ns]")
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(per_col[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _deg_exc() -> tuple[type[BaseException], ...]:
    excs: list[type[BaseException]] = [ZeroDivisionError, OverflowError]
    try:
        from factor_engine.backend.operator_errors import DataDegeneracy
        excs.insert(0, DataDegeneracy)
    except Exception:
        pass
    return tuple(excs)


# ===========================================================================
# kernels
# ===========================================================================
def _k_ts_hsic(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dependence_ext import _hsic
    from factor_engine.cleaned_operators.rolling_pack import (
        aligned_pairs, check_window, map_pair_rolling,
    )
    w = check_window(b.get("window", 60))
    out = map_pair_rolling(
        _panel(b["x"]), _panel(b["y"]), w,
        lambda a, bb: _hsic(*aligned_pairs(a, bb)),
    )
    return _rebuild(b["x"], out)


def _k_ts_energy_break_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.distribution_break import (
        _joint_shift_series, _robust_z,
    )
    w = strict_int(b.get("window", 60), "window", minimum=3)
    r = strict_int(b.get("recent_window", 20), "recent_window", minimum=2)
    p = strict_int(b.get("prior_window", 60), "prior_window", minimum=2)
    feats = np.stack([_panel(b[k]) for k in ("f1", "f2", "f3")], axis=2)
    shifts = _joint_shift_series(feats, r, p)
    rows, cols = shifts.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _robust_z(shifts[:, c], w, 3)
    return _rebuild(b["f1"], out)


def _k_ts_dmd_level_dominant_frequency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dmd import _dmd_series, dmd_feasibility
    w = int(b.get("window", 120))
    rank, dim, delay = int(b.get("rank", 4)), int(b.get("dim", 4)), int(b.get("delay", 1))
    # Mirror the pandas op's RelationalParamSpec gate (R9-OP-005): the UDF fails
    # at binding time when K = window-(dim-1)*delay < rank+2, the direct
    # ``_dmd_series`` call alone would not reproduce that failure shape.
    if not dmd_feasibility(window=w, rank=rank, dim=dim, delay=delay):
        raise ValueError(
            "ts_dmd_level_dominant_frequency: DMD requires K = window-(dim-1)*delay "
            f">= rank+2 (window={w}, dim={dim}, delay={delay}, rank={rank})"
        )
    out = _dmd_series(_panel(b["x"]), w, rank, dim, delay, "frequency")
    return _rebuild(b["x"], out)


def _k_cs_knn_tangent_residual(b: dict) -> pl.DataFrame:
    _missing = [k for k in ("f1", "f2", "f3") if b.get(k) is None]
    if _missing:
        raise TypeError(
            f"cs_knn_tangent_residual._calculate_series() missing "
            f"{len(_missing)} required positional argument: '{_missing[0]}'"
        )
    from factor_engine.cleaned_operators.cross_section_local import (
        _KNN_MIN_K, _stack_feats, _tangent_series,
    )
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
    kk = strict_int(b.get("k", 20), "k", lower=1)
    if kk < _KNN_MIN_K:
        raise ValueError(f"cs_knn_tangent_residual requires k >= {_KNN_MIN_K}")
    feats = np.stack([_panel(b.get(k)) for k in ("f1", "f2", "f3")], axis=2)
    out = _tangent_series(feats, kk)
    return _rebuild(b["f1"], out)


def _k_cs_knn_local_gradient_norm(b: dict) -> pl.DataFrame:
    _missing = [k for k in ("target", "f1", "f2", "f3") if b.get(k) is None]
    if _missing:
        raise TypeError(
            f"cs_knn_local_gradient_norm._calculate_series() missing "
            f"{len(_missing)} required positional argument: '{_missing[0]}'"
        )
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_float, strict_int
    from factor_engine.cleaned_operators.cross_section_local import (
        _KNN_MIN_K, _local_gradient_series,
    )
    kk = strict_int(b.get("k", 20), "k", lower=1)
    if kk < _KNN_MIN_K:
        raise ValueError(f"cs_knn_local_gradient_norm requires k >= {_KNN_MIN_K}")
    rg = strict_float(b.get("ridge", 1e-3), "ridge", lower=0.0)
    target = _panel(b.get("target"))
    feats = np.stack([_panel(b.get(k)) for k in ("f1", "f2", "f3")], axis=2)
    out = _local_gradient_series(target, feats, kk, rg)
    return _rebuild(b["target"], out)


def _episode_out(b: dict, which: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.state_episode_excursion import (
        _mae_series, _mfe_series, _retrace_series,
    )
    # The pandas canonical declares (x, state, scale) positionally; a call
    # without ``scale`` must fail with the UDF's binding TypeError before the
    # signed-state validation runs.
    if which in ("mfe", "mae") and b.get("scale") is None:
        cls = "StateEpisodeMfe" if which == "mfe" else "StateEpisodeMae"
        raise TypeError(
            f"{cls}._calculate_series() missing 1 required positional argument: 'scale'"
        )
    xv = _panel(b["x"])
    sv = _panel(b["state"])
    fin = np.isfinite(sv)
    bad = fin & (sv != -1.0) & (sv != 0.0) & (sv != 1.0)
    if bad.any():
        raise ValueError("state must be a signed-state panel ({-1, 0, +1})")
    scale = b.get("scale")
    if which in ("mfe", "mae") and scale is None:
        raise TypeError(
            f"state_episode_{which}._calculate_series() missing 1 required "
            "positional argument: 'scale'"
        )
    if which == "mfe":
        out = _mfe_series(xv, sv, _panel(scale))
    elif which == "mae":
        out = _mae_series(xv, sv, _panel(scale))
    else:
        out = _retrace_series(xv, sv)
    return _rebuild(b["x"], out)


def _k_state_episode_mfe(b: dict) -> pl.DataFrame:
    return _episode_out(b, "mfe")


def _k_state_episode_mae(b: dict) -> pl.DataFrame:
    return _episode_out(b, "mae")


def _k_state_episode_retrace_ratio(b: dict) -> pl.DataFrame:
    return _episode_out(b, "retrace")


def _k_ts_lempel_ziv_complexity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.complexity_ext import (
        _check_complexity_params, _lz_complexity_series,
    )
    _check_complexity_params(b.get("window", 120), bins=b.get("bins", 2))
    out = _lz_complexity_series(
        _panel(b["x"]), b.get("window", 120), b.get("bins", 2),
        b.get("min_contiguous_fraction", 0.5), b.get("min_effective_n", 16),
    )
    return _rebuild(b["x"], out)


def _k_ts_kalman_beta_uncertainty(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.state_space import (
        _BETA_WARMUP, _kalman_beta,
    )
    yv = _panel(b["y"])
    xv = _panel(b["x"])
    q = float(b.get("q", 1e-3))
    r = float(b.get("r", 1.0))
    scale_mode = b.get("scale_mode", "absolute")
    mw = b.get("min_warmup", _BETA_WARMUP)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _kalman_beta(yv[:, c], xv[:, c], q, r, "uncertainty",
                                 scale_mode=scale_mode, min_warmup=mw)
    return _rebuild(b["y"], out)


def _k_ts_l_kurtosis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.moments_ext import (
        _check_window, _l_ratio_series,
    )
    w = _check_window(b.get("window", 60), minimum=4)
    out = _l_ratio_series(
        _panel(b["x"]), w, "kurt",
        min_periods=b.get("min_periods", 20),
        min_coverage_fraction=b.get("min_coverage_fraction", 0.5),
    )
    return _rebuild(b["x"], out)


def _k_is_finite(b: dict) -> pl.DataFrame:
    base = b["x"]
    exprs = [
        pl.col(c).cast(pl.Float64, strict=False).is_finite()
        .cast(pl.Float64).fill_nan(0.0).fill_null(0.0).alias(c)
        for c in _ncols(base)
    ]
    return base.with_columns(exprs)


def _k_index_weight(b: dict) -> pl.DataFrame:
    base = b["weight"]
    arr = np.where(np.isfinite(_panel(base)), _panel(base), np.nan)
    if bool(b.get("normalize", True)):
        denom = np.nansum(arr, axis=1, keepdims=True)
        denom = np.where(np.abs(denom) > 1e-12, denom, np.nan)
        arr = arr / denom
    return _rebuild(base, arr)


def _k_ts_current_drawdown_area(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.stateful.drawdown_path import _trailing_contiguous
    w = _strict_int(b.get("window", 60), "window", lower=2)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            lo = max(0, row - w + 1)
            seg = _trailing_contiguous(xv[lo : row + 1, col])
            if seg.size < 2 or not np.all(seg > 0.0):
                continue
            vals = seg
            running_max = np.maximum.accumulate(vals)
            peak_touch = vals == running_max
            last_peak = int(np.flatnonzero(peak_touch)[-1])
            dd = 1.0 - vals / running_max
            out[row, col] = float(np.sum(dd[last_peak:]))
    return _rebuild(b["x"], out)


def _k_ts_run_concentration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_state import (
        _EPS, _run_windows, _state_series,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(int(b.get("max_run", 20)), name="max_run")
    mp = max(2, int(b.get("min_periods", 2)))
    xv, sv = _run_windows(_panel(b["x"]), _panel(b["state"]), w)
    s_valid, s_val = _state_series(sv)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        buf: deque[float] = deque(maxlen=w)
        run_len = 0
        prev_state = None
        for row in range(rows):
            if not s_valid[row, col]:
                buf.clear(); run_len = 0; prev_state = None
                continue
            if not np.isfinite(xv[row, col]):
                buf.clear(); run_len = 0; prev_state = None
                continue
            cur = s_val[row, col]
            if cur == 0.0:
                buf.clear(); run_len = 0; prev_state = None
                out[row, col] = 0.0
                continue
            if prev_state is not None and cur == prev_state:
                buf.append(xv[row, col]); run_len += 1
            else:
                buf = deque([xv[row, col]], maxlen=w); run_len = 1
            prev_state = cur
            if run_len < mp:
                out[row, col] = np.nan
                continue
            run_abs = float(sum(abs(v) for v in buf))
            run_max = float(max(abs(v) for v in buf))
            out[row, col] = run_max / (run_abs + _EPS)
    return _rebuild(b["x"], out)


def _k_ts_overnight_intraday_cov(b: dict) -> pl.DataFrame:
    ov_open = _panel(b["open"])
    cv = _panel(b["close"])
    pv = _panel(b["pre_close"])
    o = _safe_ret(ov_open, pv)
    i = _safe_ret(cv, ov_open)
    w = int(b.get("window", 60))
    rows, cols = o.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            ow = o[lo : r + 1, c]
            iw = i[lo : r + 1, c]
            ok = np.isfinite(ow) & np.isfinite(iw)
            n = int(ok.sum())
            if n < 5:
                continue
            a = ow[ok]; bb = iw[ok]
            out[r, c] = float(np.sum((a - a.mean()) * (bb - bb.mean())) / (n - 1))
    return _rebuild(b["close"], out)


def _k_ts_staleness(b: dict) -> pl.DataFrame:
    w_raw = b.get("window", 5)
    if isinstance(w_raw, (bool, np.bool_)) or not isinstance(w_raw, (int, np.integer)) or int(w_raw) < 1:
        raise TypeError("window must be a positive integer")
    w = int(w_raw)
    arr = _panel(b["x"])
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        col_arr = arr[:, col]
        for row in range(rows):
            start = max(0, row - w + 1)
            seg = col_arr[start : row + 1]
            finite_idx = np.flatnonzero(np.isfinite(seg))
            if finite_idx.size == 0:
                continue
            out[row, col] = float(row - (start + finite_idx[-1]))
    return _rebuild(b["x"], out)


def _k_ts_signature_mahalanobis_anomaly(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_transform import _sig_mahalanobis_series
    base = b["f1"]
    if int(b.get("depth", 2)) != 2:
        raise ValueError("ts_signature_mahalanobis_anomaly fixes depth = 2")
    pw = int(b.get("path_window", 20))
    hw = int(b.get("history_window", 120))
    # Mirror the pandas op's declared contract (R9-OP-010): spec minimums then
    # the strided-history relational gate — the UDF fails at binding time.
    if pw < 4:
        raise ValueError(
            f"ts_signature_mahalanobis_anomaly: path_window must be >= 4 (got {pw})"
        )
    if hw < 24:
        raise ValueError(
            f"ts_signature_mahalanobis_anomaly: history_window must be >= 24 (got {hw})"
        )
    if hw < 24 * (pw // 4):
        raise ValueError(
            "ts_signature_mahalanobis_anomaly needs >= 24 strided history "
            "signatures: history_window >= 24*(path_window//4) "
            f"(path_window={pw}, history_window={hw})"
        )
    cols = _ncols(base)
    xv = _panel(base)
    f2v = _panel(b.get("f2"))
    f3v = _panel(b.get("f3"))
    out = np.full(xv.shape, np.nan, dtype=float)
    for ci in range(len(cols)):
        fs = [xv[:, ci], f2v[:, ci], f3v[:, ci]]
        out[:, ci] = _sig_mahalanobis_series(fs, pw, hw)
    return _rebuild(base, out)


def _k_ts_scale_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_distribution import (
        _EPS, _mad, map_two_window,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window, valid_values
    ws = check_window(b.get("recent_window", 20), name="recent_window")
    wl = check_window(b.get("old_window", 40), name="old_window")
    mp = max(3, int(b.get("min_periods", 5)))

    def _fn(recent: np.ndarray, old: np.ndarray) -> float:
        ra = valid_values(recent)
        oa = valid_values(old)
        if ra.size < mp or oa.size < mp:
            return np.nan
        mad_r = _mad(ra)
        mad_o = _mad(oa)
        if (not np.isfinite(mad_r) or mad_r < _EPS) or (not np.isfinite(mad_o) or mad_o < _EPS):
            return np.nan
        return float(np.log(mad_r / mad_o))

    out = map_two_window(_panel(b["x"]), ws, wl, _fn)
    return _rebuild(b["x"], out)


def _k_ts_recurrence_trapping_time(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.recurrence_analysis import (
        _check_params, _recurrence_series, _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    w, d, dl, eq, ml = _check_params(
        b.get("window", 40), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), 2,
    )
    out = _recurrence_series(
        _panel(b["x"]), w, d, dl, eq, ml, b.get("min_periods", 10), 2,
        min_effective_fraction=_RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    return _rebuild(b["x"], out)


def _k_ts_extreme_cluster_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_tail import (
        _r62_quantile, _r62_win,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 60))
    side_kind = str(b.get("side", "absolute")).lower()
    if side_kind not in {"absolute", "upper", "lower"}:
        raise ValueError("side must be 'absolute', 'upper' or 'lower'")
    quantile = float(b.get("q", 0.9))
    if not 0.0 < quantile < 1.0:
        raise ValueError("q must be in (0, 1)")
    mp = max(2, int(b.get("min_periods", 2)))
    thr_raw = b.get("threshold", "quantile")
    use_quantile = str(thr_raw).lower() == "quantile"
    absolute_thr = None if use_quantile else float(thr_raw)
    if absolute_thr is not None and not np.isfinite(absolute_thr):
        raise ValueError("threshold must be finite or 'quantile'")
    if side_kind == "absolute" and absolute_thr is not None and absolute_thr < 0.0:
        raise ValueError("absolute threshold must be non-negative")
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    r = np.arange(rows)[:, None]
    lo, i = _r62_win(rows, w)
    valid = i <= r
    idx = np.clip(i, 0, rows - 1)
    mincnt = max(2, mp)
    for col in range(cols):
        V = xv[idx, col]
        fin = valid & np.isfinite(V)
        cnt = fin.sum(axis=1)
        nn = np.maximum(cnt, 1).astype(float)
        mean = np.where(fin, V, 0.0).sum(axis=1) / nn
        cen = np.where(fin, V - mean[:, None], 0.0)
        with np.errstate(invalid="ignore", over="ignore"):
            std0 = np.sqrt((cen * cen).sum(axis=1) / nn)
        spread = (cnt >= mincnt) & (std0 >= 1e-12)
        if use_quantile:
            if side_kind == "absolute":
                thr = _r62_quantile(np.abs(V), fin, quantile)
                ext = fin & (np.abs(V) >= thr[:, None])
            elif side_kind == "upper":
                thr = _r62_quantile(V, fin, quantile)
                ext = fin & (V >= thr[:, None])
            else:
                thr = _r62_quantile(V, fin, 1.0 - quantile)
                ext = fin & (V <= thr[:, None])
        else:
            if side_kind == "absolute":
                ext = fin & (np.abs(V) >= absolute_thr)
            elif side_kind == "upper":
                ext = fin & (V >= absolute_thr)
            else:
                ext = fin & (V <= absolute_thr)
        count = ext.sum(axis=1)
        pos = np.cumsum(fin, axis=1) - 1
        Ec = np.zeros_like(ext)
        rr, cc = np.nonzero(fin)
        Ec[rr, pos[rr, cc]] = ext[rr, cc]
        clustered = (Ec[:, :-1] & Ec[:, 1:]).sum(axis=1)
        ok = spread & (count >= mp)
        out[:, col] = np.where(ok, clustered / np.maximum(count, 1), np.nan)
    return _rebuild(b["x"], out)


def _k_ts_expected_shortfall_asymmetry(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_distribution import _es_asymmetry_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 60))
    quantile = float(b.get("q", 0.05))
    if not 0.0 < quantile < 0.5:
        raise ValueError("q must be in (0, 0.5)")
    min_tail = max(2, int(b.get("min_tail_count", 3)))
    out = _es_asymmetry_series(_panel(b["x"]), w, quantile, min_tail)
    return _rebuild(b["x"], out)


def _k_ts_leverage_effect(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_volatility import _corr, _std
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 60))
    mp = max(3, int(b.get("min_periods", 3)))
    rv = _panel(b["ret"])
    vol = map_rolling(rv, w, lambda c: _std(c, 2))
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(1, row - w + 1)
            a = rv[start - 1 : row, col]
            bb = vol[start : row + 1, col]
            out[row, col] = _corr(a, bb, mp)
    return _rebuild(b["ret"], out)


def _k_ts_value_at_argextreme(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_integer
    w = strict_integer(b.get("window", 20), "window", minimum=2)
    mode_s = str(b.get("mode", "max")).lower()
    if mode_s not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")
    vv = _panel(b.get("value"))
    sv = _panel(b.get("score"))
    rows, cols = vv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    off = 0 if b.get("include_current", False) else 1
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


def _k_ts_turning_intensity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_shape import _trailing_contiguous
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 20))
    mp = max(3, int(b.get("min_periods", 3)))

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        if v.size < mp:
            return np.nan
        d = np.diff(v)
        if d.size < 2:
            return np.nan
        mad_d = float(np.median(np.abs(d - np.median(d))))
        mags = []
        for i in range(1, d.size):
            if d[i - 1] == 0.0 or d[i] == 0.0:
                continue
            if (d[i] > 0) != (d[i - 1] > 0):
                mags.append(abs(d[i] - d[i - 1]))
        if not mags:
            return np.nan
        if mad_d == 0.0:
            return np.nan
        return float(np.mean(mags)) / mad_d

    out = map_rolling(_panel(b["x"]), w, _fn)
    return _rebuild(b["x"], out)


def _k_ts_transition_intensity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_state import (
        _EPS, _run_windows, _state_series,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    xv, sv = _run_windows(_panel(b["x"]), _panel(b["state"]), w)
    s_valid, s_val = _state_series(sv)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(1, row - w + 1)
            n = 0
            wsum = 0.0
            dx_vals = []
            for i in range(start, row + 1):
                if not (np.isfinite(xv[i, col]) and np.isfinite(xv[i - 1, col])):
                    continue
                if not (s_valid[i, col] and s_valid[i - 1, col]):
                    continue
                if s_val[i, col] != s_val[i - 1, col]:
                    d = xv[i, col] - xv[i - 1, col]
                    wsum = wsum + abs(d)
                    dx_vals.append(d)
                    n = n + 1
            if n == 0:
                continue
            if b.get("normalize", False):
                dx_arr = np.asarray(dx_vals, dtype=float)
                mad = float(np.median(np.abs(dx_arr - np.median(dx_arr))))
                if not np.isfinite(mad) or mad < _EPS:
                    continue
                out[row, col] = (wsum / float(n)) / mad
            else:
                out[row, col] = wsum / float(n)
    return _rebuild(b["x"], out)


def _k_ts_state_integral(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_state import (
        _DEFAULT_HYSTERESIS_MISSING_POLICY, HysteresisStateKernel,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    hi = float(b.get("upper", 1.0))
    lo = float(b.get("lower", 0.0))
    if not (0.0 <= lo < hi):
        raise ValueError("require 0 <= lower < upper")
    w_run = check_window(int(b.get("max_run", 120)), name="max_run")
    hl = b.get("half_life")
    lamb: float | None = None
    if hl is not None and hl != "":
        lamb = float(hl)
        if lamb <= 0.0:
            raise ValueError("half_life must be > 0")
    zv = _panel(b["z"])
    rows, cols = zv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        kernel = HysteresisStateKernel(
            hi, lo, b.get("missing_policy", _DEFAULT_HYSTERESIS_MISSING_POLICY),
            integral_cap=w_run,
        )
        for row in range(rows):
            val = zv[row, col]
            kernel.step(val, row)
            if not np.isfinite(val):
                continue
            st = kernel.current_state
            if st == 0:
                out[row, col] = 0.0
                continue
            if lamb is not None:
                s = 0.0
                for age, contrib in enumerate(reversed(kernel.contributions)):
                    s += float(contrib) * float(np.exp(-(age + 1) / lamb))
            else:
                s = float(sum(kernel.contributions))
            out[row, col] = st * s
    return _rebuild(b["z"], out)


def _k_ts_bicoherence_top_decile_mean(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.research_spectral import (
        _bicoherence_top_decile_mean, trailing_contiguous_finite,
    )
    w = int(b.get("window", 120))
    if w < 16:
        raise ValueError("ts_bicoherence_top_decile_mean requires window >= 16")
    ns = int(b.get("n_segments", 4))
    # RelationalParamSpec parity: floor(window/n_segments) >= 8 at binding time.
    if w // ns < 8:
        raise ValueError(
            "ts_bicoherence_top_decile_mean: bicoherence requires "
            f"floor(window/n_segments) >= 8 (window={w}, n_segments={ns})"
        )
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = xv[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            # Audit #58 parity: no partial warmup — require the FULL window
            # (contiguous finite) or NaN.
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < w:
                continue
            val = _bicoherence_top_decile_mean(v, ns)
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


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
        X = rv[start : fit_end + 1]
        out[row] = fn(X, rv[row])
    return out


def _k_panel_rolling_pca_explained_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_commonality
    rv = _panel(b["ret"])
    out = _rolling_pca_np(rv, int(b.get("window", 120)),
                          lambda X, c: _pca_commonality(X, c, int(b.get("n_components", 5))))
    return _rebuild(b["ret"], out)


def _k_panel_rolling_pca_loading(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_loading
    base = b["ret"]
    ids = tuple(_ncols(base))
    rv = _panel(base)
    comp = int(b.get("component", 0))
    out = _rolling_pca_np(rv, int(b.get("window", 120)),
                          lambda X, c: _pca_loading(X, c, comp, ids))
    return _rebuild(base, out)


def _k_cs_autoencoder_reconstruction_error(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _EPS as _PM_EPS
    collected = [_panel(b[k]) for k in ("f1", "f2", "f3", "f4") if b.get(k) is not None]
    if len(collected) < 2:
        raise ValueError("requires at least 2 feature panels")
    window = int(b.get("window", 120))
    rank = max(1, min(2, len(collected) - 1))
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
            sd = np.where(sd > _PM_EPS, sd, 1.0)
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


def _k_intraday_volatility_concentration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_vol_ext import (
        _concentration, _rolling_returns,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 240))
    out = _rolling_returns(_panel(b["returns"]), w, _concentration)
    return _rebuild(b["returns"], out)


def _k_cs_tail_retention(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_integer
    from factor_engine.cleaned_operators.stateful.rotation import (
        _EPS, _cohort_denominator, _exact_tail_weights, _tail_quantile, _valid_label,
    )
    lk = strict_integer(b.get("lag", 5), "lag", minimum=1)
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
    gv = None
    if group is not None and isinstance(group, (pl.DataFrame, pl.Series)):
        if isinstance(group, pl.Series):
            group = group.to_frame()
        gv = np.empty((group.height, len(_ncols(group))), dtype=object)
        for j, c in enumerate(_ncols(group)):
            gv[:, j] = group[c].to_numpy(allow_copy=True)
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
                cur_members = np.array([g == lab for g in g_row], dtype=bool)
                prev_members = np.array([g == lab for g in g_prev], dtype=bool)
                if not (np.any(cur_members) or np.any(prev_members)):
                    continue
                w_prev = np.full(cols, np.nan)
                w_cur = np.full(cols, np.nan)
                if np.any(prev_members):
                    w_prev[prev_members] = _exact_tail_weights(xv[r - lk][prev_members], q, top)
                if np.any(cur_members):
                    w_cur[cur_members] = _exact_tail_weights(xv[r][cur_members], q, top)
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


def _k_ts_persistence_diagram_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_topology import _persistence_shift_series
    w = _strict_int(b.get("window", 60), "window", lower=6)
    t = _strict_int(b.get("tau", 1), "tau", lower=1)
    m = _strict_int(b.get("embedding_dim", 3), "embedding_dim", lower=2, upper=6)
    if w - (m - 1) * t < 3:
        raise ValueError(
            "ts_persistence_diagram_shift requires window-(embedding_dim-1)*tau >= 3 "
            f"(window={w}, embedding_dim={m}, tau={t})"
        )
    out = _persistence_shift_series(_panel(b["x"]), w, t, m)
    return _rebuild(b["x"], out)


def _k_ts_kramers_moyal_diffusion(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import _km_series
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_integer
    w = strict_integer(b.get("window", 60), "window", minimum=10)
    nb = strict_integer(b.get("bins", 8), "bins", minimum=2)
    if nb < 2:
        raise ValueError("ts_kramers_moyal_diffusion requires bins >= 2")
    lg = strict_integer(b.get("lag", 1), "lag", minimum=1)
    if w <= lg:
        raise ValueError("ts_kramers_moyal_diffusion requires window > lag")
    mbc_raw = b.get("min_bin_count")
    if mbc_raw is None:
        mbc = max(3, int(np.ceil(w / nb * 0.25)))
    else:
        mbc = strict_integer(mbc_raw, "min_bin_count", minimum=1)
    if mbc > w - lg:
        raise ValueError("min_bin_count must fit available lagged transitions")
    out = _km_series(_panel(b["x"]), w, nb, lg, 2, mbc)
    return _rebuild(b["x"], out)


def _k_ts_garch_persistence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import (
        _GARCH_FIT_CACHE, _garch_path,
    )
    w = int(b.get("window", 120))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    _GARCH_FIT_CACHE.clear()
    try:
        for c in range(cols):
            for r in range(rows):
                i0 = max(0, r - w + 1)
                out[r, c] = _garch_path(xv[i0 : r + 1, c], w, "persistence", False, 0.0)
    finally:
        _GARCH_FIT_CACHE.clear()
    return _rebuild(b["x"], out)


def _k_ts_beta_break_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.feature_geometry import _beta_break_chunk
    from factor_engine.cleaned_operators.rolling_pack import check_window
    r = check_window(b.get("recent_window", 30), name="recent_window")
    p = check_window(b.get("prior_window", 90), name="prior_window")
    w = r + p
    yv = _panel(b["y"])
    xv = _panel(b["x"])
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            lo = max(0, row - w + 1)
            out[row, col] = _beta_break_chunk(yv[lo : row + 1, col], xv[lo : row + 1, col], r, p, 5)
    return _rebuild(b["y"], out)


def _k_ts_autocorr_decay_half_life(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.sequence_complexity import _vec_column_autocorr_half_life
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 60))
    ml = _strict_int(b.get("max_lag", 10), "max_lag", lower=2, upper=30)
    mp = _strict_int(b.get("min_periods", 2), "min_periods", lower=2)
    abs_ = bool(b.get("use_abs", False))
    out = _vec_column_autocorr_half_life(_panel(b["x"]), w, ml, abs_, mp)
    return _rebuild(b["x"], out)


def _k_ashare_limit_down_volume_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import _event_volume_ratio
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    out = _event_volume_ratio(_panel(b["volume"]), _panel(b["limit_down_event"]), w)
    return _rebuild(b["volume"], out)


def _k_ts_generalized_hurst_exponent(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.multifractal import _check_window, _hurst_series
    w = _check_window(b.get("window", 120))
    qv = float(b.get("q", 2.0))
    if qv not in (0.5, 1.0, 2.0, 3.0, 4.0):
        raise ValueError("q must be one of {0.5, 1, 2, 3, 4} (reviewed grid)")
    out = _hurst_series(_panel(b["x"]), w, qv)
    return _rebuild(b["x"], out)


def _k_ts_endpoint_deviation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_shape import (
        _EPS, _ols_fit, _trailing_contiguous,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 20))
    mp = max(3, int(b.get("min_periods", 3)))

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        n = v.size
        if n < mp:
            return np.nan
        bb, a, sigma = _ols_fit(v)
        x_hat_last = a + bb * float(n - 1)
        num = float(v[-1]) - x_hat_last
        if sigma < _EPS:
            return 0.0 if abs(num) < _EPS else np.nan
        return num / sigma

    out = map_rolling(_panel(b["x"]), w, _fn)
    return _rebuild(b["x"], out)


def _k_intra_realized_beta_ex_self(b: dict) -> pl.DataFrame:
    """numpy mirror of realized_beta._beta_daily(close, cap, _realized_beta, ex_self=True).

    Same day-axis / weight broadcast contract as batch4's validated
    ``_k_intra_market_model_r2_ex_self`` mirror, with the daily metric swapped
    to the realized-beta estimator.
    """
    from factor_engine.cleaned_operators.intraday.realized_beta import (
        _realized_beta, _EPS,
    )
    close, caps = b["close"], b["free_market_cap"]
    cv = _panel(close)
    rows, cols = cv.shape
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
    day_codes, groups = _day_groups(day_keys)
    uniq_days = np.unique(day_codes)
    cap_days = _time_numpy(caps, None).astype("datetime64[D]")
    cap_vals = _panel(caps)
    cap_row_by_day = {d: i for i, d in enumerate(np.unique(cap_days))}
    w_bc = np.full((rows, cols), np.nan, dtype=float)
    for k, d in enumerate(uniq_days):
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
    vals: dict[str, list[float]] = {c: [] for c in names}
    for k in range(len(groups)):
        g = groups[k]
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
                vals[c].append(float(_realized_beta(rr, mm)))
            except (ValueError, ZeroDivisionError, OverflowError):
                vals[c].append(np.nan)
    return _daily_out(close, times, vals)


def _lead_lag_day(day_px: np.ndarray, j: int, lag: int, min_obs: int) -> float:
    """numpy twin of time_structure_v2._lead_lag_panel for one day / one instrument."""
    n, nc = day_px.shape
    if nc < 2:
        return np.nan
    try:
        Pff = day_px.copy()
        for r in range(1, n):
            holes = ~np.isfinite(Pff[r]) & np.isfinite(Pff[r - 1])
            Pff[r, holes] = Pff[r - 1, holes]
        P = np.where(Pff > 0.0, Pff, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.diff(np.log(P), axis=0)
        n = L.shape[0]
        if n < 4:
            return np.nan
        stock = L[:, j]
        peers = [i for i in range(nc) if i != j]
        if not peers:
            return np.nan
        with np.errstate(invalid="ignore"):
            m_ex = np.nanmean(L[:, peers], axis=1)
        scores: list[float] = []
        for k in range(1, lag + 1):
            c_lag = _corr_skipna(stock[k:], m_ex[: n - k], min_obs)
            c_lead = _corr_skipna(stock[: n - k], m_ex[k:], min_obs)
            if np.isfinite(c_lag) and np.isfinite(c_lead):
                scores.append(c_lag - c_lead)
        return float(np.mean(scores)) if scores else np.nan
    except Exception:
        return np.nan


def _k_intra_market_lead_lag_ex_self(b: dict) -> pl.DataFrame:
    """numpy mirror of time_structure_v2._lead_lag_panel(close, lag, min_obs, industry=None)."""
    from factor_engine.cleaned_operators.intraday.time_structure_v2 import _corr_skipna
    base = b["close"]
    if isinstance(base, pl.Series):
        base = base.to_frame()
    lag = b.get("lag", 3)
    if lag is None:
        lag = 3
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be >= 1")
    min_obs = max(1, int(b.get("min_obs", 10)))
    times = _time_numpy(base, b.get("session_tz"))
    if times is None:
        raise ValueError("intra_market_lead_lag_ex_self requires a time column")
    cv = _panel(base)
    day_codes, groups = _day_groups(times)
    per_col: dict[str, list[float]] = {c: [] for c in _ncols(base)}
    for g in groups:
        for j, c in enumerate(_ncols(base)):
            per_col[c].append(_lead_lag_day(cv[g], j, lag, min_obs))
    return _daily_out(base, times, per_col)


def _k_intra_realized_correlation_ex_self(b: dict) -> pl.DataFrame:
    """numpy mirror of realized_beta._beta_daily(close, cap, _realized_corr, ex_self=True).

    Same day-axis / weight broadcast contract as batch4's validated
    ``_k_intra_market_model_r2_ex_self`` mirror, with the daily metric swapped
    to the realized-correlation estimator.
    """
    from factor_engine.cleaned_operators.intraday.realized_beta import (
        _realized_corr, _EPS,
    )
    close, caps = b["close"], b["free_market_cap"]
    cv = _panel(close)
    rows, cols = cv.shape
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
    day_codes, groups = _day_groups(day_keys)
    uniq_days = np.unique(day_codes)
    cap_days = _time_numpy(caps, None).astype("datetime64[D]")
    cap_vals = _panel(caps)
    cap_row_by_day = {d: i for i, d in enumerate(np.unique(cap_days))}
    w_bc = np.full((rows, cols), np.nan, dtype=float)
    for k, d in enumerate(uniq_days):
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
    vals: dict[str, list[float]] = {c: [] for c in names}
    for k in range(len(groups)):
        g = groups[k]
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
    return _daily_out(close, times, vals)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ts_hsic": _k_ts_hsic,
    "ts_energy_break_score": _k_ts_energy_break_score,
    "ts_dmd_level_dominant_frequency": _k_ts_dmd_level_dominant_frequency,
    "cs_knn_tangent_residual": _k_cs_knn_tangent_residual,
    "state_episode_mfe": _k_state_episode_mfe,
    "state_episode_mae": _k_state_episode_mae,
    "state_episode_retrace_ratio": _k_state_episode_retrace_ratio,
    "ts_lempel_ziv_complexity": _k_ts_lempel_ziv_complexity,
    "cs_knn_local_gradient_norm": _k_cs_knn_local_gradient_norm,
    "ts_kalman_beta_uncertainty": _k_ts_kalman_beta_uncertainty,
    "ts_l_kurtosis": _k_ts_l_kurtosis,
    "is_finite": _k_is_finite,
    "index_weight": _k_index_weight,
    "ts_current_drawdown_area": _k_ts_current_drawdown_area,
    "ts_run_concentration": _k_ts_run_concentration,
    "ts_overnight_intraday_cov": _k_ts_overnight_intraday_cov,
    "ts_staleness": _k_ts_staleness,
    "ts_signature_mahalanobis_anomaly": _k_ts_signature_mahalanobis_anomaly,
    "ts_scale_shift": _k_ts_scale_shift,
    "ts_recurrence_trapping_time": _k_ts_recurrence_trapping_time,
    "ts_extreme_cluster_ratio": _k_ts_extreme_cluster_ratio,
    "ts_expected_shortfall_asymmetry": _k_ts_expected_shortfall_asymmetry,
    "ts_leverage_effect": _k_ts_leverage_effect,
    "ts_value_at_argextreme": _k_ts_value_at_argextreme,
    "ts_turning_intensity": _k_ts_turning_intensity,
    "ts_transition_intensity": _k_ts_transition_intensity,
    "ts_state_integral": _k_ts_state_integral,
    "ts_bicoherence_top_decile_mean": _k_ts_bicoherence_top_decile_mean,
    "panel_rolling_pca_explained_ratio": _k_panel_rolling_pca_explained_ratio,
    "panel_rolling_pca_loading": _k_panel_rolling_pca_loading,
    "cs_autoencoder_reconstruction_error": _k_cs_autoencoder_reconstruction_error,
    "intraday_volatility_concentration": _k_intraday_volatility_concentration,
    "cs_tail_retention": _k_cs_tail_retention,
    "ts_persistence_diagram_shift": _k_ts_persistence_diagram_shift,
    "ts_kramers_moyal_diffusion": _k_ts_kramers_moyal_diffusion,
    "ts_garch_persistence": _k_ts_garch_persistence,
    "ts_beta_break_score": _k_ts_beta_break_score,
    "ts_autocorr_decay_half_life": _k_ts_autocorr_decay_half_life,
    "ashare_limit_down_volume_ratio": _k_ashare_limit_down_volume_ratio,
    "ts_generalized_hurst_exponent": _k_ts_generalized_hurst_exponent,
    "ts_endpoint_deviation": _k_ts_endpoint_deviation,
    "intra_realized_beta_ex_self": _k_intra_realized_beta_ex_self,
    "intra_realized_correlation_ex_self": _k_intra_realized_correlation_ex_self,
    "intra_market_lead_lag_ex_self": _k_intra_market_lead_lag_ex_self,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch3/4/5)
# ---------------------------------------------------------------------------
class _R68NativeOperator(Operator):
    _HANDLES_CALL_CONTRACT = True

    def __init__(self, canonical: str, metadata, kernel: Callable[[dict], pl.DataFrame]):
        self._canonical = canonical
        self.metadata = metadata
        self._kernel_fn = kernel

    def calculate(self, *args, **kwargs):
        try:
            args, kwargs = self._prepare_call(args, kwargs)
        except Exception:
            pass
        bound = dict(zip(self.metadata.param_names, args))
        bound.update(kwargs)
        return self._kernel_fn(bound)

    _calculate_series = calculate


def register_r68_native_batch6() -> list[str]:
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
                (canonical + ":pandas-authority-parity:r68b6").encode()
            ).hexdigest(),
            notes=(
                "R68 batch6 genuine Polars backend: numpy kernels over "
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


__all__ = ["register_r68_native_batch6"]

register_r68_native_batch6()
