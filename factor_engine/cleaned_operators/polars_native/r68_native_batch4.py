"""R68 batch4: genuine-Polars backends for 39 pandas-delegate canonicals.

Same protocol as ``r68_native_batch3``: pure ``pl.Expr`` kernels for the
formula-shaped operators; module-level NumPy authority helpers called directly
on ``pl -> numpy`` column arrays for the algorithmic ones.  **No pandas
DataFrame is constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``,
no ``iterrows``).

Daily-grain intraday kernels (``intraday_*``) build their own daily panel
(date column + instrument columns) exactly like
``rolling_pack._pl_rebuild_intraday_result`` does for the pandas-delegate UDF
they replace.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch4"

_SKIP = frozenset({
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "__fe_instrument__",
})


def _ncols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _SKIP]


def _col(frame: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).fill_nan(None)


def _panel(value: Any) -> np.ndarray:
    """A polars wide panel -> (rows, n_instruments) float64 (NaN for nulls).

    A non-panel input mirrors the pandas authority's own failure mode
    (``AttributeError: ... has no attribute 'to_numpy'``) so a caller passing
    a scalar where a panel is required fails closed the same way the
    pandas-delegate UDF it replaces does.
    """
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


def _time_numpy_utc(frame: pl.DataFrame) -> np.ndarray:
    tcol = None
    for c in ("__fe_time__", "date", "timestamp", "trade_date", "datetime"):
        if c in frame.columns:
            tcol = c
            break
    if tcol is None:
        raise ValueError("panel has no time column")
    return frame[tcol].to_numpy(allow_copy=True).astype("datetime64[ns]")


def _per_day_groups(day_keys: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    days = np.unique(day_keys)
    return days, [np.flatnonzero(day_keys == d) for d in days]


def _daily_panel(base: pl.DataFrame, day_ts: np.ndarray, vals: dict[str, list[float]]) -> pl.DataFrame:
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(vals[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _daily_out(base: pl.DataFrame, times: np.ndarray, per_col: dict[str, list[float]]) -> pl.DataFrame:
    """Build the daily pl panel (date column + instrument columns).

    Mirrors ``rolling_pack._pl_rebuild_intraday_result``: a fresh panel with
    its own daily axis, never attached to the minute producer.
    """
    day, _rows = _day_groups(times)
    day_ts = day.astype("datetime64[ns]")
    series = [pl.Series("date", day_ts)]
    for c in _ncols(base):
        series.append(pl.Series(c, np.asarray(per_col[c], dtype=float), dtype=pl.Float64))
    return pl.DataFrame(series)


def _deg_exc() -> tuple[type[BaseException], ...]:
    """(DataDegeneracy, ZeroDivisionError, OverflowError) when available."""
    excs: list[type[BaseException]] = [ZeroDivisionError, OverflowError]
    try:
        from factor_engine.backend.operator_errors import DataDegeneracy
        excs.insert(0, DataDegeneracy)
    except Exception:
        pass
    return tuple(excs)


# ===========================================================================
# kernels — direct module-level numpy authority helpers
# ===========================================================================
def _dmd_guard(canonical: str, w: int, rk: int, d: int, dl: int) -> None:
    from factor_engine.cleaned_operators.dmd import dmd_feasibility
    if not dmd_feasibility(window=w, rank=rk, dim=d, delay=dl, top_k=None):
        raise ValueError(
            f"{canonical}: DMD parameter domain infeasible "
            f"(window={w}, rank={rk}, dim={d}, delay={dl})"
        )


def _k_ts_dmd_return_dominant_growth_rate(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dmd import _dmd_series
    if int(b.get("window", 120)) < 10:
        raise ValueError("ts_dmd_return_dominant_growth_rate requires window >= 10")
    w, rk, d, dl = (
        int(b.get("window", 120)), int(b.get("rank", 4)),
        int(b.get("dim", 4)), int(b.get("delay", 1)),
    )
    _dmd_guard("ts_dmd_return_dominant_growth_rate", w, rk, d, dl)
    out = _dmd_series(_panel(b["x"]), w, rk, d, dl, "growth")
    return _rebuild(b["x"], out)


def _k_ts_dmd_return_dominant_frequency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dmd import _dmd_series
    if int(b.get("window", 120)) < 10:
        raise ValueError("ts_dmd_return_dominant_frequency requires window >= 10")
    w, rk, d, dl = (
        int(b.get("window", 120)), int(b.get("rank", 4)),
        int(b.get("dim", 4)), int(b.get("delay", 1)),
    )
    _dmd_guard("ts_dmd_return_dominant_frequency", w, rk, d, dl)
    out = _dmd_series(_panel(b["x"]), w, rk, d, dl, "frequency")
    return _rebuild(b["x"], out)


def _k_ts_cross_quantilogram(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_quantile_dynamics import (
        _check_q, _check_side, _cross_quantilogram_chunk,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_pair_rolling
    w = int(b.get("window", 120))
    tq = _check_q(b.get("target_q", 0.1))
    sq = _check_q(b.get("source_q", 0.1))
    lg = int(b.get("lag", 1))
    ts_ = str(b.get("target_side", "lower"))
    ss_ = str(b.get("source_side", "lower"))
    _check_side(ts_)
    _check_side(ss_)
    if lg < 1:
        raise ValueError("ts_cross_quantilogram requires lag >= 1")
    if w < lg + 3:
        raise ValueError("ts_cross_quantilogram requires window >= lag + 3")
    fixed = bool(b.get("fixed_threshold", False))
    out = map_pair_rolling(
        _panel(b["target"]), _panel(b["source"]), w,
        lambda a, bb: _cross_quantilogram_chunk(a, bb, tq, sq, lg, ts_, ss_, fixed),
    )
    return _rebuild(b["target"], out)


def _k_ts_quantile_crossing_spectral_concentration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_quantile_dynamics import (
        _check_q, _check_side, _hit_spectral_concentration_chunk,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    w = int(b.get("window", 120))
    q = _check_q(b.get("quantile", 0.1))
    side = str(b.get("side", "lower"))
    _check_side(side)
    if w < 8:
        raise ValueError("ts_quantile_crossing_spectral_concentration requires window >= 8")
    fixed = bool(b.get("fixed_threshold", False))
    out = map_rolling(
        _panel(b["x"]), w,
        lambda c: _hit_spectral_concentration_chunk(c, q, side, fixed),
    )
    return _rebuild(b["x"], out)


def _k_ts_spectral_quality_factor(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.spectral import (
        _check_spectral_params, _spectral_quality_factor_series,
    )
    w = _check_spectral_params(b.get("window", 60))
    out = _spectral_quality_factor_series(_panel(b["x"]), w)
    return _rebuild(b["x"], out)


def _k_ashare_limit_up_streak(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _break_mask, _consecutive_streak, _tolerance, _tradeable,
    )
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    cv = _panel(b["close"])
    lv = _panel(b["high_limit"])
    vv = _panel(b["valid_trade"])
    rows, cols = cv.shape
    condition = np.full((rows, cols), np.nan, dtype=float)
    tradeable = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if _break_mask(r, c, [cv, lv]) or not _tradeable(vv, r, c):
                continue
            if np.isfinite(lv[r, c]) and cv[r, c] >= lv[r, c] * (1.0 - tol):
                condition[r, c] = 1.0
            else:
                condition[r, c] = 0.0
            tradeable[r, c] = True
    out = _consecutive_streak(condition, tradeable, rows, cols)
    return _rebuild(b["close"], out)


def _k_ashare_limit_touch_count(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _rolling_count, _tolerance,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    side = str(b.get("side", "up")).lower()
    if side not in {"up", "down"}:
        raise ValueError("side must be 'up' or 'down'")
    hv = _panel(b["high"])
    lv = _panel(b["low"])
    hlv = _panel(b["high_limit"])
    llv = _panel(b["low_limit"])
    rows, cols = hv.shape
    condition = np.zeros((rows, cols), dtype=float)
    known = np.zeros((rows, cols), dtype=bool)
    if side == "up":
        ok = np.isfinite(hv) & np.isfinite(hlv)
        known = ok
        condition = np.where(ok & (hv >= hlv * (1.0 - tol)), 1.0, 0.0)
    else:
        ok = np.isfinite(lv) & np.isfinite(llv)
        known = ok
        condition = np.where(ok & (lv <= llv * (1.0 + tol)), 1.0, 0.0)
    out = _rolling_count(condition, known, w)
    return _rebuild(b["high"], out)


def _k_ashare_days_since_limit_up(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _days_since, strict_nonnegative_int,
    )
    ml = b.get("max_lookback")
    limit_v = None if ml is None else strict_nonnegative_int(ml, "max_lookback")
    out = _days_since(_panel(b["limit_up_event"]), limit_v)
    return _rebuild(b["limit_up_event"], out)


def _k_ashare_limit_up_volume_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import _event_volume_ratio
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    out = _event_volume_ratio(_panel(b["volume"]), _panel(b["limit_up_event"]), w)
    return _rebuild(b["volume"], out)


def _k_ts_semivariance_balance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_volatility import _EPS
    from factor_engine.cleaned_operators.rolling_pack import map_rolling, valid_values
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    mp = max(2, int(b.get("min_periods", 2)))

    def _fn(chunk: np.ndarray) -> float:
        v = valid_values(chunk)
        if v.size < mp:
            return np.nan
        pos = float(np.sum(v[v > 0.0] ** 2))
        neg = float(np.sum(v[v < 0.0] ** 2))
        denom = pos + neg
        if denom < _EPS:
            return np.nan
        return float((pos - neg) / denom)

    out = map_rolling(_panel(b["ret"]), w, _fn)
    return _rebuild(b["ret"], out)


def _k_ts_persistence_entropy_h0(b: dict) -> pl.DataFrame:
    return _persistence_entropy_out(b, True)


def _k_ts_persistence_entropy_h1(b: dict) -> pl.DataFrame:
    return _persistence_entropy_out(b, False)


def _persistence_entropy_out(b: dict, h0: bool) -> pl.DataFrame:
    from factor_engine.cleaned_operators.topology_ext import _persistence_entropy_series
    w = _strict_int(b.get("window", 120), "window", lower=6)
    t = _strict_int(b.get("tau", 1), "tau", lower=1)
    d = _strict_int(b.get("dim", 3), "dim", lower=2, upper=6)
    if w - (d - 1) * t < 3:
        raise ValueError(
            "ts_persistence_entropy requires window-(dim-1)*tau >= 3 "
            f"(window={w}, dim={d}, tau={t})"
        )
    out = _persistence_entropy_series(_panel(b["x"]), w, t, d, h0)
    return _rebuild(b["x"], out)


def _k_ts_recurrence_determinism(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rqa_ext import _rqa_series
    out = _rqa_series(
        _panel(b["x"]), b.get("window", 60), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), b.get("min_line", 4),
        b.get("min_periods", 10), "determinism", b.get("theiler"),
    )
    return _rebuild(b["x"], out)


def _k_ts_recurrence_rate(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.recurrence_analysis import (
        _check_params, _recurrence_series, _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    w, d, dl, eq, ml = _check_params(
        b.get("window", 40), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), 2,
    )
    out = _recurrence_series(
        _panel(b["x"]), w, d, dl, eq, ml, b.get("min_periods", 10), 0,
        min_effective_fraction=_RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    return _rebuild(b["x"], out)


def _k_ts_har_rv_next_vol_forecast(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import _har_rv
    w = int(b.get("window", 120))
    xv = _panel(b["rv"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _har_rv(xv[i0 : r + 1, c], w, "forecast")
    return _rebuild(b["rv"], out)


def _k_ts_garch_vol_surprise(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import (
        _GARCH_FIT_CACHE, _garch_vol_surprise,
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
                out[r, c] = _garch_vol_surprise(xv[i0 : r + 1, c], w)
    finally:
        _GARCH_FIT_CACHE.clear()
    return _rebuild(b["x"], out)


def _k_ts_kalman_level(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.state_space import _kalman_level
    xv = _panel(b["x"])
    q = float(b.get("q", 1e-4))
    r = float(b.get("r", 1.0))
    scale_mode = b.get("scale_mode", "absolute")
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _kalman_level(xv[:, c], q, r, "level", scale_mode=scale_mode)
    return _rebuild(b["x"], out)


def _k_ts_ssa_reconstruction_residual(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.hankel import (
        _check_hankel_params, _ssa_reconstruction_residual_series,
    )
    w, e, k, mcf = _check_hankel_params(
        b.get("window", 60), b.get("embedding_dim", 15),
        b.get("n_components", 3), b.get("min_contiguous_fraction", 0.8),
    )
    out = _ssa_reconstruction_residual_series(
        _panel(b["x"]), w, e, k, "strict_contiguous", mcf,
    )
    return _rebuild(b["x"], out)


def _k_ts_pettitt_change_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.glr_change import (
        _glr_feasibility, _glr_series,
    )
    w, ms = _glr_feasibility(b.get("window", 120), b.get("min_segment", 10),
                             "ts_pettitt_change_score")
    out = _glr_series(_panel(b["x"]), w, ms, "pettitt")
    return _rebuild(b["x"], out)


def _k_ts_path_signature_area(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.path_signature import _sig_area
    w = _strict_int(b.get("window", 60), "window", lower=3)
    xv = _panel(b["x"])
    yv = _panel(b["y"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _sig_area(xv[i0 : r + 1, c], yv[i0 : r + 1, c], w)
    return _rebuild(b["x"], out)


def _k_ts_lagged_mutual_information(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.nonlinear_dependence import _quantile_hist_mi
    from factor_engine.cleaned_operators.rolling_pack import (
        aligned_pairs, check_window, map_pair_rolling,
    )
    w = check_window(b.get("window", 40))
    lag_v = b.get("lag", 1)
    if lag_v < 1:
        raise ValueError("lag must be >= 1")
    nb = int(b.get("bins", 5))
    if not 2 <= nb <= 10:
        raise ValueError("bins must be in [2, 10]")
    mp = b.get("min_periods", 10)
    bc = b.get("bias_correction", True)
    raw_w = w + lag_v

    def _fn(a: np.ndarray, bb: np.ndarray) -> float:
        n = a.size
        if n < raw_w:
            return np.nan
        x_lead = a[: n - lag_v]
        y_lag = bb[lag_v:]
        pa, pb = aligned_pairs(x_lead, y_lag)
        if pa.size < mp:
            return np.nan
        return _quantile_hist_mi(pa, pb, nb, False, bias_correction=bc)

    out = map_pair_rolling(_panel(b["x"]), _panel(b["y"]), raw_w, _fn)
    return _rebuild(b["x"], out)


def _k_cs_isotonic_residual_lagged_direction(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_ext import (
        _isotonic_residual_lagged_series,
    )
    lb = int(b.get("lookback", 20))
    if lb < 1:
        raise ValueError("cs_isotonic_residual_lagged_direction requires lookback >= 1")
    out = _isotonic_residual_lagged_series(_panel(b["y"]), _panel(b["x"]), lb)
    return _rebuild(b["y"], out)


def _k_cs_shrinkage_mahalanobis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.robust_cs import _EPS
    f5 = b.get("f5")
    shrinkage = b.get("shrinkage")
    if isinstance(f5, pl.DataFrame) or isinstance(f5, pl.Series):
        panels = [b["f1"], b["f2"], b["f3"], b["f4"], f5]
        s = 0.1 if shrinkage is None else float(shrinkage)
    else:
        panels = [b["f1"], b["f2"], b["f3"], b["f4"]]
        if f5 is not None and shrinkage is not None:
            raise ValueError(
                "shrinkage was provided both as legacy fifth positional and keyword"
            )
        s = float(f5 if f5 is not None else (0.1 if shrinkage is None else shrinkage))
    if not (0.0 <= s < 1.0):
        raise ValueError("shrinkage must be in [0, 1)")
    fv = [_panel(p) for p in panels]
    n, n_cols = fv[0].shape
    p = len(fv)
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        if valid.sum() < p + 3:
            continue
        Xv = X[valid]
        mu = Xv.mean(axis=0)
        cov = np.atleast_2d(np.cov(Xv.T))
        shrunk = (1.0 - s) * cov + s * np.diag(np.diag(cov))
        try:
            icov = np.linalg.inv(shrunk + _EPS * np.eye(p))
        except np.linalg.LinAlgError:
            continue
        d = np.sqrt(np.einsum("ij,jk,ik->i", Xv - mu, icov, Xv - mu))
        out[row, valid] = d
    return _rebuild(panels[0], out)


def _k_cs_rank_churn(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.parameter_validation import strict_integer
    from factor_engine.cleaned_operators.stateful.rotation import (
        _group_pct_ranks, _valid_label,
    )
    lk = strict_integer(b.get("lag", 5), "lag", minimum=1)
    xv = _panel(b["x"])
    group = b.get("group")
    gv = None
    if group is not None and isinstance(group, (pl.DataFrame, pl.Series)):
        if isinstance(group, pl.Series):
            group = group.to_frame()
        gv = np.empty((group.height, len(_ncols(group))), dtype=object)
        for j, c in enumerate(_ncols(group)):
            gv[:, j] = group[c].to_numpy(allow_copy=True)
    rows, cols = xv.shape
    rank_panel = np.full_like(xv, np.nan, dtype=float)
    for r in range(rows):
        g_row = gv[r] if gv is not None else None
        if g_row is None:
            vals = xv[r]
            valid = np.isfinite(vals)
            if valid.sum() < 2:
                continue
            arr = vals[valid]
            order = np.argsort(arr, kind="mergesort")
            ranks = np.empty(arr.size, dtype=float)
            start = 0
            prev = arr[order[0]]
            for i in range(1, arr.size + 1):
                if i == arr.size or arr[order[i]] != prev:
                    avg = (start + i - 1) / 2.0
                    for j in range(start, i):
                        ranks[order[j]] = avg
                    start = i
                    prev = arr[order[i]] if i < arr.size else prev
            pct = ranks / (arr.size - 1)
            pos = np.flatnonzero(valid)
            rank_panel[r, pos] = pct
        else:
            rank_panel[r] = _group_pct_ranks(xv[r], g_row)
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        if r < lk:
            continue
        g_row = gv[r] if gv is not None else None
        if g_row is None:
            cur = rank_panel[r]
            prev_r = rank_panel[r - lk]
            matched = np.isfinite(cur) & np.isfinite(prev_r)
            if not np.any(matched):
                continue
            out[r] = float(np.mean(np.abs(cur[matched] - prev_r[matched])))
        else:
            labels = {v for v in g_row if _valid_label(v)}
            for lab in labels:
                mask = np.array(
                    [(g == lab) and (gv[r - lk][i] == lab) for i, g in enumerate(g_row)],
                    dtype=bool,
                )
                cur = rank_panel[r][mask]
                prev_r = rank_panel[r - lk][mask]
                matched = np.isfinite(cur) & np.isfinite(prev_r)
                if not np.any(matched):
                    continue
                out[r][mask] = float(np.mean(np.abs(cur[matched] - prev_r[matched])))
    return _rebuild(b["x"], out)


def _k_state_episode_efficiency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.state_episode_excursion import (
        _efficiency_series,
    )
    xv = _panel(b["x"])
    sv = _panel(b["state"])
    fin = np.isfinite(sv)
    bad = fin & (sv != -1.0) & (sv != 0.0) & (sv != 1.0)
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise ValueError(
            f"state must be a signed-state panel ({{-1, 0, +1}} with NaN as "
            f"missing); found {sv[r, c]!r} at row/col ({r},{c})"
        )
    out = _efficiency_series(xv, sv)
    return _rebuild(b["x"], out)


def _k_event_local_variation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.event_interval import (
        _check_event_params, _event_mask, _local_variation_series, _max_pre_age,
    )
    w, _ = _check_event_params(b.get("window", 240))
    pre = _max_pre_age(b.get("max_pre_window_age"), w)
    values = _panel(b["event"])
    _event_mask(values.reshape(-1))
    out = _local_variation_series(values, w, pre)
    return _rebuild(b["event"], out)


def _k_event_cluster_count(b: dict) -> pl.DataFrame:
    return _event_cluster_out(b, "count")


def _k_event_cluster_mean_size(b: dict) -> pl.DataFrame:
    return _event_cluster_out(b, "mean_size")


def _event_cluster_out(b: dict, which: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    condition = b["condition"]
    cv = _panel(condition)
    fin = np.isfinite(cv)
    bad = fin & (cv != 0.0) & (cv != 1.0)
    if bad.any():
        raise ValueError("condition must be a ConditionBool ({0, 1} with NaN missing)")
    w = check_window(b.get("window", 60))
    gap = int(b.get("max_gap", 3))
    if gap < 0:
        raise ValueError("max_gap must be >= 0")

    def _fn(chunk: np.ndarray) -> float:
        truth = np.isfinite(chunk) & (chunk != 0.0)
        positions = np.flatnonzero(truth)
        if positions.size == 0:
            return 0.0 if which == "count" else np.nan
        clusters = 1
        sizes: list[float] = []
        cluster_size = 1
        for i in range(1, positions.size):
            if positions[i] - positions[i - 1] > gap:
                clusters += 1
                sizes.append(cluster_size)
                cluster_size = 1
            else:
                cluster_size += 1
        sizes.append(cluster_size)
        if which == "count":
            return float(clusters)
        return float(np.mean(sizes))

    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    out = map_rolling(cv, w, _fn)
    return _rebuild(condition, out)


def _k_ts_hurst_dfa(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.sequence_complexity import (
        _dfa_hurst, _trailing_finite_suffix,
    )
    w = _strict_int(b.get("window", 120), "window", lower=2, upper=512)
    min_s = _strict_int(b.get("min_scale", 4), "min_scale", lower=2)
    ms_raw = b.get("max_scale")
    max_s = w // 4 if ms_raw is None else _strict_int(ms_raw, "max_scale", lower=2)
    if max_s <= min_s:
        raise ValueError("max_scale must be greater than min_scale")
    if max_s > w // 2:
        raise ValueError("max_scale must be <= window // 2")
    ns = _strict_int(b.get("n_scales", 6), "n_scales", lower=3)

    def _fn(chunk: np.ndarray) -> float:
        run = _trailing_finite_suffix(chunk)
        if run.size < max_s * 2 + 4:
            return np.nan
        return _dfa_hurst(run, min_s, min(max_s, run.size // 2), ns)

    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    out = map_rolling(_panel(b["x"]), w, _fn)
    return _rebuild(b["x"], out)


def _k_ts_expected_shortfall(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_tail import _r62_quantile, _r62_win
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 60))
    quantile = float(b.get("q", 0.05))
    if not 0.0 < quantile <= 0.5:
        raise ValueError("q must be in (0, 0.5] for expected-shortfall tail semantics")
    side_kind = str(b.get("side", "lower")).lower()
    if side_kind not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    min_tail = max(2, int(b.get("min_tail_count", 5)))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    r = np.arange(rows)[:, None]
    lo, i = _r62_win(rows, w)
    valid = i <= r
    idx = np.clip(i, 0, rows - 1)
    lower = side_kind == "lower"
    for col in range(cols):
        V = xv[idx, col]
        fin = valid & np.isfinite(V)
        cnt = fin.sum(axis=1)
        if lower:
            thr = _r62_quantile(V, fin, quantile)
            tail = fin & (V <= thr[:, None])
        else:
            thr = _r62_quantile(V, fin, 1.0 - quantile)
            tail = fin & (V >= thr[:, None])
        tcnt = tail.sum(axis=1)
        tsum = np.where(tail, V, 0.0).sum(axis=1)
        ok = (cnt >= min_tail) & (tcnt >= min_tail)
        out[:, col] = np.where(ok, tsum / np.maximum(tcnt, 1), np.nan)
    return _rebuild(b["x"], out)


def _k_ts_fisher_information_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_topology import _fisher_shift_series
    r = _strict_int(b.get("recent_window", 60), "recent_window", lower=8)
    p = _strict_int(b.get("prior_window", 120), "prior_window", lower=8)
    out = _fisher_shift_series(_panel(b["x"]), r, p)
    return _rebuild(b["x"], out)


# --- wavelet / spectral family (per-row trailing fixed window) --------------
def _wavelet_rows(b: dict, stat: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.wavelet_spectral import (
        _fixed_window_anchor, _wavelet_stats,
    )
    w = _fixed_window_anchor(b.get("window", 128))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _wavelet_stats(xv[i0 : r + 1, c], w, stat)
    return _rebuild(b["x"], out)


def _k_ts_wavelet_low_frequency_ratio(b: dict) -> pl.DataFrame:
    return _wavelet_rows(b, "low")


def _k_ts_wavelet_high_frequency_ratio(b: dict) -> pl.DataFrame:
    return _wavelet_rows(b, "high")


def _k_ts_wavelet_entropy(b: dict) -> pl.DataFrame:
    return _wavelet_rows(b, "entropy")


def _k_ts_wavelet_energy_slope(b: dict) -> pl.DataFrame:
    return _wavelet_rows(b, "slope")


def _k_ts_spectral_low_frequency_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.wavelet_spectral import (
        _spectral_low_ratio,
    )
    w = int(b.get("window", 128))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _spectral_low_ratio(xv[i0 : r + 1, c], w)
    return _rebuild(b["x"], out)


# --- daily-grain intraday kernels -------------------------------------------
def _k_intraday_profile_surprise_energy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import _profile_series
    hd = int(b.get("history_days", 20))
    ns = int(b.get("n_slots", 32))
    cp = float(b.get("cap", 25.0))
    if hd < 3:
        raise ValueError("intraday_profile_surprise_energy requires history_days >= 3")
    if ns < 4:
        raise ValueError("intraday_profile_surprise_energy requires n_slots >= 4")
    if cp <= 0.0:
        raise ValueError("intraday_profile_surprise_energy requires cap > 0")
    base = b["x"]
    times = _time_numpy(base, b.get("session_tz"))
    day, row_groups = _day_groups(times)
    xv = _panel(base)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(base)):
        day_vals = [xv[idx, j] for idx in row_groups]
        vals = _profile_series(day_vals, hd, ns, cp, phase=False, max_shift=0)
        per_col[c] = list(vals)
    return _daily_out(base, times, per_col)


def _k_intraday_quantile_curve_pca_residual(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import _pca_resid_series
    w = int(b.get("window", 60))
    kk = int(b.get("k", 1))
    if w < 2 or kk < 1:
        raise ValueError("intraday_quantile_curve_pca_residual requires window >= 2, k >= 1")
    base = b["returns"]
    times = _time_numpy(base, b.get("session_tz"))
    day, row_groups = _day_groups(times)
    xv = _panel(base)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(base)):
        day_vals = [xv[idx, j] for idx in row_groups]
        vals = _pca_resid_series(day_vals, w, kk)
        per_col[c] = list(vals)
    return _daily_out(base, times, per_col)


def _k_intraday_rv_signature_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.higher_moments import _rv_signature_slope
    base = b["close"]
    times = _time_numpy(base, None)
    day, row_groups = _day_groups(times)
    xv = _panel(base)
    excs = _deg_exc()
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(base)):
        vals: list[float] = []
        for idx in row_groups:
            v = xv[idx, j]
            if int(np.sum(np.isfinite(v))) < 2:
                vals.append(np.nan)
                continue
            try:
                vals.append(float(_rv_signature_slope(v)))
            except excs:
                vals.append(np.nan)
        per_col[c] = vals
    return _daily_out(base, times, per_col)


def _k_intraday_volatility_signature_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import (
        _session_local_frame, _vol_signature_slope,
    )
    mi = int(b.get("max_interval", 32))
    if mi < 4:
        raise ValueError("intraday_volatility_signature_slope requires max_interval >= 4")
    base = b["returns"]
    times = _time_numpy(base, b.get("session_tz"))
    day, row_groups = _day_groups(times)
    xv = _panel(base)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(base)):
        vals: list[float] = []
        for idx in row_groups:
            v = xv[idx, j]
            if not np.any(np.isfinite(v)):
                vals.append(np.nan)
                continue
            try:
                vals.append(float(_vol_signature_slope(v, mi)))
            except (ValueError, ZeroDivisionError, OverflowError):
                vals.append(np.nan)
        per_col[c] = vals
    return _daily_out(base, times, per_col)


def _panel_raw(value: Any) -> np.ndarray:
    """A polars wide panel -> (rows, n_instruments) object/whatever columns.

    Unlike ``_panel`` this does NOT float-cast: it preserves string period
    labels (``period_id``) verbatim for the fiscal-period walker.
    """
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


def _walk_periods_np(xv_col: np.ndarray, pv_col: np.ndarray, fn) -> np.ndarray:
    """NumPy mirror of transforms_v2._walk_periods (single column)."""
    from collections import OrderedDict
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


def _k_fin_delta_noa(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_strict import reject_ytd_growth
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import _lag_value
    reject_ytd_growth("fin_delta_noa", b.get("flow_type"))
    ta, cash = _panel(b["total_assets"]), _panel(b["cash"])
    tl = _panel(b["total_liabilities"])
    std, ltd = _panel(b["short_term_debt"]), _panel(b["long_term_debt"])
    avg_assets = _panel(b["avg_assets"])
    pid = _panel_raw(b["period_id"])
    rows, cols = ta.shape

    def _cur(order, visible, key):
        return float(visible[key])

    def _lag1(order, visible, key):
        return _lag_value(order, visible, key, 1)

    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        noa = (
            _walk_periods_np(ta[:, j], pid[:, j], _cur)
            - _walk_periods_np(cash[:, j], pid[:, j], _cur)
            - (
                _walk_periods_np(tl[:, j], pid[:, j], _cur)
                - _walk_periods_np(std[:, j], pid[:, j], _cur)
                - _walk_periods_np(ltd[:, j], pid[:, j], _cur)
            )
        )
        noa_prev = _walk_periods_np(noa, pid[:, j], _lag1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, j] = np.where(avg_assets[:, j] != 0.0, (noa - noa_prev) / avg_assets[:, j], np.nan)
    return _rebuild(b["total_assets"], out)


def _k_ts_expectile_beta_spread(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        validate_multi_configured_history,
    )
    from factor_engine.cleaned_operators.ts_model._rolling_core import quantile_fit
    y, x = b["y"], b["x"]
    qh, ql = float(b.get("q_high", 0.9)), float(b.get("q_low", 0.1))
    if not (0.0 < ql < qh < 1.0):
        raise ValueError("q_low < q_high must hold in (0, 1)")
    w = int(b.get("window", 60))
    mp_raw = int(b.get("min_periods", 10))
    validate_multi_configured_history(w, mp_raw, 1, True, fit_lag=0)
    mp = max(mp_raw, 5 * 2)  # n_coeffs = 1 feature + intercept
    yv, xv = _panel(y), _panel(x)
    rows, cols = yv.shape

    def _coeff(q: float) -> np.ndarray:
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            ycol, xcol = yv[:, c], xv[:, c]
            for row in range(rows):
                fit_end = row
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
                beta = quantile_fit(design, vy, q)
                if beta is None:
                    continue
                out[row, c] = float(beta[1])
        return out

    return _rebuild(y, _coeff(qh) - _coeff(ql))


def _k_intra_market_model_r2_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.realized_beta import (
        _market_r2, _EPS,
    )
    close, caps = b["close"], b["free_market_cap"]
    cv = _panel(close)
    rows, cols = cv.shape
    times = _time_numpy_utc(close)
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
    uniq_days, groups = _per_day_groups(day_keys)
    cap_days = _time_numpy_utc(caps).astype("datetime64[D]")
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
    out_days: list[np.datetime64] = []
    vals: dict[str, list[float]] = {c: [] for c in names}
    for k, d in enumerate(uniq_days):
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
                vals[c].append(float(_market_r2(rr, mm)))
            except (ValueError, ZeroDivisionError, OverflowError):
                vals[c].append(np.nan)
    return _daily_panel(close, np.asarray(out_days), vals)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ts_dmd_return_dominant_growth_rate": _k_ts_dmd_return_dominant_growth_rate,
    "ts_cross_quantilogram": _k_ts_cross_quantilogram,
    "intraday_profile_surprise_energy": _k_intraday_profile_surprise_energy,
    "ashare_limit_up_streak": _k_ashare_limit_up_streak,
    "ts_spectral_quality_factor": _k_ts_spectral_quality_factor,
    "ts_semivariance_balance": _k_ts_semivariance_balance,
    "ts_persistence_entropy_h0": _k_ts_persistence_entropy_h0,
    "ts_recurrence_determinism": _k_ts_recurrence_determinism,
    "ts_har_rv_next_vol_forecast": _k_ts_har_rv_next_vol_forecast,
    "intraday_quantile_curve_pca_residual": _k_intraday_quantile_curve_pca_residual,
    "cs_shrinkage_mahalanobis": _k_cs_shrinkage_mahalanobis,
    "ts_wavelet_entropy": _k_ts_wavelet_entropy,
    "ts_lagged_mutual_information": _k_ts_lagged_mutual_information,
    "cs_isotonic_residual_lagged_direction": _k_cs_isotonic_residual_lagged_direction,
    "ts_pettitt_change_score": _k_ts_pettitt_change_score,
    "ts_path_signature_area": _k_ts_path_signature_area,
    "ts_quantile_crossing_spectral_concentration": _k_ts_quantile_crossing_spectral_concentration,
    "intraday_rv_signature_slope": _k_intraday_rv_signature_slope,
    "ts_kalman_level": _k_ts_kalman_level,
    "cs_rank_churn": _k_cs_rank_churn,
    "ts_wavelet_low_frequency_ratio": _k_ts_wavelet_low_frequency_ratio,
    "ts_dmd_return_dominant_frequency": _k_ts_dmd_return_dominant_frequency,
    "ashare_limit_touch_count": _k_ashare_limit_touch_count,
    "ts_spectral_low_frequency_ratio": _k_ts_spectral_low_frequency_ratio,
    "state_episode_efficiency": _k_state_episode_efficiency,
    "event_local_variation": _k_event_local_variation,
    "ts_wavelet_high_frequency_ratio": _k_ts_wavelet_high_frequency_ratio,
    "ts_ssa_reconstruction_residual": _k_ts_ssa_reconstruction_residual,
    "event_cluster_count": _k_event_cluster_count,
    "event_cluster_mean_size": _k_event_cluster_mean_size,
    "ashare_days_since_limit_up": _k_ashare_days_since_limit_up,
    "ts_recurrence_rate": _k_ts_recurrence_rate,
    "ts_hurst_dfa": _k_ts_hurst_dfa,
    "ts_garch_vol_surprise": _k_ts_garch_vol_surprise,
    "ts_expected_shortfall": _k_ts_expected_shortfall,
    "ts_wavelet_energy_slope": _k_ts_wavelet_energy_slope,
    "ts_persistence_entropy_h1": _k_ts_persistence_entropy_h1,
    "ts_fisher_information_shift": _k_ts_fisher_information_shift,
    "intraday_volatility_signature_slope": _k_intraday_volatility_signature_slope,
    "ashare_limit_up_volume_ratio": _k_ashare_limit_up_volume_ratio,
    "fin_delta_noa": _k_fin_delta_noa,
    "intra_market_model_r2_ex_self": _k_intra_market_model_r2_ex_self,
    "ts_expectile_beta_spread": _k_ts_expectile_beta_spread,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch3)
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
            bound = dict(zip(self.metadata.param_names, args))
            bound.update(kwargs)
        except Exception as e:
            # Only a missing-OPTIONAL-panel rejection (the pandas binder accepts
            # None where the polars binder demands a panel) falls through to raw
            # binding, where the kernel applies the same fail-closed contract as
            # the pandas authority.  Relational / parameter-domain errors keep
            # the binder's raise (same contract as the UDF).
            if "required" not in str(e).lower():
                raise
            bound = dict(zip(self.metadata.param_names, args))
            bound.update(kwargs)
        return self._kernel_fn(bound)

    # Same binding contract under direct kernel-style invocation.
    _calculate_series = calculate


def register_r68_native_batch4() -> list[str]:
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
                (canonical + ":pandas-authority-parity:r68b4").encode()
            ).hexdigest(),
            notes=(
                "R68 batch4 genuine Polars backend: pure pl.Expr kernels or "
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


__all__ = ["register_r68_native_batch4"]

register_r68_native_batch4()
