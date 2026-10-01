# -*- coding: utf-8 -*-
"""R68 batch9: genuine-Polars backends for the 45 pandas-delegate canonicals.

Same protocol as ``r68_native_batch4/5/6``: numpy authority helpers called
directly on ``pl -> numpy`` column panels, plus numpy ports for the kernels
whose authority is a pandas-Series pipeline.  **No pandas DataFrame is
constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``, no
``iterrows``).  Daily-grain intraday kernels build their own daily panel from
the time column exactly like ``rolling_pack._pl_rebuild_intraday_result``.
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

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch9"

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


def _time_numpy(frame: pl.DataFrame, session_tz: Any) -> np.ndarray | None:
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


def _daily_out(base: pl.DataFrame, day: np.ndarray, arr: np.ndarray) -> pl.DataFrame:
    day_ts = day.astype("datetime64[ns]")
    series = [pl.Series("date", day_ts)]
    for j, c in enumerate(_ncols(base)):
        series.append(pl.Series(c, np.ascontiguousarray(arr[:, j]), dtype=pl.Float64))
    return pl.DataFrame(series)


# ---------------------------------------------------------------------------
# shared numpy micro-kernels (pandas-semantics ports)
# ---------------------------------------------------------------------------
def _np_shift(a: np.ndarray, n: int, fill: float = np.nan) -> np.ndarray:
    out = np.full_like(a, fill, dtype=float)
    if n < a.shape[0]:
        out[n:] = a[: a.shape[0] - n]
    return out


def _np_shift_bool(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(a.shape, dtype=bool)
    if n < a.shape[0]:
        out[n:] = a[: a.shape[0] - n]
    return out


def _np_rolling_mean(a: np.ndarray, w: int, mp: int) -> np.ndarray:
    """Trailing mean over the past ``w`` rows; requires >= mp non-NaN."""
    rows = a.shape[0]
    out = np.full(a.shape, np.nan, dtype=float)
    for j in range(a.shape[1]):
        col = a[:, j]
        csum = np.cumsum(np.where(np.isfinite(col), col, 0.0))
        ccnt = np.cumsum(np.isfinite(col).astype(np.int64))
        for r in range(rows):
            lo = max(0, r - w + 1)
            cnt = int(ccnt[r] - (ccnt[lo - 1] if lo > 0 else 0))
            if cnt < mp:
                continue
            s = float(csum[r] - (csum[lo - 1] if lo > 0 else 0.0))
            out[r, j] = s / cnt
    return out


def _np_ewm_mean(a: np.ndarray, alpha: float, mp: int) -> np.ndarray:
    """pandas ``ewm(alpha, adjust=False, min_periods=mp).mean()`` port.

    pandas (adjust=False, ignore_na=False) weights an observation arriving
    after ``g`` NaN rows at ``(1-alpha)**(g+1)`` against the new value's
    ``alpha`` — weights are based on absolute positions.
    """
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        y = np.nan
        cnt = 0
        gap = 0
        col = a[:, j]
        for i in range(rows):
            v = col[i]
            if np.isfinite(v):
                cnt += 1
                if cnt == 1:
                    y = float(v)
                else:
                    w = (1.0 - alpha) ** (gap + 1)
                    y = (w * y + alpha * float(v)) / (w + alpha)
                gap = 0
            else:
                gap += 1
            if cnt >= mp:
                out[i, j] = y
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(den) & (np.abs(den) > 1e-12), num / den, np.nan)
    return out


def _zero_to_nan(d: np.ndarray) -> np.ndarray:
    return np.where(d == 0.0, np.nan, d)


# ===========================================================================
# kernels
# ===========================================================================
def _k_ts_variogram_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    from factor_engine.cleaned_operators.sequence_complexity import (
        strict_int, _vec_column_variogram_slope,
    )
    w = check_window(b.get("window", 60))
    ml = strict_int(b.get("max_lag", 10), "max_lag", lower=1, upper=30)
    mvl = strict_int(b.get("min_valid_lags", 4), "min_valid_lags", lower=2)
    x = _panel(b["x"])
    return _rebuild(b["x"], _vec_column_variogram_slope(x, w, ml, mvl))


def _k_ts_multifractal_curvature(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.multifractal import (
        _check_window, _curvature_series,
    )
    w = _check_window(b.get("window", 120))
    return _rebuild(b["x"], _curvature_series(_panel(b["x"]), w))


def _k_ts_qn_scale(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_scale import _vec_qn_scale
    w = int(b.get("window", 60))
    if w < 2:
        raise ValueError("ts_qn_scale requires window >= 2")
    mp = max(2, int(b.get("min_periods", 8)))
    x = _panel(b["x"])
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _vec_qn_scale(x[:, c], w, mp)
    return _rebuild(b["x"], out)


def _k_ts_roughness(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    from factor_engine.cleaned_operators.alpha_language_shape import (
        _EPS, _trailing_contiguous,
    )
    w = check_window(b.get("window", 20))
    mp = max(3, int(b.get("min_periods", 3)))
    xv = _panel(b["x"])

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        if v.size < mp:
            return np.nan
        d = np.diff(v)
        if d.size < 2:
            return np.nan
        d2 = np.diff(d)
        s2 = float(np.sum(d2 * d2))
        s1 = float(np.sum(d * d))
        return s2 / (s1 + _EPS)

    return _rebuild(b["x"], map_rolling(xv, w, _fn))


def _k_ts_interval_exploration_efficiency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.interval_geometry import (
        _exploration_efficiency_series,
    )
    out = _exploration_efficiency_series(
        _panel(b["high"]), _panel(b["low"]), _panel(b["close"]),
        int(b.get("window", 20)),
    )
    return _rebuild(b["high"], out)


def _k_cs_weighted_percentile_rank(b: dict) -> pl.DataFrame:
    xv = _panel(b["x"])
    wv = _panel(b["weight"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        xr = xv[r]
        wr = wv[r]
        if np.any(np.isfinite(wr) & (wr < 0.0)):
            continue
        valid = np.isfinite(xr) & np.isfinite(wr)
        valid_idx = np.flatnonzero(valid)
        raw_weights = wr[valid]
        weight_scale = float(np.max(raw_weights)) if raw_weights.size else 0.0
        if not np.isfinite(weight_scale) or weight_scale <= 0.0:
            continue
        scaled_weights = raw_weights / weight_scale
        total = float(np.sum(scaled_weights))
        if not np.isfinite(total) or total <= 0.0:
            continue
        order = np.argsort(xr[valid], kind="stable")
        xs = xr[valid][order]
        ws = scaled_weights[order]
        n = xs.size
        lower_cum = 0.0
        rank_out = np.full(n, np.nan)
        i = 0
        while i < n:
            j = i
            tie_w = 0.0
            while j < n and xs[j] == xs[i]:
                tie_w += ws[j]
                j += 1
            for t in range(i, j):
                rank_out[t] = (lower_cum + 0.5 * tie_w) / total
            lower_cum += tie_w
            i = j
        unsorted = np.empty(n, dtype=float)
        unsorted[order] = rank_out
        out[r, valid_idx] = unsorted
    return _rebuild(b["x"], out)


def _k_ts_distance_cov(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import (
        aligned_pairs, check_window, map_pair_rolling,
    )
    from factor_engine.cleaned_operators.nonlinear_dependence import _distance_corr
    w = check_window(b.get("window", 40))
    mp = b.get("min_periods", 10)
    xv = _panel(b["x"])
    yv = _panel(b["y"])

    def _fn(a: np.ndarray, bb: np.ndarray) -> float:
        pa, pb = aligned_pairs(a, bb)
        if pa.size < mp:
            return np.nan
        return _distance_corr(pa, pb)[1]

    return _rebuild(b["x"], map_pair_rolling(xv, yv, w, _fn))


def _k_spectral_trend_share(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    from factor_engine.cleaned_operators.technical.indicators_v2 import (
        _EPS, _pi, _rfft_power_excluding_dc,
    )
    w = _pi(b.get("window", 60), "window", 4)
    tb = _pi(b.get("trend_bins", 2), "trend_bins", 1)
    close = _panel(b["close"])
    pos = np.where(close > 0.0, close, np.nan)

    def _share(a: np.ndarray) -> float:
        power = _rfft_power_excluding_dc(a)
        total = power.sum()
        if total <= _EPS:
            return 0.0
        nb = min(tb, power.size)
        return float(power[:nb].sum() / total)

    def _fn(chunk: np.ndarray) -> float:
        if chunk.size < w or not np.all(np.isfinite(chunk)):
            return np.nan
        return _share(chunk)

    return _rebuild(b["close"], map_rolling(pos, w, _fn))


def _k_ts_run_efficiency(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    from factor_engine.cleaned_operators.alpha_language_state import (
        _EPS, _run_windows, _state_series,
    )
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
                buf.clear()
                run_len = 0
                prev_state = None
                continue
            if not np.isfinite(xv[row, col]):
                buf.clear()
                run_len = 0
                prev_state = None
                continue
            cur = s_val[row, col]
            if cur == 0.0:
                buf.clear()
                run_len = 0
                prev_state = None
                out[row, col] = 0.0
                continue
            if prev_state is not None and cur == prev_state:
                buf.append(xv[row, col])
                run_len = run_len + 1
            else:
                buf = deque([xv[row, col]], maxlen=w)
                run_len = 1
            prev_state = cur
            if run_len < mp:
                out[row, col] = np.nan
                continue
            run_sum = float(sum(buf))
            run_abs = float(sum(abs(v) for v in buf))
            out[row, col] = abs(run_sum) / (run_abs + _EPS)
    return _rebuild(b["x"], out)


def _k_ts_interval_overlap_connected_component_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.interval_geometry import (
        _overlap_component_ratio_series,
    )
    out = _overlap_component_ratio_series(
        _panel(b["low"]), _panel(b["high"]), int(b.get("window", 20)),
    )
    return _rebuild(b["low"], out)


def _k_ts_autocorrelation_time(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.memory_ext import (
        _check_window, _autocorrelation_time_series,
    )
    w = _check_window(b.get("window", 120))
    ml = int(b.get("max_lag", 20))
    if ml < 1:
        raise ValueError("max_lag must be >= 1")
    if ml >= w:
        raise ValueError("max_lag must be < window")
    return _rebuild(b["x"], _autocorrelation_time_series(_panel(b["x"]), w, ml))


def _k_ts_binned_response_monotonicity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    from factor_engine.cleaned_operators.binned_response import (
        _monotonicity_series, _strict_int,
    )
    w = check_window(b.get("window", 120))
    nb = _strict_int(b.get("bins", 5), "bins", 3)
    if nb not in (3, 5):
        raise ValueError("bins must be one of {3, 5}")
    mpb = _strict_int(b.get("min_per_bin", 3), "min_per_bin", 3)
    out = _monotonicity_series(
        _panel(b["y"]), _panel(b["x"]), w, nb, mpb,
    )
    return _rebuild(b["y"], out)


def _k_ts_har_rv_next_var_forecast(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import _har_rv
    w = int(b.get("window", 120))
    xv = _panel(b["rv"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _har_rv(xv[: row + 1, col], w, "var_forecast")
    return _rebuild(b["rv"], out)


def _k_ts_har_from_return_next_vol(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.volatility import _har_from_return
    w = int(b.get("window", 120))
    xv = _panel(b["ret"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _har_from_return(xv[: row + 1, col], w, "forecast")
    return _rebuild(b["ret"], out)


def _k_ts_permutation_transition_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window
    from factor_engine.cleaned_operators.sequence_complexity import (
        strict_int, _vec_column_permutation_transition_entropy,
    )
    w = check_window(b.get("window", 60))
    ord_ = strict_int(b.get("order", 3), "order", lower=2, upper=6)
    dl = strict_int(b.get("delay", 1), "delay", lower=1)
    norm = bool(b.get("normalize", True))
    out = _vec_column_permutation_transition_entropy(
        _panel(b["x"]), w, ord_, dl, norm,
    )
    return _rebuild(b["x"], out)


def _k_ashare_one_price_limit_streak(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _consecutive_streak, _one_price_mask, _tolerance, _tradeable,
    )
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    side_kind = str(b.get("side", "up")).lower()
    if side_kind not in {"up", "down"}:
        raise ValueError("side must be 'up' or 'down'")
    ov = _panel(b["open"])
    hv = _panel(b["high"])
    lv = _panel(b["low"])
    cv = _panel(b["close"])
    vv = _panel(b["valid_trade"])
    rows, cols = cv.shape
    limit = _panel(b["high_limit"]) if side_kind == "up" else _panel(b["low_limit"])
    condition = _one_price_mask(ov, hv, lv, cv, limit, tol, rows, cols)
    for r in range(rows):
        for c in range(cols):
            if not _tradeable(vv, r, c):
                condition[r, c] = np.nan
    tradeable = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            tradeable[r, c] = _tradeable(vv, r, c)
    return _rebuild(b["close"], _consecutive_streak(condition, tradeable, rows, cols))


def _k_ts_forbidden_ordinal_pattern_excess(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.complexity_ext import (
        _MIN_EMBEDDINGS, _check_complexity_params, _forbidden_ordinal_ratio_series,
    )
    window = int(b.get("window", 120))
    order = int(b.get("order", 3))
    delay = int(b.get("delay", 1))
    min_embeddings = int(b.get("min_embeddings", _MIN_EMBEDDINGS))
    _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
    out = _forbidden_ordinal_ratio_series(
        _panel(b["x"]), window, order, delay, min_embeddings, mode="excess",
    )
    return _rebuild(b["x"], out)


def _k_ts_forbidden_ordinal_pattern_signed_excess(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.complexity_ext import (
        _MIN_EMBEDDINGS, _check_complexity_params, _forbidden_ordinal_ratio_series,
    )
    window = int(b.get("window", 120))
    order = int(b.get("order", 3))
    delay = int(b.get("delay", 1))
    min_embeddings = int(b.get("min_embeddings", _MIN_EMBEDDINGS))
    _check_complexity_params(window, order=order, delay=delay, min_embeddings=min_embeddings)
    out = _forbidden_ordinal_ratio_series(
        _panel(b["x"]), window, order, delay, min_embeddings, mode="signed_excess",
    )
    return _rebuild(b["x"], out)


def _k_fin_net_debt_issuance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        _default_require_parseable, _lag_value, _period_insert, _period_key,
    )
    from factor_engine.cleaned_operators.fiscal_strict import period_ordinal
    del period_ordinal
    s = _panel(b["short_term_loan"])
    l = _panel(b["long_term_loan"])
    bb = _panel(b["bonds_payable"])
    aa = _panel(b["avg_assets"])
    pid = b["period_id"]
    pcols = _ncols(pid)
    rows, cols = s.shape
    pv: list = [[None] * cols for _ in range(rows)]
    for j, c in enumerate(pcols):
        col_vals = pid[c].to_list()
        for i in range(rows):
            pv[i][j] = col_vals[i]
    x = s + l + bb
    delta = np.full((rows, cols), np.nan, dtype=float)
    require_parseable = _default_require_parseable()
    for c in range(cols):
        order: list = []
        visible: OrderedDict = OrderedDict()
        for i in range(rows):
            key = _period_key(pv[i][c])
            value = x[i, c]
            if key is not None and np.isfinite(value):
                if key not in visible:
                    _period_insert(order, key, require_parseable=require_parseable)
                visible[key] = float(value)
            if key is None or key not in visible:
                continue
            try:
                delta[i, c] = float(visible[key]) - _lag_value(order, visible, key, 1)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                delta[i, c] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(aa == 0.0, np.nan, delta / aa)
    return _rebuild(b["short_term_loan"], out)


def _k_ts_rank_if(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.base import strict_int_param
    w = strict_int_param(b.get("window", 20), "window", lower=2)
    mp = strict_int_param(b.get("min_periods", 5), "min_periods", lower=2)
    xv = _panel(b["x"])
    cv = _panel(b["condition"])
    finite_cv = np.isfinite(cv)
    bad = finite_cv & (cv != 0.0) & (cv != 1.0)
    if np.any(bad):
        raise ValueError(
            "condition must be a ConditionBool (values in {0, 1} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )
    truth = finite_cv & (cv == 1.0)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            xt = xv[row, col]
            if not np.isfinite(xt):
                continue
            lo = max(0, row - w + 1)
            sel = xv[lo: row + 1, col][truth[lo: row + 1, col]]
            sel = sel[np.isfinite(sel)]
            if sel.size < mp:
                continue
            less = float(np.sum(sel < xt))
            equal = float(np.sum(sel == xt))
            out[row, col] = (less + 0.5 * equal) / sel.size
    return _rebuild(b["x"], out)


def _k_ts_regression_tstat(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.daily_panel import (
        _positive_int, _slope_tstat,
    )
    w = _positive_int(b.get("window", 60), "window")
    mp_raw = b.get("min_periods")
    mp = w if mp_raw is None else _positive_int(mp_raw, "min_periods")
    add_intercept = bool(b.get("add_intercept", True))
    yv = _panel(b["y"])
    xv = _panel(b["x"])
    rows, cols = yv.shape
    out = np.full_like(yv, np.nan)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            mask = np.isfinite(yv[start: row + 1, col]) & np.isfinite(
                xv[start: row + 1, col]
            )
            if int(mask.sum()) >= mp:
                out[row, col] = _slope_tstat(
                    yv[start: row + 1, col],
                    xv[start: row + 1, col],
                    bool(add_intercept),
                )
    return _rebuild(b["y"], out)


def _k_cs_actual_lof_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.robust_cs import _EPS, _lof_row
    kk = max(2, int(b.get("k", 20)))
    feats = [_panel(b[n]) for n in ("f1", "f2", "f3", "f4") if b.get(n) is not None]
    n, n_cols = feats[0].shape
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in feats])
        valid = np.all(np.isfinite(X), axis=1)
        nv = int(valid.sum())
        eff_k = max(1, min(kk, nv - 1))
        if nv < eff_k + 2:
            continue
        Xv = X[valid]
        sd = np.std(Xv, axis=0)
        Xn = Xv / np.where(sd > _EPS, sd, 1.0)
        out[row, valid] = _lof_row(Xn, eff_k)
    return _rebuild(b["f1"], out)


def _k_fin_core_earnings_ratio(b: dict) -> pl.DataFrame:
    op = _panel(b["operating_profit"])
    inv = _panel(b["investment_income"])
    fv = _panel(b["fair_value_income"])
    asset = _panel(b["asset_deal_income"])
    other = _panel(b["other_earnings"])
    revenue = _panel(b["revenue"])
    core = op - inv - fv - asset - other
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(revenue == 0.0, np.nan, core / revenue)
    return _rebuild(b["operating_profit"], out)


def _k_fin_noncore_income_ratio(b: dict) -> pl.DataFrame:
    inv = _panel(b["investment_income"])
    fv = _panel(b["fair_value_income"])
    asset = _panel(b["asset_deal_income"])
    other = _panel(b["other_earnings"])
    nonop_rev = _panel(b["non_operating_revenue"])
    nonop_exp = _panel(b["non_operating_expense"])
    total_profit = _panel(b["total_profit"])
    noncore = inv + fv + asset + other + nonop_rev - nonop_exp
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(total_profit) == 0.0, np.nan, noncore / np.abs(total_profit))
    return _rebuild(b["investment_income"], out)


def _limit_or_panel(b: dict, name: str, default: float) -> tuple[float, np.ndarray | None]:
    value = b.get(name)
    if value is None:
        return float(default), None
    if isinstance(value, pl.DataFrame):
        arr = _panel(value)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        return float("nan"), arr
    f = float(value)
    if not np.isfinite(f) or f < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return f, None


def _k_state_slew_limit(b: dict) -> pl.DataFrame:
    lim_scalar, lim_panel = _limit_or_panel(b, "limit", 0.01)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        prev = np.nan
        for row in range(rows):
            target = xv[row, col]
            lim = float(lim_panel[row, col]) if lim_panel is not None else lim_scalar
            if not np.isfinite(target) or not np.isfinite(lim) or lim < 0.0:
                out[row, col] = np.nan
                prev = np.nan
                continue
            if not np.isfinite(prev):
                prev = target
                out[row, col] = prev
                continue
            delta = float(target) - prev
            prev = prev + float(np.clip(delta, -lim, lim))
            out[row, col] = prev
    return _rebuild(b["x"], out)


def _k_state_deadband(b: dict) -> pl.DataFrame:
    band_scalar, band_panel = _limit_or_panel(b, "band", 0.0)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        prev = np.nan
        for row in range(rows):
            target = xv[row, col]
            band = float(band_panel[row, col]) if band_panel is not None else band_scalar
            if not np.isfinite(target) or not np.isfinite(band) or band < 0.0:
                out[row, col] = np.nan
                prev = np.nan
                continue
            if not np.isfinite(prev):
                prev = target
                out[row, col] = prev
                continue
            d = float(target) - prev
            if abs(d) <= band:
                out[row, col] = prev
                continue
            prev = prev + float(np.sign(d) * (abs(d) - band))
            out[row, col] = prev
    return _rebuild(b["x"], out)


def _beta_daily_numpy(
    close: pl.DataFrame,
    weights: pl.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    ex_self: bool,
) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import DataDegeneracy, minute_of_day

    times = _time_numpy(close, None)
    cv = _panel(close)
    rows, cols = cv.shape
    if times is None:
        raise ValueError("intraday close panel requires a time column")
    day = times.astype("datetime64[D]")
    # per-stock minute log returns, hard-broken at day boundaries (P0)
    rets = np.full_like(cv, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        adjacent = (
            np.isfinite(cv[1:, :]) & np.isfinite(cv[:-1, :])
            & (cv[1:, :] > 0.0) & (cv[:-1, :] > 0.0)
        )
        rets[1:, :] = np.where(adjacent, np.log(cv[1:, :] / cv[:-1, :]), np.nan)
    if rows > 1:
        day_change = np.zeros(rows, dtype=bool)
        day_change[1:] = day[1:] != day[:-1]
        rets[day_change, :] = np.nan
    # broadcast the daily cap panel onto the minute grid by date
    wtimes = _time_numpy(weights, None)
    wv = _panel(weights)
    wcols = _ncols(weights)
    uniq_days = np.unique(day)
    wdays = wtimes.astype("datetime64[D]") if wtimes is not None else None
    w_bc = np.full((rows, cols), np.nan, dtype=float)
    for j, c in enumerate(_ncols(close)):
        if c not in wcols:
            continue
        wj = wv[:, wcols.index(c)]
        day_map = {d: i for i, d in enumerate(wdays)} if wdays is not None else {}
        for i, d in enumerate(uniq_days):
            wval = wj[day_map[d]] if d in day_map else np.nan
            w_bc[day == d, j] = wval
    w_bc = np.where(np.isfinite(w_bc) & (w_bc > 0.0), w_bc, np.nan)
    w_ret = np.where(np.isfinite(rets), w_bc, np.nan)
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(np.isfinite(rets) & np.isfinite(w_bc), rets * w_bc, 0.0), axis=1)
        den = np.nansum(np.where(np.isfinite(w_ret), w_ret, 0.0), axis=1)
        mkt = _safe_div(num, den)
    all_days = np.unique(day)
    out = np.full((all_days.size, cols), np.nan, dtype=float)
    for j, c in enumerate(_ncols(close)):
        if ex_self:
            valid = np.isfinite(rets) & np.isfinite(w_bc) & (w_bc > 0.0)
            contributions = np.where(valid, rets * w_bc, 0.0)
            effective = np.where(valid, w_bc, 0.0)
            with np.errstate(invalid="ignore"):
                num_j = contributions.sum(axis=1) - contributions[:, j]
                den_j = effective.sum(axis=1) - effective[:, j]
            m = _safe_div(num_j, den_j)
        else:
            m = mkt
        day_of_row = day
        for di, d in enumerate(all_days):
            sel = (day_of_row == d) & np.isfinite(m)
            if not sel.any():
                out[di, j] = np.nan
                continue
            rr = rets[sel, j]
            mm = m[sel]
            if int(np.sum(np.isfinite(mm))) < 2:
                out[di, j] = np.nan
                continue
            try:
                out[di, j] = float(fn(rr, mm))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                out[di, j] = np.nan
    del minute_of_day
    return _daily_out(close, all_days, out)


def _k_intra_idiosyncratic_skewness_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.realized_beta import _idio_skewness
    return _beta_daily_numpy(
        b["close"], b["free_market_cap"], _idio_skewness, ex_self=True,
    )


def _k_intra_idiosyncratic_kurtosis_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.realized_beta import _idio_kurtosis
    return _beta_daily_numpy(
        b["close"], b["free_market_cap"], _idio_kurtosis, ex_self=True,
    )


def _k_ts_pseudocount_sample_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import (
        _pseudocount_sample_entropy,
    )
    m = int(b.get("m", 2))
    r = float(b.get("r", 0.2))
    w = int(b.get("window", 200))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _pseudocount_sample_entropy(xv[: row + 1, col], m, r, w)
    return _rebuild(b["x"], out)


def _k_ts_two_state_regime_probability(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _regime_filter
    w = int(b.get("window", 120))
    tp = float(b.get("transition_prob", 0.05))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = _regime_filter(xv[: row + 1, col], w, "prob", tp)
    return _rebuild(b["x"], out)


def _k_ashare_limit_open_up_streak(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _open_limit_streak, _tolerance,
    )
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    ov = _panel(b["open"])
    limit = _panel(b["high_limit"])
    vv = _panel(b["valid_trade"])
    return _rebuild(b["open"], _open_limit_streak(ov, limit, vv, tol))


def _k_ashare_limit_open_down_streak(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _consecutive_streak, _tolerance, _tradeable,
    )
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    ov = _panel(b["open"])
    limit = _panel(b["low_limit"])
    vv = _panel(b["valid_trade"])
    rows, cols = ov.shape
    condition = np.full((rows, cols), np.nan, dtype=float)
    tradeable = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            tradeable[r, c] = _tradeable(vv, r, c)
            if not tradeable[r, c]:
                continue
            if np.isfinite(ov[r, c]) and np.isfinite(limit[r, c]):
                condition[r, c] = 1.0 if ov[r, c] <= limit[r, c] * (1.0 + tol) else 0.0
    return _rebuild(b["open"], _consecutive_streak(condition, tradeable, rows, cols))


def _k_cs_wls_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.daily_panel import _ols_residual
    yv = _panel(b["y"])
    xv = _panel(b["x"])
    wv = _panel(b["weight"])
    add_intercept = bool(b.get("add_intercept", True))
    min_obs = b.get("min_obs", 5)
    rows = yv.shape[0]
    out = np.full(yv.shape, np.nan, dtype=float)
    for row in range(rows):
        out[row] = _ols_residual(
            yv[row],
            [xv[row]],
            add_intercept=add_intercept,
            min_obs=min_obs,
            weights=wv[row],
        )
    return _rebuild(b["y"], out)


def _k_intra_abs_return_profile_cosine(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import _EPS, minute_of_day
    w = max(2, int(b.get("window", 20)))
    session_tz = b.get("session_tz")
    times = _time_numpy(b["close"], session_tz)
    cv = _panel(b["close"])
    rows, cols = cv.shape
    if times is None:
        raise ValueError("intraday close panel requires a time column")
    day = times.astype("datetime64[D]")
    minutes = minute_of_day(times)
    rets = np.full_like(cv, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, rows):
            if day[i] == day[i - 1]:
                rets[i] = np.log(cv[i] / cv[i - 1])
    rets = np.abs(rets)
    all_days = np.unique(day)
    mp = max(2, w // 2)
    col_scores: list[dict[np.datetime64, float]] = []
    for j in range(cols):
        col = rets[:, j]
        # slot matrix: day x minute-of-day (mean of non-NaN, pandas pivot_table)
        slot_of_row = minutes
        mat = {}
        for i in range(rows):
            if not np.isfinite(col[i]):
                continue
            key = (day[i], int(slot_of_row[i]))
            s, cnt = mat.get(key, (0.0, 0))
            mat[key] = (s + col[i], cnt + 1)
        slots = sorted({k[1] for k in mat})
        days_j = sorted({k[0] for k in mat})
        scores: dict[np.datetime64, float] = {}
        if days_j:
            m = np.full((len(days_j), len(slots)), np.nan, dtype=float)
            slot_idx = {s: t for t, s in enumerate(slots)}
            day_idx = {d: t for t, d in enumerate(days_j)}
            for (d, s), (sm, cnt) in mat.items():
                m[day_idx[d], slot_idx[s]] = sm / cnt
            # hist = mat.shift(1).rolling(w, min_periods=mp).mean() per slot
            hist = np.full_like(m, np.nan, dtype=float)
            for t in range(m.shape[0]):
                window = m[max(0, t - w): t]  # past w days, never today
                for s in range(m.shape[1]):
                    vals = window[:, s]
                    vals = vals[np.isfinite(vals)]
                    if vals.size >= mp:
                        hist[t, s] = float(vals.mean())
            for t, d in enumerate(days_j):
                a = m[t]
                bb = hist[t]
                valid = np.isfinite(a) & np.isfinite(bb)
                if valid.sum() < 2:
                    continue
                va, vb = a[valid], bb[valid]
                na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                if na <= _EPS or nb <= _EPS:
                    continue
                scores[d] = float(np.dot(va, vb) / (na * nb))
        col_scores.append(scores)
    # output index = union of days present in any instrument's slot matrix
    # (mirrors pd.DataFrame(out).sort_index() of the pandas authority)
    union_days = sorted({d for scores in col_scores for d in scores})
    if union_days:
        out = np.full((len(union_days), cols), np.nan, dtype=float)
        for j, scores in enumerate(col_scores):
            for t, d in enumerate(union_days):
                out[t, j] = scores.get(d, np.nan)
        return _daily_out(b["close"], np.asarray(union_days), out)
    empty = np.full((0, cols), np.nan, dtype=float)
    return _daily_out(b["close"], np.array([], dtype="datetime64[D]"), empty)


def _k_fiscal_perpetual_inventory(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_event_ops import (
        _nonnegative, _policy, _positive,
    )
    from factor_engine.cleaned_operators.fiscal_strict import period_ordinal

    def _finite(v: float) -> bool:
        return bool(np.isfinite(v))

    flow = _panel(b["flow"])
    pid = b["period_id"]
    pcols = _ncols(pid)
    rows, cols = flow.shape
    raw_periods: list = [[None] * cols for _ in range(rows)]
    for j, c in enumerate(pcols):
        col_vals = pid[c].to_list()
        for i in range(rows):
            raw_periods[i][j] = col_vals[i]
    depreciation = float(b.get("depreciation", 0.15))
    periods_per_year = _positive(b.get("periods_per_year", 4), "periods_per_year")
    warmup_periods = _nonnegative(b.get("warmup_periods", 8), "warmup_periods")
    require_consecutive = bool(b.get("require_consecutive", True))
    revision_policy = b.get("revision_policy", "latest_available")
    if not np.isfinite(depreciation) or not 0 <= depreciation < 1:
        raise ValueError("depreciation must satisfy 0 <= depreciation < 1")
    retention = (1.0 - depreciation) ** (1.0 / periods_per_year)
    policy = _policy(revision_policy)
    ordinal_values = np.full((rows, cols), np.nan, dtype=float)
    for i in range(rows):
        for j in range(cols):
            ordinal = period_ordinal(raw_periods[i][j])
            if ordinal is not None:
                ordinal_values[i, j] = ordinal
    state: list[dict[int, float]] = [dict() for _ in range(cols)]
    first_seen: list[set[int]] = [set() for _ in range(cols)]
    snapshots: list[list[dict[int, float]]] = []
    for i in range(rows):
        for j in range(cols):
            ordinal = ordinal_values[i, j]
            value = flow[i, j]
            if not np.isfinite(ordinal) or not _finite(value):
                continue
            key = int(ordinal)
            if policy == "latest_available" or key not in first_seen[j]:
                state[j][key] = float(value)
            first_seen[j].add(key)
        snapshots.append([dict(sorted(column.items())) for column in state])

    def history(i: int, j: int) -> list[tuple[int, float]]:
        current = ordinal_values[i, j]
        if not np.isfinite(current):
            return []
        events = snapshots[i][j]
        hist = [
            (int(k), float(v)) for k, v in events.items()
            if k <= int(current) and _finite(v)
        ]
        if not hist:
            return []
        hist.sort()
        if require_consecutive and hist:
            contiguous: list[tuple[int, float]] = [hist[-1]]
            for item in reversed(hist[:-1]):
                if contiguous[0][0] - item[0] != 1:
                    break
                contiguous.insert(0, item)
            hist = contiguous
        return hist

    out = np.full((rows, cols), np.nan, dtype=float)
    for i in range(rows):
        for j in range(cols):
            hist = history(i, j)
            if len(hist) <= warmup_periods:
                continue
            stock = 0.0
            for _, value in hist:
                stock = retention * stock + value
            out[i, j] = stock
    return _rebuild(b["flow"], out)


def _k_ts_max_chord_excursion(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    from factor_engine.cleaned_operators.alpha_language_shape import _EPS
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
        line = v[0] + (j / den) * (v[-1] - v[0])
        mce = float(np.max(np.abs(v - line)))
        path = float(np.sum(np.abs(np.diff(v))))
        return mce / (path + _EPS)

    return _rebuild(b["x"], map_rolling(xv, w, _fn))


def _k_intra_high_low_affinity(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import minute_of_day
    w = max(2, int(b.get("window", 20)))
    session_tz = b.get("session_tz")
    times = _time_numpy(b["high"], session_tz)
    if times is None:
        raise ValueError("intraday panel requires a time column")
    hv = _panel(b["high"])
    lv = _panel(b["low"])
    rows, cols = hv.shape
    day = times.astype("datetime64[D]")
    minutes = minute_of_day(times)
    all_days = np.unique(day)
    mp = max(2, w // 2)
    out = np.full((all_days.size, cols), np.nan, dtype=float)
    for j in range(cols):
        # per-day extreme minutes
        toh: dict[np.datetime64, float] = {}
        tol: dict[np.datetime64, float] = {}
        for d in all_days:
            sel = day == d
            vh = hv[sel, j]
            vl = lv[sel, j]
            if np.any(np.isfinite(vh)):
                i = int(np.nanargmax(vh))
                toh[d] = float(minutes[sel][i])
            else:
                toh[d] = np.nan
            if np.any(np.isfinite(vl)):
                i = int(np.nanargmin(vl))
                tol[d] = float(minutes[sel][i])
            else:
                tol[d] = np.nan
        days_j = list(all_days)
        m_toh: dict[np.datetime64, float] = {}
        m_tol: dict[np.datetime64, float] = {}
        for t, d in enumerate(days_j):
            lo = max(0, t - w)
            wh = np.array([toh[x] for x in days_j[max(0, t - w): t]], dtype=float)
            wl = np.array([tol[x] for x in days_j[max(0, t - w): t]], dtype=float)
            fh = wh[np.isfinite(wh)]
            fl = wl[np.isfinite(wl)]
            m_toh[d] = float(fh.mean()) if fh.size >= mp else np.nan
            m_tol[d] = float(fl.mean()) if fl.size >= mp else np.nan
        for t, d in enumerate(days_j):
            vals = (toh[d], m_toh[d], tol[d], m_tol[d])
            if not all(np.isfinite(v) for v in vals):
                out[t, j] = np.nan
                continue
            dev = (abs(vals[0] - vals[1]) + abs(vals[2] - vals[3])) / 120.0
            out[t, j] = 1.0 / (1.0 + dev)
    return _daily_out(b["high"], all_days, out)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ts_variogram_slope": _k_ts_variogram_slope,
    "ts_multifractal_curvature": _k_ts_multifractal_curvature,
    "ts_qn_scale": _k_ts_qn_scale,
    "ts_roughness": _k_ts_roughness,
    "ts_interval_exploration_efficiency": _k_ts_interval_exploration_efficiency,
    "cs_weighted_percentile_rank": _k_cs_weighted_percentile_rank,
    "ts_distance_cov": _k_ts_distance_cov,
    "spectral_trend_share": _k_spectral_trend_share,
    "ts_run_efficiency": _k_ts_run_efficiency,
    "ts_interval_overlap_connected_component_ratio": _k_ts_interval_overlap_connected_component_ratio,
    "ts_autocorrelation_time": _k_ts_autocorrelation_time,
    "ts_binned_response_monotonicity": _k_ts_binned_response_monotonicity,
    "ts_har_rv_next_var_forecast": _k_ts_har_rv_next_var_forecast,
    "ts_har_from_return_next_vol": _k_ts_har_from_return_next_vol,
    "ts_permutation_transition_entropy": _k_ts_permutation_transition_entropy,
    "ashare_one_price_limit_streak": _k_ashare_one_price_limit_streak,
    "ts_forbidden_ordinal_pattern_excess": _k_ts_forbidden_ordinal_pattern_excess,
    "fin_net_debt_issuance": _k_fin_net_debt_issuance,
    "ts_rank_if": _k_ts_rank_if,
    "intra_high_low_affinity": _k_intra_high_low_affinity,
    "ts_regression_tstat": _k_ts_regression_tstat,
    "cs_actual_lof_score": _k_cs_actual_lof_score,
    "fin_core_earnings_ratio": _k_fin_core_earnings_ratio,
    "fin_noncore_income_ratio": _k_fin_noncore_income_ratio,
    "state_slew_limit": _k_state_slew_limit,
    "intra_idiosyncratic_skewness_ex_self": _k_intra_idiosyncratic_skewness_ex_self,
    "ts_pseudocount_sample_entropy": _k_ts_pseudocount_sample_entropy,
    "ashare_limit_open_up_streak": _k_ashare_limit_open_up_streak,
    "cs_wls_resid": _k_cs_wls_resid,
    "intra_abs_return_profile_cosine": _k_intra_abs_return_profile_cosine,
    "fiscal_perpetual_inventory": _k_fiscal_perpetual_inventory,
    "ts_max_chord_excursion": _k_ts_max_chord_excursion,
    "ts_forbidden_ordinal_pattern_signed_excess": _k_ts_forbidden_ordinal_pattern_signed_excess,
    "intra_idiosyncratic_kurtosis_ex_self": _k_intra_idiosyncratic_kurtosis_ex_self,
    "ts_two_state_regime_probability": _k_ts_two_state_regime_probability,
    "state_deadband": _k_state_deadband,
    "ashare_limit_open_down_streak": _k_ashare_limit_open_down_streak,
}

# Source-bound execution inventory. Every entry above materializes one or
# more Polars columns to NumPy arrays and rebuilds a DataFrame; none emits a
# Polars Expr. Keep this explicit so a new kernel cannot inherit a wrong kind.
_NUMPY_KERNEL_CANONICALS = frozenset({
    "ts_variogram_slope", "ts_multifractal_curvature", "ts_qn_scale",
    "ts_roughness", "ts_interval_exploration_efficiency",
    "cs_weighted_percentile_rank", "ts_distance_cov",
    "spectral_trend_share", "ts_run_efficiency",
    "ts_interval_overlap_connected_component_ratio", "ts_autocorrelation_time",
    "ts_binned_response_monotonicity",
    "ts_har_rv_next_var_forecast", "ts_har_from_return_next_vol",
    "ts_permutation_transition_entropy", "ashare_one_price_limit_streak",
    "ts_forbidden_ordinal_pattern_excess", "fin_net_debt_issuance",
    "ts_rank_if", "intra_high_low_affinity", "ts_regression_tstat",
    "cs_actual_lof_score", "fin_core_earnings_ratio", "fin_noncore_income_ratio",
    "state_slew_limit", "intra_idiosyncratic_skewness_ex_self",
    "ts_pseudocount_sample_entropy", "ashare_limit_open_up_streak",
    "cs_wls_resid", "intra_abs_return_profile_cosine",
    "fiscal_perpetual_inventory",
    "ts_max_chord_excursion", "ts_forbidden_ordinal_pattern_signed_excess",
    "intra_idiosyncratic_kurtosis_ex_self", "ts_two_state_regime_probability",
    "state_deadband", "ashare_limit_open_down_streak",
})
_EXPR_KERNEL_CANONICALS = frozenset()
if set(_KERNELS) != _NUMPY_KERNEL_CANONICALS | _EXPR_KERNEL_CANONICALS:
    raise RuntimeError("R68 batch9 kernel execution inventory is out of sync with _KERNELS")
_KERNEL_EXECUTION_KIND = {
    **{name: ExecutionKind.POLARS_NUMPY_KERNEL for name in _NUMPY_KERNEL_CANONICALS},
    **{name: ExecutionKind.POLARS_NATIVE_EXPR for name in _EXPR_KERNEL_CANONICALS},
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch6)
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


def register_r68_native_batch9() -> list[str]:
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
            # First *genuine native* registrant wins: a pre-existing pandas
            # delegate UDF slot must be replaced, only a real POLARS_NATIVE
            # implementation blocks registration.  (canonical_polars_kind is a
            # heuristic that mislabels some simple UDF delegates — e.g. atan2 —
            # as native, so the registered op's own spec is authoritative.)
            try:
                # Use the classifier's enum consistently. A raw ExecutionKind
                # cannot be compared by identity to PolarsImplementationKind.
                cur_kind = canonical_polars_kind(canonical, production_mode=False)
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
            execution_kind=_KERNEL_EXECUTION_KIND[canonical],
            supports_lazy=False, supports_streaming=False,
            materializes_full_panel=True,
            supports_nulls=True, supports_nan=True, supports_inf=True,
            implementation_source_hash=source_hash,
            emitter_identity=f"{_SOURCE}:pl.DataFrame/numpy/pl.DataFrame:v1",
            kernel_identity=f"{_SOURCE}._KERNELS:{canonical}",
            parameter_domain_hash=parameter_hash,
            semantic_contract_hash=hashlib.sha256(
                (canonical + ":pandas-authority-parity:r68b9").encode()
            ).hexdigest(),
            notes=(
                "R68 batch9 genuine Polars backend: numpy authority kernels over "
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


__all__ = ["register_r68_native_batch9"]

register_r68_native_batch9()
