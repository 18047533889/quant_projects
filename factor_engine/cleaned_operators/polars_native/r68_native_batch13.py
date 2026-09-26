# -*- coding: utf-8 -*-
"""R68 batch13: genuine-Polars backends for the 45 pandas-delegate canonicals.

Same protocol as ``r68_native_batch4/5/6/9``: numpy authority helpers called
directly on ``pl -> numpy`` column panels, plus numpy ports for the kernels
whose authority is a pandas-Series pipeline.  **No pandas DataFrame is
constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``, no
``iterrows``).
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch13"

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


def _group_panel(value: Any) -> np.ndarray | None:
    """Group-id panel as an object array (None when the param is absent)."""
    if value is None:
        return None
    if isinstance(value, pl.Series):
        value = value.to_frame()
    if not isinstance(value, pl.DataFrame):
        return None
    cols = _ncols(value)
    if not cols:
        return None
    gv = np.empty((value.height, len(cols)), dtype=object)
    for j, c in enumerate(cols):
        gv[:, j] = value[c].to_numpy(allow_copy=True)
    return gv


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


def _unique_labels(g_row: np.ndarray, valid_fn: Callable[[Any], bool]) -> list[Any]:
    seen: list[Any] = []
    seen_set: set[Any] = set()
    for v in g_row:
        try:
            hashable = v.item() if hasattr(v, "item") else v
        except Exception:
            hashable = str(v)
        if hashable in seen_set:
            continue
        seen_set.add(hashable)
        if valid_fn(v):
            seen.append(v)
    return seen


# ===========================================================================
# kernels
# ===========================================================================
def _k_group_ex_self_quantile(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_cross import _group_peers_indices
    q = float(b.get("q", 0.5))
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    minp = max(2, int(b.get("min_peers", 2)))
    xv = _panel(b["x"])
    gv = _group_panel(b["group"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        for label in _unique_labels(gv[row], lambda v: True):
            peers = _group_peers_indices(gv[row], label, finite[row])
            if peers.size <= minp:
                continue
            for j in peers:
                others = peers[peers != j]
                if others.size < minp:
                    continue
                vals = xv[row, others]
                out[row, j] = float(np.quantile(vals, q))
    return _rebuild(b["x"], out)


def _k_cs_tail_breadth(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.stateful.rotation import (
        _EPS, _exact_tail_weights, _tail_quantile, _valid_label,
    )
    q = _tail_quantile(b.get("quantile", 0.1), "cs_tail_breadth")
    side_s = str(b.get("side", "top")).lower()
    if side_s not in ("top", "bottom"):
        raise ValueError("side must be 'top' or 'bottom'")
    top = side_s == "top"
    xv = _panel(b["x"])
    gv = _group_panel(b.get("group"))
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        g_row = gv[r] if gv is not None else None
        if g_row is None:
            w = _exact_tail_weights(xv[r], q, top)
            if float(np.sum(w)) > _EPS:
                out[r] = float(np.sum(w))
        else:
            for lab in _unique_labels(g_row, _valid_label):
                mask = g_row == lab
                w = _exact_tail_weights(xv[r][mask], q, top)
                if float(np.sum(w)) > _EPS:
                    out[r][mask] = float(np.sum(w))
    return _rebuild(b["x"], out)


def _k_group_tail_centrality(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.tail_systemic import _tail_centrality_series
    w = int(b.get("window", 120))
    q = float(b.get("quantile", 0.1))
    mp = int(b.get("min_periods", 10))
    side = str(b.get("side", "lower"))
    prior = bool(b.get("prior_threshold", True))
    if not (0.0 < q <= 0.5):
        raise ValueError("group_tail_centrality requires 0 < quantile <= 0.5 (tail state)")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if w < 5:
        raise ValueError("group_tail_centrality requires window >= 5")
    if not 1 <= mp <= w:
        raise ValueError("group_tail_centrality requires 1 <= min_periods <= window")
    if prior and mp > w - 1:
        raise ValueError(
            "INFEASIBLE_PARAMETER_DOMAIN: prior_threshold=True requires "
            "min_periods <= window - 1"
        )
    xv = _panel(b["x"])
    gv = _group_panel(b["group_id"])
    return _rebuild(
        b["x"],
        _tail_centrality_series(xv, gv, w, q, side, mp, prior),
    )


def _k_group_tail_lead_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.tail_systemic import _tail_lead_series
    w = int(b.get("window", 120))
    q = float(b.get("quantile", 0.1))
    lg = int(b.get("lag", 1))
    mp = int(b.get("min_periods", 10))
    side = str(b.get("side", "lower"))
    prior = bool(b.get("prior_threshold", True))
    mce = int(b.get("min_conditioning_events", 5))
    if not (0.0 < q <= 0.5):
        raise ValueError("group_tail_lead_score requires 0 < quantile <= 0.5 (tail state)")
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if lg < 1:
        raise ValueError("group_tail_lead_score requires lag >= 1")
    if w < lg + 2:
        raise ValueError("group_tail_lead_score requires window >= lag + 2")
    if not 1 <= mp <= w:
        raise ValueError("group_tail_lead_score requires 1 <= min_periods <= window")
    if prior and mp > w - 1:
        raise ValueError(
            "INFEASIBLE_PARAMETER_DOMAIN: prior_threshold=True requires "
            "min_periods <= window - 1"
        )
    if mce < 1:
        raise ValueError("group_tail_lead_score requires min_conditioning_events >= 1")
    xv = _panel(b["x"])
    gv = _group_panel(b["group_id"])
    return _rebuild(
        b["x"],
        _tail_lead_series(xv, gv, w, q, side, lg, mp, prior, mce),
    )


def _k_group_spd_feature_structure_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import (
        _LAM, _group_spd_shift, _spd_log_matrix, _valid_group_label,
    )
    from factor_engine.cleaned_operators.parameter_validation import strict_integer
    rw = strict_integer(b.get("reference_window", 60), "reference_window", minimum=2)
    if rw < 2:
        raise ValueError("group_spd_feature_structure_shift requires reference_window >= 2")
    if str(b.get("composition_policy", "current")) not in ("current", "intersection"):
        raise ValueError("group_spd_feature_structure_shift composition_policy must be 'current' or 'intersection'")
    mp_raw = b.get("min_peers")
    mp = 5 if mp_raw is None else strict_integer(mp_raw, "min_peers", minimum=3)
    mrd_raw = b.get("min_reference_days")
    mrd = 5 if mrd_raw is None else strict_integer(mrd_raw, "min_reference_days", minimum=2)
    if mrd > rw:
        raise ValueError("min_reference_days must be <= reference_window")
    f1, f2, f3 = _panel(b["f1"]), _panel(b["f2"]), _panel(b["f3"])
    gv = _group_panel(b["group"])
    rows, cols = f1.shape
    d = 3
    log_by_date: list[dict[Any, tuple[np.ndarray, np.ndarray]]] = []
    for t in range(rows):
        row = np.stack([f1[t], f2[t], f3[t]], axis=1)
        g_row = gv[t]
        day: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for label in _unique_labels(g_row, _valid_group_label):
            idx = np.flatnonzero(g_row == label)
            if idx.size < mp:
                continue
            m = _spd_log_matrix(row[idx], d, _LAM, mp)
            if m is not None:
                day[label] = (m, idx)
        log_by_date.append(day)
    out = np.full((rows, cols), np.nan, dtype=float)
    feats = np.stack([f1, f2, f3], axis=2)
    for t in range(rows):
        out[t] = _group_spd_shift(
            feats, gv, t, rw, log_by_date, d, _LAM, mp, mrd,
            str(b.get("composition_policy", "current")),
        )
    return _rebuild(b["f1"], out)


def _time_ns(frame: pl.DataFrame) -> np.ndarray | None:
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
        s = s.dt.convert_time_zone(str(_SESSION_TZ)).dt.replace_time_zone(None)
    return s.to_numpy(allow_copy=True).astype("datetime64[ns]")


def _k_group_current_members_tail_coexceedance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_ext import (
        _MIN_Q_ROWS, _tail_coexceedance_series,
    )
    w = int(b.get("window", 120))
    if w < _MIN_Q_ROWS + 1:
        raise ValueError(f"window must be >= {_MIN_Q_ROWS + 1} (threshold needs {_MIN_Q_ROWS} prior rows + query)")
    q = float(b.get("quantile", 0.9))
    if not (0.0 < q < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    side = str(b.get("side", "upper"))
    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")
    xv = _panel(b["x"])
    gv = _group_panel(b["group_id"])
    return _rebuild(b["x"], _tail_coexceedance_series(xv, gv, w, q, side))


def _k_group_corr_mst_length(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_ext import (
        _MIN_PAIR_ROWS, _mst_length_series,
    )
    w = int(b.get("window", 120))
    if w < _MIN_PAIR_ROWS:
        raise ValueError(f"window must be >= {_MIN_PAIR_ROWS} (a Pearson edge needs {_MIN_PAIR_ROWS} aligned rows)")
    xv = _panel(b["x"])
    gv = _group_panel(b["group_id"])
    return _rebuild(b["x"], _mst_length_series(xv, gv, w))


def _k_intraday_medrv(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.base import strict_int_param
    from factor_engine.cleaned_operators.jump_robust import (
        _MIN_FINITE, _rolling_medrv_vec,
    )
    w = strict_int_param(b.get("window", 240), "window", lower=_MIN_FINITE)
    return _rebuild(b["returns"], _rolling_medrv_vec(_panel(b["returns"]), w))


def _k_intraday_minrv(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.base import strict_int_param
    from factor_engine.cleaned_operators.jump_robust import (
        _MIN_FINITE, _minrv, _rolling_returns,
    )
    w = strict_int_param(b.get("window", 240), "window", lower=_MIN_FINITE)
    return _rebuild(b["returns"], _rolling_returns(_panel(b["returns"]), w, _minrv))


def _k_intraday_jump_test_stat(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.base import strict_int_param
    from factor_engine.cleaned_operators.jump_robust import (
        _MIN_FINITE, _jump_z, _rolling_returns,
    )
    w = strict_int_param(b.get("window", 240), "window", lower=_MIN_FINITE)
    return _rebuild(b["returns"], _rolling_returns(_panel(b["returns"]), w, _jump_z))


def _k_intraday_profile_pca_residual(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_session import (
        _minute_of_day, _pca_residual_series, _resolve_official_grid,
    )
    base = b["x"]
    sid_pl = b["session_id"]
    # _validate_session_ids equivalent on the polars panel
    if isinstance(sid_pl, pl.Series):
        sid_pl = sid_pl.to_frame()
    sid_cols = _ncols(sid_pl)
    for c in sid_cols:
        if sid_pl[c].dtype == pl.Boolean:
            raise ValueError("session_id must be integral; got a boolean column")
    sid2d = np.empty((sid_pl.height, len(sid_cols)), dtype=float)
    for j, c in enumerate(sid_cols):
        sid2d[:, j] = sid_pl[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True)
    fin = sid2d[np.isfinite(sid2d)]
    if np.any(np.abs(fin - np.round(fin)) > 1e-9):
        raise ValueError("session_id must be integral (a non-integer float is not a valid SessionID)")
    index_ns = _time_ns(base)
    if index_ns is None:
        raise ValueError("intraday_profile_pca_residual requires a time axis")
    dates = index_ns.astype("datetime64[D]").astype("int64")
    mods = _minute_of_day(index_ns)
    on_grid = index_ns.astype("datetime64[ns]").astype("int64") % 60_000_000_000 == 0
    expected, close_mod, slot_set = _resolve_official_grid(b.get("calendar"))
    arr = _pca_residual_series(
        _panel(base), sid2d, dates, mods,
        int(b.get("history_days", 20)), int(b.get("n_components", 3)),
        int(b.get("min_history_sessions", 5)),
        expected, close_mod, slot_set, on_grid,
    )
    return _rebuild(base, arr)


def _k_group_wasserstein_barycenter_distance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.group_key import is_missing_group_key
    from factor_engine.cleaned_operators.cs_state_ops import (
        _MIN_OBS, _quantiles, _wasserstein_pool,
    )
    if int(b.get("window", 60)) < 4:
        raise ValueError("group_wasserstein_barycenter_distance requires window >= 4")
    w = int(b.get("window", 60))
    mgs = max(2, int(b.get("min_group_size", 3)))
    arr = _panel(b["x"])
    garr = _group_panel(b["group"])
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        lo = max(0, r - w + 1)
        groups: dict[Any, list[int]] = {}
        for c in range(cols):
            lab = garr[r, c]
            if is_missing_group_key(lab):
                continue
            groups.setdefault(lab, []).append(c)
        for lab, members in groups.items():
            if len(members) < mgs:
                continue
            mat = arr[lo : r + 1][:, members]
            common_fin = np.all(np.isfinite(mat), axis=1)
            k = 0
            for row in reversed(common_fin):
                if row:
                    k += 1
                else:
                    break
            if k < _MIN_OBS:
                continue
            win = mat[-k:]
            mem_q: dict[int, np.ndarray] = {}
            for mi, c in enumerate(members):
                mem_q[c] = _quantiles(win[:, mi])
            if len(mem_q) < mgs:
                continue
            for c in members:
                if c not in mem_q:
                    continue
                peers = [mem_q[j] for j in mem_q if j != c]
                if len(peers) < 1:
                    continue
                bar_q = np.mean(np.stack(peers, axis=0), axis=0)
                out[r, c] = _wasserstein_pool(mem_q[c], bar_q)
    return _rebuild(b["x"], out)


def _k_panel_async_beta_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_gap import (
        _async_beta_column, _ex_self_market_returns,
    )
    window = max(2, int(b.get("window", 252)))
    min_periods = max(2, int(b.get("min_periods", 126)))
    refresh_freq = max(1, int(b.get("refresh_freq", 21)))
    rv = _panel(b["ret"])
    wv = _panel(b["weight"])
    mkt = _ex_self_market_returns(rv, wv)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        out[:, c] = _async_beta_column(rv[:, c], mkt[:, c], window, min_periods, refresh_freq)
    return _rebuild(b["ret"], out)


def _k_panel_factor_pocket_strength(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_gap import _cs_rank_pct
    window = max(2, int(b.get("window", 60)))
    min_periods = max(2, int(b.get("min_periods", 40)))
    threshold = float(b.get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"panel_factor_pocket_strength: threshold must be in (0,1), got {threshold!r}")
    rv = _panel(b["ret"])
    fv = _panel(b["factor"])
    rows, cols = rv.shape
    out_days = np.full((rows, cols), np.nan)
    cs_median = np.full(rows, np.nan)
    for s in range(rows):
        fr = _cs_rank_pct(fv[s])
        r = rv[s]
        valid_r = np.isfinite(r)
        if valid_r.sum() >= 1:
            cs_median[s] = float(np.nanmedian(r[valid_r]))
        valid = np.isfinite(fv[s]) & valid_r
        in_pocket = fr >= threshold
        for c in range(cols):
            if not np.isfinite(fv[s, c]) or not valid_r[c]:
                out_days[s, c] = np.nan
            elif in_pocket[c] and r[c] > cs_median[s]:
                out_days[s, c] = 1.0
            else:
                out_days[s, c] = 0.0
    out = np.full((rows, cols), np.nan)
    for t in range(rows):
        start = max(0, t - window + 1)
        seg = out_days[start : t + 1, :]
        for c in range(cols):
            col_seg = seg[:, c]
            valid = np.isfinite(col_seg)
            if int(valid.sum()) < min_periods:
                continue
            out[t, c] = float(np.mean(col_seg[valid]))
    return _rebuild(b["ret"], out)


def _k_cs_predictability_mosaic_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_gap import _mosaic_kernel
    window = max(2, int(b.get("window", 504)))
    min_history = max(2, int(b.get("min_history", 160)))
    clusters = max(2, int(b.get("clusters", 6)))
    lag = max(1, int(b.get("lag", 1)))
    bv = _panel(b["base_signal"])
    rv = _panel(b["realized_return"])
    sv = _panel(b["state_feature"])
    return _rebuild(b["base_signal"], _mosaic_kernel(bv, rv, sv, window, min_history, clusters, lag))


def _k_panel_predictability_mosaic_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_gap import _mosaic_kernel
    window = max(2, int(b.get("window", 504)))
    min_history = max(2, int(b.get("min_history", 160)))
    clusters = max(2, int(b.get("clusters", 6)))
    lag = max(1, int(b.get("lag", 1)))
    bv = _panel(b["base_signal"])
    rv = _panel(b["realized_return"])
    sv = _panel(b["state_feature"])
    mv = _mosaic_kernel(bv, rv, sv, window, min_history, clusters, lag)
    rows = mv.shape[0]
    out = np.full(mv.shape, np.nan)
    for t in range(rows):
        row = mv[t]
        valid = np.isfinite(row)
        if int(valid.sum()) < 1:
            continue
        out[t, :] = float(np.mean(row[valid]))
    return _rebuild(b["base_signal"], out)


def _k_ts_super_smoother(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.filter_smooth import _strict_int
    period = _strict_int(b.get("period", 10), "period", 3)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    sqrt2_pi = np.sqrt(2.0) * np.pi
    a = np.exp(-sqrt2_pi / period)
    bcoef = 2.0 * a * np.cos(sqrt2_pi / period)
    c2 = bcoef
    c3 = -(a * a)
    c1 = 1.0 - c2 - c3
    for col in range(cols):
        finite_indices = np.where(np.isfinite(xv[:, col]))[0]
        if len(finite_indices) == 0:
            continue
        if len(finite_indices) == 1:
            idx = finite_indices[0]
            out[idx, col] = xv[idx, col]
            continue
        idx0 = finite_indices[0]
        idx1 = finite_indices[1]
        y_t_minus_2 = xv[idx0, col]
        y_t_minus_1 = xv[idx1, col]
        x_t_minus_1 = xv[idx1, col]
        out[idx0, col] = y_t_minus_2
        out[idx1, col] = y_t_minus_1
        for row in range(idx1 + 1, rows):
            curr = xv[row, col]
            if not np.isfinite(curr):
                out[row, col] = np.nan
                continue
            y_t = c1 * (curr + x_t_minus_1) / 2.0 + c2 * y_t_minus_1 + c3 * y_t_minus_2
            out[row, col] = y_t
            y_t_minus_2 = y_t_minus_1
            y_t_minus_1 = y_t
            x_t_minus_1 = curr
    return _rebuild(b["x"], out)


def _k_ts_kama(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.filter_smooth import _strict_int
    er_window = _strict_int(b.get("er_window", 10), "er_window", 2)
    fast_period = _strict_int(b.get("fast_period", 2), "fast_period", 1)
    slow_period = _strict_int(b.get("slow_period", 30), "slow_period", 1)
    if fast_period >= slow_period:
        raise ValueError("fast_period must be < slow_period")
    required = er_window + 1
    mp_raw = b.get("min_periods")
    if mp_raw is None:
        min_periods = required
    else:
        min_periods = _strict_int(mp_raw, "min_periods", 1)
        if min_periods < required:
            raise ValueError(f"min_periods must be at least er_window + 1 ({required})")
    arr = _panel(b["x"])
    rows, cols = arr.shape
    # change[t] = |x[t] - x[t-er_window]|; vol[t] = rolling sum of |diff| over
    # the trailing er_window rows (pandas min_periods=er_window -> all finite).
    diff = np.full((rows, cols), np.nan)
    diff[1:] = np.abs(arr[1:] - arr[:-1])
    change = np.full((rows, cols), np.nan)
    if rows > er_window:
        change[er_window:] = np.abs(arr[er_window:] - arr[:-er_window])
    vol = np.full((rows, cols), np.nan)
    for j in range(cols):
        d = diff[:, j]
        for t in range(er_window - 1, rows):
            lo = t - er_window + 1
            seg = d[lo : t + 1]
            if np.all(np.isfinite(seg)):
                s = 0.0
                for u in range(lo, t + 1):
                    s += d[u]
                vol[t, j] = s
    epsilon = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = change / (vol + epsilon)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    alpha = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2
    alpha = np.where(np.isfinite(alpha), alpha, np.nan)
    out = np.full_like(arr, np.nan, dtype=float)
    for c in range(cols):
        last = np.nan
        contiguous = 0
        for t in range(rows):
            if not np.isfinite(arr[t, c]):
                last = np.nan
                contiguous = 0
                continue
            contiguous += 1
            if contiguous < min_periods:
                continue
            if not np.isfinite(last):
                last = arr[t, c]
            elif np.isfinite(alpha[t, c]):
                last = last + alpha[t, c] * (arr[t, c] - last)
            out[t, c] = last
    return _rebuild(b["x"], out)


def _k_ts_causal_local_linear_smoother(b: dict) -> pl.DataFrame:
    window = int(b.get("window", 20))
    min_periods = int(b.get("min_periods", 10))
    if window < 2:
        raise ValueError(f"ts_causal_local_linear_smoother requires window >= 2, got {window}")
    if min_periods < 2:
        raise ValueError(f"ts_causal_local_linear_smoother requires min_periods >= 2, got {min_periods}")
    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) cannot exceed window ({window})")
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start_idx = max(0, row - window + 1)
            window_data = xv[start_idx : row + 1, col]
            finite_mask = np.isfinite(window_data)
            finite_values = window_data[finite_mask]
            if len(finite_values) < min_periods:
                continue
            window_indices = np.arange(len(window_data), dtype=float)
            positions = window_indices[finite_mask]
            center = positions.mean()
            centered = positions - center
            scale = float(np.max(np.abs(finite_values)))
            scaled = finite_values / scale if scale > 0.0 else finite_values
            mean = scaled.mean()
            slope = np.dot(centered, scaled - mean) / np.dot(centered, centered)
            with np.errstate(over="ignore", invalid="ignore"):
                y_t = (mean + slope * (len(window_data) - 1 - center)) * scale
            if np.isfinite(y_t):
                out[row, col] = y_t
    return _rebuild(b["x"], out)


def _k_ts_butterworth_lowpass_causal(b: dict) -> pl.DataFrame:
    from scipy import signal as sp_signal
    cutoff_period = int(b.get("cutoff_period", 20))
    order = int(b.get("order", 2))
    if cutoff_period < 3:
        raise ValueError(f"ts_butterworth_lowpass_causal requires cutoff_period >= 3, got {cutoff_period}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if order > 10:
        raise ValueError(f"order must be <= 10 for numerical stability, got {order}")
    cutoff_freq = 1.0 / cutoff_period
    if cutoff_freq >= 0.5:
        raise ValueError(
            f"cutoff_period must be > 2 (cutoff_freq={cutoff_freq:.3f} >= Nyquist=0.5), got {cutoff_period}"
        )
    sos = sp_signal.butter(order, cutoff_freq, fs=1.0, btype="low", analog=False, output="sos")
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    zi0 = sp_signal.sosfilt_zi(sos)
    for col in range(cols):
        xc = xv[:, col]
        fin = np.isfinite(xc)
        if not fin.any():
            continue
        idx = np.where(fin)[0]
        brk = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, brk)
        prev_zi = None
        for run in runs:
            seg = xc[run]
            if prev_zi is None:
                zi = zi0 * seg[0]
            else:
                zi = prev_zi
            y, zi_out = sp_signal.sosfilt(sos, seg, zi=zi)
            out[run, col] = y
            prev_zi = zi_out
    return _rebuild(b["x"], out)


def _k_rolling_beta_to_market(b: dict) -> pl.DataFrame:
    w = max(2, int(b.get("window", 60)))
    mp = max(2, w // 3)
    yv = _panel(b["ret"])
    bv = _panel(b["benchmark_ret"])
    if bv.shape[1] == 1:
        bv = np.tile(bv[:, :1], (1, yv.shape[1]))
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        y = yv[:, c]
        x = bv[:, c]
        valid = np.isfinite(y) & np.isfinite(x)
        xfin = np.isfinite(x)
        cy = np.cumsum(np.where(valid, y, 0.0))
        cx = np.cumsum(np.where(valid, x, 0.0))
        cxy = np.cumsum(np.where(valid, y * x, 0.0))
        cx2 = np.cumsum(np.where(xfin, x * x, 0.0))
        cnp = np.cumsum(valid.astype(np.int64))
        cnx = np.cumsum(xfin.astype(np.int64))
        for t in range(rows):
            lo = max(0, t - w + 1)
            pre = cnp[lo - 1] if lo > 0 else 0
            n_pair = int(cnp[t] - pre)
            if n_pair < max(2, mp):
                continue
            pre_x = cx[lo - 1] if lo > 0 else 0.0
            pre_y = cy[lo - 1] if lo > 0 else 0.0
            pre_xy = cxy[lo - 1] if lo > 0 else 0.0
            sx = cx[t] - pre_x
            sy = cy[t] - pre_y
            sxy = cxy[t] - pre_xy
            cov = (sxy - sx * sy / n_pair) / (n_pair - 1)
            pre_x2 = cx2[lo - 1] if lo > 0 else 0.0
            pre_nx = cnx[lo - 1] if lo > 0 else 0
            n_x = int(cnx[t] - pre_nx)
            if n_x < max(2, mp):
                continue
            sx_all = cx[t] - (cx[lo - 1] if lo > 0 else 0.0)
            sx2 = cx2[t] - pre_x2
            var = (sx2 - sx_all * sx_all / n_x) / (n_x - 1)
            if var == 0.0 or not np.isfinite(var):
                continue
            beta = cov / var
            if np.isfinite(beta) and valid[t]:
                out[t, c] = beta
    return _rebuild(b["ret"], out)


def _k_intraday_vwap_deviation(b: dict) -> pl.DataFrame:
    base = b["close"]
    cv = _panel(b["close"])
    pv_price = _panel(b["price"])
    vol = _panel(b["volume"])
    index_ns = _time_ns(base)
    rows, cols = cv.shape
    if index_ns is not None:
        days = index_ns.astype("datetime64[D]").astype("int64")
        uniq = np.unique(days[np.isfinite(days.astype(float))])
    else:
        days = None
        uniq = np.array([0])
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        p = pv_price[:, j]
        v = vol[:, j]
        cl = cv[:, j]
        if days is None or uniq.size <= 1:
            pv = p * v
            cs_pv = np.cumsum(np.where(np.isfinite(pv), pv, 0.0) * np.isfinite(pv))
            cs_v = np.cumsum(np.where(np.isfinite(v), v, 0.0) * np.isfinite(v))
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap = np.where(cs_v != 0.0, cs_pv / cs_v, np.nan)
                dev = cl / vwap - 1.0
            out[:, j] = dev
        else:
            for day in uniq:
                mask = days == day
                p_m = p[mask]
                v_m = v[mask]
                pv = p_m * v_m
                cs_pv = np.cumsum(np.where(np.isfinite(pv), pv, 0.0) * np.isfinite(pv))
                cs_v = np.cumsum(np.where(np.isfinite(v_m), v_m, 0.0) * np.isfinite(v_m))
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = np.where(cs_v != 0.0, cs_pv / cs_v, np.nan)
                    dev = cl[mask] / vwap - 1.0
                out[mask, j] = dev
    return _rebuild(base, out)


def _k_recipe_vp_weighted_price(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composite_fastpath import pl_vp_weighted_price
    return pl_vp_weighted_price(
        b["close"], b["volume"], b.get("open"), b.get("high"), b.get("low"),
        window=int(b.get("window", 20)), min_periods=5,
    )


def _np_ewm_mean(a: np.ndarray, alpha: float, mp: int) -> np.ndarray:
    """pandas ``ewm(alpha, adjust=False, ignore_na=False, min_periods=mp).mean()`` port."""
    rows = a.shape[0]
    out = np.full(rows, np.nan, dtype=float)
    y = np.nan
    cnt = 0
    gap = 0
    for i in range(rows):
        v = a[i]
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
            out[i] = y
    return out


def _ema_frame(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = _np_ewm_mean(arr[:, j], alpha, 0)
    return out


def _k_recipe_vpmacd(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composite_fastpath import pl_vp_weighted_price
    weighted = _panel(pl_vp_weighted_price(
        b["close"], b["volume"], b.get("open"), b.get("high"), b.get("low"),
        window=20, min_periods=5,
    ))
    line = _ema_frame(weighted, 12) - _ema_frame(weighted, 26)
    signal_line = _ema_frame(line, 9)
    lam = float(b.get("lambda_param", 0.9))
    return _rebuild(b["close"], line - lam * signal_line)


def _k_recipe_vpmacd_signal(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composite_fastpath import pl_vp_weighted_price
    weighted = _panel(pl_vp_weighted_price(
        b["close"], b["volume"], b.get("open"), b.get("high"), b.get("low"),
        window=20, min_periods=5,
    ))
    line = _ema_frame(weighted, 12) - _ema_frame(weighted, 26)
    signal_line = _ema_frame(line, 9)
    lam = float(b.get("lambda_param", 0.9))
    adjusted = signal_line * lam
    rows, cols = line.shape
    out = np.zeros((rows, cols), dtype=float)
    for j in range(cols):
        prev_line = np.concatenate([[np.nan], line[:-1, j]])
        prev_adj = np.concatenate([[np.nan], adjusted[:-1, j]])
        with np.errstate(invalid="ignore"):
            golden = (line[:, j] > adjusted[:, j]) & (prev_line <= prev_adj)
            death = (line[:, j] < adjusted[:, j]) & (prev_line >= prev_adj)
        out[golden, j] = 1.0
        out[death, j] = -1.0
    return _rebuild(b["close"], out)


def _k_recipe_micro_spread(b: dict) -> pl.DataFrame:
    high = _panel(b["high"])
    low = _panel(b["low"])
    close = _panel(b["close"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(close != 0.0, (high - low) / close, np.nan)
    return _rebuild(b["high"], out)


def _k_recipe_micro_trade_imbalance(b: dict) -> pl.DataFrame:
    base = b["close"]
    cv = _panel(base)
    vol = _panel(b["volume"])
    w = int(b.get("window", 20))
    mp = int(b.get("min_periods", 2))
    index_ns = _time_ns(base)
    rows, cols = cv.shape
    if index_ns is not None:
        days = index_ns.astype("datetime64[D]").astype("int64")
        uniq = np.unique(days[np.isfinite(days.astype(float))])
    else:
        days = None
        uniq = np.array([0])
    out = np.full((rows, cols), np.nan, dtype=float)

    def _pct_change(seg: np.ndarray) -> np.ndarray:
        ret = np.full(seg.shape, np.nan)
        if seg.size > 1:
            prev = seg[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[1:] = np.where(prev != 0.0, seg[1:] / prev - 1.0, np.nan)
        return ret

    def _roll_sum(seg: np.ndarray) -> np.ndarray:
        n = seg.size
        res = np.full(n, np.nan)
        fin = np.isfinite(seg)
        csum = np.cumsum(np.where(fin, seg, 0.0))
        ccnt = np.cumsum(fin.astype(np.int64))
        for t in range(n):
            lo = max(0, t - w + 1)
            cnt = int(ccnt[t] - (ccnt[lo - 1] if lo > 0 else 0))
            if cnt < mp:
                continue
            res[t] = float(csum[t] - (csum[lo - 1] if lo > 0 else 0.0))
        return res

    for j in range(cols):
        cl = cv[:, j]
        v = vol[:, j]
        if days is None or uniq.size <= 1:
            ret = _pct_change(cl)
            mag = np.sign(ret)
            num = _roll_sum(mag * v)
            den = _roll_sum(v)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, j] = np.where(den != 0.0, num / den, np.nan)
        else:
            for day in uniq:
                mask = days == day
                ret = _pct_change(cl[mask])
                mag = np.sign(ret)
                num = _roll_sum(mag * v[mask])
                den = _roll_sum(v[mask])
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[mask, j] = np.where(den != 0.0, num / den, np.nan)
    return _rebuild(base, out)


# ---------------------------------------------------------------------------
# expanding statistics engines (numpy ports of _causal.expanding_* semantics)
# ---------------------------------------------------------------------------
def _exp_uni(xm: np.ndarray, fn, mp: int) -> np.ndarray:
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        vals = xm[:, j]
        for i in range(rows):
            if i + 1 < mp:
                continue
            valid = vals[: i + 1]
            valid = valid[np.isfinite(valid)]
            if valid.size < mp:
                continue
            try:
                out[i, j] = float(fn(valid))
            except Exception:
                pass
    return out


def _exp_bi(xm: np.ndarray, ym: np.ndarray, fn, mp: int) -> np.ndarray:
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        xc = xm[:, j]
        yc = ym[:, j]
        for i in range(rows):
            if i + 1 < mp:
                continue
            paired = np.column_stack([xc[: i + 1], yc[: i + 1]])
            paired = paired[np.all(np.isfinite(paired), axis=1)]
            if paired.shape[0] < mp:
                continue
            try:
                out[i, j] = float(fn(paired[:, 0], paired[:, 1]))
            except Exception:
                pass
    return out


def _exp_two(xm: np.ndarray, ym: np.ndarray, fn, mp: int) -> np.ndarray:
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        xc = xm[:, j]
        yc = ym[:, j]
        for i in range(rows):
            xa = xc[: i + 1]
            ya = yc[: i + 1]
            xa = xa[np.isfinite(xa)]
            ya = ya[np.isfinite(ya)]
            if xa.size < mp or ya.size < mp:
                continue
            try:
                out[i, j] = float(fn(xa, ya))
            except Exception:
                pass
    return out


def _expanding_granger_fast_np(xm: np.ndarray, ym: np.ndarray, lag: int, min_periods: int) -> np.ndarray:
    from scipy import stats as _sstats
    mxlg = int(lag)
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    k_reg = 2 * mxlg + 1
    n_comp = k_reg + 1
    lo_n = max(min_periods, 3 * mxlg + 2)
    for j in range(cols):
        xc = xm[:, j]
        yc = ym[:, j]
        pair = np.isfinite(xc) & np.isfinite(yc)
        npair = np.cumsum(pair)
        rows_idx = np.flatnonzero(npair >= lo_n)
        if rows_idx.size == 0:
            continue
        yv = yc[pair]
        xv = xc[pair]
        m = yv.size
        if m <= 3 * mxlg + 1:
            continue
        comps = np.zeros((n_comp, m), dtype=float)
        for s in range(1, mxlg + 1):
            comps[s - 1, s:] = yv[: m - s]
            comps[mxlg - 1 + s, s:] = xv[: m - s]
        comps[k_reg - 1] = 1.0
        comps[k_reg] = yv
        prod = (comps[:, None, :] * comps[None, :, :]).astype(np.longdouble)
        PC = np.cumsum(prod, axis=2)
        run_max = np.maximum.accumulate(comps[:k_reg, mxlg:], axis=1)
        run_min = np.minimum.accumulate(comps[:k_reg, mxlg:], axis=1)
        eps = np.finfo(float).eps
        ns = npair[rows_idx]
        uniq_ns, _ = np.unique(ns, return_index=True)
        vals_by_n = np.full(m + 1, np.nan)
        a0 = mxlg
        for n in uniq_ns:
            k = int(n)
            if k <= 3 * mxlg + 1:
                continue
            b0 = k - 1
            nobs = k - mxlg
            n_const = int(np.count_nonzero(run_max[:, b0 - mxlg] == run_min[:, b0 - mxlg]))
            if n_const != 1:
                continue
            S = PC[:, :, b0] - (PC[:, :, a0 - 1] if a0 > 0 else 0.0)
            yy = S[k_reg, k_reg]
            sy0 = float(np.sum(yv[a0 : b0 + 1]))
            rhs_u = S[:k_reg, k_reg]
            M_u = S[:k_reg, :k_reg]
            own_idx = list(range(mxlg)) + [k_reg - 1]
            rhs_r = rhs_u[own_idx]
            M_r = M_u[np.ix_(own_idx, own_idx)]
            M_r64 = M_r.astype(float)
            M_u64 = M_u.astype(float)
            rhs_r64 = rhs_r.astype(float)
            rhs_u64 = rhs_u.astype(float)
            try:
                b_r = np.linalg.solve(M_r64, rhs_r64)
                b_u = np.linalg.solve(M_u64, rhs_u64)
            except np.linalg.LinAlgError:
                b_r = np.linalg.pinv(M_r64) @ rhs_r64
                b_u = np.linalg.pinv(M_u64) @ rhs_u64
            try:
                r_r = (rhs_r - M_r @ b_r.astype(np.longdouble)).astype(float)
                r_u = (rhs_u - M_u @ b_u.astype(np.longdouble)).astype(float)
                b_r = b_r + np.linalg.solve(M_r64, r_r)
                b_u = b_u + np.linalg.solve(M_u64, r_u)
            except np.linalg.LinAlgError:
                pass
            ssr_r = float(yy - np.dot(rhs_r, b_r.astype(np.longdouble)))
            ssr_u = float(yy - np.dot(rhs_u, b_u.astype(np.longdouble)))
            tss = yy - sy0 * sy0 / nobs
            if tss == 0 or ssr_u == 0 or not np.isfinite(ssr_u) or (ssr_u / tss) < eps:
                continue
            f_stat = (ssr_r - ssr_u) / ssr_u / mxlg * (nobs - k_reg)
            if not np.isfinite(f_stat):
                continue
            pval = float(_sstats.f.sf(f_stat, mxlg, nobs - k_reg))
            vals_by_n[k] = pval
        out[rows_idx, j] = vals_by_n[ns]
    return out


def _adf_schwert_maxlag(nobs: int) -> int:
    maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
    maxlag = min(nobs // 2 - 1 - 1, maxlag)
    return maxlag


def _expanding_adf_fast_np(xm: np.ndarray, min_periods: int = 10) -> np.ndarray:
    from scipy.special import ndtr
    from statsmodels.tsa.adfvalues import (
        _tau_largeps, _tau_maxs, _tau_mins, _tau_smallps, _tau_stars,
    )
    tau_max = _tau_maxs["c"][0]
    tau_min = _tau_mins["c"][0]
    tau_star = _tau_stars["c"][0]
    coef_small = _tau_smallps["c"][0][::-1]
    coef_large = _tau_largeps["c"][0][::-1]
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        vals = xm[:, j]
        fin = np.isfinite(vals)
        nv = np.cumsum(fin)
        rows_idx = np.flatnonzero(nv >= min_periods)
        if rows_idx.size == 0:
            continue
        xv = vals[fin]
        m = xv.size
        ml_glob = _adf_schwert_maxlag(m)
        if ml_glob < 0:
            continue
        n_comp = ml_glob + 4
        comps = np.zeros((n_comp, m), dtype=float)
        comps[0] = 1.0
        comps[1, 1:] = xv[:-1]
        d = np.zeros(m)
        d[1:] = xv[1:] - xv[:-1]
        for s in range(0, ml_glob + 1):
            comps[2 + s, s + 1:] = d[1 : m - s]
        comps[n_comp - 1] = d
        prod = (comps[:, None, :] * comps[None, :, :]).astype(np.longdouble)
        PC = np.cumsum(prod, axis=2)
        rmax = np.maximum.accumulate(xv)
        rmin = np.minimum.accumulate(xv)
        ns = nv[rows_idx]
        uniq_ns = np.unique(ns)
        vals_by_n = np.full(m + 1, np.nan)
        yy_all = PC[n_comp - 1, n_comp - 1]
        n_U = uniq_ns.size
        aic_tab = np.full((ml_glob + 1, n_U), np.inf)
        ml_arr = np.array([_adf_schwert_maxlag(int(n)) for n in uniq_ns], dtype=np.int64)
        a0_arr = ml_arr + 1
        b0_arr = uniq_ns.astype(np.int64) - 1
        nobs_arr = uniq_ns.astype(np.int64) - 1 - ml_arr
        ok_n = (ml_arr >= 0) & (nobs_arr > 0) & (rmax[uniq_ns - 1] != rmin[uniq_ns - 1])
        for L in range(0, ml_glob + 1):
            sel = ok_n & (ml_arr >= L)
            if not sel.any():
                continue
            K = L + 2
            uu = np.flatnonzero(sel)
            a0s = a0_arr[uu]
            b0s = b0_arr[uu]
            idx = np.array([0, 1] + [2 + q for q in range(1, L + 1)])
            sub = PC[np.ix_(idx, idx)]
            M_ld = np.moveaxis(sub[:, :, b0s] - sub[:, :, a0s - 1], 2, 0)
            rhs_ld = (PC[idx, n_comp - 1][:, b0s] - PC[idx, n_comp - 1][:, a0s - 1]).T
            yy = yy_all[b0s] - yy_all[a0s - 1]
            M = M_ld.astype(float)
            rhs = rhs_ld.astype(float)
            try:
                B = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
                R = (rhs_ld - np.matmul(M_ld, B.astype(np.longdouble)[..., None])[..., 0]).astype(float)
                B = B + np.linalg.solve(M, R[..., None])[:, :, 0]
            except np.linalg.LinAlgError:
                B = np.empty_like(rhs)
                for rr in range(uu.size):
                    try:
                        B[rr] = np.linalg.solve(M[rr], rhs[rr])
                    except np.linalg.LinAlgError:
                        B[rr] = np.linalg.pinv(M[rr]) @ rhs[rr]
            ssr = (yy - np.einsum("nj,nj->n", rhs_ld, B.astype(np.longdouble))).astype(float)
            ssr = np.where(ssr > 0, ssr, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                llf = -(nobs_arr[uu] / 2.0) * (np.log(2 * np.pi) + np.log(ssr / nobs_arr[uu]) + 1.0)
                aic = -2.0 * llf + 2.0 * K
            aic_tab[L, uu] = aic
        bestL = np.argmin(aic_tab, axis=0).astype(np.int64)
        best_aic = aic_tab[bestL, np.arange(n_U)]
        good = ok_n & np.isfinite(best_aic)
        pvals = np.full(n_U, np.nan)
        for Lv in np.unique(bestL[good]):
            sel = good & (bestL == Lv)
            uu = np.flatnonzero(sel)
            K = Lv + 2
            idx = np.array([0, 1] + [2 + q for q in range(1, Lv + 1)])
            a0s = np.full(uu.size, Lv + 1, dtype=np.int64)
            b0s = b0_arr[uu]
            nobs_f = uniq_ns[uu].astype(np.float64) - 1 - Lv
            sub = PC[np.ix_(idx, idx)]
            M_ld = np.moveaxis(sub[:, :, b0s] - sub[:, :, a0s - 1], 2, 0)
            rhs_ld = (PC[idx, n_comp - 1][:, b0s] - PC[idx, n_comp - 1][:, a0s - 1]).T
            yy = yy_all[b0s] - yy_all[a0s - 1]
            M = M_ld.astype(float)
            rhs = rhs_ld.astype(float)
            e1 = np.zeros((uu.size, K, 1))
            e1[:, 1, 0] = 1.0
            try:
                B = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
                Z = np.linalg.solve(M, e1)[:, :, 0]
                R = (rhs_ld - np.matmul(M_ld, B.astype(np.longdouble)[..., None])[..., 0]).astype(float)
                B = B + np.linalg.solve(M, R[..., None])[:, :, 0]
            except np.linalg.LinAlgError:
                B = np.empty_like(rhs)
                Z = np.empty_like(rhs)
                for rr in range(uu.size):
                    try:
                        B[rr] = np.linalg.solve(M[rr], rhs[rr])
                        Z[rr] = np.linalg.solve(M[rr], e1[rr, :, 0])
                    except np.linalg.LinAlgError:
                        B[rr] = np.linalg.pinv(M[rr]) @ rhs[rr]
                        Z[rr] = np.linalg.pinv(M[rr]) @ e1[rr, :, 0]
            ssr = (yy - np.einsum("nj,nj->n", rhs_ld, B.astype(np.longdouble))).astype(float)
            inv11 = Z[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma2 = ssr / (nobs_f - K)
                denom = np.sqrt(sigma2 * inv11)
                tstat = B[:, 1] / denom
            pv = np.empty(uu.size)
            hi = tstat > tau_max
            lo = tstat < tau_min
            pv[hi] = 1.0
            pv[lo] = 0.0
            mid = ~(hi | lo) & np.isfinite(tstat)
            if mid.any():
                sm_b = tstat[mid]
                pv_mid = np.empty(sm_b.shape)
                sel_small = sm_b <= tau_star
                for coef, sel_b in ((coef_small, sel_small), (coef_large, ~sel_small)):
                    if not sel_b.any():
                        continue
                    sm_c = sm_b[sel_b]
                    acc = np.zeros(sm_c.size)
                    for cc in range(coef.size):
                        acc = acc * sm_c + coef[cc]
                    pv_mid[sel_b] = ndtr(acc)
                pv[mid] = pv_mid
            pv[~np.isfinite(tstat)] = np.nan
            pvals[uu] = pv
        vals_by_n[uniq_ns] = pvals
        out[rows_idx, j] = vals_by_n[ns]
    return out


def _expanding_ttest_ind_np(xm: np.ndarray, ym: np.ndarray, min_periods: int) -> np.ndarray:
    from scipy import special
    rows, cols = xm.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for j in range(cols):
        xc = xm[:, j]
        yc = ym[:, j]
        fx = np.isfinite(xc)
        fy = np.isfinite(yc)
        n1 = np.cumsum(fx).astype(float)
        n2 = np.cumsum(fy).astype(float)
        m1a = np.empty(xc.size)
        s1a = np.empty(xc.size)
        m2a = np.empty(xc.size)
        s2a = np.empty(xc.size)
        c1 = 0
        c2 = 0
        mm1 = 0.0
        ss1 = 0.0
        mm2 = 0.0
        ss2 = 0.0
        for i in range(xc.size):
            v = xc[i]
            if np.isfinite(v):
                c1 += 1
                d1 = v - mm1
                mm1 += d1 / c1
                ss1 += d1 * (v - mm1)
            v = yc[i]
            if np.isfinite(v):
                c2 += 1
                d2 = v - mm2
                mm2 += d2 / c2
                ss2 += d2 * (v - mm2)
            m1a[i] = mm1
            s1a[i] = ss1
            m2a[i] = mm2
            s2a[i] = ss2
        with np.errstate(divide="ignore", invalid="ignore"):
            df = n1 + n2 - 2.0
            svar = (s1a + s2a) / df
            denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
            t = (m1a - m2a) / denom
            p = 2.0 * special.stdtr(df, -np.abs(t))
        ok = (n1 >= min_periods) & (n2 >= min_periods) & (df > 0)
        out[:, j] = np.where(ok, p, np.nan)
    return out


def _k_ts_expanding_acf_statistic(b: dict) -> pl.DataFrame:
    lag = int(b.get("lag", 1))
    if lag < 0:
        from factor_engine.backend.operator_errors import FutureReferenceError
        raise FutureReferenceError(f"negative lag {lag} references future data")
    window = int(b.get("window", 20))
    mp = min(max(1, window), lag + 1)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)

    def acf_func(s: np.ndarray, lg: int) -> float:
        n = len(s)
        if lg == 0:
            return 1.0
        finite_mask = np.isfinite(s)
        if int(finite_mask.sum()) <= lg:
            return np.nan
        mean = float(s[finite_mask].mean())
        var = float(((s[finite_mask] - mean) ** 2).sum())
        if var <= 0 or not np.isfinite(var):
            return np.nan
        a = s[:-lg]
        bb = s[lg:]
        paired = np.isfinite(a) & np.isfinite(bb)
        if int(paired.sum()) <= lg:
            return np.nan
        cov = float(((a[paired] - mean) * (bb[paired] - mean)).sum())
        return cov / var

    for j in range(cols):
        vals = xv[:, j]
        for t in range(rows):
            lo = max(0, t - window + 1)
            seg = vals[lo : t + 1]
            if int(np.isfinite(seg).sum()) < mp:
                continue
            out[t, j] = acf_func(seg, lag)
    return _rebuild(b["x"], out)


def _k_ts_expanding_pacf_statistic(b: dict) -> pl.DataFrame:
    from statsmodels.tsa.stattools import pacf as sm_pacf
    lag = int(b.get("lag", 1))
    if lag < 0:
        from factor_engine.backend.operator_errors import FutureReferenceError
        raise FutureReferenceError(f"negative lag {lag} references future data")

    def _pacf_at_lag(v: np.ndarray) -> float:
        pacf_vals = sm_pacf(v, nlags=lag)
        return pacf_vals[lag] if lag < len(pacf_vals) else np.nan

    xm = _panel(b["x"])
    return _rebuild(b["x"], _exp_uni(xm, _pacf_at_lag, max(lag + 1, 3)))


def _k_ts_expanding_bartlett_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_two(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.bartlett(a, c)[1], 2))


def _k_ts_expanding_chi_square_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_bi(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.chisquare(a, c)[1], 2))


def _k_ts_expanding_pearson_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_bi(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.pearsonr(a, c)[1], 3))


def _k_ts_expanding_durbin_watson_statistic(b: dict) -> pl.DataFrame:
    from statsmodels.stats.stattools import durbin_watson
    return _rebuild(b["residuals"], _exp_uni(_panel(b["residuals"]), durbin_watson, 2))


def _k_ts_expanding_granger_pvalue(b: dict) -> pl.DataFrame:
    lag = int(b.get("lag", 5))
    if lag < 1:
        from factor_engine.backend.operator_errors import OperatorParameterError
        raise OperatorParameterError(f"granger maxlag must be >= 1, got {lag}")
    return _rebuild(
        b["x"],
        _expanding_granger_fast_np(_panel(b["x"]), _panel(b["y"]), lag, min_periods=lag + 2),
    )


def _k_ts_expanding_jarque_bera_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_uni(_panel(b["x"]), lambda v: stats.jarque_bera(v)[1], 3))


def _k_ts_expanding_kendall_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_bi(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.kendalltau(a, c)[1], 3))


def _k_ts_expanding_kpss_pvalue(b: dict) -> pl.DataFrame:
    from statsmodels.tsa.stattools import kpss

    def _kpss_pval(v: np.ndarray) -> float:
        return kpss(v, regression="c", nlags="auto")[1]

    return _rebuild(b["x"], _exp_uni(_panel(b["x"]), _kpss_pval, 10))


def _k_ts_expanding_ks_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    dist = str(b.get("dist", "norm"))
    return _rebuild(b["x"], _exp_uni(_panel(b["x"]), lambda v: stats.kstest(v, dist)[1], 2))


def _k_ts_expanding_levene_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_two(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.levene(a, c)[1], 2))


def _k_ts_expanding_lilliefors_pvalue(b: dict) -> pl.DataFrame:
    try:
        from statsmodels.stats.diagnostic import lilliefors as statsmodels_lilliefors
    except ImportError as exc:
        raise ImportError(
            "ts_expanding_lilliefors_pvalue requires statsmodels; "
            "install statsmodels>=0.14 to evaluate this operator"
        ) from exc
    return _rebuild(
        b["x"],
        _exp_uni(_panel(b["x"]), lambda v: statsmodels_lilliefors(v, dist="norm", pvalmethod="table")[1], 5),
    )


def _k_ts_expanding_spearman_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_bi(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.spearmanr(a, c)[1], 3))


def _k_ts_expanding_adf_pvalue(b: dict) -> pl.DataFrame:
    return _rebuild(b["x"], _expanding_adf_fast_np(_panel(b["x"]), min_periods=10))


def _k_ts_expanding_ttest_one_sample_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    mu = float(b.get("mu", 0.0))
    return _rebuild(b["x"], _exp_uni(_panel(b["x"]), lambda v: stats.ttest_1samp(v, mu)[1], 2))


def _k_ts_expanding_ttest_paired_pvalue(b: dict) -> pl.DataFrame:
    from scipy import stats
    return _rebuild(b["x"], _exp_bi(_panel(b["x"]), _panel(b["y"]), lambda a, c: stats.ttest_rel(a, c)[1], 2))


def _k_ts_expanding_ttest_two_sample_pvalue(b: dict) -> pl.DataFrame:
    return _rebuild(b["x"], _expanding_ttest_ind_np(_panel(b["x"]), _panel(b["y"]), min_periods=2))


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "group_ex_self_quantile": _k_group_ex_self_quantile,
    "cs_tail_breadth": _k_cs_tail_breadth,
    "group_tail_centrality": _k_group_tail_centrality,
    "group_tail_lead_score": _k_group_tail_lead_score,
    "group_spd_feature_structure_shift": _k_group_spd_feature_structure_shift,
    "group_current_members_tail_coexceedance": _k_group_current_members_tail_coexceedance,
    "group_corr_mst_length": _k_group_corr_mst_length,
    "intraday_medrv": _k_intraday_medrv,
    "intraday_minrv": _k_intraday_minrv,
    "intraday_jump_test_stat": _k_intraday_jump_test_stat,
    "intraday_profile_pca_residual": _k_intraday_profile_pca_residual,
    "group_wasserstein_barycenter_distance": _k_group_wasserstein_barycenter_distance,
    "panel_async_beta_ex_self": _k_panel_async_beta_ex_self,
    "panel_factor_pocket_strength": _k_panel_factor_pocket_strength,
    "cs_predictability_mosaic_score": _k_cs_predictability_mosaic_score,
    "panel_predictability_mosaic_score": _k_panel_predictability_mosaic_score,
    "ts_super_smoother": _k_ts_super_smoother,
    "ts_kama": _k_ts_kama,
    "ts_causal_local_linear_smoother": _k_ts_causal_local_linear_smoother,
    "ts_butterworth_lowpass_causal": _k_ts_butterworth_lowpass_causal,
    "rolling_beta_to_market": _k_rolling_beta_to_market,
    "intraday_vwap_deviation": _k_intraday_vwap_deviation,
    "recipe_vp_weighted_price": _k_recipe_vp_weighted_price,
    "recipe_vpmacd": _k_recipe_vpmacd,
    "recipe_vpmacd_signal": _k_recipe_vpmacd_signal,
    "recipe_micro_spread": _k_recipe_micro_spread,
    "recipe_micro_trade_imbalance": _k_recipe_micro_trade_imbalance,
    "ts_expanding_acf_statistic": _k_ts_expanding_acf_statistic,
    "ts_expanding_pacf_statistic": _k_ts_expanding_pacf_statistic,
    "ts_expanding_bartlett_pvalue": _k_ts_expanding_bartlett_pvalue,
    "ts_expanding_chi_square_pvalue": _k_ts_expanding_chi_square_pvalue,
    "ts_expanding_pearson_pvalue": _k_ts_expanding_pearson_pvalue,
    "ts_expanding_durbin_watson_statistic": _k_ts_expanding_durbin_watson_statistic,
    "ts_expanding_granger_pvalue": _k_ts_expanding_granger_pvalue,
    "ts_expanding_jarque_bera_pvalue": _k_ts_expanding_jarque_bera_pvalue,
    "ts_expanding_kendall_pvalue": _k_ts_expanding_kendall_pvalue,
    "ts_expanding_kpss_pvalue": _k_ts_expanding_kpss_pvalue,
    "ts_expanding_ks_pvalue": _k_ts_expanding_ks_pvalue,
    "ts_expanding_levene_pvalue": _k_ts_expanding_levene_pvalue,
    "ts_expanding_lilliefors_pvalue": _k_ts_expanding_lilliefors_pvalue,
    "ts_expanding_spearman_pvalue": _k_ts_expanding_spearman_pvalue,
    "ts_expanding_adf_pvalue": _k_ts_expanding_adf_pvalue,
    "ts_expanding_ttest_one_sample_pvalue": _k_ts_expanding_ttest_one_sample_pvalue,
    "ts_expanding_ttest_paired_pvalue": _k_ts_expanding_ttest_paired_pvalue,
    "ts_expanding_ttest_two_sample_pvalue": _k_ts_expanding_ttest_two_sample_pvalue,
}


# ---------------------------------------------------------------------------
# registration (R68 protocol, same as batch9)
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


def register_r68_native_batch13() -> list[str]:
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
                (canonical + ":pandas-authority-parity:r68b13").encode()
            ).hexdigest(),
            notes=(
                "R68 batch13 genuine Polars backend: numpy authority kernels over "
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


__all__ = ["register_r68_native_batch13"]

register_r68_native_batch13()
