# -*- coding: utf-8 -*-
"""R68 batch12: genuine-Polars backends for the pandas-delegate canonicals.

Same protocol as ``r68_native_batch5/6/7``: module-level NumPy authority
kernels called directly on ``pl -> numpy`` column arrays; **no pandas
DataFrame is constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``,
no ``iterrows``, no ``apply(axis=1)``).  Wherever the pandas authority ships a
pure-numpy helper (relation/distribution, chip_ops, advanced_information,
regression_models, advanced_topology, tail_systemic, memory_ext,
alpha_language_cross, rolling_pack, ...) that helper is imported and reused
verbatim so the native backend stays bit-identical to the pandas reference.

The only pandas algorithm re-implemented here is ``ewm(alpha, adjust=False,
ignore_na=False)`` (``_ewm_mean_np``), ported line-for-line from the pandas
2.3.3 Cython kernel and verified bit-identical (min_periods 0/3/8, NaN gaps,
constant series).
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

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch12"

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


def _panel_obj(value: Any) -> np.ndarray:
    if isinstance(value, pl.Series):
        value = value.to_frame()
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"expected a polars panel, got {type(value)!r}")
    cols = _ncols(value)
    out = np.empty((value.height, len(cols)), dtype=object)
    for j, c in enumerate(cols):
        out[:, j] = np.asarray(value[c].to_list(), dtype=object)
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


def _check_grid(*panels: pl.DataFrame) -> None:
    """Fail-closed axis gate mirroring ``strict_relation_align`` / ``_aligned``.

    The engine hands the native backend panels already normalized onto one
    grid; a shape mismatch here is the same caller bug the pandas authority
    rejects before any positional pairing.
    """
    if len(panels) < 2:
        return
    base = panels[0]
    base_cols = _ncols(base)
    for position, panel in enumerate(panels[1:], start=1):
        if panel.height != base.height or _ncols(panel) != base_cols:
            raise ValueError(
                f"relation panel {position} has a different index/columns than "
                "panel 0 (fail-closed; no silent reindex)"
            )


def _stack(panels: list[pl.DataFrame]) -> np.ndarray:
    return np.stack([_panel(p) for p in panels], axis=0)


def _rank_panels(b: dict, prefix: str, count: int = 10) -> list[pl.DataFrame]:
    return [
        b[f"{prefix}{i}"] for i in range(1, count + 1)
        if f"{prefix}{i}" in b and b[f"{prefix}{i}"] is not None
    ]


# ---------------------------------------------------------------------------
# pandas ewm(alpha, adjust=False, ignore_na=False) — exact port (pandas 2.3.3)
# ---------------------------------------------------------------------------
def _ewm_mean_np(x: np.ndarray, alpha: float, min_periods: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    old_wt_factor = 1.0 - alpha
    new_wt = alpha  # adjust=False
    weighted = x[0]
    nobs = int(weighted == weighted)
    out[0] = weighted if nobs >= min_periods else np.nan
    old_wt = 1.0
    for i in range(1, n):
        cur = x[i]
        is_obs = cur == cur
        nobs += int(is_obs)
        if weighted == weighted:
            # ignore_na=False: NaN steps still consume the decay.
            old_wt *= old_wt_factor
            if is_obs:
                if weighted != cur:
                    weighted = old_wt * weighted + new_wt * cur
                    weighted /= (old_wt + new_wt)
                old_wt = 1.0
        elif is_obs:
            weighted = cur
        out[i] = weighted if nobs >= min_periods else np.nan
    return out


def _rolling_mean_np(v: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """pandas ``rolling(window, min_periods).mean()`` (NaN-skipping count)."""
    n = v.size
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = v[lo : i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size >= min_periods and valid.size > 0:
            out[i] = float(np.sum(valid) / valid.size)
    return out


# ===========================================================================
# kernels
# ===========================================================================

# --- shareholder ------------------------------------------------------------
def _k_holder_company_ownership_hhi(b: dict) -> pl.DataFrame:
    panels = _rank_panels(b, "s")
    if not panels:
        raise TypeError("holder_company_ownership_hhi requires pandas DataFrame inputs")
    _check_grid(*panels)
    stacked = _stack(panels)
    finite = np.isfinite(stacked)
    squares = np.where(finite, stacked * stacked, np.nan)
    hhi = np.nansum(squares, axis=0)
    hhi = np.where(np.isfinite(squares).sum(axis=0) > 0, hhi, np.nan)
    return _rebuild(panels[0], hhi)


def _k_holder_concentration_change(b: dict) -> pl.DataFrame:
    raise ValueError(
        "holder_concentration_change cannot operate on daily panels; use "
        "factor_engine.storage.sources.relation.relation_snapshot_change"
    )


def _k_holder_count_change_rate(b: dict) -> pl.DataFrame:
    raise ValueError(
        "holder_count_change_rate is undefined for TopTen rows; use distinct holder "
        "snapshot metrics from storage.sources.relation"
    )


def _k_holder_observed_topk_hhi(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.ops import _rank_values
    panels = _rank_panels(b, "s")
    missing_semantic = b.get("missing_semantic", "outside_top_k")
    if len(panels) < 2:
        raise ValueError("holder_observed_topk_hhi requires at least two ranked panels")
    _check_grid(*panels)
    stacked = _stack(panels)
    values = _rank_values(stacked, missing_semantic)
    total = values.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = values / total
        hhi = np.sum(shares * shares, axis=0)
    hhi = np.where(total > 0, hhi, np.nan)
    return _rebuild(panels[0], hhi)


def _k_holder_class_js_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import _holder_js_cell
    cur_panels = [b.get(f"s{i}") for i in range(1, 6)]
    prev_panels = [b.get(f"ps{i}") for i in range(1, 6)]
    if any(p is None for p in cur_panels + prev_panels):
        raise TypeError(
            "holder_class_js_shift requires the five current and five previous "
            "class-share panels (s1..s5, ps1..ps5)"
        )
    _check_grid(*cur_panels, *prev_panels)
    cur = np.stack([_panel(p) for p in cur_panels], axis=2)
    prev = np.stack([_panel(p) for p in prev_panels], axis=2)
    rows, cols, _ = cur.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = _holder_js_cell(cur[i, j], prev[i, j])
    return _rebuild(cur_panels[0], out)


# --- technical --------------------------------------------------------------
def _qqe_columns(v: np.ndarray, length: int, smooth: int, factor: float) -> dict[str, np.ndarray]:
    n = v.size
    delta = np.full(n, np.nan)
    if n > 1:
        delta[1:] = v[1:] - v[:-1]
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    avg_gain = _ewm_mean_np(gain, 1.0 / length, length)
    avg_loss = _ewm_mean_np(loss, 1.0 / length, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
        rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss > 0.0), 0.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi)
    rsi_ma = _rolling_mean_np(rsi, smooth, smooth)
    span = 2 * length - 1
    alpha = 2.0 / (span + 1.0)
    abs_drift = np.full(n, np.nan)
    if n > 1:
        d = rsi_ma[1:] - rsi_ma[:-1]
        abs_drift[1:] = np.abs(d)
    rng = _ewm_mean_np(_ewm_mean_np(abs_drift, alpha, 0), alpha, 0)
    da = rng * factor
    base_long = rsi_ma - da
    base_short = rsi_ma + da
    basis = _rolling_mean_np(rsi_ma, span, span)

    long_band = np.full(n, np.nan)
    short_band = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    carry_long = np.nan
    carry_short = np.nan
    carry_trend = 0.0
    for i in range(n):
        s = rsi_ma[i]
        if not np.isfinite(s):
            carry_long = np.nan
            carry_short = np.nan
            carry_trend = 0.0
            continue
        bl = base_long[i]
        bs = base_short[i]
        if not np.isfinite(bl) or not np.isfinite(bs):
            carry_long = np.nan
            carry_short = np.nan
            carry_trend = 0.0
            continue
        if not np.isfinite(carry_long):
            long_band[i] = bl
            short_band[i] = bs
            trend[i] = 0.0
            carry_long = bl
            carry_short = bs
            carry_trend = 0.0
            continue
        if s > carry_long:
            long_band[i] = max(bl, carry_long)
        else:
            long_band[i] = bl
        if s < carry_short:
            short_band[i] = min(bs, carry_short)
        else:
            short_band[i] = bs
        if s > carry_short:
            carry_trend = 1.0
        elif s < carry_long:
            carry_trend = -1.0
        trend[i] = carry_trend
        carry_long = long_band[i]
        carry_short = short_band[i]
    return {
        "line": rsi_ma,
        "basis": basis,
        "long": long_band,
        "short": short_band,
        "trend": trend,
    }


def _k_QQE(b: dict) -> pl.DataFrame:
    x = b["x"]
    length = max(2, int(b.get("length", 14)))
    smooth = max(1, int(b.get("smooth", 5)))
    factor = float(b.get("factor", 4.236))
    output = str(b.get("output", "line"))
    xv = _panel(x)
    out = np.full_like(xv, np.nan)
    for c in range(xv.shape[1]):
        cols = _qqe_columns(xv[:, c], length, smooth, factor)
        out[:, c] = cols[output]
    return _rebuild(x, out)


def _k_ElderRay(b: dict) -> pl.DataFrame:
    high, low, close = b["high"], b["low"], b["close"]
    ema_p = max(1, int(b.get("ema", 13)))
    output = str(b.get("output", "bull"))
    _check_grid(high, low, close)
    alpha = 2.0 / (ema_p + 1.0)
    e = np.empty_like(_panel(close))
    cv = _panel(close)
    for c in range(cv.shape[1]):
        e[:, c] = _ewm_mean_np(cv[:, c], alpha, ema_p)
    if output == "bull":
        return _rebuild(high, _panel(high) - e)
    if output == "bear":
        return _rebuild(low, _panel(low) - e)
    if output == "spread":
        return _rebuild(high, _panel(high) - _panel(low))
    raise ValueError(f"ElderRay: unknown output {output!r}")


def _k_turnover_chip_age_cost_surface(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.chip_ops import _age_cost_column
    close, turnover = b["close"], b["turnover"]
    window = max(2, int(b.get("window", 120)))
    price_bins = max(2, int(b.get("price_bins", 64)))
    age_bins = max(2, int(b.get("age_bins", 16)))
    output = str(b.get("output", "surface_entropy"))
    _check_grid(close, turnover)
    cv = _panel(close)
    tv = _panel(turnover)
    out = np.full_like(cv, np.nan)
    for c in range(cv.shape[1]):
        col_out = _age_cost_column(cv[:, c], tv[:, c], window, price_bins, age_bins)
        out[:, c] = col_out[output]
    return _rebuild(close, out)


def _k_turnover_chip_overhang_surface(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.chip_ops import _overhang_column
    close, turnover = b["close"], b["turnover"]
    window = max(2, int(b.get("window", 120)))
    bins = max(2, int(b.get("bins", 128)))
    output = str(b.get("output", "overhang_mass"))
    _check_grid(close, turnover)
    cv = _panel(close)
    tv = _panel(turnover)
    out = np.full_like(cv, np.nan)
    for c in range(cv.shape[1]):
        col_out = _overhang_column(cv[:, c], tv[:, c], window, bins, 20.0)
        out[:, c] = col_out[output]
    return _rebuild(close, out)


# --- information / topology / regression / memory ---------------------------
def _k_ts_transfer_entropy_peak_excess(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_information import (
        _rolling_apply_2d_pair, _te_feasibility, _te_peak_window_excess, _TE_LAGS,
    )
    target, source = b["target"], b["source"]
    w = int(b.get("window", 60))
    nb = int(b.get("bins", 3))
    ratio = float(b.get("min_cells_ratio", 1.0))
    ns = max(1, int(b.get("n_surrogates", 20)))
    seed = int(b.get("seed", 0))
    min_transitions = b.get("min_transitions")
    if not (2 <= nb <= 8):
        raise ValueError("ts_transfer_entropy_peak_excess requires 2 <= bins <= 8")
    ok, reason, mt = _te_feasibility(
        window=w, bins=nb, lag=max(_TE_LAGS), min_cells_ratio=ratio,
        min_transitions=min_transitions,
    )
    if not ok:
        raise ValueError(
            f"ts_transfer_entropy_peak_excess {reason}; raise window or lower bins"
        )
    _check_grid(target, source)
    tv = _panel(target)
    sv = _panel(source)
    out = _rolling_apply_2d_pair(
        tv, sv, w,
        lambda a, bb: _te_peak_window_excess(
            a, bb, nb, mt, ratio, n_surrogates=ns, seed=seed
        )[0],
    )
    return _rebuild(target, out)


def _k_ts_huber_regression_in_sample_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.regression_models import _huber_resid_panel
    y, x = b["y"], b["x"]
    w = int(b.get("window", 20))
    mp = max(3, int(b.get("min_periods", 5)))
    _check_grid(y, x)
    yv = _panel(y)
    xv = _panel(x)
    return _rebuild(y, _huber_resid_panel(yv, xv, w=w, mp=mp, predictive=False))


def _k_ts_ridge_regression_in_sample_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.regression_models import _regression_resid
    y, x = b["y"], b["x"]
    w = int(b.get("window", 20))
    mp = max(3, int(b.get("min_periods", 5)))
    a = float(b.get("alpha", 0.1))
    if a < 0.0:
        raise ValueError("alpha must be non-negative")
    _check_grid(y, x)
    yv = _panel(y)
    xv = _panel(x)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            out[row, col] = _regression_resid(
                yv[start : row + 1, col], xv[start : row + 1, col],
                method="ridge", min_periods=mp, alpha=a,
            )
    return _rebuild(y, out)


def _k_ts_betti_1_max_persistence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_topology import _betti_series
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
    x = b["x"]
    w = strict_int(b.get("window", 60), "window", lower=6)
    t = strict_int(b.get("tau", 1), "tau", lower=1)
    m = strict_int(b.get("embedding_dim", 3), "embedding_dim", lower=2, upper=6)
    if w - (m - 1) * t < 3:
        raise ValueError(
            "ts_betti_1_max_persistence requires window-(embedding_dim-1)*tau >= 3 "
            f"(window={w}, embedding_dim={m}, tau={t})"
        )
    return _rebuild(x, _betti_series(_panel(x), w, t, m))


def _k_relation_diffusion_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.tail_systemic import _diffusion_series
    x, group = b["x"], b["group"]
    al = float(b.get("alpha", 0.5))
    st = int(b.get("steps", 2))
    if not (0.0 < al < 1.0):
        raise ValueError("relation_diffusion_score requires 0 < alpha < 1")
    if st < 1 or st > 5:
        raise ValueError("relation_diffusion_score requires 1 <= steps <= 5")
    _check_grid(x, group)
    gv = _panel_obj(group)
    return _rebuild(x, _diffusion_series(_panel(x), gv, al, st))


def _k_ts_fractional_difference_discarded_weight_mass(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.memory_ext import _fd_discarded_weight_mass
    x = b["x"]
    fdv = float(b.get("fd", 0.4))
    if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
        raise ValueError("fd must satisfy -1 < fd < 1")
    cutoff = b.get("cutoff", 20)
    if isinstance(cutoff, (bool, np.bool_)):
        raise ValueError("cutoff must be an integer, not bool")
    cf = float(cutoff)
    if not np.isfinite(cf) or cf != float(int(cf)):
        raise ValueError("cutoff must be an integer")
    c = int(cf)
    if c < 1:
        raise ValueError("cutoff must be >= 1")
    d = _fd_discarded_weight_mass(fdv, c)
    xv = _panel(x)
    return _rebuild(x, np.full(xv.shape, d, dtype=float))


def _k_ts_robust_ema(b: dict) -> pl.DataFrame:
    x = b["x"]
    span = int(b.get("span", 20))
    clip_sigma = float(b.get("clip_sigma", 3.0))
    warmup_window = int(b.get("warmup_window", 20))
    scale_floor = float(b.get("scale_floor", 1e-10))
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if clip_sigma <= 0:
        raise ValueError(f"clip_sigma must be positive, got {clip_sigma}")
    if warmup_window <= 0:
        raise ValueError(f"warmup_window must be positive, got {warmup_window}")
    if scale_floor <= 0:
        raise ValueError(f"scale_floor must be positive, got {scale_floor}")
    alpha = 2.0 / (span + 1)
    xv = _panel(x)
    out = np.full_like(xv, np.nan)
    rows, cols = xv.shape
    for c in range(cols):
        series = xv[:, c]
        col_result = np.full(rows, np.nan, dtype=np.float64)
        y_prev = np.nan
        innovations_buffer: deque = deque(maxlen=warmup_window)
        for i in range(rows):
            x_i = series[i]
            if not np.isfinite(x_i):
                col_result[i] = y_prev
                continue
            if not np.isfinite(y_prev):
                y_prev = x_i
                col_result[i] = x_i
                continue
            e_t = x_i - y_prev
            if len(innovations_buffer) >= warmup_window:
                innovations_array = np.array(innovations_buffer)
                innovations_median = np.median(innovations_array)
                mad = np.median(np.abs(innovations_array - innovations_median))
                s_t = max(1.4826 * mad, scale_floor)
                e_t_clipped = np.clip(e_t, -clip_sigma * s_t, clip_sigma * s_t)
            else:
                e_t_clipped = e_t
            y_t = y_prev + alpha * e_t_clipped
            col_result[i] = y_t
            y_prev = y_t
            innovations_buffer.append(e_t)
        out[:, c] = col_result
    return _rebuild(x, out)


# --- relation aggregation ---------------------------------------------------
def _k_relation_rank_weighted_sum(b: dict) -> pl.DataFrame:
    panels = _rank_panels(b, "s")
    missing_semantic = b.get("missing_semantic", "outside_top_k")
    if len(panels) < 2:
        raise ValueError("relation_rank_weighted_sum requires at least two ranked panels")
    from factor_engine.cleaned_operators.relation.ops import _rank_values
    _check_grid(*panels)
    stacked = _stack(panels)
    values = _rank_values(stacked, missing_semantic)
    ranks = np.arange(1, stacked.shape[0] + 1, dtype=float)[:, None, None]
    weights = 1.0 / ranks
    finite = np.isfinite(values)
    weighted = np.nansum(np.where(finite, values * weights, 0.0), axis=0)
    weight_sum = np.sum(np.where(finite, weights, 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = weighted / weight_sum
    out = np.where(np.isfinite(values).sum(axis=0) > 0, out, np.nan)
    return _rebuild(panels[0], out)


def _relation_category_out(vv: np.ndarray, cv: np.ndarray, signed: bool) -> np.ndarray:
    rows, cols = vv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        g_row = cv[row]
        for label in np.unique(g_row):
            if isinstance(label, float) and np.isnan(label):
                continue
            idx = g_row == label
            if signed:
                denom = float(np.nansum(np.abs(vv[row][idx])))
            else:
                denom = float(np.nansum(vv[row][idx]))
            if not np.isfinite(denom) or denom == 0:
                continue
            out[row][idx] = vv[row][idx] / denom
    return out


def _k_relation_category_share(b: dict) -> pl.DataFrame:
    value, category = b["value"], b["category"]
    _check_grid(value, category)
    vv = _panel(value)
    cv = _panel(category)
    if np.any((vv < 0) & np.isfinite(vv)):
        raise ValueError(
            "relation_category_share requires NON-NEGATIVE value input "
            "(R24-008); negative / signed contributions must use "
            "relation_category_signed_contribution"
        )
    return _rebuild(value, _relation_category_out(vv, cv, signed=False))


def _k_relation_category_signed_contribution(b: dict) -> pl.DataFrame:
    value, category = b["value"], b["category"]
    _check_grid(value, category)
    vv = _panel(value)
    cv = _panel(category)
    return _rebuild(value, _relation_category_out(vv, cv, signed=True))


def _k_relation_peer_weighted_mean_ex_self(b: dict) -> pl.DataFrame:
    value, weight, group = b["value"], b["weight"], b["group"]
    _check_grid(value, weight, group)
    vv = _panel(value)
    wv = _panel(weight)
    gv = _panel(group)
    rows, cols = vv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for row in range(rows):
        g_row = gv[row]
        for label in np.unique(g_row):
            group_idx = np.flatnonzero(g_row == label)
            valid = (
                np.isfinite(vv[row][group_idx])
                & np.isfinite(wv[row][group_idx])
                & (wv[row][group_idx] > 0)
            )
            valid_idx = group_idx[valid]
            if valid_idx.size == 0:
                continue
            weights = wv[row][valid_idx]
            total_w = float(np.sum(weights))
            if not np.isfinite(total_w) or total_w <= 0.0:
                continue
            weighted = float(np.sum(weights * vv[row][valid_idx]))
            for j in valid_idx:
                own_w = wv[row][j]
                denom = total_w - own_w
                if denom <= 0.0:
                    continue
                out[row][j] = (weighted - own_w * vv[row][j]) / denom
    return _rebuild(value, out)


def _k_relation_weighted_change(b: dict) -> pl.DataFrame:
    value, weight = b["value"], b["weight"]
    _check_grid(value, weight)
    vv = _panel(value)
    wv = _panel(weight)
    delta = np.full(vv.shape, np.nan, dtype=float)
    delta[1:, :] = vv[1:, :] - vv[:-1, :]
    return _rebuild(value, delta * wv)


def _k_index_member(b: dict) -> pl.DataFrame:
    member = b["member"]
    mv = _panel(member)
    out = np.where(np.isfinite(mv) & (mv != 0), 1.0, np.where(np.isfinite(mv), 0.0, np.nan))
    return _rebuild(member, out)


def _k_relation_distinct_count(b: dict) -> pl.DataFrame:
    entity_ids = b["entity_ids"]
    arr = _panel_obj(entity_ids)
    rows = []
    for row in arr:
        seen = set()
        for value in row:
            if value is None:
                continue
            text = str(value)
            if text == "nan" or text == "None":
                continue
            seen.add(text)
        rows.append(len(seen))
    counts = np.array(rows, dtype=float)
    cols = len(_ncols(entity_ids))
    return _rebuild(entity_ids, counts.reshape(-1, 1).repeat(cols, axis=1))


def _k_relation_overlap_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.ops import _jaccard
    current_ids, previous_ids = b["current_ids"], b["previous_ids"]
    method = str(b.get("method", "jaccard"))
    _check_grid(current_ids, previous_ids)
    cur_arr = _panel_obj(current_ids)
    prev_arr = _panel_obj(previous_ids)

    def _row_ids(arr: np.ndarray) -> list[set]:
        rows = []
        for row in arr:
            seen = set()
            for value in row:
                if value is None:
                    continue
                text = str(value)
                if text == "nan" or text == "None":
                    continue
                seen.add(text)
            rows.append(seen)
        return rows

    cur = _row_ids(cur_arr)
    prev = _row_ids(prev_arr)
    out = []
    for a, bb in zip(cur, prev):
        if method == "jaccard":
            out.append(_jaccard(a, bb))
        elif method == "overlap":
            inter = len(a & bb)
            out.append(float(inter / min(len(a), len(bb))) if min(len(a), len(bb)) else np.nan)
        else:
            raise ValueError(f"unknown overlap method: {method!r}")
    arr = np.asarray(out, dtype=float).reshape(-1, 1)
    cols = len(_ncols(current_ids))
    return _rebuild(current_ids, arr.repeat(cols, axis=1))


def _k_relation_topk_concentration(b: dict) -> pl.DataFrame:
    panels = _rank_panels(b, "s")
    if len(panels) < 2:
        raise ValueError("relation_topk_concentration requires at least two ranked panels")
    top_k = int(b.get("k", 5))
    if top_k < 1:
        raise ValueError("k must be >= 1")
    _check_grid(*panels)
    stacked = _stack(panels)
    n, rows, cols = stacked.shape
    if top_k > n:
        raise ValueError(
            f"relation_topk_concentration: k={top_k} exceeds the number of "
            f"available relation panels ({n})"
        )
    if np.any(stacked[np.isfinite(stacked)] < 0.0):
        raise ValueError(
            "relation_topk_concentration: relation share/weight must be non-negative"
        )
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            vals = stacked[:, r, c]
            top_slots = vals[:top_k]
            if not np.all(np.isfinite(top_slots)):
                continue
            finite = vals[np.isfinite(vals)]
            if finite.size < 2:
                continue
            total = float(np.sum(finite))
            if total <= 0.0:
                continue
            top = float(np.sum(top_slots))
            out[r, c] = top / total
    return _rebuild(panels[0], out)


def _relation_distribution_panels(b: dict) -> list[pl.DataFrame]:
    rel = b.get("relations")
    if rel is None:
        return []
    if isinstance(rel, (list, tuple)):
        return list(rel)
    return [rel]


def _k_relation_distribution_skew(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _skew
    panels = _relation_distribution_panels(b)
    if len(panels) < 3:
        raise ValueError("relation_distribution_skew requires at least three ranked panels")
    _check_grid(*panels)
    stacked = _stack(panels)
    rows, cols = stacked.shape[1], stacked.shape[2]
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = _skew(stacked[:, r, c])
    return _rebuild(panels[0], out)


def _k_relation_distribution_pearson_kurtosis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _pearson_kurtosis
    panels = _relation_distribution_panels(b)
    if len(panels) < 4:
        raise ValueError(
            "relation_distribution_pearson_kurtosis requires at least four ranked panels"
        )
    _check_grid(*panels)
    stacked = _stack(panels)
    rows, cols = stacked.shape[1], stacked.shape[2]
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = _pearson_kurtosis(stacked[:, r, c])
    return _rebuild(panels[0], out)


def _k_relation_distribution_excess_kurtosis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _pearson_kurtosis
    panels = _relation_distribution_panels(b)
    if len(panels) < 4:
        raise ValueError(
            "relation_distribution_excess_kurtosis requires at least four ranked panels"
        )
    _check_grid(*panels)
    stacked = _stack(panels)
    rows, cols = stacked.shape[1], stacked.shape[2]
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            kurt = _pearson_kurtosis(stacked[:, r, c])
            out[r, c] = kurt - 3.0 if np.isfinite(kurt) else np.nan
    return _rebuild(panels[0], out)


def _delta_np(panel: np.ndarray, window: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows > window:
        cur = panel[window:]
        prev = panel[:-window]
        ok = np.isfinite(cur) & np.isfinite(prev)
        out[window:] = np.where(ok, cur - prev, np.nan)
    return out


def _k_relation_hhi_change(b: dict) -> pl.DataFrame:
    hhi = b["hhi"]
    w = int(b.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    return _rebuild(hhi, _delta_np(_panel(hhi), w))


def _k_relation_entropy_change(b: dict) -> pl.DataFrame:
    entropy = b["entropy"]
    w = int(b.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    return _rebuild(entropy, _delta_np(_panel(entropy), w))


def _k_relation_concentration_acceleration(b: dict) -> pl.DataFrame:
    hhi = b["hhi"]
    w = int(b.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    v = _panel(hhi)
    rows, cols = v.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        prev = r - w
        prev2 = r - 2 * w
        if prev < 0 or prev2 < 0:
            continue
        for c in range(cols):
            if all(np.isfinite(t) for t in (v[r, c], v[prev, c], v[prev2, c])):
                out[r, c] = (v[r, c] - v[prev, c]) - (v[prev, c] - v[prev2, c])
    return _rebuild(hhi, out)


def _mean_panel_change_np(stacked: np.ndarray, window: int) -> np.ndarray:
    n, rows, cols = stacked.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        prev = r - window
        if prev < 0:
            continue
        for c in range(cols):
            diffs: list[float] = []
            for i in range(n):
                a = stacked[i, r, c]
                bb = stacked[i, prev, c]
                if np.isfinite(a) and np.isfinite(bb):
                    diffs.append(abs(float(a - bb)))
            if diffs:
                out[r, c] = float(np.mean(diffs))
    return out


def _k_relation_rank_mobility(b: dict) -> pl.DataFrame:
    panels = _rank_panels(b, "rank")
    if len(panels) < 2:
        raise ValueError("relation_rank_mobility requires at least two ranked panels")
    w = int(b.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    _check_grid(*panels)
    return _rebuild(panels[0], _mean_panel_change_np(_stack(panels), w))


def _k_relation_share_mobility(b: dict) -> pl.DataFrame:
    panels = _rank_panels(b, "s")
    if len(panels) < 2:
        raise ValueError("relation_share_mobility requires at least two ranked panels")
    w = int(b.get("window", 5))
    if w < 1:
        raise ValueError("window must be >= 1")
    _check_grid(*panels)
    return _rebuild(panels[0], _mean_panel_change_np(_stack(panels), w))


def _k_relation_rank_entity_mobility(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _id_value_map
    ranks = [i for i in range(1, 11) if b.get(f"s{i}") is not None]
    if not ranks:
        raise ValueError(
            "relation_rank_entity_mobility requires 4*N panels for 1 <= N <= 10 "
            "(current values, current ids, previous values, previous ids)"
        )
    rank_sets = [
        {i for i in range(1, 11) if b.get(f"{prefix}{i}") is not None}
        for prefix in ("s", "sid", "p", "psid")
    ]
    if any(ranks_set != rank_sets[0] for ranks_set in rank_sets[1:]):
        raise ValueError("current/previous value and id panels must provide the same rank slots")
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("relation rank slots must be a contiguous prefix starting at rank 1")
    cur_panels = [b[f"s{i}"] for i in ranks]
    cur_id_panels = [b[f"sid{i}"] for i in ranks]
    prev_panels = [b[f"p{i}"] for i in ranks]
    prev_id_panels = [b[f"psid{i}"] for i in ranks]
    _check_grid(*cur_panels, *cur_id_panels, *prev_panels, *prev_id_panels)
    cur = _stack(cur_panels)
    cur_id = np.stack([_panel_obj(p) for p in cur_id_panels], axis=0)
    prev = _stack(prev_panels)
    prev_id = np.stack([_panel_obj(p) for p in prev_id_panels], axis=0)
    _, rows, cols = cur.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for c in range(cols):
            cur_map = _id_value_map(cur_id[:, r, c], cur[:, r, c])
            prev_map = _id_value_map(prev_id[:, r, c], prev[:, r, c])
            if cur_map is None or prev_map is None:
                continue
            common = [k for k in cur_map if k in prev_map]
            if not common:
                continue
            out[r, c] = float(
                np.mean([abs(cur_map[k] - prev_map[k]) for k in common])
            )
    return _rebuild(cur_panels[0], out)


# --- rolling shape ----------------------------------------------------------
def _k_ts_effective_turning_rate(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    from factor_engine.cleaned_operators.alpha_language_shape import _trailing_contiguous
    x = b["x"]
    w = check_window(b.get("window", 20))
    eps = float(b.get("epsilon", 0.0))
    mp = max(2, int(b.get("min_periods", 2)))
    xv = _panel(x)

    def _sign(d: float) -> int:
        if d > eps:
            return 1
        if d < -eps:
            return -1
        return 0

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        if v.size < mp + 1:
            return np.nan
        d = np.diff(v)
        flips = 0
        active_pairs = 0
        for i in range(1, d.size):
            s0 = _sign(float(d[i - 1]))
            s1 = _sign(float(d[i]))
            if s0 == 0 or s1 == 0:
                continue
            active_pairs += 1
            if s0 != s1:
                flips += 1
        if active_pairs == 0:
            return np.nan
        return float(flips / active_pairs)

    return _rebuild(x, map_rolling(xv, w, _fn))


def _k_ts_weighted_time_centroid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    from factor_engine.cleaned_operators.alpha_language_shape import _trailing_contiguous
    weight = b["weight"]
    w = check_window(b.get("window", 20))
    mp = max(2, int(b.get("min_periods", 2)))
    wv = _panel(weight)

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        n = v.size
        if n < mp:
            return np.nan
        if np.any(v < 0.0):
            return np.nan
        scale = float(np.max(v))
        if scale <= 0.0 or n <= 1:
            return np.nan
        scaled = v / scale
        shares = scaled / float(scaled.sum())
        pos = np.arange(n, dtype=float)
        return float(2.0 * np.dot(pos, shares) / (n - 1.0) - 1.0)

    return _rebuild(weight, map_rolling(wv, w, _fn))


def _k_relation_weighted_std_ex_self(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_cross import _group_peers_indices
    x, weight, group = b["x"], b["weight"], b["group"]
    _check_grid(x, weight, group)
    xv = _panel(x)
    wv = _panel(weight)
    gv = _panel(group)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    minp = max(2, int(b.get("min_peers", 2)))
    for row in range(rows):
        g_row = gv[row]
        for label in np.unique(g_row):
            peers = _group_peers_indices(g_row, label, np.isfinite(xv[row]) & np.isfinite(wv[row]))
            if peers.size <= minp:
                continue
            for j in peers:
                others = peers[peers != j]
                w = wv[row, others]
                if np.any(w < 0.0):
                    continue
                total_w = float(w.sum())
                if total_w <= 0.0:
                    continue
                w = w / total_w
                vals = xv[row, others]
                mu = float(np.sum(w * vals))
                var = float(np.sum(w * (vals - mu) ** 2))
                out[row, j] = float(np.sqrt(max(var, 0.0)))
    return _rebuild(x, out)


# --- group / cross-section --------------------------------------------------
def _valid_label_np(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(float(value)):
        return False
    return True


def _group_shape_np(xv: np.ndarray, gv: np.ndarray, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        row_group = gv[r]
        row_x = xv[r]
        labels = {v for v in row_group if _valid_label_np(v)}
        for label in labels:
            mask = row_group == label
            values = row_x[mask]
            valid = values[np.isfinite(values)]
            if valid.size < 3:
                continue
            value = fn(valid)
            if np.isfinite(value):
                out[r, np.flatnonzero(mask)] = value
    return out


def _k_group_skewness(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _skew
    x, group = b["x"], b["group"]
    _check_grid(x, group)
    return _rebuild(x, _group_shape_np(_panel(x), _panel(group), _skew))


def _k_group_kurtosis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _kurtosis
    x, group = b["x"], b["group"]
    _check_grid(x, group)
    return _rebuild(x, _group_shape_np(_panel(x), _panel(group), _kurtosis))


def _k_group_quantile_spread(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _has_spread
    x, group = b["x"], b["group"]
    ql = float(b.get("q_low", 0.25))
    qh = float(b.get("q_high", 0.75))
    if not 0.0 < ql < qh < 1.0:
        raise ValueError("require 0 < q_low < q_high < 1")
    _check_grid(x, group)

    def _fn(values: np.ndarray) -> float:
        if values.size < 3 or not _has_spread(values):
            return np.nan
        return float(np.quantile(values, qh) - np.quantile(values, ql))

    return _rebuild(x, _group_shape_np(_panel(x), _panel(group), _fn))


def _k_group_tail_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.relation.distribution import _has_spread
    x, group = b["x"], b["group"]
    ql = float(b.get("q_low", 0.05))
    qh = float(b.get("q_high", 0.95))
    if not 0.0 < ql < qh < 1.0:
        raise ValueError("require 0 < q_low < q_high < 1")
    _check_grid(x, group)

    def _fn(values: np.ndarray) -> float:
        if values.size < 3 or not _has_spread(values):
            return np.nan
        q_lo = float(np.quantile(values, ql))
        q_hi = float(np.quantile(values, qh))
        if abs(q_lo) < 1e-12:
            return np.nan
        return float(abs(q_hi) / abs(q_lo))

    return _rebuild(x, _group_shape_np(_panel(x), _panel(group), _fn))


def _k_cs_isolation(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_cross import _row_local_gaps
    x = b["x"]
    kk = int(b.get("k", 5))
    if kk < 1:
        raise ValueError("k must be >= 1")
    group = b.get("group")
    if group is not None:
        _check_grid(x, group)
        gv = _panel(group)
    else:
        gv = None
    xv = _panel(x)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        gp, gm, _ = _row_local_gaps(xv[row], None if gv is None else gv[row], kk, finite[row])
        gp_ok = np.isfinite(gp)
        gm_ok = np.isfinite(gm)
        both = gp_ok & gm_ok
        out[row, both] = np.sqrt(gp[both] * gm[both])
        only_up = gp_ok & ~gm_ok
        out[row, only_up] = gp[only_up]
        only_down = ~gp_ok & gm_ok
        out[row, only_down] = gm[only_down]
    return _rebuild(x, out)


def _k_group_ex_self_std(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_cross import _group_ex_self_std_row
    x, group = b["x"], b["group"]
    _check_grid(x, group)
    xv = _panel(x)
    gv = _panel(group)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    for row in range(rows):
        out[row] = _group_ex_self_std_row(xv[row], gv[row], finite[row], max(2, int(b.get("min_peers", 2))))
    return _rebuild(x, out)


def _k_group_ex_self_mad(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_cross import _group_peers_indices
    x, group = b["x"], b["group"]
    _check_grid(x, group)
    xv = _panel(x)
    gv = _panel(group)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    finite = np.isfinite(xv)
    minp = max(3, int(b.get("min_peers", 3)))
    for row in range(rows):
        g_row = gv[row]
        for label in np.unique(g_row):
            peers = _group_peers_indices(g_row, label, finite[row])
            if peers.size <= minp:
                continue
            for j in peers:
                others = peers[peers != j]
                if others.size < minp:
                    continue
                vals = xv[row, others]
                med = float(np.median(vals))
                out[row, j] = float(np.median(np.abs(vals - med)))
    return _rebuild(x, out)


# --- fiscal divergence ------------------------------------------------------
def _walk_growth_np(x2d: np.ndarray, p2d: np.ndarray) -> np.ndarray:
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        _default_require_parseable, _lag_value, _period_insert, _period_key,
    )
    from factor_engine.cleaned_operators.fundamental.accruals_scores import _pct_change
    require_parseable = _default_require_parseable()
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        order: list[object] = []
        visible: dict[object, float] = {}
        for i in range(rows):
            value = x2d[i, c]
            raw_period = p2d[i, c]
            key = _period_key(raw_period)
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


def _k_fin_expense_sales_divergence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fiscal_strict import (
        GROWTH_FORBIDDEN_FLOW_TYPES, flow_types, reject_ytd_growth,
    )
    a_panel, b_panel, pid = b["period_expense"], b["operating_revenue"], b["period_id"]
    flow_type = b.get("flow_type")
    reject_ytd_growth("fin_growth", flow_type)
    if flow_type is not None:
        types = flow_types(flow_type, 2)
        for slot_label, t in zip(("a", "b"), types):
            if t in GROWTH_FORBIDDEN_FLOW_TYPES:
                raise ValueError(
                    f"fin_divergence slot {slot_label}: growth over a "
                    f"CumulativeYTDFlow input is not a period growth rate; "
                    f"convert with fin_quarter_from_cumulative first."
                )
    _check_grid(a_panel, b_panel, pid)
    av = _panel(a_panel)
    bv = _panel(b_panel)
    pv = _panel(pid)
    ga = _walk_growth_np(av, pv)
    gb = _walk_growth_np(bv, pv)
    with np.errstate(invalid="ignore"):
        return _rebuild(a_panel, ga - gb)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "holder_company_ownership_hhi": _k_holder_company_ownership_hhi,
    "holder_concentration_change": _k_holder_concentration_change,
    "holder_count_change_rate": _k_holder_count_change_rate,
    "holder_observed_topk_hhi": _k_holder_observed_topk_hhi,
    "holder_class_js_shift": _k_holder_class_js_shift,
    "QQE": _k_QQE,
    "ElderRay": _k_ElderRay,
    "turnover_chip_age_cost_surface": _k_turnover_chip_age_cost_surface,
    "turnover_chip_overhang_surface": _k_turnover_chip_overhang_surface,
    "ts_transfer_entropy_peak_excess": _k_ts_transfer_entropy_peak_excess,
    "relation_rank_weighted_sum": _k_relation_rank_weighted_sum,
    "relation_category_share": _k_relation_category_share,
    "relation_category_signed_contribution": _k_relation_category_signed_contribution,
    "relation_peer_weighted_mean_ex_self": _k_relation_peer_weighted_mean_ex_self,
    "relation_weighted_change": _k_relation_weighted_change,
    "index_member": _k_index_member,
    "relation_distinct_count": _k_relation_distinct_count,
    "relation_overlap_ratio": _k_relation_overlap_ratio,
    "ts_huber_regression_in_sample_resid": _k_ts_huber_regression_in_sample_resid,
    "ts_ridge_regression_in_sample_resid": _k_ts_ridge_regression_in_sample_resid,
    "relation_topk_concentration": _k_relation_topk_concentration,
    "relation_distribution_skew": _k_relation_distribution_skew,
    "relation_distribution_pearson_kurtosis": _k_relation_distribution_pearson_kurtosis,
    "relation_distribution_excess_kurtosis": _k_relation_distribution_excess_kurtosis,
    "relation_hhi_change": _k_relation_hhi_change,
    "relation_entropy_change": _k_relation_entropy_change,
    "relation_concentration_acceleration": _k_relation_concentration_acceleration,
    "relation_rank_mobility": _k_relation_rank_mobility,
    "relation_rank_entity_mobility": _k_relation_rank_entity_mobility,
    "relation_share_mobility": _k_relation_share_mobility,
    "ts_effective_turning_rate": _k_ts_effective_turning_rate,
    "ts_weighted_time_centroid": _k_ts_weighted_time_centroid,
    "relation_weighted_std_ex_self": _k_relation_weighted_std_ex_self,
    "ts_betti_1_max_persistence": _k_ts_betti_1_max_persistence,
    "relation_diffusion_score": _k_relation_diffusion_score,
    "ts_fractional_difference_discarded_weight_mass": _k_ts_fractional_difference_discarded_weight_mass,
    "ts_robust_ema": _k_ts_robust_ema,
    "fin_expense_sales_divergence": _k_fin_expense_sales_divergence,
    "group_skewness": _k_group_skewness,
    "group_kurtosis": _k_group_kurtosis,
    "group_quantile_spread": _k_group_quantile_spread,
    "group_tail_ratio": _k_group_tail_ratio,
    "cs_isolation": _k_cs_isolation,
    "group_ex_self_std": _k_group_ex_self_std,
    "group_ex_self_mad": _k_group_ex_self_mad,
}


# ---------------------------------------------------------------------------
# registration (R68 protocol, same as batch5/6/7)
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


def register_r68_native_batch12() -> list[str]:
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
        # Expose the pandas authority signature so ``_kernel_param_defaults``
        # sees the authored defaults (optional rank slots s3..s10, lag, ...)
        # instead of treating every declared param as required.
        op._contract_callable = ref._calculate_series
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
                (canonical + ":pandas-authority-parity:r68b12").encode()
            ).hexdigest(),
            notes=(
                "R68 batch12 genuine Polars backend: numpy kernels over "
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


__all__ = ["register_r68_native_batch12"]

register_r68_native_batch12()
