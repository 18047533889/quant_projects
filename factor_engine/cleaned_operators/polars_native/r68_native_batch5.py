# -*- coding: utf-8 -*-
"""R68 batch5: genuine-Polars backends for the next pandas-delegate canonicals.

Same protocol as ``r68_native_batch3/4``: module-level NumPy authority helpers
called directly on ``pl -> numpy`` column arrays; **no pandas DataFrame is
constructed anywhere** (no ``.to_pandas``, no ``pl.from_pandas``, no
``iterrows``).  Daily-grain intraday kernels build their own daily panel
exactly like ``rolling_pack._pl_rebuild_intraday_result``.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch5"

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


def _day_groups(times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    day = times.astype("datetime64[D]")
    days = np.unique(day)
    return day, [np.flatnonzero(day == d) for d in days]


def _daily_out(base: pl.DataFrame, times: np.ndarray, per_col: dict[str, list[float]]) -> pl.DataFrame:
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


def _safe_ret(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """numpy twin of intraday.overnight._safe_ret (a / b.replace(0,nan) - 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b == 0.0, np.nan, a / b) - 1.0
    return r


# ===========================================================================
# kernels
# ===========================================================================
def _k_ashare_days_since_limit_down(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import (
        _days_since, strict_nonnegative_int,
    )
    ml = b.get("max_lookback")
    limit_v = None if ml is None else strict_nonnegative_int(ml, "max_lookback")
    out = _days_since(_panel(b["limit_down_event"]), limit_v)
    return _rebuild(b["limit_down_event"], out)


def _k_intraday_rv_signature_curvature(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_vol_ext import (
        _rolling_returns, _rv_curvature,
    )
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 240))
    out = _rolling_returns(_panel(b["returns"]), w, _rv_curvature)
    return _rebuild(b["returns"], out)


def _hvg_out(b: dict, key: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.hvg_ext import (
        _hvg_check_params, _hvg_series,
    )
    w, mp, mn, mcf = _hvg_check_params(
        b.get("window", 60), b.get("min_periods", 8), b.get("min_nodes", 10),
        b.get("min_coverage_fraction", 0.5), f"ts_hvg_{key}",
    )
    out = _hvg_series(_panel(b["x"]), w, mp, mn, mcf, key)
    return _rebuild(b["x"], out)


def _k_ts_hvg_clustering_coefficient(b: dict) -> pl.DataFrame:
    return _hvg_out(b, "clustering")


def _k_ts_hvg_assortativity(b: dict) -> pl.DataFrame:
    return _hvg_out(b, "assortativity")


def _k_ts_hvg_motif_entropy(b: dict) -> pl.DataFrame:
    return _hvg_out(b, "motif_entropy")


def _k_cs_ridge_resid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.robust_cs import _design
    y = _panel(b["y"])
    feats = [_panel(b[k]) for k in ("x1", "x2", "x3", "x4") if b.get(k) is not None]
    a = float(b.get("alpha", 0.1))
    add_intercept = bool(b.get("add_intercept", True))
    yv, fv = y, feats
    rows = yv.shape[0]
    out = np.full(yv.shape, np.nan, dtype=float)
    for row in range(rows):
        yrow = yv[row]
        frows = [f[row] for f in fv]
        valid = np.isfinite(yrow)
        for f in frows:
            valid &= np.isfinite(f)
        if valid.sum() < 5:
            continue
        X = _design([f[valid] for f in frows], add_intercept)
        yy = yrow[valid]
        pen = a * np.eye(X.shape[1])
        if add_intercept:
            pen[0, 0] = 0.0
        beta, *_ = np.linalg.lstsq(X.T @ X + pen, X.T @ yy, rcond=None)
        res = np.full(len(yrow), np.nan)
        res[valid] = yrow[valid] - X @ beta
        out[row] = res
    return _rebuild(b["y"], out)


def _k_cs_mahalanobis_distance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.robust_cs import _EPS
    panels = [b[k] for k in ("f1", "f2", "f3", "f4") if b.get(k) is not None]
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
        cov = np.cov(Xv.T)
        try:
            icov = np.linalg.inv(cov + _EPS * np.eye(p))
        except np.linalg.LinAlgError:
            continue
        d = np.sqrt(np.einsum("ij,jk,ik->i", Xv - mu, icov, Xv - mu))
        out[row, valid] = d
    return _rebuild(panels[0], out)


def _k_ts_kalman_beta(b: dict) -> pl.DataFrame:
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
        out[:, c] = _kalman_beta(yv[:, c], xv[:, c], q, r, "beta",
                                 scale_mode=scale_mode, min_warmup=mw)
    return _rebuild(b["y"], out)


def _copula_out(b: dict, entropy: bool) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_local import _copula_cross_series
    a = _panel(b["a"])
    bb = _panel(b["b"])
    g = _strict_int(b.get("grid", 8), "grid", lower=1)
    if g not in (4, 8, 16):
        raise ValueError(f"requires grid in {{4, 8, 16}}; got {g}")
    out = _copula_cross_series(a, bb, g, entropy)
    return _rebuild(b["a"], out)


def _k_cs_rank_copula_entropy(b: dict) -> pl.DataFrame:
    return _copula_out(b, True)


def _k_cs_rank_copula_mi(b: dict) -> pl.DataFrame:
    return _copula_out(b, False)


def _composition_parts(b: dict, keys: list[str]) -> list[np.ndarray]:
    return [_panel(b[k]) for k in keys if b.get(k) is not None]


def _k_composition_clr_component(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composition import _check_zero_policy
    _check_zero_policy(b.get("zero_policy", "reject"))
    parts = _composition_parts(b, ["target", "x1", "x2", "x3", "x4", "x5", "x6", "x7"])
    if len(parts) < 3:
        raise ValueError("composition_clr_component requires >= 3 components")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(a) for a in parts])
    valid = np.all(np.isfinite(logs), axis=0)
    clr = logs[0] - logs.mean(axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = clr[valid]
    return _rebuild(b["target"], out)


def _k_composition_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.composition import _EPS, _check_zero_policy
    _check_zero_policy(b.get("zero_policy", "reject"))
    parts = _composition_parts(b, ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"])
    if len(parts) < 3:
        raise ValueError("composition_entropy requires >= 3 components")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(a) for a in parts])
    valid = np.all(np.isfinite(logs), axis=0)
    sub = logs
    w = np.exp(sub - sub.max(axis=0, keepdims=True))
    p = w / w.sum(axis=0, keepdims=True)
    ent = -np.sum(p * np.log(p + _EPS), axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = ent[valid]
    return _rebuild(b["x1"], out)


def _rqa_fixed_rr_out(b: dict, key: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rqa_ext import _r63_fixed_rr_series
    out = _r63_fixed_rr_series(
        _panel(b["x"]), b.get("window", 60), b.get("dim", 1), b.get("delay", 1),
        b.get("target_rr", 0.05), b.get("min_line", 4), b.get("min_periods", 10),
        key, b.get("theiler"),
    )
    return _rebuild(b["x"], out)


def _k_ts_rqa_determinism_fixed_rr(b: dict) -> pl.DataFrame:
    return _rqa_fixed_rr_out(b, "determinism")


def _k_ts_rqa_laminarity_fixed_rr(b: dict) -> pl.DataFrame:
    return _rqa_fixed_rr_out(b, "laminarity")


def _k_ts_matrix_profile_discord_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.sequence_anomaly import (
        _mp_stats, _validate_mp_geometry,
    )
    m, hist = _validate_mp_geometry(b.get("m", 20), b.get("history_window", 252), "discord")
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _mp_stats(xv[: r + 1, c], m, "discord", hist)
    return _rebuild(b["x"], out)


def _k_index_entry_exit_event(b: dict) -> pl.DataFrame:
    mv = _panel(b["member"])
    rows, cols = mv.shape
    out = np.zeros((rows, cols), dtype=float)
    for col in range(cols):
        prev_state: bool | None = None
        for row in range(rows):
            value = mv[row, col]
            if not np.isfinite(value):
                out[row, col] = np.nan
                prev_state = None
                continue
            state = value != 0
            if prev_state is None:
                out[row, col] = 0.0
            elif state and not prev_state:
                out[row, col] = 1.0
            elif not state and prev_state:
                out[row, col] = -1.0
            else:
                out[row, col] = 0.0
            prev_state = state
    return _rebuild(b["member"], out)


def _k_event_level_survival_share(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.parameter_validation import (
        strict_finite_scalar, strict_integer,
    )
    w = strict_integer(b.get("history_window", 60), "history_window", minimum=2)
    direction_s = str(b.get("direction", "up")).lower()
    if direction_s not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    tol = strict_finite_scalar(b.get("tolerance", 0.0), "tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be >= 0")
    ev = _panel(b["event"])
    lv = _panel(b["level"])
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        cohort: list[tuple[int, float, float, bool]] = []
        for r in range(rows):
            if r >= 1:
                prev = r - 1
                if np.isfinite(ev[prev, c]) and ev[prev, c] != 0.0 and np.isfinite(lv[prev, c]):
                    if direction_s == "up":
                        cohort.append((prev, float(lv[prev, c]), np.inf, False))
                    else:
                        cohort.append((prev, float(lv[prev, c]), -np.inf, False))
            while cohort and (r - cohort[0][0]) > w:
                cohort.pop(0)
            if not cohort:
                continue
            cur = xv[r, c]
            if not np.isfinite(cur):
                cohort = [(er, lev, ext, True) for er, lev, ext, _ in cohort]
                continue
            survived = 0.0
            observed = 0
            for k in range(len(cohort)):
                e_row, lev_, ext, censored = cohort[k]
                if censored:
                    continue
                if direction_s == "up":
                    ext = min(ext, cur)
                else:
                    ext = max(ext, cur)
                cohort[k] = (e_row, lev_, ext, False)
                observed += 1
                if direction_s == "up":
                    if ext >= lev_ - tol:
                        survived += 1.0
                else:
                    if ext <= lev_ + tol:
                        survived += 1.0
            if observed:
                out[r, c] = float(survived / observed)
    return _rebuild(b["x"], out)


def _k_ts_quantile_skew(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.robust_tail import _r63_window_quantiles
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 20))
    ql, qm, qh = float(b.get("q_low", 0.1)), float(b.get("q_mid", 0.5)), float(b.get("q_high", 0.9))
    if not 0.0 < ql < qm < qh < 1.0:
        raise ValueError("require 0 < q_low < q_mid < q_high < 1")
    mp = max(5, int(b.get("min_periods", 5)))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    Q, std, cnt = _r63_window_quantiles(xv, w, (ql, qm, qh))
    denom = Q[2] - Q[0]
    ok = (cnt >= mp) & (std >= 1e-12) & np.isfinite(denom) & (np.abs(denom) >= 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = (Q[2] + Q[0] - 2.0 * Q[1]) / denom
    out = np.where(ok, val, np.nan).reshape(rows, cols)
    return _rebuild(b["x"], out)


def _k_cs_knn_neighbor_retention(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dynamic_knn import (
        _check_decision_clock, _retention_series,
    )
    if b.get("f3") is None:
        raise TypeError("cs_knn_neighbor_retention._calculate_series() missing required argument 'f3'")
    kk = _strict_int(b.get("k", 5), "k", lower=1)
    lg = _strict_int(b.get("lag", 20), "lag", lower=1)
    _check_decision_clock(
        feature_available_at=b.get("feature_available_at", "same_day"),
        target_available_at=None,
    )
    feats = np.stack([_panel(b.get(k)) for k in ("f1", "f2", "f3")], axis=2)
    out = _retention_series(feats, kk, lg)
    return _rebuild(b["f1"], out)


def _k_ts_structural_level_strength(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.strict_params import strict_float
    from factor_engine.cleaned_operators.structural_levels import (
        _check_coverage_fraction, _check_int, _check_positive_float, _strength_series,
    )
    w = _check_int(b.get("window", 120), "window", 2)
    prom = _check_positive_float(b.get("prominence", 0.02), "prominence")
    conf = _check_int(b.get("confirmation", 3), "confirmation", 1)
    dec = strict_float(b.get("decay", 0.05), "decay", minimum=0.0)
    cut = _check_positive_float(b.get("cutoff", 0.05), "cutoff")
    mp = _check_int(b.get("min_periods", 20), "min_periods", 1)
    mcf = _check_coverage_fraction(b.get("min_coverage_fraction", 0.5))
    mpa = b.get("max_pivot_age")
    m_age = w if mpa is None else _check_int(mpa, "max_pivot_age", 1)
    out = _strength_series(_panel(b["price"]), w, prom, conf, dec, cut, mp, mcf, m_age)
    return _rebuild(b["price"], out)


def _k_ts_higuchi_fractal_dimension(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.sequence_complexity import _vec_column_higuchi_fd
    w = _strict_int(b.get("window", 120), "window", lower=2, upper=512)
    km = _strict_int(b.get("k_max", 8), "k_max", lower=1, upper=32)
    out = _vec_column_higuchi_fd(_panel(b["x"]), w, km)
    return _rebuild(b["x"], out)


def _dmd_guard(canonical: str, w: int, rk: int, d: int, dl: int, tk: int | None) -> None:
    from factor_engine.cleaned_operators.dmd import dmd_feasibility
    if not dmd_feasibility(window=w, rank=rk, dim=d, delay=dl, top_k=tk):
        extra = f", top_k={tk}" if tk is not None else ""
        raise ValueError(
            f"{canonical}: DMD parameter domain infeasible "
            f"(window={w}, rank={rk}, dim={d}, delay={dl}{extra})"
        )


def _dmd_concentration_out(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dmd import _dmd_series
    if int(b.get("window", 120)) < 10:
        raise ValueError("ts_dmd_mode_concentration requires window >= 10")
    w, rk, d, dl = (
        int(b.get("window", 120)), int(b.get("rank", 4)),
        int(b.get("dim", 4)), int(b.get("delay", 1)),
    )
    tk = int(b.get("top_k", 2))
    _dmd_guard("ts_dmd_mode_concentration", w, rk, d, dl, tk)
    out = _dmd_series(_panel(b["x"]), w, rk, d, dl, "concentration", tk)
    return _rebuild(b["x"], out)


def _k_ts_dmd_return_mode_concentration(b: dict) -> pl.DataFrame:
    return _dmd_concentration_out(b)


def _k_ts_dmd_level_mode_concentration(b: dict) -> pl.DataFrame:
    return _dmd_concentration_out(b)


def _k_ts_dmd_level_dominant_growth_rate(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dmd import _dmd_series
    if int(b.get("window", 120)) < 10:
        raise ValueError("ts_dmd_level_dominant_growth_rate requires window >= 10")
    w, rk, d, dl = (
        int(b.get("window", 120)), int(b.get("rank", 4)),
        int(b.get("dim", 4)), int(b.get("delay", 1)),
    )
    _dmd_guard("ts_dmd_level_dominant_growth_rate", w, rk, d, dl, None)
    out = _dmd_series(_panel(b["x"]), w, rk, d, dl, "growth")
    return _rebuild(b["x"], out)


def _k_ts_active_information_storage(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.markov_dynamics import _ais_series
    w = _strict_int(b.get("window", 120), "window", lower=2)
    nb = _strict_int(b.get("bins", 3), "bins", lower=2)
    k = _strict_int(b.get("history_length", 1), "history_length", lower=1)
    mh = _strict_int(b.get("min_history", 10), "min_history", lower=2)
    if not 2 <= nb <= 3:
        raise ValueError("ts_active_information_storage requires bins in [2, 3]")
    if not 1 <= k <= 2:
        raise ValueError("ts_active_information_storage requires history_length in [1, 2]")
    if w - k < 5 * nb ** (k + 1):
        raise ValueError(
            "ts_active_information_storage: AIS window cannot supply five "
            f"observations per joint cell; got window={w}, bins={nb}, "
            f"history_length={k}"
        )
    xv = _panel(b["x"])
    cols = xv.shape[1]
    out = np.column_stack([_ais_series(xv[:, c], w, nb, k, mh) for c in range(cols)])
    return _rebuild(b["x"], out)


def _k_cs_sliced_wasserstein_copula_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import (
        _EPS, _SEED, _rank_copula_proj, _sw_copula_shift,
    )
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_integer
    w = strict_integer(b.get("window", 60), "window", minimum=2)
    d_l = strict_integer(b.get("directions", 32), "directions", minimum=4)
    if w < 2 or d_l < 4:
        raise ValueError("cs_sliced_wasserstein_copula_shift requires window >= 2, directions >= 4")
    feats = np.stack([_panel(b[k]) for k in ("f1", "f2", "f3")], axis=2)
    rng = np.random.default_rng(_SEED)
    thetas = rng.normal(0.0, 1.0, size=(d_l, feats.shape[2]))
    thetas /= np.maximum(np.linalg.norm(thetas, axis=1, keepdims=True), _EPS)
    projs: list[np.ndarray | None] = [_rank_copula_proj(feats[t], thetas) for t in range(feats.shape[0])]
    out = np.full(feats.shape[0], np.nan, dtype=float)
    for t in range(feats.shape[0]):
        out[t] = _sw_copula_shift(projs, t, w, d_l)
    base = b["f1"]
    return _rebuild(base, np.repeat(out[:, None], feats.shape[1], axis=1))


def _k_ts_joint_energy_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.distribution_break import _joint_shift_series
    r = strict_int(b.get("recent_window", 20), "recent_window", minimum=2)
    p = strict_int(b.get("prior_window", 60), "prior_window", minimum=2)
    feats = np.stack([_panel(b[k]) for k in ("f1", "f2", "f3")], axis=2)
    out = _joint_shift_series(feats, r, p)
    return _rebuild(b["f1"], out)


def _k_ts_mass_concentration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import check_window, map_rolling
    w = check_window(b.get("window", 20))
    mp = max(2, int(b.get("min_periods", 2)))

    def _fn(chunk: np.ndarray) -> float:
        finite = np.isfinite(chunk)
        v = chunk[finite]
        n = v.size
        if n < mp:
            return np.nan
        if np.any(v < 0.0):
            return np.nan
        scale = float(np.max(v))
        if scale <= 0.0:
            return np.nan
        scaled = v / scale
        shares = scaled / float(scaled.sum())
        hhi = float(np.sum(shares * shares))
        if n <= 1:
            return np.nan
        return float((hhi - 1.0 / n) / (1.0 - 1.0 / n))

    out = map_rolling(_panel(b["weight"]), w, _fn)
    return _rebuild(b["weight"], out)


def _k_ts_location_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_distribution import _location_shift_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    ws = check_window(b.get("recent_window", 20), name="recent_window")
    wl = check_window(b.get("old_window", 40), name="old_window")
    mp = max(3, int(b.get("min_periods", 5)))
    out = _location_shift_series(_panel(b["x"]), ws, wl, mp)
    return _rebuild(b["x"], out)


def _k_ts_change_point_probability(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _regime_filter
    w = int(b.get("window", 120))
    tp = float(b.get("transition_prob", 0.05))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _regime_filter(xv[: r + 1, c], w, "changepoint", tp)
    return _rebuild(b["x"], out)


def _k_ts_bures_corr_shift(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_structure import _bures_shift_series
    from factor_engine.cleaned_operators.closure.strict_scalar import strict_integer
    r = strict_integer(b.get("recent_window", 20), "recent_window", minimum=2)
    p = strict_integer(b.get("prior_window", 60), "prior_window", minimum=2)
    if r < 2 or p < 2:
        raise ValueError("ts_bures_corr_shift requires recent_window, prior_window >= 2")
    mp_raw = b.get("min_pairs")
    if mp_raw is None:
        mp = max(5, int(min(r, p) // 2))
    else:
        mp = strict_integer(mp_raw, "min_pairs", minimum=2)
    if mp > min(r, p):
        raise ValueError("min_pairs must fit both recent and prior windows")
    xv = _panel(b["x"])
    yv = _panel(b["y"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - (r + p) + 1)
            if row - start + 1 < r + p:
                continue
            out[row, col] = _bures_shift_series(
                xv[start : row + 1, col], yv[start : row + 1, col], r, p, mp)
    return _rebuild(b["x"], out)


def _k_intraday_quantile_curve_pca_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_intraday import _pca_score_series
    w = int(b.get("window", 60))
    kk = int(b.get("k", 1))
    if w < 2 or kk < 1:
        raise ValueError("intraday_quantile_curve_pca_score requires window >= 2, k >= 1")
    base = b["returns"]
    times = _time_numpy(base, b.get("session_tz"))
    day, row_groups = _day_groups(times)
    xv = _panel(base)
    per_col: dict[str, list[float]] = {}
    for j, c in enumerate(_ncols(base)):
        day_vals = [xv[idx, j] for idx in row_groups]
        per_col[c] = list(_pca_score_series(day_vals, w, kk))
    return _daily_out(base, times, per_col)


def _k_cs_knn_local_linear_residual(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_local import (
        _KNN_MIN_K, _local_linear_series,
    )
    missing = [n for n in ("target", "f1", "f2", "f3") if b.get(n) is None]
    if missing:
        raise TypeError(
            f"cs_knn_local_linear_residual._calculate_series() missing "
            f"{len(missing)} required positional arguments: {', '.join(missing)}"
        )
    kk = _strict_int(b.get("k", 20), "k", lower=1)
    if kk < _KNN_MIN_K:
        raise ValueError(f"cs_knn_local_linear_residual requires k >= {_KNN_MIN_K}")
    rg = float(b.get("ridge", 1e-3))
    if not np.isfinite(rg) or rg < 0.0:
        raise ValueError("ridge must be a finite number >= 0")
    target = _panel(b["target"])
    feats = np.stack([_panel(b[k]) for k in ("f1", "f2", "f3")], axis=2)
    out = _local_linear_series(target, feats, kk, rg)
    return _rebuild(b["target"], out)


def _k_ts_state_entry_strength(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_state import (
        _DEFAULT_HYSTERESIS_MISSING_POLICY, HysteresisStateKernel,
    )
    hi = float(b.get("upper", 1.0))
    lo = float(b.get("lower", 0.0))
    if not (0.0 <= lo < hi):
        raise ValueError("require 0 <= lower < upper")
    zv = _panel(b["z"])
    policy = b.get("missing_policy", _DEFAULT_HYSTERESIS_MISSING_POLICY)
    rows, cols = zv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        kernel = HysteresisStateKernel(hi, lo, policy)
        for row in range(rows):
            val = zv[row, col]
            kernel.step(val, row)
            if not np.isfinite(val):
                continue
            if kernel.current_state == 0:
                out[row, col] = 0.0
            else:
                out[row, col] = float(kernel.entry_value)
    return _rebuild(b["z"], out)


def _k_ts_gap_survival_duration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.overnight import _EPS as _OV_EPS
    cv_all = _panel(b["close"])
    ov_open = _panel(b["open"])
    pv = _panel(b["pre_close"])
    rows, cols = cv_all.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        ov = _safe_ret(ov_open[:, col], pv[:, col])
        cv = cv_all[:, col]
        pvc = pv[:, col]
        arr = np.full(rows, np.nan, dtype=float)
        gap_age: float = np.nan
        gap_dir = 0.0
        for t in range(rows):
            if not (np.isfinite(ov[t]) and np.isfinite(cv[t]) and np.isfinite(pvc[t])):
                arr[t] = gap_age
                continue
            if abs(ov[t]) > _OV_EPS:
                gap_dir = 1.0 if ov[t] > 0 else -1.0
                gap_age = 0.0
            else:
                if gap_age is not None and np.isfinite(gap_age):
                    gap_age = gap_age + 1.0
                if gap_dir > 0 and cv[t] <= pvc[t]:
                    gap_dir, gap_age = 0.0, np.nan
                elif gap_dir < 0 and cv[t] >= pvc[t]:
                    gap_dir, gap_age = 0.0, np.nan
            arr[t] = gap_age
        out[:, col] = arr
    return _rebuild(b["close"], out)


def _k_ts_overnight_intraday_spread(b: dict) -> pl.DataFrame:
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
            if np.isfinite(ow).sum() < w or np.isfinite(iw).sum() < w:
                continue
            out[r, c] = float(np.nanmean(ow) - np.nanmean(iw))
    return _rebuild(b["close"], out)


def _k_ts_conditional_mutual_information(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.common.strict_params import strict_int
    from factor_engine.cleaned_operators.dependence_ext import _cmi, _triple_series
    from factor_engine.cleaned_operators.rolling_pack import check_window
    w = check_window(b.get("window", 120))
    nb = strict_int(b.get("bins", 3), "bins", minimum=2)
    if nb not in (2, 3):
        raise ValueError("bins must be 2 or 3 (production grid, review R4-50)")
    out = _triple_series(
        _panel(b["x"]), _panel(b["y"]), _panel(b["z"]), w,
        lambda a, bv, cv: _cmi(a, bv, cv, nb),
    )
    return _rebuild(b["x"], out)


def _k_intra_signed_tail_variation_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.higher_moments import (
        _EPS, _check_scale, _signed_jump_stats,
    )
    ts = _check_scale(b.get("threshold_scale", 3.0))
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
                pos, neg, *_ = _signed_jump_stats(v, ts)
                if not np.isfinite(pos) or not np.isfinite(neg):
                    vals.append(np.nan)
                    continue
                vals.append(float((pos - neg) / (pos + neg + _EPS)))
            except excs:
                vals.append(np.nan)
        per_col[c] = vals
    return _daily_out(base, times, per_col)


def _k_ts_variance_ratio_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.ar_meanrev import _variance_ratio_slope_vec
    xv = _panel(b["x"])
    w = int(b.get("window", 120))
    mq = int(b.get("max_q", 10))
    mp = int(b.get("min_periods", 20))
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = _variance_ratio_slope_vec(xv[:, col], w, mq, mp)
    return _rebuild(b["x"], out)


def _k_ts_recurrence_divergence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.recurrence_analysis import (
        _check_params, _recurrence_series, _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    w, d, dl, eq, ml = _check_params(
        b.get("window", 40), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), 2,
    )
    out = _recurrence_series(
        _panel(b["x"]), w, d, dl, eq, ml, b.get("min_periods", 10), 3,
        min_effective_fraction=_RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    return _rebuild(b["x"], out)


def _k_ts_multiscale_entropy_slope(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.complexity import _multiscale_entropy_slope
    ms = int(b.get("max_scale", 5))
    w = int(b.get("window", 300))
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            out[r, c] = _multiscale_entropy_slope(xv[: r + 1, c], ms, w)
    return _rebuild(b["x"], out)


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ashare_days_since_limit_down": _k_ashare_days_since_limit_down,
    "intraday_rv_signature_curvature": _k_intraday_rv_signature_curvature,
    "ts_hvg_clustering_coefficient": _k_ts_hvg_clustering_coefficient,
    "ts_hvg_assortativity": _k_ts_hvg_assortativity,
    "cs_ridge_resid": _k_cs_ridge_resid,
    "ts_kalman_beta": _k_ts_kalman_beta,
    "cs_rank_copula_entropy": _k_cs_rank_copula_entropy,
    "composition_clr_component": _k_composition_clr_component,
    "ts_rqa_determinism_fixed_rr": _k_ts_rqa_determinism_fixed_rr,
    "ts_matrix_profile_discord_score": _k_ts_matrix_profile_discord_score,
    "ts_hvg_motif_entropy": _k_ts_hvg_motif_entropy,
    "index_entry_exit_event": _k_index_entry_exit_event,
    "event_level_survival_share": _k_event_level_survival_share,
    "cs_mahalanobis_distance": _k_cs_mahalanobis_distance,
    "ts_quantile_skew": _k_ts_quantile_skew,
    "cs_knn_neighbor_retention": _k_cs_knn_neighbor_retention,
    "ts_structural_level_strength": _k_ts_structural_level_strength,
    "ts_higuchi_fractal_dimension": _k_ts_higuchi_fractal_dimension,
    "ts_dmd_return_mode_concentration": _k_ts_dmd_return_mode_concentration,
    "cs_rank_copula_mi": _k_cs_rank_copula_mi,
    "ts_active_information_storage": _k_ts_active_information_storage,
    "cs_sliced_wasserstein_copula_shift": _k_cs_sliced_wasserstein_copula_shift,
    "ts_joint_energy_shift": _k_ts_joint_energy_shift,
    "composition_entropy": _k_composition_entropy,
    "ts_mass_concentration": _k_ts_mass_concentration,
    "ts_location_shift": _k_ts_location_shift,
    "ts_change_point_probability": _k_ts_change_point_probability,
    "ts_bures_corr_shift": _k_ts_bures_corr_shift,
    "intraday_quantile_curve_pca_score": _k_intraday_quantile_curve_pca_score,
    "cs_knn_local_linear_residual": _k_cs_knn_local_linear_residual,
    "ts_state_entry_strength": _k_ts_state_entry_strength,
    "ts_rqa_laminarity_fixed_rr": _k_ts_rqa_laminarity_fixed_rr,
    "ts_dmd_level_mode_concentration": _k_ts_dmd_level_mode_concentration,
    "ts_dmd_level_dominant_growth_rate": _k_ts_dmd_level_dominant_growth_rate,
    "ts_gap_survival_duration": _k_ts_gap_survival_duration,
    "ts_conditional_mutual_information": _k_ts_conditional_mutual_information,
    "intra_signed_tail_variation_ratio": _k_intra_signed_tail_variation_ratio,
    "ts_variance_ratio_slope": _k_ts_variance_ratio_slope,
    "ts_recurrence_divergence": _k_ts_recurrence_divergence,
    "ts_overnight_intraday_spread": _k_ts_overnight_intraday_spread,
    "ts_multiscale_entropy_slope": _k_ts_multiscale_entropy_slope,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol, same as batch3/4)
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

    _calculate_series = calculate


def register_r68_native_batch5() -> list[str]:
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
                (canonical + ":pandas-authority-parity:r68b5").encode()
            ).hexdigest(),
            notes=(
                "R68 batch5 genuine Polars backend: numpy kernels over "
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


__all__ = ["register_r68_native_batch5"]

register_r68_native_batch5()
