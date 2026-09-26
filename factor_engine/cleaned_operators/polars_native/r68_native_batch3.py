# -*- coding: utf-8 -*-
"""R68 batch3: genuine-Polars backends for 45 pandas-delegate canonicals.

Replaces the ``polars_udf_pandas_delegate`` slots (``rolling_pack._PolarsUdf``
pandas round-trip) with native implementations:

* pure ``pl.Expr`` kernels for the formula-shaped operators (rolling / ewm /
  arithmetic) — no pandas, no NumPy round-trip, no ``to_pandas``;
* NumPy kernels + ``pl.Series`` rebuild for the complex-algorithm operators
  (Hankel SVD, Kalman recursion, matrix profile, survival state machine, ...).
  The NumPy authority kernel is called directly on ``pl -> numpy`` column
  arrays; **no pandas DataFrame is constructed anywhere** (no ``.to_pandas``,
  no ``pl.from_pandas``, no ``iterrows``).

Registration runs at import (ordered loader window, before gap coverage) and
follows the R65 native-replacement protocol: ``replace_backend`` + registry
register + ``record_backend_replacement_after`` + explicit
``PhysicalImplementationSpec(execution_kind=POLARS_NATIVE_EXPR)``.
First registrant wins: a canonical whose polars slot is already non-delegate
native is skipped.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from typing import Any, Callable

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator

_SOURCE = "factor_engine.cleaned_operators.polars_native.r68_native_batch3"

# Axis / identity columns never treated as instrument features.
_SKIP = frozenset({
    "__fe_time__", "date", "timestamp", "trade_date", "datetime",
    "stock_code", "instrument", "symbol", "session", "__fe_instrument__",
})

# polars 1.x renamed rolling ``min_periods`` -> ``min_samples``.
_ROLL_KW = "min_samples"
if "min_samples" not in inspect.signature(pl.Expr.rolling_sum).parameters:
    _ROLL_KW = "min_periods"
_EWM_KW = "min_samples"
if "min_samples" not in inspect.signature(pl.Expr.ewm_mean).parameters:
    _EWM_KW = "min_periods"


# ---------------------------------------------------------------------------
# pl panel <-> numpy helpers (NO pandas conversion anywhere)
# ---------------------------------------------------------------------------
def _ncols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _SKIP]


def _col(frame: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False).fill_nan(None)


def _finite(expr: pl.Expr) -> pl.Expr:
    return expr.is_finite().fill_null(False)


def _roll(expr: pl.Expr, w: int, kind: str, mp: int | None = None) -> pl.Expr:
    fn = getattr(expr, f"rolling_{kind}")
    if mp is None:
        return fn(window_size=int(w))
    return fn(**{"window_size": int(w), _ROLL_KW: int(mp)})


def _panel(value: Any) -> np.ndarray:
    """A polars wide panel -> (rows, n_instruments) float64 (NaN for nulls)."""
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


def _combine(panels: list[pl.DataFrame]) -> pl.DataFrame:
    """Horizontally stack panels' numeric columns under unique prefixed names.

    Expr kernels for multi-panel operators must be evaluated on ONE frame;
    with same-named instrument columns a bare ``pl.col(c)`` would silently
    resolve to the wrong panel's column.
    """
    data = {}
    for i, f in enumerate(panels):
        for c in _ncols(f):
            data[f"__p{i}__{c}"] = f[c].cast(pl.Float64, strict=False)
    return pl.DataFrame(data)


def _finish(base: pl.DataFrame, res: pl.DataFrame) -> pl.DataFrame:
    """Write the per-column results (named by instrument) back onto ``base``."""
    cols = _ncols(base)
    return base.with_columns([
        pl.Series(c, res[c].to_numpy(allow_copy=True), dtype=pl.Float64)
        for c in cols
    ])


def _pcol(i: int, c: str) -> pl.Expr:
    return pl.col(f"__p{i}__{c}").fill_nan(None)


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
    """The panel's time column as naive datetime64[ns] (session wall clock).

    Mirrors ``intraday._core.session_local``: a tz-aware axis is converted to
    ``tz or _SESSION_TZ`` wall clock and de-localised; a naive axis is
    assumed local and passes through untouched.
    """
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


# ===========================================================================
# pure pl.Expr kernels
# ===========================================================================
def _k_ashare_failed_limit_count(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ashare.state_machine import _tolerance
    high, low, close = b["high"], b["low"], b["close"]
    w = _strict_int(b.get("window", 20), "window", lower=2)
    tol = _tolerance(b.get("tick_tolerance", 0.005))
    side = str(b.get("side", "up")).lower()
    if side not in {"up", "down"}:
        raise ValueError("side must be 'up' or 'down'")
    cols = _ncols(high)
    comb = _combine([high, low, close, b["high_limit"], b["low_limit"]])
    exprs = []
    for c in cols:
        H, L, C = _pcol(0, c), _pcol(1, c), _pcol(2, c)
        HL, LL = _pcol(3, c), _pcol(4, c)
        if side == "up":
            known = H.is_not_null() & C.is_not_null() & HL.is_not_null()
            touched = H >= HL * (1.0 - tol)
            held = C < HL * (1.0 - tol)
        else:
            known = L.is_not_null() & C.is_not_null() & LL.is_not_null()
            touched = L <= LL * (1.0 + tol)
            held = C > LL * (1.0 + tol)
        mark = pl.when(known & touched & held).then(1.0).otherwise(
            pl.when(known).then(0.0).otherwise(None))
        s = _roll(mark, w, "sum", mp=1)
        exprs.append(pl.when(known).then(s).otherwise(None)
                     .fill_null(float("nan")).cast(pl.Float64).alias(c))
    return _finish(high, comb.select(exprs))


def _k_vwap_distance_pct(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.indicators_v2 import _pi
    close, volume = b["close"], b["volume"]
    w = _pi(b.get("window", 20), "window", 2)
    cols = _ncols(close)
    comb = _combine([close, volume])
    exprs = []
    for c in cols:
        C, V = _pcol(0, c), _pcol(1, c)
        pos = pl.when(C > 0).then(C).otherwise(None)
        valid = C.is_not_null() & V.is_not_null() & (V > 0)
        num = _roll(pl.when(valid).then(C * V).otherwise(None), w, "sum", mp=w)
        den = _roll(pl.when(valid).then(V).otherwise(None), w, "sum", mp=w)
        vwap = num / pl.when(den > 0).then(den).otherwise(None)
        exprs.append(((pos - vwap) / pos).fill_null(float("nan")).cast(pl.Float64).alias(c))
    return _finish(close, comb.select(exprs))


def _k_donchian_channel_position(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.indicators_v2 import _pi
    high, low, close = b["high"], b["low"], b["close"]
    w = _pi(b.get("window", 20), "window", 2)
    cols = _ncols(high)
    comb = _combine([high, low, close])
    exprs = []
    for c in cols:
        H, L, C = _pcol(0, c), _pcol(1, c), _pcol(2, c)
        u = _roll(H, w, "max", mp=w)
        lo = _roll(L, w, "min", mp=w)
        width = u - lo
        den = pl.when(width != 0).then(width).otherwise(None)
        exprs.append(((C - lo) / den).fill_null(float("nan")).cast(pl.Float64).alias(c))
    return _finish(high, comb.select(exprs))


def _k_consolidation_range_pct(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.indicators_v2 import _pi
    close = b["close"]
    w = _pi(b.get("window", 20), "window", 2)
    exprs = []
    for c in _ncols(close):
        v = _col(close, c)
        pos = pl.when(v > 0).then(v).otherwise(None)
        hi = _roll(pos, w, "max", mp=w)
        lo = _roll(pos, w, "min", mp=w)
        exprs.append(((hi - lo) / pos).fill_null(float("nan")).cast(pl.Float64).alias(c))
    return close.with_columns(exprs)


def _k_ts_gap_reversion_ratio(b: dict) -> pl.DataFrame:
    close, open_px, pre_close = b["close"], b["open"], b["pre_close"]
    thr = float(b.get("threshold", 0.01))
    cols = _ncols(close)
    comb = _combine([close, open_px, pre_close])
    exprs = []
    for c in cols:
        C, O, P = _pcol(0, c), _pcol(1, c), _pcol(2, c)
        pc = pl.when(P != 0).then(P).otherwise(None)
        oc = pl.when(O != 0).then(O).otherwise(None)
        o = O / pc - 1.0
        i = C / oc - 1.0
        oa = o.abs()
        den = pl.when(oa != 0).then(oa).otherwise(None)
        ratio = -i / den
        exprs.append(pl.when(oa > thr).then(ratio).otherwise(None)
                     .fill_null(float("nan")).cast(pl.Float64).alias(c))
    return _finish(close, comb.select(exprs))


def _k_event_frequency(b: dict) -> pl.DataFrame:
    condition = b["condition"]
    w = _strict_int(b.get("window", 20), "window", lower=2)
    mp = max(1, _strict_int(b.get("min_periods", 1), "min_periods", lower=1))
    arr = _panel(condition)
    bad = np.isfinite(arr) & (arr != 0.0) & (arr != 1.0)
    if bad.any():
        raise ValueError(
            f"condition must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )
    exprs = []
    for c in _ncols(condition):
        v = _col(condition, c)
        fin = _finite(v)
        n = _roll(fin.cast(pl.Float64), w, "sum", mp=1)
        hits = _roll((fin & (v != 0)).cast(pl.Float64), w, "sum", mp=1)
        exprs.append(pl.when(n >= mp).then(hits / n).otherwise(None)
                     .fill_null(float("nan")).cast(pl.Float64).alias(c))
    return condition.with_columns(exprs)


def _k_ts_upper_partial_moment(b: dict) -> pl.DataFrame:
    x = b["x"]
    w = _strict_int(b.get("window", 20), "window", lower=2)
    thr = float(b.get("threshold", 0.0))
    ord_ = float(b.get("order", 1.0))
    if ord_ < 1.0:
        raise ValueError("order must be >= 1")
    mp = max(2, _strict_int(b.get("min_periods", 2), "min_periods", lower=2))
    exprs = []
    for c in _ncols(x):
        v = pl.when(_finite(_col(x, c))).then(_col(x, c)).otherwise(None)
        above = (v - thr).clip(0.0, None)
        s = _roll(above.pow(ord_), w, "sum", mp=mp)
        n = _roll(_finite(_col(x, c)).cast(pl.Float64), w, "sum", mp=1)
        exprs.append(pl.when(n >= mp).then(s / n).otherwise(None)
                     .fill_null(float("nan")).cast(pl.Float64).alias(c))
    return x.with_columns(exprs)


# ===========================================================================
# NumPy kernels (polars -> numpy -> authority kernel -> pl.Series rebuild)
# ===========================================================================
def _k_relation_hhi(b: dict) -> pl.DataFrame:
    names = [f"s{i}" for i in range(1, 11) if b.get(f"s{i}") is not None]
    if len(names) < 2:
        raise ValueError("relation_hhi requires at least two ranked panels")
    arrs = [_panel(b[n]) for n in names]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs[1:]):
        raise ValueError("relation_hhi panels must share the same grid")
    stacked = np.stack(arrs, axis=0)
    ms = str(b.get("missing_semantic", "outside_top_k"))
    choices = ("structural_zero", "outside_top_k", "not_reported",
               "source_missing", "unknown")
    if ms not in choices:
        raise ValueError(f"missing_semantic must be one of {choices!r}; got {ms!r}")
    if ms in ("structural_zero", "outside_top_k"):
        values = np.nan_to_num(stacked, nan=0.0)
    else:
        values = stacked
    total = values.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = values / total
        hhi = np.sum(shares * shares, axis=0)
    hhi = np.where(total > 0, hhi, np.nan)
    return _rebuild(b[names[0]], hhi)


def _k_ts_hankel_singular_gap(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.hankel import (
        _check_hankel_params, _hankel_singular_gap_series,
    )
    w, e, _, mcf = _check_hankel_params(
        b.get("window", 60), b.get("embedding_dim", 15), None,
        b.get("min_contiguous_fraction", 0.8),
    )
    out = _hankel_singular_gap_series(_panel(b["x"]), w, e, "strict_contiguous", mcf)
    return _rebuild(b["x"], out)


def _k_ts_vector_state_local_density(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.candle_state_space import _local_density_series
    out = _local_density_series(
        _panel(b["f1"]), _panel(b["f2"]), _panel(b["f3"]), _panel(b["f4"]),
        b.get("window", 60), b.get("k", 5),
    )
    return _rebuild(b["f1"], out)


def _k_ts_vector_state_mahalanobis(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.candle_state_space import _mahalanobis_series
    out = _mahalanobis_series(
        _panel(b["f1"]), _panel(b["f2"]), _panel(b["f3"]), _panel(b["f4"]),
        b.get("window", 60), b.get("shrinkage", 0.5),
    )
    return _rebuild(b["f1"], out)


def _matrix_profile_out(b: dict, index: int) -> pl.DataFrame:
    from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
    arrs = _matrix_profile_series(
        _panel(b["x"]), b.get("window", 120),
        b.get("subsequence_length", 10), b.get("history", 80),
    )
    return _rebuild(b["x"], arrs[index])


def _k_ts_matrix_profile_neighbor_dispersion(b: dict) -> pl.DataFrame:
    return _matrix_profile_out(b, 3)


def _k_ts_matrix_profile_motif_frequency(b: dict) -> pl.DataFrame:
    return _matrix_profile_out(b, 2)


def _k_ts_recurrence_diagonal_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.recurrence_analysis import (
        _check_params, _recurrence_series, _RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    w, d, dl, eq, ml = _check_params(
        b.get("window", 40), b.get("dim", 1), b.get("delay", 1),
        b.get("eps_fraction", 0.1), 2,
    )
    out = _recurrence_series(
        _panel(b["x"]), w, d, dl, eq, ml, b.get("min_periods", 10), 1,
        min_effective_fraction=_RQA_MIN_EFFECTIVE_FRACTION_DEFAULT,
    )
    return _rebuild(b["x"], out)


def _k_ts_local_lyapunov_exponent(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series
    w = _strict_int(b.get("window", 120), "window", lower=2)
    tau_i = _strict_int(b.get("tau", 1), "tau", lower=1)
    dim = _strict_int(b.get("embedding_dim", 3), "embedding_dim", lower=2, upper=6)
    H = _strict_int(b.get("horizon", 5), "horizon", lower=1)
    ma = _strict_int(b.get("min_anchors", 3), "min_anchors", lower=1)
    pt_raw = b.get("physical_time", False)
    if not isinstance(pt_raw, (bool, np.bool_)):
        raise ValueError("physical_time must be boolean")
    pt = bool(pt_raw)
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = _lyapunov_series(xv[:, c], w, tau_i, dim, H, ma, pt)
    return _rebuild(b["x"], out)


def _survival_out(b: dict, index: int) -> pl.DataFrame:
    from factor_engine.cleaned_operators.stateful.survival import _survival_kernel
    hw = _strict_int(b.get("history_window", 60), "history_window", lower=1)
    mc = _strict_int(b.get("min_completed_runs", 5), "min_completed_runs", lower=1)
    ip = b.get("inactive_policy", "nan")
    gp = b.get("gap_policy", "nan")
    sv = _panel(b["state"])
    rows, cols = sv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        res = _survival_kernel(sv[:, col], hw, mc, mc, mc, 1.0,
                               inactive_policy=ip, gap_policy=gp)
        out[:, col] = res[index]
    return _rebuild(b["state"], out)


def _k_ts_state_age_percentile(b: dict) -> pl.DataFrame:
    return _survival_out(b, 1)


def _k_ts_state_residual_life(b: dict) -> pl.DataFrame:
    return _survival_out(b, 3)


def _k_ts_conditional_transfer_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.conditional_dependence import (
        _tri_rolling, _conditional_te_window,
    )
    w = _strict_int(b.get("window", 60), "window", lower=2)
    nb = _strict_int(b.get("bins", 2), "bins", lower=2)
    lg = _strict_int(b.get("lag", 1), "lag", lower=1)
    ratio_raw = b.get("min_cells_ratio", 1.0)
    if isinstance(ratio_raw, (bool, np.bool_)) or not isinstance(ratio_raw, (int, float, np.integer, np.floating)):
        raise ValueError("min_cells_ratio must be a float")
    ratio = float(ratio_raw)
    if ratio < 0.0:
        raise ValueError("min_cells_ratio must be >= 0")
    if nb not in (2, 3):
        raise ValueError("ts_conditional_transfer_entropy requires bins in {2, 3}")
    if lg < 1:
        raise ValueError("ts_conditional_transfer_entropy requires lag >= 1")
    if w < lg + 2:
        raise ValueError("ts_conditional_transfer_entropy requires window >= lag + 2")
    mt_raw = b.get("min_transitions")
    if mt_raw is None:
        mt = max(30, 2 * nb * nb * nb)
    else:
        mt = max(lg + 2, _strict_int(mt_raw, "min_transitions", lower=1))
    required = 3 * (nb ** 4)
    mt = max(mt, required, lg + 2)
    mt = max(mt, int(np.ceil(ratio * nb * nb * nb * nb)))
    if w - lg < mt:
        raise ValueError(
            f"ts_conditional_transfer_entropy window-lag ({w - lg}) < required "
            f"transitions ({mt}) with bins={nb}"
        )
    out = _tri_rolling(
        _panel(b["target"]), _panel(b["source"]), _panel(b["condition"]), w,
        lambda a, bb, cc: _conditional_te_window(a, bb, cc, nb, lg, mt, ratio),
    )
    return _rebuild(b["target"], out)


def _k_ts_jump_bipower_proxy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.alpha_language_volatility import (
        _trailing_contiguous, _EPS,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    w = _strict_int(b.get("window", 20), "window", lower=2)
    mp = max(3, _strict_int(b.get("min_periods", 3), "min_periods", lower=1))
    rv = _panel(b["ret"])

    def _fn(chunk: np.ndarray) -> float:
        v = _trailing_contiguous(chunk)
        n = v.size
        if n < mp:
            return np.nan
        rv2 = float(np.sum(v * v))
        if rv2 < _EPS:
            return np.nan
        bv = float(np.sum(np.abs(v[1:]) * np.abs(v[:-1])))
        bv = (np.pi / 2.0) * bv
        return float(max(rv2 - bv, 0.0) / (rv2 + _EPS))

    return _rebuild(b["ret"], map_rolling(rv, w, _fn))


def _k_ts_upper_tail_coexceedance_probability(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.nonlinear_dependence import (
        aligned_pairs, _tail_dependence,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_pair_rolling
    w = _strict_int(b.get("window", 60), "window", lower=2)
    quantile = float(b.get("q", 0.9))
    if not 0.5 < quantile < 1.0:
        raise ValueError("q must be in (0.5, 1) for the upper tail")
    min_tail = _strict_int(b.get("min_tail_count", 5), "min_tail_count", lower=2)
    xv = _panel(b["x"])
    yv = _panel(b["y"])

    def _fn(a: np.ndarray, bb: np.ndarray) -> float:
        pa, pb = aligned_pairs(a, bb)
        return _tail_dependence(pa, pb, quantile, True, min_tail)

    return _rebuild(b["x"], map_pair_rolling(xv, yv, w, _fn))


def _kalman_level_out(b: dict, out_stat: str) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.state_space import _kalman_level
    x = b["x"]
    q = float(b.get("q", 1e-4))
    r = float(b.get("r", 1.0))
    scale_mode = b.get("scale_mode", "absolute")
    xv = _panel(x)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = _kalman_level(xv[:, col], q, r, out_stat, scale_mode=scale_mode)
    return _rebuild(x, out)


def _k_ts_kalman_innovation_z(b: dict) -> pl.DataFrame:
    return _kalman_level_out(b, "innovation_z")


def _k_ts_kalman_beta_change(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.ts_model.state_space import (
        _kalman_beta, _BETA_WARMUP,
    )
    from factor_engine.cleaned_operators.ts_model._rolling_core import aligned
    y, x = b["y"], b["x"]
    q = float(b.get("q", 1e-3))
    r = float(b.get("r", 1.0))
    scale_mode = b.get("scale_mode", "absolute")
    min_warmup = _strict_int(b.get("min_warmup", _BETA_WARMUP), "min_warmup", lower=1)
    # aligned() works on pandas panels; on polars panels a strict shape/name
    # check is the equivalent alignment guard (the engine never delivers
    # misaligned panels to a backend slot).
    if _panel(y).shape != _panel(x).shape or _ncols(y) != _ncols(x):
        raise ValueError("ts_kalman_beta_change requires aligned y/x panels")
    yv = _panel(y)
    xv = _panel(x)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        out[:, col] = _kalman_beta(yv[:, col], xv[:, col], q, r, "beta_change",
                                   scale_mode=scale_mode, min_warmup=min_warmup)
    return _rebuild(y, out)


def _k_intraday_volatility_time_centroid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday_vol_ext import (
        _rolling_returns, _time_centroid,
    )
    w = _strict_int(b.get("window", 240), "window", lower=2)
    out = _rolling_returns(_panel(b["returns"]), w, _time_centroid)
    return _rebuild(b["returns"], out)


def _k_event_interval_memory(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.event_interval import (
        _check_event_params, _max_pre_age, _event_mask, _interval_memory_series,
    )
    w, _ = _check_event_params(b.get("window", 240))
    pre = _max_pre_age(b.get("max_pre_window_age"), w)
    values = _panel(b["event"])
    _event_mask(values.reshape(-1))
    out = _interval_memory_series(values, w, pre)
    return _rebuild(b["event"], out)


def _k_event_fano_factor(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.event_interval import (
        _check_event_params, _event_mask, _fano_factor_series,
    )
    w, blk = _check_event_params(b.get("window", 240), b.get("block", 20))
    values = _panel(b["event"])
    _event_mask(values.reshape(-1))
    out = _fano_factor_series(values, w, blk)
    return _rebuild(b["event"], out)


def _k_ts_cross_extremogram(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.advanced_quantile_dynamics import (
        _check_q, _check_side, _cross_extremogram_chunk,
    )
    from factor_engine.cleaned_operators.rolling_pack import map_pair_rolling
    w = int(b.get("window", 120))
    tq = _check_q(b.get("target_q", 0.1))
    sq = _check_q(b.get("source_q", 0.1))
    lg = int(b.get("lag", 1))
    ts_ = b.get("target_side", "lower")
    ss_ = b.get("source_side", "lower")
    ft = b.get("fixed_threshold", False)
    _check_side(ts_)
    _check_side(ss_)
    if lg < 1:
        raise ValueError("ts_cross_extremogram requires lag >= 1")
    if w < lg + 2:
        raise ValueError("ts_cross_extremogram requires window >= lag + 2")
    out = map_pair_rolling(
        _panel(b["target"]), _panel(b["source"]), w,
        lambda a, bb: _cross_extremogram_chunk(a, bb, tq, sq, lg, ts_, ss_, ft),
    )
    return _rebuild(b["target"], out)


def _k_cs_hartigan_dip(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cs_state_ops import _dip_statistic
    mc = max(2, _strict_int(b.get("min_cross", 100), "min_cross", lower=1))
    arr = _panel(b["x"])
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        row = arr[r]
        finite = row[np.isfinite(row)]
        if finite.size < mc:
            continue
        dip = _dip_statistic(np.sort(finite))
        out[r, :] = float(np.sqrt(finite.size) * dip)
    return _rebuild(b["x"], out)


def _k_ts_generalized_hurst_spread_q1_q4(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.multifractal import (
        _check_window, _hurst_spread_series,
    )
    w = _check_window(b.get("window", 120))
    out = _hurst_spread_series(_panel(b["x"]), w)
    return _rebuild(b["x"], out)


def _k_cs_knn_graph_dirichlet_energy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.dynamic_knn import _dirichlet_energy_series
    kk = _strict_int(b.get("k", 5), "k", lower=1)
    fa = b.get("feature_available_at", "same_day")
    ta = b.get("target_available_at", "close_of_t")
    if fa != "same_day" or ta != "close_of_t":
        raise ValueError(
            "cs_knn_graph_dirichlet_energy requires feature_available_at="
            "'same_day' and target_available_at='close_of_t'"
        )
    feats = np.stack([_panel(b[n]) for n in ("f1", "f2", "f3")], axis=2)
    out = _dirichlet_energy_series(_panel(b["target"]), feats, kk)
    return _rebuild(b["target"], out)


def _k_cs_knn_local_moran(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section_ext import _local_moran_series
    kk = _strict_int(b.get("k", 5), "k", lower=1)
    feats = np.stack([_panel(b[n]) for n in ("f1", "f2", "f3")], axis=2)
    out = _local_moran_series(_panel(b["target"]), feats, kk)
    return _rebuild(b["target"], out)


def _k_ts_tail_ratio(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.rolling_pack import map_rolling
    from factor_engine.cleaned_operators.robust_tail import _has_spread, valid_values
    w = _strict_int(b.get("window", 60), "window", lower=2)
    ql, qh = float(b.get("q_low", 0.05)), float(b.get("q_high", 0.95))
    if not 0.0 < ql < qh < 1.0:
        raise ValueError("require 0 < q_low < q_high < 1")
    mp = max(5, _strict_int(b.get("min_periods", 5), "min_periods", lower=1))

    def _fn(chunk: np.ndarray) -> float:
        valid = valid_values(chunk)
        if not _has_spread(valid, min_count=mp):
            return np.nan
        q_lo = float(np.quantile(valid, ql))
        q_hi = float(np.quantile(valid, qh))
        if abs(q_lo) < 1e-12:
            return np.nan
        return float(abs(q_hi) / abs(q_lo))

    return _rebuild(b["x"], map_rolling(_panel(b["x"]), w, _fn))


def _k_ts_spectral_centroid(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.spectral import (
        _check_spectral_params, _spectral_centroid_series,
    )
    w = _check_spectral_params(b.get("window", 60))
    out = _spectral_centroid_series(_panel(b["x"]), w)
    return _rebuild(b["x"], out)


def _k_ts_spectral_peak_concentration(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.spectral import (
        _check_spectral_params, _spectral_peak_concentration_series,
    )
    w = _check_spectral_params(b.get("window", 60))
    out = _spectral_peak_concentration_series(_panel(b["x"]), w)
    return _rebuild(b["x"], out)


def _k_ts_cov_if(b: dict) -> pl.DataFrame:
    x, y, condition = b["x"], b["y"], b["condition"]
    w = _strict_int(b.get("window", 20), "window", lower=2)
    mp = _strict_int(b.get("min_periods", 2), "min_periods", lower=2, upper=w)
    xv = _panel(x)
    yv = _panel(y)
    cv = _panel(condition)
    if xv.shape != yv.shape or xv.shape != cv.shape:
        raise ValueError("ts_cov_if panels must be aligned")
    bad = np.isfinite(cv) & (cv != 0.0) & (cv != 1.0)
    if bad.any():
        raise ValueError(
            "condition must be a ConditionBool (values in {0, 1} with NaN as missing)"
        )
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            start = max(0, r - w + 1)
            xs = xv[start: r + 1, col]
            ys = yv[start: r + 1, col]
            cs = cv[start: r + 1, col]
            mask = (np.isfinite(xs) & np.isfinite(ys) & np.isfinite(cs) & (cs == 1.0))
            xa = xs[mask].astype(float)
            ya = ys[mask].astype(float)
            if xa.size < mp:
                continue
            a = xa.astype(np.longdouble)
            bbb = ya.astype(np.longdouble)
            cov = np.sum((a - a.mean()) * (bbb - bbb.mean())) / (xa.size - 1)
            if not np.isfinite(cov) or abs(cov) > np.finfo(float).max:
                continue
            result = float(cov)
            if cov != 0 and result == 0:
                continue
            out[r, col] = result
    return _rebuild(x, out)


def _k_ts_opening_mispricing_score(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday.overnight import _EPS
    close, open_px, pre_close = b["close"], b["open"], b["pre_close"]
    cv = _panel(close)
    ov0 = _panel(open_px)
    pv = _panel(pre_close)
    if not (cv.shape == ov0.shape == pv.shape):
        raise ValueError("ts_opening_mispricing_score panels must be aligned")
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = np.where(pv != 0.0, pv, np.nan)
        oc = np.where(ov0 != 0.0, ov0, np.nan)
        o = ov0 / pc - 1.0
        i = cv / oc - 1.0
    rows, cols = o.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(b.get("window", 60))
    mp = max(6, w // 5)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            a, bb = o[start: row + 1, col], i[start: row + 1, col]
            valid = np.isfinite(a) & np.isfinite(bb)
            if valid.sum() < mp or np.var(a[valid]) <= _EPS:
                continue
            beta = float(np.cov(a[valid], bb[valid])[0, 1] / np.var(a[valid]))
            expected = beta * o[row, col]
            out[row, col] = float(o[row, col] - expected)
    return _rebuild(close, out)


def _k_ts_l_skewness(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.moments_ext import (
        _check_window, _l_ratio_series,
    )
    w = _check_window(b.get("window", 60), minimum=4)
    out = _l_ratio_series(
        _panel(b["x"]), w, "skew",
        min_periods=b.get("min_periods", 20),
        min_coverage_fraction=b.get("min_coverage_fraction", 0.5),
    )
    return _rebuild(b["x"], out)


def _k_ts_hvg_degree_entropy(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.hvg_ext import _hvg_check_params, _hvg_series
    w, mp, mn, mcf = _hvg_check_params(
        b.get("window", 60), b.get("min_periods", 8), b.get("min_nodes", 10),
        b.get("min_coverage_fraction", 0.5), "ts_hvg_degree_entropy",
    )
    out = _hvg_series(_panel(b["x"]), w, mp, mn, mcf, "degree_entropy")
    return _rebuild(b["x"], out)


def _k_ts_pickands_tail_index(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.evt_allan import _pickands_tail
    from factor_engine.cleaned_operators.gemini_v2_common import trailing_contiguous_finite
    w_raw = b.get("window", 120)
    if int(w_raw) < 6:
        raise ValueError("ts_pickands_tail_index requires window >= 6")
    side_s = str(b.get("side", "upper")).lower()
    if side_s not in {"upper", "lower"}:
        raise ValueError("ts_pickands_tail_index requires side in {'upper','lower'}")
    k = int(b.get("k", 10))
    w = int(w_raw)
    arr = _panel(b["x"])
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo: r + 1])
            if v.size < 4 * int(k) + 1:
                continue
            val = _pickands_tail(v, int(k), side_s == "upper")
            if np.isfinite(val):
                out[r, c] = val
    return _rebuild(b["x"], out)


def _k_ts_delay_intrinsic_dimension(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intrinsic_dimension import _intrinsic_dim_series
    w = _strict_int(b.get("window", 120), "window", lower=2)
    dim = _strict_int(b.get("embedding_dim", 3), "embedding_dim", lower=2)
    kk = _strict_int(b.get("k", 5), "k", lower=2)
    dl = _strict_int(b.get("delay", 1), "delay", lower=1)
    if w - (dim - 1) * dl < kk + 1:
        raise ValueError(
            "ts_delay_intrinsic_dimension requires window-(embedding_dim-1)*delay >= k+1"
        )
    tw_raw = b.get("theiler_window")
    tw = _strict_int(tw_raw, "theiler_window", lower=0) if tw_raw is not None else dim * dl
    n_embed = w - (dim - 1) * dl
    if n_embed < kk + tw + 1:
        raise ValueError(
            "ts_delay_intrinsic_dimension requires N_embed >= k + effective_theiler + 1"
        )
    out = _intrinsic_dim_series(_panel(b["x"]), w, dim, kk, dl, tw)
    return _rebuild(b["x"], out)


def _k_cs_knn_distance(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.robust_cs import (
        _knn_blockwise, _EPS,
    )
    kk = max(2, int(b.get("k", 5)))
    fv = [_panel(b[n]) for n in ("f1", "f2", "f3", "f4") if b.get(n) is not None]
    n, n_cols = fv[0].shape
    out = np.full((n, n_cols), np.nan, dtype=float)
    for row in range(n):
        X = np.column_stack([f[row] for f in fv])
        valid = np.all(np.isfinite(X), axis=1)
        if valid.sum() < kk + 1:
            continue
        Xv = X[valid]
        sd = np.std(Xv, axis=0)
        Xn = Xv / np.where(sd > _EPS, sd, 1.0)
        out[row, valid] = _knn_blockwise(Xn, kk)
    return _rebuild(b["f1"], out)


def _k_ts_recovery_fraction(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.stateful.drawdown_path import (
        _EPS, _trailing_contiguous,
    )
    xv = _panel(b["x"])
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = _strict_int(b.get("window", 60), "window", lower=2)
    for col in range(cols):
        for row in range(rows):
            lo = max(0, row - w + 1)
            seg = _trailing_contiguous(xv[lo: row + 1, col])
            if seg.size < 2 or not np.all(seg > 0.0):
                continue
            vals = seg
            p = float(np.max(vals))
            p_pos = int(vals[::-1].argmax())
            p_pos = len(vals) - 1 - p_pos
            t = float(np.min(vals[p_pos:]))
            if p - t <= _EPS:
                out[row, col] = 1.0
                continue
            rf = (float(vals[-1]) - t) / (p - t + _EPS)
            out[row, col] = float(np.clip(rf, 0.0, 1.0))
    return _rebuild(b["x"], out)


def _k_fin_working_capital_accruals(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.fundamental.quality_v2 import _reject_ytd_growth
    from factor_engine.cleaned_operators.fundamental.transforms_v2 import (
        _period_key, _period_insert, _lag_value, _default_require_parseable,
    )
    _reject_ytd_growth("fin_working_capital_accruals", b.get("flow_type"))
    ca = _panel(b["current_assets"])
    cash = _panel(b["cash"])
    cl = _panel(b["current_liabilities"])
    std = _panel(b["short_term_debt"])
    tp = _panel(b["tax_payable"])
    avg = _panel(b["avg_assets"])
    pid = _panel(b["period_id"])
    shape = ca.shape
    if not all(a.shape == shape for a in (cash, cl, std, tp, avg, pid)):
        raise ValueError("fin_working_capital_accruals panels must be aligned")
    require_parseable = _default_require_parseable()

    def walk(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        rows, cols = x.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            order: list = []
            visible: dict = {}
            xv = x[:, c]
            pv = p[:, c]
            for i in range(rows):
                value = xv[i]
                key = _period_key(pv[i])
                if key is not None and np.isfinite(value):
                    if key not in visible:
                        _period_insert(order, key, require_parseable=require_parseable)
                    visible[key] = float(value)
                if key is None or key not in visible:
                    continue
                try:
                    out[i, c] = float(visible[key]) - _lag_value(order, visible, key, 1)
                except (ValueError, ZeroDivisionError, FloatingPointError,
                        np.linalg.LinAlgError):
                    out[i, c] = np.nan
        return out

    d_ca = walk(ca, pid)
    d_cash = walk(cash, pid)
    d_cl = walk(cl, pid)
    d_std = walk(std, pid)
    d_tp = walk(tp, pid)
    wc = (d_ca - d_cash) - (d_cl - d_std - d_tp)
    denom = np.where(avg != 0.0, avg, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = wc / denom
    return _rebuild(b["current_assets"], out)


def _k_atr_pct(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.technical.indicators_v2 import _pi
    high, low, close = b["high"], b["low"], b["close"]
    w = _pi(b.get("window", 20), "window", 2)
    cols = _ncols(high)
    comb = _combine([high, low, close])
    exprs = []
    for c in cols:
        H, L, C = _pcol(0, c), _pcol(1, c), _pcol(2, c)
        prev = C.shift(1)
        a = H - L
        bcomp = (H - prev).abs()
        ccomp = (L - prev).abs()
        tr = pl.max_horizontal(a, bcomp, ccomp)
        tr = pl.when(a.is_null() | bcomp.is_null() | ccomp.is_null()).then(None).otherwise(tr)
        atr = tr.ewm_mean(alpha=1.0 / w, adjust=False, **{_EWM_KW: w})
        # pandas (ignore_na=False) carries the previous filtered state through
        # NaN rows in its OUTPUT; polars leaves them null — forward-fill parity.
        atr = atr.fill_null(strategy="forward")
        pos = pl.when(C > 0).then(C).otherwise(None)
        exprs.append((atr / pos).fill_null(float("nan")).cast(pl.Float64).alias(c))
    return _finish(high, comb.select(exprs))


def _k_panel_rolling_pca_resid_momentum(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_resid
    ret = b["ret"]
    w = int(b.get("window", 120))
    n_components = int(b.get("n_components", 5))
    rv = _panel(ret)
    rows, cols = rv.shape
    resid = np.full((rows, cols), np.nan, dtype=float)
    lag = 1
    for row in range(rows):
        fit_end = row - lag
        if fit_end < 0:
            continue
        if fit_end < w - 1:
            continue
        start = max(0, fit_end - w + 1)
        X = rv[start: fit_end + 1]
        resid[row] = _pca_resid(X, rv[row], n_components)
    base = _rebuild(ret, resid)
    exprs = []
    for c in _ncols(base):
        exprs.append(_roll(_col(base, c), w, "sum", mp=10)
                     .fill_null(float("nan")).cast(pl.Float64).alias(c))
    return base.with_columns(exprs)


def _k_intra_bar_range_persistence(b: dict) -> pl.DataFrame:
    from factor_engine.cleaned_operators.intraday._core import minute_of_day
    high, low = b["high"], b["low"]
    w = max(2, int(b.get("window", 20)))
    session_tz = b.get("session_tz")
    times = _time_numpy(high, session_tz)
    if times is None:
        raise ValueError("intra_bar_range_persistence requires a time axis column")
    h = _panel(high)
    l = _panel(low)
    with np.errstate(divide="ignore", invalid="ignore"):
        ranges = np.where(
            (h > 0) & (l > 0) & np.isfinite(h) & np.isfinite(l),
            np.log(h / l), np.nan,
        )
    days = times.astype("datetime64[D]")
    slots = minute_of_day(times)
    uniq_days = np.unique(days)
    day_pos = {d: i for i, d in enumerate(uniq_days)}
    uniq_slots = np.unique(slots)
    slot_pos = {s: i for i, s in enumerate(uniq_slots)}
    cols = _ncols(high)
    n_days, n_slots = len(uniq_days), len(uniq_slots)
    out = np.full((n_days, len(cols)), np.nan, dtype=float)
    mp_hist = max(2, w // 2)
    eps = 1e-12
    for j, c in enumerate(cols):
        mat = np.full((n_days, n_slots), np.nan, dtype=float)
        cnt = np.zeros((n_days, n_slots), dtype=float)
        vals = ranges[:, j]
        for i in range(len(times)):
            if not np.isfinite(vals[i]):
                continue
            di = day_pos[days[i]]
            si = slot_pos[slots[i]]
            s = mat[di, si]
            mat[di, si] = vals[i] if not np.isfinite(s) else (s + vals[i])
            cnt[di, si] += 1.0
        with np.errstate(invalid="ignore"):
            mat = np.where(cnt > 0, mat / np.where(cnt > 0, cnt, 1.0), np.nan)
        # trailing mean of the PAST w days per slot (shift(1).rolling(w))
        hist = np.full((n_days, n_slots), np.nan, dtype=float)
        for d in range(n_days):
            lo = max(0, d - w)
            block = mat[lo:d]  # past days only
            fin = np.isfinite(block)
            cnt = fin.sum(axis=0)
            ssum = np.where(fin, block, 0.0).sum(axis=0)
            hist[d] = np.where(cnt >= mp_hist, ssum / np.where(cnt > 0, cnt, 1.0), np.nan)
        for d in range(n_days):
            a = mat[d]
            bb = hist[d]
            valid = np.isfinite(a) & np.isfinite(bb)
            if valid.sum() < 2:
                continue
            va, vb = a[valid], bb[valid]
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            if na <= eps or nb <= eps:
                continue
            out[d, j] = float(np.dot(va, vb) / (na * nb))
    # daily output; shape-preserving only when the panel itself is daily
    base = b["high"]
    if n_days == base.height:
        return base.with_columns([
            pl.Series(c, np.ascontiguousarray(out[:, j]), dtype=pl.Float64)
            for j, c in enumerate(cols)
        ])
    from factor_engine.backend.operator_errors import OperatorShapeError
    raise OperatorShapeError(
        "intra_bar_range_persistence produced a daily panel; the polars runtime "
        "requires a daily source panel for this operator"
    )



def _k_ts_quantile_beta_spread(b: dict) -> pl.DataFrame:
    """beta(q_high) - beta(q_low) (pinball LP, in-sample; b23r re-assert native)."""
    from factor_engine.cleaned_operators.ts_model._rolling_core import pinball_quantile_fit
    from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
        validate_multi_configured_history,
    )
    y = _panel(b["y"])
    x = _panel(b["x"])
    qh, ql = float(b.get("q_high", 0.9)), float(b.get("q_low", 0.1))
    if not (0.0 < ql < qh < 1.0):
        raise ValueError("q_low < q_high must hold in (0, 1)")
    w = int(b.get("window", 60))
    mp_req = int(b.get("min_periods", 10))
    mp = max(mp_req, 10)  # max(min_periods, 5*n_coeffs), n_coeffs=2
    validate_multi_configured_history(w, mp_req, 1, True, fit_lag=0)
    rows, cols = y.shape

    def _beta(q: float) -> np.ndarray:
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            ycol, xcol = y[:, col], x[:, col]
            for row in range(rows):
                start = max(0, row - w + 1)
                seg_y = ycol[start : row + 1]
                seg_x = xcol[start : row + 1]
                valid = np.isfinite(seg_y) & np.isfinite(seg_x)
                if valid.sum() < mp:
                    continue
                vy, vx = seg_y[valid], seg_x[valid]
                if np.std(vx) <= 0.0:
                    continue
                design = np.column_stack([np.ones(vy.size), vx])
                coef = pinball_quantile_fit(design, vy, q)
                if coef is None:
                    continue
                out[row, col] = float(coef[1])
        return out

    return _rebuild(b["y"], _beta(qh) - _beta(ql))


_KERNELS: dict[str, Callable[[dict], pl.DataFrame]] = {
    "ashare_failed_limit_count": _k_ashare_failed_limit_count,
    "vwap_distance_pct": _k_vwap_distance_pct,
    "ts_hankel_singular_gap": _k_ts_hankel_singular_gap,
    "relation_hhi": _k_relation_hhi,
    "fin_working_capital_accruals": _k_fin_working_capital_accruals,
    "ts_vector_state_local_density": _k_ts_vector_state_local_density,
    "ts_recurrence_diagonal_entropy": _k_ts_recurrence_diagonal_entropy,
    "ts_local_lyapunov_exponent": _k_ts_local_lyapunov_exponent,
    "ts_state_age_percentile": _k_ts_state_age_percentile,
    "ts_conditional_transfer_entropy": _k_ts_conditional_transfer_entropy,
    "ts_state_residual_life": _k_ts_state_residual_life,
    "ts_jump_bipower_proxy": _k_ts_jump_bipower_proxy,
    "event_frequency": _k_event_frequency,
    "ts_upper_tail_coexceedance_probability": _k_ts_upper_tail_coexceedance_probability,
    "ts_kalman_innovation_z": _k_ts_kalman_innovation_z,
    "intraday_volatility_time_centroid": _k_intraday_volatility_time_centroid,
    "ts_vector_state_mahalanobis": _k_ts_vector_state_mahalanobis,
    "ts_gap_reversion_ratio": _k_ts_gap_reversion_ratio,
    "event_interval_memory": _k_event_interval_memory,
    "donchian_channel_position": _k_donchian_channel_position,
    "atr_pct": _k_atr_pct,
    "ts_cross_extremogram": _k_ts_cross_extremogram,
    "event_fano_factor": _k_event_fano_factor,
    "cs_hartigan_dip": _k_cs_hartigan_dip,
    "ts_generalized_hurst_spread_q1_q4": _k_ts_generalized_hurst_spread_q1_q4,
    "cs_knn_graph_dirichlet_energy": _k_cs_knn_graph_dirichlet_energy,
    "ts_matrix_profile_neighbor_dispersion": _k_ts_matrix_profile_neighbor_dispersion,
    "panel_rolling_pca_resid_momentum": _k_panel_rolling_pca_resid_momentum,
    "intra_bar_range_persistence": _k_intra_bar_range_persistence,
    "consolidation_range_pct": _k_consolidation_range_pct,
    "ts_matrix_profile_motif_frequency": _k_ts_matrix_profile_motif_frequency,
    "cs_knn_local_moran": _k_cs_knn_local_moran,
    "ts_tail_ratio": _k_ts_tail_ratio,
    "ts_spectral_centroid": _k_ts_spectral_centroid,
    "ts_cov_if": _k_ts_cov_if,
    "ts_opening_mispricing_score": _k_ts_opening_mispricing_score,
    "ts_l_skewness": _k_ts_l_skewness,
    "ts_hvg_degree_entropy": _k_ts_hvg_degree_entropy,
    "ts_pickands_tail_index": _k_ts_pickands_tail_index,
    "ts_delay_intrinsic_dimension": _k_ts_delay_intrinsic_dimension,
    "ts_spectral_peak_concentration": _k_ts_spectral_peak_concentration,
    "cs_knn_distance": _k_cs_knn_distance,
    "ts_upper_partial_moment": _k_ts_upper_partial_moment,
    "ts_recovery_fraction": _k_ts_recovery_fraction,
    "ts_kalman_beta_change": _k_ts_kalman_beta_change,
    "ts_quantile_beta_spread": _k_ts_quantile_beta_spread,
}


# ---------------------------------------------------------------------------
# registration (R65 protocol)
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


def register_r68_native_batch3() -> list[str]:
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
            raise RuntimeError(f"R68 native registration missing pandas authority: {canonical}")
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
                continue  # first registrant wins: another native kernel owns the slot
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
                (canonical + ":pandas-authority-parity:r68b3").encode()
            ).hexdigest(),
            notes=(
                "R68 batch3 genuine Polars backend: pure pl.Expr kernels or "
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


__all__ = ["register_r68_native_batch3"]

# Register on import: the ordered loader imports this module inside the
# building window (before gap coverage), so the native slots win over the
# pandas-delegate UDF bridge.
register_r68_native_batch3()
