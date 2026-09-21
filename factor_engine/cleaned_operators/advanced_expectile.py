# -*- coding: utf-8 -*-
"""Expectile / asymmetric least-squares operators (2026-08 market language, P1).

Quantiles only track "how much probability lies below a threshold"; expectiles
additionally weight the *magnitude* of the tail observations, which makes them
the natural asymmetric-loss location used in VaR/ES and tail-risk work.

* ``ts_expectile``  — windowed expectile ``e_tau = argmin_e sum |tau - I(x<e)| (x-e)^2``.
* ``ts_expectile_beta`` — IRLS asymmetric least-squares regression
  ``y = a + b·x + eps`` minimising the same loss; the slope ``b`` answers
  "how strong is X's effect when Y is in its low/high state".

Both are deterministic (fixed-point / IRLS convergence), prefix-causal, and
fail closed to NaN on degenerate windows.
"""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import (
    aligned_pairs,
    frame_like,
    map_pair_rolling,
    map_rolling,
    register_polars_udf,
)

_EPS = 1e-12


def _trailing_window_matrix(values: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing-window gather for one column, NaN-aware.

    Returns ``(W, valid, seg_n)``: row ``r`` of ``W`` holds rows
    ``max(0, r-w+1) .. r`` of ``values`` in a ``w``-wide row (short leading rows
    are NaN padded at the tail); ``valid`` marks the positions that are both
    in-window and finite — exactly the authority's ``seg[np.isfinite(seg)]`` —
    and ``seg_n = min(r+1, w)`` is that ``seg.size``.  Only the *set* of valid
    positions is used downstream, so the padding side is immaterial.
    """
    rows = values.shape[0]
    pos = np.arange(w)
    start = np.maximum(0, np.arange(rows) - w + 1)
    gather = np.clip(start[:, None] + pos[None, :], 0, rows - 1)
    inwin = np.ascontiguousarray(pos[None, :] < (np.arange(rows) + 1)[:, None])
    W = np.where(inwin, values[gather], np.nan)
    return W, inwin & np.isfinite(W), inwin.sum(axis=1)


def _trailing_run_matrix(values: np.ndarray, w: int):
    """Trailing *contiguous finite* run ending at every row, NaN padded.

    Row ``r`` holds the maximal trailing finite run of ``values[max(0,r-w+1)..r]``
    starting at column 0 (never re-connected across a gap; a NaN at ``r`` yields
    the empty run), so the padded tail of the row is exactly the authority's
    ``_trailing_contiguous`` / ``_trailing_contiguous_finite`` block.
    """
    rows = values.shape[0]
    idx = np.arange(rows)
    last_bad = np.where(np.isfinite(values), -1, idx)
    np.maximum.accumulate(last_bad, out=last_bad)
    start = np.maximum(np.maximum(0, idx - w + 1), last_bad + 1)
    length = np.where(np.isfinite(values), idx - start + 1, 0)
    pos = np.arange(w)
    gather = np.clip(start[:, None] + pos[None, :], 0, rows - 1)
    V = np.where(pos[None, :] < length[:, None], values[gather], np.nan)
    return V, length


def _padded_median(A: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """``np.median`` of the leading ``cnt`` entries of each row of ``A``.

    ``A`` carries NaN in the unused tail (sorted last), so the two middle order
    statistics of the valid prefix are picked directly; the average of the two
    middles reproduces ``np.median`` bit for bit (including ``cnt < 1`` -> NaN).
    """
    S = np.sort(A, axis=1)
    ridx = np.arange(A.shape[0])
    lo = (cnt - 1) // 2
    hi = cnt // 2
    return (S[ridx, lo] + S[ridx, hi]) / 2.0


# P1-16: the fixed-point / IRLS iterate is emitted ONLY when the maximum
# absolute update between iterations satisfies a RELATIVE tolerance; a window
# that does not converge within ``_MAX_ITER_*`` iterations emits NaN — a stale
# last-iterate must never masquerade as a valid factor.
_CONV_TOL = 1e-6
_MAX_ITER_EXPECTILE = 200
_MAX_ITER_IRLS = 40

# P1-17: default minimum effective-tail sample size.  An extreme expectile
# (tau=0.01/0.99) estimated from ``N_eff * min(tau, 1-tau) < n_min``
# observations is wildly unstable and emits NaN.  Kept at 3 (not 5) so the
# well-posed tau=0.5 / window=6 case (expectile == mean, N_eff*0.5 = 3) still
# emits a finite value.
_DEFAULT_N_MIN = 3


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
    param_specs: dict | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="expectile",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "expectile", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
        panel_params=("y","x") if name == "ts_expectile_beta" else ("x",),
        scalar_params=("window","tau","n_min"),
        param_specs={
            "window": ParamSpec(dtype=int, min=4 if name == "ts_expectile_beta" else 3,
                              default=60, history_semantics="max_rows", param_role=ParamRole.HORIZON),
            "tau": ParamSpec(dtype=float, min=float(np.nextafter(0.,1.)),
                            max=float(np.nextafter(1.,0.)), default=.1,
                            param_role=ParamRole.NUMERICAL, searchable=False),
            "n_min": ParamSpec(dtype=int,min=1,default=_DEFAULT_N_MIN,
                              param_role=ParamRole.SUPPORT_POLICY,searchable=False),
        },
    )


def _parameters(window, tau, n_min, minimum):
    from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_float
    return (
        strict_int(window,"window",minimum=minimum),
        strict_float(tau,"tau",minimum=float(np.nextafter(0.,1.)),maximum=float(np.nextafter(1.,0.))),
        strict_int(n_min,"n_min",minimum=1),
    )


def _expectile(vals: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    """Newey-Powell fixed-point expectile of a finite value array.

    Only a fixed point with normalized update and estimating-score residual
    at most 1e-12 within ``_MAX_ITER_EXPECTILE`` is emitted; otherwise NaN.
    P1-17: requires ``N_eff * min(tau, 1-tau) >= n_min`` effective tail
    observations, else the extreme-tail estimate is unstable -> NaN.
    """
    v = vals[np.isfinite(vals)]
    if v.size < 3:
        return np.nan
    tt = float(tau)
    nmin = int(n_min)
    if v.size * min(tt, 1.0 - tt) < nmin:
        return np.nan
    # A large level must not relax the stopping criterion.  Work in a shared
    # centered, scale-normalized training coordinate system; the half-sum
    # midpoint avoids overflow when the observed range spans both signs.
    center = float(v.min() / 2.0 + v.max() / 2.0)
    shifted = v - center
    scale = float(np.max(np.abs(shifted)))
    if scale == 0.0:
        return center
    if not np.isfinite(scale):
        return np.nan
    normalized = shifted / scale
    e = float(np.mean(normalized))
    converged = False
    for _ in range(_MAX_ITER_EXPECTILE):
        w = np.abs(tt - (normalized < e).astype(float))
        s = float(w.sum())
        if s <= _EPS:
            converged = True
            break
        e_new = float(np.sum(w * normalized)) / s
        residual = normalized - e_new
        score = float(np.dot(np.where(residual >= 0.0, tt, 1.0 - tt), residual)) / s
        if abs(e_new - e) <= 1e-12 and abs(score) <= 1e-12:
            e = e_new
            converged = True
            break
        e = e_new
    result = center + scale * e
    return result if converged and np.isfinite(result) else np.nan


def _expectile_chunk(chunk: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    return _expectile(chunk, tau, n_min)


@register_operator(
    name="ts_expectile",
    category="expectile",
    business_category="expectile",
    canonical="ts_expectile",
    source="advanced_expectile",
)
class TsExpectile(SeriesOperator):
    """窗口 expectile（幅度敏感的分位——极端值有多大会影响结果）。

    ``e_tau = argmin_e sum |tau - I(x<e)| (x-e)^2``，tau=0.5 即均值。与 quantile
    不同：越过 tau 边界的观测的幅度进入目标函数，对 return/volume/turnover/
    fundamental-change 的厚尾更敏感。PIT 安全（trailing 窗口）。P1。
    """

    metadata = _metadata(
        "ts_expectile",
        "窗口 expectile e_tau（不对称最小二乘，幅度敏感）。",
        ["x", "window", "tau", "n_min"],
        domain="price_volume",
        # P0-15: the output carries the unit of the single panel input ``x`` —
        # there is NO ``target`` parameter, so ``same_as:target`` was
        # unresolvable.
        unit="same_as:x",
        cost=4,
        param_specs={
            "n_min": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: float = 0.1,
        n_min: int = _DEFAULT_N_MIN,
        **_: Any,
    ) -> pd.DataFrame:
        w, tt, nmin = _parameters(window, tau, n_min, 3)
        xv = x.to_numpy(dtype=float)
        out = np.empty_like(xv)
        for c in range(xv.shape[1]):
            out[:, c] = _expectile_batch(xv[:, c], w, tt, nmin)
        return frame_like(x, out)


def _expectile_batch(
    values: np.ndarray, window: int, tau: float, n_min: int
) -> np.ndarray:
    """Row-parallel ``_expectile``: every trailing window iterates together.

    The Newey-Powell fixed point is run on the whole panel at once (each row
    frozen as soon as it converges, exactly like the authority's ``break``), so
    the per-window Python loop disappears while the convergence contract —
    normalized update AND estimating score within 1e-12, else NaN — is kept.
    """
    rows = values.shape[0]
    w = max(3, int(window))
    tt = float(tau)
    nmin = int(n_min)
    W, mask, _ = _trailing_window_matrix(values, w)
    cnt = mask.sum(axis=1)
    out = np.full(rows, np.nan, dtype=float)
    tail = min(tt, 1.0 - tt)
    gate = (cnt >= 3) & (cnt * tail >= nmin)
    if not gate.any():
        return out
    with np.errstate(invalid="ignore"):
        vmin = np.fmin.reduce(W, axis=1, initial=np.inf)
        vmax = np.fmax.reduce(W, axis=1, initial=-np.inf)
    center = vmin / 2.0 + vmax / 2.0
    shifted = W - center[:, None]
    with np.errstate(invalid="ignore"):
        scale = np.fmax.reduce(np.abs(shifted), axis=1, initial=-np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = shifted / scale[:, None]
    A = np.where(mask, norm, 0.0)
    mf = mask.astype(float)
    with np.errstate(invalid="ignore"):
        e = np.nansum(norm, axis=1) / np.maximum(cnt, 1)
    live = gate & np.isfinite(scale) & (scale != 0.0)
    conv = np.zeros(rows, dtype=bool)
    if live.any():
        ridx = np.arange(rows)
        for _ in range(_MAX_ITER_EXPECTILE):
            act = live & ~conv
            if not act.any():
                break
            wgt = np.abs(tt - np.where(A < e[:, None], 1.0, 0.0)) * mf
            s = wgt.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                e_new = (wgt * A).sum(axis=1) / s
            resid = (A - e_new[:, None]) * mf
            score = np.sum(
                np.where(resid >= 0.0, tt, 1.0 - tt) * resid, axis=1
            ) / s
            stopped = np.abs(e_new - e) <= 1e-12
            stopped &= np.abs(score) <= 1e-12
            empty = s <= _EPS
            e = np.where(act & ~empty, e_new, e)
            conv = conv | (act & (stopped | empty))
        result = center + scale * e
        ok = live & conv & np.isfinite(result)
        out[ok] = result[ok]
    zero = gate & (scale == 0.0)
    out[zero] = center[zero]
    return out


def _expectile_slope(yv: np.ndarray, xv: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    """IRLS asymmetric least-squares slope of y ~ a + b·x at expectile tau.

    Only an IRLS iterate with normalized prediction and score convergence
    within ``_MAX_ITER_IRLS`` is emitted; a near-singular /
    high-leverage design that does not settle returns NaN instead of a bogus
    slope.
    P1-17: requires ``N_eff * min(tau, 1-tau) >= n_min`` effective tail
    observations, else the extreme-tail estimate is unstable -> NaN.
    """
    yy, xx = aligned_pairs(yv, xv)
    n = yy.size
    if n < 4:
        return np.nan
    tt = float(tau)
    nmin = int(n_min)
    if n * min(tt, 1.0 - tt) < nmin:
        return np.nan
    xcenter = float(xx.min() / 2.0 + xx.max() / 2.0)
    ycenter = float(yy.min() / 2.0 + yy.max() / 2.0)
    centered_x, centered_y = xx - xcenter, yy - ycenter
    sx = float(np.max(np.abs(centered_x)))
    sy = float(np.max(np.abs(centered_y)))
    if sx == 0.0 or not np.isfinite(sx) or not np.isfinite(sy):
        return np.nan
    if sy == 0.0:
        return 0.0
    target = centered_y / sy
    A = np.column_stack((centered_x / sx, np.ones(n)))
    try:
        beta, _, rank, _ = np.linalg.lstsq(A, target, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    if rank < 2:
        return np.nan
    converged = False
    for _ in range(_MAX_ITER_IRLS):
        r = target - A @ beta
        w = np.abs(tt - (r < 0.0).astype(float))
        sw = float(w.sum())
        if sw <= _EPS:
            converged = True
            break
        W = np.sqrt(w)
        try:
            beta_new, _, rank, _ = np.linalg.lstsq(A * W[:, None], target * W, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        if rank < 2 or not np.isfinite(beta_new).all():
            return np.nan
        residual = target - A @ beta_new
        next_weights = np.where(residual >= 0.0, tt, 1.0 - tt)
        score = A.T @ (next_weights * residual) / next_weights.sum()
        prediction_change = np.max(np.abs(A @ (beta_new - beta)))
        if prediction_change <= 1e-12 and np.max(np.abs(score)) <= 1e-12:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged:
        return np.nan
    # Restore slope units without an overflowing/underflowing intermediate
    # ratio sy/sx when the final slope itself is representable.
    mantissa, exponent = math.frexp(float(beta[0]))
    my, ey = math.frexp(sy)
    mx, ex = math.frexp(sx)
    try:
        result = math.ldexp(mantissa * my / mx, exponent + ey - ex)
    except OverflowError:
        return np.nan
    return result if np.isfinite(result) else np.nan


def _expectile_beta_chunk(yc: np.ndarray, xc: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    return _expectile_slope(yc, xc, tau, n_min)


@register_operator(
    name="ts_expectile_beta",
    category="expectile",
    business_category="expectile",
    canonical="ts_expectile_beta",
    source="advanced_expectile",
)
class TsExpectileBeta(SeriesOperator):
    """不对称最小二乘回归斜率（y 的低/高状态下 X 的影响强度）。

    窗口内用 IRLS 解 ``min sum |tau - I(y < a+bx)| (y - a - bx)^2``，输出斜率
    ``b_tau``。tau=0.1 与 0.9 会得到完全不同的市场机制（杀跌 vs 追涨），比普通
    OLS beta 更适合高波动/极端换手/财务恶化。PIT 安全。P1。
    """

    metadata = _metadata(
        "ts_expectile_beta",
        "expectile 回归斜率 b_tau（不对称最小二乘）。",
        ["y", "x", "window", "tau", "n_min"],
        domain="price_volume",
        # A regression slope carries unit(y)/unit(x) — NOT a fixed ratio.  With
        # y = price and x = amount the slope is per-amount price, which is not
        # dimensionless (P2-7).  The Typed Unit Algebra derives this at runtime.
        unit="unit(y)/unit(x)",
        cost=5,
        param_specs={
            "n_min": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        window: int = 60,
        tau: float = 0.1,
        n_min: int = _DEFAULT_N_MIN,
        **_: Any,
    ) -> pd.DataFrame:
        w, tt, nmin = _parameters(window, tau, n_min, 4)
        if not y.index.equals(x.index) or not y.columns.equals(x.columns):
            raise ValueError("expectile regression inputs must have identical axes")
        return frame_like(
            y,
            map_pair_rolling(
                y.to_numpy(dtype=float),
                x.to_numpy(dtype=float),
                w,
                lambda a, b: _expectile_beta_chunk(a, b, tt, nmin),
            ),
        )


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_expectile", "ts_expectile_beta"})
    for _canon in ("ts_expectile", "ts_expectile_beta"):
        register_polars_udf(_canon)


_register_surface()
