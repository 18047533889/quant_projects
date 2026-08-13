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

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamSpec, ParamRole, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    frame_like,
    map_pair_rolling,
    map_rolling,
    register_polars_udf,
)

_EPS = 1e-12

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
        param_specs=dict(param_specs) if param_specs else {},
    )


def _expectile(vals: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    """Newey-Powell fixed-point expectile of a finite value array.

    P1-16: only a CONVERGED fixed point (max |update| <= ``_CONV_TOL`` relative
    within ``_MAX_ITER_EXPECTILE``) is emitted; otherwise NaN.
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
    e = float(np.mean(v))
    converged = False
    for _ in range(_MAX_ITER_EXPECTILE):
        w = np.abs(tt - (v < e).astype(float))
        s = float(w.sum())
        if s <= _EPS:
            converged = True
            break
        e_new = float(np.sum(w * v)) / s if s != 0 else np.nan
        if abs(e_new - e) <= _CONV_TOL * max(1.0, abs(e)):
            e = e_new
            converged = True
            break
        e = e_new
    return e if converged else np.nan


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
        w = int(window)
        tt = float(tau)
        nmin = int(n_min)
        if not (0.0 < tt < 1.0):
            raise ValueError("ts_expectile requires 0 < tau < 1")
        if w < 3:
            raise ValueError("ts_expectile requires window >= 3")
        if nmin < 1:
            raise ValueError("ts_expectile requires n_min >= 1")
        return frame_like(
            x,
            map_rolling(x.to_numpy(dtype=float), w, lambda c: _expectile_chunk(c, tt, nmin)),
        )


def _expectile_slope(yv: np.ndarray, xv: np.ndarray, tau: float, n_min: int = _DEFAULT_N_MIN) -> float:
    """IRLS asymmetric least-squares slope of y ~ a + b·x at expectile tau.

    P1-16: only a CONVERGED IRLS iterate (max |beta update| <= ``_CONV_TOL``
    relative within ``_MAX_ITER_IRLS``) is emitted; a near-singular /
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
    sx = float(np.std(xx))
    if sx <= _EPS:
        return np.nan
    A = np.vstack([xx, np.ones(n)]).T
    try:
        beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    converged = False
    for _ in range(_MAX_ITER_IRLS):
        r = yy - A @ beta
        w = np.abs(tt - (r < 0.0).astype(float))
        sw = float(w.sum())
        if sw <= _EPS:
            converged = True
            break
        W = np.sqrt(w)
        try:
            beta_new = np.linalg.lstsq(A * W[:, None], yy * W, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan
        if float(np.max(np.abs(beta_new - beta))) <= _CONV_TOL * max(1.0, float(np.max(np.abs(beta)))):
            beta = beta_new
            converged = True
            break
        beta = beta_new
    return float(beta[0]) if converged else np.nan


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
        w = int(window)
        tt = float(tau)
        nmin = int(n_min)
        if not (0.0 < tt < 1.0):
            raise ValueError("ts_expectile_beta requires 0 < tau < 1")
        if w < 4:
            raise ValueError("ts_expectile_beta requires window >= 4")
        if nmin < 1:
            raise ValueError("ts_expectile_beta requires n_min >= 1")
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
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"ts_expectile", "ts_expectile_beta"})
    for _canon in ("ts_expectile", "ts_expectile_beta"):
        register_polars_udf(_canon)


_register_surface()
