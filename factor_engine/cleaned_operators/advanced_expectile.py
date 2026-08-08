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

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    aligned_pairs,
    frame_like,
    map_pair_rolling,
    map_rolling,
    register_polars_udf,
)

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
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
    )


def _expectile(vals: np.ndarray, tau: float) -> float:
    """Newey-Powell fixed-point expectile of a finite value array."""
    v = vals[np.isfinite(vals)]
    if v.size < 3:
        return np.nan
    e = float(np.mean(v))
    for _ in range(200):
        w = np.abs(tau - (v < e).astype(float))
        s = float(w.sum())
        if s <= _EPS:
            break
        e_new = float(np.sum(w * v)) / s
        if abs(e_new - e) <= 1e-12 * max(1.0, abs(e)):
            e = e_new
            break
        e = e_new
    return e


def _expectile_chunk(chunk: np.ndarray, tau: float) -> float:
    return _expectile(chunk, tau)


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
        ["x", "window", "tau"],
        domain="price_volume",
        unit="same_as:target",
        cost=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        tau: float = 0.1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        tt = float(tau)
        if not (0.0 < tt < 1.0):
            raise ValueError("ts_expectile requires 0 < tau < 1")
        if w < 3:
            raise ValueError("ts_expectile requires window >= 3")
        return frame_like(
            x,
            map_rolling(x.to_numpy(dtype=float), w, lambda c: _expectile_chunk(c, tt)),
        )


def _expectile_slope(yv: np.ndarray, xv: np.ndarray, tau: float) -> float:
    """IRLS asymmetric least-squares slope of y ~ a + b·x at expectile tau."""
    yy, xx = aligned_pairs(yv, xv)
    n = yy.size
    if n < 4:
        return np.nan
    sx = float(np.std(xx))
    if sx <= _EPS:
        return np.nan
    A = np.vstack([xx, np.ones(n)]).T
    try:
        beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    for _ in range(40):
        r = yy - A @ beta
        w = np.abs(tau - (r < 0.0).astype(float))
        sw = float(w.sum())
        if sw <= _EPS:
            break
        W = np.sqrt(w)
        try:
            beta_new = np.linalg.lstsq(A * W[:, None], yy * W, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan
        if float(np.max(np.abs(beta_new - beta))) <= 1e-10 * max(1.0, float(np.max(np.abs(beta)))):
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0])


def _expectile_beta_chunk(yc: np.ndarray, xc: np.ndarray, tau: float) -> float:
    return _expectile_slope(yc, xc, tau)


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
        ["y", "x", "window", "tau"],
        domain="price_volume",
        # A regression slope carries unit(y)/unit(x) — NOT a fixed ratio.  With
        # y = price and x = amount the slope is per-amount price, which is not
        # dimensionless (P2-7).  The Typed Unit Algebra derives this at runtime.
        unit="unit(y)/unit(x)",
        cost=5,
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        window: int = 60,
        tau: float = 0.1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        tt = float(tau)
        if not (0.0 < tt < 1.0):
            raise ValueError("ts_expectile_beta requires 0 < tau < 1")
        if w < 4:
            raise ValueError("ts_expectile_beta requires window >= 4")
        return frame_like(
            y,
            map_pair_rolling(
                y.to_numpy(dtype=float),
                x.to_numpy(dtype=float),
                w,
                lambda a, b: _expectile_beta_chunk(a, b, tt),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {"ts_expectile", "ts_expectile_beta"}
    )
    for _canon in ("ts_expectile", "ts_expectile_beta"):
        register_polars_udf(_canon)


_register_surface()
