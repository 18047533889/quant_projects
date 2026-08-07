# -*- coding: utf-8 -*-
"""Extreme-value tail shape and quantile relationship operators (2026-08 V3).

* ``ts_hill_tail_index``        — Hill estimator of the tail index ξ (shape, not
  size): how thick the extreme tail is (P1).  Upper/lower via ``side``.
* ``ts_quantile_regression_beta`` — exact quantile-regression slope β_q via a
  linear program (P2 research): the marginal relationship of ``x`` when the
  stock is in its own q-th return state.

Deterministic (fixed threshold fraction, LP with HiGHS), prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="extreme_tail",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "extreme_tail", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _column_map(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out


# ---------------------------------------------------------------------------
# ts_hill_tail_index
# ---------------------------------------------------------------------------

def _hill_series(series: np.ndarray, window: int, side: str, tail_fraction: float, min_tail_count: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    frac = min(max(_EPS, float(tail_fraction)), 0.5)
    mtc = max(3, int(min_tail_count))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        y = chunk if side == "upper" else -chunk
        y = y[np.isfinite(y)]
        y = y[y > 0.0]
        if y.size < mtc:
            continue
        k = int(np.floor(y.size * frac))
        if k < mtc:
            continue
        ys = np.sort(y)
        threshold = ys[y.size - k - 1]
        if threshold <= 0.0 or not np.isfinite(threshold):
            continue
        top = ys[y.size - k :]
        xi = float(np.mean(np.log(top / threshold)))
        out[t] = xi
    return out


@register_operator(
    name="ts_hill_tail_index",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_hill_tail_index",
    source="extreme_tail",
)
class TsHillTailIndex(SeriesOperator):
    """Hill 尾部指数 ξ（上/下尾厚度形状，非大小）。

    对窗口内 Y（upper=+x / lower=-x，仅保留 Y>0）排序，取 ``k=floor(n*frac)``
    个大值：``ξ = (1/k) Σ log(Y_{n-j+1}/Y_{n-k})``。ξ 越大尾部越厚、极端观测越
    容易远离普通观测。``min_tail_count`` 内样本不足则 NaN。与 ES/partial
    moment（尾部大小）互补。P1。
    """

    metadata = _metadata(
        "ts_hill_tail_index",
        "Hill 尾部指数 ξ（上/下尾厚度形状）。",
        ["x", "window", "side", "tail_fraction", "min_tail_count"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        side: str = "upper",
        tail_fraction: float = 0.2,
        min_tail_count: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        side_k = str(side).lower()
        if side_k not in {"upper", "lower"}:
            raise ValueError("ts_hill_tail_index requires side in {'upper','lower'}")
        return _frame_like_result(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _hill_series(s, window, side_k, tail_fraction, min_tail_count),
            ),
        )


def _frame_like_result(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return frame_like(template, values)


# ---------------------------------------------------------------------------
# ts_quantile_regression_beta (P2 research)
# ---------------------------------------------------------------------------

def _quantile_beta(x: np.ndarray, y: np.ndarray, q: float) -> float:
    n = x.shape[0]
    if n < 3:
        return np.nan
    xc = x - x.mean()
    if float(np.sum(xc * xc)) <= _EPS:
        return np.nan
    from scipy.optimize import linprog

    # variables: [alpha, beta, u_1..u_n, v_1..v_n];  alpha + beta x_i + u_i - v_i = y_i
    dim = 2 + 2 * n
    c = np.zeros(dim)
    c[2 : 2 + n] = q
    c[2 + n :] = 1.0 - q
    A_eq = np.zeros((n, dim))
    for i in range(n):
        A_eq[i, 0] = 1.0
        A_eq[i, 1] = x[i]
        A_eq[i, 2 + i] = 1.0
        A_eq[i, 2 + n + i] = -1.0
    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n)
    try:
        result = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    except Exception:
        return np.nan
    if result is None or result.status != 0 or result.x is None:
        return np.nan
    beta = float(result.x[1])
    if not np.isfinite(beta):
        return np.nan
    return beta


def _quantile_beta_series(y: np.ndarray, x: np.ndarray, window: int, q: float) -> np.ndarray:
    rows, cols = y.shape
    w = max(2, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for t in range(rows):
            lo = max(0, t - w + 1)
            if t - lo + 1 < w:
                continue
            xw = x[lo : t + 1, c]
            yw = y[lo : t + 1, c]
            finite = np.isfinite(xw) & np.isfinite(yw)
            if int(finite.sum()) < 3:
                continue
            out[t, c] = _quantile_beta(xw[finite], yw[finite], q)
    return out


@register_operator(
    name="ts_quantile_regression_beta",
    category="extreme_tail",
    business_category="extreme_tail",
    canonical="ts_quantile_regression_beta",
    source="extreme_tail",
    status="experimental",
)
class TsQuantileRegressionBeta(SeriesOperator):
    """精确分位数回归斜率 β_q（LP 求解）。

    最小化 ``Σ ρ_q(y - α - βx)``（``ρ_q(u)=u(q-I(u<0))``），exact 线性规划
    (HiGHS)，不用不稳定的 IRLS 近似。q=0.1 回答"股票自己处于最差收益状态时 x
    对它的边际关系"。P2 / Research。
    """

    metadata = _metadata(
        "ts_quantile_regression_beta",
        "分位数回归斜率 β_q（精确 LP）。",
        ["y", "x", "window", "quantile"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(
        self, y: pd.DataFrame, x: pd.DataFrame, window: int = 120, quantile: float = 0.5, **_: Any
    ) -> pd.DataFrame:
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("ts_quantile_regression_beta requires quantile in (0, 1)")
        return _frame_like_result(
            y,
            _quantile_beta_series(y.to_numpy(dtype=float), x.to_numpy(dtype=float), window, q),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"ts_hill_tail_index"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"ts_quantile_regression_beta"}
    )


_register_surface()
