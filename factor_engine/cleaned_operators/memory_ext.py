# -*- coding: utf-8 -*-
"""Serial-dependence memory operators (2026-08 geometry/math expansion).

Two complements:

* ``ts_autocorrelation_time`` — integrated autocorrelation time ``tau_int`` of
  the window.  Sample autocorrelations ``rho(k)`` are computed over aligned
  finite pairs for ``k = 1..max_lag``; the sum ``1 + 2*sum rho(k)`` accumulates
  only while ``rho(k) > 0`` (stops at the first non-positive value) and is then
  normalised by ``max_lag`` so the output stays on a comparable scale.
* ``ts_fractional_difference`` — the fractional differencing transform
  ``y_t = sum_{k=0}^{min(cutoff, t)} w_k * x_{t-k}`` with the binomial-weight
  recursion ``w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k`` for ``d in (-1, 1)``.
  ``d > 0`` removes long memory (stationarising), ``d < 0`` is fractional
  integration.

All operators are trailing-window / causal, deterministic and NaN-safe: the
autocorrelation sums stop on missing lags, and a fractional-difference output
is NaN whenever any input in its causal support is NaN (values are never
re-connected across a gap).

NOTE on the parameter name: the fractional differencing order is exposed as
``fd`` (not ``d``) because the framework's base layer auto-normalises the
integer-typed parameter named ``d`` and would reject the required fractional
value ``0.4``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="memory",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "memory", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:long_memory",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _check_window(window: int) -> int:
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    return w


def _autocorrelation_time(chunk: np.ndarray, max_lag: int) -> float:
    valid = chunk[np.isfinite(chunk)]
    n = int(valid.size)
    if n < 3:
        return np.nan
    mu = float(np.mean(valid))
    denom = float(np.sum((valid - mu) ** 2))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    tau = 1.0
    m = int(max_lag)
    total_len = int(chunk.shape[0])
    for k in range(1, m + 1):
        if total_len <= k:
            break
        x = chunk[: total_len - k]
        y = chunk[k:]
        finite = np.isfinite(x) & np.isfinite(y)
        cnt = int(finite.sum())
        if cnt < 2:
            break
        num = float(np.sum((x[finite] - mu) * (y[finite] - mu)))
        rho = num / denom
        if not np.isfinite(rho) or rho <= 0.0:
            break
        tau += 2.0 * rho
    return tau / m


def _autocorrelation_time_series(x2d: np.ndarray, window: int, max_lag: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    ml = int(max_lag)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _autocorrelation_time(col[i0 : r + 1], ml)
    return out


def _fd_weights(fd: float, cutoff: int) -> np.ndarray:
    c = int(cutoff)
    w = np.empty(c + 1, dtype=float)
    w[0] = 1.0
    for k in range(1, c + 1):
        w[k] = -w[k - 1] * (fd - k + 1) / k
    return w


def _fractional_difference_series(x2d: np.ndarray, fd: float, cutoff: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    c = int(cutoff)
    w = _fd_weights(fd, c)
    for col in range(cols):
        colv = x2d[:, col]
        for r in range(rows):
            k = c if c < r else r  # min(cutoff, t)
            seg = colv[r - k : r + 1]  # x_{t-k}..x_t
            if not np.isfinite(seg).all():
                out[r, col] = np.nan
                continue
            rev = seg[::-1]  # x_t, x_{t-1}, ..., x_{t-k}
            out[r, col] = float(np.dot(rev, w[: k + 1]))
    return out


@register_operator(
    name="ts_autocorrelation_time",
    category="memory",
    business_category="memory",
    canonical="ts_autocorrelation_time",
    source="memory_ext",
)
class TsAutocorrelationTime(SeriesOperator):
    """积分自相关时间 tau_int = 1 + 2*sum rho(k)（rho>0 时累加），按 max_lag 归一。

    大 -> 强序列依赖（动量/慢变量）；小 -> 近白噪声。随机游走等非平稳序列
    会给出持续为正的自相关，tau_int 偏大，正好表达其长记忆。P1。
    """

    metadata = _metadata(
        "ts_autocorrelation_time",
        "积分自相关时间（正自相关累加，按 max_lag 归一化）。",
        ["x", "window", "max_lag"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, max_lag: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        ml = int(max_lag)
        if ml < 1:
            raise ValueError("max_lag must be >= 1")
        return frame_like(x, _autocorrelation_time_series(x.to_numpy(dtype=float), w, ml))


@register_operator(
    name="ts_fractional_difference",
    category="memory",
    business_category="memory",
    canonical="ts_fractional_difference",
    source="memory_ext",
)
class TsFractionalDifference(SeriesOperator):
    """分数差分变换 y_t = sum_k w_k x_{t-k}（二项权重递归，d∈(-1,1)）。

    d>0 移除长记忆（平稳化）；d<0 为分数积分。输出长度与输入一致；
    因果支持内任意 NaN -> NaN。P1。
    """

    metadata = _metadata(
        "ts_fractional_difference",
        "分数差分变换（二项权重，d∈(-1,1) 内），长度保持。",
        ["x", "fd", "cutoff"],
        unit="series",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, fd: float = 0.4, cutoff: int = 20, **_: Any) -> pd.DataFrame:
        fdv = float(fd)
        if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
            raise ValueError("fd must satisfy -1 < fd < 1")
        c = int(cutoff)
        if c < 1:
            raise ValueError("cutoff must be >= 1")
        return frame_like(x, _fractional_difference_series(x.to_numpy(dtype=float), fdv, c))


_NEW_CANONICALS = (
    "ts_autocorrelation_time",
    "ts_fractional_difference",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
