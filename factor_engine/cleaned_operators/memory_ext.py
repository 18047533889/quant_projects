# -*- coding: utf-8 -*-
"""Serial-dependence memory operators (2026-08 geometry/math expansion).

Two complements:

* ``ts_autocorrelation_time`` — integrated autocorrelation time ``tau_int`` of
  the window, in bars (review R4-56).  Sample autocorrelations ``rho(k)`` are
  computed for ``k = 1..max_lag`` over the *trailing contiguous finite window*
  (review R4-57: the same cohort is used for the mean, the total variance and
  every lag-k numerator — never a re-connected axis); the sum
  ``1 + 2*sum rho(k)`` accumulates only while ``rho(k) > 0`` (stops at the
  first non-positive value) and is NOT divided by ``max_lag``, so white noise
  gives ``~1`` bar regardless of ``max_lag``.  The precise spelling
  ``ts_integrated_autocorrelation_time`` is an alias of this canonical.
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


def _trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Trailing contiguous run of finite values (suffix after the last gap)."""
    finite = np.isfinite(chunk)
    if not np.any(finite):
        return np.array([], dtype=float)
    bad = np.flatnonzero(~finite)
    if bad.size == 0:
        return chunk
    return chunk[int(bad[-1]) + 1 :]


def _autocorrelation_time(chunk: np.ndarray, max_lag: int) -> float:
    # Review R4-57: the mean / total variance and every lag-k numerator use the
    # SAME cohort — the trailing contiguous finite window.  Values are never
    # re-connected across a gap.
    contig = _trailing_contiguous_finite(chunk)
    n = int(contig.size)
    if n < 3:
        return np.nan
    mu = float(np.mean(contig))
    denom = float(np.sum((contig - mu) ** 2))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    tau = 1.0
    m = int(max_lag)
    for k in range(1, m + 1):
        if n <= k:
            break
        x = contig[: n - k]
        y = contig[k:]
        num = float(np.sum((x - mu) * (y - mu)))
        rho = num / denom
        if not np.isfinite(rho) or rho <= 0.0:
            break
        tau += 2.0 * rho
    # Review R4-56: true integrated autocorrelation time in bars — do NOT divide
    # by max_lag (white noise would collapse to 1/max_lag and max_lag 20 vs 40
    # would mechanically halve the output).
    return tau


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
            # R11 #41: the fractional filter has a causal support of ``cutoff+1``
            # terms (x_t .. x_{t-cutoff}).  The startup region ``r < cutoff``
            # cannot apply the declared filter — a shortened prefix is a DIFFERENT
            # transform under the same factor name, so it fails closed (NaN)
            # until the full filter history is available (min_periods =
            # cutoff + 1 rows).
            if r < c:
                continue
            seg = colv[r - c : r + 1]  # x_{t-c}..x_t
            if not np.isfinite(seg).all():
                out[r, col] = np.nan
                continue
            rev = seg[::-1]  # x_t, x_{t-1}, ..., x_{t-c}
            out[r, col] = float(np.dot(rev, w))
    return out


@register_operator(
    name="ts_autocorrelation_time",
    category="memory",
    business_category="memory",
    canonical="ts_autocorrelation_time",
    source="memory_ext",
)
class TsAutocorrelationTime(SeriesOperator):
    """积分自相关时间 ``tau_int = 1 + 2*sum rho(k)``（rho>0 时累加），单位 bars。

    传统 integrated autocorrelation time，输出**不除以 max_lag**（R4-56）：
    白噪声 → τ≈1 bar，与 max_lag 无关；max_lag 只截断求和的滞后上限。
    大 -> 强序列依赖（动量/慢变量）；小 -> 近白噪声。随机游走等非平稳序列
    会给出持续为正的自相关，tau_int 偏大，正好表达其长记忆。统计量在尾部
    连续有限窗口上计算（R4-57），跨 gap 不重连。P1。
    """

    metadata = _metadata(
        "ts_autocorrelation_time",
        "积分自相关时间（bars，正自相关累加，不除以 max_lag）。",
        ["x", "window", "max_lag"],
        unit="bars",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 120, max_lag: int = 20, **_: Any) -> pd.DataFrame:
        w = _check_window(window)
        ml = int(max_lag)
        # R5-39: lag truncation is capped by the window — ``max_lag >= window``
        # is a nonsensical search node (lags beyond the window are never
        # computed, so it manufactures duplicate factors).
        if ml < 1:
            raise ValueError("max_lag must be >= 1")
        if ml >= w:
            raise ValueError("max_lag must be < window")
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
    因果支持内任意 NaN -> NaN。R11 #41：启动区 ``r < cutoff``（分数滤波
    历史未收敛）为 NaN，直到完整 ``cutoff+1`` 个因果项可用。P1。
    """

    metadata = _metadata(
        "ts_fractional_difference",
        "分数差分变换（二项权重，d∈(-1,1) 内），启动区 NaN。",
        ["x", "fd", "cutoff"],
        unit="series",
        cost=3,
    )

    def _calculate_series(self, x: pd.DataFrame, fd: float = 0.4, cutoff: int = 20, **_: Any) -> pd.DataFrame:
        fdv = float(fd)
        if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
            raise ValueError("fd must satisfy -1 < fd < 1")
        if isinstance(cutoff, (bool, np.bool_)):
            raise ValueError("cutoff must be an integer, not bool")
        cf = float(cutoff)
        if not np.isfinite(cf) or cf != float(int(cf)):
            raise ValueError("cutoff must be an integer")
        c = int(cf)
        if c < 1:
            raise ValueError("cutoff must be >= 1")
        return frame_like(x, _fractional_difference_series(x.to_numpy(dtype=float), fdv, c))


_NEW_CANONICALS = (
    "ts_autocorrelation_time",
    "ts_fractional_difference",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _register_precise_alias() -> None:
    """R4-56: expose the precise name ``ts_integrated_autocorrelation_time``.

    The canonical stays ``ts_autocorrelation_time`` (polars_geometry_math
    hard-codes that name in its polars-backend list, so a canonical rename is
    not loadable without editing that module).  The precise spelling is
    registered as an alias so new DSL expressions can use it.
    """
    from cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_alias(
        "ts_integrated_autocorrelation_time",
        "ts_autocorrelation_time",
    )


_register_surface()
_register_precise_alias()
