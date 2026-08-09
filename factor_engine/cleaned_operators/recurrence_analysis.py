# -*- coding: utf-8 -*-
"""Recurrence Quantification Analysis (RQA) operators (2026-08 deepening).

RQA characterises the *recurrence structure* of a path: how often the system
returns to similar states and how the returns are organised into diagonal /
vertical line structures.  Unlike ``state_density`` (which only looks near the
current point) and the matrix-profile family (subsequence matching), RQA
summarises the whole trailing window's self-revisit geometry.

Shared kernel: for each trailing window, phase-space embedding points
``X_i = [x_i, x_{i+delay}, ..]`` (``dim`` ``x`` ``delay``) build the recurrence
matrix ``R_ij = 1[‖X_i - X_j‖ ≤ ε]`` with ``ε`` taken as a fixed quantile of
the pairwise distances (scale-invariant, deterministic — no search parameter).
From ``R`` four statistics are derived:

* ``ts_recurrence_rate``           — fraction of off-diagonal recurrences.
* ``ts_recurrence_diagonal_entropy`` — entropy of diagonal line lengths.
* ``ts_recurrence_trapping_time``  — mean vertical line length (how long the
  path is typically "stuck" in a state).
* ``ts_recurrence_divergence``     — 1 / longest diagonal line (local divergence
  proxy, cheaper and more stable than a local Lyapunov exponent).

All operators are trailing-window, prefix-causal and deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _trailing_contiguous_finite(chunk: np.ndarray) -> np.ndarray:
    """Longest trailing contiguous finite run (never re-connects across a gap).

    R5 P0-03: a NaN at the current row returns the empty block so the caller
    emits NaN instead of re-using the previous contiguous historical run.
    """
    n = chunk.shape[0]
    if n == 0 or not np.isfinite(chunk[-1]):
        return chunk[:0]
    end = n
    while end > 0 and np.isfinite(chunk[end - 1]):
        end -= 1
    return chunk[end:]


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_recurrence",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_recurrence", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:path_geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _recurrence_stats_window(
    v: np.ndarray,
    dim: int,
    delay: int,
    eps_fraction: float,
    min_line: int,
) -> tuple[float, float, float, float] | None:
    """RQA statistics (rate, diag_entropy, trapping_time, divergence) for one
    finite window ``v``.  Returns None when the embedding is degenerate.

    ``ε = eps_fraction · scale`` where scale is the window's robust dispersion
    (1.4826·MAD).  The threshold is therefore scale-relative and deterministic;
    a quantile-of-distances threshold would pin the recurrence rate to the
    quantile regardless of the signal, which is uninformative."""
    M = v.shape[0] - (dim - 1) * delay
    if M < 4:
        return None
    if dim == 1:
        P = v[:M].reshape(-1, 1)
    else:
        P = np.stack([v[i : i + M] for i in range(0, dim * delay, delay)], axis=1)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= _EPS:
        scale = float(np.std(v))
    if not np.isfinite(scale) or scale <= _EPS:
        return None
    # R6-113: phase-space distance grows ~√dim in dim dimensions; a fixed
    # epsilon does not represent a fixed neighbourhood density across dim.
    # Normalise epsilon by √dim for comparability.
    eps = float(eps_fraction) * scale / float(np.sqrt(max(1, int(dim))))
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    R = (D <= eps) & ~np.eye(M, dtype=bool)

    total_pairs = M * (M - 1)
    if total_pairs <= 0:
        return None
    rate = float(R.sum()) / total_pairs

    # Diagonal line lengths (upper triangle, off the main diagonal).
    diag_lengths: list[int] = []
    for off in range(1, M):
        length = 0
        i, j = 0, off
        while j < M:
            if R[i, j]:
                length += 1
            else:
                if length >= min_line:
                    diag_lengths.append(length)
                length = 0
            i += 1
            j += 1
        if length >= min_line:
            diag_lengths.append(length)

    Lmax = max(diag_lengths, default=0)
    if diag_lengths:
        lens = np.asarray(diag_lengths, dtype=float)
        uniq, counts = np.unique(lens, return_counts=True)
        p = counts / counts.sum()
        ent = -float(np.sum(p * np.log(p)))
        # R6-115: the previous normalisation divided by log(#unique lengths seen
        # in THIS window), so two windows with identical probability structure
        # got different values purely because one happened to contain a rare
        # extra length.  Normalise by the FIXED theoretical support of the
        # diagonal-length distribution instead: lengths run from ``min_line`` to
        # ``M`` (the phase-space size), so the maximum entropy is log(M-min_line+1)
        # regardless of which lengths actually appeared.
        support = max(2, M - int(min_line) + 1)
        ent /= np.log(support)
    else:
        ent = np.nan

    # Vertical line lengths (consecutive recurrences down a column).
    vert_lengths: list[int] = []
    for j in range(M):
        length = 0
        for i in range(M):
            if R[i, j]:
                length += 1
            else:
                if length >= min_line:
                    vert_lengths.append(length)
                length = 0
        if length >= min_line:
            vert_lengths.append(length)
    trapping = float(np.mean(vert_lengths)) if vert_lengths else np.nan
    divergence = 1.0 / Lmax if Lmax >= min_line else np.nan
    return rate, ent, trapping, divergence


def _recurrence_series(
    values: np.ndarray,
    window: int,
    dim: int,
    delay: int,
    eps_fraction: float,
    min_line: int,
    min_periods: int,
    which: int,
) -> np.ndarray:
    rows, cols = values.shape
    w = max(2, int(window))
    mp = max(4, int(min_periods))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            chunk = values[lo : r + 1, c]
            # Audit P1-B: never compress NaN out of the time axis.  A phase-space
            # embedding coordinate that is missing invalidates the embedding; the
            # longest trailing contiguous finite run keeps the time structure and
            # never re-connects data across a gap.
            v = _trailing_contiguous_finite(chunk)
            if v.size < mp:
                continue
            stats = _recurrence_stats_window(v, dim, delay, eps_fraction, min_line)
            if stats is None:
                continue
            out[r, c] = stats[which]
    return out


def _check_params(window: int, dim: int, delay: int, eps_fraction: float, min_line: int) -> tuple[int, int, int, float, int]:
    w = max(2, int(window))
    d = max(1, int(dim))
    dl = max(1, int(delay))
    ef = float(eps_fraction)
    if not 0.0 < ef < 1.0:
        raise ValueError("eps_fraction must be in (0, 1)")
    if d * dl > 4:
        raise ValueError("dimension*delay must be <= 4 (embedding support)")
    ml = max(2, int(min_line))
    return w, d, dl, ef, ml


@register_operator(
    name="ts_recurrence_rate",
    category="time_series_recurrence",
    business_category="time_series_recurrence",
    canonical="ts_recurrence_rate",
    source="recurrence_analysis",
)
class TsRecurrenceRate(SeriesOperator):
    """递归率 RR: 窗口内相空间点对的回访比例 ``Σ R_ij / (M·(M-1))``。

    ε = eps_fraction × 窗口稳健尺度（1.4826·MAD），尺度无关、确定性（若用点对
    距离的分位数会把 RR 钉死在分位数值上，失去信息）。RR 高 = 系统经常回到
    类似状态（周期性 / 强自相关结构）；低 = 路径持续探索新状态。与 state_density
    （只看当前点附近）不同：RR 看**整条窗口路径自身**的回访结构。PIT 安全。
    """

    metadata = _metadata(
        "ts_recurrence_rate",
        "递归率 RR：窗口路径自身的回访占比。",
        ["x", "window", "dim", "delay", "eps_fraction", "min_periods"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 40,
        dim: int = 1,
        delay: int = 1,
        eps_fraction: float = 0.1,
        min_periods: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        w, d, dl, eq, ml = _check_params(window, dim, delay, eps_fraction, 2)
        out = _recurrence_series(x.to_numpy(dtype=float), w, d, dl, eq, ml, min_periods, 0)
        return frame_like(x, out)


@register_operator(
    name="ts_recurrence_diagonal_entropy",
    category="time_series_recurrence",
    business_category="time_series_recurrence",
    canonical="ts_recurrence_diagonal_entropy",
    source="recurrence_analysis",
)
class TsRecurrenceDiagonalEntropy(SeriesOperator):
    """递归图对角线段长度的熵 ``-Σ p(l) log p(l)``（按不同长度数归一）。

    衡量重复轨迹的持续长度是否高度多样：高 = 递归的持续时间分散（各种长度的
    重复都有）；低 = 递归几乎只有一种典型时长。与 permutation entropy（顺序
    复杂性）不同，这是**递归结构的形状**。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_recurrence_diagonal_entropy",
        "对角线长度分布熵（[0,1]，高=长度多样）。",
        ["x", "window", "dim", "delay", "eps_fraction", "min_periods"],
        unit="entropy",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 40,
        dim: int = 1,
        delay: int = 1,
        eps_fraction: float = 0.1,
        min_periods: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        w, d, dl, eq, ml = _check_params(window, dim, delay, eps_fraction, 2)
        out = _recurrence_series(x.to_numpy(dtype=float), w, d, dl, eq, ml, min_periods, 1)
        return frame_like(x, out)


@register_operator(
    name="ts_recurrence_trapping_time",
    category="time_series_recurrence",
    business_category="time_series_recurrence",
    canonical="ts_recurrence_trapping_time",
    source="recurrence_analysis",
)
class TsRecurrenceTrappingTime(SeriesOperator):
    """递归捕捉时间: 垂直线段长度的均值 ``E[v]``。

    系统进入某类状态后平均"卡"多久。与 laminarity（垂直线点的占比）互补：
    laminarity 回答"卡住的密度"，trapping time 回答"卡住以后平均卡多久"。
    高 = 状态锁定后持续自重复。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_recurrence_trapping_time",
        "垂直线平均长度 E[v]（状态平均卡住时长）。",
        ["x", "window", "dim", "delay", "eps_fraction", "min_periods"],
        unit="bars",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 40,
        dim: int = 1,
        delay: int = 1,
        eps_fraction: float = 0.1,
        min_periods: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        w, d, dl, eq, ml = _check_params(window, dim, delay, eps_fraction, 2)
        out = _recurrence_series(x.to_numpy(dtype=float), w, d, dl, eq, ml, min_periods, 2)
        return frame_like(x, out)


@register_operator(
    name="ts_recurrence_divergence",
    category="time_series_recurrence",
    business_category="time_series_recurrence",
    canonical="ts_recurrence_divergence",
    source="recurrence_analysis",
)
class TsRecurrenceDivergence(SeriesOperator):
    """递归发散度 ``1 / L_max``（L_max = 最长对角线长度）。

    局部轨迹发散程度的简单代理：最长重复延续段越短 → 发散度越高（路径很少
    沿相同轨迹走远）。比直接做局部 Lyapunov 更稳定、更便宜。PIT 安全、确定性。
    """

    metadata = _metadata(
        "ts_recurrence_divergence",
        "1/最长对角线长度（局部发散代理）。",
        ["x", "window", "dim", "delay", "eps_fraction", "min_periods"],
        # R6-114: DIV = 1/L_max has units of 1/bar (inverse of a length in
        # bars), NOT a dimensionless ratio.  Declared honestly as inverse_bars.
        unit="inverse_bars",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 40,
        dim: int = 1,
        delay: int = 1,
        eps_fraction: float = 0.1,
        min_periods: int = 10,
        **_: Any,
    ) -> pd.DataFrame:
        w, d, dl, eq, ml = _check_params(window, dim, delay, eps_fraction, 2)
        out = _recurrence_series(x.to_numpy(dtype=float), w, d, dl, eq, ml, min_periods, 3)
        return frame_like(x, out)


_NEW_CANONICALS = (
    "ts_recurrence_rate",
    "ts_recurrence_diagonal_entropy",
    "ts_recurrence_trapping_time",
    "ts_recurrence_divergence",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
