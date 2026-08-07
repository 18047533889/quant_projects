# -*- coding: utf-8 -*-
"""Candle state-space operators (2026-08 geometry/math expansion).

A candle is reduced to a 4-dimensional *state vector* per row per symbol (e.g.
body ratio, wick ratios, range/ATR), so a run of candles becomes a point
trajectory through feature space.  This module characterises how the *current
candle as an object* sits inside its own history:

* ``ts_vector_state_mahalanobis``          — anomaly of today's state vector vs
  the trailing covariance of its own history (shrinkage-regularised
  Mahalanobis distance).  High = today's candle is atypical for the symbol.
* ``ts_vector_state_local_density``        — local density of today's state
  among its prior neighbours (inverse k-th nearest distance).  High = common
  state, low = rare / isolated state.
* ``ts_multivariate_matrix_profile_novelty`` — distance of the current trailing
  subsequence of the caller-supplied scalar ``x`` to its nearest *prior*
  subsequence under z-normalised Euclidean distance (novelty).
* ``ts_matrix_profile_motif_age``          — normalised age of the nearest
  historical match of the current subsequence (same matrix-profile kernel).

Shared kernels are private to this module.  All operators are trailing-window,
prefix-causal (row ``r`` uses rows ``<= r`` only) and deterministic; NaN inputs
are dropped from the window, and a window with no valid rows emits NaN.  Every
operator is per-column (each symbol processed independently), so one symbol's
state never leaks into another's.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="candle_state_space",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "candle_state_space", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:vector_state",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# kernels (per column; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _mahalanobis_series(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    window: int, shrinkage: float,
) -> np.ndarray:
    rows, cols = f1.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    lam = float(shrinkage)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")
    if w < 2:
        raise ValueError("window must be >= 2")
    feat = (f1, f2, f3, f4)
    p = len(feat)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            win = np.stack([feat[j][i0 : r + 1, c] for j in range(p)], axis=1)  # (n, p)
            if not np.all(np.isfinite(win[r - i0])):
                continue  # current state vector must be complete
            finite = np.all(np.isfinite(win), axis=1)
            valid = win[finite].astype(float)
            if valid.shape[0] < 2:
                continue  # need at least two aligned rows for a covariance
            z = win[r - i0].astype(float)
            mu = np.median(valid, axis=0)
            cov = np.atleast_2d(np.cov(valid, rowvar=False, ddof=1))
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            shrunk = (1.0 - lam) * cov + lam * np.diag(np.diag(cov))
            prec = np.linalg.pinv(shrunk + _EPS * np.eye(p))
            d = z - mu
            D = float(np.sqrt(max(0.0, float(d @ prec @ d))))
            out[r, c] = D
    return out


def _local_density_series(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray,
    window: int, k: int,
) -> np.ndarray:
    rows, cols = f1.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    kk = int(k)
    if w < 2:
        raise ValueError("window must be >= 2")
    if kk < 1:
        raise ValueError("k must be >= 1")
    feat = (f1, f2, f3, f4)
    p = len(feat)
    for c in range(cols):
        for r in range(rows):
            i0 = max(0, r - w + 1)
            win = np.stack([feat[j][i0 : r + 1, c] for j in range(p)], axis=1)  # (n, p)
            if not np.all(np.isfinite(win[r - i0])):
                continue
            finite = np.all(np.isfinite(win), axis=1)
            valid = win[finite].astype(float)
            if valid.shape[0] < kk + 1:
                continue  # need current + k prior valid rows
            # standardise each feature by the trailing window mean / std
            mu = valid.mean(axis=0)
            sd = valid.std(axis=0)
            sd = np.where(sd > _EPS, sd, 1.0)
            z = (valid - mu) / sd
            zc = z[-1]  # current row is the last valid row of the trailing window
            prior = z[:-1]  # exclude the current point
            if prior.shape[0] < kk:
                continue
            dists = np.linalg.norm(prior - zc, axis=1)
            rk = float(np.partition(dists, kk - 1)[kk - 1])  # k-th nearest distance
            out[r, c] = 1.0 / (rk + _EPS)
    return out


def _z_normalize(sub: np.ndarray) -> np.ndarray:
    mu = sub.mean()
    sd = sub.std()
    if sd <= _EPS:
        return np.zeros_like(sub, dtype=float)
    return (sub - mu) / sd


def _matrix_profile_series(
    x2d: np.ndarray, window: int, subsequence_length: int, history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared matrix-profile kernel -> (novelty, motif_age/history) per row.

    For every row ``t`` the current trailing subsequence is ``x[t-L+1..t]`` and
    the historical search band for prior-subsequence start indices is
    ``[t-history .. t-L]`` (clamped to the data).  ``window`` is validated as an
    upper bound (``window >= history``) and also clamps the band, which is
    redundant once ``window >= history`` holds — it keeps the parameter an
    explicit part of the kernel contract.
    """
    rows, cols = x2d.shape
    novelty = np.full((rows, cols), np.nan, dtype=float)
    age = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    L = int(subsequence_length)
    h = int(history)
    if L < 2:
        raise ValueError("subsequence_length must be >= 2")
    if h < L:
        raise ValueError("history must be >= subsequence_length")
    if w < h:
        raise ValueError("window must be >= history")
    for c in range(cols):
        x = x2d[:, c]
        for r in range(rows):
            if r + 1 < L:
                continue  # current subsequence not complete yet
            z = x[r - L + 1 : r + 1]
            if not np.all(np.isfinite(z)):
                continue
            s0 = max(0, r - h, r - w)
            s1 = r - L  # latest start index for a strictly prior subsequence
            if s0 > s1:
                continue
            zz = _z_normalize(z)
            best_d = np.inf
            best_s = -1
            for s in range(s0, s1 + 1):
                cand = x[s : s + L]
                if not np.all(np.isfinite(cand)):
                    continue
                cz = _z_normalize(cand)
                d = float(np.linalg.norm(zz - cz))
                if d < best_d:
                    best_d = d
                    best_s = s
            if np.isfinite(best_d):
                novelty[r, c] = best_d
                age[r, c] = (r - best_s) / float(h)
    return novelty, age


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_vector_state_mahalanobis",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_vector_state_mahalanobis",
    source="candle_state_space",
)
class TsVectorStateMahalanobis(SeriesOperator):
    """状态向量马氏距离：今天的蜡烛对象相对自身历史协方差的异常度。

    ``z_t=(f1..f4)_t`` 对比窗口内逐特征中位数 μ 与收缩协方差
    ``(1-λ)Σ+λ·diag(Σ)``；距离越大 → 今天蜡烛相对自身历史越异常。P1。
    """

    metadata = _metadata(
        "ts_vector_state_mahalanobis",
        "4D 状态向量相对窗口收缩协方差的马氏距离（蜡烛对象异常度）。",
        ["f1", "f2", "f3", "f4", "window", "shrinkage"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, f4: pd.DataFrame,
        window: int = 60, shrinkage: float = 0.5, **_: Any,
    ) -> pd.DataFrame:
        arr = _mahalanobis_series(
            f1.to_numpy(dtype=float), f2.to_numpy(dtype=float),
            f3.to_numpy(dtype=float), f4.to_numpy(dtype=float),
            window, shrinkage,
        )
        return frame_like(f1, arr)


@register_operator(
    name="ts_vector_state_local_density",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_vector_state_local_density",
    source="candle_state_space",
)
class TsVectorStateLocalDensity(SeriesOperator):
    """状态向量局部密度：今天的状态在自身先验近邻中的疏密程度。

    各特征按窗口标准差标准化后，取当前点距窗口内前序点第 k 近的距离 r_k，
    密度 = 1/(r_k+eps)。高 → 常见状态；低 → 稀有/孤立状态。P1。
    """

    metadata = _metadata(
        "ts_vector_state_local_density",
        "当前状态到前序窗口第 k 近邻距离的倒数（局部密度）。",
        ["f1", "f2", "f3", "f4", "window", "k"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, f4: pd.DataFrame,
        window: int = 60, k: int = 5, **_: Any,
    ) -> pd.DataFrame:
        arr = _local_density_series(
            f1.to_numpy(dtype=float), f2.to_numpy(dtype=float),
            f3.to_numpy(dtype=float), f4.to_numpy(dtype=float),
            window, k,
        )
        return frame_like(f1, arr)


@register_operator(
    name="ts_multivariate_matrix_profile_novelty",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_multivariate_matrix_profile_novelty",
    source="candle_state_space",
)
class TsMultivariateMatrixProfileNovelty(SeriesOperator):
    """矩阵轮廓新颖度：当前子序列与其最近历史子序列的 z 归一欧氏距离。

    ``x`` 为调用方传入的标量序列；当前尾随子序列长度 L，搜索历史带
    ``[t-history .. t-L]`` 内最近的前序匹配。距离大 → 当前模式新颖。P1。
    """

    metadata = _metadata(
        "ts_multivariate_matrix_profile_novelty",
        "当前子序列到最近历史子序列的 z 归一化距离（新颖度）。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        novelty, _age = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, novelty)


@register_operator(
    name="ts_matrix_profile_motif_age",
    category="candle_state_space",
    business_category="candle_state_space",
    canonical="ts_matrix_profile_motif_age",
    source="candle_state_space",
)
class TsMatrixProfileMotifAge(SeriesOperator):
    """矩阵轮廓模式年龄：最近历史匹配起点距今的归一化年龄。

    与新颖度共享同一矩阵轮廓核；``s*`` 为最近匹配的前序子序列起点，
    ``age = t - s*``，输出 ``age/history``（∈(0,1]）。P1。
    """

    metadata = _metadata(
        "ts_matrix_profile_motif_age",
        "当前子序列最近历史匹配的归一化年龄 age/history。",
        ["x", "window", "subsequence_length", "history"],
        unit="ratio",
        cost=8,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120,
        subsequence_length: int = 10, history: int = 80, **_: Any,
    ) -> pd.DataFrame:
        _novelty, age = _matrix_profile_series(
            x.to_numpy(dtype=float), window, subsequence_length, history,
        )
        return frame_like(x, age)


_NEW_CANONICALS = (
    "ts_vector_state_mahalanobis",
    "ts_vector_state_local_density",
    "ts_multivariate_matrix_profile_novelty",
    "ts_matrix_profile_motif_age",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
