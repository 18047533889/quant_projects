# -*- coding: utf-8 -*-
"""Matrix-profile style sequence-anomaly operators (P3, experimental).

Brute-force z-normalised Euclidean distance between the trailing subsequence
and historical subsequences.  Expensive (O(n^2) in the window) and limited to
short windows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.ts_model._rolling_core import frame_like, metadata

_CANONICALS: list[str] = []


def _register(name: str, description: str, params: list[str], unit: str, fn, cost: int = 9):
    @register_operator(
        name=name,
        category="time_series_regression",
        business_category="time_series_regression",
        canonical=name,
        source="ts_model.sequence_anomaly",
        backend="pandas_numpy",
        status="experimental",
    )
    class _AnomalyOp(SeriesOperator):
        metadata = metadata(name, description, params, unit=unit, cost=cost)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {name}
    )
    return _AnomalyOp


def _apply(x: pd.DataFrame, fn) -> pd.DataFrame:
    xv = x.to_numpy(dtype=float)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            out[row, col] = fn(xv[: row + 1, col])
    return frame_like(x, out)


def _zscore(vec: np.ndarray) -> np.ndarray:
    sd = float(np.std(vec))
    if sd <= 1e-12:
        return vec * 0.0
    return (vec - np.mean(vec)) / sd


def _mp_stats(vals: np.ndarray, m: int, stat: str) -> float:
    seg = vals[-200:]
    finite = seg[np.isfinite(seg)]
    m = max(3, int(m))
    n = len(finite)
    if n < m + 3:
        return np.nan
    last = _zscore(finite[n - m :])
    distances: list[float] = []
    for i in range(0, n - m):
        cand = _zscore(finite[i : i + m])
        d = float(np.sqrt(np.sum((last - cand) ** 2) / m))
        distances.append(d)
    if len(distances) == 0:
        return np.nan
    if stat == "discord":
        return float(min(distances))
    if stat == "motif":
        return float(min(distances))
    if stat == "recurrence":
        thr = float(np.median(distances)) * 0.5
        return float(sum(d < thr for d in distances))
    return float(np.mean(distances))


_register("ts_matrix_profile_discord_score", "末尾子序列到最近历史子序列距离（离群度）。", ["x", "m"], "level",
           lambda x, m=20: _apply(x, lambda v: _mp_stats(v, int(m), "discord")))
_register("ts_matrix_profile_motif_distance", "末尾子序列到最相似历史模式的距离。", ["x", "m"], "level",
           lambda x, m=20: _apply(x, lambda v: _mp_stats(v, int(m), "motif")))
_register("ts_motif_recurrence_count", "相似历史模式出现次数（近邻计数）。", ["x", "m"], "count",
           lambda x, m=20: _apply(x, lambda v: _mp_stats(v, int(m), "recurrence")))
