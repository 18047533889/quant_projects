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
from cleaned_operators.candle_state_space import _matrix_profile_series
from cleaned_operators.registry import OperatorRegistry
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

    # R5-50: live extend mutator, never a frozenset reassignment.
    _surface.extend_research_only({name})
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


def _mp_stats(vals: np.ndarray, m: int, stat: str, history_window: int = 252) -> float:
    """Unified matrix-profile statistic over the shared kernel.

    The legacy implementation compressed NaN (``seg[np.isfinite(seg)]``), which
    bridged gaps, and used a hard-coded 200-row invisible tail (audit P0).  The
    shared ``_matrix_profile_series`` kernel keeps the time axis intact
    (contiguous-finite patterns only), enforces an exclusion zone and takes an
    explicit history window — there is no invisible tail.  ``discord`` and
    ``motif`` are the same min-distance quantity and are both superseded by the
    novelty canonical (``ts_matrix_profile_novelty``).

    ``history_window`` caps the lookback (round-7 audit): the kernel was called
    with ``history = window = n``, an implicit expanding-history operator where
    every row searches all of ``x[:t+1]`` — the factor then depends on the data
    start date and the cost grows O(n * history).  When ``history_window`` is
    finite the profile runs on the trailing window only.
    """
    v = np.asarray(vals, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    m = max(3, int(m))
    hist = int(history_window)
    if hist <= 0:
        hist = v.shape[0]
    n = v.shape[0]
    lo = max(0, n - hist)
    vw = v[lo:]
    nw = vw.shape[0]
    if nw < m + 3:
        return np.nan
    # Explicit band (no hard-coded 200): window == history == trailing band.
    novelty, _age, frequency, dispersion = _matrix_profile_series(
        vw, window=nw, subsequence_length=m, history=nw
    )
    last = nw - 1
    if stat in {"discord", "motif"}:
        # P1-93: ``motif`` is an ALIAS of ``discord`` — both return the trailing
        # subsequence's matrix-profile novelty (its minimum distance to any
        # historical subsequence).  The two names are intentionally the same
        # quantity; ``ts_matrix_profile_discord_score`` is the canonical and
        # ``ts_matrix_profile_motif_distance`` is kept as a registry
        # compatibility alias (its metadata documents the relationship).
        return float(novelty[last, 0])
    if stat == "recurrence":
        return float(frequency[last, 0]) if np.isfinite(frequency[last, 0]) else np.nan
    return float(dispersion[last, 0]) if np.isfinite(dispersion[last, 0]) else np.nan


_register("ts_matrix_profile_discord_score", "末尾子序列到最近历史子序列距离（离群度，novelty canonical，history_window 截断回溯）。", ["x", "m", "history_window"], "level",
           lambda x, m=20, history_window=252: _apply(x, lambda v: _mp_stats(v, int(m), "discord", int(history_window))))
# P1-93 / round-7: ``motif_distance`` is the same novelty value as
# ``discord_score``.  The canonical is ``ts_matrix_profile_discord_score``; this
# name is a registry compatibility alias — a single canonical, NOT a second
# independent research candidate that mining would double-search.
OperatorRegistry.register_compat_alias(
    "ts_matrix_profile_motif_distance",
    "ts_matrix_profile_discord_score",
    migration_reason="legacy name for the identical matrix-profile novelty distance",
    deprecated_since="2026-08",
    removal_version="1.0",
)
_register("ts_motif_recurrence_count", "相似历史模式出现次数（近邻计数，history_window 截断回溯）。", ["x", "m", "history_window"], "count",
           lambda x, m=20, history_window=252: _apply(x, lambda v: _mp_stats(v, int(m), "recurrence", int(history_window))))
