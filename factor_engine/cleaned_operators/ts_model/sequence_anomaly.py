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

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec, RelationalParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model._rolling_core import frame_like, metadata

_CANONICALS: list[str] = []

# Model-audit Phase 4 (search-space hygiene): ``m`` is the matrix-profile
# subsequence length — an estimator-resolution grid knob, never a full-search
# dimension (M-115/M-162/M-170).  ``history_window`` caps the lookback and is
# the alpha horizon (HORIZON, searched).
_MP_PARAM_SPECS: dict[str, ParamSpec] = {
    "m": ParamSpec(dtype=int, min=3, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False),
    "history_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON, searchable=True),
}
# M-11xx: matrix-profile relational feasibility for the sequence-anomaly family:
# the subsequence must be strictly shorter than the history band, and the band
# must clear the exclusion zone (``m + m//4``) so a non-empty historical search
# band exists (otherwise the trailing row is guaranteed all-NaN).  Rejected at
# binding, never run-then-NaN.
_MP_RELATIONAL_SPECS: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "m < history_window",
        "ts_matrix_profile_discord_score / ts_motif_recurrence_count require m < "
        "history_window (m={m}, history_window={history_window})",
    ),
    RelationalParamSpec(
        "history_window >= m + m // 4",
        "ts_matrix_profile_discord_score / ts_motif_recurrence_count require "
        "history_window >= m + m//4 (exclusion-zone feasibility; history_window="
        "{history_window}, m={m})",
    ),
]


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
        metadata = metadata(name, description, params, unit=unit, cost=cost,
                            param_specs=_MP_PARAM_SPECS)
        metadata.relational_specs = list(_MP_RELATIONAL_SPECS)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _CANONICALS.append(name)
    import factor_engine.cleaned_operators.operator_surface as _surface

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

    M-101: ``m`` (subsequence length) is validated strictly as an integer >= 3
    (``strict_int``, see ``cleaned_operators.closure.strict_scalar``).  ``m < 3``,
    a fractional value, NaN/Inf or a bool now RAISE instead of being clamped or
    truncated — the legacy ``max(3, int(m))`` silently coerced ``m=2`` to ``3``
    and ``m=20.7`` to ``20``, manufacturing factors that never matched any
    declared window.

    M-102: ``history_window`` is validated strictly as a finite positive
    integer.  The legacy ``history_window <= 0`` silently expanded to the full
    history (an implicit expanding-history operator whose output depends on the
    data start date); that is now a contract error and raises.  ``history_window``
    caps the lookback (round-7 audit): the profile runs on the trailing window
    only — every row searches its trailing ``history_window`` band, so the cost
    stays O(n * history_window) and the output is invariant to history before the
    band.
    """
    v = np.asarray(vals, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    m = strict_int(m, "m", lower=3)
    hist = strict_int(history_window, "history_window", lower=1)
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


_register("ts_matrix_profile_discord_score", "末尾子序列到最近历史子序列距离（离群度，novelty canonical，history_window 截断回溯）。m 必须为 >=3 整数，history_window 必须为正整数，非法值报错不静默截断（M-101/M-102）。", ["x", "m", "history_window"], "level",
           lambda x, m=20, history_window=252: _apply(x, lambda v: _mp_stats(v, m, "discord", history_window)))
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
_register("ts_motif_recurrence_count", "相似历史模式出现次数（近邻计数，history_window 截断回溯）。m 必须为 >=3 整数，history_window 必须为正整数，非法值报错不静默截断（M-101/M-102）。", ["x", "m", "history_window"], "count",
           lambda x, m=20, history_window=252: _apply(x, lambda v: _mp_stats(v, m, "recurrence", history_window)))
