# -*- coding: utf-8 -*-
"""Shared helpers for the 2026-08 stateful rule/episode/rotation pack.

Every operator in this package is a forward per-column recursion or a
trailing-window / cross-sectional transform that never looks past the current
row (prefix-causal).  Missing values follow ``missing_policy="break"``: a NaN
input emits NaN at that row and re-baselines the internal recursion state, so a
missing observation is never treated as False (see ``ts_time_since_change``).
Degenerate windows return NaN rather than an invented number; all divisions use
a safe epsilon.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    category: str = "time_series_state",
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def panel_or_scalar(value: Any, row: int, col: int) -> float:
    """Resolve a parameter that may be a scalar or a per-cell panel value.

    A DataFrame parameter is indexed cell-wise ``[row, col]`` so each stock
    gets its *own* threshold/limit on every day.  The legacy behaviour (first
    column broadcast to every stock) is a cross-sectional bug: e.g.
    ``state_slew_limit(x, atr_panel)`` silently used stock ``A``'s ATR for all
    stocks.  A scalar still broadcasts.  Missing cells return NaN (the caller
    treats NaN like a broken input).
    """
    if isinstance(value, pd.DataFrame):
        if row < 0 or row >= value.shape[0]:
            return np.nan
        v = value.iloc[row, col]
        return float(v)
    if isinstance(value, pd.Series):
        if col < 0 or col >= value.shape[0]:
            return np.nan
        return float(value.iloc[col])
    return float(value)


def truth_mask(panel: pd.DataFrame) -> np.ndarray:
    """Finite, non-zero truth (condition uses finite AND != 0)."""
    v = panel.to_numpy()
    return np.isfinite(v) & (v != 0)


def finite_panel(panel: pd.DataFrame) -> np.ndarray:
    return panel.to_numpy()


def register_stateful_surface(canonicals: list[str]) -> None:
    """Append stateful-pack canonicals to the reviewed extended surface."""
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(canonicals)
    )
