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


def panel_or_scalar(value: Any, row: int) -> float:
    """Resolve a parameter that may be a scalar or a per-row panel value."""
    if isinstance(value, pd.DataFrame):
        v = value.iloc[row]
        if v.ndim == 0 or len(v) == 0:
            return np.nan
        return float(v.iloc[0])
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
