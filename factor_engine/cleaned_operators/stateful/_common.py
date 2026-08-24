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

from factor_engine.cleaned_operators.base import OperatorMetadata
from factor_engine.cleaned_operators.rolling_pack import frame_like

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


def assert_condition_bool(panel: pd.DataFrame, name: str = "condition") -> None:
    """R11 #144: a condition input must be a ConditionBool.

    Accepted values: {0, 1} (or boolean True/False); NaN = missing/unknown.  Any
    other finite numeric value (e.g. 5.0, -0.03) is neither a probability nor a
    boolean — silently treating it as "truthy" is a hidden semantic.  Fail the
    call (raise) instead of guessing.

    Mirrors ``cleaned_operators.common.daily_panel._assert_condition_bool``.
    """
    cv = panel.to_numpy()
    finite = np.isfinite(cv)
    bad = finite & (cv != 0.0) & (cv != 1.0)
    if np.any(bad):
        raise ValueError(
            f"{name} must be a ConditionBool (values in {{0, 1}} with NaN as "
            f"missing); found {int(bad.sum())} finite value(s) outside {{0, 1}}"
        )


def truth_mask(panel: pd.DataFrame) -> np.ndarray:
    """ConditionBool truth: finite values equal to exactly 1.

    ``1`` = true event, ``0`` = false, ``NaN`` = unknown.  Unknown (NaN) cells
    are masked False here — a boolean mask cannot carry "unknown", so callers
    that need to distinguish unknown from false must handle NaN separately per
    the operator's documented missing policy (censor to NaN / keep prior state).
    """
    v = panel.to_numpy()
    return np.isfinite(v) & (v == 1)


def finite_panel(panel: pd.DataFrame) -> np.ndarray:
    return panel.to_numpy()


def register_stateful_surface(canonicals: list[str]) -> None:
    """Append stateful-pack canonicals to the reviewed extended surface."""
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(canonicals))
