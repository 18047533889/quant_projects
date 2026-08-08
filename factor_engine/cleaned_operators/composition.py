# -*- coding: utf-8 -*-
"""Compositional Data (CoDa / Aitchison) primitives (2026-08-08 Gemini V2 round).

A *composition* is a vector of positive parts that sum to (or describe) a
whole.  The operators operate on aligned panels (date x instrument), one part
per panel, and are all elementwise in the (date, instrument) cell — they never
touch the time axis, so PIT is trivially preserved.

* ``composition_clr_component``      — the centred log-ratio of one component:
  ``clr_i = ln x_i - mean(ln x)`` (P0).
* ``composition_aitchison_distance`` — ``‖clr(A) - clr(B)‖₂`` between two
  same-dimension compositions (P0).
* ``composition_ilr_balance``        — the normalised isometric balance between
  two groups of components ``√(rs/(r+s))·ln(gmean(A)/gmean(B))`` (P0).
* ``composition_entropy``            — Shannon entropy of the closed
  composition ``H = -Σ p_i ln p_i`` (P1).
* ``composition_js_divergence``      — Jensen-Shannon divergence between two
  closed compositions (P1).

Contract:

* ``zero_policy`` is fixed to ``"reject"`` — a component that is missing (NaN)
  or non-positive (a structural zero) invalidates the whole composition cell
  and fails closed.  No arbitrary ``1e-6`` imputation is ever applied (a fixed
  additive constant would be meaningless across money magnitudes), and no
  negative flow is silently fed into a log-ratio.
* At least three components are required (the family's 3-8 component surface).
  Aitchison distance / ILR balance / JS divergence take two same-dimension
  compositions (up to 6 components each side).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.gemini_v2_common import (
    _align,
    frame_like,
    register_dual,
    union_extended,
)

_EPS = 1e-12


def _check_zero_policy(zero_policy: str) -> None:
    if str(zero_policy) != "reject":
        raise ValueError("zero_policy must be 'reject' (structural zeros invalidate the cell)")


def _parts(*frames: pd.DataFrame | None) -> list[pd.DataFrame]:
    return [f for f in frames if f is not None]


def _comp_logs(frames: Sequence[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Stacked logs of aligned parts and the all-finite cell mask."""
    aligned = _align(*frames)
    arrs = [a.to_numpy(dtype=float) for a in aligned]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(a) for a in arrs])  # (P, R, C) — non-positive -> -inf/nan
    valid = np.all(np.isfinite(logs), axis=0)
    return logs, valid


def _close(logs: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Closure of parts [i0, i1) -> probability vector (P, R, C)."""
    sub = logs[i0:i1]
    w = np.exp(sub - sub.max(axis=0, keepdims=True))
    return w / w.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _composition_clr_component(
    target: pd.DataFrame,
    x1: pd.DataFrame | None = None,
    x2: pd.DataFrame | None = None,
    x3: pd.DataFrame | None = None,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    x7: pd.DataFrame | None = None,
    zero_policy: str = "reject",
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    parts = _parts(target, x1, x2, x3, x4, x5, x6, x7)
    if len(parts) < 3:
        raise ValueError("composition_clr_component requires >= 3 components")
    logs, valid = _comp_logs(parts)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    clr = logs[0] - logs.mean(axis=0)
    out[valid] = clr[valid]
    return frame_like(target, out)


def _composition_entropy(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    x7: pd.DataFrame | None = None,
    x8: pd.DataFrame | None = None,
    zero_policy: str = "reject",
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    parts = _parts(x1, x2, x3, x4, x5, x6, x7, x8)
    if len(parts) < 3:
        raise ValueError("composition_entropy requires >= 3 components")
    logs, valid = _comp_logs(parts)
    p = _close(logs, 0, len(parts))
    ent = -np.sum(p * np.log(p + _EPS), axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = ent[valid]
    return frame_like(x1, out)


def _composition_aitchison_distance(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        raise ValueError("composition_aitchison_distance requires equal-size compositions (2..6 parts each)")
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    la = logs[:na]
    lb = logs[na:]
    clr_a = la - la.mean(axis=0, keepdims=True)
    clr_b = lb - lb.mean(axis=0, keepdims=True)
    dist = np.sqrt(np.sum((clr_a - clr_b) ** 2, axis=0))
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = dist[valid]
    return frame_like(x1, out)


def _composition_ilr_balance(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("composition_ilr_balance requires >= 1 component per side")
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    gmean_a = np.exp(logs[:na].mean(axis=0))
    gmean_b = np.exp(logs[na:].mean(axis=0))
    r = float(na)
    s = float(len(b))
    balance = np.sqrt(r * s / (r + s)) * np.log(gmean_a / gmean_b + _EPS)
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = balance[valid]
    return frame_like(x1, out)


def _composition_js_divergence(
    x1: pd.DataFrame,
    x2: pd.DataFrame,
    x3: pd.DataFrame,
    y1: pd.DataFrame,
    y2: pd.DataFrame,
    y3: pd.DataFrame,
    x4: pd.DataFrame | None = None,
    x5: pd.DataFrame | None = None,
    x6: pd.DataFrame | None = None,
    y4: pd.DataFrame | None = None,
    y5: pd.DataFrame | None = None,
    y6: pd.DataFrame | None = None,
    zero_policy: str = "reject",
) -> pd.DataFrame:
    _check_zero_policy(zero_policy)
    a = _parts(x1, x2, x3, x4, x5, x6)
    b = _parts(y1, y2, y3, y4, y5, y6)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        raise ValueError("composition_js_divergence requires equal-size compositions (2..6 parts each)")
    logs, valid = _comp_logs([*a, *b])
    na = len(a)
    p = _close(logs, 0, na)
    q = _close(logs, na, na + len(b))
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + _EPS) - np.log(m + _EPS)), axis=0) + 0.5 * np.sum(
        q * (np.log(q + _EPS) - np.log(m + _EPS)), axis=0
    )
    out = np.full(logs.shape[1:], np.nan, dtype=float)
    out[valid] = js[valid]
    return frame_like(x1, out)


_SPECS: dict[str, dict[str, Any]] = {
    "composition_clr_component": {
        "fn": _composition_clr_component,
        "params": ["target", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "zero_policy"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "ratio",
        "cost": 2,
        "tags_extra": ["fundamental"],
        "output_unit": "ratio",
    },
    "composition_entropy": {
        "fn": _composition_entropy,
        "params": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "zero_policy"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "entropy",
        "cost": 2,
        "tags_extra": ["fundamental"],
        "output_unit": "entropy",
    },
    "composition_aitchison_distance": {
        "fn": _composition_aitchison_distance,
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6", "zero_policy"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "ratio",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "ratio",
    },
    "composition_ilr_balance": {
        "fn": _composition_ilr_balance,
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6", "zero_policy"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "level",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "level",
    },
    "composition_js_divergence": {
        "fn": _composition_js_divergence,
        "params": ["x1", "x2", "x3", "y1", "y2", "y3", "x4", "x5", "x6", "y4", "y5", "y6", "zero_policy"],
        "category": "compositional_data",
        "domain": "composition",
        "unit": "entropy",
        "cost": 3,
        "tags_extra": ["fundamental"],
        "output_unit": "entropy",
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="composition",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_extended(*_SPECS.keys())


_register()
