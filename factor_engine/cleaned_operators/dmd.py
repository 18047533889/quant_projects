# -*- coding: utf-8 -*-
"""Dynamic Mode Decomposition primitives (2026-08-08 Gemini V2 round).

Hankel / delay-embedded DMD on a single series: the trailing window is
embedded into an ``L x K`` Hankel matrix (``L = dim`` rows, column stride
``delay``), and ``X = H[:, :-1]``, ``Y = H[:, 1:]`` approximate the linear
Koopman-style propagator ``Ã = U_r^T Y V_r Σ_r^{-1}`` from a rank-``r`` SVD of
``X``.  The eigenvalues of ``Ã`` give per-mode growth ``g = ln|λ|/Δt`` and
frequency ``f = arg(λ)/(2π·Δt)``; mode energy ``E_j = |b_j|²·Σ_t |λ_j|^{2t}``
(with ``b = U_r^T H[:, 0]``) selects the dominant mode.

All three operators are Research-surface: SVD/eigen decomposition is
numerically delicate, so they are excluded from the default mining grammar
and fail closed whenever the embedding, the singular-value rank, or the
condition number is degenerate.

* ``ts_dmd_dominant_growth_rate``
* ``ts_dmd_dominant_frequency``  (NaN when the dominant mode has no
  imaginary component)
* ``ts_dmd_mode_concentration``  (top-``top_k`` mode energy share)

Deterministic (no randomized SVD / subsampling), strict-PIT, NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_research,
)

_EPS = 1e-12


def _hankel_dmd(v: np.ndarray, rank: int, dim: int, delay: int) -> dict[str, Any] | None:
    n = v.shape[0]
    L = max(2, int(dim))
    dl = max(1, int(delay))
    K = n - (L - 1) * dl
    if K < rank + 2 or K < 4:
        return None
    H = np.stack([v[i : i + K] for i in range(0, (L - 1) * dl + 1, dl)], axis=0)
    X = H[:, :-1]
    Y = H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(int(rank), S.shape[0] - 1)
    if r < 1:
        return None
    if S[0] <= _EPS or S[r - 1] <= _EPS * S[0]:
        return None
    cond = S[0] / S[r - 1]
    if cond > 1e12:
        return None
    Ur = U[:, :r]
    Sr_inv = np.diag(1.0 / S[:r])
    Vr = Vt[:r, :].T
    A_tilde = Ur.T @ Y @ Vr @ Sr_inv
    eig_vals = np.linalg.eigvals(A_tilde)
    eig_vals = eig_vals[np.isfinite(eig_vals)]
    if eig_vals.size == 0:
        return None
    b = Ur.T @ H[:, 0]
    # per-mode reconstruction energy E_j = |b_j|^2 * sum_{t=0}^{K-1} |lam_j|^{2t}
    Kf = float(K)
    energies = np.abs(b) ** 2 * np.where(
        np.abs(np.abs(eig_vals) - 1.0) < _EPS,
        Kf,
        (1.0 - np.abs(eig_vals) ** (2 * Kf)) / np.maximum(1.0 - np.abs(eig_vals) ** 2, _EPS),
    )
    order = np.argsort(-energies)
    return {
        "eig": eig_vals[order],
        "energy": energies[order],
    }


def _dmd_series(x2d: np.ndarray, window: int, rank: int, dim: int, delay: int, which: str, top_k: int = 2) -> np.ndarray:
    rows, cols = x2d.shape
    w = max(10, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < int(dim) * int(delay) + int(rank) + 3:
                continue
            res = _hankel_dmd(v, int(rank), int(dim), int(delay))
            if res is None:
                continue
            lam0 = res["eig"][0]
            if which == "growth":
                val = float(np.log(max(abs(lam0), _EPS)))
            elif which == "frequency":
                if abs(lam0.imag) < 1e-9:
                    continue
                val = float(np.angle(lam0) / (2.0 * np.pi))
            else:
                n_modes = min(int(top_k), res["energy"].shape[0])
                total = float(res["energy"].sum())
                if total <= _EPS:
                    continue
                val = float(res["energy"][:n_modes].sum() / total)
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_dmd_dominant_growth_rate(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_dominant_growth_rate requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "growth")
    return frame_like(x, out)


def _ts_dmd_dominant_frequency(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_dominant_frequency requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "frequency")
    return frame_like(x, out)


def _ts_dmd_mode_concentration(x: pd.DataFrame, window: int = 120, rank: int = 4, dim: int = 4, delay: int = 1, top_k: int = 2) -> pd.DataFrame:
    if int(window) < 10:
        raise ValueError("ts_dmd_mode_concentration requires window >= 10")
    out = _dmd_series(x.to_numpy(dtype=float), int(window), int(rank), int(dim), int(delay), "concentration", int(top_k))
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_dmd_dominant_growth_rate": {
        "fn": _ts_dmd_dominant_growth_rate,
        "params": ["x", "window", "rank", "dim", "delay"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "level",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "level",
    },
    "ts_dmd_dominant_frequency": {
        "fn": _ts_dmd_dominant_frequency,
        "params": ["x", "window", "rank", "dim", "delay"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "cycles",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "cycles",
    },
    "ts_dmd_mode_concentration": {
        "fn": _ts_dmd_mode_concentration,
        "params": ["x", "window", "rank", "dim", "delay", "top_k"],
        "category": "dynamic_mode",
        "domain": "dynamical_systems",
        "unit": "ratio",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "ratio",
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
            source="dmd",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_research(*_SPECS.keys())


_register()
