# -*- coding: utf-8 -*-
"""Dynamic Mode Decomposition primitives (2026-08-08 Gemini V2 round).

Hankel / delay-embedded DMD on a single series: the trailing window is
embedded into an ``L x K`` Hankel matrix (``L = dim`` rows, column stride
``delay``), and ``X = H[:, :-1]``, ``Y = H[:, 1:]`` approximate the linear
Koopman-style propagator ``Ã = U_r^T Y V_r Σ_r^{-1}`` from a rank-``r`` SVD of
``X``.  The eigenvalues of ``Ã`` give per-mode growth ``g = ln|λ|/Δt`` and
frequency ``f = arg(λ)/(2π·Δt)``; the *exact* DMD modes are
``Φ = Y V Σ^{-1} W`` (``W`` = eigenvectors of ``Ã``) and the amplitudes are the
pseudoinverse solution ``b = Φ† x_1``.  Mode energy is the finite-horizon sum
``E_j = Σ_{t=0}^{K-1} |b_j λ_j^t|²`` — never a closed-form geometric-series
with a denominator clamp (that formula went unstable when ``|λ| > 1``).

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

from cleaned_operators.base import ParamSpec, RelationalParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    union_research,
)

_EPS = 1e-12

# R5 P1-01/P1-02: ``rank``/``dim``/``delay`` are validated ints (a fractional
# value is rejected, never silently truncated).
_DMD_PARAM_SPECS = {
    "window": ParamSpec(dtype=int, min=10),
    "rank": ParamSpec(dtype=int, min=1),
    "dim": ParamSpec(dtype=int, min=2),
    "delay": ParamSpec(dtype=int, min=1),
    "top_k": ParamSpec(dtype=int, min=1),
}
# R6-157: ``keys(param_specs) ⊆ param_names`` is a registry invariant.  The
# dominant-growth/frequency canonicals take ``window/rank/dim/delay`` only —
# ``top_k`` belongs solely to ``ts_dmd_mode_concentration``.  A shared spec
# dict would declare a contract for a parameter the canonical does not have.
_DMD_BASE_SPEC = {k: _DMD_PARAM_SPECS[k] for k in ("window", "rank", "dim", "delay")}
_DMD_CONCENTRATION_SPEC = dict(_DMD_PARAM_SPECS)
# R6-200/201: feasibility relations for the DMD family — K = window-(dim-1)*delay
# is the embedding column count, and rank <= min(dim, K-1), K >= rank+2 are
# required for a valid SVD / propagator.  top_k <= rank stops silent clipping.
_DMD_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window - (dim - 1) * delay >= rank + 2",
        "DMD requires K = window-(dim-1)*delay >= rank+2 "
        "(window={window}, dim={dim}, delay={delay}, rank={rank})",
    ),
    RelationalParamSpec(
        "rank <= dim",
        "DMD requires rank <= dim (rank={rank}, dim={dim})",
    ),
]
_DMD_CONCENTRATION_RELATIONAL_SPECS = list(_DMD_RELATIONAL_SPECS) + [
    RelationalParamSpec(
        "top_k <= rank",
        "ts_dmd_mode_concentration requires top_k <= rank (top_k={top_k}, rank={rank})",
    )
]


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
    # R5 P1-02: an infeasible rank is REJECTED (fail-closed), never silently
    # clipped down — ``rank=4`` and ``rank=3`` must not compile to the same
    # factor, which the old ``min(rank, S.shape[0]-1)`` did.
    r = int(rank)
    if r < 1 or r > S.shape[0]:
        return None
    if S[0] <= _EPS or S[r - 1] <= _EPS * S[0]:
        return None
    cond = S[0] / S[r - 1]
    if cond > 1e12:
        return None
    Ur = U[:, :r]
    Vr = Vt[:r, :].T
    Sr_inv = np.diag(1.0 / S[:r])
    A_tilde = Ur.T @ Y @ Vr @ Sr_inv
    # P0-06 rewrite: keep the (eigval, eigvector) PAIRS together so the filtered
    # subset stays aligned (eigenvalues alone cannot be re-paired to modes).
    eig_vals, W = np.linalg.eig(A_tilde)
    finite = np.isfinite(eig_vals)
    if not finite.all():
        eig_vals = eig_vals[finite]
        W = W[:, finite]
    if eig_vals.size == 0:
        return None
    # exact DMD modes Phi = Y V Sigma^{-1} W  and  amplitudes b = Phi^dagger x_1.
    # (The old code used the SVD-coordinate projection ``Ur.T @ H[:, 0]`` as if
    # it were the mode amplitude — that is not the standard DMD amplitude and
    # mis-orders the dominant mode / energies.)
    Phi = Y @ (Vr / S[:r]) @ W          # (L, r) exact modes
    x1 = X[:, 0]
    b = np.linalg.pinv(Phi) @ x1        # (r,) mode amplitudes
    # finite-horizon energy E_j = sum_{t=0}^{K-1} |b_j lam_j^t|^2 — a direct
    # K-term sum (stable), never the closed-form-with-clamp.
    rho = np.abs(eig_vals) ** 2
    tgrid = np.arange(K, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        powers = rho[:, None] ** tgrid          # (r, K)
        e_sum = np.sum(powers, axis=1)
    for j in range(eig_vals.size):
        if not np.isfinite(e_sum[j]) and rho[j] > 1.0:
            # last-term-dominated stable form (avoids overflow of rho^K)
            e_sum[j] = np.exp(min((K - 1) * np.log(rho[j]), 700.0)) * (rho[j] / (rho[j] - 1.0))
    energies = (np.abs(b) ** 2) * e_sum
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
                # R6-202: for a real-valued time series the eigenvalues come in
                # conjugate pairs (λ, conj(λ)) with near-identical energies.  The
                # raw sign of arg(λ) can flip day-to-day from tie ordering even
                # though both signs describe the SAME oscillation.  Canonical
                # frequency uses |arg λ|/(2π) so the dominant frequency is stable
                # and unambiguous for a real series.
                val = float(abs(np.angle(lam0)) / (2.0 * np.pi))
            else:
                # R6-200: top_k > rank is rejected, never silently clipped —
                # rank=3 with top_k=3,4,10 must not compile to the same factor.
                tk = int(top_k)
                if tk > int(rank):
                    continue
                # R6-203: mode concentration must count a conjugate pair ONCE —
                # a real oscillation splits its energy between the +f and -f
                # modes, so top_k=1 on the split spectrum understates the mode's
                # concentration.  Merge conjugate pairs (equal |λ| and equal
                # |arg λ|, distinct values) and sum the pair energies before
                # ranking.
                eig = np.asarray(res["eig"])
                energy = np.asarray(res["energy"], dtype=float)
                merged_e: list[float] = []
                used = np.zeros(eig.shape[0], dtype=bool)
                for i in range(eig.shape[0]):
                    if used[i]:
                        continue
                    pair_e = float(energy[i])
                    for j in range(i + 1, eig.shape[0]):
                        if used[j]:
                            continue
                        is_conj = (
                            eig[i] != eig[j]
                            and abs(abs(eig[i]) - abs(eig[j])) < 1e-6
                            and abs(abs(np.angle(eig[i])) - abs(np.angle(eig[j]))) < 1e-6
                        )
                        if is_conj:
                            pair_e += float(energy[j])
                            used[j] = True
                            break
                    merged_e.append(pair_e)
                merged_e.sort(reverse=True)
                tk = min(tk, len(merged_e))
                total = float(sum(merged_e))
                if total <= _EPS:
                    continue
                val = float(sum(merged_e[:tk]) / total)
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
        "param_specs": _DMD_BASE_SPEC,
        "relational_specs": _DMD_RELATIONAL_SPECS,
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
        "param_specs": _DMD_BASE_SPEC,
        "relational_specs": _DMD_RELATIONAL_SPECS,
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
        "param_specs": _DMD_CONCENTRATION_SPEC,
        "relational_specs": _DMD_CONCENTRATION_RELATIONAL_SPECS,
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
            param_specs=spec.get("param_specs"),
            relational_specs=spec.get("relational_specs"),
        )
    union_research(*_SPECS.keys())


_register()
