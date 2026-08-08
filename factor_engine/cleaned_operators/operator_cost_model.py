# -*- coding: utf-8 -*-
"""Parameter-aware operator cost model (R6 §32).

The static ``cost:N`` metadata tag labels a nominal complexity band, but the
REAL cost of an operator depends on its parameters: HVG motif entropy is
O(window³), Qn / Hodges-Lehmann O(window²), bicoherence O(window·F²), DMD
O(window·d² + d³), kernel Granger O(window³).  A search budget that treats
``window=20`` and ``window=500`` as the same cost will let a couple of heavy
operators eat the whole expression budget.

This module provides ``runtime_cost(canonical, params)`` and
``memory_cost(canonical, params)`` — deterministic, parameter-aware complexity
estimates — plus a registry of complexity kernels keyed by operator family.
It is a governance *primitive*: search / AlphaProbe budget allocation should
query these instead of the static ``cost`` tag; wiring into the search loop is
tracked separately.

Deterministic (no random sampling / no timing), pure functions of the bound
parameters.
"""
from __future__ import annotations

import math
from typing import Any, Callable

# Reference nominal cost for ``window=120`` on a mid-cost daily operator; used
# to normalise the absolute figures into multiples of one standard rolling op.
_REF_WINDOW = 120.0
_REF_COST = 120.0  # one pass over a 120-row window, per column


def _poly_cubic(w: float, _p: dict[str, Any]) -> float:
    return w ** 3


def _poly_quad(w: float, _p: dict[str, Any]) -> float:
    return w ** 2


def _poly_linear(w: float, _p: dict[str, Any]) -> float:
    return w


def _bicoherence(w: float, p: dict[str, Any]) -> float:
    ns = max(2, int(p.get("n_segments", 4)))
    seg = w / ns
    f = min(32.0, seg / 2.0)
    return w * f * f


def _dmd(w: float, p: dict[str, Any]) -> float:
    d = float(p.get("dim", 4))
    return w * d * d + d ** 3


def _hvg_motif(w: float, _p: dict[str, Any]) -> float:
    return w ** 3


def _pairwise(w: float, _p: dict[str, Any]) -> float:
    # Qn / Hodges-Lehmann / BDS / recurrence build all O(n²) pairs per row.
    return w * w


def _rqa(w: float, _p: dict[str, Any]) -> float:
    # M x M recurrence matrix built once per window.
    return w * w


def _topology(w: float, _p: dict[str, Any]) -> float:
    # Rips complex over a capped point cloud (<=12 pts) — small but super-linear
    # in the window because the cloud is built from the window.
    return w * 12 * 12


def _matrix_profile(w: float, p: dict[str, Any]) -> float:
    L = float(p.get("subsequence_length", 10))
    h = float(p.get("history", 80))
    return h * L * math.log(max(L, 2.0))


# canonical prefix -> complexity kernel.  Matching is longest-prefix-first so a
# specific family (ts_hvg_motif_entropy) beats its generic prefix (ts_hvg_).
_COMPLEXITY_KERNELS: list[tuple[str, Callable[[float, dict[str, Any]], float], str]] = [
    ("ts_hvg_motif_entropy", _hvg_motif, "O(W^3)"),
    ("ts_hvg_", _pairwise, "O(W^2) pairwise graph"),
    ("ts_qn_scale", _pairwise, "O(W^2) pairwise diffs"),
    ("ts_hodges_lehmann_location", _pairwise, "O(W^2) pairwise midpoints"),
    ("ts_bds_statistic", _pairwise, "O(W^2) correlation integrals"),
    ("ts_recurrence_", _rqa, "O(M^2) recurrence matrix"),
    ("ts_bicoherence_max", _bicoherence, "O(W*F^2)"),
    ("ts_dmd_", _dmd, "O(W*d^2 + d^3)"),
    ("ts_kernel_granger_score", _poly_cubic, "O(W^3) kernel solve"),
    ("ts_residualized_hsic", _poly_cubic, "O(W^3) kernel smoother"),
    ("ts_persistence_", _topology, "Rips over capped cloud"),
    ("ts_matrix_profile", _matrix_profile, "O(history*L*log L)"),
    ("ts_signature_mahalanobis_anomaly", _pairwise, "O(W) sigs x covariance"),
    ("ts_multivariate_matrix_profile", _matrix_profile, "O(history*L*log L)"),
]

# Canonicals explicitly declared research-only (never auto-search) — their cost
# is out of the search budget by construction.
_RESEARCH_ONLY_PREFIXES = ("ts_hvg_motif_entropy", "ts_dmd_", "ts_persistence_",)


def _bind_window(params: dict[str, Any]) -> float:
    w = params.get("window") or params.get("recent_window") or _REF_WINDOW
    try:
        return max(1.0, float(w))
    except (TypeError, ValueError):
        return _REF_WINDOW


def runtime_cost(canonical: str, params: dict[str, Any] | None = None) -> float:
    """Estimated per-column runtime cost in units of one 120-row rolling op.

    ``params`` are the bound parameter values (``window`` / ``dim`` / ``lag`` /
    …).  Returns a dimensionless multiple of the reference cost so a search
    budget can compare ``window=20`` and ``window=500`` fairly.
    """
    params = params or {}
    w = _bind_window(params)
    for prefix, kernel, _label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            return max(1.0, kernel(w, params) / _REF_COST)
    # default: linear-in-window (the standard rolling family)
    return max(1.0, w / _REF_WINDOW)


def memory_cost(canonical: str, params: dict[str, Any] | None = None) -> float:
    """Estimated peak per-column working set in rows²-equivalents.

    Operators that materialise an n×n matrix (recurrence, RQA, covariance,
    pairwise diffs) have O(window²) memory; everything else is O(window).
    """
    params = params or {}
    w = _bind_window(params)
    for prefix, kernel, _label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            # kernels returning w*w (pairwise / rqa / dmd / bicoherence) carry
            # an O(W²) working set; linear/cubic kernels are O(W).
            if prefix in ("ts_hvg_", "ts_qn_scale", "ts_hodges_lehmann_location",
                          "ts_bds_statistic", "ts_recurrence_", "ts_dmd_",
                          "ts_bicoherence_max", "ts_signature_mahalanobis_anomaly"):
                return max(1.0, (w * w) / (_REF_WINDOW * _REF_WINDOW))
            return max(1.0, w / _REF_WINDOW)
    return max(1.0, w / _REF_WINDOW)


def is_research_only(canonical: str) -> bool:
    """Whether the canonical is excluded from default search by construction."""
    return any(canonical.startswith(p) for p in _RESEARCH_ONLY_PREFIXES)


def complexity_label(canonical: str) -> str:
    """Human-readable asymptotic label for a canonical (for catalog/docs)."""
    for prefix, _kernel, label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            return label
    return "O(W)"


__all__ = [
    "runtime_cost",
    "memory_cost",
    "is_research_only",
    "complexity_label",
    "_COMPLEXITY_KERNELS",
]
