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
    ("ts_bicoherence_", _bicoherence, "O(W*F^2)"),
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
    w = (
        params.get("window")
        or params.get("recent_window")
        or params.get("history_window")
        or params.get("path_window")
        or _REF_WINDOW
    )
    try:
        return max(1.0, float(w))
    except (TypeError, ValueError):
        return _REF_WINDOW


# R9-P1-047: explicit cost contracts for the heavy families.  An operator may
# declare ``metadata.cost_model`` (preferred) OR be listed here — both are the
# operator's OWN ``cost_model(params, shape) -> (runtime, memory)`` contract, so
# the cost of these families is no longer guessed from a name prefix.  The
# prefix table below remains only as the fallback for undeclared operators.
def _dmd_cost_contract(params: dict[str, Any], _shape: tuple[int, int] | None) -> tuple[float, float]:
    w = _bind_window(params)
    d = float(params.get("dim", 4))
    rt = w * d * d + d ** 3
    mem = w * w
    return max(1.0, rt / _REF_COST), max(1.0, mem / (_REF_WINDOW * _REF_WINDOW))


def _pairwise_cost_contract(params: dict[str, Any], _shape: tuple[int, int] | None) -> tuple[float, float]:
    w = _bind_window(params)
    rt = w * w
    return max(1.0, rt / _REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW))


def _cubic_cost_contract(params: dict[str, Any], _shape: tuple[int, int] | None) -> tuple[float, float]:
    w = _bind_window(params)
    rt = w ** 3
    return max(1.0, rt / _REF_COST), max(1.0, w / _REF_WINDOW)


def _bicoherence_cost_contract(params: dict[str, Any], _shape: tuple[int, int] | None) -> tuple[float, float]:
    w = _bind_window(params)
    ns = max(2, int(params.get("n_segments", 4)))
    f = min(32.0, (w / ns) / 2.0)
    rt = w * f * f
    return max(1.0, rt / _REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW))


_EXPLICIT_COST_CONTRACTS: dict[str, Callable[[dict[str, Any], tuple[int, int] | None], tuple[float, float]]] = {
    "ts_dmd_dominant_growth_rate": _dmd_cost_contract,
    "ts_dmd_dominant_frequency": _dmd_cost_contract,
    "ts_dmd_mode_concentration": _dmd_cost_contract,
    "ts_hvg_motif_entropy": _cubic_cost_contract,
    "ts_qn_scale": _pairwise_cost_contract,
    "ts_hodges_lehmann_location": _pairwise_cost_contract,
    "ts_bds_statistic": _pairwise_cost_contract,
    "ts_kernel_granger_score": _cubic_cost_contract,
    "ts_residualized_hsic": _cubic_cost_contract,
    "ts_bicoherence_top_decile_mean": _bicoherence_cost_contract,
    "ts_bicoherence_max": _bicoherence_cost_contract,
    "ts_recurrence_quantification_analysis": _pairwise_cost_contract,
    "ts_persistence_entropy_h0": _cubic_cost_contract,
    "ts_persistence_entropy_h1": _cubic_cost_contract,
}


def _declared_cost_model(canonical: str) -> Callable[[dict[str, Any], tuple[int, int] | None], tuple[float, float]] | None:
    """The operator's declared ``metadata.cost_model`` if it has one.

    R9-P1-047: an operator that declares its own ``cost_model(params, shape) ->
    (runtime, memory)`` is trusted verbatim over the prefix-name complexity
    table.  Registry lookups before ``load_all`` return ``None`` and fall back
    to the prefix model — the declared contract simply wins when present.
    """
    if canonical in _EXPLICIT_COST_CONTRACTS:
        return _EXPLICIT_COST_CONTRACTS[canonical]
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        cm = getattr(getattr(op, "metadata", None), "cost_model", None)
        return cm if callable(cm) else None
    except Exception:  # pragma: no cover - registry not loaded / import order
        return None


def runtime_cost(canonical: str, params: dict[str, Any] | None = None, shape: tuple[int, int] | None = None) -> float:
    """Estimated per-column runtime cost in units of one 120-row rolling op.

    ``params`` are the bound parameter values (``window`` / ``dim`` / ``lag`` /
    …).  Returns a dimensionless multiple of the reference cost so a search
    budget can compare ``window=20`` and ``window=500`` fairly.  A declared
    ``metadata.cost_model`` (R9-P1-047) takes precedence over the prefix table.
    """
    params = params or {}
    declared = _declared_cost_model(canonical)
    if declared is not None:
        rt, _mem = declared(params, shape)
        return max(1.0, float(rt))
    w = _bind_window(params)
    for prefix, kernel, _label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            return max(1.0, kernel(w, params) / _REF_COST)
    # default: linear-in-window (the standard rolling family)
    return max(1.0, w / _REF_WINDOW)


def memory_cost(canonical: str, params: dict[str, Any] | None = None, shape: tuple[int, int] | None = None) -> float:
    """Estimated peak per-column working set in rows²-equivalents.

    Operators that materialise an n×n matrix (recurrence, RQA, covariance,
    pairwise diffs) have O(window²) memory; everything else is O(window).  A
    declared ``metadata.cost_model`` (R9-P1-047) takes precedence.
    """
    params = params or {}
    declared = _declared_cost_model(canonical)
    if declared is not None:
        _rt, mem = declared(params, shape)
        return max(1.0, float(mem))
    w = _bind_window(params)
    for prefix, kernel, _label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            # kernels returning w*w (pairwise / rqa / dmd / bicoherence) carry
            # an O(W²) working set; linear/cubic kernels are O(W).
            if prefix in ("ts_hvg_", "ts_qn_scale", "ts_hodges_lehmann_location",
                          "ts_bds_statistic", "ts_recurrence_", "ts_dmd_",
                          "ts_bicoherence_", "ts_signature_mahalanobis_anomaly"):
                return max(1.0, (w * w) / (_REF_WINDOW * _REF_WINDOW))
            return max(1.0, w / _REF_WINDOW)
    return max(1.0, w / _REF_WINDOW)


def has_declared_cost_contract(canonical: str) -> bool:
    """Whether the canonical declares its own ``cost_model`` (R9-P1-047)."""
    return _declared_cost_model(canonical) is not None


def search_budget_gate(
    canonical: str,
    params: dict[str, Any] | None = None,
    *,
    runtime_budget: float | None = None,
    memory_budget: float | None = None,
    shape: tuple[int, int] | None = None,
) -> bool:
    """R9-P1-046: the single gate an AlphaProbe / AlphaMiner-style generator
    calls BEFORE allocating search budget to ``(canonical, params)``.

    Returns ``True`` iff the parameterised operator fits the given budgets
    (``None`` budget = unbounded on that axis).  A generator that calls this for
    every candidate gets parameter-aware accounting (window=20 vs window=500,
    dim/rank/k/bins, panel/cross-sectional shape) instead of the static
    ``cost:N`` tag.
    """
    params = params or {}
    rt = runtime_cost(canonical, params, shape=shape)
    if runtime_budget is not None and rt > float(runtime_budget):
        return False
    mem = memory_cost(canonical, params, shape=shape)
    if memory_budget is not None and mem > float(memory_budget):
        return False
    return True


#: Default mining treats an operator as "heavy" above this multiple of one
#: reference rolling op.
_DEFAULT_HEAVY_RUNTIME = 8.0


def default_mining_allowed(
    canonical: str,
    params: dict[str, Any] | None = None,
    *,
    heavy_runtime: float = _DEFAULT_HEAVY_RUNTIME,
) -> bool:
    """R9-P1-046: whether ``(canonical, params)`` may enter DEFAULT mining.

    An operator whose parameterised runtime exceeds ``heavy_runtime`` is
    excluded UNLESS it declares its own cost contract (``metadata.cost_model``)
    — a heavy operator with no declared contract must not silently consume the
    default search budget.  Research-only families are already excluded by
    construction (see :func:`is_research_only`).
    """
    params = params or {}
    if is_research_only(canonical):
        return False
    rt = runtime_cost(canonical, params)
    if rt <= float(heavy_runtime):
        return True
    return has_declared_cost_contract(canonical)


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
    "search_budget_gate",
    "default_mining_allowed",
    "has_declared_cost_contract",
    "is_research_only",
    "complexity_label",
    "_COMPLEXITY_KERNELS",
]
