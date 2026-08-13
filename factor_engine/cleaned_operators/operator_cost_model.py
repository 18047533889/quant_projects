# -*- coding: utf-8 -*-
"""Parameter- and shape-aware operator cost model (R6 §32, R9-P1-046/047, R10).

The static ``cost:N`` metadata tag labels a nominal complexity band, but the
REAL cost of an operator depends on its parameters AND on the shape of the
panel it runs over: HVG motif entropy is O(W³), Qn / Hodges-Lehmann O(W²),
bicoherence O(W·F²), DMD O(W·d² + d³), kernel Granger O(W³) — and every one of
those per-window costs is multiplied by the cross-section size N and the time
length T.  A search budget that treats ``window=20`` and ``window=500`` (or a
500-stock panel and a 5000-stock panel) as the same cost will let a couple of
heavy operators eat the whole expression budget.

This module provides three layers:

* :func:`estimate_runtime(canonical, params, shape) -> (runtime, memory)` —
  the production estimator.  It dispatches to the operator's *declared* cost
  contract (``metadata.cost_model`` / :data:`_EXPLICIT_COST_CONTRACTS`) when
  one is present, otherwise to the per-category *complexity-class* model.  When
  neither can resolve the cost the operator is :class:`CostResolution.UNKNOWN`
  and :func:`estimate_runtime` raises :class:`CostUnknownError` — it never
  falls back to a cheap default (R10 #35).

* :data:`CostShape` + the per-category complexity classes (R10 #32).  The
  estimate is a function of the ACTUAL shape (T, N, window, group_size,
  n_features, k, bins, backend), not of the window alone.

* The R9 search gates :func:`search_budget_gate` and
  :func:`default_mining_allowed`, hardened per R10 #31/#34/#35: a declared
  cost contract is NOT a free pass (budget is still enforced) and an operator
  whose cost is UNKNOWN is unselectable in production search.

``runtime_cost`` / ``memory_cost`` remain as the deterministic per-column
estimates (used by contract hashing and research-mode planning) and keep the
cheap linear-in-window default ONLY for the research/non-production path.

Deterministic (no random sampling / no timing), pure functions of the bound
parameters and shape.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Callable

# Reference nominal cost for ``window=120`` on a mid-cost daily operator; used
# to normalise the absolute figures into multiples of one standard rolling op.
_REF_WINDOW = 120.0
_REF_COST = 120.0  # one pass over a 120-row window, per column
# Reference A-share panel used for the shape-aware normalisation (T * N * W).
_REF_T = 250.0
_REF_N = 5000.0


# ---------------------------------------------------------------------------
# CostShape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostShape:
    """The actual panel / operator shape a cost estimate is computed over.

    Dimensions (R10 #32):

    * ``T``   — time length (rows).  Default 250 (one year of daily bars).
    * ``N``   — cross-sectional size (A股 ~5000 names).  Default 5000.
    * ``window`` — trailing window (overrides the operator's ``window`` param).
    * ``group_size`` — average group breadth for ``group_*`` operators.
    * ``n_features`` — feature count for matrix-inversion / regression ops.
    * ``k``   — neighbour count for KNN-family operators.
    * ``bins`` — number of bins for bucketing operators.
    * ``backend`` — runtime backend (``polars`` / ``pandas_numpy`` / ``sql``).
    """

    T: float = _REF_T
    N: float = _REF_N
    window: float | None = None
    group_size: float | None = None
    n_features: float | None = None
    k: float | None = None
    bins: float | None = None
    backend: str = "polars"

    @classmethod
    def from_tuple(cls, shape: tuple[int, int]) -> "CostShape":
        """Build from the legacy ``(T, N)`` tuple accepted by old contracts."""
        return cls(T=float(shape[0]), N=float(shape[1]))

    def to_tuple(self) -> tuple[float, float]:
        """The ``(T, N)`` view consumed by legacy declared cost contracts."""
        return (float(self.T), float(self.N))


class CostResolution(enum.Enum):
    """How reliably an operator's cost can be resolved.

    * ``DECLARED`` — the operator declares its own ``cost_model`` (trusted).
    * ``CLASS``    — resolved through the per-category complexity-class model.
    * ``UNKNOWN``  — unresolvable: no declared contract and no known class.
      Such an operator is UNSELECTABLE in production search (R10 #35).
    """

    DECLARED = "declared"
    CLASS = "class"
    UNKNOWN = "unknown"


class ComplexityClass(enum.Enum):
    """Per-category complexity classes used by the shape-aware model (R10 #32)."""

    TS_ROLLING = "ts_rolling"          # O(T*N*W)
    CS_SORT_RANK = "cs_sort_rank"      # O(T*N*logN)
    KNN_GRAPH = "knn_graph"            # O(T*N^2) (or an ANN contract)
    KERNEL_GRAM = "kernel_gram"        # O(T*W^2) time, O(W^2) MEMORY
    MATRIX_INV = "matrix_inv"          # O(p^3)
    TRANSPORT = "transport"            # documented complexity
    GROUP = "group"                    # scales with group breadth
    UNKNOWN = "unknown"


class CostUnknownError(RuntimeError):
    """Raised by :func:`estimate_runtime` when an operator's cost cannot be
    resolved (R10 #35).  Production search must treat it as unselectable."""


# ---------------------------------------------------------------------------
# Parameter binding helpers
# ---------------------------------------------------------------------------
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


def _bind_group_size(params: dict[str, Any]) -> float:
    g = params.get("group_size") or params.get("groups") or params.get("n_groups") or 1.0
    try:
        return max(1.0, float(g))
    except (TypeError, ValueError):
        return 1.0


def _bind_n_features(params: dict[str, Any]) -> float:
    p = (
        params.get("n_features")
        or params.get("dim")
        or params.get("n_components")
        or params.get("rank")
        or 4.0
    )
    try:
        return max(1.0, float(p))
    except (TypeError, ValueError):
        return 4.0


def _coerce_shape(shape: CostShape | tuple[int, int] | None) -> CostShape | None:
    if shape is None:
        return None
    if isinstance(shape, CostShape):
        return shape
    if isinstance(shape, (tuple, list)) and len(shape) >= 2:
        try:
            return CostShape(T=float(shape[0]), N=float(shape[1]))
        except (TypeError, ValueError):
            return None
    return None


def _shape_as_tuple(shape: CostShape | tuple[int, int] | None) -> tuple[float, float] | None:
    """Legacy ``(T, N)`` view passed to declared cost contracts."""
    s = _coerce_shape(shape)
    return None if s is None else (float(s.T), float(s.N))


def _shape_factor(shape: CostShape | tuple[int, int] | None) -> tuple[float, float]:
    """(T, N) panel factors; per-column mode (shape is None) uses (1, 1)."""
    s = _coerce_shape(shape)
    if s is None:
        return 1.0, 1.0
    return (float(s.T or 1.0), float(s.N or 1.0))


# ---------------------------------------------------------------------------
# Specific complexity kernels (longest-prefix wins)
# ---------------------------------------------------------------------------
def _poly_cubic(w: float, _p: dict[str, Any]) -> float:
    return w ** 3


def _poly_quad(w: float, _p: dict[str, Any]) -> float:
    return w ** 2


def _poly_linear(w: float, _p: dict[str, Any]) -> float:
    return w


def _bicoherence(w: float, p: dict[str, Any]) -> float:
    ns = max(2, int(p.get("n_segments", 4)))
    seg = w / ns if ns != 0 else np.nan
    f = np.where(2.0) != 0, min(32.0, seg / 2.0), np.nan)
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

# Kernel families whose peak working set is an n×n matrix -> O(W²) memory.
# R10 #33: kernel / cubic-cost operators must NEVER be estimated at O(W) memory.
_QUADRATIC_MEMORY_PREFIXES: frozenset[str] = frozenset({
    "ts_hvg_motif_entropy",   # HVG visibility graph -> W×W adjacency
    "ts_hvg_",                # pairwise graph
    "ts_qn_scale",            # pairwise diffs
    "ts_hodges_lehmann_location",
    "ts_bds_statistic",
    "ts_recurrence_",         # M×M recurrence matrix
    "ts_bicoherence_",
    "ts_dmd_",                # windowed covariance/correlation matrices
    "ts_kernel_granger_score",  # Gram matrix
    "ts_residualized_hsic",     # Gram matrix
    "ts_persistence_",          # persistence barcode over W×W filtration
    "ts_signature_mahalanobis_anomaly",
})


def _match_kernel(canonical: str) -> tuple[str, Callable[[float, dict[str, Any]], float], str] | None:
    """Longest-prefix complexity-kernel match for ``canonical``."""
    best: tuple[str, Callable[[float, dict[str, Any]], float], str] | None = None
    for prefix, kernel, label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, kernel, label)
    return best


# ---------------------------------------------------------------------------
# Per-category complexity-class model (R10 #32)
# ---------------------------------------------------------------------------
#: Mapping from existing operator category/family prefixes to the complexity
#: class that governs its shape-aware cost estimate.  These are category
#: assignments (not research/cost-TIER decisions — those use operator_surface
#: metadata, see :func:`is_research_only`).
_CLASS_OVERRIDES: list[tuple[str, ComplexityClass]] = [
    # -- KNN / graph / distance-matrix: O(T*N^2) (or an ANN contract) --
    ("cs_knn_", ComplexityClass.KNN_GRAPH),
    ("cs_local_density", ComplexityClass.KNN_GRAPH),
    ("cs_local_curvature", ComplexityClass.KNN_GRAPH),
    ("cs_local_moran", ComplexityClass.KNN_GRAPH),
    ("cs_neighbor_gap", ComplexityClass.KNN_GRAPH),
    ("cs_isolation", ComplexityClass.KNN_GRAPH),
    ("cs_mahalanobis_", ComplexityClass.KNN_GRAPH),
    ("cs_robust_mahalanobis_", ComplexityClass.KNN_GRAPH),
    ("cs_relative_density", ComplexityClass.KNN_GRAPH),
    ("cs_shrinkage_mahalanobis", ComplexityClass.KNN_GRAPH),
    ("cs_actual_lof_score", ComplexityClass.KNN_GRAPH),
    # -- matrix inversion / regression solvers: O(p^3) --
    ("cs_ridge_resid", ComplexityClass.MATRIX_INV),
    ("cs_wls_resid", ComplexityClass.MATRIX_INV),
    ("cs_multi_resid", ComplexityClass.MATRIX_INV),
    ("cs_spline_resid", ComplexityClass.MATRIX_INV),
    ("cs_robust_resid", ComplexityClass.MATRIX_INV),
    ("cs_quantile_resid", ComplexityClass.MATRIX_INV),
    ("cs_regression", ComplexityClass.MATRIX_INV),
    ("ts_ridge_regression_", ComplexityClass.MATRIX_INV),
    ("ts_huber_regression_", ComplexityClass.MATRIX_INV),
    ("ts_multi_regression_", ComplexityClass.MATRIX_INV),
    ("ts_expectile_regression_", ComplexityClass.MATRIX_INV),
    ("ts_quantile_regression_", ComplexityClass.MATRIX_INV),
    ("ts_ar_", ComplexityClass.MATRIX_INV),
    ("ts_cov_if", ComplexityClass.MATRIX_INV),
    ("ts_partial_corr", ComplexityClass.MATRIX_INV),
    # -- kernel Gram matrices: O(W^2) MEMORY, O(T*W^2) time --
    ("ts_kernel_granger_score", ComplexityClass.KERNEL_GRAM),
    ("ts_residualized_hsic", ComplexityClass.KERNEL_GRAM),
    ("ts_hsic", ComplexityClass.KERNEL_GRAM),
    ("ts_distance_correlation_partial_proxy", ComplexityClass.KERNEL_GRAM),
    ("ts_distance_corr", ComplexityClass.KERNEL_GRAM),
    ("ts_distance_cov", ComplexityClass.KERNEL_GRAM),
    ("ts_mmd_rbf_shift", ComplexityClass.KERNEL_GRAM),
    ("ts_upper_tail_dependence", ComplexityClass.KERNEL_GRAM),
    ("ts_lower_tail_dependence", ComplexityClass.KERNEL_GRAM),
    ("ts_conditional_mutual_information", ComplexityClass.KERNEL_GRAM),
    ("ts_chatterjee_xi", ComplexityClass.KERNEL_GRAM),
    # -- transport / distribution distance: documented complexity --
    ("intra_profile_earth_mover_distance", ComplexityClass.TRANSPORT),
    ("intraday_return_wasserstein_shift", ComplexityClass.TRANSPORT),
    ("ts_quantile_transport_", ComplexityClass.TRANSPORT),
    ("ts_wasserstein_shift", ComplexityClass.TRANSPORT),
]


def _category_class(canonical: str) -> ComplexityClass:
    """Assign ``canonical`` to a per-category complexity class.

    Every registered operator is assigned a class; ``UNKNOWN`` is reserved for
    operators that are not even known to the registry (see :func:`cost_resolution`).
    """
    for prefix, cls in _CLASS_OVERRIDES:
        if canonical.startswith(prefix):
            return cls
    if canonical.startswith("cs_"):
        # cross-sectional sort/rank/quantile family: O(T*N*logN)
        return ComplexityClass.CS_SORT_RANK
    if canonical.startswith((
        "group_", "relation_", "holder_", "composition_", "index_", "peer_",
    )):
        # group / relation / shareholder families scale with group breadth
        return ComplexityClass.GROUP
    # Every remaining time-series-shaped family is modelled as a rolling-window
    # pass: O(T*N*W).  Elementwise / primitive ops are a sub-case with W=1.
    return ComplexityClass.TS_ROLLING


def _class_cost(
    cls: ComplexityClass,
    params: dict[str, Any],
    shape: CostShape | tuple[int, int] | None,
) -> tuple[float, float]:
    """Return the shape-aware ``(runtime, memory)`` for a complexity class,
    normalised to units of one 120-row rolling op (per column for ``shape=None``).
    """
    s = _coerce_shape(shape)
    if s is None:
        T = N = 1.0
        w = _bind_window(params)
        g = _bind_group_size(params)
        p = _bind_n_features(params)
    else:
        T = float(s.T or 1.0)
        N = float(s.N or 1.0)
        w = float(s.window) if s.window else _bind_window(params)
        g = float(s.group_size) if s.group_size else _bind_group_size(params)
        p = float(s.n_features) if s.n_features else _bind_n_features(params)

    if cls is ComplexityClass.TS_ROLLING:
        rt = T * N * w / _REF_COST if _REF_COST != 0 else np.nan
        mem = w / _REF_WINDOW if _REF_WINDOW != 0 else np.nan
    elif cls is ComplexityClass.CS_SORT_RANK:
        rt = T * N * max(2.0, math.log2(max(N, 2.0))) / _REF_COST if _REF_COST != 0 else np.nan
        mem = N / _REF_WINDOW if _REF_WINDOW != 0 else np.nan
    elif cls is ComplexityClass.KNN_GRAPH:
        rt = T * N * N / _REF_COST if _REF_COST != 0 else np.nan
        mem = np.where((_REF_WINDOW * _REF_WINDOW) != 0, N * N / (_REF_WINDOW * _REF_WINDOW), np.nan)
    elif cls is ComplexityClass.KERNEL_GRAM:
        rt = T * w * w / _REF_COST if _REF_COST != 0 else np.nan
        mem = np.where((_REF_WINDOW * _REF_WINDOW) != 0, w * w / (_REF_WINDOW * _REF_WINDOW), np.nan)
    elif cls is ComplexityClass.MATRIX_INV:
        rt = T * p ** 3 / _REF_COST if _REF_COST != 0 else np.nan
        mem = np.where((_REF_WINDOW * _REF_WINDOW) != 0, p * p / (_REF_WINDOW * _REF_WINDOW), np.nan)
    elif cls is ComplexityClass.TRANSPORT:
        rt = T * N * w * w / _REF_COST if _REF_COST != 0 else np.nan
        mem = np.where((_REF_WINDOW * _REF_WINDOW) != 0, w * w / (_REF_WINDOW * _REF_WINDOW), np.nan)
    elif cls is ComplexityClass.GROUP:
        rt = T * N * g / _REF_COST if _REF_COST != 0 else np.nan
        mem = N / _REF_WINDOW if _REF_WINDOW != 0 else np.nan
    else:
        raise CostUnknownError(f"no complexity class for {cls!r}")
    return max(1.0, float(rt)), max(1.0, float(mem))


# ---------------------------------------------------------------------------
# Declared cost contracts (R9-P1-047)
# ---------------------------------------------------------------------------
#: Reference A-share cross-section for shape scaling inside declared contracts.
def _dmd_cost_contract(params: dict[str, Any], shape: tuple[float, float] | None) -> tuple[float, float]:
    w = _bind_window(params)
    d = float(params.get("dim", 4))
    T, N = _shape_factor(shape)
    rt = (w * d * d + d ** 3) * T * N
    mem = w * w
    return np.where(_REF_COST), max(1.0, mem / (_REF_WINDOW * _REF_WINDOW)) != 0, max(1.0, rt / _REF_COST), max(1.0, mem / (_REF_WINDOW * _REF_WINDOW)), np.nan)


def _pairwise_cost_contract(params: dict[str, Any], shape: tuple[float, float] | None) -> tuple[float, float]:
    w = _bind_window(params)
    T, N = _shape_factor(shape)
    rt = w * w * T * N
    return np.where(_REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW)) != 0, max(1.0, rt / _REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW)), np.nan)


def _cubic_cost_contract(params: dict[str, Any], shape: tuple[float, float] | None) -> tuple[float, float]:
    w = _bind_window(params)
    T, N = _shape_factor(shape)
    rt = w ** 3 * T * N
    # R10 #33: kernel/cubic-cost operators build a W×W Gram structure — their
    # memory estimate must be O(W²), never O(W).
    mem = w * w
    return np.where(_REF_COST), max(1.0, mem / (_REF_WINDOW * _REF_WINDOW)) != 0, max(1.0, rt / _REF_COST), max(1.0, mem / (_REF_WINDOW * _REF_WINDOW)), np.nan)


def _bicoherence_cost_contract(params: dict[str, Any], shape: tuple[float, float] | None) -> tuple[float, float]:
    w = _bind_window(params)
    ns = max(2, int(params.get("n_segments", 4)))
    f = np.where(ns) / 2.0) != 0, min(32.0, (w / ns) / 2.0), np.nan)
    T, N = _shape_factor(shape)
    rt = w * f * f * T * N
    return np.where(_REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW)) != 0, max(1.0, rt / _REF_COST), max(1.0, rt / (_REF_WINDOW * _REF_WINDOW)), np.nan)


_EXPLICIT_COST_CONTRACTS: dict[str, Callable[[dict[str, Any], tuple[float, float] | None], tuple[float, float]]] = {
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


def _declared_cost_model(canonical: str) -> Callable[[dict[str, Any], tuple[float, float] | None], tuple[float, float]] | None:
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


def _is_known_canonical(canonical: str) -> bool:
    """Whether ``canonical`` is a known registered operator (runtime or catalog)."""
    try:
        from cleaned_operators.registry import OperatorRegistry

        resolved = OperatorRegistry._aliases.get(canonical, canonical)
        return (
            resolved in OperatorRegistry._operators
            or resolved in OperatorRegistry._catalog
        )
    except Exception:  # pragma: no cover - registry not loaded / import order
        return False


def cost_resolution(canonical: str) -> CostResolution:
    """How reliably ``canonical``'s cost can be resolved (R10 #35).

    An operator with a declared cost contract is ``DECLARED``; a known
    registered operator is resolved through the complexity-class model
    (``CLASS``); anything else is ``UNKNOWN`` and must be unselectable in
    production search.
    """
    if _declared_cost_model(canonical) is not None:
        return CostResolution.DECLARED
    if _is_known_canonical(canonical):
        return CostResolution.CLASS
    return CostResolution.UNKNOWN


def complexity_class(canonical: str) -> ComplexityClass:
    """The per-category complexity class for ``canonical`` (``UNKNOWN`` when the
    operator is not known at all)."""
    if cost_resolution(canonical) is CostResolution.UNKNOWN:
        return ComplexityClass.UNKNOWN
    return _category_class(canonical)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def estimate_runtime(
    canonical: str,
    params: dict[str, Any] | None = None,
    shape: CostShape | tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Shape-aware production estimate ``(runtime, memory)``.

    Dispatches to the operator's declared cost contract when one is present
    (passing the ``(T, N)`` shape view), otherwise to the per-category
    complexity-class model.  When the cost cannot be resolved at all
    (:class:`CostResolution.UNKNOWN`) this raises :class:`CostUnknownError`
    (R10 #35) instead of falling back to a cheap default that would let a
    heavy/unknown operator through the search budget.
    """
    params = params or {}
    declared = _declared_cost_model(canonical)
    if declared is not None:
        rt, mem = declared(params, _shape_as_tuple(shape))
        return max(1.0, float(rt)), max(1.0, float(mem))

    kernel = _match_kernel(canonical)
    if kernel is not None:
        prefix, fn, _label = kernel
        w = _bind_window(params)
        T, N = _shape_factor(shape)
        base = fn(w, params)
        rt = base * T * N / _REF_COST if _REF_COST != 0 else np.nan
        if prefix in _QUADRATIC_MEMORY_PREFIXES:
            mem = np.where((_REF_WINDOW * _REF_WINDOW) != 0, (w * w) / (_REF_WINDOW * _REF_WINDOW), np.nan)
        else:
            mem = w / _REF_WINDOW if _REF_WINDOW != 0 else np.nan
        return max(1.0, float(rt)), max(1.0, float(mem))

    cls = _category_class(canonical)
    if not _is_known_canonical(canonical):
        raise CostUnknownError(
            f"cost contract for {canonical!r} cannot be resolved (UNKNOWN); "
            "it is unselectable in production search"
        )
    return _class_cost(cls, params, shape)


def runtime_cost(canonical: str, params: dict[str, Any] | None = None, shape: CostShape | tuple[int, int] | None = None) -> float:
    """Estimated per-column runtime cost in units of one 120-row rolling op.

    ``params`` are the bound parameter values (``window`` / ``dim`` / ``lag`` /
    …).  A declared ``metadata.cost_model`` (R9-P1-047) takes precedence over
    the prefix table.  When a cost cannot be resolved this falls back to the
    cheap linear-in-window default — acceptable ONLY for research-mode planning
    / contract hashing; production search must use :func:`estimate_runtime`.
    """
    params = params or {}
    declared = _declared_cost_model(canonical)
    if declared is not None:
        rt, _mem = declared(params, _shape_as_tuple(shape))
        return max(1.0, float(rt))
    kernel = _match_kernel(canonical)
    if kernel is not None:
        _prefix, fn, _label = kernel
        w = _bind_window(params)
        T, N = _shape_factor(shape)
        base = fn(w, params)
        return np.where(_REF_COST) != 0, max(1.0, base * T * N / _REF_COST), np.nan)
    cls = _category_class(canonical)
    if cls is not ComplexityClass.UNKNOWN:
        rt, _mem = _class_cost(cls, params, shape)
        return rt
    w = _bind_window(params)
    return np.where(_REF_WINDOW) != 0, max(1.0, w / _REF_WINDOW), np.nan)


def memory_cost(canonical: str, params: dict[str, Any] | None = None, shape: CostShape | tuple[int, int] | None = None) -> float:
    """Estimated peak per-column working set in rows²-equivalents.

    Operators that materialise an n×n matrix (recurrence, RQA, covariance,
    pairwise diffs, kernel Gram) have O(window²) memory — never O(window)
    (R10 #33).  A declared ``metadata.cost_model`` (R9-P1-047) takes precedence.
    """
    params = params or {}
    declared = _declared_cost_model(canonical)
    if declared is not None:
        _rt, mem = declared(params, _shape_as_tuple(shape))
        return max(1.0, float(mem))
    kernel = _match_kernel(canonical)
    if kernel is not None:
        prefix, _fn, _label = kernel
        w = _bind_window(params)
        if prefix in _QUADRATIC_MEMORY_PREFIXES:
            return np.where((_REF_WINDOW * _REF_WINDOW)) != 0, max(1.0, (w * w) / (_REF_WINDOW * _REF_WINDOW)), np.nan)
        return np.where(_REF_WINDOW) != 0, max(1.0, w / _REF_WINDOW), np.nan)
    cls = _category_class(canonical)
    if cls is not ComplexityClass.UNKNOWN:
        _rt, mem = _class_cost(cls, params, shape)
        return mem
    w = _bind_window(params)
    return np.where(_REF_WINDOW) != 0, max(1.0, w / _REF_WINDOW), np.nan)


def has_declared_cost_contract(canonical: str) -> bool:
    """Whether the canonical declares its own ``cost_model`` (R9-P1-047)."""
    return _declared_cost_model(canonical) is not None


# ---------------------------------------------------------------------------
# Search gates
# ---------------------------------------------------------------------------
def search_budget_gate(
    canonical: str,
    params: dict[str, Any] | None = None,
    *,
    runtime_budget: float | None = None,
    memory_budget: float | None = None,
    shape: CostShape | tuple[int, int] | None = None,
) -> bool:
    """R9-P1-046: the single gate an AlphaProbe / AlphaMiner-style generator
    calls BEFORE allocating search budget to ``(canonical, params)``.

    Returns ``True`` iff the parameterised operator fits the given budgets
    (``None`` budget = unbounded on that axis).  An operator whose cost is
    UNKNOWN (R10 #35) fails the gate regardless of budget — production search
    never falls back to a cheap default for an unresolvable cost contract.
    """
    if cost_resolution(canonical) is CostResolution.UNKNOWN:
        return False
    rt, mem = estimate_runtime(canonical, params, shape=shape)
    if runtime_budget is not None and rt > float(runtime_budget):
        return False
    if memory_budget is not None and mem > float(memory_budget):
        return False
    return True


#: Default mining treats an operator as "heavy" above this multiple of one
#: reference rolling op.
_DEFAULT_HEAVY_RUNTIME = 8.0
#: Default mining treats an operator as memory-heavy above this working-set
#: multiple (in rows²-equivalents of the reference 120-row op).
_DEFAULT_HEAVY_MEMORY = 8.0


def default_mining_allowed(
    canonical: str,
    params: dict[str, Any] | None = None,
    *,
    heavy_runtime: float = _DEFAULT_HEAVY_RUNTIME,
    heavy_memory: float = _DEFAULT_HEAVY_MEMORY,
    shape: CostShape | tuple[int, int] | None = None,
) -> bool:
    """R9-P1-046 / R10 #31: whether ``(canonical, params)`` may enter DEFAULT mining.

    An operator is searchable in default mining IFF:

    * its cost contract resolves (a declared ``metadata.cost_model`` OR a known
      complexity-class estimate — an UNKNOWN cost is unselectable, R10 #35);
    * its estimated runtime <= ``heavy_runtime``; AND
    * its estimated memory  <= ``heavy_memory``.

    A declared cost contract only means the estimate is RELIABLE; it does NOT
    grant budget overage (R10 #31).  Research-only families are additionally
    excluded by their authoring tier (R10 #34), never by name prefix.
    """
    params = params or {}
    if is_research_only(canonical):
        return False
    if cost_resolution(canonical) is CostResolution.UNKNOWN:
        return False
    rt, mem = estimate_runtime(canonical, params, shape=shape)
    if rt > float(heavy_runtime):
        return False
    if mem > float(heavy_memory):
        return False
    return True


def is_research_only(canonical: str) -> bool:
    """Whether the canonical is excluded from default search by its authoring
    tier.

    R10 #34: the tier decision comes from the unified surface metadata
    (``operator_surface.classify_canonical``), never from the operator name
    prefix (e.g. ``name.startswith('research')``).
    """
    try:
        from cleaned_operators.operator_surface import classify_canonical

        return classify_canonical(canonical) == "research"
    except Exception:  # pragma: no cover - surface module importable
        return False


def complexity_label(canonical: str) -> str:
    """Human-readable asymptotic label for a canonical (for catalog/docs).

    Kept identical to the R9 behavior: the kernel table supplies the label and
    everything else is the standard ``O(W)`` band.  Changing this string would
    alter every registry contract hash, so the per-category mapping is exposed
    through :func:`complexity_class` instead.
    """
    for prefix, _kernel, label in _COMPLEXITY_KERNELS:
        if canonical.startswith(prefix):
            return label
    return "O(W)"


__all__ = [
    "CostShape",
    "CostResolution",
    "ComplexityClass",
    "CostUnknownError",
    "estimate_runtime",
    "runtime_cost",
    "memory_cost",
    "search_budget_gate",
    "default_mining_allowed",
    "has_declared_cost_contract",
    "cost_resolution",
    "complexity_class",
    "is_research_only",
    "complexity_label",
    "_COMPLEXITY_KERNELS",
]
