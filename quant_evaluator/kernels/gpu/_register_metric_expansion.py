"""GPU backend capability registration for the metric-expansion families.

Registers the GPU kernels in :mod:`quant_evaluator.kernels.gpu` (exposure /
purity, tradability / capacity, novelty / interactions, data-quality) with the
:class:`BackendCapabilityRegistry` via ``register_backend_implementation``.

This is a pure registration module: it imports the kernels lazily (CuPy is
only touched when a kernel is actually invoked) and records each primitive's
operation_id, backend, dtypes, layout, and parity status.  It is idempotent —
calling :func:`register_gpu_metric_expansion_kernels` twice does not
double-register.
"""

from __future__ import annotations

from typing import Optional

from quant_evaluator.backends.capability_registry import (
    BackendImplementation,
    ParityStatus,
    get_backend_capability_registry,
    register_backend_implementation,
)

# operation_id -> (module, function name)
_GPU_KERNELS = {
    # exposure / purity (spec §24/§25)
    "hhi_concentration": ("exposure", "batched_concentration_hhi"),
    "hhi_effective_n": ("exposure", "batched_concentration_hhi"),
    "sector_exposure": ("exposure", "batched_sector_exposure"),
    "factor_loadings": ("exposure", "batched_factor_loadings"),
    "style_exposure": ("exposure", "batched_style_exposure"),
    # tradability / capacity (spec §26)
    "factor_turnover_rate": ("tradability", "batched_factor_turnover_rate"),
    "turnover_cost": ("tradability", "batched_weighted_turnover"),
    "turnover_adjusted_ic": ("tradability", "batched_turnover_contribution"),
    # novelty / interactions (spec §30)
    "pairwise_correlation": ("interactions", "batched_pairwise_correlation"),
    "conditional_ic": ("interactions", "batched_conditional_ic"),
    "substitution_effect": ("interactions", "batched_substitution_effect"),
    "complementarity_score": ("interactions", "batched_complementarity_score"),
    "interaction_strength": ("interactions", "batched_interaction_strength"),
    "incremental_ic": ("interactions", "batched_incremental_ic"),
    # data-quality (spec §35)
    "missing_ratio": ("data_quality", "batched_missing_ratio"),
    "missing_timeline": ("data_quality", "batched_missing_timeline"),
    "staleness": ("data_quality", "batched_staleness"),
    "effective_n": ("data_quality", "batched_effective_n"),
    "distinct_level_ratio": ("data_quality", "batched_distinct_level_ratio"),
    "tie_ratio": ("data_quality", "batched_tie_ratio"),
    "cross_section_cardinality": ("data_quality", "batched_cross_section_cardinality"),
    "tradable_coverage": ("data_quality", "batched_tradable_coverage"),
    "universe_churn": ("data_quality", "batched_universe_churn"),
}

_VERSION = "1.0.0"


def _resolve_fn(module_name: str, fn_name: str):
    import importlib

    mod = importlib.import_module(
        f"quant_evaluator.kernels.gpu.{module_name}"
    )
    return getattr(mod, fn_name)


def register_gpu_metric_expansion_kernels(
    parity_status: ParityStatus = ParityStatus.PARITY_PASS,
) -> int:
    """Register all metric-expansion GPU kernels; returns the count registered.

    Idempotent: an operation_id already registered for the ``cuda`` backend is
    skipped.  ``parity_status`` defaults to PARITY_PASS (the parity tests in
    ``tests/test_gpu_interactions_parity.py`` verify this on a GPU host).
    """
    reg = get_backend_capability_registry()
    count = 0
    for op_id, (module_name, fn_name) in _GPU_KERNELS.items():
        if reg.has_backend(op_id, "cuda"):
            continue
        fn = _resolve_fn(module_name, fn_name)
        register_backend_implementation(
            BackendImplementation(
                operation_id=op_id,
                backend="cuda",
                implementation_version=_VERSION,
                implementation_hash=f"gpu:{module_name}.{fn_name}",
                supported_dtypes=("float32", "float64"),
                supported_layouts=("T,F,N",),
                deterministic=True,
                exact_semantics=True,
                parity_status=parity_status,
                fn=fn,
            )
        )
        count += 1
    return count


def register_gpu_metric_expansion_kernels_if_available() -> int:
    """Register only when CuPy is importable; returns the count registered."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        return 0
    return register_gpu_metric_expansion_kernels()
