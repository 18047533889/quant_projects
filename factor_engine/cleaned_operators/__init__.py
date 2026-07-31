# -*- coding: utf-8 -*-
"""可独立交付的统一算子库（唯一 production runtime 层）。"""
from cleaned_operators.registry import OperatorRegistry

from cleaned_operators import common  # noqa: F401
from cleaned_operators import price_volume  # noqa: F401
from cleaned_operators import technical  # noqa: F401
from cleaned_operators import fundamental  # noqa: F401
from cleaned_operators import microstructure  # noqa: F401
from cleaned_operators import _aliases  # noqa: F401

__all__ = [
    "OperatorRegistry",
    "common",
    "price_volume",
    "technical",
    "fundamental",
    "microstructure",
]

_LOAD_MODULES = (
    "cleaned_operators.common.elementwise",
    "cleaned_operators.common.time_series",
    "cleaned_operators.common.shift_cum",
    "cleaned_operators.common.cross_sectional",
    "cleaned_operators.common.group",
    "cleaned_operators.common.data_cleaning",
    "cleaned_operators.common.statistics",
    "cleaned_operators.common.daily_panel",
    "cleaned_operators.common.gtja_compat",
    "cleaned_operators.common.scalar_compare",
    "cleaned_operators.common.scalar_where",
    "cleaned_operators.lqtp_compat",
    "cleaned_operators.common.polars_ops",
    "cleaned_operators.common.group_polars",
    "cleaned_operators.common.shift_polars",
    "cleaned_operators.common.polars_extended",
    "cleaned_operators.common.polars_auto",
    "cleaned_operators.common.polars_data_cleaning",
    "cleaned_operators.common.polars_statistics",
    "cleaned_operators.common.polars_np_parity",
    "cleaned_operators.common.polars_math_extended",
    "cleaned_operators.common.polars_batch_mirror",
    "cleaned_operators.common.polars_daily_native",
    "cleaned_operators.research_polars",
    "cleaned_operators.price_volume.ops",
    "cleaned_operators.price_volume.polars_price_volume",
    "cleaned_operators.technical.signal",
    "cleaned_operators.technical.polars_signal",
    "cleaned_operators.fundamental.ops",
    "cleaned_operators.microstructure.ops",
    "cleaned_operators.microstructure.polars_microstructure",
    "cleaned_operators.semantic_hardening",
    "cleaned_operators.operator_overhaul",
    "cleaned_operators.composite_fastpath",
    "cleaned_operators.composite_fastpath_fixes",
    "cleaned_operators.layer_primitives",
    "cleaned_operators.layer_composite_fixes",
)

_LOADED = False


def load_all() -> None:
    global _LOADED
    if _LOADED:
        if OperatorRegistry.lifecycle() == "frozen":
            return
        from cleaned_operators.registry import RegistryInitializationError
        raise RegistryInitializationError(
            f"loaded registry is unexpectedly {OperatorRegistry.lifecycle()!r}"
        )
    if OperatorRegistry.lifecycle() != "building":
        from cleaned_operators.registry import RegistryInitializationError
        raise RegistryInitializationError(
            f"unloaded registry cannot initialize from {OperatorRegistry.lifecycle()!r}"
        )

    # Load all historical/native modules first, then reviewed replacements while
    # the registry is still mutable. No physical implementation may be swapped
    # after layer_governance seals duplicate registration.
    for mod in _LOAD_MODULES:
        __import__(mod, fromlist=["*"])
    __import__("cleaned_operators.production_repairs", fromlist=["*"])
    __import__("cleaned_operators.price_volume.technical_extensions", fromlist=["*"])
    __import__("cleaned_operators.price_volume.technical_structure_repairs", fromlist=["*"])
    __import__("cleaned_operators.price_volume.candle_patterns_extended", fromlist=["*"])

    from cleaned_operators._dedupe import apply_operator_deduplication
    apply_operator_deduplication()

    from cleaned_operators.operator_overhaul import finalize_operator_overhaul
    finalize_operator_overhaul()

    from cleaned_operators.lqtp_policy_patch import apply_lqtp_policy_patch
    apply_lqtp_policy_patch()

    from cleaned_operators.research_factor_enable import enable_research_factor_runtime
    enable_research_factor_runtime()

    from cleaned_operators.layer_governance import finalize_layer_governance
    finalize_layer_governance()

    from cleaned_operators.layer_governance_post import apply_post_governance
    apply_post_governance()

    from cleaned_operators.production_hardening import apply_production_hardening
    apply_production_hardening()

    from backend.sql_pushdown.sql_registry import register_sql_backends
    register_sql_backends()

    OperatorRegistry.finalize()
    OperatorRegistry.freeze()
    _LOADED = True
