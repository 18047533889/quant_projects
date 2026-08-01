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

_REVIEWED_EXTENSIONS = (
    "cleaned_operators.production_repairs",
    "cleaned_operators.price_volume.technical_extensions",
    "cleaned_operators.price_volume.technical_structure_repairs",
    "cleaned_operators.price_volume.candle_patterns_extended",
    "cleaned_operators.price_volume.candle_geometry_v2",
    "cleaned_operators.price_volume.candle_pattern_engine_v2",
    "cleaned_operators.price_volume.candle_pattern_engine_repairs_v2",
    "cleaned_operators.price_volume.structure_patterns_v2",
    "cleaned_operators.price_volume.structure_patterns_extra_v2",
    "cleaned_operators.price_volume.structure_patterns_extra_repairs_v2",
    "cleaned_operators.price_volume.liquidity_v2",
    "cleaned_operators.price_volume.liquidity_naming_v2",
    "cleaned_operators.technical.indicators_v2",
    "cleaned_operators.common.stable_high_moments_v2",
    "cleaned_operators.fundamental.transforms_v2",
    "cleaned_operators.fundamental.transforms_repairs_v2",
    "cleaned_operators.fundamental.flow_semantics_v2",
    "cleaned_operators.fundamental.expectation_v2",
    "cleaned_operators.fundamental.parameter_contract_v2",
    "cleaned_operators.production_policy_extensions_v2",
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

    from backend.production_signature_v2 import apply_production_signature_v2
    apply_production_signature_v2()

    from cleaned_operators.registration_audit import install_registration_audit
    install_registration_audit()

    for mod in _LOAD_MODULES:
        __import__(mod, fromlist=["*"])

    for mod in _REVIEWED_EXTENSIONS:
        __import__(mod, fromlist=["*"])

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

    from cleaned_operators.registration_audit import finalize_registration_audit
    finalize_registration_audit()

    from backend.sql_pushdown.sql_registry import register_sql_backends
    register_sql_backends()

    # Replace the legacy first-seen fiscal SQL lowering with exact ordinal,
    # revision-aware semantics before evidence and final contracts are consumed.
    from backend.sql_pushdown.fiscal_v2 import apply_fiscal_sql_v2
    apply_fiscal_sql_v2()

    # Evidence validation must observe the final SQL emitter set, including the
    # fiscal-v2 lowering installed above.  Running this earlier makes the
    # implementation-bound emitter contract fail closed for the wrong reason.
    from cleaned_operators.production_certification_overlay import apply_evidence_certification_overlay
    apply_evidence_certification_overlay()

    from cleaned_operators.contract_hardening import apply_final_contract_hardening
    apply_final_contract_hardening()

    OperatorRegistry.finalize()
    OperatorRegistry.freeze()
    _LOADED = True
