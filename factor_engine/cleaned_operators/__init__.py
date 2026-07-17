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


def load_all() -> None:
    for mod in _LOAD_MODULES:
        __import__(mod, fromlist=["*"])
    from cleaned_operators._dedupe import apply_operator_deduplication

    apply_operator_deduplication()

    from cleaned_operators.operator_overhaul import finalize_operator_overhaul

    finalize_operator_overhaul()

    from cleaned_operators.layer_governance import finalize_layer_governance

    finalize_layer_governance()

    from cleaned_operators.layer_governance_post import apply_post_governance

    apply_post_governance()

    # Final production hardening is intentionally performed after the registry
    # has been sealed: replacements require an explicit reason/version and
    # therefore remain auditable in OperatorRegistry._replacement_history.
    from cleaned_operators.layer_regression_fusion import install_rolling_ols_fusion
    from cleaned_operators.layer_native_polars_final import install_final_native_polars
    from cleaned_operators.layer_final_audit import apply_final_operator_audit

    install_rolling_ols_fusion()
    install_final_native_polars()
    apply_final_operator_audit()

    # Any module-level historical tier sets are narrowed to the final registry
    # only after all canonical renames, backend replacements and surface moves.
    from backend.active_capabilities import synchronize_active_backend_sets

    synchronize_active_backend_sets()
