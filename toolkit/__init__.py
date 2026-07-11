"""Lightweight post-processing toolkits for CogAlpha factors."""

from toolkit.cross_sectional import (
    apply_cross_sectional_transform,
    cs_demean,
    cs_minmax,
    cs_quantile_bucket,
    cs_rank,
    cs_rank_gauss,
    cs_robust_zscore,
    cs_scale,
    cs_signed_power,
    cs_winsorize,
    cs_zscore,
)
from toolkit.registry import (
    CROSS_SECTIONAL_TRANSFORM_NAMES,
    EXPOSED_CROSS_SECTIONAL_TRANSFORM_NAMES,
    TransformSpec,
    get_transform_spec,
    is_allowed_transform,
)
from toolkit.alpha_tools import (
    build_alpha_tools_facade,
    get_active_tool_functions,
    get_active_tool_names,
    get_active_tool_specs,
    get_inactive_tool_names,
)

__all__ = [
    "CROSS_SECTIONAL_TRANSFORM_NAMES",
    "EXPOSED_CROSS_SECTIONAL_TRANSFORM_NAMES",
    "TransformSpec",
    "apply_cross_sectional_transform",
    "cs_demean",
    "cs_minmax",
    "cs_quantile_bucket",
    "cs_rank",
    "cs_rank_gauss",
    "cs_robust_zscore",
    "cs_scale",
    "cs_signed_power",
    "cs_winsorize",
    "cs_zscore",
    "build_alpha_tools_facade",
    "get_active_tool_functions",
    "get_active_tool_names",
    "get_active_tool_specs",
    "get_inactive_tool_names",
    "get_transform_spec",
    "is_allowed_transform",
]
