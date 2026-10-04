"""Source-evaluation metric surface, separate from source-auto certification.

Implementation support describes kernels the source evaluator can compute.
It does not grant source-auto routing or performance qualification.
"""
from __future__ import annotations

from quant_evaluator.runtime.gpu_quantile_shape_adapter import (
    GPU_LINEAR_QUANTILE_METRICS,
)
from quant_evaluator.runtime.source_auto_evidence import SOURCE_AUTO_METRICS


SOURCE_SUMMARY_METRICS = frozenset({
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
})
SOURCE_IMPLEMENTED_METRICS = (
    SOURCE_AUTO_METRICS | GPU_LINEAR_QUANTILE_METRICS | SOURCE_SUMMARY_METRICS
)
SOURCE_SERIES_METRICS = frozenset({"rank_ic_series", "pearson_ic_series"})

__all__ = (
    "SOURCE_IMPLEMENTED_METRICS",
    "SOURCE_SUMMARY_METRICS",
    "SOURCE_SERIES_METRICS",
)
