"""Stable public facade for GTJA185 batch evaluation.

Implementation is split into contracts, vectorized leakage-safe metrics, and the
FactorEngine/DataAccess runner.  This module preserves the original import surface
for AutoFactorEvaluation callers and tests.
"""
from __future__ import annotations

from .gtja185_metrics import (
    apply_point_in_time_universe as _apply_point_in_time_universe,
    apply_validation_fdr as _apply_validation_fdr,
    daily_ic as _daily_ic,
    long_short_series as _long_short_series,
    normalize_market_frame as _normalize_market_frame,
    performance_stats as _performance_stats,
    price_forward_returns as _price_forward_returns,
    purged_split_mask as _purged_split_mask,
    purify_series as _purify_series,
    ranking_rows as _ranking_rows,
    route_factor as _route_factor,
    split_boundaries as _split_boundaries,
    summary_stats as _summary_stats,
)
from .gtja185_models import (
    BatchEvaluationConfig,
    BatchRunSummary,
    FactorRunRecord,
    SplitBoundaries,
)
from .gtja185_runner import (
    build_synthetic_market_frame,
    main,
    run_gtja185_evaluation,
)

__all__ = [
    "BatchEvaluationConfig",
    "BatchRunSummary",
    "FactorRunRecord",
    "SplitBoundaries",
    "build_synthetic_market_frame",
    "run_gtja185_evaluation",
    "main",
    "_apply_point_in_time_universe",
    "_apply_validation_fdr",
    "_daily_ic",
    "_long_short_series",
    "_normalize_market_frame",
    "_performance_stats",
    "_price_forward_returns",
    "_purged_split_mask",
    "_purify_series",
    "_ranking_rows",
    "_route_factor",
    "_split_boundaries",
    "_summary_stats",
]


if __name__ == "__main__":
    raise SystemExit(main())
