"""evaluation — Modular factor evaluation framework.

Usage::

    import pandas as pd
    from evaluation import EvalConfig, run_evaluation, serialize_results

    factor_df = pd.read_parquet("factor_output.parquet")
    market_df = pd.read_parquet("NQ.parquet")

    cfg = EvalConfig(vwap_col="average")
    raw = run_evaluation(factor_df, market_df, cfg)       # pandas objects
    json_safe = serialize_results(raw, cfg)                # JSON-ready dict
"""

from .config import EvalConfig
from .pipeline import (
    evaluate_horizon,
    print_brief,
    run_evaluation,
    serialize_results,
)
from .labels import MarketStatusLabel
from .preprocessing import prepare_base

__all__ = [
    "EvalConfig",
    "evaluate_horizon",
    "print_brief",
    "run_evaluation",
    "serialize_results",
    "MarketStatusLabel",
    "prepare_base",
]
