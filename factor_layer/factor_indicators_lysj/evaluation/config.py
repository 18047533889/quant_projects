from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EvalConfig:
    """Global configuration for the factor evaluation pipeline.

    Contains column mappings for input data and all evaluation parameters.
    No file paths — data is passed directly to the pipeline.
    """

    # -- Market data column mapping --
    timestamp_col: str = "data"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    vwap_col: str = "average"

    # -- Factor input column mapping --
    factor_col: str = "factor"
    factor_timestamp_col: str = "timestamp"

    # -- Evaluation parameters --
    horizons: Tuple[int, ...] = (1, 5, 10, 20)
    zscore_window: int = 200
    winsor_quantile: float = 0.01
    n_quantiles: int = 10
    min_obs_per_day: int = 30
    holding_fee_rate: float = 0.00002
    quantile_lookback: int = 1000

    # -- MarketStatusLabel parameters --
    vol_window: int = 120
    vol_rank_window: int = 7200
    vol_low_pct: float = 1.0 / 3.0
    vol_high_pct: float = 2.0 / 3.0

    # -- Status methods to evaluate --
    status_methods: Tuple[str, ...] = ("volatility_regime", "intraday_session")
