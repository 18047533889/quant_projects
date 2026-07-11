"""Stable contracts for the GTJA185 evaluation pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BatchEvaluationConfig:
    market: str = "ashare"
    dataset: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    universe_id: str = "A_SHARE_ALL_A_EX_ST"
    instrument_filter: tuple[str, ...] = ()
    backend: str = "pandas"
    run_mode: str = "research"
    horizons: tuple[int, ...] = (1, 5, 21)
    batch_size: int = 8
    min_assets: int = 20
    n_quantiles: int = 5
    winsor_mad: float = 5.0
    train_fraction: float = 0.60
    valid_fraction: float = 0.20
    annualization: int = 252
    forward_price_field: str = "vwap"
    entry_lag: int = 1
    cost_bps: float = 10.0
    fdr_alpha: float = 0.10
    universe_dataset: str | None = None
    universe_membership_field: str = "is_member"
    tradability_field: str | None = "is_tradable"
    require_point_in_time_universe: bool = False
    materialize_staging: bool = False
    publish: bool = False
    factor_version: str = "gtja185.v1"
    strict: bool = True
    resume: bool = True
    limit: int | None = None
    selected_factors: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.horizons or any(int(horizon) <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.min_assets < 2:
            raise ValueError("min_assets must be >= 2")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if not math.isfinite(float(self.winsor_mad)) or self.winsor_mad <= 0:
            raise ValueError("winsor_mad must be finite and positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.valid_fraction < 1:
            raise ValueError("valid_fraction must be in (0, 1)")
        if self.train_fraction + self.valid_fraction >= 1:
            raise ValueError("train_fraction + valid_fraction must be < 1")
        if self.annualization < 1:
            raise ValueError("annualization must be positive")
        if self.entry_lag < 1:
            raise ValueError("entry_lag must be >= 1 to prevent same-bar execution bias")
        if not math.isfinite(float(self.cost_bps)) or self.cost_bps < 0:
            raise ValueError("cost_bps must be a finite non-negative number")
        if not 0 < float(self.fdr_alpha) <= 1:
            raise ValueError("fdr_alpha must be in (0, 1]")
        if self.publish and not self.materialize_staging:
            raise ValueError("publish=True requires materialize_staging=True")
        if str(self.run_mode).lower() not in {"research", "production"}:
            raise ValueError("run_mode must be research or production")
        if str(self.run_mode).lower() == "production" and not self.require_point_in_time_universe:
            raise ValueError(
                "production GTJA185 evaluation requires require_point_in_time_universe=True"
            )
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: str
    valid_end: str
    first_date: str
    last_date: str


@dataclass
class FactorRunRecord:
    factor_name: str
    formula_hash: str
    status: str
    compile_seconds: float = 0.0
    execute_seconds: float = 0.0
    finite_values: int = 0
    coverage: float = 0.0
    direction: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)
    route: str = "rejected"
    staging: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class BatchRunSummary:
    run_id: str
    status: str
    pack_name: str
    pack_version: str
    pack_hash: str
    source_catalog_hash: str
    snapshot_id: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    factor_count: int
    succeeded: int
    failed: int
    skipped: int
    output_dir: str
    config_hash: str
    split_boundaries: dict[str, str]
    route_counts: dict[str, int]
    records: list[dict[str, Any]]
