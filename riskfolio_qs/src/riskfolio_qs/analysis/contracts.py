"""Stable contracts for position-output analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class OptimizationArtifacts:
    """Optimizer artifacts loaded from memory or a CLI output directory."""

    target_positions: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    metadata: pd.DataFrame
    run_manifest: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    resolved_params: dict[str, Any] = field(default_factory=dict)
    input_dir: Path | None = None


@dataclass(slots=True)
class AnalysisEnrichment:
    """Optional point-in-time data used by P1 analysis."""

    benchmark: pd.DataFrame | None = None
    tradable: pd.DataFrame | None = None
    market_amount: pd.DataFrame | None = None
    industry: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    factor_exposure: pd.DataFrame | None = None
    factor_cov: pd.DataFrame | None = None
    specific_var: pd.DataFrame | None = None
    factor_specs: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionAnalysisResult:
    """Structured result of P0/P1 position analysis."""

    position_summary: pd.DataFrame
    holdings_detail: pd.DataFrame
    turnover_detail: pd.DataFrame
    exposure_summary: pd.DataFrame
    risk_contribution: pd.DataFrame
    constraint_summary: pd.DataFrame
    quality_checks: pd.DataFrame
    metadata: dict[str, Any]
    output_dir: Path | None = None

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "failed"))


EMPTY_EXPOSURE_COLUMNS = [
    "portfolio_exposure",
    "benchmark_exposure",
    "active_exposure",
    "asset_coverage",
    "weight_coverage",
    "exposure_date",
    "snapshot_match",
]

EMPTY_RISK_COLUMNS = [
    "variance_contribution",
    "variance_contribution_pct",
    "marginal_risk",
    "variance",
    "volatility",
    "risk_horizon_days",
    "annualization_factor",
    "exposure_date",
    "covariance_date",
    "specific_risk_date",
]

EMPTY_CONSTRAINT_COLUMNS = [
    "scope",
    "reported_value",
    "recomputed_value",
    "limit",
    "slack",
    "utilization",
    "is_binding",
    "is_violated",
    "recompute_status",
    "reported_violation",
    "reported_slack",
    "difference",
]

EMPTY_QUALITY_COLUMNS = [
    "check_id",
    "date",
    "scope",
    "severity",
    "status",
    "observed",
    "expected",
    "difference",
    "message",
]


def empty_exposure_summary() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[], [], []], names=["date", "exposure_type", "exposure_name"]
    )
    return pd.DataFrame(index=index, columns=EMPTY_EXPOSURE_COLUMNS)


def empty_risk_contribution() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[], [], [], []],
        names=["date", "risk_basis", "component_type", "component_name"],
    )
    return pd.DataFrame(index=index, columns=EMPTY_RISK_COLUMNS)


def empty_constraint_summary() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [[], []], names=["date", "constraint_name"]
    )
    return pd.DataFrame(index=index, columns=EMPTY_CONSTRAINT_COLUMNS)


def empty_quality_checks() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_QUALITY_COLUMNS)


__all__ = [
    "AnalysisEnrichment",
    "OptimizationArtifacts",
    "PositionAnalysisResult",
]
