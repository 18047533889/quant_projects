"""Typed evidence returned by robust statistical producers."""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class HACEvidence:
    estimate: float
    standard_error: Optional[float]
    t_statistic: Optional[float]
    n_time: int
    kernel: str
    bandwidth: int
    bandwidth_rule: str
    status: str


@dataclass(frozen=True)
class PairedBootstrapEvidence:
    mean_difference: Optional[float]
    confidence_interval: Tuple[Optional[float], Optional[float]]
    n_time: int
    block_length: int
    repetitions: int
    seed: int
    interval_method: str
    status: str


@dataclass(frozen=True)
class DSREvidence:
    observed_sharpe: Optional[float]
    benchmark_max_sharpe: Optional[float]
    probability: Optional[float]
    n_time: int
    returns_frequency: str
    annualization_factor: float
    family_sharpe_scale: str
    skewness: Optional[float]
    kurtosis: Optional[float]
    effective_trial_count: int
    trial_ledger_ref: str
    status: str


@dataclass(frozen=True)
class PBOEvidence:
    probability_backtest_overfit: Optional[float]
    split_logits: Tuple[float, ...]
    n_time: int
    n_candidates: int
    n_splits: int
    purpose: str
    candidate_universe_complete: bool
    common_cost_spec_ref: str
    status: str


@dataclass(frozen=True)
class RetentionEvidence:
    train_value: float
    validation_value: float
    signed_difference: float
    retention_ratio: Optional[float]
    ratio_applicable: bool
    sign_reversal: bool
    effective_windows: int
    paired_interval: Tuple[Optional[float], Optional[float]]
    cost_basis: str
    status: str


@dataclass(frozen=True)
class HorizonCurveEvidence:
    horizons: Tuple[int, ...]
    predictive_ic: Tuple[float, ...]
    observation_counts: Tuple[int, ...]
    half_life: Optional[float]
    fit_status: str
    label_refs: Tuple[str, ...]
    zero_tolerance: float = 1e-12


@dataclass(frozen=True)
class RegimeEvidence:
    regime_kind: str
    labels: Tuple[int, ...]
    available_times: Tuple[object, ...]
    conditional_values: Mapping[int, Optional[float]]
    conditional_counts: Mapping[int, int]
    state_ref: Optional[str]
    status: str
