"""Production drift, PIT realised-metric, and training gate contracts."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, rankdata, wasserstein_distance


class MonitorAction(str, enum.Enum):
    MONITOR = "monitor"
    CAMPAIGN_REQUEST = "campaign_request"
    PROMOTION_BLOCK = "promotion_block"


class LifecycleReason(str, enum.Enum):
    WAITING_LABEL = "waiting_label"
    DATA_PENDING = "data_pending"
    BUDGET_STOP = "budget_stop"
    INVALID_SPEC = "invalid_spec"
    IMPLEMENTATION_ERROR = "implementation_error"
    REJECTED_QUALITY = "rejected_quality"


class LifecycleAction(str, enum.Enum):
    WAIT_FOR_MATURITY = "wait_for_maturity"
    WAIT_FOR_DATA = "wait_for_data"
    REQUEST_BUDGET = "request_budget"
    TERMINAL_REJECT = "terminal_reject"
    RETRY_AFTER_FIX = "retry_after_fix"


@dataclass(frozen=True)
class LifecycleDisposition:
    action: LifecycleAction
    retryable: bool
    terminal: bool
    gc_candidate: bool


_LIFECYCLE_POLICY = {
    LifecycleReason.WAITING_LABEL: LifecycleDisposition(LifecycleAction.WAIT_FOR_MATURITY, True, False, False),
    LifecycleReason.DATA_PENDING: LifecycleDisposition(LifecycleAction.WAIT_FOR_DATA, True, False, False),
    LifecycleReason.BUDGET_STOP: LifecycleDisposition(LifecycleAction.REQUEST_BUDGET, True, False, False),
    LifecycleReason.INVALID_SPEC: LifecycleDisposition(LifecycleAction.TERMINAL_REJECT, False, True, True),
    LifecycleReason.IMPLEMENTATION_ERROR: LifecycleDisposition(LifecycleAction.RETRY_AFTER_FIX, True, False, False),
    LifecycleReason.REJECTED_QUALITY: LifecycleDisposition(LifecycleAction.TERMINAL_REJECT, False, True, True),
}


@dataclass(frozen=True)
class LifecycleEvent:
    work_intent_id: str
    reason: LifecycleReason
    original_cause: str
    occurred_at: str
    prior_state: str
    next_state: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, LifecycleReason):
            raise TypeError("LifecycleEvent.reason must be LifecycleReason")
        for name in ("work_intent_id", "original_cause", "occurred_at", "prior_state", "next_state"):
            if not getattr(self, name):
                raise ValueError(f"LifecycleEvent.{name} is required")

    @property
    def disposition(self) -> LifecycleDisposition:
        return _LIFECYCLE_POLICY[self.reason]


@dataclass(frozen=True)
class DriftThreshold:
    monitor: float
    retrain: float | None = None
    block: float | None = None

    def action(self, value: float) -> MonitorAction:
        if self.block is not None and value >= self.block:
            return MonitorAction.PROMOTION_BLOCK
        if self.retrain is not None and value >= self.retrain:
            return MonitorAction.CAMPAIGN_REQUEST
        return MonitorAction.MONITOR


DRIFT_DIMENSIONS = (
    "psi", "ks", "wasserstein", "prediction_rank", "coefficient",
    "ic_decay", "coverage", "regime_occupancy", "missingness",
)

DRIFT_CATEGORIES = {
    "psi": "feature", "ks": "feature", "wasserstein": "feature",
    "prediction_rank": "predictive", "coefficient": "exposure",
    "ic_decay": "predictive", "coverage": "data",
    "regime_occupancy": "feature", "missingness": "data",
}


@dataclass(frozen=True)
class CampaignRequestPolicy:
    """Pre-approved, finite authority for a new research campaign."""

    campaign_id: str
    candidate_budget: int
    baseline_artifact_id: str
    baseline_feature_version: str
    rollback_artifact_id: str
    rollback_feature_version: str

    def __post_init__(self) -> None:
        for name in (
            "campaign_id", "baseline_artifact_id", "baseline_feature_version",
            "rollback_artifact_id", "rollback_feature_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be >= 1")
        if ((self.rollback_artifact_id == self.baseline_artifact_id)
                != (self.rollback_feature_version == self.baseline_feature_version)):
            raise ValueError("rollback must bind a matching artifact and feature version")


@dataclass(frozen=True)
class DriftContract:
    """Maps every required production drift dimension to an explicit action."""

    thresholds: dict[str, DriftThreshold] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(DRIFT_DIMENSIONS) - set(self.thresholds)
        if missing:
            raise ValueError(f"missing drift thresholds: {sorted(missing)}")

    def assess(
        self,
        metrics: dict[str, float],
        *,
        campaign_policy: CampaignRequestPolicy | None = None,
        realised_metrics: dict[str, "RealisedMetric"] | None = None,
        asof: str | None = None,
    ) -> dict[str, MonitorAction]:
        missing = set(DRIFT_DIMENSIONS) - set(metrics)
        if missing:
            raise ValueError(f"missing drift metrics: {sorted(missing)}")
        actions = {name: self.thresholds[name].action(float(metrics[name])) for name in DRIFT_DIMENSIONS}
        requests = [name for name, action in actions.items() if action is MonitorAction.CAMPAIGN_REQUEST]
        if requests and campaign_policy is None:
            raise ValueError("drift campaign request requires a bounded CampaignRequestPolicy")
        predictive_requests = [name for name in requests if DRIFT_CATEGORIES[name] == "predictive"]
        if predictive_requests:
            if realised_metrics is None or asof is None:
                raise ValueError("predictive drift requires mature realised metrics and asof")
            for name in predictive_requests:
                metric = realised_metrics.get(name)
                if metric is None:
                    raise ValueError(f"missing realised metric for predictive drift: {name}")
                metric.require_available(asof)
        return actions


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def population_stability_index(reference: Any, current: Any, bins: int = 10) -> float:
    ref, cur = _finite(reference), _finite(current)
    if not len(ref) or not len(cur):
        return float("inf")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_count, _ = np.histogram(ref, edges)
    cur_count, _ = np.histogram(cur, edges)
    ref_prob = np.maximum(ref_count / len(ref), 1e-12)
    cur_prob = np.maximum(cur_count / len(cur), 1e-12)
    return float(np.sum((cur_prob - ref_prob) * np.log(cur_prob / ref_prob)))


def production_drift_metrics(
    reference: Any,
    current: Any,
    *,
    reference_prediction: Any,
    current_prediction: Any,
    reference_coefficient: Any,
    current_coefficient: Any,
    reference_ic: float,
    current_ic: float,
    reference_coverage: float,
    current_coverage: float,
    reference_regime_occupancy: Any,
    current_regime_occupancy: Any,
    reference_missingness: float,
    current_missingness: float,
) -> dict[str, float]:
    """Compute the nine standardised non-negative production drift magnitudes."""
    ref, cur = _finite(reference), _finite(current)
    rp, cp = _finite(reference_prediction), _finite(current_prediction)
    rc, cc = _finite(reference_coefficient), _finite(current_coefficient)
    ro, co = _finite(reference_regime_occupancy), _finite(current_regime_occupancy)
    if any(len(x) == 0 for x in (ref, cur, rp, cp, rc, cc, ro, co)):
        raise ValueError("drift inputs must contain finite observations")
    return {
        "psi": population_stability_index(ref, cur),
        "ks": float(ks_2samp(ref, cur).statistic),
        "wasserstein": float(wasserstein_distance(ref, cur)),
        "prediction_rank": float(ks_2samp(rankdata(rp), rankdata(cp)).statistic),
        "coefficient": float(wasserstein_distance(rc, cc)),
        "ic_decay": max(0.0, float(reference_ic) - float(current_ic)),
        "coverage": abs(float(current_coverage) - float(reference_coverage)),
        "regime_occupancy": float(wasserstein_distance(ro, co)),
        "missingness": abs(float(current_missingness) - float(reference_missingness)),
    }


@dataclass(frozen=True)
class RealisedMetric:
    prediction_id: str
    metric_name: str
    value: float | None
    horizon_sessions: int
    decision_at: str
    metric_available_at: str
    observed_at: str

    def require_available(self, asof: str) -> None:
        if asof < self.metric_available_at:
            raise ValueError("realised metric is not PIT-available")


@dataclass(frozen=True)
class ModelCostContract:
    max_fit_seconds: float
    max_peak_memory_bytes: int
    max_artifact_bytes: int
    max_prediction_latency_seconds: float
    min_rows_per_second: float

    def failures(self, observed: dict[str, float]) -> list[str]:
        checks = (
            ("fit_seconds", observed["fit_seconds"] <= self.max_fit_seconds),
            ("peak_memory_bytes", observed["peak_memory_bytes"] <= self.max_peak_memory_bytes),
            ("artifact_bytes", observed["artifact_bytes"] <= self.max_artifact_bytes),
            ("prediction_latency_seconds", observed["prediction_latency_seconds"] <= self.max_prediction_latency_seconds),
            ("rows_per_second", observed["rows_per_second"] >= self.min_rows_per_second),
        )
        return [name for name, passed in checks if not passed]


@dataclass(frozen=True)
class DataQualityCertificate:
    certificate_id: str
    cohort: str
    feature_snapshot_hash: str
    universe_snapshot_hash: str
    source_snapshot_hash: str
    valid: bool
    reasons: tuple[str, ...] = ()

    def require_valid(self, expected_cohort: str) -> None:
        if self.cohort != expected_cohort or not self.valid:
            raise ValueError(f"data quality gate failed for {expected_cohort}: {self.reasons}")
