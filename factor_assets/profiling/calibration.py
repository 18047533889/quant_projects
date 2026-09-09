"""Development-only calibration artifacts for V5 factor health policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CalibrationObservation:
    factor_id: str
    family_id: str
    evaluation_ref: str
    rank_ic: float
    rank_icir_raw: float
    split_role: str = "DEVELOPMENT"
    passed_basic_data_checks: bool = True

    def __post_init__(self) -> None:
        if self.split_role not in {"TRAIN", "DEVELOPMENT"}:
            raise ValueError("sealed test/holdout observations cannot calibrate policy")
        if not self.passed_basic_data_checks:
            raise ValueError("reference observations must share basic data checks")
        if not all((self.factor_id, self.family_id, self.evaluation_ref)):
            raise ValueError("factor, family and evaluation references are required")


@dataclass(frozen=True)
class DevelopmentCalibrationArtifact:
    calibration_id: str
    policy_id: str
    cutoff_as_of: str
    cohort_kind: str
    method: str
    observation_count: int
    family_counts: tuple[tuple[str, int], ...]
    reference_candidate_hash: str
    percentile_bands: tuple[tuple[str, tuple[float, ...]], ...]
    source_evaluation_refs: tuple[str, ...]
    synthetic_example: bool
    admission_authority: bool = False

    def __post_init__(self) -> None:
        if self.cohort_kind != "DEVELOPMENT_REFERENCE_COHORT":
            raise ValueError("only development reference cohorts are supported")
        if self.admission_authority:
            raise ValueError("uncertified development calibration cannot be admission authority")
        if self.observation_count < 2 or len(self.family_counts) < 2:
            raise ValueError("calibration requires multiple observations and families")

    def to_dict(self) -> dict:
        return asdict(self)

    def persist(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DevelopmentCalibrationArtifact":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["family_counts"] = tuple((k, int(v)) for k, v in raw["family_counts"])
        raw["percentile_bands"] = tuple((k, tuple(v)) for k, v in raw["percentile_bands"])
        raw["source_evaluation_refs"] = tuple(raw["source_evaluation_refs"])
        return cls(**raw)


@dataclass(frozen=True)
class PolicyRescoreTrace:
    old_policy_ref: str
    new_policy_ref: str
    source_evaluation_refs: tuple[str, ...]
    reused_raw_metric_hash: str
    recomputed_metric_ids: tuple[str, ...] = ()
    rematerialized_factor_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.recomputed_metric_ids or self.rematerialized_factor_refs:
            raise ValueError("policy-only rescore must not recompute metrics or FE materializations")


def build_development_calibration(
    observations: Sequence[CalibrationObservation], *, policy_id: str,
    cutoff_as_of: str, synthetic_example: bool,
    max_per_family: int = 3,
) -> DevelopmentCalibrationArtifact:
    """Build a frozen family-capped cohort; inputs are never winner-filtered."""
    if max_per_family < 1:
        raise ValueError("max_per_family must be positive")
    ordered = sorted(observations, key=lambda row: (row.family_id, row.factor_id))
    counts: dict[str, int] = {}
    cohort: list[CalibrationObservation] = []
    for row in ordered:
        if counts.get(row.family_id, 0) < max_per_family:
            cohort.append(row)
            counts[row.family_id] = counts.get(row.family_id, 0) + 1
    if len(counts) < 2:
        raise ValueError("family-aware calibration requires at least two families")
    # Fixed display percentiles; never recomputed from a current winners list.
    probs = (.975, .90, .80, .65, .50, .30)
    def quantiles(name: str) -> tuple[float, ...]:
        values = sorted(float(getattr(row, name)) for row in cohort)
        return tuple(values[min(len(values) - 1, int(p * (len(values) - 1)))] for p in probs)
    payload = [asdict(row) for row in cohort]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return DevelopmentCalibrationArtifact(
        calibration_id=f"cal:{policy_id}:{digest[:16]}", policy_id=policy_id,
        cutoff_as_of=cutoff_as_of, cohort_kind="DEVELOPMENT_REFERENCE_COHORT",
        method="FAMILY_CAPPED_FIXED_COHORT_EMPIRICAL_QUANTILES",
        observation_count=len(cohort), family_counts=tuple(sorted(counts.items())),
        reference_candidate_hash=digest,
        percentile_bands=(("rank_ic", quantiles("rank_ic")),
                          ("rank_icir_raw", quantiles("rank_icir_raw"))),
        source_evaluation_refs=tuple(row.evaluation_ref for row in cohort),
        synthetic_example=synthetic_example, admission_authority=False,
    )


def build_policy_rescore_trace(
    *, old_policy_ref: str, new_policy_ref: str,
    raw_metrics: Mapping[str, Mapping[str, float]],
) -> PolicyRescoreTrace:
    canonical = json.dumps(raw_metrics, sort_keys=True)
    return PolicyRescoreTrace(
        old_policy_ref, new_policy_ref, tuple(sorted(raw_metrics)),
        hashlib.sha256(canonical.encode()).hexdigest(),
    )
