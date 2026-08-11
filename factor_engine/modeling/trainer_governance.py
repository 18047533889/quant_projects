# -*- coding: utf-8 -*-
"""Fail-closed trainer search/comparison governance contracts.

This module deliberately does not depend on ``modeling.evaluation``: evaluation
implementations may evolve independently, while trainer comparison semantics
remain explicit, versioned, and auditable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

__all__ = [
    "GovernanceContract",
    "CandidateExposureLedger",
    "ComparisonRecord",
    "ComparisonStatus",
    "apply_versioned_transform",
    "common_oos_cohort",
    "enforce_coverage",
    "incremental_value",
    "validate_pit_categorical_vocabulary",
]


class ComparisonStatus:
    COMPUTED = "computed"
    NOT_COMPUTABLE = "not_computable"
    CODE_ERROR = "code_error"


@dataclass(frozen=True)
class GovernanceContract:
    """Version pins and hard budgets for a governed trainer run."""

    version: str
    objective: str = "rank_ic"
    objective_transform_version: str = "identity-v1"
    label_transform_version: str = "identity-v1"
    min_coverage: float = 1.0
    family_budget: int = 1
    uncertainty_method: str = "none-v1"
    categorical_vocabulary_version: str | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("governance contract version is required")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if self.family_budget < 1:
            raise ValueError("family_budget must be >= 1")
        for name in (self.objective_transform_version, self.label_transform_version):
            if name not in {"identity-v1", "negate-v1", "log1p-v1"}:
                raise ValueError(f"unknown transform version {name!r}")


@dataclass
class CandidateExposureLedger:
    """Records every exposed candidate and enforces the family search budget."""

    family_budget: int
    _identities: list[str] = field(default_factory=list)

    def expose(self, identity: str) -> None:
        if identity in self._identities:
            raise ValueError(f"candidate {identity!r} exposed more than once")
        if len(self._identities) >= self.family_budget:
            raise ValueError(
                f"family exposure budget {self.family_budget} exceeded by {identity!r}"
            )
        self._identities.append(identity)

    @property
    def identities(self) -> tuple[str, ...]:
        return tuple(self._identities)

    def digest(self) -> str:
        payload = json.dumps(self._identities, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparisonRecord:
    status: str
    value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            ComparisonStatus.COMPUTED,
            ComparisonStatus.NOT_COMPUTABLE,
            ComparisonStatus.CODE_ERROR,
        }:
            raise ValueError(f"unknown comparison status {self.status!r}")
        if self.status == ComparisonStatus.COMPUTED:
            if self.value is None or not np.isfinite(self.value):
                raise ValueError("computed comparison requires a finite value")
        elif self.value is not None:
            raise ValueError("non-computed comparison cannot carry a value")
        if self.status == ComparisonStatus.CODE_ERROR and not self.reason:
            raise ValueError("code_error requires a reason")


def apply_versioned_transform(values: np.ndarray, version: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if version == "identity-v1":
        return arr.copy()
    if version == "negate-v1":
        return -arr
    if version == "log1p-v1":
        if np.any(arr <= -1.0):
            raise ValueError("log1p-v1 domain requires values > -1")
        return np.log1p(arr)
    raise ValueError(f"unknown transform version {version!r}")


def enforce_coverage(*, observed: int, expected: int, minimum: float) -> float:
    if expected <= 0:
        raise ValueError("expected cohort size must be positive")
    coverage = observed / expected
    if coverage < minimum:
        raise ValueError(f"coverage {coverage:.6f} below hard minimum {minimum:.6f}")
    return coverage


def common_oos_cohort(candidate_keys: Iterable[Iterable[Any]]) -> tuple[Any, ...]:
    cohorts = [set(keys) for keys in candidate_keys]
    if not cohorts:
        raise ValueError("at least one OOS cohort is required")
    common = set.intersection(*cohorts)
    if not common:
        raise ValueError("candidate OOS cohorts have empty intersection")
    return tuple(sorted(common, key=lambda value: repr(value)))


def incremental_value(candidate: ComparisonRecord, baseline: ComparisonRecord) -> ComparisonRecord:
    if candidate.status == ComparisonStatus.CODE_ERROR:
        raise RuntimeError(candidate.reason or "candidate comparison code error")
    if baseline.status == ComparisonStatus.CODE_ERROR:
        raise RuntimeError(baseline.reason or "baseline comparison code error")
    if candidate.status != ComparisonStatus.COMPUTED or baseline.status != ComparisonStatus.COMPUTED:
        return ComparisonRecord(
            ComparisonStatus.NOT_COMPUTABLE,
            reason="candidate or baseline objective is not computable",
        )
    return ComparisonRecord(ComparisonStatus.COMPUTED, candidate.value - baseline.value)


def validate_pit_categorical_vocabulary(
    values: Iterable[Any], *, approved: Iterable[Any], version: str
) -> tuple[Any, ...]:
    if not version:
        raise ValueError("categorical vocabulary version is required")
    approved_set = set(approved)
    observed = set(values)
    unknown = observed - approved_set
    if unknown:
        raise ValueError(f"unknown PIT categorical values for {version}: {sorted(unknown, key=repr)!r}")
    return tuple(sorted(observed, key=repr))
