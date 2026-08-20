"""OOS prediction, content identity, schema, and dependency contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable
import enum


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PredictionRow:
    date: str
    stock: str
    prediction: float | None
    artifact_id: str
    model_version: str
    training_cutoff: str
    available_at: str
    feature_schema_hash: str
    source_snapshot_hash: str
    feature_snapshot_id: str
    feature_snapshot_hash: str

    def validate(self) -> None:
        required = ("artifact_id", "feature_schema_hash", "source_snapshot_hash",
                    "feature_snapshot_id", "feature_snapshot_hash")
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"prediction row missing identity: {missing}")
        if self.available_at < self.training_cutoff:
            raise ValueError("artifact available_at precedes training_cutoff")


@dataclass(frozen=True)
class PredictionCacheIdentity:
    artifact_id: str
    feature_snapshot_id: str
    feature_snapshot_hash: str
    source_snapshot_hash: str
    date: str
    stock: str

    def key(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class RetrainFingerprint:
    model_semantic: str
    training_cutoff: str
    snapshot_hash: str
    policy_hash: str
    code_hash: str

    def value(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    retrain_fingerprint: str
    machine: dict[str, Any]
    started_at: str
    finished_at: str | None = None
    retries: int = 0
    log_uri: str | None = None


@dataclass(frozen=True)
class ModelArtifactIdentity:
    artifact_id: str
    content_hash: str
    fit_fingerprint: str
    producing_run_id: str

    @classmethod
    def from_content(cls, content: Any, fit_fingerprint: str, producing_run_id: str) -> "ModelArtifactIdentity":
        digest = stable_hash(content)
        return cls("artifact-" + digest, digest, fit_fingerprint, producing_run_id)


@dataclass(frozen=True)
class RuntimeEnvironment:
    library_versions: dict[str, str]
    blas: str
    device: str
    determinism_flags: tuple[str, ...]
    thread_count: int

    @property
    def runtime_environment_hash(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class ModelFitFingerprint:
    frozen_params_hash: str
    preprocessing_hash: str
    training_cohort_hash: str
    weights_hash: str
    feature_schema_hash: str
    label_hash: str
    code_hash: str

    def value(self) -> str:
        return stable_hash(self.__dict__)


class PredictionStatus(str, enum.Enum):
    OK = "ok"
    MISSING_FEATURE = "missing_feature"
    OUT_OF_UNIVERSE = "out_of_universe"
    NO_ACTIVE_ARTIFACT = "no_active_artifact"
    NUMERICAL_FAILURE = "numerical_failure"
    CLOCK_VIOLATION = "clock_violation"


class FallbackPolicy(str, enum.Enum):
    NO_SIGNAL = "no_signal"
    USE_PREVIOUS_ACTIVE_ARTIFACT = "use_previous_active_artifact"
    USE_BASELINE_MODEL = "use_baseline_model"
    ZERO_SCORE = "zero_score"


class DeploymentState(str, enum.Enum):
    ACTIVE = "active"
    DEGRADED_STALE = "degraded_stale"
    RETIRED = "retired"
    REVOKED = "revoked"


@dataclass(frozen=True)
class FeatureBundle:
    canonical: str
    operator_semantic_version: str
    params: dict[str, Any]
    normalized_ast_hash: str
    source_snapshot: str
    semantic_version: str

    def identity(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class TrainingCohortSummary:
    n_dates: int
    n_stocks: int
    n_rows: int
    industry_coverage: dict[str, int]
    cap_distribution: dict[str, float]
    missingness: dict[str, float]
    label_distribution: dict[str, float]
    feature_summary: dict[str, dict[str, float]]


@dataclass
class ArtifactDependencyDAG:
    """Edges point from an artifact to immutable dependencies."""

    edges: dict[str, set[str]] = field(default_factory=dict)

    def add(self, artifact_id: str, *dependencies: str) -> None:
        self.edges.setdefault(artifact_id, set()).update(dependencies)

    def invalidate(self, changed: set[str]) -> set[str]:
        invalid = set(changed)
        while True:
            newly_invalid = {artifact for artifact, deps in self.edges.items() if deps & invalid}
            expanded = invalid | newly_invalid
            if expanded == invalid:
                return invalid
            invalid = expanded


@dataclass(frozen=True)
class ArtifactSchemaMigration:
    from_version: str
    to_version: str
    converter_id: str

    def convert(self, payload: dict[str, Any], converter: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        if not self.converter_id:
            raise ValueError("schema migration requires an explicit converter")
        return converter(payload)


@dataclass(frozen=True)
class FrozenModelSchema:
    learner: str
    version: str
    required_fields: tuple[str, ...]
    field_types: dict[str, type]
    shapes: dict[str, tuple[int | None, ...]] = field(default_factory=dict)

    def validate(self, payload: dict[str, Any]) -> None:
        if set(payload) != set(self.required_fields):
            raise ValueError(f"{self.learner} frozen fields do not match schema {self.version}")
        for name, expected_type in self.field_types.items():
            if not isinstance(payload[name], expected_type):
                raise TypeError(f"{self.learner}.{name} must be {expected_type.__name__}")
        for name, expected_shape in self.shapes.items():
            actual = getattr(payload[name], "shape", None)
            if actual is None or len(actual) != len(expected_shape):
                raise ValueError(f"{self.learner}.{name} shape mismatch")
            if any(want is not None and got != want for got, want in zip(actual, expected_shape)):
                raise ValueError(f"{self.learner}.{name} shape mismatch")
