# -*- coding: utf-8 -*-
"""Model-score DSL, typed IR, artifact resolution, and frozen scoring."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import numpy as np

from modeling.artifact import ModelArtifact, PredictionContext
from modeling.contracts import ApplicationWindow
from modeling.learners.base import LearnerSpec, get_learner
from modeling.model_catalog import (
    ModelArtifactCatalog,
    ModelArtifactCatalogRecord,
    parse_catalog_timestamp,
    validate_artifact_id,
)

__all__ = [
    "ArtifactStore",
    "ArtifactResolver",
    "ScoringContext",
    "Deployment",
    "DeploymentState",
    "FrozenPredictor",
    "FrozenScorer",
    "ModelArtifactResolutionContext",
    "ModelScoreExpr",
    "TypedModelScoreIR",
    "ArtifactResolutionNode",
    "ModelFeatureReadNode",
    "FrozenScoreNode",
    "ModelScoreBlock",
    "ModelScoreResolutionStatus",
    "NoLegalArtifactError",
    "score_asof",
    "replay_historical_scores",
    "model_score",
    "lower_model_score",
    "ModelScoreOperatorStub",
    "identity_of",
    "configure_default_resolver",
]


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


@dataclass
class ArtifactStore:
    """JSON artifact storage.  Catalog metadata is managed separately."""

    base_dir: str | None = None

    def path_for(self, artifact_id: str) -> str:
        validate_artifact_id(artifact_id)
        if not self.base_dir:
            raise ValueError("ArtifactStore.base_dir must be set")
        return os.path.join(self.base_dir, f"{artifact_id}.json")

    def put(self, artifact: ModelArtifact) -> str:
        """Atomically persist an artifact and make the rename durable."""
        if not self.base_dir:
            raise ValueError("ArtifactStore.base_dir must be set before put()")
        os.makedirs(self.base_dir, exist_ok=True)
        path = self.path_for(artifact.artifact_id)
        artifact.save(path)
        return artifact.artifact_id

    def get(self, artifact_id: str) -> ModelArtifact | None:
        if not self.base_dir:
            return None
        path = self.path_for(artifact_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        learner_cls = get_learner(data["learner_name"])
        spec_data = data["learner_spec"]
        spec = LearnerSpec(
            learner_name=spec_data["learner_name"],
            family=spec_data["family"],
            hyperparams=dict(spec_data.get("hyperparams") or {}),
            random_seed=spec_data.get("random_seed"),
        )
        return ModelArtifact.from_dict(data, learner_cls(spec))

    def list(self) -> list[str]:
        if not self.base_dir or not os.path.isdir(self.base_dir):
            return []
        return sorted(
            fname[:-5] for fname in os.listdir(self.base_dir) if fname.endswith(".json")
        )

    def _catalog_path(self) -> str:
        if not self.base_dir:
            raise ValueError("ArtifactStore.base_dir must be set")
        return os.path.join(self.base_dir, "deployments.json")

    def save_deployments(self, deployments: dict[str, list["Deployment"]]) -> None:
        """Durably persist deployment metadata without serialising predictors."""
        if not self.base_dir:
            raise ValueError("ArtifactStore.base_dir must be set")
        os.makedirs(self.base_dir, exist_ok=True)
        rows = [
            {
                "artifact_id": deployment.artifact.artifact_id,
                "state": deployment.state,
                "activation_at": deployment.activation_at,
                "retired_at": deployment.retired_at,
                "revoked_at": deployment.revoked_at,
                "canary_fraction": deployment.canary_fraction,
            }
            for model_deployments in deployments.values()
            for deployment in model_deployments
        ]
        target = self._catalog_path()
        fd, temporary = tempfile.mkstemp(prefix=".deployments-", suffix=".tmp", dir=self.base_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1, default=_json_default)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_deployments(self) -> dict[str, list["Deployment"]]:
        path = self._catalog_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        deployments: dict[str, list[Deployment]] = {}
        for row in rows:
            artifact = self.get(row["artifact_id"])
            if artifact is None:
                raise LookupError(f"deployment artifact {row['artifact_id']!r} is missing")
            deployment = Deployment(
                artifact=artifact,
                state=row["state"],
                activation_at=row.get("activation_at"),
                retired_at=row.get("retired_at"),
                revoked_at=row.get("revoked_at"),
                canary_fraction=float(row.get("canary_fraction", 0.0)),
            )
            deployments.setdefault(artifact.model_name, []).append(deployment)
        return deployments


@dataclass(frozen=True)
class ScoringContext:
    """Request-scoped production scoring authority."""

    asof: Any
    clock_id: str
    schema_hash: str
    application_window: ApplicationWindow | None = None
    dates: Any = None
    production: bool = True
    max_stale_age_seconds: float | None = None


class DeploymentState:
    ACTIVE = "ACTIVE"
    CANARY = "CANARY"
    RETIRED = "RETIRED"
    REVOKED = "REVOKED"


@dataclass(frozen=True)
class Deployment:
    artifact: ModelArtifact
    state: str = DeploymentState.CANARY
    activation_at: Any = None
    retired_at: Any = None
    revoked_at: Any = None
    canary_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.state not in {
            DeploymentState.ACTIVE, DeploymentState.CANARY,
            DeploymentState.RETIRED, DeploymentState.REVOKED,
        }:
            raise ValueError(f"invalid deployment state {self.state!r}")
        if not 0.0 <= self.canary_fraction <= 1.0:
            raise ValueError("canary_fraction must be in [0, 1]")


@dataclass(frozen=True)
class ModelArtifactResolutionContext:
    tenant: str = "default"
    project: str = "default"
    market: str = "default"
    namespace: str = "default"
    access_policy: str = "internal"
    asof: Any = None


class ModelScoreResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    NO_LEGAL_ARTIFACT = "NO_LEGAL_ARTIFACT"


class NoLegalArtifactError(LookupError):
    """Production model scoring found no artifact legal at the requested as-of."""


@dataclass(frozen=True)
class ArtifactResolutionResult:
    status: ModelScoreResolutionStatus
    artifact: ModelArtifact | None = None
    record: ModelArtifactCatalogRecord | None = None


@dataclass
class ArtifactResolver:
    """Resolve the latest artifact legal for a request-scoped as-of context."""

    store: ArtifactStore
    catalog: ModelArtifactCatalog | None = None
    registry: dict[str, list[ModelArtifact]] = field(default_factory=dict)
    deployments: dict[str, list[Deployment]] = field(default_factory=dict)

    def register(
        self,
        artifact: ModelArtifact,
        *,
        context: ModelArtifactResolutionContext | None = None,
    ) -> None:
        if self.catalog is None:
            self.registry.setdefault(artifact.model_name, []).append(artifact)
            return
        ctx = context or ModelArtifactResolutionContext()
        if not self.store.base_dir:
            raise ValueError("catalog-backed registration requires ArtifactStore.base_dir")
        self.store.put(artifact)
        self.catalog.register(
            artifact,
            self.store.path_for(artifact.artifact_id),
            namespace=ctx.namespace,
            tenant=ctx.tenant,
            project=ctx.project,
            market=ctx.market,
            access_classification=ctx.access_policy,
        )

    def resolve_result(
        self,
        model_name: str,
        asof: Any,
        training_cutoff: Any = None,
        *,
        context: ModelArtifactResolutionContext | None = None,
    ) -> ArtifactResolutionResult:
        """Resolve artifact for production scoring with active deployment and certification filters."""
        if asof is None:
            raise ValueError("resolve_result requires a non-null asof timestamp")
        requested_asof = parse_catalog_timestamp(asof)
        requested_cutoff = (
            parse_catalog_timestamp(training_cutoff)
            if training_cutoff is not None
            else None
        )
        if self.catalog is not None:
            ctx = context or ModelArtifactResolutionContext(asof=asof)
            legal: list[ModelArtifactCatalogRecord] = []
            for record in self.catalog.snapshot(
                model_name=model_name,
                namespace=ctx.namespace,
                tenant=ctx.tenant,
                project=ctx.project,
                market=ctx.market,
            ):
                # Filter by time constraints only (promotion/certification checked in production path)
                if record.training_cutoff_timestamp > requested_asof:
                    continue
                if record.available_at_timestamp > requested_asof:
                    continue
                if requested_cutoff is not None and record.training_cutoff_timestamp > requested_cutoff:
                    continue
                legal.append(record)
            if not legal:
                return ArtifactResolutionResult(ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT)
            record = max(
                legal,
                key=lambda r: (
                    r.training_cutoff_timestamp,
                    r.available_at_timestamp,
                    r.semantic_version,
                    r.artifact_id,
                ),
            )
            artifact = self.store.get(record.artifact_id)
            if artifact is None:
                raise LookupError(
                    f"cataloged artifact {record.artifact_id!r} is missing from storage"
                )
            return ArtifactResolutionResult(
                ModelScoreResolutionStatus.RESOLVED, artifact, record
            )

        legal_artifacts: list[ModelArtifact] = []
        for artifact in self.registry.get(model_name, []):
            m = artifact.manifest
            cutoff = parse_catalog_timestamp(m.training_cutoff)
            available = parse_catalog_timestamp(m.available_at)
            if cutoff > requested_asof or available > requested_asof:
                continue
            if requested_cutoff is not None and cutoff > requested_cutoff:
                continue
            legal_artifacts.append(artifact)
        if not legal_artifacts:
            return ArtifactResolutionResult(ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT)
        artifact = max(
            legal_artifacts,
            key=lambda a: (
                parse_catalog_timestamp(a.manifest.training_cutoff),
                parse_catalog_timestamp(a.manifest.available_at),
                a.manifest.model_version,
                a.artifact_id,
            ),
        )
        return ArtifactResolutionResult(ModelScoreResolutionStatus.RESOLVED, artifact)

    def deploy(
        self,
        artifact: ModelArtifact,
        *,
        state: str = DeploymentState.CANARY,
        activation_at: Any = None,
        retired_at: Any = None,
        revoked_at: Any = None,
        canary_fraction: float = 0.0,
    ) -> Deployment:
        deployment = Deployment(
            artifact, state, activation_at, retired_at, revoked_at, canary_fraction
        )
        self.deployments.setdefault(artifact.model_name, []).append(deployment)
        if self.store.base_dir:
            self.store.put(artifact)
            self.store.save_deployments(self.deployments)
        return deployment

    def recover(self) -> None:
        """Recover persisted deployments after process restart."""
        self.deployments = self.store.load_deployments()

    def resolve(
        self,
        model_name: str,
        asof: Any,
        training_cutoff: Any = None,
        *,
        context: ModelArtifactResolutionContext | None = None,
        production: bool = False,
        max_stale_age_seconds: float | None = None,
    ) -> ModelArtifact | None:
        if production:
            candidates: list[tuple[Any, ModelArtifact]] = []
            for deployment in self.deployments.get(model_name, []):
                if deployment.state != DeploymentState.ACTIVE:
                    continue
                if deployment.activation_at is None or deployment.activation_at > asof:
                    continue
                if deployment.retired_at is not None and asof >= deployment.retired_at:
                    continue
                if deployment.revoked_at is not None and asof >= deployment.revoked_at:
                    continue
                artifact = deployment.artifact
                if not artifact.manifest.certification_hash:
                    continue
                if not artifact.is_legal_asof(asof, training_cutoff):
                    continue
                if max_stale_age_seconds is not None:
                    age = _age_seconds(asof, artifact.manifest.available_at)
                    if age > max_stale_age_seconds:
                        continue
                candidates.append((deployment.activation_at, artifact))
            if not candidates:
                return None
            return max(candidates, key=lambda item: (item[0], item[1].artifact_id))[1]

        return self.resolve_result(
            model_name, asof, training_cutoff, context=context
        ).artifact


def _age_seconds(asof: Any, available_at: Any) -> float:
    def _dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    return (_dt(asof) - _dt(available_at)).total_seconds()


class FrozenScorer:
    """Fit-free production scoring object."""

    __slots__ = ("_artifact",)

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact

    @property
    def artifact_id(self) -> str:
        return self._artifact.artifact_id

    def score(self, features: np.ndarray, context: PredictionContext) -> np.ndarray:
        return self._artifact.predict(np.asarray(features, dtype=np.float64), context=context)


class FrozenPredictor:
    """Immutable production predictor; deliberately has no ``fit`` method."""

    __slots__ = ("_artifact",)

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact

    def predict(self, features: np.ndarray, context: ScoringContext) -> np.ndarray:
        manifest = self._artifact.manifest
        if context.asof is None or not context.clock_id or not context.schema_hash:
            raise ValueError("production ScoringContext requires asof, clock_id, and schema_hash")
        if context.clock_id != manifest.decision_clock_id:
            raise ValueError("CLOCK_MISMATCH")
        if context.schema_hash != manifest.feature_schema_hash:
            raise ValueError("SCHEMA_MISMATCH")
        if not self._artifact.is_legal_asof(context.asof):
            raise ValueError("ARTIFACT_NOT_AVAILABLE_ASOF")
        if context.application_window is None or context.dates is None:
            raise ValueError("production ScoringContext requires application_window and dates")
        prediction_context = PredictionContext(
            application_window=context.application_window,
            dates=context.dates,
            asof=context.asof,
            feature_schema_hash=context.schema_hash,
        )
        return self._artifact.predict(
            np.asarray(features, dtype=np.float64), context=prediction_context
        )


_default_resolver: ArtifactResolver | None = None


def configure_default_resolver(resolver: ArtifactResolver | None) -> None:
    """Legacy single-process compatibility hook; request-scoped resolver preferred."""
    global _default_resolver
    _default_resolver = resolver


def identity_of(artifact: ModelArtifact) -> dict[str, Any]:
    m = artifact.manifest
    return {
        "model_name": m.model_name,
        "version": m.model_version,
        "artifact_id": m.artifact_id,
        "decision_clock_id": m.decision_clock_id,
        "label_contract_id": m.label_contract_id,
        "feature_schema_hash": m.feature_schema_hash,
        "data_source_hash": m.data_source_hash,
        "universe_hash": m.universe_hash,
        "training_cutoff": m.training_cutoff,
    }


@dataclass(frozen=True)
class ModelScoreExpr:
    model_name: str
    feature_names: tuple[str, ...] = ()
    resolution_policy: str = "latest_legal_asof"
    missing_artifact_policy: str = "fail_closed"

    def __post_init__(self) -> None:
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_score requires a non-empty model name")
        if self.resolution_policy != "latest_legal_asof":
            raise ValueError(f"unsupported artifact resolution policy {self.resolution_policy!r}")
        if self.missing_artifact_policy not in {"fail_closed", "nan"}:
            raise ValueError(
                "missing_artifact_policy must be 'fail_closed' or research-only 'nan'"
            )


@dataclass(frozen=True)
class ModelFeatureReadNode:
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactResolutionNode:
    model_name: str
    resolution_policy: str
    missing_artifact_policy: str


@dataclass(frozen=True)
class FrozenScoreNode:
    resolution: ArtifactResolutionNode
    features: ModelFeatureReadNode


@dataclass(frozen=True)
class TypedModelScoreIR:
    model_name: str
    resolution: ArtifactResolutionNode
    feature_read: ModelFeatureReadNode
    frozen_score: FrozenScoreNode
    output_semantic_kind: str = "ModelScoreBlock"

    @property
    def semantic_identity(self) -> Mapping[str, Any]:
        return {
            "op": "model_score",
            "model_name": self.model_name,
            "resolution_policy": self.resolution.resolution_policy,
            "missing_artifact_policy": self.resolution.missing_artifact_policy,
            "feature_names": self.feature_read.feature_names,
            "output_semantic_kind": self.output_semantic_kind,
        }


@dataclass(frozen=True)
class ModelScoreBlock:
    values: np.ndarray
    artifact_id: str
    model_name: str
    asof: Any
    cache_identity: tuple[Any, ...]


def model_score(
    model_name: str,
    *feature_names: str,
    missing_artifact_policy: str = "fail_closed",
) -> ModelScoreExpr:
    return ModelScoreExpr(
        model_name=model_name,
        feature_names=tuple(feature_names),
        missing_artifact_policy=missing_artifact_policy,
    )


def lower_model_score(expr: ModelScoreExpr) -> TypedModelScoreIR:
    resolution = ArtifactResolutionNode(
        model_name=expr.model_name,
        resolution_policy=expr.resolution_policy,
        missing_artifact_policy=expr.missing_artifact_policy,
    )
    features = ModelFeatureReadNode(expr.feature_names)
    return TypedModelScoreIR(
        model_name=expr.model_name,
        resolution=resolution,
        feature_read=features,
        frozen_score=FrozenScoreNode(resolution, features),
    )


def _score_block(
    ir: TypedModelScoreIR,
    features: np.ndarray,
    asof: Any,
    resolver: ArtifactResolver,
    *,
    training_cutoff: Any = None,
    context: ModelArtifactResolutionContext | None = None,
) -> ModelScoreBlock:
    if asof is None:
        raise ValueError("production scoring requires a non-null asof timestamp")
    result = resolver.resolve_result(
        ir.model_name,
        asof,
        training_cutoff,
        context=context,
    )
    if result.status is ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT:
        if ir.resolution.missing_artifact_policy == "nan":
            values = np.full(np.asarray(features).shape[0], np.nan, dtype=np.float64)
            return ModelScoreBlock(
                values, "", ir.model_name, asof,
                (ir.model_name, str(asof), "NO_LEGAL_ARTIFACT"),
            )
        raise NoLegalArtifactError(
            f"NO_LEGAL_ARTIFACT: model {ir.model_name!r} at {asof!r}"
        )
    if result.artifact is None:
        raise RuntimeError("resolved artifact result is missing its artifact payload")
    artifact = result.artifact
    prediction_context = PredictionContext(
        application_window=ApplicationWindow(start=asof, end=asof),
        dates=np.full(len(np.asarray(features)), asof, dtype=object),
        asof=asof,
        feature_schema_hash=artifact.manifest.feature_schema_hash,
    )
    values = FrozenScorer(artifact).score(features, prediction_context)
    m = artifact.manifest
    return ModelScoreBlock(
        values=values,
        artifact_id=artifact.artifact_id,
        model_name=ir.model_name,
        asof=asof,
        cache_identity=(
            ir.model_name,
            str(asof),
            artifact.artifact_id,
            m.training_cutoff,
            m.feature_schema_hash,
            m.data_source_hash,
            m.universe_hash,
            m.decision_clock_id,
        ),
    )


def score_asof(
    model_name: str,
    features: np.ndarray,
    asof: Any,
    *,
    training_cutoff: Any = None,
    resolver: ArtifactResolver | None = None,
    context: ModelArtifactResolutionContext | None = None,
    scoring_context: ScoringContext | None = None,
) -> np.ndarray:
    res = resolver if resolver is not None else _default_resolver
    if res is None:
        raise LookupError("no ArtifactResolver configured for score_asof")

    if scoring_context is not None and scoring_context.production:
        artifact = res.resolve(
            model_name, asof, training_cutoff,
            production=True,
            max_stale_age_seconds=scoring_context.max_stale_age_seconds,
        )
        if artifact is None:
            raise LookupError(
                f"no legal as-of artifact for model {model_name!r} at {asof!r}"
            )
        return FrozenPredictor(artifact).predict(
            np.asarray(features, dtype=np.float64),
            scoring_context,
        )

    ir = lower_model_score(model_score(model_name))
    return _score_block(
        ir,
        features,
        asof,
        res,
        training_cutoff=training_cutoff,
        context=context,
    ).values


def replay_historical_scores(
    model_name: str,
    features_by_date: dict[Any, np.ndarray],
    resolver: ArtifactResolver,
) -> dict[Any, np.ndarray]:
    ir = lower_model_score(model_score(model_name))
    out: dict[Any, np.ndarray] = {}
    for date in sorted(features_by_date):
        out[date] = _score_block(
            ir, features_by_date[date], date, resolver
        ).values
    return out


ModelScoreOperatorStub = ModelScoreExpr
