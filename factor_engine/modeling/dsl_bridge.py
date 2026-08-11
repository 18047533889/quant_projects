# -*- coding: utf-8 -*-
"""Model-score DSL, typed IR, artifact resolution, and frozen scoring."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from modeling.artifact import ModelArtifact
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
    "ModelArtifactResolutionContext",
    "ModelScoreExpr",
    "TypedModelScoreIR",
    "ArtifactResolutionNode",
    "ModelFeatureReadNode",
    "FrozenScoreNode",
    "ModelScoreBlock",
    "ModelScoreResolutionStatus",
    "NoLegalArtifactError",
    "FrozenScorer",
    "model_score",
    "lower_model_score",
    "score_asof",
    "replay_historical_scores",
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
        path = self.path_for(artifact.artifact_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(artifact.to_dict(), fh, indent=1, default=_json_default)
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
    """Resolve the latest artifact legal for a request-scoped as-of context.

    A durable ``ModelArtifactCatalog`` is authoritative when configured.  The
    in-memory registry remains as a compatibility path for existing tests and
    local research callers.
    """

    store: ArtifactStore
    catalog: ModelArtifactCatalog | None = None
    registry: dict[str, list[ModelArtifact]] = field(default_factory=dict)

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

        legal: list[ModelArtifact] = []
        for artifact in self.registry.get(model_name, []):
            m = artifact.manifest
            cutoff = parse_catalog_timestamp(m.training_cutoff)
            available = parse_catalog_timestamp(m.available_at)
            if cutoff > requested_asof or available > requested_asof:
                continue
            if requested_cutoff is not None and cutoff > requested_cutoff:
                continue
            legal.append(artifact)
        if not legal:
            return ArtifactResolutionResult(ModelScoreResolutionStatus.NO_LEGAL_ARTIFACT)
        artifact = max(
            legal,
            key=lambda a: (
                parse_catalog_timestamp(a.manifest.training_cutoff),
                parse_catalog_timestamp(a.manifest.available_at),
                a.manifest.model_version,
                a.artifact_id,
            ),
        )
        return ArtifactResolutionResult(ModelScoreResolutionStatus.RESOLVED, artifact)

    def resolve(
        self,
        model_name: str,
        asof: Any,
        training_cutoff: Any = None,
        *,
        context: ModelArtifactResolutionContext | None = None,
    ) -> ModelArtifact | None:
        return self.resolve_result(
            model_name, asof, training_cutoff, context=context
        ).artifact


class FrozenScorer:
    """Fit-free production scoring object.

    This type deliberately exposes only ``score``.  Training objects and their
    mutable ``fit`` surface never appear in the scoring API.
    """

    __slots__ = ("_artifact",)

    def __init__(self, artifact: ModelArtifact) -> None:
        self._artifact = artifact

    @property
    def artifact_id(self) -> str:
        return self._artifact.artifact_id

    def score(self, features: np.ndarray) -> np.ndarray:
        return self._artifact.predict(np.asarray(features, dtype=np.float64))


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
    values = FrozenScorer(artifact).score(features)
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
) -> np.ndarray:
    res = resolver if resolver is not None else _default_resolver
    if res is None:
        raise LookupError("no ArtifactResolver configured for score_asof")
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


# Compatibility alias for callers that imported the documented stub name.
ModelScoreOperatorStub = ModelScoreExpr
