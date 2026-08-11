# -*- coding: utf-8 -*-
"""Factor-DSL as-of artifact bridge (Model Layer Major Redesign taskbook
§49 / §50 / §51 / §22 / §13.12).

Resolves a ``model_score("name")`` DSL expression to a **legal as-of**
:class:`modeling.artifact.ModelArtifact`.  The bridge is additive and
self-contained — it does not edit the DSL compiler or ``api/mining_integration``.

* §50  — an artifact is legal at ``asof`` only when its ``training_cutoff <=
  asof`` AND its ``available_at <= asof``; among legal artifacts the LATEST
  (largest ``training_cutoff``) is selected.
* §13.12 / §51 — a future-trained artifact is NEVER returned for history.
* §22  — scoring goes through the frozen artifact path (``artifact.predict``);
  ``fit`` is never invoked during scoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modeling.artifact import ModelArtifact
from modeling.learners.base import LearnerSpec, get_learner

__all__ = [
    "ArtifactStore",
    "ArtifactResolver",
    "score_asof",
    "replay_historical_scores",
    "ModelScoreOperatorStub",
    "identity_of",
]


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Artifact store (§49) — persistence for ModelArtifact JSON.
# --------------------------------------------------------------------------- #
@dataclass
class ArtifactStore:
    """JSON artifact store.  ``put``/``get`` round-trip a
    :class:`~modeling.artifact.ModelArtifact` through ``artifact.to_dict()``.
    """

    base_dir: str | None = None

    def put(self, artifact: ModelArtifact) -> str:
        """Serialize ``artifact`` to ``base_dir/<artifact_id>.json``; return id."""
        if not self.base_dir:
            raise ValueError("ArtifactStore.base_dir must be set before put()")
        os.makedirs(self.base_dir, exist_ok=True)
        path = os.path.join(self.base_dir, f"{artifact.artifact_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(artifact.to_dict(), fh, indent=1, default=_json_default)
        return artifact.artifact_id

    def get(self, artifact_id: str) -> ModelArtifact | None:
        """Load an artifact by id, rebuilding its learner from the registry."""
        if not self.base_dir:
            return None
        path = os.path.join(self.base_dir, f"{artifact_id}.json")
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


# --------------------------------------------------------------------------- #
# As-of resolver (§50 / §13.12).
# --------------------------------------------------------------------------- #
@dataclass
class ArtifactResolver:
    """Registry of artifacts per ``model_name`` + as-of resolution.

    ``resolve`` picks the LATEST artifact whose ``training_cutoff <= asof`` AND
    ``available_at <= asof``.  It NEVER returns a future-trained artifact for a
    historical as-of (``training_cutoff > asof`` is rejected, §13.12 / §51).
    """

    store: ArtifactStore
    registry: dict[str, list[ModelArtifact]] = field(default_factory=dict)

    def register(self, artifact: ModelArtifact) -> None:
        self.registry.setdefault(artifact.model_name, []).append(artifact)

    def resolve(
        self,
        model_name: str,
        asof: Any,
        training_cutoff: Any = None,
    ) -> ModelArtifact | None:
        legal: list[ModelArtifact] = []
        for art in self.registry.get(model_name, []):
            if not art.is_legal_asof(asof, training_cutoff):
                continue
            # §50 / §13.12: never a future-trained artifact.
            if art.manifest.training_cutoff and asof is not None:
                if asof < art.manifest.training_cutoff:
                    continue
            legal.append(art)
        if not legal:
            return None
        legal.sort(
            key=lambda a: (
                a.manifest.training_cutoff,
                a.manifest.available_at,
                a.manifest.model_version,
            )
        )
        return legal[-1]


# --------------------------------------------------------------------------- #
# Score entry points (§22 / §51).
# --------------------------------------------------------------------------- #
_default_resolver: ArtifactResolver | None = None


def configure_default_resolver(resolver: ArtifactResolver | None) -> None:
    """Install the process-wide default resolver used by :func:`score_asof`."""
    global _default_resolver
    _default_resolver = resolver


def score_asof(
    model_name: str,
    features: np.ndarray,
    asof: Any,
    *,
    training_cutoff: Any = None,
    resolver: ArtifactResolver | None = None,
) -> np.ndarray:
    """Score ``features`` with the artifact legal at ``asof``.

    Resolves via §50 and predicts through the frozen artifact path (§22).
    Raises ``LookupError`` when no legal artifact exists.
    """
    res = resolver if resolver is not None else _default_resolver
    if res is None:
        raise LookupError("no ArtifactResolver configured for score_asof")
    artifact = res.resolve(model_name, asof, training_cutoff)
    if artifact is None:
        raise LookupError(
            f"no legal as-of artifact for model {model_name!r} at {asof!r}"
        )
    return artifact.predict(np.asarray(features, dtype=np.float64))


def replay_historical_scores(
    model_name: str,
    features_by_date: dict[Any, np.ndarray],
    resolver: ArtifactResolver,
) -> dict[Any, np.ndarray]:
    """§51 — score each date with the artifact legal that date.

    Asserts (fail closed) that the resolved artifact was not trained in the
    future relative to the scored date.
    """
    out: dict[Any, np.ndarray] = {}
    for date in sorted(features_by_date):
        artifact = resolver.resolve(model_name, date)
        if artifact is None:
            raise LookupError(
                f"no legal as-of artifact for model {model_name!r} at {date!r}"
            )
        tc = artifact.manifest.training_cutoff
        if tc and date is not None and date < tc:
            raise AssertionError(
                f"replay leak: artifact trained at {tc!r} used for {date!r} "
                "(future-trained artifact must never score history)"
            )
        out[date] = artifact.predict(
            np.asarray(features_by_date[date], dtype=np.float64)
        )
    return out


# --------------------------------------------------------------------------- #
# DSL semantic identity (§49 / §22).
# --------------------------------------------------------------------------- #
def identity_of(artifact: ModelArtifact) -> dict[str, Any]:
    """The lineage facts every ``model_score(...)`` node must carry (§22)."""
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


@dataclass
class ModelScoreOperatorStub:
    """Placeholder documenting the DSL ``model_score("name")`` node contract.

    A ``model_score("name")`` expression resolves to :func:`score_asof`: the
    DSL compiler emits a node whose ``semantic_identity`` carries every lineage
    fact §22 requires (via :func:`identity_of`).  This stub is documentation of
    that contract; the DSL-compiler integration is a separate task and does not
    import or edit this module.
    """

    model_name: str
    artifact: ModelArtifact | None = None
    asof: Any = None
    training_cutoff: Any = None

    @property
    def semantic_identity(self) -> dict[str, Any]:
        if self.artifact is not None:
            return identity_of(self.artifact)
        return {
            "model_name": self.model_name,
            "version": "",
            "artifact_id": "",
            "decision_clock_id": "",
            "label_contract_id": "",
            "feature_schema_hash": "",
            "data_source_hash": "",
            "universe_hash": "",
            "training_cutoff": self.training_cutoff,
        }

    def resolve(self, features: np.ndarray) -> np.ndarray:
        """Score this node exactly as the DSL would: via :func:`score_asof`."""
        return score_asof(
            self.model_name,
            features,
            self.asof,
            training_cutoff=self.training_cutoff,
        )
