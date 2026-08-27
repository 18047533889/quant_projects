"""Similarity fingerprint + ANN index artifacts for factor_assets (DLIB-FA-006/52).

- :class:`SimilarityFingerprintArtifact` — the embedding fingerprint of a factor
  that feeds the incremental pipeline: Fingerprint -> ANN shortlist -> Exact
  MultiView Similarity -> RefinedPairArtifact -> Sparse Similarity Graph Update.
- :class:`ANNIndexArtifact` — records the ANN backend capability / index params /
  seed / recall benchmark / version so a production index is reproducible and
  auditable.  For 100k production a real approximate backend (HNSW/IVF/IVFPQ/
  Annoy) is required; a flat exact index is ``EXACT_FLAT``, not ANN.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from factor_assets.contracts._canonical import canonical_digest

__all__ = [
    "SimilarityFingerprintArtifact",
    "ANNIndexArtifact",
    "ANNIndexCapability",
]


class ANNIndexCapability(Enum):
    """Capability of a search index (DLIB-FA-052).

    An index that answers queries EXACTLY (brute-force over every stored
    vector) is NOT an approximate-nearest-neighbour (ANN) index, even when it
    is backed by the ``faiss`` library.  For 100k production a real approximate
    backend (HNSW/IVF/IVFPQ/Annoy) is required.
    """

    EXACT_FLAT = "EXACT_FLAT"
    APPROXIMATE = "APPROXIMATE"


@dataclass(frozen=True)
class SimilarityFingerprintArtifact:
    """Embedding fingerprint of a factor (DLIB-FA-006).

    ``content_hash`` is derived-only over the factor id, embedding, embedding
    spec, snapshot, universe, and window.  The fingerprint feeds the
    incremental pipeline: Fingerprint -> ANN shortlist -> Exact MultiView
    Similarity -> RefinedPairArtifact -> Sparse Similarity Graph Update.
    """

    factor_id: str
    embedding: tuple[float, ...]
    embedding_spec: str
    snapshot: str
    universe: str
    window: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.embedding:
            raise ValueError("embedding cannot be empty")
        if not self.embedding_spec:
            raise ValueError("embedding_spec is required")
        if not self.snapshot:
            raise ValueError("snapshot is required")
        if not self.universe:
            raise ValueError("universe is required")
        if not self.window:
            raise ValueError("window is required")
        for v in self.embedding:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError("embedding must contain non-boolean numbers")
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError("embedding must contain finite values")
        object.__setattr__(self, "embedding", tuple(self.embedding))
        computed = canonical_digest(
            self.factor_id,
            self.embedding,
            self.embedding_spec,
            self.snapshot,
            self.universe,
            self.window,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed fingerprint content "
                "hash; a caller may not self-report an arbitrary hash — FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "embedding": list(self.embedding),
            "embedding_spec": self.embedding_spec,
            "snapshot": self.snapshot,
            "universe": self.universe,
            "window": self.window,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ANNIndexArtifact:
    """Reproducible, auditable ANN index record (DLIB-FA-052).

    Records the backend capability, index params, seed, recall benchmark, and
    version.  For 100k production a real approximate backend (HNSW/IVF/IVFPQ/
    Annoy) is required; a flat exact index is ``EXACT_FLAT``, not ANN.
    """

    index_id: str
    backend: str
    capability: ANNIndexCapability
    index_params: Mapping[str, object]
    seed: Optional[int] = None
    recall_benchmark: Optional[float] = None
    version: str = "1.0"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.index_id:
            raise ValueError("index_id is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not isinstance(self.capability, ANNIndexCapability):
            raise TypeError("capability must be an ANNIndexCapability")
        if not isinstance(self.index_params, Mapping):
            raise TypeError("index_params must be a mapping")
        object.__setattr__(self, "index_params", MappingProxyType(dict(self.index_params)))
        if self.recall_benchmark is not None:
            if isinstance(self.recall_benchmark, bool) or not isinstance(
                self.recall_benchmark, (int, float)
            ):
                raise TypeError("recall_benchmark must be a non-boolean number or None")
            rb = float(self.recall_benchmark)
            if rb != rb or rb in (float("inf"), float("-inf")):
                raise ValueError("recall_benchmark must be finite")
            if not 0.0 <= rb <= 1.0:
                raise ValueError("recall_benchmark must be in [0, 1]")
            object.__setattr__(self, "recall_benchmark", rb)
        computed = canonical_digest(
            self.index_id,
            self.backend,
            self.capability.value,
            self.index_params,
            self.seed,
            self.recall_benchmark,
            self.version,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed ANN index content "
                "hash; a caller may not self-report an arbitrary hash — FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "index_id": self.index_id,
            "backend": self.backend,
            "capability": self.capability.value,
            "index_params": dict(self.index_params),
            "seed": self.seed,
            "recall_benchmark": self.recall_benchmark,
            "version": self.version,
            "content_hash": self.content_hash,
        }
