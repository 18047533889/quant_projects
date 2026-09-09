"""Incremental cluster assignment for factor_assets (QRP-P6-INC6 / DLIB-FA-010).

The production pipeline is: full ClusterSetVersion refresh (week/month) +
DAILY incremental assignment of NEW candidate factors into the EXISTING
cluster family, WITHOUT re-clustering the whole universe and WITHOUT mutating
the immutable production ``ClusterVersionArtifact`` / ``ClusterSetVersion``.

This module owns that daily path:

- :func:`incremental_assign` — fingerprinted new factor  ->  closest existing
  logical cluster, record an IMMUTABLE :class:`IncrementalClusterAssignment`.
- :func:`build_incremental_cluster_version` — copy-on-write: produce a NEW
  ``ClusterVersionArtifact`` (overlay) for each touched logical cluster at a
  NEW ``cluster_set_version_ref``; the production version artifact (deep-frozen,
  content-hash verified) is untouched.
- :func:`build_incremental_lineage_edges` — lineage edge per touched cluster
  ``(parent_cluster_version_ref -> new_cluster_version_ref)`` carrying a
  ``requested_by`` reason and the new-factor members.
- ANN acceleration — when the fingerprint carries an :class:`ANNIndexArtifact`
  whose capability is APPROXIMATE, use :class:`AnnoyANNIndex` against the
  members' stored embeddings; otherwise exact brute-force cosine.
- Unknown pair handling — a pair with NO measured similarity is NEVER treated
  as a computed zero (DLIB-FA-007): it is excluded from candidate ranking and
  a factor with no meaningful family affinity is honestly recorded as
  ``PENDING_GLOBAL_REFRESH`` (never silently "assigned" nowhere by silence).
- ``min_cluster_size`` in the merge policy is re-applied via
  :func:`_small_cluster_merge_policy_apply`; a disallowed small cluster can
  only be redirected to another cluster backed by that target's own qualified
  affinity evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterSetVersionArtifact,
    ClusterVersionArtifact,
    IncrementalAssignmentKind,
    IncrementalClusterAssignment,
    LogicalCluster,
)
from factor_assets.contracts.fingerprint import (
    ANNIndexArtifact,
    ANNIndexCapability,
    SimilarityFingerprintArtifact,
)
from factor_assets.contracts._canonical import canonical_digest

try:
    from factor_assets.similarity.ann import (  # noqa: F401
        AnnoyANNIndex,
        ANNBackend,
        IndexCapability,
    )
    ANN_IMPORTS_OK = True
except Exception:  # pragma: no cover
    ANN_IMPORTS_OK = False

try:
    from factor_assets.similarity.ann import AnnoyANNIndex as _Annoy  # noqa: F401
    import annoy  # noqa: F401

    ANNOY_AVAILABLE = True
except Exception:  # pragma: no cover
    ANNOY_AVAILABLE = False

__all__ = [
    "UNKNOWN_AFFINITY_FLOOR",
    "DEFAULT_MAX_CANDIDATES",
    "IncrementalPolicy",
    "PairwiseEvidenceStatus",
    "CertifiedPairwiseEvidence",
    "IncrementalCandidate",
    "IncrementalAssignResult",
    "IncrementalLineageEdge",
    "incremental_assign",
    "build_incremental_cluster_version",
    "build_incremental_lineage_edges",
]

#: The semantic floor for a measured affinity to count as "measured".  NEVER
#: ``0.0``: an unmeasured pair is not "same family" (DLIB-FA-007) and punting
#: to ``0`` would silence an unknown as if it were "dis-similar".  When no
#: pair at all is measurable the assignment is PENDING_GLOBAL_REFRESH (unknown
#: is preserved, never zero).
UNKNOWN_AFFINITY_FLOOR = 0.25

DEFAULT_MAX_CANDIDATES = 32


class PairwiseEvidenceStatus(Enum):
    MEASURED_LOW = "MEASURED_LOW"
    UNMEASURED = "UNMEASURED"
    APPROXIMATE = "APPROXIMATE"
    CERTIFIED = "CERTIFIED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class CertifiedWindowEvidence:
    window_ref: str
    signed_similarity: Optional[float]
    status: PairwiseEvidenceStatus
    pair_count: int
    n_days: int
    confidence_interval: Tuple[Optional[float], Optional[float]]
    uncertainty_scale: str


@dataclass(frozen=True)
class CertifiedPairwiseEvidence:
    """FA-side bounded view of QE pairwise evidence; contains no raw values."""

    factor_id_a: str
    factor_id_b: str
    signed_similarity: Optional[float]
    status: PairwiseEvidenceStatus
    pair_count: int
    window_ref: str
    universe_ref: str
    sample_ref: str
    windows: Tuple[CertifiedWindowEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id_a or not self.factor_id_b or self.factor_id_a == self.factor_id_b:
            raise ValueError("pairwise evidence requires two distinct factor IDs")
        if not isinstance(self.status, PairwiseEvidenceStatus):
            raise TypeError("status must be a PairwiseEvidenceStatus")
        if self.pair_count < 0:
            raise ValueError("pair_count must be non-negative")
        if not self.window_ref or not self.universe_ref or not self.sample_ref:
            raise ValueError("window_ref, universe_ref, and sample_ref are required")
        if self.status in (PairwiseEvidenceStatus.CERTIFIED, PairwiseEvidenceStatus.MEASURED_LOW):
            if self.signed_similarity is None or not np.isfinite(self.signed_similarity):
                raise ValueError("measured pairwise evidence requires finite signed_similarity")
            if not -1.0 <= float(self.signed_similarity) <= 1.0:
                raise ValueError("signed_similarity must be in [-1, 1]")
            if self.pair_count < 1:
                raise ValueError("measured pairwise evidence requires pair_count >= 1")
        elif self.signed_similarity is not None:
            raise ValueError("unmeasured/approximate evidence cannot carry certified similarity")
        object.__setattr__(self, "windows", tuple(self.windows))


@dataclass(frozen=True)
class IncrementalPolicy:
    """Immutable policy knobs for incremental assignment (QRP-P6-INC6).

    All knobs are validated at construction so a typo fails closed instead of
    silently changing assignment behaviour.
    """

    #: Affinity threshold (cosine similarity, in [-1,1]) that gates
    #: ASSIGNED vs PENDING_GLOBAL_REFRESH.
    affinity_threshold: float = 0.5
    #: If the top candidate's affinity is within ``ambiguity_gap`` of the
    #: runner-up, the assignment is AMBIGUOUS (deferred to Global Refresh fold,
    #: never guessed).
    ambiguity_gap: float = 0.05
    #: Only the top ``max_candidates`` affinity candidates are considered.
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    #: A candidate affinity below this floor is treated as "not meaningfully
    #: measured" (UNKNOWN), never a computed zero.
    min_measure_floor: float = UNKNOWN_AFFINITY_FLOOR
    min_pair_count: int = 30
    min_window_days: int = 20
    min_certified_windows: int = 1
    cluster_support_k: int = 3
    min_cluster_support: int = 2
    require_medoid_support: bool = True

    def __post_init__(self) -> None:
        for name in ("affinity_threshold", "ambiguity_gap", "min_measure_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a non-boolean number")
            fv = float(value)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.affinity_threshold) <= 1.0:
            raise ValueError("affinity_threshold must be in [0, 1]")
        if float(self.ambiguity_gap) < 0.0:
            raise ValueError("ambiguity_gap must be non-negative")
        if isinstance(self.max_candidates, bool) or not isinstance(
            self.max_candidates, int
        ):
            raise TypeError("max_candidates must be an int")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        for name in ("min_pair_count", "min_window_days", "min_certified_windows",
                     "cluster_support_k", "min_cluster_support"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if self.min_cluster_support > self.cluster_support_k:
            raise ValueError("min_cluster_support cannot exceed cluster_support_k")
        if not isinstance(self.require_medoid_support, bool):
            raise TypeError("require_medoid_support must be bool")


@dataclass(frozen=True)
class IncrementalCandidate:
    """A single affinity measurement between the new factor and an existing
    member factor of a logical cluster (never an unmeasured zero)."""

    factor_id: str
    similarity: float
    cluster_id: str
    evidence_status: PairwiseEvidenceStatus = PairwiseEvidenceStatus.APPROXIMATE
    pair_count: int = 0
    window_ref: Optional[str] = None
    sample_ref: Optional[str] = None
    signed_similarity: Optional[float] = None
    confidence_interval: Tuple[Optional[float], Optional[float]] = (None, None)
    support_count: int = 1
    support_member_ids: Tuple[str, ...] = ()
    medoid_supported: Optional[bool] = None
    aggregation_method: str = "member_affinity"
    rejection_reason: Optional[str] = None
    support_confidence_intervals: Tuple[Tuple[Optional[float], Optional[float]], ...] = ()
    support_evidence_refs: Tuple[Tuple[str, Optional[str], Optional[str]], ...] = ()

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.cluster_id:
            raise ValueError("cluster_id is required")
        if isinstance(self.similarity, bool) or not isinstance(
            self.similarity, (int, float)
        ):
            raise TypeError("similarity must be a non-boolean number")
        sim = float(self.similarity)
        if sim != sim or sim in (float("inf"), float("-inf")):
            raise ValueError("similarity must be finite")
        if not -1.0 <= sim <= 1.0:
            raise ValueError("similarity must be in [-1, 1]")
        object.__setattr__(self, "similarity", sim)
        if not isinstance(self.evidence_status, PairwiseEvidenceStatus):
            raise TypeError("evidence_status must be a PairwiseEvidenceStatus")
        if self.pair_count < 0:
            raise ValueError("pair_count must be non-negative")
        if isinstance(self.support_count, bool) or self.support_count < 0:
            raise ValueError("support_count must be a non-negative integer")
        object.__setattr__(self, "support_member_ids", tuple(self.support_member_ids))
        object.__setattr__(self, "support_confidence_intervals",
                           tuple(self.support_confidence_intervals))
        object.__setattr__(self, "support_evidence_refs", tuple(self.support_evidence_refs))
        if self.evidence_status is PairwiseEvidenceStatus.CERTIFIED:
            if self.pair_count < 1 or not self.window_ref or not self.sample_ref:
                raise ValueError("CERTIFIED candidate requires count/window/sample evidence")
            if self.signed_similarity is None or abs(float(self.signed_similarity)) != sim:
                raise ValueError("candidate affinity must equal abs(signed_similarity)")

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "similarity": self.similarity,
            "cluster_id": self.cluster_id,
            "evidence_status": self.evidence_status.value,
            "pair_count": self.pair_count,
            "window_ref": self.window_ref,
            "sample_ref": self.sample_ref,
            "signed_similarity": self.signed_similarity,
            "confidence_interval": self.confidence_interval,
            "support_count": self.support_count,
            "support_member_ids": list(self.support_member_ids),
            "medoid_supported": self.medoid_supported,
            "aggregation_method": self.aggregation_method,
            "rejection_reason": self.rejection_reason,
            "support_confidence_intervals": self.support_confidence_intervals,
            "support_evidence_refs": self.support_evidence_refs,
        }


@dataclass(frozen=True)
class IncrementalAssignResult:
    """Outcome of an incremental assignment batch (QRP-P6-INC6).

    ``assignments`` contains one :class:`IncrementalClusterAssignment` per
    submitted fingerprint (with the batch id + created_at provenance).
    ``candidates`` preserves the per-cluster top measured affinities for audit.
    """

    batch_id: str
    assignments: Tuple[IncrementalClusterAssignment, ...]
    candidates: Tuple[IncrementalCandidate, ...]
    created_at: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id is required")
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            )
        computed = canonical_digest(
            self.batch_id,
            tuple(a.to_dict() for a in self.assignments),
            tuple(c.to_dict() for c in self.candidates),
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed incremental "
                "assignment-batch content hash; a caller may not self-report an "
                "arbitrary hash — FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "assignments": [a.to_dict() for a in self.assignments],
            "candidates": [c.to_dict() for c in self.candidates],
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }


def _as_touchable(kind: IncrementalAssignmentKind) -> bool:
    return kind is IncrementalAssignmentKind.ASSIGNED


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------


def _to_embedding(fingerprint: SimilarityFingerprintArtifact) -> np.ndarray:
    return np.asarray(list(fingerprint.embedding), dtype=np.float64)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(
            "zero / non-finite embeddings are not admissible for cosine "
            "similarity assignment — FAIL CLOSED"
        )
    return matrix / norms[:, np.newaxis]


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of a (d,) query against an (n, d) matrix."""
    qn = _normalize_rows(query.reshape(1, -1))[0]
    mn = _normalize_rows(matrix)
    return qn @ mn.T


def _collect_member_embeddings(
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    member_ids: Sequence[str],
) -> tuple[list[str], Optional[np.ndarray]]:
    """(present_ids, matrix) of member fingerprints present in the map.

    Members with a missing fingerprint are skipped (never measured as zero).
    """
    rows: list[str] = []
    vectors: list[np.ndarray] = []
    for fid in member_ids:
        fp = fingerprints_by_id.get(fid)
        if fp is None:
            continue
        if fp.factor_id != fid:
            raise ValueError("fingerprint mapping key must match fingerprint.factor_id")
        rows.append(fid)
        vectors.append(_to_embedding(fp))
    if not vectors:
        return [], None
    return rows, np.vstack(vectors)


def _fingerprint_domain(fp: SimilarityFingerprintArtifact) -> tuple[str, ...]:
    """Semantic measurement domain in which embedding geometry is comparable."""
    return (
        fp.embedding_spec,
        fp.snapshot,
        fp.universe,
        fp.window,
        fp.preprocessing_ref,
        fp.mask_policy,
        fp.direction,
        fp.aggregation_method,
        fp.embedding_model_version,
    )


def _validate_fingerprint_domains(
    queries: Sequence[SimilarityFingerprintArtifact],
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    member_ids: set[str],
) -> None:
    for key, fp in fingerprints_by_id.items():
        if key != fp.factor_id:
            raise ValueError("fingerprint mapping key must match fingerprint.factor_id")
    for query in queries:
        bound = fingerprints_by_id.get(query.factor_id)
        if bound is None or bound.content_hash != query.content_hash:
            raise ValueError("query fingerprint must be content-bound in fingerprints_by_id")
        query_domain = _fingerprint_domain(query)
        for member_id in member_ids:
            member = fingerprints_by_id.get(member_id)
            if member is None:
                continue
            if _fingerprint_domain(member) != query_domain:
                raise ValueError(
                    "incompatible fingerprint semantic domains: embedding spec, "
                    "snapshot, universe, window, and preprocessing identity must match"
                )


def _validate_ann_index(ann_index: ANNIndexArtifact, dim: int) -> None:
    if not isinstance(ann_index, ANNIndexArtifact):
        raise TypeError("ann_index must be an ANNIndexArtifact")
    if not isinstance(ann_index.capability, ANNIndexCapability):
        raise TypeError(
            "ann_index.capability must be an ANNIndexArtifact.ANNIndexCapability"
        )
    declared = ann_index.index_params.get("embedding_dim")
    if declared is not None and isinstance(declared, (int, float)) and not isinstance(
        declared, bool
    ):
        if int(declared) != dim:
            raise ValueError(
                f"ann_index embedding_dim {int(declared)} does not match the "
                f"stored embedding dim {dim}; a mismatched index must not be used"
            )


def _validate_ann_membership(
    ann_index: ANNIndexArtifact,
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    member_ids: Sequence[str],
) -> None:
    ordered = tuple(sorted(member_ids))
    expected_members = canonical_digest(tuple(
        (factor_id, fingerprints_by_id[factor_id].content_hash)
        for factor_id in ordered
    ))
    domains = {_fingerprint_domain(fingerprints_by_id[factor_id]) for factor_id in ordered}
    if len(domains) != 1:
        raise ValueError("ANN members span incompatible fingerprint semantic domains")
    expected_spec = canonical_digest(next(iter(domains)))
    if ann_index.member_set_hash != expected_members:
        raise ValueError("ANN member_set_hash does not match actual index membership")
    if ann_index.embedding_spec_hash != expected_spec:
        raise ValueError("ANN embedding_spec_hash does not match actual fingerprint domain")


def _select_exact(
    query: np.ndarray,
    fingerprint_map: Mapping[str, SimilarityFingerprintArtifact],
    member_ids: Sequence[str],
    cluster_id: str,
    floor: float,
) -> list[IncrementalCandidate]:
    present, matrix = _collect_member_embeddings(fingerprint_map, member_ids)
    if not present or matrix is None:
        return []
    scores = _cosine_scores(query, matrix)
    out: list[IncrementalCandidate] = []
    for fid, sim in zip(present, scores):
        s = float(sim)
        status = (
            PairwiseEvidenceStatus.MEASURED_LOW
            if s < floor else PairwiseEvidenceStatus.APPROXIMATE
        )
        out.append(IncrementalCandidate(
            factor_id=fid, similarity=s, cluster_id=cluster_id,
            evidence_status=status,
        ))
    out.sort(key=lambda c: c.similarity, reverse=True)
    return out


def _select_ann(
    query: np.ndarray,
    fingerprint_map: Mapping[str, SimilarityFingerprintArtifact],
    member_ids: Sequence[str],
    cluster_id: str,
    ann_index: ANNIndexArtifact,
    max_candidates: int,
    floor: float,
) -> list[IncrementalCandidate]:
    """Approximate nearest-neighbour selection via Annoy.

    An ``EXACT_FLAT`` index is NOT ANN (DLIB-FA-052): if the supplied artifact
    carries that capability we refuse to call it ANN and fall back to exact.
    Any Annoy runtime failure also falls back to exact so the semantic core
    (family affinity) is never silently dropped.
    """
    present, matrix = _collect_member_embeddings(fingerprint_map, member_ids)
    if not present or matrix is None:
        return []
    _validate_ann_index(ann_index, matrix.shape[1])
    if ann_index.capability is not ANNIndexCapability.APPROXIMATE:
        return _select_exact(query, fingerprint_map, present, cluster_id, floor)
    try:
        n_trees = int(ann_index.index_params.get("n_trees", 10))
        idx = AnnoyANNIndex(embedding_dim=matrix.shape[1], n_trees=n_trees)
        idx.build(list(present), np.ascontiguousarray(matrix, dtype=np.float32))
        results = idx.search(
            np.ascontiguousarray(query, dtype=np.float32),
            k=max_candidates,
            min_similarity=floor,
        )
    except Exception:
        return _select_exact(query, fingerprint_map, present, cluster_id, floor)
    out: list[IncrementalCandidate] = []
    for r in results:
        if r.similarity_score is None or float(r.similarity_score) < floor:
            continue
        out.append(IncrementalCandidate(factor_id=r.factor_id, similarity=float(r.similarity_score), cluster_id=cluster_id))
    out.sort(key=lambda c: c.similarity, reverse=True)
    return out


# ---------------------------------------------------------------------------
# policy helpers
# ---------------------------------------------------------------------------


def _small_cluster_merge_policy_apply(
    chosen_cluster_id: Optional[str],
    cluster_sizes: Mapping[str, int],
    min_cluster_size: int,
) -> Optional[str]:
    """Re-apply the families.py ``MERGE_NEAREST`` min-cluster-size policy (the
    SINGLE governance policy for small clusters — no second policy here).

    When the chosen cluster is below ``min_cluster_size``, return ``None`` so
    the caller must choose from another target's own measured candidates.
    """
    if min_cluster_size <= 1:
        return chosen_cluster_id
    if chosen_cluster_id is None:
        return None
    if int(cluster_sizes.get(chosen_cluster_id, 0)) >= min_cluster_size:
        return chosen_cluster_id
    # Cluster size is not similarity evidence.  Redirecting to the largest
    # cluster here used to attach the chosen cluster's high affinity to a
    # completely different target.  Callers must select another target from
    # that target's own qualified candidate evidence, or leave the factor
    # pending when none exists.
    return None


def _classify(
    chosen: Optional[str],
    top_sim: Optional[float],
    runner_sim: Optional[float],
    policy: IncrementalPolicy,
    any_measured: bool,
) -> tuple[Optional[str], IncrementalAssignmentKind]:
    """Classify per DLIB-FA-010."""
    if not any_measured:
        return None, IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
    if chosen is None or top_sim is None:
        return None, IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
    if top_sim < float(policy.affinity_threshold):
        return None, IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
    if runner_sim is not None and (top_sim - runner_sim) < float(policy.ambiguity_gap):
        return chosen, IncrementalAssignmentKind.AMBIGUOUS
    return chosen, IncrementalAssignmentKind.ASSIGNED


def _formal_candidates(
    fingerprint: SimilarityFingerprintArtifact,
    cluster_versions: Mapping[str, ClusterVersionArtifact],
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    evidence_by_pair: Mapping[Tuple[str, str], CertifiedPairwiseEvidence],
    policy: IncrementalPolicy,
) -> tuple[list[IncrementalCandidate], list[IncrementalCandidate]]:
    """Resolve comparable QE evidence; absent/incomparable pairs stay unknown."""
    qualified: list[IncrementalCandidate] = []
    audited: list[IncrementalCandidate] = []
    for cluster_id, cluster in cluster_versions.items():
        for member_id in cluster.member_factor_ids:
            evidence = evidence_by_pair.get((fingerprint.factor_id, member_id))
            if evidence is None:
                evidence = evidence_by_pair.get((member_id, fingerprint.factor_id))
            if evidence is None:
                continue
            if {evidence.factor_id_a, evidence.factor_id_b} != {fingerprint.factor_id, member_id}:
                raise ValueError("pairwise evidence key does not match its factor IDs")
            member = fingerprints_by_id.get(member_id)
            if member is None:
                continue
            if (evidence.window_ref != fingerprint.window or
                    evidence.window_ref != member.window or
                    evidence.universe_ref != fingerprint.universe or
                    evidence.universe_ref != member.universe):
                continue
            if evidence.status is not PairwiseEvidenceStatus.CERTIFIED:
                continue
            signed = float(evidence.signed_similarity)
            affinity = abs(signed)  # explicit orientation rule; sign remains auditable
            comparable_windows = [w for w in evidence.windows
                                  if w.status is PairwiseEvidenceStatus.CERTIFIED]
            signs = {np.sign(float(w.signed_similarity)) for w in comparable_windows
                     if w.signed_similarity is not None and float(w.signed_similarity) != 0.0}
            intervals = [w.confidence_interval for w in comparable_windows
                         if w.confidence_interval[0] is not None]
            sufficient = (evidence.pair_count >= policy.min_pair_count and
                          len(comparable_windows) >= policy.min_certified_windows and
                          all(w.pair_count >= policy.min_pair_count and
                              w.n_days >= policy.min_window_days
                              for w in comparable_windows))
            stable_sign = len(signs) <= 1
            if not sufficient or not stable_sign or len(intervals) < policy.min_certified_windows:
                status = PairwiseEvidenceStatus.UNCERTAIN
            else:
                # For a signed interval not crossing zero, this is the lower
                # bound on absolute affinity. A zero-crossing interval has 0.
                abs_lowers = [0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
                              for lo, hi in intervals]
                abs_uppers = [max(abs(lo), abs(hi)) for lo, hi in intervals]
                if affinity < policy.min_measure_floor and max(abs_uppers) < policy.min_measure_floor:
                    status = PairwiseEvidenceStatus.MEASURED_LOW
                elif min(abs_lowers) >= policy.affinity_threshold:
                    status = PairwiseEvidenceStatus.CERTIFIED
                else:
                    status = PairwiseEvidenceStatus.UNCERTAIN
            candidate = IncrementalCandidate(
                member_id, affinity, cluster_id, status, evidence.pair_count,
                evidence.window_ref, evidence.sample_ref, signed,
                intervals[0] if intervals else (None, None),
            )
            audited.append(candidate)
            if status is PairwiseEvidenceStatus.CERTIFIED:
                qualified.append(candidate)
    qualified.sort(key=lambda item: item.similarity, reverse=True)
    return qualified, audited


def _aggregate_cluster_support(
    candidates: Sequence[IncrementalCandidate],
    cluster_versions: Mapping[str, ClusterVersionArtifact],
    policy: IncrementalPolicy,
) -> tuple[list[IncrementalCandidate], list[IncrementalCandidate]]:
    """Apply the declared medoid/top-k rule to already-certified members."""
    grouped: dict[str, list[IncrementalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.cluster_id, []).append(candidate)
    eligible: list[IncrementalCandidate] = []
    audited: list[IncrementalCandidate] = []
    for cluster_id in sorted(cluster_versions):
        members = sorted(
            grouped.get(cluster_id, ()),
            key=lambda item: (-item.similarity, item.factor_id),
        )
        if not members:
            continue
        selected = members[:policy.cluster_support_k]
        medoid_id = cluster_versions[cluster_id].representative_factor_id
        medoid = next((item for item in members if item.factor_id == medoid_id), None)
        medoid_supported = medoid is not None
        evidence_members = list(selected)
        if medoid is not None and all(item.factor_id != medoid.factor_id for item in evidence_members):
            evidence_members.append(medoid)
        support_count = len(evidence_members)
        top_k_mean = float(np.mean([item.similarity for item in selected]))
        affinity = min(top_k_mean, medoid.similarity) if medoid is not None else top_k_mean
        reasons = []
        if support_count < policy.min_cluster_support:
            reasons.append("INSUFFICIENT_TOP_K_SUPPORT")
        if policy.require_medoid_support and not medoid_supported:
            reasons.append("MEDOID_NOT_SUPPORTED")
        anchor = medoid if medoid is not None else selected[0]
        signed = None
        if anchor.signed_similarity is not None:
            signed = float(np.copysign(affinity, anchor.signed_similarity))
        summary = IncrementalCandidate(
            factor_id=anchor.factor_id, similarity=affinity, cluster_id=cluster_id,
            evidence_status=(PairwiseEvidenceStatus.UNCERTAIN if reasons
                             else anchor.evidence_status),
            pair_count=min(item.pair_count for item in evidence_members),
            window_ref=anchor.window_ref, sample_ref=anchor.sample_ref,
            signed_similarity=(None if reasons else signed),
            confidence_interval=(None, None),
            support_count=support_count,
            support_member_ids=tuple(item.factor_id for item in evidence_members),
            medoid_supported=medoid_supported,
            aggregation_method="min(medoid_affinity,top_k_mean)",
            rejection_reason=";".join(reasons) or None,
            support_confidence_intervals=tuple(
                item.confidence_interval for item in evidence_members
            ),
            support_evidence_refs=tuple(
                (item.factor_id, item.window_ref, item.sample_ref)
                for item in evidence_members
            ),
        )
        audited.append(summary)
        if not reasons:
            eligible.append(summary)
    eligible.sort(key=lambda item: (-item.similarity, item.cluster_id))
    return eligible, audited


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def incremental_assign(
    fingerprints: Sequence[SimilarityFingerprintArtifact],
    cluster_versions: Mapping[str, ClusterVersionArtifact],
    fingerprints_by_id: Mapping[str, SimilarityFingerprintArtifact],
    policy: Optional[IncrementalPolicy] = None,
    *,
    batch_id: Optional[str] = None,
    ann_index: Optional[ANNIndexArtifact] = None,
    min_cluster_size: int = 1,
    min_cluster_policy: str = "KEEP_SMALL",
    certified_pairwise: Optional[Mapping[Tuple[str, str], CertifiedPairwiseEvidence]] = None,
) -> IncrementalAssignResult:
    """Assign new fingerprinted factors into the existing cluster versioning
    WITHOUT mutating the production ``ClusterVersionArtifact`` set.

    :param fingerprints: new-factor fingerprints (their ``factor_id`` must not
        already be a member of ``cluster_versions``; violation FAILS CLOSED).
    :param cluster_versions: logical_cluster_id -> ClusterVersionArtifact of the
        CURRENT production cluster version (never mutated here).
    :param fingerprints_by_id: factor_id -> :class:`SimilarityFingerprintArtifact`
        covering BOTH the existing members and the new factors (members' stored
        embeddings are the nearest-neighbour library).
    :param policy: assignment policy (default :class:`IncrementalPolicy`).
    :param ann_index: optional :class:`ANNIndexArtifact` describing an ANN
        approximate index over the MEMBERS' embeddings.
    :param min_cluster_size / min_cluster_policy: the SAME governance knobs
        re-applied via :func:`_small_cluster_merge_policy_apply` (there is
        exactly one min-cluster policy; ``MERGE_NEAREST`` merges the chosen
        small cluster into the largest family).
    """
    if policy is None:
        policy = IncrementalPolicy()
    if not isinstance(policy, IncrementalPolicy):
        raise TypeError("policy must be an IncrementalPolicy")
    if not fingerprints:
        raise ValueError("fingerprints must be non-empty")
    if not cluster_versions:
        raise ValueError("cluster_versions must be non-empty")

    new_ids = [fp.factor_id for fp in fingerprints]
    if len(set(new_ids)) != len(new_ids):
        raise ValueError("new fingerprint factor_ids must be unique")
    member_all: set[str] = set()
    for cv in cluster_versions.values():
        member_all.update(cv.member_factor_ids)
    for fp in fingerprints:
        if fp.factor_id in member_all:
            raise ValueError(
                f"factor {fp.factor_id!r} is already a cluster member; only "
                "new factors can be incrementally assigned"
            )
    _validate_fingerprint_domains(fingerprints, fingerprints_by_id, member_all)

    if ann_index is not None and ann_index.capability is ANNIndexCapability.APPROXIMATE:
        ann_fingerprints = list(fingerprints) + [
            fingerprints_by_id[factor_id]
            for factor_id in member_all if factor_id in fingerprints_by_id
        ]
        if any(not fingerprint.production_spec_complete
               for fingerprint in ann_fingerprints):
            raise ValueError(
                "incomplete production fingerprint cannot enter approximate "
                "ANN index/recall; bind every required production identity first"
            )
        _validate_ann_membership(
            ann_index, fingerprints_by_id, sorted(member_all)
        )

    if batch_id is None:
        batch_id = hashlib.sha256(
            canonical_digest(tuple(sorted(new_ids))).encode("ascii")
        ).hexdigest()[:16]

    if min_cluster_policy not in ("KEEP_SMALL", "MERGE_NEAREST", "MARK_UNSTABLE"):
        raise ValueError(
            "min_cluster_policy must be one of 'KEEP_SMALL' / 'MERGE_NEAREST' / "
            "'MARK_UNSTABLE'"
        )

    version_ref = _cluster_set_version_ref(cluster_versions)
    parent_set_hash = canonical_digest(tuple(
        (cid, cluster_versions[cid].content_hash) for cid in sorted(cluster_versions)
    ))
    cluster_sizes = {
        cid: len(cv.member_factor_ids) for cid, cv in cluster_versions.items()
    }

    assignments: list[IncrementalClusterAssignment] = []
    candidates_out: list[IncrementalCandidate] = []

    # Build one ANN index for the immutable cluster-set membership and reuse it
    # for every query in this batch.  The previous factor x cluster loop built
    # an Annoy index repeatedly, making index cost scale with query count.
    batch_ann = None
    member_to_cluster: dict[str, str] = {}
    if ann_index is not None and ann_index.capability is ANNIndexCapability.APPROXIMATE:
        member_ids: list[str] = []
        for cid, cv in cluster_versions.items():
            for member_id in cv.member_factor_ids:
                if member_id in member_to_cluster and member_to_cluster[member_id] != cid:
                    raise ValueError("a member factor cannot belong to multiple logical clusters")
                member_to_cluster[member_id] = cid
                member_ids.append(member_id)
        present, matrix = _collect_member_embeddings(fingerprints_by_id, member_ids)
        if present and matrix is not None:
            _validate_ann_index(ann_index, matrix.shape[1])
            try:
                index = AnnoyANNIndex(
                    embedding_dim=matrix.shape[1],
                    n_trees=int(ann_index.index_params.get("n_trees", 10)),
                )
                index.build(present, np.ascontiguousarray(matrix, dtype=np.float32))
                batch_ann = index
            except Exception:
                batch_ann = None

    for fp in fingerprints:
        query = _to_embedding(fp)
        best: list[IncrementalCandidate] = []
        any_measured = False
        if batch_ann is not None:
            grouped: dict[str, list[IncrementalCandidate]] = {}
            results = batch_ann.search(
                np.ascontiguousarray(query, dtype=np.float32),
                k=min(policy.max_candidates, len(member_to_cluster)),
                min_similarity=policy.min_measure_floor,
            )
            for result in results:
                if result.similarity_score is None:
                    continue
                cid = member_to_cluster[result.factor_id]
                grouped.setdefault(cid, []).append(
                    IncrementalCandidate(result.factor_id, float(result.similarity_score), cid)
                )
            best = [max(candidates, key=lambda item: item.similarity) for candidates in grouped.values()]
            any_measured = bool(best)
        for cid, cv in (() if batch_ann is not None else cluster_versions.items()):
            if not cv.member_factor_ids:
                continue
            if ann_index is not None:
                # Batch ANN construction failed or was unavailable: exact
                # fallback preserves semantics without rebuilding an index in
                # the factor x cluster loop.
                cands = _select_exact(
                    query,
                    fingerprints_by_id,
                    cv.member_factor_ids,
                    cid,
                    policy.min_measure_floor,
                )
            else:
                cands = _select_exact(
                    query,
                    fingerprints_by_id,
                    cv.member_factor_ids,
                    cid,
                    policy.min_measure_floor,
                )
            if cands:
                any_measured = True
                best.append(cands[0])
        if certified_pairwise is not None:
            # ANN/cosine candidates are recall hints only.  Formal assignment
            # is based solely on comparable QE evidence and never upgrades an
            # approximation to CERTIFIED by inference.
            best, audited = _formal_candidates(
                fp, cluster_versions, fingerprints_by_id,
                certified_pairwise, policy,
            )
            if (policy.cluster_support_k > 1 or policy.min_cluster_support > 1
                    or policy.require_medoid_support):
                best, support_audit = _aggregate_cluster_support(
                    best, cluster_versions, policy,
                )
                candidates_out.extend(support_audit)
                candidates_out.extend(
                    candidate for candidate in audited
                    if candidate.evidence_status is not PairwiseEvidenceStatus.CERTIFIED
                )
            else:
                candidates_out.extend(audited)
            any_measured = bool(audited)
        else:
            candidates_out.extend(best)
        eligible_best = [
            candidate for candidate in best
            if candidate.evidence_status is not PairwiseEvidenceStatus.MEASURED_LOW
        ]
        best_sorted = sorted(eligible_best, key=lambda c: (-c.similarity, c.cluster_id, c.factor_id))
        if best_sorted:
            top = best_sorted[0]
            runner = best_sorted[1] if len(best_sorted) >= 2 else None
            chosen = top.cluster_id
            if min_cluster_policy == "MERGE_NEAREST":
                chosen = _small_cluster_merge_policy_apply(chosen, cluster_sizes, min_cluster_size)
                if chosen is None:
                    eligible = [
                        candidate for candidate in best_sorted
                        if cluster_sizes.get(candidate.cluster_id, 0) >= min_cluster_size
                        and candidate.similarity >= policy.affinity_threshold
                    ]
                    if eligible:
                        top = eligible[0]
                        runner = eligible[1] if len(eligible) >= 2 else None
                        chosen = top.cluster_id
            runner_sim = None
            if chosen == top.cluster_id:
                runner_sim = runner.similarity if runner else None
            top_sim = top.similarity if chosen == top.cluster_id else None
            chosen, kind = _classify(
                chosen, top_sim, runner_sim, policy, any_measured
            )
        else:
            chosen, kind = _classify(None, None, None, policy, any_measured)
        assignments.append(
            IncrementalClusterAssignment(
                factor_id=fp.factor_id,
                logical_cluster_id=chosen,
                kind=kind,
                cluster_set_version_ref=version_ref,
                affinity=(
                    next(
                        (c.similarity for c in best_sorted if c.cluster_id == chosen),
                        None,
                    ) if chosen is not None else None
                ),
                parent_cluster_set_hash=(
                    parent_set_hash if kind is IncrementalAssignmentKind.ASSIGNED else ""
                ),
            )
        )
    return IncrementalAssignResult(
        batch_id=batch_id,
        assignments=tuple(assignments),
        candidates=tuple(candidates_out),
        created_at=datetime.now(timezone.utc).isoformat(),
        content_hash="",  # derived in __post_init__
    )


def _cluster_set_version_ref(
    cluster_versions: Mapping[str, ClusterVersionArtifact],
) -> str:
    refs = {cv.cluster_set_version_ref for cv in cluster_versions.values()}
    if len(refs) != 1:
        raise ValueError(
            "cluster_versions must all reference the SAME cluster_set_version_ref"
        )
    return next(iter(refs))


# ---------------------------------------------------------------------------
# copy-on-write cluster version + lineage
# ---------------------------------------------------------------------------


def build_incremental_cluster_version(
    production_version_artifacts: Mapping[str, ClusterVersionArtifact],
    assignments: Sequence[IncrementalClusterAssignment],
    *,
    new_cluster_set_version_id: str,
    overlay_suffix: str = "inc",
) -> tuple[ClusterVersionArtifact, ...]:
    """Copy-on-write overlay cluster versions (QRP-P6-INC6).

    For every logical cluster that gained an ``ASSIGNED`` factor, produce a NEW
    ``ClusterVersionArtifact`` whose ``member_factor_ids`` are the production
    members PLUS the new factor, referencing the NEW cluster-set-version id.
    The production artifact (and its ``created_at`` / ``content_hash``) is
    NEVER mutated in place.

    The production ClusterSetVersionArtifact itself is not created here — the
    caller owns the next version id (e.g. ``csv2``); this function produces
    the per-cluster overlay artifacts that live under it.

    :returns: tuple of NEW ClusterVersionArtifact (one per ASSIGNED cluster),
        sorted by logical_cluster_id.
    """
    if not production_version_artifacts:
        raise ValueError("production_version_artifacts must be non-empty")
    if not new_cluster_set_version_id:
        raise ValueError("new_cluster_set_version_id is required")
    base_version = _cluster_set_version_ref(production_version_artifacts)
    parent_set_hash = canonical_digest(tuple(
        (cid, production_version_artifacts[cid].content_hash)
        for cid in sorted(production_version_artifacts)
    ))
    if new_cluster_set_version_id == base_version:
        raise ValueError("new cluster set version id must differ from its parent")
    for key, artifact in production_version_artifacts.items():
        if key != artifact.logical_cluster_id:
            raise ValueError("production cluster mapping key/logical id mismatch")

    touched: dict[str, list[str]] = {}
    assigned_to: dict[str, str] = {}
    for a in assignments:
        if a.cluster_set_version_ref != base_version:
            raise ValueError("stale incremental assignment parent version (CAS mismatch)")
        if not _as_touchable(a.kind):
            continue
        if a.parent_cluster_set_hash != parent_set_hash:
            raise ValueError("stale incremental assignment parent content hash (CAS mismatch)")
        if a.logical_cluster_id is None:
            raise ValueError(
                "an ASSIGNED incremental assignment must carry a logical_cluster_id"
            )
        if a.logical_cluster_id not in production_version_artifacts:
            raise ValueError(
                f"logical_cluster_id {a.logical_cluster_id!r} is not in the "
                "production cluster version set"
            )
        previous = assigned_to.setdefault(a.factor_id, a.logical_cluster_id)
        if previous != a.logical_cluster_id:
            raise ValueError("a factor cannot be assigned to multiple partition clusters")
        touched.setdefault(a.logical_cluster_id, []).append(a.factor_id)

    out: list[ClusterVersionArtifact] = []
    for cid in sorted(production_version_artifacts):
        prod = production_version_artifacts[cid]
        added = tuple(sorted(set(touched.get(cid, ()))))
        new_members = tuple(prod.member_factor_ids) + added
        if len(set(new_members)) != len(new_members):
            raise ValueError("overlay collision: a new member already in the cluster")
        out.append(
            ClusterVersionArtifact(
                logical_cluster_id=prod.logical_cluster_id,
                cluster_set_version_ref=new_cluster_set_version_id,
                member_factor_ids=new_members,
                representative_factor_id=prod.representative_factor_id,
                scale=prod.scale,
                content_hash="",  # derived in __post_init__
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class IncrementalLineageEdge:
    """Copy-on-write lineage edge for an incremental cluster version.

    ``parent_cluster_version_ref`` is the PRODUCTION ClusterVersionArtifact ref
    (its logical id + its cluster_set_version ref + short content hash); the NEW
    overlay artifact ref is the edge head.  ``requested_by`` is the reason the
    edge exists (per the prompt: lineage edge records parent -> new +
    requested_by / 理由).
    """

    logical_cluster_id: str
    parent_cluster_version_ref: str
    new_cluster_version_ref: str
    added_factor_ids: tuple[str, ...]
    requested_by: str = "incremental_assign"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.logical_cluster_id:
            raise ValueError("logical_cluster_id is required")
        if not self.parent_cluster_version_ref:
            raise ValueError("parent_cluster_version_ref is required")
        if not self.new_cluster_version_ref:
            raise ValueError("new_cluster_version_ref is required")
        if not self.added_factor_ids:
            raise ValueError("added_factor_ids cannot be empty")
        object.__setattr__(self, "added_factor_ids", tuple(self.added_factor_ids))
        computed = canonical_digest(
            self.logical_cluster_id,
            self.parent_cluster_version_ref,
            self.new_cluster_version_ref,
            self.added_factor_ids,
            self.requested_by,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed incremental-lineage "
                "edge content hash; a caller may not self-report an arbitrary "
                "hash — FAIL CLOSED"
            )

    def to_dict(self) -> dict:
        return {
            "logical_cluster_id": self.logical_cluster_id,
            "parent_cluster_version_ref": self.parent_cluster_version_ref,
            "new_cluster_version_ref": self.new_cluster_version_ref,
            "added_factor_ids": list(self.added_factor_ids),
            "requested_by": self.requested_by,
            "content_hash": self.content_hash,
        }


def build_incremental_lineage_edges(
    production_version_artifacts: Mapping[str, ClusterVersionArtifact],
    overlay_versions: Sequence[ClusterVersionArtifact],
    assignments: Sequence[IncrementalClusterAssignment],
    *,
    requested_by: str = "incremental_assign",
) -> tuple[IncrementalLineageEdge, ...]:
    """Build copy-on-write lineage edges parent_production -> new_overlay.

    One edge per touched logical cluster; ``added_factor_ids`` is the new
    factor(s) added in this overlay.  The production artifact (the parent) is
    never mutated — an edge only ADDS a child.
    """
    if not production_version_artifacts:
        raise ValueError("production_version_artifacts must be non-empty")
    touch: dict[str, set[str]] = {}
    for a in assignments:
        if _as_touchable(a.kind) and a.logical_cluster_id:
            touch.setdefault(a.logical_cluster_id, set()).add(a.factor_id)

    overlay_by_id: dict[str, ClusterVersionArtifact] = {}
    for ov in overlay_versions:
        if ov.logical_cluster_id in overlay_by_id:
            raise ValueError("overlay logical cluster ids must be unique")
        overlay_by_id[ov.logical_cluster_id] = ov
    if set(overlay_by_id) != set(production_version_artifacts):
        raise ValueError("overlay must be a full cluster-set snapshot")
    edges: list[IncrementalLineageEdge] = []
    for cid in sorted(production_version_artifacts):
        prod = production_version_artifacts[cid]
        ov = overlay_by_id[cid]
        if ov.logical_cluster_id != prod.logical_cluster_id:
            raise ValueError("overlay child logical id does not match parent")
        parent_members = set(prod.member_factor_ids)
        child_members = set(ov.member_factor_ids)
        removed = parent_members - child_members
        actual_added = child_members - parent_members
        declared_added = touch.get(cid, set())
        if removed:
            raise ValueError("incremental overlay contains undeclared member removals")
        if actual_added != declared_added:
            raise ValueError("declared assignment additions do not match actual child diff")
        if not actual_added:
            continue
        edges.append(
            IncrementalLineageEdge(
                logical_cluster_id=cid,
                parent_cluster_version_ref=(
                    f"{prod.logical_cluster_id}@{prod.cluster_set_version_ref}"
                    f"#{prod.content_hash[:16]}"
                ),
                new_cluster_version_ref=(
                    f"{ov.logical_cluster_id}@{ov.cluster_set_version_ref}"
                    f"#{ov.content_hash[:16]}"
                ),
                added_factor_ids=tuple(sorted(actual_added)),
                requested_by=requested_by,
                content_hash="",
            )
        )
    return tuple(edges)
