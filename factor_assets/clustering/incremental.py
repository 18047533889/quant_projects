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
  :func:`_small_cluster_merge_policy_apply`, which mirrors the families.py
  ``MERGE_NEAREST`` policy (merge the chosen small cluster into the family
  with the LARGEST membership) — there is NO second, divergent
  min-cluster-size policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class IncrementalCandidate:
    """A single affinity measurement between the new factor and an existing
    member factor of a logical cluster (never an unmeasured zero)."""

    factor_id: str
    similarity: float
    cluster_id: str

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

    def to_dict(self) -> dict:
        return {
            "factor_id": self.factor_id,
            "similarity": self.similarity,
            "cluster_id": self.cluster_id,
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
        rows.append(fid)
        vectors.append(_to_embedding(fp))
    if not vectors:
        return [], None
    return rows, np.vstack(vectors)


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
        if s < floor:
            continue  # below floor = unmeasured/not meaningful, never a zero
        out.append(IncrementalCandidate(factor_id=fid, similarity=s, cluster_id=cluster_id))
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

    When the chosen cluster is below ``min_cluster_size`` and the policy is
    ``MERGE_NEAREST``, the new factor is merged into the cluster with the
    LARGEST membership (the strongest family) so a near-singleton does not win
    by isolation.  ``KEEP_SMALL`` / ``MARK_UNSTABLE`` keep the chosen cluster.
    """
    if min_cluster_size <= 1:
        return chosen_cluster_id
    if chosen_cluster_id is None:
        return None
    if int(cluster_sizes.get(chosen_cluster_id, 0)) >= min_cluster_size:
        return chosen_cluster_id
    if not cluster_sizes:
        return chosen_cluster_id
    largest_id = max(cluster_sizes, key=lambda c: (cluster_sizes[c], c))
    return largest_id


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
    cluster_sizes = {
        cid: len(cv.member_factor_ids) for cid, cv in cluster_versions.items()
    }

    assignments: list[IncrementalClusterAssignment] = []
    candidates_out: list[IncrementalCandidate] = []

    for fp in fingerprints:
        query = _to_embedding(fp)
        best: list[IncrementalCandidate] = []
        any_measured = False
        for cid, cv in cluster_versions.items():
            if not cv.member_factor_ids:
                continue
            if ann_index is not None:
                cands = _select_ann(
                    query,
                    fingerprints_by_id,
                    cv.member_factor_ids,
                    cid,
                    ann_index,
                    policy.max_candidates,
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
        candidates_out.extend(best)
        best_sorted = sorted(best, key=lambda c: c.similarity, reverse=True)
        if best_sorted:
            top = best_sorted[0]
            runner = best_sorted[1] if len(best_sorted) >= 2 else None
            chosen = top.cluster_id
            if min_cluster_policy == "MERGE_NEAREST":
                chosen = _small_cluster_merge_policy_apply(
                    chosen, cluster_sizes, min_cluster_size
                )
            runner_sim = None
            if chosen == top.cluster_id:
                runner_sim = runner.similarity if runner else None
            top_sim = top.similarity
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
                    best_sorted[0].similarity
                    if chosen is not None and best_sorted
                    else None
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

    touched: dict[str, list[str]] = {}
    for a in assignments:
        if not _as_touchable(a.kind):
            continue
        if a.logical_cluster_id is None:
            raise ValueError(
                "an ASSIGNED incremental assignment must carry a logical_cluster_id"
            )
        if a.logical_cluster_id not in production_version_artifacts:
            raise ValueError(
                f"logical_cluster_id {a.logical_cluster_id!r} is not in the "
                "production cluster version set"
            )
        touched.setdefault(a.logical_cluster_id, []).append(a.factor_id)

    out: list[ClusterVersionArtifact] = []
    for cid in sorted(touched):
        prod = production_version_artifacts[cid]
        added = tuple(sorted(set(touched[cid])))
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
    touch: dict[str, list[str]] = {}
    for a in assignments:
        if _as_touchable(a.kind) and a.logical_cluster_id:
            touch.setdefault(a.logical_cluster_id, []).append(a.factor_id)

    overlay_by_id = {ov.logical_cluster_id: ov for ov in overlay_versions}
    edges: list[IncrementalLineageEdge] = []
    for cid in sorted(touch):
        prod = production_version_artifacts[cid]
        ov = overlay_by_id.get(cid)
        if ov is None:
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
                added_factor_ids=tuple(sorted(set(touch[cid]))),
                requested_by=requested_by,
                content_hash="",
            )
        )
    return tuple(edges)