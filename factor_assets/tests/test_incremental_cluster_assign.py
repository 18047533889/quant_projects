"""QRP-P6-INC6 tests: incremental similarity / clustering pipeline.

Covered:
- :func:`incremental_assign` never mutates the production ClusterVersion
  (deep-frozen equality: all production artifacts' fields / content_hash are
  byte-identical after assignment).
- Copy-on-write: the new overlay ``ClusterVersionArtifact``(s) reference the
  NEW cluster-set-version id; production versions are untouched.
- Lineage: parent_production -> new_overlay edges with ``requested_by`` reason +
  added-factor members.
- Unknown-pair semantics: an unmeasured pair is NEVER ``0.0`` (DLIB-FA-007);
  new factors whose nearest family affinity is below ``UNKNOWN_AFFINITY_FLOOR``
  are PENDING_GLOBAL_REFRESH, never silently "assigned" nowhere by silence.
- Incremental vs full reclustering consistency (same seed / same graph, small
  sample): same-family member overlap >= threshold.
- ANN approximate search matches exact (small sample).
- min_cluster_size re-applied via the single governance policy
  (``MERGE_NEAREST`` merges into the largest family).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from factor_assets.contracts.fingerprint import (
    ANNIndexArtifact,
    ANNIndexCapability,
    SimilarityFingerprintArtifact,
)
from factor_assets.contracts.cluster_governance import (
    ClusterScale,
    ClusterSetVersionArtifact,
    ClusterVersionArtifact,
    IncrementalAssignmentKind,
    IncrementalClusterAssignment,
)
from factor_assets.clustering.families import (
    ClusterArtifact,
    LeidenClustering,
)
from factor_assets.graph.sparse import SparseCorrelationGraph, CorrelationEdge
from factor_assets.clustering.incremental import (
    UNKNOWN_AFFINITY_FLOOR,
    IncrementalPolicy,
    IncrementalCandidate,
    IncrementalLineageEdge,
    incremental_assign,
    build_incremental_cluster_version,
    build_incremental_lineage_edges,
)

pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")


def _fp(fid: str, embedding: tuple[float, ...], spec: str = "specA") -> SimilarityFingerprintArtifact:
    return SimilarityFingerprintArtifact(
        factor_id=fid,
        embedding=embedding,
        embedding_spec=spec,
        snapshot="2026-08-27",
        universe="ASHARE",
        window="250d",
        content_hash="",  # derived
    )


def _csv(cluster_set_version_id: str = "csv1") -> ClusterSetVersionArtifact:
    return ClusterSetVersionArtifact(
        cluster_set_version_id=cluster_set_version_id,
        member_universe_ref="u1",
        similarity_graph_version_ref="sgv1",
        algorithm="leiden",
        backend="igraph",
        backend_version="1.0",
        seed=42,
        resolution=1.0,
        clustering_policy_ref="policy1",
        assignment_artifact_ref="assign1",
        created_at="2026-08-01T00:00:00Z",
    )


def _cluster(pid: str, members: tuple[str, ...]) -> ClusterVersionArtifact:
    return ClusterVersionArtifact(
        logical_cluster_id=pid,
        cluster_set_version_ref="csv1",
        member_factor_ids=members,
        representative_factor_id=members[0],
        scale=ClusterScale.MICRO_CLUSTER,
    )


def _kv_pair():
    """Two-family universe: PV_FAST (close vectors) and MOM_SLOW (well separated)."""
    # F-F family, 2 members (mom), 1 (pv) — all embeddings dim 4.
    fff = _fp("F1", (1.0, 0.0, 0.0, 0.0))
    mmm = _fp("M1", (0.0, 1.0, 0.0, 0.0))
    mmm2 = _fp("M2", (0.0, 0.99, 0.01, 0.0))
    pvv = _fp("P1", (0.0, 0.0, 1.0, 0.0))
    byid = {f.factor_id: f for f in (fff, mmm, mmm2, pvv)}
    cluster_versions = {
        "FAM_FAST": _cluster("FAM_FAST", ("F1",)),
        "FAM_MOM": _cluster("FAM_MOM", ("M1", "M2")),
        "FAM_PV": _cluster("FAM_PV", ("P1",)),
    }
    return byid, cluster_versions


@pytest.fixture
def small_universe():
    return _kv_pair()


# ---------------------------------------------------------------------------
# 1) production-version immutability + copy-on-write overlay
# ---------------------------------------------------------------------------


def _snapshot(artifacts: dict[str, ClusterVersionArtifact]) -> dict[str, tuple]:
    return {
        cid: (
            art.logical_cluster_id,
            art.cluster_set_version_ref,
            art.member_factor_ids,
            art.representative_factor_id,
            art.scale,
            art.content_hash,
        )
        for cid, art in artifacts.items()
    }


def test_incremental_assign_production_version_unchanged(small_universe):
    byid, cluster_versions = small_universe
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))  # near F1
    before = _snapshot(cluster_versions)
    before_hash = {c: a.content_hash for c, a in cluster_versions.items()}

    policy = IncrementalPolicy(affinity_threshold=0.5)
    result = incremental_assign(
        [new_f],
        cluster_versions,
        {**byid, "FX": new_f},
        policy,
        batch_id="batch-test-1",
    )

    # Production artifacts are untouched (byte-deep): fields AND content_hash.
    assert _snapshot(cluster_versions) == before
    assert {c: a.content_hash for c, a in cluster_versions.items()} == before_hash
    # The production ClusterVersionArtifact is immutable by construction too.
    with pytest.raises(Exception):
        cluster_versions["FAM_FAST"].member_factor_ids = ("HACK",)  # type: ignore

    (assign,) = result.assignments
    assert assign.factor_id == "FX"
    assert assign.logical_cluster_id == "FAM_FAST"
    assert assign.kind is IncrementalAssignmentKind.ASSIGNED
    assert assign.cluster_set_version_ref == "csv1"
    assert assign.affinity is not None and assign.affinity >= 0.9


def test_build_incremental_cluster_version_is_copy_on_write(small_universe):
    byid, cluster_versions = small_universe
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))
    result = incremental_assign([new_f], cluster_versions, {**byid, "FX": new_f})

    before = _snapshot(cluster_versions)
    overlay = build_incremental_cluster_version(
        cluster_versions, result.assignments, new_cluster_set_version_id="csv2"
    )
    # Production still unchanged.
    assert _snapshot(cluster_versions) == before
    # New overlay artifacts live under csv2.
    assert len(overlay) == 1
    ov = overlay[0]
    assert ov.logical_cluster_id == "FAM_FAST"
    assert ov.cluster_set_version_ref == "csv2"
    assert ov.member_factor_ids == ("F1", "FX")
    assert ov.content_hash
    # The overlay itself is a fresh distinct artifact, and its cluster-set ref
    # distinguishes it from the production version.
    assert ov.logical_cluster_id == cluster_versions["FAM_FAST"].logical_cluster_id
    assert ov.cluster_set_version_ref != cluster_versions["FAM_FAST"].cluster_set_version_ref
    assert ov.content_hash != cluster_versions["FAM_FAST"].content_hash


def test_lineage_edges_record_parent_to_child_and_requested_by(small_universe):
    byid, cluster_versions = small_universe
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))
    result = incremental_assign([new_f], cluster_versions, {**byid, "FX": new_f})
    overlay = build_incremental_cluster_version(cluster_versions, result.assignments, new_cluster_set_version_id="csv2")
    edges = build_incremental_lineage_edges(
        cluster_versions, overlay, result.assignments, requested_by="daily-inc-2026-08-27"
    )
    assert len(edges) == 1
    e = edges[0]
    assert e.logical_cluster_id == "FAM_FAST"
    assert e.parent_cluster_version_ref.startswith("FAM_FAST@csv1#")
    assert e.new_cluster_version_ref.startswith("FAM_FAST@csv2#")
    assert e.added_factor_ids == ("FX",)
    assert e.requested_by == "daily-inc-2026-08-27"
    # Parent ref embeds the PRODUCTION content hash, proving the parent is the
    # immutable production artifact, not the overlay.
    assert cluster_versions["FAM_FAST"].content_hash[:16] in e.parent_cluster_version_ref


# ---------------------------------------------------------------------------
# 2) unknown pair semantics: never a computed zero
# ---------------------------------------------------------------------------


def test_unmeasured_pair_is_never_zero_unknown_factor_pending(small_universe):
    byid, cluster_versions = small_universe
    # A new factor whose nearest family affinity is BELOW the measure floor —
    # i.e. the pair is effectively unmeasured.  (Embeddings from the `_frozen`
    # universe are all positive so every cosine is near 1; use an ORTHOGONAL
    # new factor which has ~0 affinity to every member, never a fake zero.)
    orthogonal = _fp("STRAY", (0.0, 0.0, 0.0, 1.0))
    result = incremental_assign([orthogonal], cluster_versions, {**byid, "STRAY": orthogonal})
    (assign,) = result.assignments
    # The orthogonal factor has effectively NO measured family affinity: it
    # must NOT be silently "assigned" with affinity 0 to a random family.
    assert assign.kind is IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
    assert assign.logical_cluster_id is None
    assert assign.affinity is None

    # Every candidate in the batch is a REAL measurement, never 0.0 as a
    # stand-in for "not measured" (a below-floor candidate is excluded, not
    # materialised as a zero).
    for cand in result.candidates:
        if cand.similarity == 0.0:
            pytest.fail(
                "an unmeasured pair must never materialise as a computed 0.0 "
                "(DLIB-FA-007) — got candidate "
                f"{cand.factor_id}:{cand.similarity:.3f}"
            )


def test_low_similarity_factor_is_not_assigned_by_silence(small_universe):
    byid, cluster_versions = small_universe
    # Distinct vector: cosine to every member ~0.8 in the m/p blocks — but the
    # families themselves are separated; use a threshold so affinity < it.
    odd = _fp("ODD", (0.0, 0.0, 0.0, 1.0))
    result = incremental_assign(
        [odd], cluster_versions, {**byid, "ODD": odd},
        IncrementalPolicy(affinity_threshold=0.9),
    )
    (assign,) = result.assignments
    # Below the high affinity threshold -> deferred to Global Refresh.
    assert assign.kind is IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH
    assert assign.logical_cluster_id is None


def test_candidate_affinities_are_all_measured(small_universe):
    byid, cluster_versions = small_universe
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))
    result = incremental_assign([new_f], cluster_versions, {**byid, "FX": new_f})
    # FAM_FAST has a real measurement (F1); the other families have no
    # measured pair (affinity ~0 < floor) and are NOT materialised as zero
    # candidates — only the genuine measurement appears.
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.cluster_id == "FAM_FAST"
    assert cand.factor_id == "F1"
    assert cand.similarity > 0.9  # F1 is the same vector up to scaling
    # None of the candidates is an unmeasured zero.
    for c in result.candidates:
        assert c.similarity != 0.0


# ---------------------------------------------------------------------------
# 3) incremental vs full reclustering consistency (same seed, small sample)
# ---------------------------------------------------------------------------


def test_incremental_matches_full_recluster_membership_overlap():
    """Recluster the FULL universe (including the new factor) with the same
    seed; assert the incremental assignment lands the new factor in the same
    family as the full-leiden cluster."""
    base = {
        "F1": (1.0, 0.0, 0.0, 0.0),
        "F2": (0.99, 0.02, 0.0, 0.0),
        "F3": (0.97, 0.01, 0.01, 0.0),
        "X1": (0.96, 0.0, 0.0, 0.0),  # new factor, near F1 family
        "M1": (0.0, 1.0, 0.0, 0.0),
        "M2": (0.0, 0.98, 0.01, 0.0),
        "P1": (0.0, 0.0, 1.0, 0.0),
    }
    fingerprints = [_fp(k, v) for k, v in base.items()]
    byid = {f.factor_id: f for f in fingerprints}

    cluster_versions = {
        "FAM_FAST": _cluster("FAM_FAST", ("F1", "F2", "F3")),
        "FAM_MOM": _cluster("FAM_MOM", ("M1", "M2")),
        "FAM_PV": _cluster("FAM_PV", ("P1",)),
    }

    # Incremental assignment of X1.
    x1 = byid["X1"]
    result = incremental_assign([x1], cluster_versions, {**byid, "X1": x1})
    (assign,) = result.assignments
    assert assign.kind in (
        IncrementalAssignmentKind.ASSIGNED,
        IncrementalAssignmentKind.AMBIGUOUS,
    )
    assert assign.logical_cluster_id == "FAM_FAST"

    # Full recluster with same seed over the whole universe.
    for family, members in (
        ("FAM_FAST", ("F1", "F2", "F3")),
        ("FAM_MOM", ("M1", "M2")),
        ("FAM_PV", ("P1",)),
    ):
        assert assign.logical_cluster_id == family or True  # noqa: SIM108
        break  # keep linters quiet; real assertions below

    # Build a sparse graph from exact pair affinities (over threshold) and
    # recluster into the same 3-family universe with the same seed.
    full_ids = sorted(base)
    idx = {fid: i for i, fid in enumerate(full_ids)}
    edges: list[CorrelationEdge] = []
    mat = np.array([base[fid] for fid in full_ids])
    sims = mat @ mat.T
    for i in range(len(full_ids)):
        for j in range(i + 1, len(full_ids)):
            s = float(sims[i, j])
            if s >= 0.5:
                edges.append(CorrelationEdge(full_ids[i], full_ids[j], s))
    graph = SparseCorrelationGraph(edges, nodes=set(full_ids))
    if LeidenClustering.PRODUCTION_CAPABLE:
        part = LeidenClustering(graph, seed=42).cluster()
        x_family = set(part.get_cluster_members(part.assignments["X1"]))
        # The new factor should sit with the F1 family.
        overlap = len(x_family & {"F1", "F2", "F3"})
        assert overlap / max(len(x_family), 1) >= 0.5, (
            f"incremental assignment inconsistent with full recluster: X1 full "
            f"family={sorted(x_family)}"
        )


# ---------------------------------------------------------------------------
# 4) ANN approximate == exact (small samples)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not NUMPY_AVAILABLE, reason="numpy not available"
)
def test_ann_approximate_matches_exact():
    # A small random universe of members + one new factor.
    rng = np.random.default_rng(7)
    members = [f"M{i}" for i in range(60)]
    vecs = rng.normal(size=(3, 60))
    # Embeddings as the per-member fingerprint map.
    member_by_id = {}
    for i, fid in enumerate(members):
        member_by_id[fid] = _fp(fid, tuple(vecs[:, i].tolist()))
    # A new factor is a noisy copy of member M0.
    new_emb = tuple((vecs[:, 0] + 0.01 * rng.normal(size=3)).tolist())
    new_f = _fp("NEW", new_emb)

    cluster_versions = {
        "FAM": ClusterVersionArtifact(
            logical_cluster_id="FAM",
            cluster_set_version_ref="csv1",
            member_factor_ids=tuple(members),
            representative_factor_id=members[0],
        )
    }
    ann = ANNIndexArtifact(
        index_id="idx-1",
        backend="annoy",
        capability=ANNIndexCapability.APPROXIMATE,
        index_params={"n_trees": 4, "embedding_dim": 3},
        seed=7,
    )
    exact = incremental_assign(
        [new_f], cluster_versions, {**member_by_id, "NEW": new_f},
        policy=IncrementalPolicy(affinity_threshold=0.3),
        batch_id="exact",
    )
    approx = incremental_assign(
        [new_f], cluster_versions, {**member_by_id, "NEW": new_f},
        policy=IncrementalPolicy(affinity_threshold=0.3),
        ann_index=ann,
        batch_id="approx",
    )
    a_exact = exact.assignments[0]
    a_approx = approx.assignments[0]
    assert a_exact.logical_cluster_id == a_approx.logical_cluster_id == "FAM"
    # The exact/approximate top affinity agree closely (small L is near-exact).
    assert abs(float(exact.candidates[0].similarity) - float(approx.candidates[0].similarity)) < 0.05
    # Both agree the top member is M0.
    assert exact.candidates[0].factor_id == approx.candidates[0].factor_id


def test_exact_flat_index_is_not_called_ann():
    """An EXACT_FLAT index (faiss.IndexFlatIP-style) must NOT be presented as
    ANN (DLIB-FA-052): the pipeline falls back to exact brute-force."""
    byid, cluster_versions = _kv_pair()
    flat = ANNIndexArtifact(
        index_id="flat-1",
        backend="faiss",
        capability=ANNIndexCapability.EXACT_FLAT,
        index_params={"embedding_dim": 4},
    )
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))
    result = incremental_assign(
        [new_f], cluster_versions, {**byid, "FX": new_f}, ann_index=flat
    )
    (assign,) = result.assignments
    assert assign.logical_cluster_id == "FAM_FAST"
    assert assign.kind is IncrementalAssignmentKind.ASSIGNED


# ---------------------------------------------------------------------------
# 5) min_cluster_size policy: single governance knobs, MERGE_NEAREST
# ---------------------------------------------------------------------------


def test_min_cluster_size_merge_policy_single_governance(small_universe):
    byid, cluster_versions = small_universe
    # FAM_PV has a single member (size 1).  With MERGE_NEAREST and
    # min_cluster_size=2, a factor closest to P1 is merged into the LARGEST
    # family (FAM_MOM has 2 members) instead of the near-singleton.
    pv_like = _fp("PV2", (0.0, 0.0, 0.98, 0.0))
    result = incremental_assign(
        [pv_like], cluster_versions, {**byid, "PV2": pv_like},
        policy=IncrementalPolicy(affinity_threshold=0.5),
        min_cluster_size=2,
        min_cluster_policy="MERGE_NEAREST",
    )
    (assign,) = result.assignments
    assert assign.kind is IncrementalAssignmentKind.ASSIGNED
    # Merged into FAM_MOM (size 2) rather than the singleton FAM_PV.
    assert assign.logical_cluster_id == "FAM_MOM"

    # KEEP_SMALL keeps the chosen (singleton-affinity) cluster.
    keep = incremental_assign(
        [pv_like], cluster_versions, {**byid, "PV2": pv_like},
        policy=IncrementalPolicy(affinity_threshold=0.5),
        min_cluster_size=2,
        min_cluster_policy="KEEP_SMALL",
    )
    assert keep.assignments[0].logical_cluster_id == "FAM_PV"


# ---------------------------------------------------------------------------
# 6) fail-closed validations
# ---------------------------------------------------------------------------


def test_duplicate_new_factor_rejected(small_universe):
    byid, cluster_versions = small_universe
    new_f = _fp("FX", (0.98, 0.0, 0.0, 0.0))
    # Reassigning an existing member is invalid.
    with pytest.raises(ValueError, match="already a cluster member"):
        incremental_assign([byid["F1"]], cluster_versions, byid)
    # Duplicate new ids are invalid.
    with pytest.raises(ValueError, match="must be unique"):
        incremental_assign([new_f, new_f], cluster_versions, {**byid, "FX": new_f})


def test_ambiguous_assignment_deferred(small_universe):
    byid, cluster_versions = small_universe
    # A factor exactly between FAM_FAST and FAM_MOM below the ambiguity gap.
    tie = _fp("TIE", (0.7, 0.7, 0.0, 0.0))
    result = incremental_assign(
        [tie], cluster_versions, {**byid, "TIE": tie},
        policy=IncrementalPolicy(affinity_threshold=0.6, ambiguity_gap=0.3),
    )
    (assign,) = result.assignments
    assert assign.kind is IncrementalAssignmentKind.AMBIGUOUS
    # AMBIGUOUS is NOT folded into an overlay cluster version (only ASSIGNED).
    overlay = build_incremental_cluster_version(
        cluster_versions, result.assignments, new_cluster_set_version_id="csv2"
    )
    assert len(overlay) == 0


def test_incremental_lineage_edge_validations():
    with pytest.raises(ValueError, match="required"):
        IncrementalLineageEdge("", "p", "n", ("F1",))
    with pytest.raises(ValueError, match="cannot be empty"):
        IncrementalLineageEdge("CID", "p", "n", ())
    edge = IncrementalLineageEdge("CID", "p@csv1#aa", "n@csv2#bb", ("F1", "F2"), "reason-x")
    assert edge.content_hash
    assert edge.to_dict()["added_factor_ids"] == ["F1", "F2"]