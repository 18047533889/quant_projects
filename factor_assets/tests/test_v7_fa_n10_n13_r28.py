"""V7 independent regressions for FA admissionN10-N13 and R28."""

from dataclasses import replace

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.clustering.incremental import (
    PairwiseEvidenceStatus,
    build_incremental_cluster_version,
    build_incremental_lineage_edges,
    incremental_assign,
)
from factor_assets.contracts.admission import AdmissionDecision
from factor_assets.contracts.cluster_governance import (
    ClusterVersionArtifact,
    IncrementalAssignmentKind,
    IncrementalClusterAssignment,
)
from factor_assets.contracts.fingerprint import SimilarityFingerprintArtifact
from factor_assets.contracts.fingerprint import ANNIndexArtifact, ANNIndexCapability
from factor_assets.contracts._canonical import canonical_digest
from factor_assets.contracts.lifecycle import HealthState, LifecycleState
from factor_assets.tests.assembly.test_production_and_similarity_artifact import (
    make_admission,
    make_asset,
    make_spec,
    make_treatment,
)


def _cluster(cid="C", version="cs1", members=("old",)):
    return ClusterVersionArtifact(
        logical_cluster_id=cid,
        cluster_set_version_ref=version,
        member_factor_ids=members,
        representative_factor_id=members[0],
    )


def _assignment(fid="new", cid="C", version="cs1", parent_hash="fixture-parent-hash"):
    return IncrementalClusterAssignment(
        factor_id=fid,
        logical_cluster_id=cid,
        kind=IncrementalAssignmentKind.ASSIGNED,
        cluster_set_version_ref=version,
        affinity=0.95,
        parent_cluster_set_hash=parent_hash,
    )


def _parent_hash(parents):
    return canonical_digest(tuple(
        (cid, parents[cid].content_hash) for cid in sorted(parents)
    ))


def _fp(fid, vector, *, spec="rank-v1", snapshot="s1", universe="u1", window="w1"):
    return SimilarityFingerprintArtifact(
        factor_id=fid,
        embedding=vector,
        embedding_spec=spec,
        snapshot=snapshot,
        universe=universe,
        window=window,
        preprocessing_ref="prep-v1",
        mask_policy="pairwise-valid",
        direction="signed",
        aggregation_method="daily-then-time",
        embedding_model_version="embed-v1",
        value_ref=f"values:{fid}",
        profile_ref=f"profile:{fid}",
    )


@pytest.mark.parametrize("selection_policy", ["manual", "family_robust"])
def test_n10_public_production_rejects_complete_rejected_admission(selection_policy):
    asset = make_asset("F1")
    rejected = make_admission("F1", decision=AdmissionDecision.REJECTED)
    kwargs = {}
    if selection_policy != "manual":
        from factor_assets.tests.assembly.test_production_and_similarity_artifact import make_decision
        kwargs["selection_decisions"] = [make_decision("F1")]
    with pytest.raises(ValueError, match="APPROVED"):
        FactorSetAssembler().assemble(
            make_spec("v7", "V7", selection_policy),
            [asset],
            admission_artifacts={"F1": rejected},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
            **kwargs,
        )


def test_n10_approved_matching_record_still_assembles():
    result = FactorSetAssembler().assemble(
        make_spec("v7-ok", "V7", "manual"),
        [make_asset("F1")],
        admission_artifacts={"F1": make_admission("F1")},
        treatment_selection_artifacts={"F1": make_treatment("F1")},
        production=True,
    )
    assert result.factor_ids == ("F1",)


def test_n10_production_rejects_treatment_for_different_factor_version():
    with pytest.raises(ValueError, match="factor_version"):
        FactorSetAssembler().assemble(
            make_spec("v7-version", "V7", "manual"),
            [make_asset("F1")],
            admission_artifacts={"F1": make_admission("F1", factor_version="v1")},
            treatment_selection_artifacts={"F1": make_treatment("F1", factor_version="v2")},
            production=True,
        )


@pytest.mark.parametrize(
    "asset",
    [
        replace(make_asset("F1"), lifecycle_state=LifecycleState.RETIRED),
        replace(make_asset("F1"), health_state=HealthState.RETIRED),
    ],
    ids=["lifecycle-retired", "health-retired"],
)
def test_n10_production_rejects_current_retired_asset_despite_old_approval(asset):
    with pytest.raises(ValueError, match="current (lifecycle|health)"):
        FactorSetAssembler().assemble(
            make_spec("v7-retired", "V7", "manual"),
            [asset],
            admission_artifacts={"F1": make_admission("F1")},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )


def test_n10_production_rejects_admission_not_bound_to_latest_asset_evidence():
    with pytest.raises(ValueError, match="latest evidence"):
        FactorSetAssembler().assemble(
            make_spec("v7-evidence", "V7", "manual"),
            [make_asset("F1")],
            admission_artifacts={
                "F1": make_admission("F1", evidence_refs=("obsolete-bundle",))
            },
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )


@pytest.mark.parametrize("change", ["spec", "snapshot", "universe", "window"])
def test_n11_same_vector_in_different_semantic_domain_is_not_comparable(change):
    old = _fp("old", (1.0, 0.0))
    values = dict(spec="rank-v1", snapshot="s1", universe="u1", window="w1")
    values[change] = "different"
    new = _fp("new", (1.0, 0.0), **values)
    with pytest.raises(ValueError, match="semantic domain"):
        incremental_assign([new], {"C": _cluster()}, {"old": old, "new": new})


def test_n11_mapping_key_must_equal_fingerprint_factor_id():
    old = _fp("old", (1.0, 0.0))
    new = _fp("new", (1.0, 0.0))
    with pytest.raises(ValueError, match="mapping key"):
        incremental_assign([new], {"C": _cluster()}, {"WRONG": old, "new": new})


def test_n12_overlay_enforces_parent_cas_partition_and_new_identity():
    parents = {"A": _cluster("A", "cs2", ("a",)), "B": _cluster("B", "cs2", ("b",))}
    with pytest.raises(ValueError, match="CAS"):
        build_incremental_cluster_version(parents, [_assignment(cid="A", version="cs1")], new_cluster_set_version_id="cs3")
    with pytest.raises(ValueError, match="multiple"):
        build_incremental_cluster_version(
            parents,
            [_assignment(cid="A", version="cs2", parent_hash=_parent_hash(parents)),
             _assignment(cid="B", version="cs2", parent_hash=_parent_hash(parents))],
            new_cluster_set_version_id="cs3",
        )
    with pytest.raises(ValueError, match="differ"):
        build_incremental_cluster_version(parents, [], new_cluster_set_version_id="cs2")


def test_n12_overlay_is_full_snapshot_and_preserves_untouched_cluster():
    parents = {"A": _cluster("A", members=("a",)), "B": _cluster("B", members=("b",))}
    children = build_incremental_cluster_version(
        parents, [_assignment(cid="A", parent_hash=_parent_hash(parents))], new_cluster_set_version_id="cs2"
    )
    assert {child.logical_cluster_id for child in children} == {"A", "B"}
    assert next(c for c in children if c.logical_cluster_id == "B").member_factor_ids == ("b",)


def test_n13_lineage_uses_actual_member_diff_and_rejects_false_claim():
    parent = {"C": _cluster()}
    unchanged_child = replace(parent["C"], cluster_set_version_ref="cs2", content_hash="")
    with pytest.raises(ValueError, match="actual child diff"):
        build_incremental_lineage_edges(parent, [unchanged_child], [_assignment(parent_hash=_parent_hash(parent))])


def test_r28_exact_low_measurement_is_preserved_not_unknown():
    old = _fp("old", (1.0, 0.0))
    new = _fp("new", (0.1, 0.995))
    result = incremental_assign([new], {"C": _cluster()}, {"old": old, "new": new})
    assert result.candidates
    assert result.candidates[0].evidence_status is PairwiseEvidenceStatus.MEASURED_LOW
    assert result.candidates[0].similarity == pytest.approx(0.09999875, rel=1e-5)
    assert result.assignments[0].kind is IncrementalAssignmentKind.PENDING_GLOBAL_REFRESH


def _ann_hashes(fingerprints, members):
    member_hash = canonical_digest(tuple(
        (fid, fingerprints[fid].content_hash) for fid in sorted(members)
    ))
    first = fingerprints[sorted(members)[0]]
    domain = (
        first.embedding_spec, first.snapshot, first.universe, first.window,
        first.preprocessing_ref, first.mask_policy, first.direction,
        first.aggregation_method, first.embedding_model_version,
    )
    return member_hash, canonical_digest(domain)


def test_r28_ann_identity_binds_actual_membership_and_builds_once_per_batch(monkeypatch):
    import factor_assets.clustering.incremental as module

    members = {"m1": _fp("m1", (1.0, 0.0)), "m2": _fp("m2", (0.0, 1.0))}
    queries = [_fp("q1", (0.9, 0.1)), _fp("q2", (0.1, 0.9))]
    mapping = {**members, **{q.factor_id: q for q in queries}}
    member_hash, spec_hash = _ann_hashes(mapping, members)
    bad = ANNIndexArtifact(
        "bad", "annoy", ANNIndexCapability.APPROXIMATE,
        {"embedding_dim": 2}, member_set_hash="wrong",
        embedding_spec_hash=spec_hash,
    )
    with pytest.raises(ValueError, match="actual index membership"):
        incremental_assign(queries, {"C": _cluster(members=tuple(members))}, mapping, ann_index=bad)

    builds = []
    class FakeIndex:
        def __init__(self, **kwargs): pass
        def build(self, ids, matrix): builds.append(tuple(ids))
        def search(self, query, **kwargs): return []
    monkeypatch.setattr(module, "AnnoyANNIndex", FakeIndex)
    good = ANNIndexArtifact(
        "good", "annoy", ANNIndexCapability.APPROXIMATE,
        {"embedding_dim": 2}, member_set_hash=member_hash,
        embedding_spec_hash=spec_hash,
    )
    incremental_assign(
        queries, {"C": _cluster(members=tuple(members))}, mapping, ann_index=good
    )
    assert builds == [("m1", "m2")]
