"""DLIB-FA-012/013/014/015/016 library governance + assembly evidence tests.

- FactorLibraryVersionArtifact: versioned, hash-addressed library snapshot;
  requires cluster_set_version_ref (a library spans many clusters).
- AssemblyEvidence / LibraryCandidateEvidence: typed scores, missing quality
  is NOT_ELIGIBLE never a recency fallback.
- RepresentativeSelection: MAX_IC binds to ic_evidence_refs + window/snapshot/
  split provenance, not a bare float.
- AggregationFitArtifact: IC_WEIGHTED / INVERSE_VARIANCE weights are fit on
  TRAIN only, bound to the fit split.
- SimilarityResult measurement status: vanishing 0.0 score is a deliberate,
  recorded measurement (DLIB-FA-016).
"""

from types import MappingProxyType

import pytest

from factor_assets.contracts.assembly_evidence import (
    AssemblyEvidence,
    LibraryCandidateEvidence,
    AssemblyPolicy,
    EvidenceMaturity,
)
from factor_assets.contracts.library_governance import (
    FactorLibraryStatus,
    FactorLibraryDefinition,
    FactorLibraryMembership,
    FactorLibraryVersionArtifact,
    NewLibraryProposal,
)
from factor_assets.aggregation.specs import AggregationFitArtifact, AggregationSpec, WeightingScheme


# ---------------------------------------------------------------------------
# DLIB-FA-012: factor library governance
# ---------------------------------------------------------------------------


def make_membership(factor_ref="F1", score=0.8):
    return FactorLibraryMembership(
        factor_definition_ref=factor_ref,
        selected_treatment_ref=f"treat_{factor_ref}",
        orientation=1,
        logical_cluster_id="CL_A",
        cluster_set_version_ref="csv1",
        admission_ref=f"ad_{factor_ref}",
        evaluation_ref=f"ev_{factor_ref}",
        assembly_score=score,
        selection_rank=0,
    )


def test_library_version_requires_cluster_set_version_ref():
    """A FactorLibrary spans many clusters — cluster_set_version_ref is
    REQUIRED (DLIB-FA-012)."""
    with pytest.raises(ValueError, match="cluster_set_version_ref"):
        FactorLibraryVersionArtifact(
            library_version_id="v1",
            logical_library_id="LIB_MOM",
            members=(make_membership(),),
            cluster_set_version_ref="",
            selection_policy_ref="sp1",
            evidence_snapshot_ref="es1",
            snapshot_ref="snap1",
            universe_ref="u1",
            created_at="2024-01-01T00:00:00Z",
        )


def test_library_version_is_immutable_versioned_snapshot():
    lib = FactorLibraryVersionArtifact(
        library_version_id="v102",
        logical_library_id="LIB_MOM",
        members=(make_membership("F1"), make_membership("F2", 0.7)),
        cluster_set_version_ref="csv1",
        selection_policy_ref="sp1",
        evidence_snapshot_ref="es1",
        snapshot_ref="snap1",
        universe_ref="u1",
        created_at="2024-01-01T00:00:00Z",
        status=FactorLibraryStatus.PRODUCTION,
    )
    assert lib.content_hash
    assert lib.factor_definition_refs == ("F1", "F2")
    assert lib.status is FactorLibraryStatus.PRODUCTION
    # Promotion changes only the active pointer; the previous version (v101)
    # is preserved, never mutated.
    v101 = FactorLibraryVersionArtifact(
        library_version_id="v101",
        logical_library_id="LIB_MOM",
        members=(make_membership("F1"),),
        cluster_set_version_ref="csv0",
        selection_policy_ref="sp0",
        evidence_snapshot_ref="es0",
        snapshot_ref="snap0",
        universe_ref="u0",
        created_at="2023-01-01T00:00:00Z",
    )
    assert v101.content_hash != lib.content_hash
    # Duplicate factor definitions rejected.
    with pytest.raises(ValueError, match="duplicate"):
        FactorLibraryVersionArtifact(
            library_version_id="v3",
            logical_library_id="LIB_MOM",
            members=(make_membership("F1"), make_membership("F1")),
            cluster_set_version_ref="csv1",
            selection_policy_ref="sp1",
            evidence_snapshot_ref="es1",
            snapshot_ref="snap1",
            universe_ref="u1",
            created_at="2024-01-01T00:00:00Z",
        )


def test_new_library_proposal():
    proposal = NewLibraryProposal(
        proposal_id="prop1",
        logical_library_id="LIB_VOL",
        name="Volume library",
        purpose="Volume-timing library for a low-latency book",
        rationale="Stable volume family emerged from clustering",
    )
    assert proposal.status == "PENDING"
    assert proposal.created_at
    with pytest.raises(ValueError, match="rationale"):
        NewLibraryProposal("p2", "LIB_VOL", "V", "P", "")


def test_library_definition_carries_proxy_policy_refs():
    definition = FactorLibraryDefinition(
        logical_library_id="LIB_MOM",
        name="Momentum library",
        purpose="Cross-sectional momentum for the flagship book",
        selection_policy_ref="sel:v1",
        cluster_budget_policy_ref="cb:v1",
    )
    assert definition.selection_policy_ref == "sel:v1"
    assert definition.to_dict()["logical_library_id"] == "LIB_MOM"


# ---------------------------------------------------------------------------
# DLIB-FA-013: assembly evidence
# ---------------------------------------------------------------------------


def test_assembly_evidence_typed_scores():
    ev = AssemblyEvidence(
        factor_id="F1",
        quality_score=0.8,
        stability_score=0.7,
        evidence_refs=("e1", "e2"),
        maturity=EvidenceMaturity.MATURE,
    )
    assert ev.quality_score == 0.8
    assert ev.content_hash
    assert ev.to_dict()["maturity"] == "MATURE"


def test_assembly_evidence_rejects_nan_quality():
    """A missing/NaN quality is never admissible — a member is never scored on
    a fabricated value (DLIB-FA-013)."""
    with pytest.raises(ValueError, match="finite"):
        AssemblyEvidence("F1", quality_score=float("nan"))


def test_library_candidate_evidence_is_not_eligible_without_quality():
    """AssemblyEvidence with maturity NOT_ELIGIBLE records that a member could
    not be assessed — production assembly must NOT fall back to recency."""
    candidate = LibraryCandidateEvidence(
        factor_definition_ref="F1",
        quality_score=0.0,
        maturity=EvidenceMaturity.NOT_ELIGIBLE,
    )
    assert candidate.maturity is EvidenceMaturity.NOT_ELIGIBLE


def test_assembly_policy_replaces_magic_lambda():
    policy = AssemblyPolicy("pol1", "1.0", quality_weight=0.6, redundancy_weight=0.4)
    assert policy.quality_weight == 0.6
    assert policy.redundancy_weight == 0.4
    assert policy.max_per_microcluster == 1
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        AssemblyPolicy("p2", "1.0", quality_weight=1.5)
    # Legacy behavior preserved by default weights.
    default = AssemblyPolicy("p3", "1.0")
    assert default.quality_weight == 0.5 and default.redundancy_weight == 0.5


def test_assembly_policy_wired_into_diverse_assembler():
    """The MMR diverse policy's magic lambda_weight=0.5 is replaced by the
    typed AssemblyPolicy (DLIB-FA-013)."""
    from factor_assets.assembly.engine import FactorSetAssembler
    from factor_assets.contracts.factor_set import FactorSetSpec
    from factor_assets.contracts.asset import FactorAsset, AssetMetadata
    from factor_assets.contracts.evidence_ref import EvidenceBundleRef
    from factor_assets.contracts.lineage import LineageRef
    from factor_assets.contracts.lifecycle import LifecycleState
    from factor_assets.selection.policy import SelectionDecision, SelectionReason

    def make_asset(fid):
        return FactorAsset(
            metadata=AssetMetadata(
                factor_id=fid,
                canonical_repr=fid,
                canonical_hash=f"hash_{fid}",
                frequency="daily",
                domains=("cross-sectional",),
                timing="daily",
            ),
            lineage=LineageRef(factor_id=fid, parents=()),
            lifecycle_state=LifecycleState.EVALUATED,
            registered_at="2024-01-01T00:00:00Z",
            latest_evidence_ref=EvidenceBundleRef(
                bundle_id=f"bundle_{fid}",
                evaluation_run_id=f"run_{fid}",
                factor_ids=(fid,),
                timestamp="2024-01-01T00:00:00Z",
                qe_version="1.0",
            ),
        )

    spec = FactorSetSpec(
        set_id="set-1",
        name="Diverse",
        selection_policy="diverse",
        universe_ref="u1",
        data_snapshot_ref="snap1",
        split_ref="split1",
        max_factors=2,
    )

    def make_decision(fid, quality):
        decision = SelectionDecision(
            decision_id=f"SD_{fid}_20240101T000000000",
            factor_id=fid,
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=("e1",),
            gate_results=("g1",),
        )
        object.__setattr__(
            decision, "metadata", MappingProxyType({"quality": quality})
        )
        return decision

    def similarity(a, b):
        if {a, b} == {"F1", "F2"}:
            return 0.95
        return 0.1

    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2"), make_asset("F3")],
        selection_decisions=[
            make_decision("F1", 1.0),
            make_decision("F2", 0.9),
            make_decision("F3", 0.8),
        ],
        similarity_provider=similarity,
        assembly_policy=AssemblyPolicy("pol1", "1.0", quality_weight=0.5, redundancy_weight=0.5),
    )
    # MMR picks F1 (highest quality) then F3 (diverse), not F2 (redundant).
    assert result.factor_ids == ("F1", "F3")


# ---------------------------------------------------------------------------
# DLIB-FA-014: MAX_IC binds to evidence refs
# ---------------------------------------------------------------------------


def test_representative_selection_max_ic_binds_evidence():
    from factor_assets.aggregation.representatives import (
        RepresentativeSelection,
        RepresentativeSelectionMethod,
    )

    selection = RepresentativeSelection(
        selection_id="sel1",
        family="family_mom",
        method=RepresentativeSelectionMethod.MAX_IC,
        selected_factor_ids=("F1",),
        candidate_factor_ids=("F1", "F2", "F3"),
        timestamp="2024-01-01T00:00:00Z",
        ic_values=(0.05,),
        ic_evidence_refs=("ev_ref_1",),
        ic_window_ref="2024-01-01/2024-12-31",
        ic_snapshot_ref="snap1",
        ic_split_ref="split1",
    )
    assert selection.ic_evidence_refs == ("ev_ref_1",)
    assert selection.ic_window_ref is not None
    # Mismatched evidence refs length fails closed.
    with pytest.raises(ValueError, match="ic_evidence_refs"):
        RepresentativeSelection(
            selection_id="sel2",
            family="family_mom",
            method=RepresentativeSelectionMethod.MAX_IC,
            selected_factor_ids=("F1", "F2"),
            timestamp="2024-01-01T00:00:00Z",
            ic_evidence_refs=("ev_ref_1",),  # only one for two factors
        )


# ---------------------------------------------------------------------------
# DLIB-FA-015: aggregation weights fit on TRAIN only
# ---------------------------------------------------------------------------


def test_aggregation_fit_requires_split_ref():
    spec = AggregationSpec(
        spec_id="agg1",
        name="A1",
        weighting_scheme=WeightingScheme.IC_WEIGHTED,
        factor_ids=("F1", "F2"),
    )
    with pytest.raises(ValueError, match="fit_split_ref"):
        AggregationFitArtifact(
            fit_id="fit1",
            spec=spec,
            fit_split_ref="",  # must be TRAIN split
            fit_window_ref="2024",
            fit_snapshot_ref="snap1",
            fit_universe_ref="u1",
            fit_method="qequantile",
            computed_weights=(0.6, 0.4),
        )
    fit = AggregationFitArtifact(
        fit_id="fit1",
        spec=spec,
        fit_split_ref="train/2024H1",
        fit_window_ref="2024",
        fit_snapshot_ref="snap1",
        fit_universe_ref="u1",
        fit_method="rank_ic_normalized",
        computed_weights=(0.6, 0.4),
    )
    assert fit.fit_split_ref == "train/2024H1"
    assert fit.created_at


# ---------------------------------------------------------------------------
# DLIB-FA-016: SimilarityResult measurement status
# ---------------------------------------------------------------------------


def test_similarity_result_zero_is_a_recorded_measurement():
    from factor_assets.similarity.exact import (
        SimilarityResult,
        SimilarityMethod,
        SimilarityMeasurementStatus,
    )

    computed_zero = SimilarityResult(
        factor_id_a="F1",
        factor_id_b="F2",
        similarity_score=0.0,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=100,
        measurement_status=SimilarityMeasurementStatus.COMPUTED_VALUE,
    )
    assert computed_zero.value_is_admissible is True
    assert computed_zero.is_high_similarity() is False  # genuinely 0.0

    unknown = SimilarityResult(
        factor_id_a="F1",
        factor_id_b="F2",
        similarity_score=0.0,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=0,
        measurement_status=SimilarityMeasurementStatus.UNKNOWN,
    )
    # An UNKNOWN measurement must NOT be conflated with a computed zero.
    assert unknown.value_is_admissible is False
    assert unknown.is_high_similarity() is False
    assert unknown.is_significant() is False

    # Legacy constructions default to COMPUTED_VALUE (behaviour preserved).
    legacy = SimilarityResult(
        factor_id_a="F1",
        factor_id_b="F2",
        similarity_score=0.8,
        method=SimilarityMethod.PEARSON,
        timestamp="2024-01-01T00:00:00Z",
        sample_size=50,
    )
    assert legacy.measurement_status is SimilarityMeasurementStatus.COMPUTED_VALUE
    assert legacy.is_high_similarity(0.7) is True