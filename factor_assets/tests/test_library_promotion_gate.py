"""QRP-P7 library promotion / rollback gate tests.

Covers the fail-closed promotion rules (label not mature / wrong return basis /
low rank_ic / duplicate similarity), the approval path, the merge-suggested
REVIEW path, artifact immutability (derived-only content hash), and rollback
lineage reachability / unreachability.
"""

from __future__ import annotations

import pytest

from factor_assets.contracts.library_governance import (
    FactorLibraryStatus,
    FactorLibraryMembership,
    FactorLibraryVersionArtifact,
)
from factor_assets.library import (
    VWAP_TO_VWAP_BASIS,
    DEFAULT_MIN_RANK_IC,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_REJECT_DUPLICATES,
    PromotionDecision,
    PromotionReasonCode,
    CandidateEvaluationRef,
    PromotionDecisionArtifact,
    RollbackDecisionArtifact,
    PromotionGate,
    RollbackGate,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_membership(factor_ref: str = "F1", score: float = 0.8) -> FactorLibraryMembership:
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


def make_library(version_id: str = "v102", factor_refs=("F1", "F2")) -> FactorLibraryVersionArtifact:
    return FactorLibraryVersionArtifact(
        library_version_id=version_id,
        logical_library_id="LIB_MOM",
        members=tuple(make_membership(f) for f in factor_refs),
        cluster_set_version_ref="csv1",
        selection_policy_ref="sp1",
        evidence_snapshot_ref="es1",
        snapshot_ref="snap1",
        universe_ref="u1",
        created_at="2024-01-01T00:00:00Z",
        status=FactorLibraryStatus.PRODUCTION,
    )


def gate(**kwargs) -> PromotionGate:
    return PromotionGate(**kwargs)


def eval_ref(candidate_ref: str = "F9", **overrides) -> CandidateEvaluationRef:
    base = dict(
        candidate_ref=candidate_ref,
        rank_ic=0.05,
        return_basis=VWAP_TO_VWAP_BASIS,
    )
    base.update(overrides)
    return CandidateEvaluationRef(**base)


# ---------------------------------------------------------------------------
# reject: label not mature
# ---------------------------------------------------------------------------


def test_rejects_label_not_mature_flag():
    artifact = gate().evaluate("v102", eval_ref(label_maturity=False))
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.LABEL_NOT_MATURE in artifact.reason_codes


@pytest.mark.parametrize("status", ["not_computed", "label_not_mature", "insufficient_data", "failed"])
def test_rejects_not_computed_evidence_statuses(status):
    artifact = gate().evaluate("v102", eval_ref(evidence_status=status))
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.LABEL_NOT_MATURE in artifact.reason_codes


def test_computed_evidence_status_does_not_reject():
    artifact = gate().evaluate("v102", eval_ref(evidence_status="computed"))
    assert artifact.decision is PromotionDecision.APPROVE
    assert PromotionReasonCode.LABEL_NOT_MATURE not in artifact.reason_codes


# ---------------------------------------------------------------------------
# reject: wrong return basis (defensive; whole warehouse is vwap to vwap)
# ---------------------------------------------------------------------------


def test_rejects_wrong_return_basis():
    artifact = gate().evaluate("v102", eval_ref(return_basis="close_to_close"))
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.RETURN_BASIS_WRONG in artifact.reason_codes


# ---------------------------------------------------------------------------
# reject: rank_ic below / missing
# ---------------------------------------------------------------------------


def test_rejects_rank_ic_below_threshold():
    artifact = gate().evaluate("v102", eval_ref(rank_ic=0.005))
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.RANK_IC_BELOW_THRESHOLD in artifact.reason_codes


def test_rejects_missing_rank_ic():
    artifact = gate().evaluate("v102", eval_ref(rank_ic=None))
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.RANK_IC_BELOW_THRESHOLD in artifact.reason_codes


# ---------------------------------------------------------------------------
# reject / review: duplicate similarity to existing member
# ---------------------------------------------------------------------------


def test_rejects_duplicate_by_similarity():
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F1"),
        library_member_refs=("F1",),
        similarity_fn=lambda c, m: 0.99,
    )
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.DUPLICATE_OF_EXISTING_MEMBER in artifact.reason_codes


def test_similarity_equal_to_threshold_is_not_duplicate():
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1",),
        similarity_fn=lambda c, m: 0.7,  # threshold boundary: must be STRICTLY above
    )
    assert artifact.decision is PromotionDecision.APPROVE
    assert PromotionReasonCode.QUALIFIES in artifact.reason_codes


def test_merge_suggested_when_duplicates_not_rejected():
    artifact = gate(reject_duplicates=False).evaluate(
        "v102",
        eval_ref(candidate_ref="F1"),
        library_member_refs=("F1",),
        similarity_fn=lambda c, m: 0.95,
    )
    assert artifact.decision is PromotionDecision.REVIEW
    assert PromotionReasonCode.MERGE_SUGGESTED in artifact.reason_codes


def test_unmeasured_similarity_forces_review():
    artifact = gate(reject_duplicates=False).evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1",),
        similarity_fn=lambda c, m: None,
    )
    assert artifact.decision is PromotionDecision.REVIEW
    assert PromotionReasonCode.MERGE_SUGGESTED in artifact.reason_codes


def test_missing_similarity_fn_with_members_forces_review():
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1", "F2"),
        similarity_fn=None,
    )
    assert artifact.decision is PromotionDecision.REVIEW
    assert PromotionReasonCode.MERGE_SUGGESTED in artifact.reason_codes


def test_similarity_fn_non_finite_fails_closed():
    with pytest.raises(ValueError, match="non-finite"):
        gate().evaluate(
            "v102",
            eval_ref(candidate_ref="F9"),
            library_member_refs=("F1",),
            similarity_fn=lambda c, m: float("nan"),
        )


# ---------------------------------------------------------------------------
# approve path
# ---------------------------------------------------------------------------


def test_approves_qualified_candidate():
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1", "F2"),
        similarity_fn=lambda c, m: 0.1,
    )
    assert artifact.decision is PromotionDecision.APPROVE
    assert artifact.reason_codes == (PromotionReasonCode.QUALIFIES,)
    assert artifact.candidate_ref == "F9"
    assert artifact.library_version_ref == "v102"


def test_approve_binds_evaluation_and_treatment_refs():
    artifact = gate().evaluate(
        "v102",
        eval_ref(
            candidate_ref="F9",
            evaluation_ref="QE_EVAL_9",
            treatment_optimization_ref="FO_TREAT_9",
        ),
    )
    assert artifact.evaluation_ref == "QE_EVAL_9"
    assert artifact.treatment_optimization_ref == "FO_TREAT_9"


def test_promotion_decision_binds_real_library_artifact():
    lib = make_library()
    artifact = gate().evaluate(
        lib.content_hash,
        eval_ref(candidate_ref="F9"),
    )
    assert artifact.library_version_ref == lib.content_hash


# ---------------------------------------------------------------------------
# artifact immutability / fail-closed
# ---------------------------------------------------------------------------


def test_artifact_reject_requires_reject_reason():
    with pytest.raises(ValueError, match="REJECT requires"):
        PromotionDecisionArtifact(
            decision=PromotionDecision.REJECT,
            reason_codes=(PromotionReasonCode.QUALIFIES,),
            candidate_ref="F9",
            library_version_ref="v102",
            observation_metadata={},
        )


def test_artifact_approve_must_not_carry_reject_reason():
    with pytest.raises(ValueError, match="reject reason codes"):
        PromotionDecisionArtifact(
            decision=PromotionDecision.APPROVE,
            reason_codes=(
                PromotionReasonCode.QUALIFIES,
                PromotionReasonCode.LABEL_NOT_MATURE,
            ),
            candidate_ref="F9",
            library_version_ref="v102",
            observation_metadata={},
        )


def test_artifact_hash_is_derived_only():
    artifact = gate().evaluate("v102", eval_ref())
    tampered = artifact.to_dict()
    tampered["candidate_ref"] = "DIFFERENT"
    with pytest.raises(ValueError, match="content hash"):
        PromotionDecisionArtifact.from_dict(tampered)


def test_artifact_roundtrip():
    artifact = gate().evaluate("v102", eval_ref())
    restored = PromotionDecisionArtifact.from_dict(artifact.to_dict())
    assert restored.decision is artifact.decision
    assert restored.reason_codes == artifact.reason_codes
    assert restored.content_hash == artifact.content_hash


def test_gate_threshold_validation():
    with pytest.raises(ValueError, match="non-negative"):
        gate(min_rank_ic=-0.01)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        gate(duplicate_similarity_threshold=1.5)
    with pytest.raises(ValueError, match="finite"):
        gate(min_rank_ic=float("nan"))


# ---------------------------------------------------------------------------
# rollback lineage
# ---------------------------------------------------------------------------


def test_rollback_approves_when_target_is_ancestor():
    lineage = {"v102": ("v101",), "v101": ("v100",), "v100": ("v99",), "v99": ()}
    artifact = RollbackGate().validate("v102", "v100", lineage)
    assert artifact.decision is PromotionDecision.APPROVE
    assert artifact.lineage_path == ("v101", "v100")


def test_rollback_approves_direct_parent():
    artifact = RollbackGate().validate("v102", "v101", {"v102": ("v101",), "v101": ()})
    assert artifact.decision is PromotionDecision.APPROVE
    assert artifact.lineage_path == ("v101",)


def test_rollback_rejects_unrelated_version():
    lineage = {
        "v102": ("v101",),
        "v101": ("v100",),
        "v100": (),
        "unrelated": ("v1x",),
        "v1x": (),
    }
    artifact = RollbackGate().validate("v102", "unrelated", lineage)
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.LINEAGE_NOT_REACHABLE in artifact.reason_codes
    assert artifact.lineage_path == ()


def test_rollback_rejects_when_lineage_walk_ends():
    # v99 is never recorded as a parent — the walk terminates without reaching it.
    lineage = {"v102": ("v101",), "v101": ("v100",), "v100": ()}
    artifact = RollbackGate().validate("v102", "v99", lineage)
    assert artifact.decision is PromotionDecision.REJECT
    assert artifact.lineage_path == ()


def test_rollback_rejects_same_version():
    with pytest.raises(ValueError, match="cannot roll back to itself"):
        RollbackGate().validate("v102", "v102", {"v102": ()})


def test_rollback_rejects_cycle_without_target():
    # A cycle (v102 -> v101 -> v102) must terminate without reaching v99.
    lineage = {"v102": ("v101",), "v101": ("v102",)}
    artifact = RollbackGate().validate("v102", "v99", lineage)
    assert artifact.decision is PromotionDecision.REJECT
    assert artifact.lineage_path == ()


def test_rollback_artifact_roundtrip():
    artifact = RollbackGate().validate(
        "v102", "v100", {"v102": ("v101",), "v101": ("v100",), "v100": ()}
    )
    restored = RollbackDecisionArtifact.from_dict(artifact.to_dict())
    assert restored.lineage_path == ("v101", "v100")
    assert restored.content_hash == artifact.content_hash


def test_rollback_reject_requires_right_reason_code():
    with pytest.raises(ValueError, match="exactly the reason"):
        RollbackDecisionArtifact(
            decision=PromotionDecision.APPROVE,
            reason_codes=(PromotionReasonCode.LINEAGE_NOT_REACHABLE,),
            current_library_version_ref="v102",
            target_library_version_ref="v100",
            lineage_path=(),
        )


# ---------------------------------------------------------------------------
# defaults match FA conventions
# ---------------------------------------------------------------------------


def test_defaults_match_fa_conventions():
    assert DEFAULT_MIN_RANK_IC == 0.02  # MinimumICGate admission IC floor
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.7  # admission similarity threshold
    assert DEFAULT_REJECT_DUPLICATES is True
    assert VWAP_TO_VWAP_BASIS == "vwap_to_vwap"  # LabelBundle.price_convention


# ---------------------------------------------------------------------------
# positive / protective counter-examples (fail-closed can never silently PAN)
# ---------------------------------------------------------------------------


def test_wrong_return_basis_not_flagged_when_basis_correct():
    artifact = gate().evaluate("v102", eval_ref(return_basis=VWAP_TO_VWAP_BASIS))
    assert artifact.decision is PromotionDecision.APPROVE
    assert PromotionReasonCode.RETURN_BASIS_WRONG not in artifact.reason_codes


def test_rank_ic_above_threshold_not_rejected():
    artifact = gate().evaluate("v102", eval_ref(rank_ic=DEFAULT_MIN_RANK_IC + 0.001))
    assert PromotionReasonCode.RANK_IC_BELOW_THRESHOLD not in artifact.reason_codes


def test_all_measured_below_threshold_approves():
    # Every member measured low: uniqueness IS proven -> APPROVE (never confused
    # with an unmeasured None).
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1", "F2"),
        similarity_fn=lambda c, m: {"F1": 0.1, "F2": 0.2}.get(m, 0.0),
    )
    assert artifact.decision is PromotionDecision.APPROVE
    assert artifact.reason_codes == (PromotionReasonCode.QUALIFIES,)


def test_partial_unmeasured_similarity_forces_review():
    # UNKNOWN for ANY member is unprovable uniqueness — REVIEW, never a silent
    # APPROVE just because the *measured* ones are all below threshold.
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1", "F2"),
        similarity_fn=lambda c, m: None if m == "F1" else 0.1,
    )
    assert artifact.decision is PromotionDecision.REVIEW
    assert PromotionReasonCode.MERGE_SUGGESTED in artifact.reason_codes


def test_reject_takes_precedence_over_unmeasured_review():
    # Fail-closed REJECT reasons must never be downgraded to REVIEW by the
    # unmeasurable-similarity branch (label not mature + unmeasurable -> REJECT).
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9", label_maturity=False),
        library_member_refs=("F1", "F2"),
        similarity_fn=None,
    )
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.LABEL_NOT_MATURE in artifact.reason_codes
    assert PromotionReasonCode.MERGE_SUGGESTED not in artifact.reason_codes


def test_duplicate_reject_takes_precedence_over_unmeasured():
    # A measured duplicate above threshold stays REJECT even when a sibling
    # member is unmeasured.
    artifact = gate().evaluate(
        "v102",
        eval_ref(candidate_ref="F9"),
        library_member_refs=("F1", "F2"),
        similarity_fn=lambda c, m: None if m == "F2" else 0.95,
    )
    assert artifact.decision is PromotionDecision.REJECT
    assert PromotionReasonCode.DUPLICATE_OF_EXISTING_MEMBER in artifact.reason_codes


def test_promotion_artifact_is_immutable():
    artifact = gate().evaluate("v102", eval_ref())
    with pytest.raises(AttributeError):
        artifact.candidate_ref = "DIFFERENT"  # noqa: B018 — frozen dataclass


def test_rollback_decision_artifact_is_immutable():
    artifact = RollbackGate().validate(
        "v102", "v100", {"v102": ("v101",), "v101": ("v100",), "v100": ()}
    )
    with pytest.raises(AttributeError):
        artifact.lineage_path = ()  # noqa: B018 — frozen dataclass