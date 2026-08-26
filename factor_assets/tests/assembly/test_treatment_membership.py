"""Tests for FactorSet membership auto-treatment refs (FA track).

FA is CONSUME-only for auto-treatment: it carries ``treatment_selection_ref`` /
``preprocess_policy_ref`` / ``preprocess_state_ref`` on membership, populated
from a TreatmentSelectionArtifact, and never recomputes or fabricates the
selected treatment.  In production mode assembly fails closed when a
treatment_selection_ref is required but missing.
"""

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.admission import (
    AdmissionDecision,
    FactorAdmissionArtifact,
)
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.factor_set import FactorMembership, FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact


def make_asset(factor_id):
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr=f"identity({factor_id})",
        canonical_hash=f"hash-{factor_id}",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    return FactorAsset(
        metadata=metadata,
        lineage=LineageRef(factor_id=factor_id, parents=()),
        lifecycle_state=LifecycleState.APPROVED,
        registered_at="2024-01-01T00:00:00Z",
        family="family-a",
        latest_evidence_ref=EvidenceBundleRef(
            bundle_id=f"bundle-{factor_id}",
            evaluation_run_id="run-1",
            factor_ids=(factor_id,),
            timestamp="2024-01-01T00:00:00Z",
            qe_version="1.0",
        ),
    )


def make_spec(set_id, name, policy="manual", **kwargs):
    kwargs.setdefault("data_snapshot_ref", "snapshot:default")
    kwargs.setdefault("universe_ref", "universe:default")
    kwargs.setdefault("split_ref", "split:default")
    return FactorSetSpec(set_id, name, policy, **kwargs)


def make_admission(factor_id):
    return FactorAdmissionArtifact(
        factor_id=factor_id,
        decision=AdmissionDecision.APPROVED,
        quality=0.8,
        factor_version="v1",
        health_state_ref="lifecycle:APPROVED",
        similarity_ref="sim-hash-abc",
        novelty_ref="novelty-1",
        cluster_id=3,
        orientation=1,
        reason="APPROVED",
        evidence_refs=(f"bundle-{factor_id}",),
        gate_results=("gate-1",),
        policy_ref="policy:1.0",
        created_at="2024-08-01T00:00:00Z",
    )


def make_treatment(factor_id):
    return TreatmentSelectionArtifact(
        factor_id=factor_id,
        factor_version="v1",
        raw_baseline_evidence_ref="ref:raw_baseline",
        factor_profile_ref="ref:profile",
        eligibility_policy_ref="ref:eligibility",
        search_space_ref="ref:search_space",
        all_trial_refs=("ref:trial_a",),
        pareto_candidate_refs=("ref:p_1",),
        winner_recipe={"preprocess": "zscore", "winsorize": 0.01},
        winner_policy_identity="policy:auto-treat/v3",
        absolute_metric_refs={"sharpe": "ref:sharpe"},
        delta_metric_refs={"delta_sharpe": "ref:delta_sharpe"},
        dimension_scores={"return": 0.8, "robustness": 0.65},
        hard_gate_results={"min_obs": "PASS", "non_nan": "PASS"},
        soft_floor_results={"min_sharpe": 0.2},
        robustness_evidence="ref:robustness",
        complexity_score=0.42,
        snapshot_ref="ref:snapshot",
        universe_ref="ref:universe",
        split_ref="ref:split",
        created_at="2026-08-25T00:00:00+00:00",
    )


# --- membership field validation ---------------------------------------------


class TestMembershipTreatmentFieldValidation:
    def test_fields_default_to_none(self):
        m = FactorMembership(factor_id="F1")
        assert m.treatment_selection_ref is None
        assert m.preprocess_policy_ref is None
        assert m.preprocess_state_ref is None

    def test_non_empty_strings_accepted(self):
        m = FactorMembership(
            factor_id="F1",
            treatment_selection_ref="ref:treat",
            preprocess_policy_ref="policy:pp1",
            preprocess_state_ref="state:2024-01",
        )
        assert m.treatment_selection_ref == "ref:treat"
        assert m.preprocess_policy_ref == "policy:pp1"
        assert m.preprocess_state_ref == "state:2024-01"

    def test_empty_string_rejected_for_each(self):
        with pytest.raises(ValueError, match="non-empty string"):
            FactorMembership(factor_id="F1", treatment_selection_ref="")
        with pytest.raises(ValueError, match="non-empty string"):
            FactorMembership(factor_id="F1", preprocess_policy_ref="")
        with pytest.raises(ValueError, match="non-empty string"):
            FactorMembership(factor_id="F1", preprocess_state_ref="")

    def test_non_string_rejected(self):
        with pytest.raises(TypeError, match="must be a str or None"):
            FactorMembership(factor_id="F1", treatment_selection_ref=123)


# --- assembler consume-only wiring -------------------------------------------

class TestTreatmentMembershipAssembler:
    def test_production_populates_treatment_selection_ref(self):
        spec = make_spec("set-1", "manual")
        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            admission_artifacts={"F1": make_admission("F1")},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )
        (membership,) = result.memberships
        assert membership.treatment_selection_ref == make_treatment("F1").content_hash

    def test_production_fails_closed_without_treatment_ref(self):
        spec = make_spec("set-1", "manual")
        with pytest.raises(ValueError, match="treatment_selection_ref"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"F1": make_admission("F1")},
                production=True,
            )

    def test_non_production_keeps_none_without_artifact(self):
        spec = make_spec("set-1", "manual")
        result = FactorSetAssembler().assemble(
            spec, [make_asset("F1")], admission_artifacts={"F1": make_admission("F1")}
        )
        (membership,) = result.memberships
        assert membership.treatment_selection_ref is None

    def test_artifact_key_must_match_factor_id(self):
        spec = make_spec("set-1", "manual")
        with pytest.raises(ValueError, match="must match artifact.factor_id"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                treatment_selection_artifacts={"WRONG": make_treatment("F1")},
            )

    def test_treatment_artifacts_must_be_typed(self):
        spec = make_spec("set-1", "manual")
        with pytest.raises(TypeError, match="TreatmentSelectionArtifact"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                treatment_selection_artifacts={"F1": "not-an-artifact"},
            )

    def test_treatment_ref_flows_into_assembly_hash(self):
        spec = make_spec("set-1", "manual")
        base = FactorSetAssembler().assemble(
            spec, [make_asset("F1")], created_at="2024-02-01T00:00:00Z"
        ).assembly_hash
        treated = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            created_at="2024-02-01T00:00:00Z",
        ).assembly_hash
        assert treated != base
