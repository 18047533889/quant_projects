"""Tests for production-mode assembly, admission-artifact consumption, and
MMR consuming SimilarityArtifact (with backward-compatible bare-float provider)."""

import pytest
from types import MappingProxyType

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.admission import (
    AdmissionDecision,
    FactorAdmissionArtifact,
)
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.factor_set import FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.contracts.similarity import SimilarityArtifact
from factor_assets.selection import SelectionDecision, SelectionReason


def make_asset(factor_id, *, frequency="daily", domains=("price",), state=LifecycleState.APPROVED):
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr=f"identity({factor_id})",
        canonical_hash=f"hash-{factor_id}",
        frequency=frequency,
        domains=domains,
        timing=frequency,
    )
    return FactorAsset(
        metadata=metadata,
        lineage=LineageRef(factor_id=factor_id, parents=()),
        lifecycle_state=state,
        registered_at="2024-01-01T00:00:00Z",
        family="family-a" if factor_id != "F3" else "family-b",
        latest_evidence_ref=EvidenceBundleRef(
            bundle_id=f"bundle-{factor_id}",
            evaluation_run_id="run-1",
            factor_ids=(factor_id,),
            timestamp="2024-01-01T00:00:00Z",
            qe_version="1.0",
        ),
    )


def make_spec(set_id, name, policy, **kwargs):
    kwargs.setdefault("data_snapshot_ref", "snapshot:default")
    kwargs.setdefault("universe_ref", "universe:default")
    kwargs.setdefault("split_ref", "split:default")
    return FactorSetSpec(set_id, name, policy, **kwargs)


def make_decision(factor_id, *, approved=True, timestamp="2024-01-01T00:00:00Z"):
    return SelectionDecision(
        decision_id=f"decision-{factor_id}-{timestamp}",
        factor_id=factor_id,
        approved=approved,
        reason=(
            SelectionReason.APPROVED
            if approved
            else SelectionReason.REJECTED_GATE_FAILURE
        ),
        timestamp=timestamp,
        policy_version="1.0",
        evidence_refs=("evidence-1",),
        gate_results=("gate-1",),
    )


def make_admission(factor_id, **overrides):
    defaults = dict(
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
        evidence_refs=("bundle-" + factor_id,),
        gate_results=("gate-1",),
        policy_ref="policy:1.0",
        created_at="2024-08-01T00:00:00Z",
    )
    defaults.update(overrides)
    return FactorAdmissionArtifact(**defaults)


def make_treatment(factor_id, **overrides):
    """Build a TreatmentSelectionArtifact for the consume-only auto-treatment wiring."""
    from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact

    defaults = dict(
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
    defaults.update(overrides)
    return TreatmentSelectionArtifact(**defaults)


class TestProductionModeMandatory:
    def test_production_requires_complete_membership_provenance(self):
        spec = make_spec("set-1", "Prod", "manual")
        # No admission artifact, no family on the asset -> cluster_id/orientation
        # unresolvable -> production must fail closed.
        asset = make_asset("F1")
        asset = FactorAsset(
            metadata=asset.metadata,
            lineage=asset.lineage,
            lifecycle_state=asset.lifecycle_state,
            registered_at=asset.registered_at,
            family=None,
        )
        with pytest.raises(ValueError, match="production assembly requires"):
            FactorSetAssembler().assemble(spec, [asset], production=True)

    def test_production_succeeds_with_admission_artifacts(self):
        spec = make_spec("set-1", "Prod", "manual")
        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            admission_artifacts={"F1": make_admission("F1")},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )
        (membership,) = result.memberships
        assert membership.health_state_ref == "lifecycle:APPROVED"
        assert membership.cluster_id == 3
        assert membership.orientation == 1
        assert membership.representative_of == "cluster:3"
        assert membership.factor_version == "v1"

    def test_production_fails_when_admission_artifact_missing_health(self):
        spec = make_spec("set-1", "Prod", "manual")
        artifact = make_admission("F1", health_state_ref=None)
        with pytest.raises(ValueError, match="health_state_ref"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"F1": artifact},
                treatment_selection_artifacts={"F1": make_treatment("F1")},
                production=True,
            )

    def test_production_fails_when_admission_artifact_missing_cluster(self):
        spec = make_spec("set-1", "Prod", "manual")
        artifact = make_admission("F1", cluster_id=None)
        with pytest.raises(ValueError, match="cluster_id"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"F1": artifact},
                treatment_selection_artifacts={"F1": make_treatment("F1")},
                production=True,
            )

    def test_production_fails_when_admission_artifact_missing_orientation(self):
        spec = make_spec("set-1", "Prod", "manual")
        artifact = make_admission("F1", orientation=None)
        with pytest.raises(ValueError, match="orientation"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"F1": artifact},
                treatment_selection_artifacts={"F1": make_treatment("F1")},
                production=True,
            )

    def test_production_factor_version_comes_from_admission_artifact(self):
        # factor_version is always derivable from asset metadata (canonical_hash
        # is required), but an admission artifact's factor_version takes
        # precedence when present.
        spec = make_spec("set-1", "Prod", "manual")
        artifact = make_admission("F1", factor_version="v2")
        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            admission_artifacts={"F1": artifact},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )
        (membership,) = result.memberships
        assert membership.factor_version == "v2"

    def test_non_production_keeps_none_fields_without_artifact(self):
        spec = make_spec("set-1", "Dev", "manual")
        result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
        (membership,) = result.memberships
        # Without an admission artifact, health/cluster/orientation all stay
        # None (not fabricated) in non-production mode.
        assert membership.health_state_ref is None
        assert membership.orientation is None
        assert membership.cluster_id is None
        assert membership.representative_of is None

    def test_admission_artifact_key_must_match_factor_id(self):
        spec = make_spec("set-1", "Prod", "manual")
        with pytest.raises(ValueError, match="must match artifact.factor_id"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"WRONG": make_admission("F1")},
            )

    def test_admission_artifacts_must_be_typed(self):
        spec = make_spec("set-1", "Prod", "manual")
        with pytest.raises(TypeError, match="FactorAdmissionArtifact"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1")],
                admission_artifacts={"F1": "not-an-artifact"},
            )

    def test_production_with_non_manual_policy_and_artifacts(self):
        spec = make_spec("set-1", "Prod", "pareto_front")
        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            selection_decisions=[make_decision("F1")],
            admission_artifacts={"F1": make_admission("F1")},
            treatment_selection_artifacts={"F1": make_treatment("F1")},
            production=True,
        )
        (membership,) = result.memberships
        assert membership.health_state_ref == "lifecycle:APPROVED"
        assert membership.cluster_id == 3
        assert membership.orientation == 1


class TestMMRConsumesSimilarityArtifact:
    def test_mmr_uses_primary_view_of_similarity_artifact(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity_provider(a, b):
            if {a, b} == {"F1", "F2"}:
                return SimilarityArtifact(
                    factor_a=a,
                    factor_b=b,
                    views={"rank_corr": 0.95, "pnl_corr": 0.9},
                    snapshot_ref="snapshot:default",
                    window_ref="2024-01-01/2024-12-31",
                    universe_ref="universe:default",
                    producer="qe",
                    created_at="2024-08-01T00:00:00Z",
                )
            return SimilarityArtifact(
                factor_a=a,
                factor_b=b,
                views={"rank_corr": 0.1},
                snapshot_ref="snapshot:default",
                window_ref="2024-01-01/2024-12-31",
                universe_ref="universe:default",
                producer="qe",
                created_at="2024-08-01T00:00:00Z",
            )

        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1"), make_asset("F2"), make_asset("F3")],
            selection_decisions=[
                decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                decided("F3", 0.8, "2024-01-01T00:00:00Z"),
            ],
            similarity_provider=similarity_provider,
        )
        # MMR picks F1 (highest quality) then F3 (diverse), not F2 (redundant).
        assert result.factor_ids == ("F1", "F3")

    def test_mmr_uses_custom_primary_view(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity_provider(a, b):
            if {a, b} == {"F1", "F2"}:
                return SimilarityArtifact(
                    factor_a=a,
                    factor_b=b,
                    views={"rank_corr": 0.1, "pnl_corr": 0.95},
                    snapshot_ref="snapshot:default",
                    window_ref="2024-01-01/2024-12-31",
                    universe_ref="universe:default",
                    producer="qe",
                    created_at="2024-08-01T00:00:00Z",
                    primary_view="pnl_corr",
                )
            return SimilarityArtifact(
                factor_a=a,
                factor_b=b,
                views={"rank_corr": 0.1, "pnl_corr": 0.1},
                snapshot_ref="snapshot:default",
                window_ref="2024-01-01/2024-12-31",
                universe_ref="universe:default",
                producer="qe",
                created_at="2024-08-01T00:00:00Z",
                primary_view="pnl_corr",
            )

        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1"), make_asset("F2"), make_asset("F3")],
            selection_decisions=[
                decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                decided("F3", 0.8, "2024-01-01T00:00:00Z"),
            ],
            similarity_provider=similarity_provider,
        )
        # With pnl_corr as primary, F1 and F2 are near-duplicates -> F3 chosen.
        assert result.factor_ids == ("F1", "F3")

    def test_mmr_artifact_provider_none_value_fails_closed(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity_provider(a, b):
            # F1/F2 pair returns None (unmeasured).  An UNKNOWN similarity must
            # NOT be treated as zero — MMR must fail closed rather than reward
            # exactly the pairs it cannot assess.
            if {a, b} == {"F1", "F2"}:
                return None
            return SimilarityArtifact(
                factor_a=a,
                factor_b=b,
                views={"rank_corr": 0.1},
                snapshot_ref="snapshot:default",
                window_ref="2024-01-01/2024-12-31",
                universe_ref="universe:default",
                producer="qe",
                created_at="2024-08-01T00:00:00Z",
            )

        with pytest.raises(ValueError, match="UNKNOWN"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1"), make_asset("F2"), make_asset("F3")],
                selection_decisions=[
                    decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                    decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                    decided("F3", 0.8, "2024-01-01T00:00:00Z"),
                ],
                similarity_provider=similarity_provider,
            )


class TestBackwardCompatBareFloatProvider:
    def test_bare_float_provider_still_works(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity(a, b):
            if {a, b} == {"F1", "F2"}:
                return 0.95
            return 0.1

        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1"), make_asset("F2"), make_asset("F3")],
            selection_decisions=[
                decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                decided("F3", 0.8, "2024-01-01T00:00:00Z"),
            ],
            similarity_provider=similarity,
        )
        assert result.factor_ids == ("F1", "F3")

    def test_bare_float_provider_negative_similarity_abs_normalized(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity(a, b):
            if {a, b} == {"F1", "F2"}:
                return -0.95  # strong negative correlation -> still redundant
            return 0.1

        result = FactorSetAssembler().assemble(
            spec,
            [make_asset("F1"), make_asset("F2"), make_asset("F3")],
            selection_decisions=[
                decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                decided("F3", 0.8, "2024-01-01T00:00:00Z"),
            ],
            similarity_provider=similarity,
        )
        assert result.factor_ids == ("F1", "F3")

    def test_bare_float_provider_none_fails_closed(self):
        spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

        def decided(factor_id, quality, timestamp):
            d = make_decision(factor_id, timestamp=timestamp)
            object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
            return d

        def similarity(a, b):
            # F1/F2 pair returns None (unmeasured).  An UNKNOWN similarity must
            # NOT be treated as zero — MMR fails closed.
            if {a, b} == {"F1", "F2"}:
                return None
            return 0.1

        with pytest.raises(ValueError, match="UNKNOWN"):
            FactorSetAssembler().assemble(
                spec,
                [make_asset("F1"), make_asset("F2"), make_asset("F3")],
                selection_decisions=[
                    decided("F1", 1.0, "2024-01-01T00:00:00Z"),
                    decided("F2", 0.9, "2024-01-01T00:00:00Z"),
                    decided("F3", 0.8, "2024-01-01T00:00:00Z"),
                ],
                similarity_provider=similarity,
            )
