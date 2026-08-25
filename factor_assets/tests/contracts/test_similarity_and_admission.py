"""Tests for the SimilarityArtifact and FactorAdmissionArtifact contracts."""

import dataclasses

import pytest

from factor_assets.contracts.admission import (
    ADMISSION_DECISION_APPROVED,
    ADMISSION_DECISION_REJECTED,
    ADMISSION_DECISION_SHADOWED,
    AdmissionDecision,
    FactorAdmissionArtifact,
)
from factor_assets.contracts.similarity import (
    DEFAULT_SIMILARITY_VIEW,
    SIMILARITY_VIEW_KEYS,
    SimilarityArtifact,
    SimilarityView,
)
from factor_assets.similarity import SimilarityMethod, SimilarityResult
from factor_assets.selection import SelectionDecision, SelectionReason


def _artifact(
    a="F001", b="F002", views=None, snapshot="snapshot:2024-08", window="2024-01-01/2024-12-31",
    universe="universe:ashare", producer="qe", created_at="2024-08-01T00:00:00Z",
    primary_view=DEFAULT_SIMILARITY_VIEW,
):
    return SimilarityArtifact(
        factor_a=a,
        factor_b=b,
        views=views if views is not None else {"rank_corr": 0.9, "pnl_corr": 0.7},
        snapshot_ref=snapshot,
        window_ref=window,
        universe_ref=universe,
        producer=producer,
        created_at=created_at,
        primary_view=primary_view,
    )


def _admission(factor_id="F001", decision=AdmissionDecision.APPROVED, **overrides):
    defaults = dict(
        factor_id=factor_id,
        decision=decision,
        quality=0.8,
        factor_version="v1",
        health_state_ref="lifecycle:APPROVED",
        similarity_ref="sim-hash-abc",
        novelty_ref="novelty-1",
        cluster_id=3,
        orientation=1,
        reason="APPROVED",
        evidence_refs=("bundle-F001",),
        gate_results=("gate-1",),
        policy_ref="policy:1.0",
        created_at="2024-08-01T00:00:00Z",
    )
    defaults.update(overrides)
    return FactorAdmissionArtifact(**defaults)


class TestSimilarityArtifactConstruction:
    def test_basic_construction(self):
        artifact = _artifact()
        assert artifact.factor_a == "F001"
        assert artifact.factor_b == "F002"
        assert artifact.views["rank_corr"] == 0.9
        assert artifact.snapshot_ref == "snapshot:2024-08"
        assert artifact.window_ref == "2024-01-01/2024-12-31"
        assert artifact.universe_ref == "universe:ashare"
        assert artifact.producer == "qe"
        assert artifact.primary_view == DEFAULT_SIMILARITY_VIEW == "rank_corr"

    def test_requires_factor_a(self):
        with pytest.raises(ValueError, match="factor_a"):
            _artifact(a="")

    def test_requires_factor_b(self):
        with pytest.raises(ValueError, match="factor_b"):
            _artifact(b="")

    def test_rejects_same_factor_pair(self):
        with pytest.raises(ValueError, match="must differ"):
            _artifact(a="F1", b="F1")

    def test_requires_at_least_one_non_none_view(self):
        with pytest.raises(ValueError, match="at least one non-None"):
            _artifact(views={"rank_corr": None, "pnl_corr": None})

    def test_rejects_non_finite_view(self):
        with pytest.raises(ValueError, match="finite"):
            _artifact(views={"rank_corr": float("nan")})
        with pytest.raises(ValueError, match="finite"):
            _artifact(views={"rank_corr": float("inf")})

    def test_rejects_bool_view(self):
        with pytest.raises(TypeError, match="non-boolean"):
            _artifact(views={"rank_corr": True})

    def test_mixed_none_views_allowed(self):
        artifact = _artifact(views={"rank_corr": 0.9, "pnl_corr": None})
        assert artifact.views["pnl_corr"] is None
        assert artifact.views["rank_corr"] == 0.9

    def test_allows_legacy_view_keys_beyond_canonical_set(self):
        artifact = _artifact(views={"rank_corr": 0.9, "custom_view": 0.4})
        assert artifact.views["custom_view"] == 0.4

    def test_primary_view_must_exist(self):
        with pytest.raises(ValueError, match="primary_view"):
            _artifact(views={"pnl_corr": 0.9}, primary_view="rank_corr")

    def test_primary_view_must_be_non_none(self):
        with pytest.raises(ValueError, match="primary_view"):
            _artifact(views={"rank_corr": None, "pnl_corr": 0.9})

    def test_primary_view_none_requires_single_view(self):
        artifact = _artifact(views={"rank_corr": 0.9}, primary_view=None)
        assert artifact.primary_value == 0.9
        with pytest.raises(ValueError, match="exactly one view"):
            _artifact(views={"rank_corr": 0.9, "pnl_corr": 0.8}, primary_view=None)

    def test_frozen(self):
        artifact = _artifact()
        with pytest.raises(dataclasses.FrozenInstanceError):
            artifact.factor_a = "X"  # type: ignore

    def test_default_snapshot_window_universe_are_none(self):
        artifact = SimilarityArtifact(factor_a="A", factor_b="B", views={"rank_corr": 0.5})
        assert artifact.snapshot_ref is None
        assert artifact.window_ref is None
        assert artifact.universe_ref is None

    def test_view_keys_constant(self):
        assert "rank_corr" in SIMILARITY_VIEW_KEYS
        assert "pnl_corr" in SIMILARITY_VIEW_KEYS
        assert "top_overlap" in SIMILARITY_VIEW_KEYS
        assert "bottom_overlap" in SIMILARITY_VIEW_KEYS
        assert "residual_similarity" in SIMILARITY_VIEW_KEYS
        assert "horizon_similarity" in SIMILARITY_VIEW_KEYS


class TestSimilarityArtifactHash:
    def test_hash_is_deterministic_and_content_sensitive(self):
        first = _artifact()
        second = _artifact()
        assert first.similarity_spec_hash == second.similarity_spec_hash

        # Different view value -> different hash.
        different_view = _artifact(views={"rank_corr": 0.91, "pnl_corr": 0.7})
        assert different_view.similarity_spec_hash != first.similarity_spec_hash

        # Different snapshot -> different hash.
        different_snapshot = _artifact(snapshot="snapshot:2024-09")
        assert different_snapshot.similarity_spec_hash != first.similarity_spec_hash

        # Different factor pair -> different hash.
        different_pair = _artifact(a="F001", b="F003")
        assert different_pair.similarity_spec_hash != first.similarity_spec_hash

    def test_hash_is_stable_across_producer_and_created_at(self):
        first = _artifact(producer="qe", created_at="2024-08-01T00:00:00Z")
        second = _artifact(producer="manual", created_at="2025-01-01T00:00:00Z")
        assert first.similarity_spec_hash == second.similarity_spec_hash

    def test_hash_is_stable_across_view_insertion_order(self):
        ordered = _artifact(views={"rank_corr": 0.9, "pnl_corr": 0.7})
        unordered = _artifact(views={"pnl_corr": 0.7, "rank_corr": 0.9})
        assert ordered.similarity_spec_hash == unordered.similarity_spec_hash

    def test_supplied_hash_is_preserved_when_it_matches(self):
        # A caller-supplied hash is only accepted when it equals the hash
        # recomputed from the views/provenance (identity is derived, not
        # self-reported).  Passing the recomputed hash round-trips exactly.
        recomputed = SimilarityArtifact(
            factor_a="F001", factor_b="F002", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", recomputed, views={"rank_corr": 0.8}
        )
        assert artifact.similarity_spec_hash == recomputed
        assert artifact.primary_value == 0.8

    def test_for_spec_rejects_mismatched_supplied_hash(self):
        # A caller must not self-report an arbitrary hash: deserialize with a
        # stored hash that does not match the recomputed spec hash fails closed.
        with pytest.raises(ValueError, match="does not match"):
            SimilarityArtifact.for_spec(
                "F001", "F002", "not-the-real-hash", views={"rank_corr": 0.8}
            )

    def test_for_spec_accepts_recomputed_hash(self):
        recomputed = SimilarityArtifact(
            factor_a="F001", factor_b="F002", views={"rank_corr": 0.8}
        ).similarity_spec_hash
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", recomputed, views={"rank_corr": 0.8}
        )
        assert artifact.similarity_spec_hash == recomputed

    def test_for_spec_requires_hash(self):
        # An empty hash falls back to auto-computation (still valid).
        artifact = SimilarityArtifact.for_spec(
            "F001", "F002", "", views={"rank_corr": 0.8}
        )
        assert len(artifact.similarity_spec_hash) == 64

    def test_primary_value_and_view_accessor(self):
        artifact = _artifact()
        assert artifact.primary_value == 0.9
        assert artifact.view("pnl_corr") == 0.7
        assert artifact.view("missing") is None

    def test_to_dict_roundtrip(self):
        artifact = _artifact()
        data = artifact.to_dict()
        assert data["factor_a"] == "F001"
        assert data["views"] == {"rank_corr": 0.9, "pnl_corr": 0.7}
        assert data["similarity_spec_hash"] == artifact.similarity_spec_hash
        restored = SimilarityArtifact(
            factor_a=data["factor_a"],
            factor_b=data["factor_b"],
            views=data["views"],
            snapshot_ref=data["snapshot_ref"],
            window_ref=data["window_ref"],
            universe_ref=data["universe_ref"],
            similarity_spec_hash=data["similarity_spec_hash"],
            primary_view=data["primary_view"],
            created_at=artifact.created_at,
            producer=artifact.producer,
        )
        assert restored == artifact

    def test_from_similarity_result_adaptation(self):
        result = SimilarityResult(
            factor_id_a="F001",
            factor_id_b="F002",
            similarity_score=0.85,
            method=SimilarityMethod.PEARSON,
            timestamp="2024-01-01T00:00:00Z",
            sample_size=1000,
            universe_ref="US_500",
            period_start="2024-01-01",
            period_end="2024-12-31",
        )
        artifact = SimilarityArtifact.from_similarity_result(
            result, snapshot_ref="snapshot:2024-08"
        )
        assert artifact.views["rank_corr"] == 0.85
        assert artifact.universe_ref == "US_500"
        assert artifact.window_ref == "2024-01-01/2024-12-31"
        assert artifact.snapshot_ref == "snapshot:2024-08"
        assert artifact.producer == "legacy:SimilarityResult"

    def test_with_provenance_recomputes_hash(self):
        artifact = _artifact()
        augmented = artifact.with_provenance(
            snapshot_ref="snapshot:2025", universe_ref="universe:hs300"
        )
        assert augmented.snapshot_ref == "snapshot:2025"
        assert augmented.universe_ref == "universe:hs300"
        assert augmented.window_ref == artifact.window_ref
        assert augmented.factor_a == artifact.factor_a
        # Provenance change is a change to measurement identity: the spec hash
        # must be recomputed, not carried over stale.
        assert augmented.similarity_spec_hash != artifact.similarity_spec_hash
        recomputed = SimilarityArtifact(
            factor_a="F001",
            factor_b="F002",
            views={"rank_corr": 0.9, "pnl_corr": 0.7},
            snapshot_ref="snapshot:2025",
            window_ref="2024-01-01/2024-12-31",
            universe_ref="universe:hs300",
        )
        assert augmented.similarity_spec_hash == recomputed.similarity_spec_hash
        assert augmented.created_at == artifact.created_at

    def test_views_are_immutable_deep_frozen(self):
        from types import MappingProxyType

        artifact = _artifact()
        assert isinstance(artifact.views, MappingProxyType)
        with pytest.raises(TypeError):
            artifact.views["rank_corr"] = 0.5  # type: ignore[misc]

    def test_views_snapshot_isolated_from_caller_dict(self):
        source = {"rank_corr": 0.9}
        artifact = SimilarityArtifact(
            factor_a="A", factor_b="B", views=source
        )
        # Mutating the caller's dict must not mutate the artifact's snapshot.
        source["rank_corr"] = 0.0
        assert artifact.views["rank_corr"] == 0.9
        assert artifact.primary_value == 0.9


class TestSimilarityView:
    def test_similarity_view_construction(self):
        view = SimilarityView(key="rank_corr", value=0.9)
        assert view.key == "rank_corr"
        assert view.value == 0.9
        assert view.is_absolute is True

    def test_similarity_view_requires_key(self):
        with pytest.raises(ValueError, match="key"):
            SimilarityView(key="", value=0.5)

    def test_similarity_view_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            SimilarityView(key="rank_corr", value=float("nan"))


class TestFactorAdmissionArtifact:
    def test_basic_construction(self):
        artifact = _admission()
        assert artifact.factor_id == "F001"
        assert artifact.decision is ADMISSION_DECISION_APPROVED
        assert artifact.quality == 0.8
        assert artifact.factor_version == "v1"
        assert artifact.health_state_ref == "lifecycle:APPROVED"
        assert artifact.similarity_ref == "sim-hash-abc"
        assert artifact.novelty_ref == "novelty-1"
        assert artifact.cluster_id == 3
        assert artifact.orientation == 1
        assert artifact.reason == "APPROVED"
        assert artifact.evidence_refs == ("bundle-F001",)
        assert artifact.gate_results == ("gate-1",)
        assert artifact.policy_ref == "policy:1.0"

    def test_frozen(self):
        artifact = _admission()
        with pytest.raises(dataclasses.FrozenInstanceError):
            artifact.factor_id = "X"  # type: ignore

    def test_requires_factor_id(self):
        with pytest.raises(ValueError, match="factor_id"):
            _admission(factor_id="")

    def test_requires_decision_enum(self):
        with pytest.raises(TypeError, match="AdmissionDecision"):
            _admission(decision="APPROVED")

    def test_approved_requires_evidence_refs(self):
        with pytest.raises(ValueError, match="evidence_refs"):
            _admission(evidence_refs=())

    def test_approved_requires_gate_results(self):
        with pytest.raises(ValueError, match="gate_results"):
            _admission(gate_results=())

    def test_rejected_does_not_require_evidence(self):
        artifact = _admission(decision=AdmissionDecision.REJECTED, evidence_refs=(), gate_results=())
        assert artifact.is_rejected
        assert not artifact.is_approved

    def test_shadowed_status(self):
        artifact = _admission(decision=AdmissionDecision.SHADOWED, evidence_refs=(), gate_results=())
        assert artifact.is_shadowed
        assert artifact.decision is ADMISSION_DECISION_SHADOWED

    def test_orientation_validation(self):
        with pytest.raises(ValueError, match="orientation"):
            _admission(orientation=0)
        with pytest.raises(ValueError, match="orientation"):
            _admission(orientation=2)

    def test_quality_must_be_finite(self):
        with pytest.raises(ValueError, match="finite"):
            _admission(quality=float("nan"))

    def test_decision_properties(self):
        assert _admission().is_approved
        assert not _admission().is_rejected
        assert ADMISSION_DECISION_APPROVED.value == "APPROVED"
        assert ADMISSION_DECISION_REJECTED.value == "REJECTED"
        assert ADMISSION_DECISION_SHADOWED.value == "SHADOWED"


class TestFactorAdmissionArtifactHash:
    def test_hash_is_deterministic_and_content_sensitive(self):
        first = _admission()
        second = _admission()
        assert first.content_hash == second.content_hash

        changed_cluster = _admission(cluster_id=4)
        assert changed_cluster.content_hash != first.content_hash

        changed_health = _admission(health_state_ref="lifecycle:PRODUCTION_READY")
        assert changed_health.content_hash != first.content_hash

        changed_quality = _admission(quality=0.9)
        assert changed_quality.content_hash != first.content_hash

        changed_reason = _admission(reason="SHADOWED_COEXIST")
        assert changed_reason.content_hash != first.content_hash

    def test_hash_is_stable_across_created_at(self):
        first = _admission(created_at="2024-08-01T00:00:00Z")
        second = _admission(created_at="2025-01-01T00:00:00Z")
        assert first.content_hash == second.content_hash

    def test_hash_changes_with_gate_results(self):
        first = _admission()
        second = _admission(gate_results=("gate-1", "gate-2"))
        assert second.content_hash != first.content_hash

    def test_from_selection_decision_approved(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=True,
            reason=SelectionReason.APPROVED,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=("e1",),
            gate_results=("g1",),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(
            decision,
            factor_version="v1",
            quality=0.8,
            health_state_ref="lifecycle:APPROVED",
            cluster_id=2,
            orientation=1,
            policy_ref="policy:1.0",
        )
        assert artifact.decision is AdmissionDecision.APPROVED
        assert artifact.factor_version == "v1"
        assert artifact.cluster_id == 2
        assert artifact.orientation == 1
        assert artifact.evidence_refs == ("e1",)
        assert artifact.gate_results == ("g1",)
        assert artifact.policy_ref == "policy:1.0"
        # content_hash is derived and deterministic: an artifact built from the
        # same decision content recomputes the identical hash.
        rebuilt = FactorAdmissionArtifact.from_selection_decision(
            decision,
            factor_version="v1",
            quality=0.8,
            health_state_ref="lifecycle:APPROVED",
            cluster_id=2,
            orientation=1,
            policy_ref="policy:1.0",
        )
        assert artifact.content_hash == rebuilt.content_hash

    def test_from_selection_decision_shadowed(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=False,
            reason=SelectionReason.SHADOW,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(decision)
        assert artifact.decision is AdmissionDecision.SHADOWED

    def test_from_selection_decision_rejected(self):
        decision = SelectionDecision(
            decision_id="SD_F001",
            factor_id="F001",
            approved=False,
            reason=SelectionReason.REJECTED_GATE_FAILURE,
            timestamp="2024-01-01T00:00:00Z",
            policy_version="1.0",
            evidence_refs=(),
            gate_results=(),
        )
        artifact = FactorAdmissionArtifact.from_selection_decision(decision)
        assert artifact.decision is AdmissionDecision.REJECTED
        assert artifact.reason == "REJECTED_GATE_FAILURE"

    def test_to_dict(self):
        artifact = _admission()
        data = artifact.to_dict()
        assert data["decision"] == "APPROVED"
        assert data["content_hash"] == artifact.content_hash
        assert data["cluster_id"] == 3

    def test_content_hash_is_derived_and_fails_closed_on_mismatch(self):
        # A caller must not self-report an arbitrary content hash: passing a
        # stored hash that does not equal the recomputed content hash FAILS
        # CLOSED, so content and hash can never diverge.
        with pytest.raises(ValueError, match="does not match"):
            _admission(content_hash="forged-hash")

    def test_evidence_refs_and_gate_results_are_construction_time_snapshot(self):
        # Tuples passed by the caller are snapshotted at construction time; a
        # caller mutating a list cannot change the artifact's frozen evidence.
        source_evidence = ["bundle-X"]
        source_gates = ["gate-X"]
        artifact = FactorAdmissionArtifact(
            factor_id="F001",
            decision=AdmissionDecision.APPROVED,
            quality=0.8,
            reason="APPROVED",
            evidence_refs=source_evidence,
            gate_results=source_gates,
        )
        source_evidence.append("tampered")
        source_gates.append("tampered")
        assert artifact.evidence_refs == ("bundle-X",)
        assert artifact.gate_results == ("gate-X",)
