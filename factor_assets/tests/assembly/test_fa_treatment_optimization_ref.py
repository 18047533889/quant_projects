"""P5-CLOSURE: FA wires the FO treatment-optimization reference (INT-2).

Closes the three reported gaps:
1. ``FactorSetSpec.treatment_optimization_ref`` (next to ``data_snapshot_ref``).
2. ``FactorLibraryMembership.treatment_optimization_ref`` (next to
   ``selected_treatment_ref``).
3. ``assembly/engine.py`` explicit ``content_hash`` reads (fail-closed — no more
   ``getattr(selection_artifact, "content_hash", None)`` silent ``None``).

All three are CONSUME-only: FA carries the FO optimization-run reference as a
bare str / dict PURE-DTO (matching FO ``LibrarySnapshotRef.to_dict()``) and
never imports the FO contract.
"""

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.assembly.engine import _read_content_hash
from factor_assets.contracts.factor_set import FactorMembership, FactorSetSpec
from factor_assets.contracts.library_governance import FactorLibraryMembership
from factor_assets.errors import CapabilityError
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact


# ---------------------------------------------------------------------------
# 1) FactorSetSpec.treatment_optimization_ref
# ---------------------------------------------------------------------------


class TestFactorSetSpecTreatmentOptimizationRef:
    def test_defaults_to_none_and_round_trips(self):
        spec = FactorSetSpec(
            set_id="s1",
            name="N",
            selection_policy="manual",
            data_snapshot_ref="snap:1",
        )
        assert spec.treatment_optimization_ref is None
        d = spec.to_dict()
        assert "treatment_optimization_ref" in d
        assert d["treatment_optimization_ref"] is None
        assert FactorSetSpec.from_dict(d).treatment_optimization_ref is None

    def test_old_format_dict_without_key_round_trips(self):
        """Backward compatible: a pre-closure spec dict has no key at all."""
        old = {
            "set_id": "s1",
            "name": "N",
            "selection_policy": "manual",
            "data_snapshot_ref": "snap:1",
        }
        spec = FactorSetSpec.from_dict(old)
        assert spec.treatment_optimization_ref is None
        assert spec.data_snapshot_ref == "snap:1"

    def test_str_shorthand_round_trips(self):
        spec = FactorSetSpec(
            set_id="s1",
            name="N",
            selection_policy="manual",
            data_snapshot_ref="snap:1",
            treatment_optimization_ref="opt-hash-1",
        )
        assert spec.treatment_optimization_ref == "opt-hash-1"
        assert (
            FactorSetSpec.from_dict(spec.to_dict()).treatment_optimization_ref
            == "opt-hash-1"
        )

    def test_dict_content_hash_normalized_and_full_mapping_preserved(self):
        # dict with content_hash -> canonical string transport form
        spec = FactorSetSpec(
            set_id="s1",
            name="N",
            selection_policy="manual",
            data_snapshot_ref="snap:1",
            treatment_optimization_ref={
                "library_version_ref": "lv1",
                "content_hash": "opt-hash-2",
            },
        )
        assert spec.treatment_optimization_ref == "opt-hash-2"
        # full mapping without content_hash key is preserved as-is
        full = FactorSetSpec(
            set_id="s1",
            name="N",
            selection_policy="manual",
            data_snapshot_ref="snap:1",
            treatment_optimization_ref={"library_version_ref": "lv1"},
        )
        assert full.treatment_optimization_ref == {"library_version_ref": "lv1"}

    def test_invalid_refs_fail_closed(self):
        with pytest.raises(TypeError, match="treatment_optimization_ref"):
            FactorSetSpec(
                set_id="s1",
                name="N",
                selection_policy="manual",
                data_snapshot_ref="snap:1",
                treatment_optimization_ref=123,
            )
        with pytest.raises(ValueError, match="non-empty"):
            FactorSetSpec(
                set_id="s1",
                name="N",
                selection_policy="manual",
                data_snapshot_ref="snap:1",
                treatment_optimization_ref="",
            )


# ---------------------------------------------------------------------------
# 2) FactorLibraryMembership.treatment_optimization_ref
# ---------------------------------------------------------------------------


class TestFactorLibraryMembershipTreatmentOptimizationRef:
    def test_defaults_to_none_and_to_dict_carries_key(self):
        m = FactorLibraryMembership(factor_definition_ref="d1")
        assert m.treatment_optimization_ref is None
        assert m.to_dict()["treatment_optimization_ref"] is None

    def test_str_and_dict_forms(self):
        m1 = FactorLibraryMembership(
            factor_definition_ref="d1", treatment_optimization_ref="opt-hash-1"
        )
        assert m1.treatment_optimization_ref == "opt-hash-1"
        m2 = FactorLibraryMembership(
            factor_definition_ref="d1",
            treatment_optimization_ref={"library_version_ref": "lv1", "content_hash": "ch2"},
        )
        assert m2.treatment_optimization_ref == "ch2"

    def test_invalid_refs_fail_closed(self):
        with pytest.raises(TypeError, match="treatment_optimization_ref"):
            FactorLibraryMembership(factor_definition_ref="d1", treatment_optimization_ref=123)
        with pytest.raises(ValueError, match="non-empty"):
            FactorLibraryMembership(factor_definition_ref="d1", treatment_optimization_ref="")

    def test_ref_flows_into_content_hash(self):
        """The membership's ref changes the library version content hash."""
        from factor_assets.contracts.library_governance import (
            FactorLibraryVersionArtifact,
        )

        def lib(opt):
            return FactorLibraryVersionArtifact(
                library_version_id="v1",
                logical_library_id="LIB_MOM",
                members=(
                    FactorLibraryMembership(
                        factor_definition_ref="F1",
                        selected_treatment_ref="st1",
                        treatment_optimization_ref=opt,
                    ),
                ),
                cluster_set_version_ref="csv1",
                selection_policy_ref="sp1",
                evidence_snapshot_ref="es1",
                snapshot_ref="snap1",
                universe_ref="u1",
                created_at="2024-01-01T00:00:00Z",
            )

        base = lib(None)
        with_ref = lib("opt-hash-9")
        assert with_ref.content_hash != base.content_hash


# ---------------------------------------------------------------------------
# 3) FactorMembership field validation (engine carries it on membership)
# ---------------------------------------------------------------------------


class TestFactorMembershipTreatmentOptimizationRef:
    def test_defaults_to_none(self):
        m = FactorMembership(factor_id="F1")
        assert m.treatment_optimization_ref is None

    def test_non_empty_string_accepted_empty_and_non_string_rejected(self):
        assert (
            FactorMembership(factor_id="F1", treatment_optimization_ref="ch1")
            .treatment_optimization_ref
            == "ch1"
        )
        with pytest.raises(ValueError, match="non-empty"):
            FactorMembership(factor_id="F1", treatment_optimization_ref="")
        # P0-FA-014 (Task #105): the membership canonicalizes the ref into its
        # transport form (str shorthand / dict PURE-DTO), so a non-str value
        # fails with the canonicalization error rather than a "str or None" one.
        with pytest.raises((TypeError, ValueError), match="treatment_optimization_ref"):
            FactorMembership(factor_id="F1", treatment_optimization_ref=123)

    def test_content_hash_sensitive_assembly_hash(self):
        """Changing the spec's ref changes the assembly_hash (and membership's)."""
        from factor_assets.contracts.asset import AssetMetadata, FactorAsset
        from factor_assets.contracts.lifecycle import LifecycleState
        from factor_assets.contracts.lineage import LineageRef
        from factor_assets.contracts.evidence_ref import EvidenceBundleRef

        def asset(fid):
            metadata = AssetMetadata(
                factor_id=fid,
                canonical_repr=f"identity({fid})",
                canonical_hash=f"hash-{fid}",
                frequency="daily",
                domains=("price",),
                timing="daily",
            )
            return FactorAsset(
                metadata=metadata,
                lineage=LineageRef(factor_id=fid, parents=()),
                lifecycle_state=LifecycleState.APPROVED,
                registered_at="2024-01-01T00:00:00Z",
                family="family-a",
                latest_evidence_ref=EvidenceBundleRef(
                    bundle_id=f"bundle-{fid}",
                    evaluation_run_id="run-1",
                    factor_ids=(fid,),
                    timestamp="2024-01-01T00:00:00Z",
                    qe_version="1.0",
                ),
            )

        def spec(opt=None):
            return FactorSetSpec(
                set_id="s1",
                name="N",
                selection_policy="manual",
                data_snapshot_ref="snap:1",
                universe_ref="universe:default",
                split_ref="split:default",
                treatment_optimization_ref=opt,
            )

        assembler = FactorSetAssembler()
        hashes = {
            "plain": assembler.assemble(spec(), [asset("F1")], created_at="2024-02-01T00:00:00Z").assembly_hash,
            "with_opt": assembler.assemble(spec("opt-hash-1"), [asset("F1")], created_at="2024-02-01T00:00:00Z").assembly_hash,
            "other_opt": assembler.assemble(spec("opt-hash-2"), [asset("F1")], created_at="2024-02-01T00:00:00Z").assembly_hash,
        }
        assert hashes["with_opt"] != hashes["plain"]
        assert hashes["other_opt"] != hashes["with_opt"]


# ---------------------------------------------------------------------------
# 4) engine._read_content_hash fail-closed
# ---------------------------------------------------------------------------


class TestReadContentHashFailClosed:
    def test_dict_with_content_hash_reads_explicitly(self):
        assert _read_content_hash({"content_hash": "abc"}, "F1") == "abc"

    def test_typed_artifact_reads_content_hash(self):
        artifact = _make_treatment("F1")
        assert _read_content_hash(artifact, "F1") == artifact.content_hash

    def test_dict_without_content_hash_fails_closed(self):
        with pytest.raises(CapabilityError, match="content_hash"):
            _read_content_hash({"foo": "bar"}, "F1")

    def test_bare_str_fails_closed(self):
        with pytest.raises(CapabilityError, match="bare str"):
            _read_content_hash("not-an-artifact", "F1")

    def test_opaque_object_fails_closed(self):
        with pytest.raises(CapabilityError, match="content_hash"):
            _read_content_hash(object(), "F1")

    def test_empty_content_hash_fails_closed(self):
        with pytest.raises(CapabilityError, match="content_hash"):
            _read_content_hash({"content_hash": ""}, "F1")

    def test_assembler_populates_membership_ref_from_artifact(self):
        """Integration: assemble() lifts the ref via the new explicit reader."""
        artifact = _make_treatment("F1")
        from factor_assets.contracts.admission import (
            AdmissionDecision,
            FactorAdmissionArtifact,
        )
        from factor_assets.contracts.asset import AssetMetadata, FactorAsset
        from factor_assets.contracts.lifecycle import LifecycleState
        from factor_assets.contracts.lineage import LineageRef
        from factor_assets.contracts.evidence_ref import EvidenceBundleRef

        metadata = AssetMetadata(
            factor_id="F1",
            canonical_repr="identity(F1)",
            canonical_hash="hash-F1",
            frequency="daily",
            domains=("price",),
            timing="daily",
        )
        asset = FactorAsset(
            metadata=metadata,
            lineage=LineageRef(factor_id="F1", parents=()),
            lifecycle_state=LifecycleState.APPROVED,
            registered_at="2024-01-01T00:00:00Z",
            family="family-a",
            latest_evidence_ref=EvidenceBundleRef(
                bundle_id="bundle-F1",
                evaluation_run_id="run-1",
                factor_ids=("F1",),
                timestamp="2024-01-01T00:00:00Z",
                qe_version="1.0",
            ),
        )
        admission = FactorAdmissionArtifact(
            factor_id="F1",
            decision=AdmissionDecision.APPROVED,
            quality=0.8,
            factor_version="v1",
            health_state_ref="lifecycle:APPROVED",
            cluster_id=3,
            orientation=1,
            reason="APPROVED",
            evidence_refs=("bundle-F1",),
            gate_results=("gate-1",),
            universe_ref="universe:default", snapshot_ref="snap:1",
            split_ref="split:default", recipe_ref="recipe:default",
            data_as_of="2024-01-01T00:00:00Z", created_at="2024-01-01T00:00:00Z",
        )
        spec = FactorSetSpec(
            set_id="s1",
            name="N",
            selection_policy="manual",
            data_snapshot_ref="snap:1",
            universe_ref="universe:default",
            split_ref="split:default",
            recipe_ref="recipe:default",
            selection_as_of="2026-01-01T00:00:00Z",
        )
        result = FactorSetAssembler().assemble(
            spec,
            [asset],
            admission_artifacts={"F1": admission},
            treatment_selection_artifacts={"F1": artifact},
            production=True,
            created_at="2024-02-01T00:00:00Z",
        )
        (membership,) = result.memberships
        assert membership.treatment_selection_ref == artifact.content_hash


def _make_treatment(factor_id):
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
