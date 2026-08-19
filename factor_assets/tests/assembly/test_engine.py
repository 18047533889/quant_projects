from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.factor_set import FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
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


def test_empty_candidates_fail_closed():
    spec = FactorSetSpec("set-1", "Empty", "manual")
    try:
        FactorSetAssembler().assemble(spec, [])
    except ValueError as exc:
        assert "no factor assets" in str(exc)
    else:
        raise AssertionError("empty assembly must fail")


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


def test_non_manual_assembly_rejects_unproven_approval():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    forged = SelectionDecision(
        decision_id="forged",
        factor_id="F1",
        approved=True,
        reason=SelectionReason.APPROVED,
        timestamp="2024-01-01T00:00:00Z",
        policy_version="1.0",
        evidence_refs=(),
        gate_results=(),
    )
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")], selection_decisions=[forged])
    except ValueError as exc:
        assert "approved selection decisions require" in str(exc)
    else:
        raise AssertionError("unproven approvals must fail closed")


def test_non_manual_assembly_rejects_evidence_free_approved_asset():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    asset = make_asset("F1")
    asset = FactorAsset(
        metadata=asset.metadata,
        lineage=asset.lineage,
        lifecycle_state=asset.lifecycle_state,
        registered_at=asset.registered_at,
        family=asset.family,
    )
    try:
        FactorSetAssembler().assemble(spec, [asset], selection_decisions=[make_decision("F1")])
    except ValueError as exc:
        assert "lacks evidence bundle" in str(exc)
    else:
        raise AssertionError("evidence-free approved assets must fail closed")


def test_non_manual_selection_requires_decisions():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")])
    except ValueError as exc:
        assert "selection_decisions are required" in str(exc)
    else:
        raise AssertionError("non-manual selection must require admission decisions")


def test_non_manual_selection_admits_only_latest_approved_decisions():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("approved"), make_asset("rejected"), make_asset("missing")],
        selection_decisions=[
            make_decision("approved"),
            make_decision("rejected", approved=True),
            make_decision(
                "rejected", approved=False, timestamp="2024-02-01T00:00:00Z"
            ),
        ],
    )
    assert result.factor_ids == ("approved",)


def test_non_manual_selection_is_order_invariant_for_decision_history():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    early_rejection = make_decision(
        "F1", approved=False, timestamp="2024-01-01T00:00:00Z"
    )
    later_approval = make_decision(
        "F1", approved=True, timestamp="2024-02-01T00:00:00Z"
    )
    duplicate = make_decision(
        "F2", approved=True, timestamp="2024-01-15T00:00:00Z"
    )
    result_in_order = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[early_rejection, later_approval, duplicate, duplicate],
    )
    result_reversed = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[duplicate, duplicate, later_approval, early_rejection],
    )

    assert result_in_order.factor_ids == ("F1", "F2")
    assert result_reversed.factor_ids == result_in_order.factor_ids


def test_equal_timestamp_conflicting_selection_decisions_fail_closed():
    spec = FactorSetSpec("set-1", "Automated", "family_robust")
    first = make_decision("F1", approved=True)
    conflicting = make_decision("F1", approved=False)

    try:
        FactorSetAssembler().assemble(
            spec,
            [make_asset("F1")],
            selection_decisions=[first, conflicting],
        )
    except ValueError as exc:
        assert "conflicting selection decisions" in str(exc)
    else:
        raise AssertionError("conflicting same-timestamp decisions must fail")


def test_manual_selection_remains_backward_compatible_without_decisions():
    spec = FactorSetSpec("set-1", "Manual", "manual")
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    assert result.factor_ids == ("F1",)


def test_max_factors_and_deterministic_ordering():
    spec = FactorSetSpec("set-1", "Capped", "manual", max_factors=2)
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F3"), make_asset("F1"), make_asset("F2")],
        created_at="2024-02-01T00:00:00Z",
    )
    assert result.factor_ids == ("F1", "F2")
    assert result.created_at == "2024-02-01T00:00:00Z"
    assert result.families == ("family-a",)


def test_duplicate_factor_id_candidates_are_deduplicated_before_cap():
    spec = FactorSetSpec("set-1", "Capped", "manual", max_factors=2)
    duplicate = make_asset("F1")
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), duplicate, duplicate, make_asset("F3")]
    )
    assert result.factor_ids == ("F1", "F2")
    assert result.size == 2


def test_conflicting_duplicate_factor_id_candidates_fail_closed():
    spec = FactorSetSpec("set-1", "Conflicting", "manual")
    first = make_asset("F1")
    conflicting = FactorAsset(
        metadata=AssetMetadata(
            factor_id="F1",
            canonical_repr="different(F1)",
            canonical_hash="different-hash-F1",
            frequency="daily",
            domains=("price",),
            timing="daily",
        ),
        lineage=LineageRef(factor_id="F1", parents=()),
        lifecycle_state=LifecycleState.APPROVED,
        registered_at="2024-01-01T00:00:00Z",
        family="family-a",
    )

    try:
        FactorSetAssembler().assemble(spec, [first, conflicting])
    except ValueError as exc:
        assert "conflicting candidate assets for factor_id: F1" in str(exc)
    else:
        raise AssertionError("conflicting duplicate factor assets must fail")


def test_basic_spec_filters():
    spec = FactorSetSpec(
        "set-1", "Filtered", "manual", frequency="daily",
        required_domains=("price",), excluded_domains=("fundamental",),
        min_lifecycle_state="APPROVED",
    )
    candidates = [
        make_asset("good"),
        make_asset("wrong-frequency", frequency="weekly"),
        make_asset("wrong-domain", domains=("fundamental",)),
        make_asset("too-early", state=LifecycleState.EVALUATED),
    ]
    assert FactorSetAssembler().assemble(spec, candidates).factor_ids == ("good",)


def test_family_constraint_limits_each_family():
    spec = FactorSetSpec("set-1", "Families", "manual", max_factors=3, family_constraints="max_per_family=1")
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), make_asset("F3"), make_asset("F1")],
    )
    assert result.factor_ids == ("F1", "F3")


def test_unknown_family_constraint_syntax_fails_closed():
    spec = FactorSetSpec("set-1", "Families", "manual", family_constraints="one_per_family")
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")])
    except ValueError as exc:
        assert "max_per_family=N" in str(exc)
    else:
        raise AssertionError("ambiguous family constraints must fail")



def test_ranking_is_evidence_based_not_lexicographic():
    # F1 admitted most recently (2024-03) so it ranks ahead of F2 (2024-01)
    # despite the lexicographic order F1 < F2 being coincidentally aligned;
    # use F2 recent / F1 old so the two orderings genuinely disagree.
    spec = FactorSetSpec("set-1", "Ranked", "pareto_front", max_factors=1)
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[
            make_decision("F1", timestamp="2024-01-01T00:00:00Z"),
            make_decision("F2", timestamp="2024-03-01T00:00:00Z"),
        ],
    )
    assert result.factor_ids == ("F2",)


def test_manual_ranking_remains_lexicographic():
    spec = FactorSetSpec("set-1", "Manual", "manual", max_factors=1)
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), make_asset("F1")]
    )
    assert result.factor_ids == ("F1",)


def test_unknown_selection_policy_fails_closed():
    spec = FactorSetSpec("set-1", "Bogus", "aggressive_growth")
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")])
    except ValueError as exc:
        assert "unknown selection_policy" in str(exc)
    else:
        raise AssertionError("unknown selection policies must fail closed")


def test_memberships_carry_decision_and_evidence_refs():
    spec = FactorSetSpec("set-1", "Provenance", "pareto_front")
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1")],
        selection_decisions=[make_decision("F1")],
    )
    (membership,) = result.memberships
    assert membership.factor_id == "F1"
    assert membership.role == "member"
    assert membership.family_id == "family-a"
    assert membership.selection_decision_ref == "decision-F1-2024-01-01T00:00:00Z"
    assert membership.evidence_ref == "bundle-F1"
    assert membership.reason == "APPROVED"


def test_manual_memberships_have_null_decision_refs():
    spec = FactorSetSpec("set-1", "Manual", "manual")
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    (membership,) = result.memberships
    assert membership.factor_id == "F1"
    assert membership.selection_decision_ref is None
    assert membership.reason is None
    assert membership.evidence_ref == "bundle-F1"


def test_assembly_hash_deterministic_and_content_sensitive():
    spec = FactorSetSpec("set-1", "Hashed", "pareto_front")
    decisions = [make_decision("F1"), make_decision("F2")]
    first = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), make_asset("F1")],
        selection_decisions=decisions, created_at="2024-02-01T00:00:00Z",
    )
    second = FactorSetAssembler().assemble(
        spec, [make_asset("F1"), make_asset("F2")],
        selection_decisions=list(reversed(decisions)),
        created_at="2025-01-01T00:00:00Z",
    )
    # Same members + spec + decision refs: hash is order- and time-invariant.
    assert first.assembly_hash == second.assembly_hash
    assert first.policy_hash == second.policy_hash

    # Changing the membership content changes the assembly hash.
    changed_decisions = [make_decision("F1"), make_decision("F2", timestamp="2024-06-01T00:00:00Z")]
    changed = FactorSetAssembler().assemble(
        spec, [make_asset("F1"), make_asset("F2")],
        selection_decisions=changed_decisions, created_at="2024-02-01T00:00:00Z",
    )
    assert changed.assembly_hash != first.assembly_hash

    # Changing the policy semantics changes the policy hash only.
    other_spec = FactorSetSpec("set-2", "Hashed", "family_robust")
    other = FactorSetAssembler().assemble(
        other_spec, [make_asset("F1"), make_asset("F2")],
        selection_decisions=decisions, created_at="2024-02-01T00:00:00Z",
    )
    assert other.policy_hash != first.policy_hash


def test_factor_set_split_and_snapshot_refs_flow_from_spec():
    spec = FactorSetSpec(
        "set-1", "Refs", "manual",
        universe_ref="universe:ashare-v3", split_ref="split:oos-2024h1",
    )
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    assert result.snapshot_ref == "universe:ashare-v3"
    assert result.split_ref == "split:oos-2024h1"
    assert result.universe_ref == "universe:ashare-v3"


def test_factor_set_artifact_round_trip_from_assembly():
    from factor_assets.contracts.factor_set import FactorSetArtifact

    spec = FactorSetSpec(
        "set-1", "Artifact", "pareto_front",
        universe_ref="universe:ashare-v3", split_ref="split:oos-2024h1",
    )
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F1"), make_asset("F2")],
        selection_decisions=[make_decision("F1"), make_decision("F2")],
        created_at="2024-02-01T00:00:00Z",
    )
    artifact = FactorSetArtifact(
        set_id=result.set_id,
        name=result.name,
        members=result.memberships,
        created_at=result.created_at,
        policy_hash=result.policy_hash,
        assembly_hash=result.assembly_hash,
        universe_ref=result.universe_ref,
        snapshot_ref=result.snapshot_ref,
        split_ref=result.split_ref,
        spec=result.spec,
    )
    assert artifact.factor_ids == ("F1", "F2")
    assert artifact.contains("F1") and not artifact.contains("F9")
    assert artifact.split_ref == "split:oos-2024h1"


def test_factor_set_artifact_rejects_duplicate_members():
    from factor_assets.contracts.factor_set import FactorMembership, FactorSetArtifact

    member = FactorMembership(factor_id="F1")
    try:
        FactorSetArtifact(
            set_id="set-1", name="Bad", members=(member, member),
            created_at="2024-02-01T00:00:00Z",
            policy_hash="ph", assembly_hash="ah",
        )
    except ValueError as exc:
        assert "duplicate factor_id" in str(exc)
    else:
        raise AssertionError("duplicate members must fail")


def test_validation_status_orthogonal_to_lifecycle():
    from factor_assets.contracts.asset import AssetMetadata, FactorAsset
    from factor_assets.contracts.lifecycle import (
        HealthState, LifecycleState, ValidationStatus,
    )

    asset = FactorAsset(
        metadata=make_asset("F1").metadata,
        lineage=make_asset("F1").lineage,
        lifecycle_state=LifecycleState.PRODUCTION_READY,
        registered_at="2024-01-01T00:00:00Z",
        validation_status=ValidationStatus.STALE,
        health_state=HealthState.DEPRECATED,
    )
    # Stage says certified, validation says evidence is stale, health says
    # discouraged — all three dimensions are independently representable.
    assert asset.is_production_ready
    assert not asset.is_validation_current
    assert not asset.is_usable

    # Defaults preserve legacy construction (UNVALIDATED / ACTIVE).
    legacy = make_asset("F2")
    assert legacy.validation_status == ValidationStatus.UNVALIDATED
    assert legacy.health_state == HealthState.ACTIVE
