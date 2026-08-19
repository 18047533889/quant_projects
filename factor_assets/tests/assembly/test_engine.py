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


    assert FactorSetAssembler.__name__ == "FactorSetAssembler"
