from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.assembly_evidence import AssemblyPolicy
from factor_assets.contracts.factor_set import FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.evidence_ref import EvidenceBundleRef
from factor_assets.selection import SelectionDecision, SelectionReason
from types import MappingProxyType

import pytest
from types import SimpleNamespace


def test_wrong_universe_snapshot_is_rejected_at_public_assembly_entry():
    spec = make_spec("set-wrong-universe", "Wrong universe", "manual", universe_ref="universe:A")
    wrong = SimpleNamespace(universe_ref="universe:B", snapshot_id="snapshot:B")
    with pytest.raises(ValueError, match="does not match spec.universe_ref"):
        FactorSetAssembler().assemble(spec, [make_asset("F1")], universe_snapshot=wrong)


def test_assembly_binds_exact_universe_snapshot_to_lineage_and_hash():
    spec = make_spec("set-dated", "Dated", "manual", universe_ref="universe:A")
    first = SimpleNamespace(universe_ref="universe:A", snapshot_id="snapshot:A:2020")
    second = SimpleNamespace(universe_ref="universe:A", snapshot_id="snapshot:A:2021")
    a = FactorSetAssembler().assemble(spec, [make_asset("F1")], universe_snapshot=first)
    b = FactorSetAssembler().assemble(spec, [make_asset("F1")], universe_snapshot=second)
    assert a.universe_snapshot_ref == "snapshot:A:2020"
    assert b.universe_snapshot_ref == "snapshot:A:2021"
    assert a.assembly_hash != b.assembly_hash


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
    """Build a FactorSetSpec with the now-required provenance refs."""
    kwargs.setdefault("data_snapshot_ref", "snapshot:default")
    kwargs.setdefault("universe_ref", "universe:default")
    kwargs.setdefault("split_ref", "split:default")
    kwargs.setdefault("recipe_ref", "recipe:default")
    kwargs.setdefault("selection_as_of", "2026-01-01T00:00:00Z")
    return FactorSetSpec(set_id, name, policy, **kwargs)


def test_empty_candidates_fail_closed():
    spec = make_spec("set-1", "Empty", "manual")
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
    spec = make_spec("set-1", "Automated", "family_robust")
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
    spec = make_spec("set-1", "Automated", "family_robust")
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
    spec = make_spec("set-1", "Automated", "family_robust")
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")])
    except ValueError as exc:
        assert "selection_decisions are required" in str(exc)
    else:
        raise AssertionError("non-manual selection must require admission decisions")


def test_non_manual_selection_admits_only_latest_approved_decisions():
    spec = make_spec("set-1", "Automated", "family_robust")
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
    spec = make_spec("set-1", "Automated", "family_robust")
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
    spec = make_spec("set-1", "Automated", "family_robust")
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
    spec = make_spec("set-1", "Manual", "manual")
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    assert result.factor_ids == ("F1",)


def test_max_factors_and_deterministic_ordering():
    spec = make_spec("set-1", "Capped", "manual", max_factors=2)
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F3"), make_asset("F1"), make_asset("F2")],
        created_at="2024-02-01T00:00:00Z",
    )
    assert result.factor_ids == ("F1", "F2")
    assert result.created_at == "2024-02-01T00:00:00Z"
    assert result.families == ("family-a",)


def test_duplicate_factor_id_candidates_are_deduplicated_before_cap():
    spec = make_spec("set-1", "Capped", "manual", max_factors=2)
    duplicate = make_asset("F1")
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), duplicate, duplicate, make_asset("F3")]
    )
    assert result.factor_ids == ("F1", "F2")
    assert result.size == 2


def test_conflicting_duplicate_factor_id_candidates_fail_closed():
    spec = make_spec("set-1", "Conflicting", "manual")
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
    spec = make_spec(
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
    spec = make_spec("set-1", "Families", "manual", max_factors=3, family_constraints="max_per_family=1")
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), make_asset("F3"), make_asset("F1")],
    )
    assert result.factor_ids == ("F1", "F3")


def test_unknown_family_constraint_syntax_fails_closed():
    spec = make_spec("set-1", "Families", "manual", family_constraints="one_per_family")
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
    spec = make_spec("set-1", "Ranked", "pareto_front", max_factors=1)
    old = make_decision("F1", timestamp="2024-01-01T00:00:00Z")
    recent = make_decision("F2", timestamp="2024-03-01T00:00:00Z")
    object.__setattr__(old, "metadata", MappingProxyType({"objectives": {"quality": .1}}))
    object.__setattr__(recent, "metadata", MappingProxyType({"objectives": {"quality": .2}}))
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[
            old, recent,
        ],
        assembly_policy=AssemblyPolicy("pareto-test", "1", required_objectives=("quality",)),
    )
    assert result.factor_ids == ("F2",)


def test_manual_ranking_remains_lexicographic():
    spec = make_spec("set-1", "Manual", "manual", max_factors=1)
    result = FactorSetAssembler().assemble(
        spec, [make_asset("F2"), make_asset("F1")]
    )
    assert result.factor_ids == ("F1",)


def test_unknown_selection_policy_fails_closed():
    spec = make_spec("set-1", "Bogus", "aggressive_growth")
    try:
        FactorSetAssembler().assemble(spec, [make_asset("F1")])
    except ValueError as exc:
        assert "unknown selection_policy" in str(exc)
    else:
        raise AssertionError("unknown selection policies must fail closed")


def test_memberships_carry_decision_and_evidence_refs():
    spec = make_spec("set-1", "Provenance", "family_robust")
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
    spec = make_spec("set-1", "Manual", "manual")
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    (membership,) = result.memberships
    assert membership.factor_id == "F1"
    assert membership.selection_decision_ref is None
    assert membership.reason is None
    assert membership.evidence_ref == "bundle-F1"


def test_assembly_hash_deterministic_and_content_sensitive():
    spec = make_spec("set-1", "Hashed", "family_robust")
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
    other_spec = make_spec("set-2", "Hashed", "family_robust", max_factors=1)
    other = FactorSetAssembler().assemble(
        other_spec, [make_asset("F1"), make_asset("F2")],
        selection_decisions=decisions, created_at="2024-02-01T00:00:00Z",
    )
    assert other.policy_hash != first.policy_hash


def test_assembly_hash_is_canonical_over_all_semantic_fields():
    """The assembly hash is a canonical serialization over EVERY semantic field
    of the membership + snapshot/universe/split/policy identity — a change to
    any one of them must change the hash, so a new field never silently drifts
    out of the hash."""
    spec = make_spec("set-1", "Canonical", "manual")
    base = FactorSetAssembler().assemble(
        spec, [make_asset("F1")], created_at="2024-02-01T00:00:00Z"
    )
    h = base.assembly_hash

    # Changing the data snapshot / universe / split identity changes the hash.
    assert FactorSetAssembler().assemble(
        make_spec("set-1", "Canonical", "manual", data_snapshot_ref="snapshot:OTHER"),
        [make_asset("F1")], created_at="2024-02-01T00:00:00Z",
    ).assembly_hash != h
    assert FactorSetAssembler().assemble(
        make_spec("set-1", "Canonical", "manual", universe_ref="universe:OTHER"),
        [make_asset("F1")], created_at="2024-02-01T00:00:00Z",
    ).assembly_hash != h
    assert FactorSetAssembler().assemble(
        make_spec("set-1", "Canonical", "manual", split_ref="split:OTHER"),
        [make_asset("F1")], created_at="2024-02-01T00:00:00Z",
    ).assembly_hash != h

    # Changing the admission provenance (cluster_id / orientation / factor_version)
    # — which flow into the membership — must change the assembly hash too.
    from factor_assets.contracts.admission import AdmissionDecision, FactorAdmissionArtifact

    def admission(cluster_id, orientation, factor_version):
        return FactorAdmissionArtifact(
            factor_id="F1",
            decision=AdmissionDecision.APPROVED,
            quality=0.8,
            factor_version=factor_version,
            health_state_ref="lifecycle:APPROVED",
            cluster_id=cluster_id,
            orientation=orientation,
            reason="APPROVED",
            evidence_refs=("bundle-F1",),
            gate_results=("gate-1",),
            universe_ref="universe:default", snapshot_ref="snapshot:default",
            split_ref="split:default", recipe_ref="recipe:default",
            data_as_of="2024-01-01T00:00:00Z", created_at="2024-01-01T00:00:00Z",
        )

    # Auto-treatment artifacts so production mode has a consume-only
    # treatment_selection_ref (production now requires one).
    from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact

    def treatment(factor_version="v1"):
        return TreatmentSelectionArtifact(
            factor_id="F1",
            factor_version=factor_version,
            raw_baseline_evidence_ref="ref:raw",
            factor_profile_ref="ref:profile",
            eligibility_policy_ref="ref:elig",
            search_space_ref="ref:ss",
            all_trial_refs=("ref:t_a",),
            pareto_candidate_refs=("ref:p_1",),
            winner_recipe={"preprocess": "zscore"},
            winner_policy_identity="policy:auto-treat/v3",
            absolute_metric_refs={"sharpe": "ref:s"},
            delta_metric_refs={"delta_sharpe": "ref:d"},
            dimension_scores={"return": 0.8, "robustness": 0.65},
            hard_gate_results={"min_obs": "PASS"},
            soft_floor_results={"min_sharpe": 0.2},
            robustness_evidence="ref:rob",
            complexity_score=0.42,
            snapshot_ref="ref:snapshot",
            universe_ref="ref:universe",
            split_ref="ref:split",
            created_at="2026-08-25T00:00:00+00:00",
        )

    prod = FactorSetAssembler().assemble(
        spec, [make_asset("F1")],
        admission_artifacts={"F1": admission(3, 1, "v1")},
        treatment_selection_artifacts={"F1": treatment()},
        production=True, created_at="2024-02-01T00:00:00Z",
    )
    assert prod.assembly_hash != h  # membership provenance now included
    assert FactorSetAssembler().assemble(
        spec, [make_asset("F1")],
        admission_artifacts={"F1": admission(4, 1, "v1")},
        treatment_selection_artifacts={"F1": treatment()},
        production=True, created_at="2024-02-01T00:00:00Z",
    ).assembly_hash != prod.assembly_hash
    assert FactorSetAssembler().assemble(
        spec, [make_asset("F1")],
        admission_artifacts={"F1": admission(3, 1, "v2")},
        treatment_selection_artifacts={"F1": treatment("v2")},
        production=True, created_at="2024-02-01T00:00:00Z",
    ).assembly_hash != prod.assembly_hash


def test_factor_set_split_and_snapshot_refs_flow_from_spec():
    spec = make_spec(
        "set-1", "Refs", "manual",
        universe_ref="universe:ashare-v3", split_ref="split:oos-2024h1",
    )
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    # snapshot_ref is the explicit data_snapshot_ref (defaulted by make_spec),
    # NOT the universe — the universe must not impersonate the snapshot.
    assert result.snapshot_ref == "snapshot:default"
    assert result.split_ref == "split:oos-2024h1"
    assert result.universe_ref == "universe:ashare-v3"


def test_explicit_data_snapshot_ref_is_authoritative():
    # data_snapshot_ref is explicit provenance: it must not be aliased to
    # universe_ref, and a spec without it fails closed (no universe fallback).
    spec = make_spec(
        "set-1", "Snapshot", "manual",
        universe_ref="universe:ashare-v3",
        data_snapshot_ref="snapshot:2024-08-01T00:00:00Z",
    )
    result = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    assert result.snapshot_ref == "snapshot:2024-08-01T00:00:00Z"
    assert result.universe_ref == "universe:ashare-v3"


def test_spec_without_data_snapshot_ref_fails_closed():
    # The universe must NOT impersonate the snapshot: a spec with no
    # data_snapshot_ref is rejected at construction.
    try:
        FactorSetSpec("set-2", "Legacy", "manual", universe_ref="universe:ashare-v3")
    except ValueError as exc:
        assert "data_snapshot_ref is required" in str(exc)
    else:
        raise AssertionError("spec without data_snapshot_ref must fail closed")


def test_diverse_requires_similarity_provider():
    """diverse (MMR) fails closed without a similarity provider."""
    spec = make_spec("set-1", "Diverse", "diverse")
    try:
        FactorSetAssembler().assemble(
            spec, [make_asset("F1")], selection_decisions=[make_decision("F1")]
        )
    except ValueError as exc:
        assert "similarity_provider" in str(exc)
    else:
        raise AssertionError("diverse without similarity_provider must fail closed")


def test_diverse_mmr_selects_diverse_factors():
    """diverse uses real MMR: quality minus max similarity to selected."""
    spec = make_spec("set-1", "Diverse", "diverse", max_factors=2)

    def decided(factor_id, quality, timestamp):
        d = make_decision(factor_id, timestamp=timestamp)
        object.__setattr__(d, "metadata", MappingProxyType({"quality": quality}))
        return d

    # F1 and F2 are near-duplicates (high similarity); F3 is distinct.
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
    # MMR picks F1 (highest quality) then F3 (diverse), not F2 (redundant).
    assert result.factor_ids == ("F1", "F3")


def test_family_robust_uses_composite_score():
    """family_robust ranks by composite robustness score, not just recency."""
    spec = make_spec("set-1", "Robust", "family_robust", max_factors=1)

    def decided(factor_id, score, timestamp):
        d = make_decision(factor_id, timestamp=timestamp)
        object.__setattr__(d, "metadata", MappingProxyType({"family_robust": score}))
        return d

    # F1 has a higher composite robustness score but is LESS recent.
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[
            decided("F1", 0.9, "2024-01-01T00:00:00Z"),
            decided("F2", 0.5, "2024-06-01T00:00:00Z"),
        ],
    )
    assert result.factor_ids == ("F1",)


def test_artifact_to_legacy_view():
    """FactorSetArtifact.to_legacy_view() produces the deprecated FactorSet."""
    from factor_assets.contracts.factor_set import FactorSet

    spec = make_spec("set-1", "LegacyView", "manual")
    artifact = FactorSetAssembler().assemble(spec, [make_asset("F1")])
    legacy = artifact.to_legacy_view()
    assert isinstance(legacy, FactorSet)
    assert legacy.factor_ids == ("F1",)
    assert legacy.snapshot_ref == "snapshot:default"
    assert legacy.memberships == artifact.members


def test_pareto_front_ranking_uses_dominance_not_recency():
    # F2 dominates F1 on both objectives (rank_ic and sharpe) despite F1
    # having the MORE RECENT admission decision — dominance must win over
    # the timestamp-priority ordering.
    def decided(factor_id, objectives, timestamp):
        d = make_decision(factor_id, timestamp=timestamp)
        object.__setattr__(d, "metadata", MappingProxyType({"objectives": dict(objectives)}))
        return d

    spec = make_spec("set-1", "Pareto", "pareto_front", max_factors=1)
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[
            decided("F1", {"rank_ic": 0.01, "sharpe": 0.5}, "2024-06-01T00:00:00Z"),
            decided("F2", {"rank_ic": 0.05, "sharpe": 1.2}, "2024-01-01T00:00:00Z"),
        ],
        assembly_policy=AssemblyPolicy("pareto-test", "1", required_objectives=("rank_ic", "sharpe")),
    )
    assert result.factor_ids == ("F2",)


def test_pareto_front_ranking_ignores_partial_objectives():
    # One decision lacking the objective set makes the objective space
    # undefined — ranking falls back to recency ordering rather than
    # scoring a factor on -inf defaults.
    def decided(factor_id, objectives, timestamp):
        d = make_decision(factor_id, timestamp=timestamp)
        object.__setattr__(d, "metadata", MappingProxyType({"objectives": dict(objectives)}))
        return d

    spec = make_spec("set-1", "Partial", "pareto_front", max_factors=1)
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2")],
        selection_decisions=[
            decided("F1", {}, "2024-06-01T00:00:00Z"),
            decided("F2", {"rank_ic": 0.05}, "2024-01-01T00:00:00Z"),
        ],
        assembly_policy=AssemblyPolicy("pareto-test", "1", required_objectives=("rank_ic",)),
    )
    assert result.factor_ids == ("F2",)


def test_pareto_front_without_frozen_objectives_fails_closed():
    # Metadata-only decisions (no objectives anywhere): all points are
    # mutually non-dominated; ordering preserves the previous behaviour.
    spec = make_spec("set-1", "NoObj", "pareto_front", max_factors=1)
    with pytest.raises(ValueError, match="policy-frozen required_objectives"):
        FactorSetAssembler().assemble(
            spec, [make_asset("F1"), make_asset("F2")],
            selection_decisions=[make_decision("F1"), make_decision("F2")],
        )


def test_factor_set_artifact_round_trip_from_assembly():
    from factor_assets.contracts.factor_set import FactorSetArtifact

    spec = make_spec(
        "set-1", "Artifact", "family_robust",
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
            snapshot_ref="snapshot:default", universe_ref="universe:default",
            split_ref="split:default",
        )
    except ValueError as exc:
        assert "duplicate factor_id" in str(exc)
    else:
        raise AssertionError("duplicate members must fail")


def test_validation_status_orthogonal_to_lifecycle():
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


# ---------------------------------------------------------------------------
# AssemblyPolicy runtime enforcement (P1-FA-013 / audit item 3)
# ---------------------------------------------------------------------------


def test_assembly_policy_max_per_microcluster_is_enforced():
    """A supplied AssemblyPolicy is a budget contract, not just ranking
    weights: per-microcluster budget (max_per_microcluster) must actually cap
    how many members share one grouping.  A policy that is accepted but never
    applied is the audit bug."""
    # All three assets share the same family (microcluster grouping).
    assets = [make_asset("F1"), make_asset("F2"), make_asset("F3")]
    # Override the family so F3 shares it too — make_asset gives F3 family-b.
    from dataclasses import replace as _replace

    assets = [_replace(a, family="shared") for a in assets]
    spec = make_spec("set-policy-1", "Policy", "manual", max_factors=10)
    result = FactorSetAssembler().assemble(
        spec,
        assets,
        selection_decisions=[make_decision("F1"), make_decision("F2"), make_decision("F3")],
        assembly_policy=AssemblyPolicy(
            "pol-micro", "1.0", max_per_microcluster=1, max_per_macrocluster=5
        ),
    )
    # Only ONE member may be admitted from the shared microcluster.
    assert result.factor_ids == ("F1",)


def test_assembly_policy_max_per_macrocluster_is_enforced():
    """Per-macrocluster budget (max_per_macrocluster) caps how many members
    share the macro grouping."""
    from dataclasses import replace as _replace

    assets = [make_asset("F1"), make_asset("F2"), make_asset("F3")]
    # F1/F2 share macro "mom"; F3 is its own macro bucket.
    assets = [_replace(a, family="mom" if a.factor_id != "F3" else "vol") for a in assets]
    spec = make_spec("set-policy-2", "Policy", "manual", max_factors=10)
    result = FactorSetAssembler().assemble(
        spec,
        assets,
        selection_decisions=[make_decision("F1"), make_decision("F2"), make_decision("F3")],
        assembly_policy=AssemblyPolicy(
            "pol-macro", "1.0", max_per_microcluster=1, max_per_macrocluster=1
        ),
    )
    # At most one member from the "mom" macro bucket; F3 stays (its own bucket).
    # No actual cluster-membership evidence: every unknown belongs to the
    # shared UNKNOWN buckets and cannot evade the macro budget via factor_id.
    assert result.factor_ids == ("F1",)


def test_assembly_policy_capacity_budget_is_enforced():
    """capacity_budget caps the total assembled set before max_factors."""
    spec = make_spec("set-policy-3", "Policy", "manual", max_factors=10)
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2"), make_asset("F3")],
        selection_decisions=[make_decision("F1"), make_decision("F2"), make_decision("F3")],
        assembly_policy=AssemblyPolicy(
            "pol-cap", "1.0", max_per_microcluster=5, max_per_macrocluster=5,
            capacity_budget=2,
        ),
    )
    assert result.factor_ids == ("F1", "F2")


def test_assembly_policy_without_policy_preserves_legacy_behaviour():
    """No policy supplied -> no budget is silently applied (a default policy
    must not cap the assembly behind the caller's back)."""
    spec = make_spec("set-nopolicy", "NoPolicy", "manual", max_factors=10)
    result = FactorSetAssembler().assemble(
        spec,
        [make_asset("F1"), make_asset("F2"), make_asset("F3")],
        selection_decisions=[make_decision("F1"), make_decision("F2"), make_decision("F3")],
    )
    assert result.factor_ids == ("F1", "F2", "F3")


def test_assembly_policy_rejects_inverted_cluster_budgets():
    """A microcluster is a subset of a macrocluster — a policy with
    max_per_microcluster > max_per_macrocluster fails closed at construction."""
    with pytest.raises(ValueError, match="max_per_microcluster must be <= "):
        AssemblyPolicy("pol-bad", "1.0", max_per_microcluster=3, max_per_macrocluster=2)
