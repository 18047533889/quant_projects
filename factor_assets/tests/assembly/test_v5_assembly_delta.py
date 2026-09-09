from dataclasses import replace

import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.assembly_evidence import AssemblyClusterMembership, AssemblyPolicy
from factor_assets.contracts.factor_set import FactorSetSpec
from factor_assets.tests.assembly.test_engine import make_asset, make_decision, make_spec
from factor_assets.tests.assembly.test_production_and_similarity_artifact import make_admission, make_treatment


def _cluster(fid, micro, macro, representative=None):
    return AssemblyClusterMembership(
        fid, micro, macro, "clusters:v1", representative, f"cluster-evidence:{fid}"
    )


def test_t22_real_micro_and_macro_constraints_and_unknown_bucket():
    assets = [make_asset(x) for x in ("A", "B", "C", "D")]
    memberships = {
        "A": _cluster("A", "m1", "M"), "B": _cluster("B", "m1", "M"),
        "C": _cluster("C", "m2", "M"), "D": _cluster("D", None, None),
    }
    policy = AssemblyPolicy("p", "1", max_per_microcluster=1, max_per_macrocluster=2)
    result = FactorSetAssembler().assemble(
        make_spec("s", "s", "manual"), assets, assembly_policy=policy,
        cluster_memberships=memberships,
    )
    assert result.factor_ids == ("A", "C", "D")
    assert [(m.microcluster_id, m.macrocluster_id) for m in result.members] == [
        ("m1", "M"), ("m2", "M"), (None, None)
    ]


def test_t23_policy_identity_changes_while_member_content_can_match():
    spec = make_spec("s", "s", "manual")
    assets = [make_asset("A")]
    p1 = AssemblyPolicy("p", "1", quality_weight=.5, redundancy_weight=.5)
    p2 = AssemblyPolicy("p", "2", quality_weight=.7, redundancy_weight=.3)
    a = FactorSetAssembler().assemble(spec, assets, assembly_policy=p1)
    b = FactorSetAssembler().assemble(spec, assets, assembly_policy=p2)
    assert a.factor_ids == b.factor_ids
    assert a.policy_hash != b.policy_hash
    assert a.assembly_hash != b.assembly_hash


def test_t24_required_pareto_objective_missing_rejects_only_that_candidate():
    spec = make_spec("s", "s", "pareto_front")
    full = replace(make_decision("A"), metadata={"objectives": {"quality": .8, "cost": .7}})
    missing = replace(make_decision("B"), metadata={"objectives": {"quality": .9}})
    policy = AssemblyPolicy("p", "1", required_objectives=("quality", "cost"))
    result = FactorSetAssembler().assemble(
        spec, [make_asset("A"), make_asset("B")],
        selection_decisions=[full, missing], assembly_policy=policy,
    )
    assert result.factor_ids == ("A",)


def test_t25_as_of_uses_actual_utc_and_does_not_select_future_decision():
    spec = make_spec("s", "s", "family_robust", selection_as_of="2024-01-01T01:30:00Z")
    early = make_decision("A", timestamp="2024-01-01T09:00:00+08:00")
    future = make_decision("A", approved=False, timestamp="2024-01-01T02:00:00Z")
    result = FactorSetAssembler().assemble(
        spec, [make_asset("A")], selection_decisions=[early, future]
    )
    assert result.factor_ids == ("A",)


def test_t25_only_explicit_cluster_representative_is_marked():
    assets = [make_asset(x) for x in ("A", "B", "C")]
    memberships = {x: _cluster(x, "m1", "M", representative="B") for x in ("A", "B", "C")}
    result = FactorSetAssembler().assemble(
        make_spec("s", "s", "manual"), assets, cluster_memberships=memberships
    )
    assert [m.factor_id for m in result.members if m.representative_of] == ["B"]


def test_t25_production_context_freeze_rejects_future_or_wrong_recipe():
    spec = make_spec("s", "s", "manual", selection_as_of="2024-08-02T00:00:00Z")
    with pytest.raises(ValueError, match="recipe_ref"):
        FactorSetAssembler().assemble(
            spec, [make_asset("A")], production=True,
            admission_artifacts={"A": make_admission("A", recipe_ref="recipe:wrong")},
            treatment_selection_artifacts={"A": make_treatment("A")},
        )
    with pytest.raises(ValueError, match="unavailable"):
        FactorSetAssembler().assemble(
            spec, [make_asset("A")], production=True,
            admission_artifacts={"A": make_admission("A", created_at="2024-08-03T00:00:00Z")},
            treatment_selection_artifacts={"A": make_treatment("A")},
        )
