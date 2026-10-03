import math

import pytest

from factor_assets.tests.assembly.test_v6_constrained_mmr import assemble, evidence, policy


@pytest.mark.parametrize(
    ("budget", "expected_ids", "expected_rejections"),
    [
        (0.6000000000000001, ("A", "B", "C"), {}),
        (math.nextafter(0.6000000000000001, -math.inf), ("A", "B"), {"C": "TURNOVER_BUDGET"}),
    ],
)
def test_turnover_sum_preserves_exact_boundary_and_adjacent_float(
    budget, expected_ids, expected_rejections
):
    scores = {"A": 0.99, "B": 0.98, "C": 0.97}
    turnovers = {"A": 0.1, "B": 0.2, "C": 0.3}
    result = assemble(
        list(scores),
        maximum=3,
        provider=lambda _a, _b: 0.0,
        evidence_map={
            fid: evidence(fid, score, turnover=turnovers[fid], health=0.9)
            for fid, score in scores.items()
        },
        assembly_policy=policy(
            max_per_microcluster=3,
            max_per_macrocluster=3,
            turnover_budget=budget,
        ),
    )
    assert result.factor_ids == expected_ids
    assert dict(result.selection_evidence.rejections) == expected_rejections
    expected_used = [sum(turnovers[fid] for fid in expected_ids[:i]) for i in range(1, len(expected_ids) + 1)]
    assert [
        step.constraint_headroom["turnover_remaining"]
        for step in result.selection_evidence.steps
    ] == [budget - used for used in expected_used]


def test_none_family_and_missing_membership_keep_hashable_unknown_cluster_counts():
    from dataclasses import replace

    from factor_assets.tests.assembly.test_engine import make_asset

    assets = [
        replace(make_asset("A"), family=None),
        replace(make_asset("B"), family="other"),
    ]
    result = assemble(
        ["A", "B"],
        maximum=2,
        assets=assets,
        provider=lambda _a, _b: pytest.fail("infeasible candidate must not trigger similarity"),
        evidence_map={"A": evidence("A", 0.9), "B": evidence("B", 0.8)},
        assembly_policy=policy(max_per_microcluster=1, max_per_macrocluster=1),
        family_constraints="max_per_family=1",
        cluster_memberships={},
    )
    assert result.factor_ids == ("A",)
    assert result.selection_evidence.rejections == {"B": "MICROCLUSTER_LIMIT"}


def test_similarity_provider_error_propagates_with_same_pair_order():
    calls = []

    def provider(a, b):
        calls.append((a, b))
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        assemble(["A", "B"], provider=provider)
    assert calls == [("B", "A")]
