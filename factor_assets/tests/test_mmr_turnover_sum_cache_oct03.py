import math
import random
from dataclasses import replace

import pytest

from factor_assets.contracts.assembly_evidence import AssemblyClusterMembership
from factor_assets.tests.assembly.test_v6_constrained_mmr import (
    assemble,
    evidence,
    make_asset,
    policy,
)


def _old_scan_oracle(
    assets,
    typed,
    memberships,
    *,
    target,
    family_limit,
    max_micro,
    max_macro,
    turnover_budget,
    similarity,
    quality_weight=0.5,
    redundancy_weight=0.5,
):
    """Independent former selected-list feasibility and sum scan."""
    by_id = {asset.factor_id: asset for asset in assets}
    remaining = set(by_id)
    selected = []
    redundancy = {factor_id: 0.0 for factor_id in remaining}
    rejections = {}
    provider_pairs = []
    steps = []

    def groups(factor_id):
        membership = memberships.get(factor_id)
        return (
            membership.microcluster_id if membership and membership.microcluster_id else "UNKNOWN_MICRO",
            membership.macrocluster_id if membership and membership.macrocluster_id else "UNKNOWN_MACRO",
        )

    def rejection(factor_id):
        asset = by_id[factor_id]
        if family_limit is not None and sum(
            member.family == asset.family for member in selected
        ) >= family_limit:
            return "FAMILY_LIMIT"
        micro, macro = groups(factor_id)
        if sum(groups(member.factor_id)[0] == micro for member in selected) >= max_micro:
            return "MICROCLUSTER_LIMIT"
        if sum(groups(member.factor_id)[1] == macro for member in selected) >= max_macro:
            return "MACROCLUSTER_LIMIT"
        item = typed[factor_id]
        if item.turnover_score is None:
            return "MISSING_TURNOVER_EVIDENCE"
        used = sum(typed[member.factor_id].turnover_score or 0.0 for member in selected)
        if used + item.turnover_score > turnover_budget:
            return "TURNOVER_BUDGET"
        return None

    while remaining and len(selected) < target:
        for factor_id in sorted(tuple(remaining)):
            why = rejection(factor_id)
            if why is not None:
                rejections[factor_id] = why
                remaining.remove(factor_id)
        if not remaining:
            break
        chosen_id = None
        chosen_score = float("-inf")
        for factor_id in sorted(remaining):
            score = (
                quality_weight * typed[factor_id].quality_score
                - redundancy_weight * redundancy[factor_id]
            )
            if score > chosen_score:
                chosen_id, chosen_score = factor_id, score
        selected.append(by_id[chosen_id])
        remaining.remove(chosen_id)
        used_after_commit = sum(
            typed[member.factor_id].turnover_score or 0.0 for member in selected
        )
        steps.append(
            (
                chosen_id,
                typed[chosen_id].quality_score,
                redundancy[chosen_id],
                chosen_score,
                target - len(selected),
                len(assets) - len(selected),
                turnover_budget - used_after_commit,
            )
        )
        if len(selected) >= target:
            break
        for factor_id in sorted(tuple(remaining)):
            why = rejection(factor_id)
            if why is not None:
                rejections[factor_id] = why
                remaining.remove(factor_id)
                continue
            provider_pairs.append((factor_id, chosen_id))
            redundancy[factor_id] = max(
                redundancy[factor_id], abs(similarity(factor_id, chosen_id))
            )

    for factor_id in remaining:
        rejections[factor_id] = "SELECTION_LIMIT_REACHED"
    return tuple(item.factor_id for item in selected), rejections, provider_pairs, steps


def _assert_output_matches_oracle(result, actual_pairs, expected):
    ids, rejections, provider_pairs, expected_steps = expected
    assert result.factor_ids == ids
    assert dict(result.selection_evidence.rejections) == rejections
    assert actual_pairs == provider_pairs
    steps = [
        (
            step.factor_id,
            step.quality,
            step.max_redundancy,
            step.objective,
            step.constraint_headroom["set_slots"],
            step.constraint_headroom["capacity_slots"],
            step.constraint_headroom["turnover_remaining"],
        )
        for step in result.selection_evidence.steps
    ]
    assert steps == expected_steps


@pytest.mark.parametrize(
    ("budget", "expected_ids"),
    [
        (1e16, ("A", "B", "C", "D")),
        (math.nextafter(1e16, math.inf), ("A", "B", "C", "D", "E")),
    ],
)
def test_compensated_turnover_sum_matches_old_scan_at_adjacent_budgets(budget, expected_ids):
    # Python 3.12 sum([-1e16, 1, 1e16, 1]) == 2. A running += loses a unit
    # and changes E's decision. Finite negative turnover remains legal under
    # the current evidence contract.
    ids = ("A", "B", "C", "D", "E")
    turnover = (-1e16, 1.0, 1e16, 1.0, 1e16)
    quality = {factor_id: 0.95 - index * 0.05 for index, factor_id in enumerate(ids)}
    assets = [make_asset(factor_id) for factor_id in ids]
    typed = {
        factor_id: evidence(factor_id, quality[factor_id], turnover=turnover[index])
        for index, factor_id in enumerate(ids)
    }
    similarity = lambda _a, _b: 0.0
    expected = _old_scan_oracle(
        assets,
        typed,
        {},
        target=len(ids),
        family_limit=None,
        max_micro=len(ids),
        max_macro=len(ids),
        turnover_budget=budget,
        similarity=similarity,
    )
    assert expected[0] == expected_ids
    actual_pairs = []
    result = assemble(
        ids,
        maximum=len(ids),
        assets=assets,
        provider=lambda a, b: actual_pairs.append((a, b)) or 0.0,
        evidence_map=typed,
        assembly_policy=policy(
            max_per_microcluster=len(ids),
            max_per_macrocluster=len(ids),
            turnover_budget=budget,
        ),
    )
    _assert_output_matches_oracle(result, actual_pairs, expected)


def test_large_tied_batch_matches_old_scan_constraints_and_selection_trace():
    count, target = 1000, 50
    rng = random.Random(80103)
    assets, typed, memberships = [], {}, {}
    for index in range(count):
        factor_id = f"F{index:04d}"
        family = None if index % 13 == 0 else f"family-{index % 80:02d}"
        asset = replace(make_asset(factor_id), family=family)
        assets.append(asset)
        quality = 0.75 + (index % 9) / 100.0
        typed[factor_id] = evidence(
            factor_id, quality, turnover=0.02 + rng.randrange(0, 5) / 100.0
        )
        memberships[factor_id] = AssemblyClusterMembership(
            factor_id,
            f"micro-{index % 24:02d}",
            f"macro-{index % 12:02d}",
            "clusters:turnover-cache-oracle",
            evidence_ref=f"cluster:{factor_id}",
        )

    family_limit, max_micro, max_macro = 3, 3, 5
    turnover_budget = 10.0
    similarity = lambda a, b: ((int(a[1:]) * 31 + int(b[1:]) * 17) % 101) / 100
    expected = _old_scan_oracle(
        assets,
        typed,
        memberships,
        target=target,
        family_limit=family_limit,
        max_micro=max_micro,
        max_macro=max_macro,
        turnover_budget=turnover_budget,
        similarity=similarity,
    )
    actual_pairs = []
    result = assemble(
        [asset.factor_id for asset in assets],
        maximum=target,
        assets=assets,
        provider=lambda a, b: actual_pairs.append((a, b)) or similarity(a, b),
        evidence_map=typed,
        assembly_policy=policy(
            max_per_microcluster=max_micro,
            max_per_macrocluster=max_macro,
            turnover_budget=turnover_budget,
        ),
        family_constraints=f"max_per_family={family_limit}",
        cluster_memberships=memberships,
    )
    _assert_output_matches_oracle(result, actual_pairs, expected)


def test_zero_target_is_rejected_before_similarity_calls():
    with pytest.raises(ValueError, match="max_factors must be >= 1"):
        assemble(
            ["A", "B"],
            maximum=0,
            assembly_policy=policy(turnover_budget=1.0),
        )


def test_one_target_matches_scan_and_does_not_call_similarity_provider():
    ids = ("A", "B", "C")
    typed = {fid: evidence(fid, 0.9 - index * 0.1, turnover=0.1) for index, fid in enumerate(ids)}
    calls = []
    result = assemble(
        ids,
        maximum=1,
        provider=lambda a, b: calls.append((a, b)) or 0.0,
        evidence_map=typed,
        assembly_policy=policy(turnover_budget=1.0),
    )
    assert result.factor_ids == ("A",)
    assert calls == []


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, -math.inf])
def test_turnover_evidence_still_rejects_boolean_and_nonfinite_values(bad):
    with pytest.raises((TypeError, ValueError)):
        evidence("bad-turnover", 0.5, turnover=bad)
