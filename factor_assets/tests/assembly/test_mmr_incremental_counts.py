from dataclasses import replace
from time import perf_counter
import math
import pytest

from factor_assets.contracts.assembly_evidence import AssemblyClusterMembership
from factor_assets.tests.assembly.test_v6_constrained_mmr import (
    assemble,
    evidence,
    make_asset,
    policy,
)


@pytest.mark.parametrize("turnover_budget", [3.4, math.nextafter(3.4, math.inf)])
def test_incremental_constraint_counts_match_independent_scan_oracle(turnover_budget):
    """Compare MMR output/evidence/calls with the former scan-based rules."""
    count = 220
    target = 34
    assets = [
        replace(make_asset(f"F{i:03d}"), family=f"family-{i % 45:02d}")
        for i in range(count)
    ]
    scores = {asset.factor_id: 0.55 + (i % 37) / 100 for i, asset in enumerate(assets)}
    typed = {
        asset.factor_id: evidence(
            asset.factor_id,
            scores[asset.factor_id],
            turnover=0.05 + (i % 6) * 0.04,
            health=0.4 if i % 19 == 0 else 0.9,
        )
        for i, asset in enumerate(assets)
    }
    memberships = {
        asset.factor_id: AssemblyClusterMembership(
            asset.factor_id, f"micro-{i % 17:02d}", f"macro-{i % 9:02d}",
            "clusters:oracle", evidence_ref=f"cluster:{asset.factor_id}",
        )
        for i, asset in enumerate(assets)
    }
    family_limit = 2
    max_micro = 3
    max_macro = 5
    health_floor = 0.5
    q_weight = r_weight = 0.5

    def similarity(a, b):
        ai, bi = int(a[1:]), int(b[1:])
        return ((min(ai, bi) * 31 + max(ai, bi) * 17) % 101) / 100

    # Deliberately use the former selected-list scans as the independent
    # feasibility oracle, including their rejection priority and float sums.
    selected = []
    remaining = {asset.factor_id for asset in assets}
    redundancy = {factor_id: 0.0 for factor_id in remaining}
    rejected = {}
    expected_pairs = []
    expected_redundancy = []
    by_id = {asset.factor_id: asset for asset in assets}
    groups = lambda fid: (
        memberships[fid].microcluster_id,
        memberships[fid].macrocluster_id,
    )

    def reason(fid):
        candidate = by_id[fid]
        if sum(item.family == candidate.family for item in selected) >= family_limit:
            return "FAMILY_LIMIT"
        micro, macro = groups(fid)
        if sum(groups(item.factor_id)[0] == micro for item in selected) >= max_micro:
            return "MICROCLUSTER_LIMIT"
        if sum(groups(item.factor_id)[1] == macro for item in selected) >= max_macro:
            return "MACROCLUSTER_LIMIT"
        item_evidence = typed[fid]
        if item_evidence.health_score is None:
            return "MISSING_HEALTH_EVIDENCE"
        if item_evidence.health_score < health_floor:
            return "HEALTH_FLOOR"
        if item_evidence.turnover_score is None:
            return "MISSING_TURNOVER_EVIDENCE"
        used = sum(typed[item.factor_id].turnover_score or 0.0 for item in selected)
        if used + item_evidence.turnover_score > turnover_budget:
            return "TURNOVER_BUDGET"
        return None

    scan_started = perf_counter()
    while remaining and len(selected) < target:
        for fid in sorted(tuple(remaining)):
            rejection = reason(fid)
            if rejection is not None:
                rejected[fid] = rejection
                remaining.remove(fid)
        if not remaining:
            break
        chosen_id = None
        best_score = float("-inf")
        for fid in sorted(remaining):
            score = q_weight * scores[fid] - r_weight * redundancy[fid]
            if score > best_score:
                chosen_id, best_score = fid, score
        assert chosen_id is not None
        selected.append(by_id[chosen_id])
        remaining.remove(chosen_id)
        expected_redundancy.append(redundancy[chosen_id])
        if len(selected) >= target:
            break
        for fid in sorted(tuple(remaining)):
            rejection = reason(fid)
            if rejection is not None:
                rejected[fid] = rejection
                remaining.remove(fid)
                continue
            expected_pairs.append((fid, chosen_id))
            redundancy[fid] = max(redundancy[fid], similarity(fid, chosen_id))

    for fid in remaining:
        rejected[fid] = "SELECTION_LIMIT_REACHED"

    scan_seconds = perf_counter() - scan_started
    actual_pairs = []
    optimized_started = perf_counter()
    result = assemble(
        [asset.factor_id for asset in assets],
        maximum=target,
        provider=lambda a, b: actual_pairs.append((a, b)) or similarity(a, b),
        evidence_map=typed,
        assembly_policy=policy(
            quality_weight=q_weight,
            redundancy_weight=r_weight,
            max_per_microcluster=max_micro,
            max_per_macrocluster=max_macro,
            turnover_budget=turnover_budget,
            health_floor=health_floor,
        ),
        assets=assets,
        family_constraints=f"max_per_family={family_limit}",
        cluster_memberships=memberships,
    )
    optimized_seconds = perf_counter() - optimized_started

    expected_ids = tuple(asset.factor_id for asset in selected)
    assert result.factor_ids == expected_ids
    assert dict(result.selection_evidence.rejections) == rejected
    assert actual_pairs == expected_pairs
    assert [step.max_redundancy for step in result.selection_evidence.steps] == expected_redundancy
    assert [step.factor_id for step in result.selection_evidence.steps] == list(expected_ids)
    assert [
        step.objective for step in result.selection_evidence.steps
    ] == [
        q_weight * scores[fid] - r_weight * red
        for fid, red in zip(expected_ids, expected_redundancy)
    ]
    print(
        "bounded MMR AB: 220 candidates, target 34; "
        f"scan oracle {scan_seconds:.6f}s, public optimized path {optimized_seconds:.6f}s; "
        f"selected {len(expected_ids)}, rejected {len(rejected)}, "
        f"provider calls {len(expected_pairs)} (both paths)"
    )
