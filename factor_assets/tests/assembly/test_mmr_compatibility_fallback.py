from dataclasses import replace

import pytest

from factor_assets.contracts.assembly_evidence import AssemblyClusterMembership
from factor_assets.tests.assembly.test_engine import make_asset
from factor_assets.tests.assembly.test_v6_constrained_mmr import assemble, evidence, policy


def test_list_family_with_limit_matches_scan_fallback():
    assets = [
        replace(make_asset("A"), family=["shared"]),
        replace(make_asset("B"), family=["shared"]),
        replace(make_asset("C"), family=["other"]),
    ]
    result = assemble(
        ["A", "B", "C"], maximum=3, assets=assets,
        evidence_map={fid: evidence(fid, score) for fid, score in [("A", .9), ("B", .8), ("C", .7)]},
        provider=lambda *_: 0.0,
        family_constraints="max_per_family=1",
        assembly_policy=policy(max_per_microcluster=3, max_per_macrocluster=3),
    )
    assert result.factor_ids == ("A", "C")
    assert dict(result.selection_evidence.rejections) == {"B": "FAMILY_LIMIT"}


def test_list_family_without_limit_is_never_hashed_or_compared():
    assets = [replace(make_asset(fid), family=["unhashable"]) for fid in ("A", "B")]
    result = assemble(
        ["A", "B"], maximum=2, assets=assets,
        evidence_map={"A": evidence("A", .9), "B": evidence("B", .8)},
        provider=lambda *_: 0.0,
        assembly_policy=policy(max_per_microcluster=2, max_per_macrocluster=2),
    )
    assert result.factor_ids == ("A", "B")


def test_list_cluster_ids_match_original_scan_fallback():
    assets = [make_asset("A"), make_asset("B")]
    class HashableList(list):
        __hash__ = object.__hash__

    memberships = {
        fid: AssemblyClusterMembership(
            fid, HashableList(["same"]), HashableList(["same"]), "clusters:test", evidence_ref=f"cluster:{fid}"
        )
        for fid in ("A", "B")
    }
    result = assemble(
        ["A", "B"], maximum=2, assets=assets,
        evidence_map={"A": evidence("A", .9), "B": evidence("B", .8)},
        provider=lambda *_: pytest.fail("infeasible candidate must not reach similarity"),
        assembly_policy=policy(max_per_microcluster=1, max_per_macrocluster=1),
        cluster_memberships=memberships,
    )
    assert result.factor_ids == ("A",)
    assert dict(result.selection_evidence.rejections) == {"B": "MICROCLUSTER_LIMIT"}


def test_non_reflexive_nan_family_matches_original_scan_behavior():
    nan = float("nan")
    assets = [replace(make_asset(fid), family=nan) for fid in ("A", "B", "C")]
    result = assemble(
        ["A", "B", "C"], maximum=3, assets=assets,
        evidence_map={fid: evidence(fid, score) for fid, score in [("A", .9), ("B", .8), ("C", .7)]},
        provider=lambda *_: 0.0,
        family_constraints="max_per_family=1",
        assembly_policy=policy(max_per_microcluster=3, max_per_macrocluster=3),
    )
    assert result.factor_ids == ("A", "B", "C")
    assert dict(result.selection_evidence.rejections) == {}
