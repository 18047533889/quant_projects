from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from factor_assets.assembly import FactorSetAssembler
from factor_assets.assembly.engine import _diverse_mmr_rank
from factor_assets.contracts.assembly_evidence import (
    AssemblyClusterMembership,
    AssemblyEvidence,
    AssemblyPolicy,
)
from factor_assets.contracts.similarity import SimilarityArtifact
from factor_assets.tests.assembly.test_engine import make_asset, make_decision, make_spec


def decided(factor_id, quality=None):
    decision = make_decision(factor_id, timestamp="2026-01-01T00:00:00Z")
    metadata = {} if quality is None else {"quality": quality}
    object.__setattr__(decision, "metadata", MappingProxyType(metadata))
    return decision


def evidence(factor_id, score, *, turnover=None, health=None):
    return AssemblyEvidence(
        factor_id,
        quality_score=score,
        quality_definition="v6.cross_sectional_quality",
        quality_unit="unit_interval",
        quality_policy_ref="quality-policy:v6",
        evidence_refs=(f"quality:{factor_id}",),
        turnover_score=turnover,
        health_score=health,
    )


def policy(**kwargs):
    limits = {"max_per_microcluster": 5, "max_per_macrocluster": 5}
    limits.update(kwargs)
    return AssemblyPolicy(
        "mmr-v6",
        "2.0.0",
        selection_algorithm_version="constrained_mmr.v2",
        quality_definition="v6.cross_sectional_quality",
        quality_unit="unit_interval",
        quality_policy_ref="quality-policy:v6",
        similarity_view="rank_corr",
        **limits,
    )


def assemble(ids, *, maximum=2, provider=lambda a, b: 0.0, evidence_map=None,
             assembly_policy=None, assets=None, family_constraints=None, **kwargs):
    return FactorSetAssembler().assemble(
        make_spec("v6-set", "V6", "diverse", max_factors=maximum,
                  family_constraints=family_constraints),
        assets or [make_asset(fid) for fid in ids],
        selection_decisions=[decided(fid) for fid in ids],
        similarity_provider=provider,
        assembly_evidence=evidence_map or {fid: evidence(fid, 1.0 - i / 10) for i, fid in enumerate(ids)},
        assembly_policy=assembly_policy or policy(),
        **kwargs,
    )


def test_x10_missing_quality_is_rejected_not_replaced_by_recency_or_batch_size():
    qualities = {"A": evidence("A", 0.9), "B": evidence("B", 0.8)}
    first = assemble(["missing", "A", "B"], evidence_map=qualities)
    expanded = assemble(["missing", "A", "B", "unrelated"], evidence_map=qualities)
    assert first.factor_ids == expanded.factor_ids == ("A", "B")
    assert first.selection_evidence.rejections["missing"] == "MISSING_QUALITY_EVIDENCE"


@pytest.mark.parametrize("quality", [None, np.nan])
def test_retired_order_only_mmr_cannot_fallback_coerce_or_rank_unbounded(quality):
    assets = [make_asset(f"F{i}") for i in range(20)]
    decisions = {asset.factor_id: decided(asset.factor_id, quality) for asset in assets}
    calls = []
    with pytest.raises(RuntimeError, match="retired"):
        _diverse_mmr_rank(
            assets,
            decisions,
            lambda a, b: calls.append((a, b)) or np.nan,
            policy(),
        )
    # Retirement occurs before recency fallback, float coercion, pair calls,
    # or an accidental complete ranking that ignores FactorSetSpec.max_factors.
    assert calls == []


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf, True, 1.01, -1.01])
def test_x11_invalid_bare_similarity_fails_closed(bad):
    with pytest.raises((TypeError, ValueError), match="similarity"):
        assemble(["A", "B"], provider=lambda a, b: bad)


def test_x11_legal_negative_correlation_uses_magnitude():
    result = assemble(["A", "B", "C"], provider=lambda a, b: -0.8 if {a, b} == {"A", "B"} else 0.0)
    assert result.factor_ids == ("A", "C")


def test_x11_production_refuses_bare_similarity_without_pair_evidence():
    from factor_assets.tests.assembly.test_production_and_similarity_artifact import (
        make_admission,
        make_treatment,
    )

    ids = ["A", "B"]
    with pytest.raises(TypeError, match="typed SimilarityArtifact"):
        assemble(
            ids,
            provider=lambda a, b: 0.2,
            production=True,
            admission_artifacts={fid: make_admission(fid) for fid in ids},
            treatment_selection_artifacts={fid: make_treatment(fid) for fid in ids},
        )


def test_x11_typed_similarity_requires_exact_pair_and_context():
    def wrong_pair(a, b):
        return SimilarityArtifact(
            "X", "Y", {"rank_corr": 0.2}, snapshot_ref="snapshot:default",
            universe_ref="universe:default", window_ref="2025", metric="spearman",
        )
    with pytest.raises(ValueError, match="pair"):
        assemble(["A", "B"], provider=wrong_pair)

    def wrong_context(a, b):
        return SimilarityArtifact(
            a, b, {"rank_corr": 0.2}, snapshot_ref="snapshot:wrong",
            universe_ref="universe:default", window_ref="2025", metric="spearman",
        )
    with pytest.raises(ValueError, match="snapshot"):
        assemble(["A", "B"], provider=wrong_context)


def test_x12_k1_performs_zero_pair_calls():
    calls = []
    result = assemble(["A", "B", "C"], maximum=1, provider=lambda a, b: calls.append((a, b)) or 0.0)
    assert result.factor_ids == ("A",)
    assert calls == []


def test_x12_pair_calls_scale_with_f_times_k_and_are_cached():
    calls = []
    ids = [f"F{i:02d}" for i in range(20)]
    result = assemble(
        ids,
        maximum=3,
        provider=lambda a, b: calls.append(tuple(sorted((a, b)))) or 0.0,
        evidence_map={fid: evidence(fid, 1.0 - i / 40) for i, fid in enumerate(ids)},
    )
    assert result.size == 3
    assert len(calls) == 19 + 18
    assert len(set(calls)) == len(calls)


def test_x13_infeasible_b_never_penalizes_c_and_rejections_are_structured():
    assets = [
        replace(make_asset("A"), family="same"),
        replace(make_asset("B"), family="same"),
        replace(make_asset("C"), family="C"),
        replace(make_asset("D"), family="D"),
    ]
    scores = {"A": 1.0, "B": 0.99, "C": 0.8, "D": 0.7}
    similarities = {frozenset(("B", "C")): 0.9}
    result = assemble(
        list(scores),
        assets=assets,
        evidence_map={fid: evidence(fid, score) for fid, score in scores.items()},
        provider=lambda a, b: similarities.get(frozenset((a, b)), 0.0),
        assembly_policy=policy(max_per_microcluster=5, max_per_macrocluster=5),
        family_constraints="max_per_family=1",
    )
    assert result.factor_ids == ("A", "C")
    assert result.selection_evidence.algorithm_version == "constrained_mmr.v2"
    assert result.selection_evidence.rejections["B"] == "FAMILY_LIMIT"
    assert [step.factor_id for step in result.selection_evidence.steps] == ["A", "C"]


def test_x13_cluster_and_resource_constraints_are_checked_before_commit():
    scores = {"A": 1.0, "B": 0.95, "C": 0.9, "D": 0.8, "E": 0.7}
    memberships = {
        "A": AssemblyClusterMembership("A", "m1", "M1", "clusters:v6", evidence_ref="c:A"),
        "B": AssemblyClusterMembership("B", "m1", "M1", "clusters:v6", evidence_ref="c:B"),
        "C": AssemblyClusterMembership("C", "m2", "M1", "clusters:v6", evidence_ref="c:C"),
        "D": AssemblyClusterMembership("D", "m3", "M2", "clusters:v6", evidence_ref="c:D"),
        "E": AssemblyClusterMembership("E", "m4", "M3", "clusters:v6", evidence_ref="c:E"),
    }
    typed = {
        "A": evidence("A", 1.0, turnover=0.2, health=0.9),
        "B": evidence("B", 0.95, turnover=0.1, health=0.9),
        "C": evidence("C", 0.9, turnover=0.1, health=0.9),
        "D": evidence("D", 0.8, turnover=0.4, health=0.9),
        "E": evidence("E", 0.7, turnover=0.1, health=0.5),
    }
    calls = []
    result = assemble(
        list(scores), maximum=3, evidence_map=typed,
        provider=lambda a, b: calls.append(frozenset((a, b))) or 0.0,
        assembly_policy=policy(
            max_per_microcluster=1, max_per_macrocluster=2,
            turnover_budget=0.5, health_floor=0.6,
        ),
        cluster_memberships=memberships,
    )
    assert result.factor_ids == ("A", "C")
    assert result.selection_evidence.rejections == {
        "B": "MICROCLUSTER_LIMIT",
        "D": "TURNOVER_BUDGET",
        "E": "HEALTH_FLOOR",
    }
    assert all("B" not in pair for pair in calls)
