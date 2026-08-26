"""Tests for the tiered (funnel) evaluation policy and scheduler."""

import pytest

from factor_optimizer.search.tiered_evaluation import (
    EvaluationTier,
    TieredEvaluationPolicy,
    TieredEvaluationScheduler,
)


def _policy():
    return TieredEvaluationPolicy(
        tiers=(
            EvaluationTier("Tier1", 1.0, 0),
            EvaluationTier("Tier2", 5.0, 1),
            EvaluationTier("Tier3", 25.0, 3),
            EvaluationTier("Tier4", 100.0, 4),
        )
    )


def test_tier_survivor_only_promotes():
    """Only candidates that pass a tier advance; failures are pruned."""
    sched = TieredEvaluationScheduler(_policy())
    assert sched.tier_name_for("a") == "Tier1"
    # 'a' passes Tier1 -> promotes to Tier2
    nxt = sched.advance("a", "Tier1", passed=True)
    assert nxt == "Tier2"
    assert sched.tier_name_for("a") == "Tier2"
    # 'b' fails Tier1 -> pruned, never reaches Tier2
    assert sched.advance("b", "Tier1", passed=False) is None
    assert sched.is_pruned("b")
    with pytest.raises(ValueError):
        sched.tier_name_for("b")


def test_full_backtest_only_for_survivors():
    """Tier4 (full backtest) is scheduled only for funnel survivors."""
    sched = TieredEvaluationScheduler(_policy())
    # A candidate pruned at Tier2 must never see Tier4.
    assert sched.advance("x", "Tier1", True) == "Tier2"
    assert sched.advance("x", "Tier2", False) is None
    assert sched.is_pruned("x")
    # A survivor advances through every tier and finally hits Tier4.
    assert sched.advance("y", "Tier1", True) == "Tier2"
    assert sched.advance("y", "Tier2", True) == "Tier3"
    assert sched.advance("y", "Tier3", True) == "Tier4"
    assert sched.tier_name_for("y") == "Tier4"
    assert sched.fidelity_for("y") == 4
    # Completing Tier4 ends the funnel (full backtest ran for the survivor).
    assert sched.advance("y", "Tier4", True) is None
    assert sched.is_completed("y")


def test_tiered_config_validation():
    # Non-positive cost multiplier rejected.
    with pytest.raises(ValueError):
        EvaluationTier("T1", 0.0, 0)
    with pytest.raises(ValueError):
        EvaluationTier("T1", -3.0, 0)
    # Fidelity out of [0,4] rejected.
    with pytest.raises(ValueError):
        EvaluationTier("T1", 1.0, 9)
    # Non-increasing cost order rejected (funnel must get more expensive).
    with pytest.raises(ValueError):
        TieredEvaluationPolicy(
            tiers=(
                EvaluationTier("Tier1", 5.0, 0),
                EvaluationTier("Tier2", 2.0, 1),
            )
        )
    # Non-increasing fidelity order rejected.
    with pytest.raises(ValueError):
        TieredEvaluationPolicy(
            tiers=(
                EvaluationTier("Tier1", 1.0, 3),
                EvaluationTier("Tier2", 2.0, 1),
            )
        )
    # Empty / non-list tiers rejected.
    with pytest.raises(ValueError):
        TieredEvaluationPolicy(tiers=[])
    # Duplicate names rejected.
    with pytest.raises(ValueError):
        TieredEvaluationPolicy(
            tiers=(
                EvaluationTier("Tier1", 1.0, 0),
                EvaluationTier("Tier1", 2.0, 1),
            )
        )


def test_policy_dict_round_trip():
    policy = _policy()
    restored = TieredEvaluationPolicy.from_dict(policy.to_dict())
    assert restored.tiers == policy.tiers
    sched = TieredEvaluationScheduler(policy)
    sched.advance("a", "Tier1", True)
    sched.advance("b", "Tier1", False)
    restored_sched = TieredEvaluationScheduler.from_dict(sched.to_dict())
    assert restored_sched.tier_name_for("a") == "Tier2"
    assert restored_sched.is_pruned("b")
    assert restored_sched.policy.to_dict() == policy.to_dict()
