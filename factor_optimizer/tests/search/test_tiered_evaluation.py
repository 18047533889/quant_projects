"""Tests for the tiered (funnel) evaluation policy and scheduler."""

import pytest

from factor_optimizer.search.tiered_evaluation import (
    EvaluationTier,
    TierEvaluationOutcome,
    TieredEvaluationPolicy,
    TieredEvaluationScheduler,
)


def _advance(sched, candidate, passed, recipe="recipe:v1"):
    job = sched.issue(candidate, recipe)
    outcome = TierEvaluationOutcome(
        outcome_id=f"outcome:{job.job_id}", job_id=job.job_id,
        candidate_id=candidate, recipe_version=recipe, tier_name=job.tier_name,
        attempt=job.attempt, evaluation_ref=f"evaluation:{job.job_id}", passed=passed,
    )
    return sched.advance(outcome)


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
    nxt = _advance(sched, "a", True)
    assert nxt == "Tier2"
    assert sched.tier_name_for("a") == "Tier2"
    # 'b' fails Tier1 -> pruned, never reaches Tier2
    assert _advance(sched, "b", False) is None
    assert sched.is_pruned("b")
    with pytest.raises(ValueError):
        sched.tier_name_for("b")


def test_full_backtest_only_for_survivors():
    """Tier4 (full backtest) is scheduled only for funnel survivors."""
    sched = TieredEvaluationScheduler(_policy())
    # A candidate pruned at Tier2 must never see Tier4.
    assert _advance(sched, "x", True) == "Tier2"
    assert _advance(sched, "x", False) is None
    assert sched.is_pruned("x")
    # A survivor advances through every tier and finally hits Tier4.
    assert _advance(sched, "y", True) == "Tier2"
    assert _advance(sched, "y", True) == "Tier3"
    assert _advance(sched, "y", True) == "Tier4"
    assert sched.tier_name_for("y") == "Tier4"
    assert sched.fidelity_for("y") == 4
    # Completing Tier4 ends the funnel (full backtest ran for the survivor).
    assert _advance(sched, "y", True) is None
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
    _advance(sched, "a", True)
    _advance(sched, "b", False)
    restored_sched = TieredEvaluationScheduler.from_dict(sched.to_dict())
    assert restored_sched.tier_name_for("a") == "Tier2"
    assert restored_sched.is_pruned("b")
    assert restored_sched.policy.to_dict() == policy.to_dict()


def _outcome(job, **changes):
    values = dict(
        outcome_id=f"outcome:{job.job_id}", job_id=job.job_id,
        candidate_id=job.candidate_id, recipe_version=job.recipe_version,
        tier_name=job.tier_name, attempt=job.attempt,
        evaluation_ref=f"evaluation:{job.job_id}", passed=True,
    )
    values.update(changes)
    return TierEvaluationOutcome(**values)


def test_v6_rejects_unissued_skip_wrong_recipe_job_and_attempt():
    sched = TieredEvaluationScheduler(_policy())
    job = sched.issue("a", "recipe:v1")
    with pytest.raises(ValueError, match="tier_name"):
        sched.advance(_outcome(job, tier_name="Tier4"))
    with pytest.raises(ValueError, match="recipe_version"):
        sched.advance(_outcome(job, recipe_version="recipe:v2"))
    with pytest.raises(ValueError, match="unknown issued job"):
        sched.advance(_outcome(job, job_id="not-issued"))
    with pytest.raises(ValueError, match="attempt"):
        sched.advance(_outcome(job, attempt=2))


def test_v6_strict_bool_idempotence_terminal_and_restart_replay():
    sched = TieredEvaluationScheduler(_policy())
    job = sched.issue("a", "recipe:v1")
    with pytest.raises(TypeError, match="strict bool"):
        _outcome(job, passed="false")
    outcome = _outcome(job)
    assert sched.advance(outcome) == "Tier2"
    assert sched.advance(outcome) == "Tier2"
    restored = TieredEvaluationScheduler.from_dict(sched.to_dict())
    assert restored.advance(outcome) == "Tier2"
    with pytest.raises(ValueError, match="different payload"):
        restored.advance(_outcome(job, evaluation_ref="evaluation:tampered"))

    for _ in range(3):
        _advance(restored, "a", True)
    assert restored.is_completed("a")
    with pytest.raises(ValueError, match="completed"):
        restored.tier_for("a")


def test_v6_same_evaluation_ref_cannot_count_as_independent_confirmation():
    policy = TieredEvaluationPolicy(tiers=(
        EvaluationTier("Tier1", 1.0, 0, promote_after=2),
        EvaluationTier("Tier2", 2.0, 1),
    ))
    sched = TieredEvaluationScheduler(policy)
    first = sched.issue("a", "recipe:v1")
    assert sched.advance(_outcome(first, evaluation_ref="evaluation:same")) == "Tier1"
    second = sched.issue("a", "recipe:v1")
    with pytest.raises(ValueError, match="evaluation_ref was already consumed"):
        sched.advance(_outcome(
            second, outcome_id="outcome:second", evaluation_ref="evaluation:same"
        ))
    assert sched.promote_count("a", "Tier1") == 1
