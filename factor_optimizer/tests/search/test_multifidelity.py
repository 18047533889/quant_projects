"""Tests for multi-fidelity evaluation."""

import pytest

from factor_optimizer.search.multifidelity import (
    FidelityTier,
    FidelitySpec,
    MultiFidelityScheduler,
    PromotionCriteria,
    FIDELITY_TIERS,
)


def test_fidelity_tier_enum():
    assert FidelityTier.L0.value == 0
    assert FidelityTier.L4.value == 4
    assert len(list(FidelityTier)) == 5


def test_fidelity_spec_validation():
    with pytest.raises(ValueError, match="sample_fraction"):
        FidelitySpec(tier=FidelityTier.L0, sample_fraction=0.0, num_metrics=1)

    with pytest.raises(ValueError, match="sample_fraction"):
        FidelitySpec(tier=FidelityTier.L0, sample_fraction=1.5, num_metrics=1)

    with pytest.raises(ValueError, match="num_metrics"):
        FidelitySpec(tier=FidelityTier.L0, sample_fraction=0.1, num_metrics=0)

    with pytest.raises(ValueError, match="cost_multiplier"):
        FidelitySpec(tier=FidelityTier.L0, sample_fraction=0.1, num_metrics=1, cost_multiplier=-1.0)


def test_fidelity_spec_creation():
    spec = FidelitySpec(
        tier=FidelityTier.L2,
        sample_fraction=0.5,
        num_metrics=5,
        cross_validate=False,
        compute_robustness=False,
        cost_multiplier=10.0,
    )

    assert spec.tier == FidelityTier.L2
    assert spec.sample_fraction == 0.5
    assert spec.num_metrics == 5
    assert not spec.cross_validate
    assert spec.cost_multiplier == 10.0


def test_fidelity_tiers_defaults():
    assert FidelityTier.L0 in FIDELITY_TIERS
    assert FidelityTier.L4 in FIDELITY_TIERS

    l0_spec = FIDELITY_TIERS[FidelityTier.L0]
    assert l0_spec.sample_fraction == 0.05
    assert l0_spec.num_metrics == 1
    assert not l0_spec.cross_validate

    l4_spec = FIDELITY_TIERS[FidelityTier.L4]
    assert l4_spec.sample_fraction == 1.0
    assert l4_spec.cross_validate
    assert l4_spec.compute_robustness


def test_promotion_criteria_min_score():
    criteria = PromotionCriteria(min_score=0.7)

    assert criteria.should_promote(score=0.8, rank=0, total=10)
    assert not criteria.should_promote(score=0.6, rank=0, total=10)


def test_promotion_criteria_top_k():
    criteria = PromotionCriteria(top_k_fraction=0.3)

    # Top 30% of 10 = top 3
    assert criteria.should_promote(score=0.5, rank=0, total=10)
    assert criteria.should_promote(score=0.5, rank=2, total=10)
    assert not criteria.should_promote(score=0.5, rank=3, total=10)


@pytest.mark.parametrize("total", [1, 2, 3, 4])
def test_promotion_criteria_top_k_keeps_best_small_cohort(total):
    criteria = PromotionCriteria(top_k_fraction=0.2)

    assert criteria.should_promote(score=0.5, rank=0, total=total)
    assert not criteria.should_promote(score=0.5, rank=1, total=total)


@pytest.mark.parametrize("total", [0, -1])
def test_promotion_criteria_top_k_rejects_empty_cohort(total):
    criteria = PromotionCriteria(top_k_fraction=0.2)

    assert not criteria.should_promote(score=0.5, rank=0, total=total)


def test_promotion_criteria_improvement():
    criteria = PromotionCriteria(min_improvement=0.1)

    assert criteria.should_promote(score=0.8, rank=0, total=10, baseline_score=0.6)
    assert not criteria.should_promote(score=0.65, rank=0, total=10, baseline_score=0.6)


def test_promotion_criteria_require_all():
    criteria = PromotionCriteria(
        min_score=0.7,
        top_k_fraction=0.5,
        require_all=True,
    )

    # Both conditions met
    assert criteria.should_promote(score=0.8, rank=2, total=10)

    # Only score met
    assert not criteria.should_promote(score=0.8, rank=6, total=10)

    # Only rank met
    assert not criteria.should_promote(score=0.6, rank=2, total=10)


def test_promotion_criteria_require_any():
    criteria = PromotionCriteria(
        min_score=0.7,
        top_k_fraction=0.5,
        require_all=False,
    )

    # Both conditions met
    assert criteria.should_promote(score=0.8, rank=2, total=10)

    # Only score met
    assert criteria.should_promote(score=0.8, rank=6, total=10)

    # Only rank met
    assert criteria.should_promote(score=0.6, rank=2, total=10)

    # Neither met
    assert not criteria.should_promote(score=0.6, rank=6, total=10)


def test_multifidelity_scheduler_defaults():
    scheduler = MultiFidelityScheduler()

    assert len(scheduler.tier_specs) == 5
    assert FidelityTier.L0 in scheduler.tier_specs

    # Check default promotion criteria
    assert FidelityTier.L0 in scheduler.promotion_criteria
    assert FidelityTier.L3 in scheduler.promotion_criteria
    assert FidelityTier.L4 not in scheduler.promotion_criteria  # No promotion from L4


def test_multifidelity_scheduler_get_spec():
    scheduler = MultiFidelityScheduler()

    l0_spec = scheduler.get_spec(FidelityTier.L0)
    assert l0_spec.tier == FidelityTier.L0
    assert l0_spec.sample_fraction == 0.05

    l4_spec = scheduler.get_spec(FidelityTier.L4)
    assert l4_spec.tier == FidelityTier.L4
    assert l4_spec.sample_fraction == 1.0


def test_multifidelity_scheduler_next_tier():
    scheduler = MultiFidelityScheduler()

    assert scheduler.next_tier(FidelityTier.L0) == FidelityTier.L1
    assert scheduler.next_tier(FidelityTier.L3) == FidelityTier.L4
    assert scheduler.next_tier(FidelityTier.L4) is None


def test_multifidelity_scheduler_should_promote():
    scheduler = MultiFidelityScheduler()

    # L0 default: top 50%
    assert scheduler.should_promote(FidelityTier.L0, score=0.8, rank=2, total=10)
    assert not scheduler.should_promote(FidelityTier.L0, score=0.8, rank=6, total=10)

    # L4 has no promotion criteria
    assert not scheduler.should_promote(FidelityTier.L4, score=1.0, rank=0, total=10)


def test_multifidelity_scheduler_estimate_cost():
    scheduler = MultiFidelityScheduler()

    cost_l0 = scheduler.estimate_cost(FidelityTier.L0, num_evaluations=10)
    cost_l4 = scheduler.estimate_cost(FidelityTier.L4, num_evaluations=10)

    assert cost_l0 < cost_l4
    assert cost_l0 == 1.0 * 10  # L0 multiplier is 1.0
    assert cost_l4 == 50.0 * 10  # L4 multiplier is 50.0


def test_multifidelity_scheduler_estimate_duration():
    scheduler = MultiFidelityScheduler()

    duration_l0 = scheduler.estimate_duration_ms(FidelityTier.L0)
    duration_l4 = scheduler.estimate_duration_ms(FidelityTier.L4)

    assert duration_l0 < duration_l4
    assert duration_l0 == 50
    assert duration_l4 == 5000


def test_multifidelity_scheduler_optimal_tier_for_budget():
    scheduler = MultiFidelityScheduler()

    # Very low budget: only L0
    tier = scheduler.optimal_tier_for_budget(remaining_budget=0.5)
    assert tier == FidelityTier.L0

    # Medium budget: L2
    tier = scheduler.optimal_tier_for_budget(remaining_budget=15.0)
    assert tier == FidelityTier.L2

    # High budget: L4
    tier = scheduler.optimal_tier_for_budget(remaining_budget=100.0)
    assert tier == FidelityTier.L4


def test_multifidelity_scheduler_custom_tiers():
    custom_tiers = {
        FidelityTier.L0: FidelitySpec(
            tier=FidelityTier.L0,
            sample_fraction=0.1,
            num_metrics=1,
            cost_multiplier=2.0,
        ),
    }

    scheduler = MultiFidelityScheduler(tier_specs=custom_tiers)

    spec = scheduler.get_spec(FidelityTier.L0)
    assert spec.cost_multiplier == 2.0
    assert spec.sample_fraction == 0.1


def test_multifidelity_scheduler_custom_promotion():
    custom_promotion = {
        FidelityTier.L0: PromotionCriteria(min_score=0.9),
    }

    scheduler = MultiFidelityScheduler(promotion_criteria=custom_promotion)

    # Custom criteria applied
    assert scheduler.should_promote(FidelityTier.L0, score=0.95, rank=9, total=10)
    assert not scheduler.should_promote(FidelityTier.L0, score=0.85, rank=0, total=10)


def test_fidelity_tier_ordering():
    tiers = list(FidelityTier)
    assert tiers == [
        FidelityTier.L0,
        FidelityTier.L1,
        FidelityTier.L2,
        FidelityTier.L3,
        FidelityTier.L4,
    ]


def test_promotion_criteria_no_conditions():
    criteria = PromotionCriteria()

    # No conditions means always promote
    assert criteria.should_promote(score=0.0, rank=99, total=100)
