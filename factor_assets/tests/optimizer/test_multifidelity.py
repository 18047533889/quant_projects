"""Tests for multi-fidelity policy."""

import pytest
from factor_assets.optimizer.multifidelity import (
    FidelityTier,
    FidelitySpec,
    PromotionCriteria,
    MultiFidelityPolicy,
    DEFAULT_FIDELITY_SPECS,
)


def test_fidelity_spec_creation():
    """Test fidelity specification creation."""
    spec = FidelitySpec(
        tier=FidelityTier.L0,
        sample_fraction=0.05,
        universe_fraction=0.5,
        metrics=["rank_ic"],
        cost_multiplier=1.0,
        typical_duration_s=5,
        description="Quick screening",
    )

    assert spec.tier == FidelityTier.L0
    assert spec.sample_fraction == 0.05
    assert "rank_ic" in spec.metrics


def test_promotion_criteria_creation():
    """Test promotion criteria creation."""
    criteria = PromotionCriteria(
        min_metric_value=0.02,
        metric_name="rank_ic",
        max_candidates_per_tier=50,
    )

    assert criteria.min_metric_value == 0.02
    assert criteria.metric_name == "rank_ic"


def test_default_fidelity_specs():
    """Test default fidelity specifications."""
    assert FidelityTier.L0 in DEFAULT_FIDELITY_SPECS
    assert FidelityTier.L4 in DEFAULT_FIDELITY_SPECS

    l0_spec = DEFAULT_FIDELITY_SPECS[FidelityTier.L0]
    assert l0_spec.sample_fraction == 0.05

    l4_spec = DEFAULT_FIDELITY_SPECS[FidelityTier.L4]
    assert l4_spec.sample_fraction == 1.0
    assert "robustness" in l4_spec.metrics


def test_multifidelity_policy_creation():
    """Test policy creation with defaults."""
    policy = MultiFidelityPolicy()

    assert policy.fidelity_specs is not None
    assert policy.promotion_criteria is not None


def test_multifidelity_policy_get_spec():
    """Test getting tier specification."""
    policy = MultiFidelityPolicy()

    spec = policy.get_spec(FidelityTier.L1)
    assert spec.tier == FidelityTier.L1
    assert spec.sample_fraction == 0.15


def test_multifidelity_policy_should_promote_success():
    """Test successful promotion."""
    policy = MultiFidelityPolicy()

    metrics = {"rank_ic": 0.03}

    # L0 → L1 requires rank_ic >= 0.01
    assert policy.should_promote(FidelityTier.L0, metrics)


def test_multifidelity_policy_should_promote_failure():
    """Test failed promotion."""
    policy = MultiFidelityPolicy()

    metrics = {"rank_ic": 0.005}

    # L0 → L1 requires rank_ic >= 0.01
    assert not policy.should_promote(FidelityTier.L0, metrics)


def test_multifidelity_policy_l3_to_l4_promotion():
    """Test L3 → L4 promotion requires plateau analysis."""
    policy = MultiFidelityPolicy()

    metrics = {"rank_ic": 0.05}

    # Without plateau analysis, should not promote
    assert not policy.should_promote(FidelityTier.L3, metrics)

    # With sufficient plateau analysis, should promote
    plateau_analysis = {
        "plateau_stability": 0.8,
        "neighbor_survival_rate": 0.7,
    }
    assert policy.should_promote(FidelityTier.L3, metrics, plateau_analysis)


def test_multifidelity_policy_l3_to_l4_insufficient_stability():
    """Test L3 → L4 fails with insufficient stability."""
    policy = MultiFidelityPolicy()

    metrics = {"rank_ic": 0.05}
    plateau_analysis = {
        "plateau_stability": 0.5,  # Below 0.7 threshold
        "neighbor_survival_rate": 0.7,
    }

    assert not policy.should_promote(FidelityTier.L3, metrics, plateau_analysis)


def test_multifidelity_policy_cannot_promote_from_l4():
    """Test cannot promote from L4."""
    policy = MultiFidelityPolicy()

    metrics = {"rank_ic": 1.0}

    assert not policy.should_promote(FidelityTier.L4, metrics)


def test_multifidelity_policy_get_next_tier():
    """Test getting next tier."""
    policy = MultiFidelityPolicy()

    assert policy.get_next_tier(FidelityTier.L0) == FidelityTier.L1
    assert policy.get_next_tier(FidelityTier.L3) == FidelityTier.L4
    assert policy.get_next_tier(FidelityTier.L4) is None


def test_multifidelity_policy_estimate_cost():
    """Test cost estimation."""
    policy = MultiFidelityPolicy()

    cost_l0 = policy.estimate_cost(FidelityTier.L0, num_candidates=10)
    cost_l4 = policy.estimate_cost(FidelityTier.L4, num_candidates=10)

    # L4 should be more expensive than L0
    assert cost_l4 > cost_l0


def test_multifidelity_policy_select_tier_for_budget():
    """Test tier selection for budget."""
    policy = MultiFidelityPolicy()

    # Small budget should select L0
    tier = policy.select_tier_for_budget(budget=5.0, num_candidates=10)
    assert tier == FidelityTier.L0

    # Large budget should select higher tier
    tier = policy.select_tier_for_budget(budget=1000.0, num_candidates=10)
    assert tier in {FidelityTier.L3, FidelityTier.L4}


def test_multifidelity_policy_serialization():
    """Test spec serialization."""
    spec = DEFAULT_FIDELITY_SPECS[FidelityTier.L0]
    data = spec.to_dict()

    assert data["tier"] == "L0"
    assert data["sample_fraction"] == 0.05


def test_promotion_criteria_serialization():
    """Test criteria serialization."""
    criteria = PromotionCriteria(
        min_metric_value=0.02,
        metric_name="rank_ic",
    )

    data = criteria.to_dict()
    assert data["min_metric_value"] == 0.02
    assert data["metric_name"] == "rank_ic"
