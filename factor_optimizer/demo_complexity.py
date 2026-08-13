#!/usr/bin/env python3
"""
Demonstration script for the complexity profile system.

Shows how to use ComplexityProfile, ComplexityEstimator, ComplexityBudget,
and BudgetTracker in a realistic optimization scenario.
"""

from factor_optimizer.complexity import (
    ComplexityProfile,
    ComplexityEstimator,
    ComplexityBudget,
    BudgetTracker,
    create_default_budget,
)


def demo_basic_profile():
    """Demonstrate ComplexityProfile creation and serialization."""
    print("=" * 70)
    print("DEMO 1: ComplexityProfile")
    print("=" * 70)

    profile = ComplexityProfile(
        operator_count=8,
        max_depth=4,
        lookback_periods=60,
        stateful_operators=2,
        estimated_cost=35.0,
        memory_estimate=128.0,
    )

    print(f"Created profile:")
    print(f"  Operators: {profile.operator_count}")
    print(f"  Depth: {profile.max_depth}")
    print(f"  Lookback: {profile.lookback_periods}")
    print(f"  Cost: {profile.estimated_cost}")
    print(f"  Memory: {profile.memory_estimate} MB")

    # Serialize and deserialize
    serialized = profile.to_dict()
    restored = ComplexityProfile.from_dict(serialized)
    print(f"\n✓ Serialization round-trip successful")
    print()


def demo_budget_checking():
    """Demonstrate budget creation and checking."""
    print("=" * 70)
    print("DEMO 2: ComplexityBudget")
    print("=" * 70)

    budget = ComplexityBudget(
        max_cost=100.0,
        max_operator_count=15,
        max_lookback_periods=252,
        strict=True,
    )

    print(f"Budget constraints:")
    print(f"  Max cost: {budget.max_cost}")
    print(f"  Max operators: {budget.max_operator_count}")
    print(f"  Max lookback: {budget.max_lookback_periods}")
    print(f"  Strict mode: {budget.strict}")

    # Test with profile within budget
    profile_ok = ComplexityProfile(
        operator_count=8,
        lookback_periods=60,
        estimated_cost=50.0,
    )

    results = budget.check(profile_ok)
    print(f"\nProfile 1 check: {results}")
    print(f"  Within budget: {budget.is_within_budget(profile_ok)}")

    # Test with profile exceeding budget
    profile_exceed = ComplexityProfile(
        operator_count=20,
        lookback_periods=300,
        estimated_cost=150.0,
    )

    results = budget.check(profile_exceed)
    print(f"\nProfile 2 check: {results}")
    print(f"  Within budget: {budget.is_within_budget(profile_exceed)}")
    print()


def demo_estimator():
    """Demonstrate ComplexityEstimator with mock adapter."""
    print("=" * 70)
    print("DEMO 3: ComplexityEstimator")
    print("=" * 70)

    # Mock FE adapter
    class MockFEAdapter:
        def estimate_complexity(self, factor_def):
            # Simulate different complexities based on factor name
            if "simple" in factor_def:
                return {
                    "operator_count": 3,
                    "max_depth": 2,
                    "lookback_periods": 20,
                    "estimated_cost": 15.0,
                }
            else:
                return {
                    "operator_count": 10,
                    "max_depth": 5,
                    "lookback_periods": 100,
                    "estimated_cost": 75.0,
                }

    estimator = ComplexityEstimator(fe_adapter=MockFEAdapter())

    # Estimate complexity
    profile1 = estimator.estimate("simple_rolling_mean")
    print(f"Simple factor: cost={profile1.estimated_cost}, ops={profile1.operator_count}")

    profile2 = estimator.estimate("complex_multi_factor")
    print(f"Complex factor: cost={profile2.estimated_cost}, ops={profile2.operator_count}")

    # Estimate mutation delta
    delta = estimator.estimate_mutation_delta(
        profile1, "window_adjust", {"new_window": 100}
    )
    print(f"\nMutation delta (window 20→100):")
    print(f"  Cost: {profile1.estimated_cost:.1f} → {delta.estimated_cost:.1f}")
    print(f"  Lookback: {profile1.lookback_periods} → {delta.lookback_periods}")
    print()


def demo_budget_tracker():
    """Demonstrate BudgetTracker in optimization scenario."""
    print("=" * 70)
    print("DEMO 4: BudgetTracker (Optimization Scenario)")
    print("=" * 70)

    # Setup
    budget = create_default_budget(cost_multiplier=1.0)
    tracker = BudgetTracker(budget=budget)

    print(f"Budget: max_cost={budget.max_cost}, max_ops={budget.max_operator_count}")
    print()

    # Simulate evaluating multiple candidates
    candidates = [
        ("candidate_A", ComplexityProfile(operator_count=5, estimated_cost=30.0)),
        ("candidate_B", ComplexityProfile(operator_count=8, estimated_cost=50.0)),
        ("candidate_C", ComplexityProfile(operator_count=15, estimated_cost=90.0)),
        ("candidate_D", ComplexityProfile(operator_count=25, estimated_cost=150.0)),
        ("candidate_E", ComplexityProfile(operator_count=12, estimated_cost=70.0)),
    ]

    print("Evaluating candidates:")
    for cand_id, profile in candidates:
        is_within = tracker.record(profile, cand_id)
        status = "✓ PASS" if is_within else "✗ REJECT"
        print(f"  {cand_id}: cost={profile.estimated_cost:6.1f}, ops={profile.operator_count:2d} → {status}")

    # Show statistics
    print()
    stats = tracker.stats()
    print(f"Statistics:")
    print(f"  Total evaluated: {stats['total_evaluated']}")
    print(f"  Violations: {stats['violations_count']}")
    print(f"  Violation rate: {stats['violation_rate']:.1%}")
    print(f"  Max observed cost: {stats['max_observed_cost']}")
    print(f"  Cost utilization: {stats['budget_cost_utilization']:.1%}")
    print()


def demo_end_to_end():
    """Demonstrate complete end-to-end workflow."""
    print("=" * 70)
    print("DEMO 5: End-to-End Workflow")
    print("=" * 70)

    # Mock FE adapter
    class MockFEAdapter:
        def estimate_complexity(self, factor_def):
            import random
            random.seed(hash(factor_def) % 2**32)
            return {
                "operator_count": random.randint(3, 15),
                "estimated_cost": random.uniform(10.0, 100.0),
                "lookback_periods": random.randint(10, 200),
            }

    # Setup
    estimator = ComplexityEstimator(fe_adapter=MockFEAdapter())
    budget = ComplexityBudget(max_cost=80.0, max_operator_count=12, strict=False)
    tracker = BudgetTracker(budget=budget)

    # Simulate optimization loop
    factor_definitions = [
        "rolling_mean_20",
        "ewm_alpha_0.1",
        "rank_zscore_60",
        "complex_composite_factor",
        "simple_momentum",
    ]

    print("Optimization loop:")
    accepted = []
    rejected = []

    for factor_def in factor_definitions:
        # Estimate complexity
        profile = estimator.estimate(factor_def)

        # Check budget
        is_within = tracker.record(profile, factor_def)

        if is_within:
            print(f"  ✓ {factor_def:30s} cost={profile.estimated_cost:5.1f}")
            accepted.append(factor_def)
        else:
            print(f"  ✗ {factor_def:30s} cost={profile.estimated_cost:5.1f} [REJECTED]")
            rejected.append(factor_def)

    print()
    print(f"Results: {len(accepted)} accepted, {len(rejected)} rejected")
    print()


if __name__ == "__main__":
    demo_basic_profile()
    demo_budget_checking()
    demo_estimator()
    demo_budget_tracker()
    demo_end_to_end()

    print("=" * 70)
    print("✅ All demonstrations completed successfully!")
    print("=" * 70)
