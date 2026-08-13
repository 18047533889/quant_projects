"""
Example 04: Factor Optimization Search (Standalone)

Demonstrates the FO (Factor Optimizer) workflow:
1. Define a parent factor as the search starting point
2. Generate mutations (parameter sweeps, operator substitutions)
3. Track multi-fidelity evaluation budget (L0-L4)
4. Maintain Pareto frontier of IC vs complexity
5. Guide search with evidence feedback

This standalone version includes inline implementations for demonstration.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np


class MutationType(Enum):
    """Types of mutations that can be applied to factors."""
    PARAMETER_SWEEP = "parameter_sweep"
    OPERATOR_SUBSTITUTION = "operator_substitution"
    COMPOSITION = "composition"


@dataclass
class MutationSpec:
    """Specification for a factor mutation."""
    mutation_id: str
    mutation_type: MutationType
    parent_expression: str
    mutated_expression: str
    complexity_delta: int


@dataclass
class EvaluationBudget:
    """Multi-fidelity evaluation budget tracker."""
    l0_remaining: int  # Quick (10 dates, 100 symbols)
    l1_remaining: int  # Fast (60 dates, 500 symbols)
    l2_remaining: int  # Medium (120 dates, 1000 symbols)
    l3_remaining: int  # Full (250 dates, 3000 symbols)
    l4_remaining: int  # Extended (500 dates, 3000 symbols)

    def consume(self, level: int) -> bool:
        """Consume budget at given fidelity level."""
        if level == 0 and self.l0_remaining > 0:
            self.l0_remaining -= 1
            return True
        elif level == 1 and self.l1_remaining > 0:
            self.l1_remaining -= 1
            return True
        elif level == 2 and self.l2_remaining > 0:
            self.l2_remaining -= 1
            return True
        elif level == 3 and self.l3_remaining > 0:
            self.l3_remaining -= 1
            return True
        elif level == 4 and self.l4_remaining > 0:
            self.l4_remaining -= 1
            return True
        return False


@dataclass
class CandidateFactor:
    """A candidate factor with evaluation results."""
    expression: str
    complexity: int
    ic_mean: Optional[float] = None
    rank_ic_ir: Optional[float] = None
    evaluation_level: Optional[int] = None
    parent_expression: Optional[str] = None
    mutation_type: Optional[MutationType] = None


@dataclass
class ParetoFrontier:
    """Tracks the Pareto frontier of IC vs complexity."""
    candidates: List[CandidateFactor] = field(default_factory=list)

    def add(self, candidate: CandidateFactor) -> bool:
        """Add candidate if non-dominated."""
        if candidate.ic_mean is None:
            return False

        # Check if dominated by existing candidates
        for existing in self.candidates:
            if existing.ic_mean is None:
                continue
            # Existing dominates if better IC and lower/equal complexity
            if (existing.ic_mean >= candidate.ic_mean and
                existing.complexity <= candidate.complexity and
                (existing.ic_mean > candidate.ic_mean or existing.complexity < candidate.complexity)):
                return False

        # Remove candidates dominated by new candidate
        self.candidates = [
            c for c in self.candidates
            if c.ic_mean is None or
               not (candidate.ic_mean >= c.ic_mean and
                    candidate.complexity <= c.complexity and
                    (candidate.ic_mean > c.ic_mean or candidate.complexity < c.complexity))
        ]

        self.candidates.append(candidate)
        return True


def generate_mutations(parent_expression: str, parent_complexity: int) -> List[MutationSpec]:
    """Generate mutations from a parent factor."""
    mutations = []

    # Parameter sweep: delay window
    for delay in [10, 15, 30, 40]:
        mutations.append(MutationSpec(
            mutation_id=f"mut_delay_{delay}",
            mutation_type=MutationType.PARAMETER_SWEEP,
            parent_expression=parent_expression,
            mutated_expression=f"ts_rank(close / ts_delay(close, {delay}), 120)",
            complexity_delta=0,
        ))

    # Parameter sweep: rank window
    for rank_window in [60, 90, 150]:
        mutations.append(MutationSpec(
            mutation_id=f"mut_rank_{rank_window}",
            mutation_type=MutationType.PARAMETER_SWEEP,
            parent_expression=parent_expression,
            mutated_expression=f"ts_rank(close / ts_delay(close, 20), {rank_window})",
            complexity_delta=0,
        ))

    # Operator substitution
    mutations.append(MutationSpec(
        mutation_id="mut_op_zscore",
        mutation_type=MutationType.OPERATOR_SUBSTITUTION,
        parent_expression=parent_expression,
        mutated_expression="ts_zscore(close / ts_delay(close, 20), 120)",
        complexity_delta=0,
    ))

    # Composition
    mutations.append(MutationSpec(
        mutation_id="mut_volume_weight",
        mutation_type=MutationType.COMPOSITION,
        parent_expression=parent_expression,
        mutated_expression="ts_rank(close / ts_delay(close, 20), 120) * cs_rank(volume)",
        complexity_delta=2,
    ))

    return mutations


def simulate_qe_evaluation(expression: str, level: int, seed: int) -> Tuple[float, float]:
    """Simulate QE evaluation at given fidelity level."""
    np.random.seed(hash(expression + str(level) + str(seed)) % 2**32)

    # Base IC from expression hash
    base_ic = (hash(expression) % 100) / 1000.0 - 0.02

    # Noise decreases with fidelity level
    noise_levels = {0: 0.03, 1: 0.02, 2: 0.01, 3: 0.005, 4: 0.002}
    noise = np.random.randn() * noise_levels[level]

    ic_mean = base_ic + noise
    rank_ic_ir = ic_mean / 0.15 * (1 + np.random.randn() * 0.1)

    return ic_mean, rank_ic_ir


def main():
    """Run factor optimization search example."""

    print("=" * 70)
    print("EXAMPLE 04: Factor Optimization Search")
    print("=" * 70)
    print()

    # Step 1: Define parent factor
    print("STEP 1: Define Parent Factor")
    print("-" * 70)

    parent_expression = "ts_rank(close / ts_delay(close, 20), 120)"
    parent_complexity = 3

    print(f"Parent expression: {parent_expression}")
    print(f"Parent complexity: {parent_complexity}")
    print()

    # Evaluate parent
    parent_ic, parent_ir = simulate_qe_evaluation(parent_expression, level=3, seed=0)
    print(f"Parent evaluation (L3 - full fidelity):")
    print(f"  IC Mean: {parent_ic:.4f}")
    print(f"  Rank IC IR: {parent_ir:.2f}")
    print()

    # Step 2: Initialize search
    print("STEP 2: Initialize Search State")
    print("-" * 70)
    print()

    budget = EvaluationBudget(
        l0_remaining=100,
        l1_remaining=50,
        l2_remaining=20,
        l3_remaining=10,
        l4_remaining=3,
    )

    print("Evaluation budget:")
    print(f"  L0 (quick):    {budget.l0_remaining} evaluations")
    print(f"  L1 (fast):     {budget.l1_remaining} evaluations")
    print(f"  L2 (medium):   {budget.l2_remaining} evaluations")
    print(f"  L3 (full):     {budget.l3_remaining} evaluations")
    print(f"  L4 (extended): {budget.l4_remaining} evaluations")
    print()

    frontier = ParetoFrontier()
    parent_candidate = CandidateFactor(
        expression=parent_expression,
        complexity=parent_complexity,
        ic_mean=parent_ic,
        rank_ic_ir=parent_ir,
        evaluation_level=3,
    )
    frontier.add(parent_candidate)
    print("Pareto frontier initialized with parent.")
    print()

    # Step 3: Generate and evaluate mutations
    print("STEP 3: Generate and Evaluate Mutations")
    print("-" * 70)
    print()

    mutations = generate_mutations(parent_expression, parent_complexity)
    print(f"Generated {len(mutations)} mutations:")
    for mut in mutations:
        print(f"  - {mut.mutation_id} ({mut.mutation_type.value})")
        print(f"    {mut.mutated_expression}")
    print()

    print("Evaluating with progressive fidelity...")
    print()

    candidates: List[CandidateFactor] = []

    for mut in mutations:
        mut_complexity = parent_complexity + mut.complexity_delta

        # L0 quick check
        if not budget.consume(0):
            print(f"  ⚠ Budget exhausted at L0, skipping {mut.mutation_id}")
            continue

        ic_l0, ir_l0 = simulate_qe_evaluation(mut.mutated_expression, level=0, seed=1)
        print(f"  {mut.mutation_id} @ L0: IC={ic_l0:.4f}, IR={ir_l0:.2f}")

        # Gate: only promote if promising
        if ic_l0 < parent_ic - 0.02:
            print(f"    ✗ Failed L0 gate (IC too low)")
            continue

        # L2 medium check
        if not budget.consume(2):
            print(f"    ⚠ Budget exhausted at L2")
            continue

        ic_l2, ir_l2 = simulate_qe_evaluation(mut.mutated_expression, level=2, seed=2)
        print(f"    → L2: IC={ic_l2:.4f}, IR={ir_l2:.2f}")

        if ic_l2 < parent_ic - 0.01:
            print(f"    ✗ Failed L2 gate")
            continue

        # L3 full evaluation
        if not budget.consume(3):
            print(f"    ⚠ Budget exhausted at L3")
            candidate = CandidateFactor(
                expression=mut.mutated_expression,
                complexity=mut_complexity,
                ic_mean=ic_l2,
                rank_ic_ir=ir_l2,
                evaluation_level=2,
                parent_expression=parent_expression,
                mutation_type=mut.mutation_type,
            )
            candidates.append(candidate)
            continue

        ic_l3, ir_l3 = simulate_qe_evaluation(mut.mutated_expression, level=3, seed=3)
        print(f"    → L3: IC={ic_l3:.4f}, IR={ir_l3:.2f}")

        candidate = CandidateFactor(
            expression=mut.mutated_expression,
            complexity=mut_complexity,
            ic_mean=ic_l3,
            rank_ic_ir=ir_l3,
            evaluation_level=3,
            parent_expression=parent_expression,
            mutation_type=mut.mutation_type,
        )
        candidates.append(candidate)

        if frontier.add(candidate):
            print(f"    ✓ Added to Pareto frontier")
        else:
            print(f"    - Not on frontier (dominated)")

    print()
    print(f"Evaluated {len(candidates)} candidates")
    print()

    # Step 4: Examine Pareto frontier
    print("STEP 4: Examine Pareto Frontier")
    print("-" * 70)
    print()

    sorted_frontier = sorted(frontier.candidates, key=lambda c: c.complexity)

    print(f"Pareto frontier contains {len(sorted_frontier)} factors:")
    print()

    for candidate in sorted_frontier:
        print(f"Complexity {candidate.complexity}:")
        print(f"  Expression: {candidate.expression}")
        print(f"  IC Mean: {candidate.ic_mean:.4f}")
        print(f"  Rank IC IR: {candidate.rank_ic_ir:.2f}")
        print(f"  Eval Level: L{candidate.evaluation_level}")
        if candidate.parent_expression:
            print(f"  Mutation: {candidate.mutation_type.value if candidate.mutation_type else 'N/A'}")
        else:
            print(f"  (Parent factor)")
        print()

    # Step 5: Budget summary
    print("STEP 5: Budget Summary")
    print("-" * 70)
    print()

    print("Budget consumed:")
    print(f"  L0: {100 - budget.l0_remaining}/100")
    print(f"  L1: {50 - budget.l1_remaining}/50")
    print(f"  L2: {20 - budget.l2_remaining}/20")
    print(f"  L3: {10 - budget.l3_remaining}/10")
    print(f"  L4: {3 - budget.l4_remaining}/3")
    print()

    # Step 6: Understanding the workflow
    print("STEP 6: Understanding the Optimization Workflow")
    print("-" * 70)
    print()
    print("The factor optimization workflow:")
    print()
    print("1. Mutation Generation (FO)")
    print("   - Parameter sweeps: vary window lengths")
    print("   - Operator substitution: swap similar operators")
    print("   - Composition: combine with other factors")
    print("   - Grammar ensures syntactic validity")
    print()
    print("2. Progressive Fidelity Evaluation (FO + QE)")
    print("   - L0: Quick filter (~1s)")
    print("   - L2: Medium check (~15s)")
    print("   - L3: Full evaluation (~60s)")
    print("   - Early stopping saves budget")
    print()
    print("3. Pareto Frontier Tracking (FO)")
    print("   - Multi-objective: maximize IC, minimize complexity")
    print("   - Non-dominated solutions preserved")
    print("   - Guides search toward high IC/complexity ratio")
    print()
    print("4. Integration Points")
    print("   - FA: Register promising candidates as assets")
    print("   - QE: Provides evaluation at each fidelity level")
    print("   - FP: Preprocessing strategy informed by complexity")
    print()

    print("=" * 70)
    print("Example 04 Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
