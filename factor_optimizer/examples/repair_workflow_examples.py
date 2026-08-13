"""Example usage of repair strategies and admission decisions."""

from factor_optimizer.policy.repair import (
    DiagnosisKind,
    DiagnosisRecord,
    RepairMapper,
)
from factor_optimizer.policy.decisions import (
    AdmissionCriteria,
    AdmissionPolicy,
)


def example_low_ic_repair():
    """Example: Factor with low IC gets interaction term added."""
    print("=" * 60)
    print("Example 1: Low IC Factor Repair")
    print("=" * 60)

    diagnosis = DiagnosisRecord(
        trial_id="trial_001",
        diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
        severity=0.6,
        evidence={
            "ic": 0.03,
            "ic_std": 0.05,
            "lookback_periods": 20,
        },
    )

    print(f"\nDiagnosis: {diagnosis.diagnosis_kind.value}")
    print(f"Severity: {diagnosis.severity}")
    print(f"Evidence: IC={diagnosis.evidence['ic']:.3f}")

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=3)

    print(f"\n{len(proposals)} repair proposals generated:")
    for i, proposal in enumerate(proposals, 1):
        print(f"\n  Proposal {i}:")
        print(f"    Strategy: {proposal.strategy.value}")
        print(f"    Mutation: {proposal.mutation_type}")
        print(f"    Parameters: {proposal.parameters}")
        print(f"    Confidence: {proposal.confidence:.2f}")
        print(f"    Expected: {proposal.expected_improvement}")

    mutation_spec = mapper.generate_mutation_spec(proposals[0])
    print(f"\nGenerated MutationSpec:")
    print(f"  Type: {mutation_spec['mutation_type']}")
    print(f"  Params: {mutation_spec['parameters']}")
    print(f"  Source: {mutation_spec['provenance']['source']}")


def example_high_turnover_repair():
    """Example: High turnover factor gets increased horizon."""
    print("\n\n" + "=" * 60)
    print("Example 2: High Turnover Factor Repair")
    print("=" * 60)

    diagnosis = DiagnosisRecord(
        trial_id="trial_002",
        diagnosis_kind=DiagnosisKind.HIGH_TURNOVER,
        severity=0.8,
        evidence={
            "turnover_rate": 0.85,
            "prediction_horizon": 1,
            "holding_period": 1,
        },
    )

    print(f"\nDiagnosis: {diagnosis.diagnosis_kind.value}")
    print(f"Severity: {diagnosis.severity}")
    print(f"Evidence: Turnover={diagnosis.evidence['turnover_rate']:.2f}")

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=2)

    print(f"\n{len(proposals)} repair proposals generated:")
    for i, proposal in enumerate(proposals, 1):
        print(f"\n  Proposal {i}:")
        print(f"    Strategy: {proposal.strategy.value}")
        print(f"    Parameters: {proposal.parameters}")
        print(f"    Confidence: {proposal.confidence:.2f}")


def example_admission_workflow():
    """Example: Complete admission decision workflow."""
    print("\n\n" + "=" * 60)
    print("Example 3: Admission Decision Workflow")
    print("=" * 60)

    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.2,
        max_lookback_periods=200,
        min_parent_quality=0.5,
    )

    policy = AdmissionPolicy(criteria=criteria)

    print("\nAdmission Criteria:")
    print(f"  Max Complexity: {criteria.max_complexity_cost}")
    print(f"  Min Expected Value: {criteria.min_expected_value}")
    print(f"  Max Lookback: {criteria.max_lookback_periods}")

    test_cases = [
        {
            "name": "Good Mutation",
            "mutation_id": "mut_001",
            "trial_id": "trial_001",
            "complexity_cost": 75.0,
            "expected_value": 0.5,
            "parent_quality": 0.7,
        },
        {
            "name": "Too Complex",
            "mutation_id": "mut_002",
            "trial_id": "trial_002",
            "complexity_cost": 150.0,
            "expected_value": 0.6,
            "parent_quality": 0.8,
        },
        {
            "name": "Low Expected Value",
            "mutation_id": "mut_003",
            "trial_id": "trial_003",
            "complexity_cost": 50.0,
            "expected_value": 0.1,
            "parent_quality": 0.6,
        },
    ]

    for test in test_cases:
        name = test.pop("name")
        decision = policy.decide(**test)

        print(f"\n{name}:")
        print(f"  Verdict: {decision.verdict.value}")
        print(f"  Approved: {decision.is_approved()}")
        if decision.rejection_reasons:
            reasons = ", ".join(r.value for r in decision.rejection_reasons)
            print(f"  Rejection Reasons: {reasons}")

    stats = policy.get_admission_stats()
    print(f"\nAdmission Statistics:")
    print(f"  Total Decisions: {stats['total']}")
    print(f"  Admitted: {stats['admitted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Admission Rate: {stats['admission_rate']:.1%}")


def example_complete_repair_workflow():
    """Example: Complete diagnosis → repair → admission workflow."""
    print("\n\n" + "=" * 60)
    print("Example 4: Complete Repair Workflow")
    print("=" * 60)

    diagnosis = DiagnosisRecord(
        trial_id="trial_004",
        diagnosis_kind=DiagnosisKind.NUMERICAL_INSTABILITY,
        severity=0.7,
        evidence={
            "nan_count": 150,
            "inf_count": 20,
            "total_observations": 1000,
        },
    )

    print(f"\nStep 1: Diagnosis")
    print(f"  Kind: {diagnosis.diagnosis_kind.value}")
    print(f"  Severity: {diagnosis.severity}")
    print(f"  NaN ratio: {diagnosis.evidence['nan_count'] / diagnosis.evidence['total_observations']:.1%}")

    mapper = RepairMapper()
    proposals = mapper.propose_repairs(diagnosis, max_proposals=2)

    print(f"\nStep 2: Generate Repairs")
    print(f"  Generated {len(proposals)} proposals")
    best_proposal = proposals[0]
    print(f"  Best Strategy: {best_proposal.strategy.value}")
    print(f"  Confidence: {best_proposal.confidence:.2f}")

    ranked = mapper.rank_repairs(proposals)
    print(f"\nStep 3: Rank by Priority")
    for i, p in enumerate(ranked, 1):
        print(f"  Rank {i}: {p.strategy.value} (confidence={p.confidence:.2f})")

    mutation_spec = mapper.generate_mutation_spec(ranked[0])
    print(f"\nStep 4: Generate MutationSpec")
    print(f"  Mutation Type: {mutation_spec['mutation_type']}")
    print(f"  Parameters: {mutation_spec['parameters']}")

    criteria = AdmissionCriteria(
        max_complexity_cost=100.0,
        min_expected_value=0.15,
    )
    policy = AdmissionPolicy(criteria=criteria)

    decision = policy.decide(
        mutation_id="mut_004",
        trial_id=diagnosis.trial_id,
        complexity_cost=80.0,
        expected_value=0.4,
    )

    print(f"\nStep 5: Admission Decision")
    print(f"  Verdict: {decision.verdict.value}")
    print(f"  Approved: {decision.is_approved()}")

    if decision.is_approved():
        print("\n✓ Mutation admitted for evaluation")
    else:
        print(f"\n✗ Mutation rejected: {decision.primary_rejection_reason().value}")


def example_multiple_diagnoses_ranking():
    """Example: Ranking repairs from multiple diagnoses."""
    print("\n\n" + "=" * 60)
    print("Example 5: Multi-Diagnosis Repair Ranking")
    print("=" * 60)

    diagnoses = [
        DiagnosisRecord(
            trial_id="t1",
            diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
            severity=0.8,
            evidence={"operator_count": 50},
        ),
        DiagnosisRecord(
            trial_id="t2",
            diagnosis_kind=DiagnosisKind.TIMING_VIOLATION,
            severity=0.95,
            evidence={"lookahead_days": 1},
        ),
        DiagnosisRecord(
            trial_id="t3",
            diagnosis_kind=DiagnosisKind.LOW_SIGNAL,
            severity=0.5,
            evidence={"ic": 0.02},
        ),
    ]

    mapper = RepairMapper()
    all_proposals = []

    print("\nDiagnoses:")
    for diag in diagnoses:
        print(f"  {diag.trial_id}: {diag.diagnosis_kind.value} (severity={diag.severity:.2f})")
        proposals = mapper.propose_repairs(diag, max_proposals=1)
        all_proposals.extend(proposals)

    print(f"\nRanking {len(all_proposals)} repairs by priority:")
    ranked = mapper.rank_repairs(all_proposals)

    for i, proposal in enumerate(ranked, 1):
        print(f"\n  Rank {i}:")
        print(f"    Trial: {proposal.diagnosis_record.trial_id}")
        print(f"    Diagnosis: {proposal.diagnosis_record.diagnosis_kind.value}")
        print(f"    Strategy: {proposal.strategy.value}")
        print(f"    Confidence: {proposal.confidence:.2f}")
        print(f"    Severity: {proposal.diagnosis_record.severity:.2f}")


if __name__ == "__main__":
    example_low_ic_repair()
    example_high_turnover_repair()
    example_admission_workflow()
    example_complete_repair_workflow()
    example_multiple_diagnoses_ranking()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
