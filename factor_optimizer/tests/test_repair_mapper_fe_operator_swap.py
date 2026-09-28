"""Legacy repair proposals must use real FE operator fields, not placeholders."""

from types import SimpleNamespace

from factor_optimizer.adapters.factor_engine import create_fe_adapter
from factor_optimizer.policy.repair import (
    DiagnosisKind,
    DiagnosisRecord,
    RepairMapper,
    RepairStrategy,
)


def _diagnosis(evidence):
    return DiagnosisRecord(
        trial_id="trial-1", diagnosis_kind=DiagnosisKind.HIGH_COMPLEXITY,
        evidence=evidence,
    )


def test_operator_swap_is_not_proposed_without_a_real_replacement():
    mapper = RepairMapper()
    for evidence in ({}, {"operator": "ts_mean"},
                     {"operator": "ts_mean", "replacement_operator": "ts_mean"}):
        proposals = mapper.propose_repairs(_diagnosis(evidence))
        assert all(p.strategy != RepairStrategy.OPERATOR_SWAP for p in proposals)


def test_operator_swap_uses_fe_contract_and_canonical_resolution():
    mapper = RepairMapper()
    diagnosis = _diagnosis({
        "operator": "Mean", "replacement_operator": "ts_median",
    })
    proposals = mapper.propose_repairs(diagnosis)
    swaps = [p for p in proposals if p.strategy == RepairStrategy.OPERATOR_SWAP]
    assert len(swaps) == 1
    assert swaps[0].parameters == {
        "target_operator": "Mean", "replacement_operator": "ts_median",
    }
    adapter = create_fe_adapter()
    validation = adapter.validate_mutation(
        SimpleNamespace(parameters=swaps[0].parameters), None,
    )
    assert validation["is_legal"] is True
    assert validation["metadata"]["canonical_operators"] == {
        "target_operator": "ts_mean", "replacement_operator": "ts_median",
    }
