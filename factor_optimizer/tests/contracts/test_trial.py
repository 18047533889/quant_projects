"""Tests for Trial contract."""

import pytest
from datetime import datetime
from factor_optimizer.contracts.trial import Trial, TrialStatus


def test_trial_creation():
    """Test basic Trial creation."""
    trial = Trial(
        trial_id="trial_001",
        mutation_id="mut_001",
        status=TrialStatus.PROPOSED,
        parent_factor_ids=["f1"],
    )

    assert trial.trial_id == "trial_001"
    assert trial.mutation_id == "mut_001"
    assert trial.status == TrialStatus.PROPOSED
    assert trial.parent_factor_ids == ["f1"]


def test_trial_status_update():
    """Test updating trial status."""
    trial = Trial(
        trial_id="t1",
        mutation_id="m1",
        status=TrialStatus.PROPOSED,
    )

    trial.update_status(TrialStatus.LEGAL, legality_check={"is_legal": True})
    assert trial.status == TrialStatus.LEGAL
    assert trial.legality_check["is_legal"] is True


def test_trial_terminal_status():
    """Test terminal status detection."""
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)
    assert not trial.is_terminal()

    trial.update_status(TrialStatus.LEGAL)
    assert not trial.is_terminal()

    trial.update_status(TrialStatus.EVALUATED)
    assert trial.is_terminal()


def test_trial_success():
    """Test successful trial detection."""
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.LEGAL)
    assert not trial.is_successful()

    trial.update_status(TrialStatus.EVALUATED, evaluation_ref="eval_001")
    assert trial.is_successful()
    assert trial.evaluation_ref == "eval_001"


def test_trial_failure():
    """Test failed trial."""
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.VALIDATING)

    trial.update_status(TrialStatus.FAILED, failure_reason="Compilation error")
    assert trial.is_terminal()
    assert not trial.is_successful()
    assert trial.failure_reason == "Compilation error"


def test_trial_serialization():
    """Test Trial serialization round-trip."""
    original = Trial(
        trial_id="t1",
        mutation_id="m1",
        status=TrialStatus.EVALUATED,
        parent_factor_ids=["f1", "f2"],
        legality_check={"is_legal": True},
        evaluation_ref="eval_001",
        metadata={"note": "test"},
    )

    serialized = original.to_dict()
    assert serialized["trial_id"] == "t1"
    assert serialized["status"] == "evaluated"
    assert serialized["evaluation_ref"] == "eval_001"

    restored = Trial.from_dict(serialized)
    assert restored.trial_id == original.trial_id
    assert restored.status == original.status
    assert restored.evaluation_ref == original.evaluation_ref


def test_trial_metadata_update():
    """Test metadata accumulation."""
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.PROPOSED)

    trial.update_status(TrialStatus.LEGAL, metadata={"step": 1})
    trial.update_status(TrialStatus.EVALUATING, metadata={"step": 2})

    assert trial.metadata["step"] == 2


def test_trial_status_enum():
    """Test TrialStatus enum values."""
    assert TrialStatus.PROPOSED.value == "proposed"
    assert TrialStatus.ILLEGAL.value == "illegal"
    assert TrialStatus.EVALUATED.value == "evaluated"
    assert TrialStatus.DUPLICATE.value == "duplicate"


def test_trial_duplicate_status():
    """Test duplicate detection."""
    trial = Trial(trial_id="t1", mutation_id="m1", status=TrialStatus.DUPLICATE)
    assert trial.is_terminal()
    assert not trial.is_successful()
