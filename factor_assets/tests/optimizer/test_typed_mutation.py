"""Tests for typed mutation."""

import pytest
from datetime import datetime
from factor_assets.optimizer.typed_mutation import (
    TypedMutation,
    MutationContext,
    MutationResult,
    MutationStatus,
)


def test_typed_mutation_creation():
    """Test basic mutation creation."""
    mutation = TypedMutation(
        mutation_id="mut_001",
        parent_asset_id="asset_123",
        mutation_type="adjust_window",
        parameters={"window": 20, "direction": "increase"},
    )

    assert mutation.mutation_id == "mut_001"
    assert mutation.parent_asset_id == "asset_123"
    assert mutation.mutation_type == "adjust_window"
    assert mutation.parameters["window"] == 20
    assert mutation.version == "0.1.0"


def test_typed_mutation_immutable():
    """Test that mutation is immutable."""
    mutation = TypedMutation(
        mutation_id="mut_001",
        parent_asset_id="asset_123",
        mutation_type="adjust_window",
        parameters={"window": 20},
    )

    with pytest.raises(AttributeError):
        mutation.mutation_id = "mut_002"


def test_typed_mutation_with_context():
    """Test mutation with context."""
    context = {"campaign_id": "camp_001", "trial_id": "trial_002"}
    mutation = TypedMutation(
        mutation_id="mut_001",
        parent_asset_id="asset_123",
        mutation_type="adjust_window",
        parameters={"window": 20},
        context=context,
    )

    assert mutation.context["campaign_id"] == "camp_001"
    assert mutation.context["trial_id"] == "trial_002"


def test_mutation_context():
    """Test mutation context."""
    context = MutationContext(
        campaign_id="camp_001",
        trial_id="trial_003",
        budget_remaining=500,
    )

    assert context.campaign_id == "camp_001"
    assert context.trial_id == "trial_003"
    assert context.budget_remaining == 500


def test_mutation_result_success():
    """Test successful mutation result."""
    result = MutationResult(
        mutation_id="mut_001",
        status=MutationStatus.EXECUTED,
        child_asset_id="asset_456",
        error_message="Mutation applied successfully",
    )

    assert result.status == MutationStatus.EXECUTED
    assert result.child_asset_id == "asset_456"
    assert result.is_success()


def test_mutation_result_failure():
    """Test failed mutation result."""
    result = MutationResult(
        mutation_id="mut_001",
        status=MutationStatus.FAILED,
        child_asset_id=None,
        error_message="Invalid parameters: ValueError: window must be positive",
    )

    assert result.status == MutationStatus.FAILED
    assert result.child_asset_id is None
    assert not result.is_success()
    assert "ValueError" in result.error_message


def test_mutation_result_skipped():
    """Test skipped mutation result."""
    result = MutationResult(
        mutation_id="mut_001",
        status=MutationStatus.CANCELLED,
        child_asset_id=None,
        error_message="Budget exhausted",
    )

    assert result.status == MutationStatus.CANCELLED
    assert not result.is_success()
