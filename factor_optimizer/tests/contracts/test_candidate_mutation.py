"""Tests for CandidateMutation contract."""

import pytest
from datetime import datetime
from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from factor_optimizer.errors import MissingInputError


def test_candidate_mutation_creation():
    """Test basic CandidateMutation creation."""
    mutation = CandidateMutation(
        mutation_id="mut_001",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["factor_123"],
        mutation_type="parameter_tune",
        parameters={"operator_name": "ma", "parameter_name": "window", "new_value": 20},
    )

    assert mutation.mutation_id == "mut_001"
    assert mutation.mutation_spec_version == "0.1.0"
    assert mutation.parent_factor_ids == ["factor_123"]
    assert mutation.mutation_type == "parameter_tune"
    assert mutation.parameters["new_value"] == 20


def test_candidate_mutation_validation():
    """Test that required fields are validated."""
    with pytest.raises(MissingInputError, match="mutation_id is required"):
        CandidateMutation(
            mutation_id="",
            mutation_spec_version="0.1.0",
            parent_factor_ids=["f1"],
            mutation_type="test",
            parameters={},
        )

    with pytest.raises(MissingInputError, match="parent_factor_id"):
        CandidateMutation(
            mutation_id="m1",
            mutation_spec_version="0.1.0",
            parent_factor_ids=[],
            mutation_type="test",
            parameters={},
        )


def test_candidate_mutation_serialization():
    """Test to_dict and from_dict round-trip."""
    original = CandidateMutation(
        mutation_id="mut_002",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1", "f2"],
        mutation_type="linear_combination",
        parameters={"weight_a": 0.6, "weight_b": 0.4},
        mechanism_hypothesis="Combine momentum and value",
        expected_signatures=["higher_ic", "lower_turnover"],
        created_at=datetime(2026, 8, 13, 10, 0, 0),
    )

    serialized = original.to_dict()
    assert serialized["mutation_id"] == "mut_002"
    assert serialized["parent_factor_ids"] == ["f1", "f2"]
    assert serialized["parameters"]["weight_a"] == 0.6

    restored = CandidateMutation.from_dict(serialized)
    assert restored.mutation_id == original.mutation_id
    assert restored.parent_factor_ids == original.parent_factor_ids
    assert restored.parameters == original.parameters
    assert restored.mechanism_hypothesis == original.mechanism_hypothesis


def test_candidate_mutation_immutable():
    """Test that CandidateMutation is immutable (frozen dataclass)."""
    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="test",
        parameters={},
    )

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        mutation.mutation_id = "m2"
