"""Tests for MutationValidator."""

import pytest
from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from factor_optimizer.grammar.validation import MutationValidator, ValidationResult
from factor_optimizer.grammar.registry import MutationRegistry
from factor_optimizer.grammar.mutation_spec import (
    MutationSpec,
    ParameterSpec,
    ParameterKind,
    ParameterRole,
)


def test_validation_result_creation():
    """Test ValidationResult creation."""
    result = ValidationResult(True)
    assert result.is_valid
    assert len(result.errors) == 0
    assert len(result.warnings) == 0

    result = ValidationResult(False, errors=["error1", "error2"])
    assert not result.is_valid
    assert len(result.errors) == 2


def test_validation_result_bool():
    """Test ValidationResult boolean conversion."""
    assert ValidationResult(True)
    assert not ValidationResult(False)


def test_validator_unknown_mutation_type():
    """Test validation fails for unknown mutation type."""
    registry = MutationRegistry()
    validator = MutationValidator(registry)

    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="nonexistent",
        parameters={},
    )

    result = validator.validate(mutation)
    assert not result.is_valid
    assert any("Unknown mutation type" in e for e in result.errors)


def test_validator_valid_mutation():
    """Test validation passes for valid mutation."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test_op",
        version="0.1.0",
        description="Test",
        parameters=[
            ParameterSpec(
                name="window",
                kind=ParameterKind.INTEGER,
                role=ParameterRole.WINDOW,
                min_value=1,
                max_value=100,
            )
        ],
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="test_op",
        parameters={"window": 20},
    )

    result = validator.validate(mutation)
    assert result.is_valid
    assert len(result.errors) == 0


def test_validator_invalid_parameters():
    """Test validation fails for invalid parameters."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test_op",
        version="0.1.0",
        description="Test",
        parameters=[
            ParameterSpec(
                name="window",
                kind=ParameterKind.INTEGER,
                role=ParameterRole.WINDOW,
                min_value=1,
                max_value=100,
            )
        ],
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    # Out of range
    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="test_op",
        parameters={"window": 200},
    )

    result = validator.validate(mutation)
    assert not result.is_valid
    assert any("must be <=" in e for e in result.errors)


def test_validator_parent_count_single():
    """Test parent count validation for single-parent mutation."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="single_parent",
        version="0.1.0",
        description="Test",
        requires_single_parent=True,
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    # Wrong number of parents
    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1", "f2"],
        mutation_type="single_parent",
        parameters={},
    )

    result = validator.validate(mutation)
    assert not result.is_valid
    assert any("requires exactly 1 parent" in e for e in result.errors)


def test_validator_parent_count_multiple():
    """Test parent count validation for multi-parent mutation."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="multi_parent",
        version="0.1.0",
        description="Test",
        requires_single_parent=False,
        requires_multiple_parents=True,
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    # Only one parent
    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="multi_parent",
        parameters={},
    )

    result = validator.validate(mutation)
    assert not result.is_valid
    assert any("requires 2+ parents" in e for e in result.errors)


def test_validator_version_mismatch_warning():
    """Test version mismatch produces warning."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.2.0",
        description="Test",
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="test",
        parameters={},
    )

    result = validator.validate(mutation)
    assert len(result.warnings) > 0
    assert any("version mismatch" in w for w in result.warnings)


def test_validator_unknown_parameter():
    """Test validation catches unknown parameters."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
        parameters=[
            ParameterSpec(name="known", kind=ParameterKind.INTEGER, role=ParameterRole.SCALAR)
        ],
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    mutation = CandidateMutation(
        mutation_id="m1",
        mutation_spec_version="0.1.0",
        parent_factor_ids=["f1"],
        mutation_type="test",
        parameters={"known": 1, "unknown": 2},
    )

    result = validator.validate(mutation)
    assert not result.is_valid
    assert any("Unknown parameter" in e for e in result.errors)


def test_validator_batch():
    """Test batch validation."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    mutations = [
        CandidateMutation(
            mutation_id="m1",
            mutation_spec_version="0.1.0",
            parent_factor_ids=["f1"],
            mutation_type="test",
            parameters={},
        ),
        CandidateMutation(
            mutation_id="m2",
            mutation_spec_version="0.1.0",
            parent_factor_ids=["f2"],
            mutation_type="nonexistent",
            parameters={},
        ),
    ]

    results = validator.validate_batch(mutations)
    assert len(results) == 2
    assert results["m1"].is_valid
    assert not results["m2"].is_valid


def test_validator_quick_check():
    """Test quick parameter validation."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
        parameters=[
            ParameterSpec(
                name="window",
                kind=ParameterKind.INTEGER,
                role=ParameterRole.WINDOW,
                min_value=1,
            )
        ],
    )
    registry.register(spec)
    validator = MutationValidator(registry)

    is_valid, errors = validator.quick_check("test", {"window": 20})
    assert is_valid

    is_valid, errors = validator.quick_check("test", {})
    assert not is_valid
    assert any("window" in e for e in errors)
