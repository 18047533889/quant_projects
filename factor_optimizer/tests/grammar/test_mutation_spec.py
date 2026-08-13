"""Tests for MutationSpec and parameter validation."""

import pytest
from factor_optimizer.grammar.mutation_spec import (
    MutationSpec,
    ParameterSpec,
    ParameterKind,
    ParameterRole,
)


def test_parameter_spec_creation():
    """Test basic ParameterSpec creation."""
    param = ParameterSpec(
        name="window",
        kind=ParameterKind.INTEGER,
        role=ParameterRole.WINDOW,
        min_value=1,
        max_value=252,
    )

    assert param.name == "window"
    assert param.kind == ParameterKind.INTEGER
    assert param.role == ParameterRole.WINDOW
    assert param.min_value == 1
    assert param.max_value == 252


def test_parameter_spec_enum_validation():
    """Test enum parameter requires allowed_values."""
    with pytest.raises(ValueError, match="ENUM kind requires allowed_values"):
        ParameterSpec(
            name="method",
            kind=ParameterKind.ENUM,
            role=ParameterRole.CATEGORICAL,
        )

    # Should work with allowed_values
    param = ParameterSpec(
        name="method",
        kind=ParameterKind.ENUM,
        role=ParameterRole.CATEGORICAL,
        allowed_values=["sma", "ema", "wma"],
    )
    assert param.allowed_values == ["sma", "ema", "wma"]


def test_parameter_value_validation_integer():
    """Test integer parameter validation."""
    param = ParameterSpec(
        name="window",
        kind=ParameterKind.INTEGER,
        role=ParameterRole.WINDOW,
        min_value=1,
        max_value=100,
    )

    # Valid
    is_valid, error = param.validate_value(50)
    assert is_valid
    assert error is None

    # Too small
    is_valid, error = param.validate_value(0)
    assert not is_valid
    assert "must be >=" in error

    # Too large
    is_valid, error = param.validate_value(101)
    assert not is_valid
    assert "must be <=" in error

    # Wrong type
    is_valid, error = param.validate_value("50")
    assert not is_valid
    assert "must be an integer" in error


def test_parameter_value_validation_float():
    """Test float parameter validation."""
    param = ParameterSpec(
        name="decay",
        kind=ParameterKind.FLOAT,
        role=ParameterRole.DECAY,
        min_value=0.0,
        max_value=1.0,
    )

    is_valid, error = param.validate_value(0.5)
    assert is_valid

    is_valid, error = param.validate_value(-0.1)
    assert not is_valid


def test_parameter_value_validation_enum():
    """Test enum parameter validation."""
    param = ParameterSpec(
        name="method",
        kind=ParameterKind.ENUM,
        role=ParameterRole.CATEGORICAL,
        allowed_values=["sma", "ema"],
    )

    is_valid, error = param.validate_value("sma")
    assert is_valid

    is_valid, error = param.validate_value("wma")
    assert not is_valid
    assert "must be one of" in error


def test_parameter_value_validation_boolean():
    """Test boolean parameter validation."""
    param = ParameterSpec(
        name="normalize",
        kind=ParameterKind.BOOLEAN,
        role=ParameterRole.STRUCTURAL,
    )

    is_valid, error = param.validate_value(True)
    assert is_valid

    is_valid, error = param.validate_value(False)
    assert is_valid

    is_valid, error = param.validate_value(1)
    assert not is_valid


def test_parameter_value_validation_required():
    """Test required parameter validation."""
    param = ParameterSpec(
        name="window",
        kind=ParameterKind.INTEGER,
        role=ParameterRole.WINDOW,
        required=True,
    )

    is_valid, error = param.validate_value(None)
    assert not is_valid
    assert "is required" in error


def test_parameter_value_validation_optional():
    """Test optional parameter with default."""
    param = ParameterSpec(
        name="window",
        kind=ParameterKind.INTEGER,
        role=ParameterRole.WINDOW,
        required=False,
        default=20,
    )

    is_valid, error = param.validate_value(None)
    assert is_valid


def test_mutation_spec_creation():
    """Test MutationSpec creation."""
    spec = MutationSpec(
        mutation_type="window_adjust",
        version="0.1.0",
        description="Adjust window parameter",
        parameters=[
            ParameterSpec(
                name="new_window",
                kind=ParameterKind.INTEGER,
                role=ParameterRole.WINDOW,
                min_value=1,
            )
        ],
    )

    assert spec.mutation_type == "window_adjust"
    assert spec.version == "0.1.0"
    assert len(spec.parameters) == 1


def test_mutation_spec_validate_parameters():
    """Test parameter dictionary validation."""
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test mutation",
        parameters=[
            ParameterSpec(name="window", kind=ParameterKind.INTEGER, role=ParameterRole.WINDOW, min_value=1),
            ParameterSpec(name="decay", kind=ParameterKind.FLOAT, role=ParameterRole.DECAY, required=False),
        ],
    )

    # Valid
    is_valid, errors = spec.validate_parameters({"window": 20})
    assert is_valid
    assert len(errors) == 0

    # Missing required
    is_valid, errors = spec.validate_parameters({"decay": 0.5})
    assert not is_valid
    assert any("window" in e and "required" in e for e in errors)

    # Unknown parameter
    is_valid, errors = spec.validate_parameters({"window": 20, "unknown": 123})
    assert not is_valid
    assert any("Unknown parameter" in e for e in errors)


def test_mutation_spec_get_parameter():
    """Test getting parameter by name."""
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
        parameters=[
            ParameterSpec(name="p1", kind=ParameterKind.INTEGER, role=ParameterRole.SCALAR),
            ParameterSpec(name="p2", kind=ParameterKind.FLOAT, role=ParameterRole.DECAY),
        ],
    )

    param = spec.get_parameter("p1")
    assert param is not None
    assert param.name == "p1"

    param = spec.get_parameter("nonexistent")
    assert param is None


def test_mutation_spec_parent_requirements():
    """Test parent count requirements."""
    # Single parent
    spec1 = MutationSpec(
        mutation_type="single",
        version="0.1.0",
        description="Single parent",
        requires_single_parent=True,
    )
    assert spec1.requires_single_parent
    assert not spec1.requires_multiple_parents

    # Multiple parents
    spec2 = MutationSpec(
        mutation_type="multi",
        version="0.1.0",
        description="Multiple parents",
        requires_multiple_parents=True,
        requires_single_parent=False,
    )
    assert not spec2.requires_single_parent
    assert spec2.requires_multiple_parents


def test_mutation_spec_duplicate_parameters():
    """Test duplicate parameter name detection."""
    with pytest.raises(ValueError, match="Duplicate parameter names"):
        MutationSpec(
            mutation_type="test",
            version="0.1.0",
            description="Test",
            parameters=[
                ParameterSpec(name="window", kind=ParameterKind.INTEGER, role=ParameterRole.WINDOW),
                ParameterSpec(name="window", kind=ParameterKind.INTEGER, role=ParameterRole.WINDOW),
            ],
        )
