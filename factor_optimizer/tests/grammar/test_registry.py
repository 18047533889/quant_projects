"""Tests for MutationRegistry."""

import pytest
from factor_optimizer.grammar.registry import MutationRegistry, get_mutation_registry
from factor_optimizer.grammar.mutation_spec import (
    MutationSpec,
    ParameterSpec,
    ParameterKind,
    ParameterRole,
)


def test_registry_creation():
    """Test creating an empty registry."""
    registry = MutationRegistry()
    assert len(registry.list_mutation_types()) == 0


def test_registry_register():
    """Test registering a mutation spec."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test_mutation",
        version="0.1.0",
        description="Test mutation",
    )

    registry.register(spec)
    assert "test_mutation" in registry.list_mutation_types()


def test_registry_duplicate_registration():
    """Test that duplicate registration raises error."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
    )

    registry.register(spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_get():
    """Test retrieving mutation specs."""
    registry = MutationRegistry()
    spec = MutationSpec(
        mutation_type="test",
        version="0.1.0",
        description="Test",
    )

    registry.register(spec)

    retrieved = registry.get("test")
    assert retrieved is not None
    assert retrieved.mutation_type == "test"

    not_found = registry.get("nonexistent")
    assert not_found is None


def test_registry_list_specs():
    """Test listing all specs."""
    registry = MutationRegistry()

    spec1 = MutationSpec(mutation_type="m1", version="0.1.0", description="M1")
    spec2 = MutationSpec(mutation_type="m2", version="0.1.0", description="M2")

    registry.register(spec1)
    registry.register(spec2)

    specs = registry.list_specs()
    assert len(specs) == 2
    assert any(s.mutation_type == "m1" for s in specs)
    assert any(s.mutation_type == "m2" for s in specs)


def test_global_registry():
    """Test global registry access."""
    registry = get_mutation_registry()
    assert registry is not None

    # Should have default mutations registered
    types = registry.list_mutation_types()
    assert len(types) > 0
    assert "parameter_tune" in types
    assert "window_adjust" in types


def test_default_mutations_registered():
    """Test that default mutations are properly registered."""
    registry = get_mutation_registry()

    # Check parameter_tune
    param_tune = registry.get("parameter_tune")
    assert param_tune is not None
    assert param_tune.requires_single_parent

    # Check window_adjust
    window_adjust = registry.get("window_adjust")
    assert window_adjust is not None
    assert len(window_adjust.parameters) == 1
    assert window_adjust.parameters[0].name == "new_window"

    # Check linear_combination
    linear_combo = registry.get("linear_combination")
    assert linear_combo is not None
    assert linear_combo.requires_multiple_parents
    assert not linear_combo.requires_single_parent


def test_default_mutation_parameter_specs():
    """Test parameter specifications of default mutations."""
    registry = get_mutation_registry()

    # Window adjust should have min/max constraints
    spec = registry.get("window_adjust")
    param = spec.get_parameter("new_window")
    assert param.kind == ParameterKind.INTEGER
    assert param.role == ParameterRole.WINDOW
    assert param.min_value == 1
    assert param.max_value == 252

    # Decay adjust should have [0,1] range
    spec = registry.get("decay_adjust")
    param = spec.get_parameter("new_decay")
    assert param.kind == ParameterKind.FLOAT
    assert param.role == ParameterRole.DECAY
    assert param.min_value == 0.0
    assert param.max_value == 1.0


def test_registry_version():
    """Test registry version tracking."""
    registry = MutationRegistry()
    version = registry.version()
    assert version is not None
    assert isinstance(version, str)
