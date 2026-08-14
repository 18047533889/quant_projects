"""
Test that mutation registry isolation works correctly across tests.

Verifies that the conftest.py fixture properly resets the global registry
to prevent test contamination.
"""

from factor_optimizer.grammar.registry import (
    get_mutation_registry,
    _GLOBAL_REGISTRY,
)
from factor_optimizer.grammar.mutation_spec import (
    MutationSpec,
    ParameterSpec,
    ParameterKind,
    ParameterRole,
)


def test_mutation_registry_isolation_first():
    """
    First test: register a custom mutation.

    This should not leak to the next test.
    """
    registry = get_mutation_registry()

    # Get baseline count
    initial_types = registry.list_mutation_types()
    initial_count = len(initial_types)

    # Register a test-specific mutation
    test_spec = MutationSpec(
        mutation_type="test_mutation_first",
        version="0.1.0",
        description="Test mutation for isolation",
        parameters=[
            ParameterSpec(
                name="test_param",
                kind=ParameterKind.FLOAT,
                role=ParameterRole.SCALAR,
                description="Test parameter",
            )
        ],
        requires_single_parent=True,
    )
    registry.register(test_spec)

    # Verify registration
    assert registry.get("test_mutation_first") is not None
    assert len(registry.list_mutation_types()) == initial_count + 1
    assert "test_mutation_first" in registry.list_mutation_types()


def test_mutation_registry_isolation_second():
    """
    Second test: verify previous registration is gone.

    Registry should be fresh with only default mutations.
    """
    registry = get_mutation_registry()

    # Should not find the mutation from the first test
    assert registry.get("test_mutation_first") is None
    assert "test_mutation_first" not in registry.list_mutation_types()

    # Register a different mutation
    test_spec = MutationSpec(
        mutation_type="test_mutation_second",
        version="0.1.0",
        description="Second test mutation",
        parameters=[
            ParameterSpec(
                name="test_param2",
                kind=ParameterKind.INTEGER,
                role=ParameterRole.WINDOW,
                description="Test parameter 2",
            )
        ],
        requires_single_parent=True,
    )
    registry.register(test_spec)

    # Verify only default mutations plus this one exist
    assert registry.get("test_mutation_second") is not None
    assert registry.get("test_mutation_first") is None
    assert "test_mutation_second" in registry.list_mutation_types()


def test_mutation_registry_isolation_third():
    """
    Third test: verify both previous registrations are gone.

    Registry should be completely fresh with only defaults.
    """
    registry = get_mutation_registry()

    # Should not find either previous test mutation
    assert registry.get("test_mutation_first") is None
    assert registry.get("test_mutation_second") is None
    assert "test_mutation_first" not in registry.list_mutation_types()
    assert "test_mutation_second" not in registry.list_mutation_types()

    # Should still have all default mutations
    all_types = registry.list_mutation_types()
    assert len(all_types) > 0

    # Verify we can still access standard mutations
    assert registry.get("parameter_tune") is not None
    assert registry.get("window_adjust") is not None
    assert registry.get("operator_swap") is not None
