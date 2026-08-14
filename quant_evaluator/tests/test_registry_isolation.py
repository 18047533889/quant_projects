"""
Test that backend registry isolation works correctly across tests.

Verifies that the conftest.py fixture properly resets the global registry
to prevent test contamination.
"""

from quant_evaluator.backends.selector import BackendType


def test_registry_isolation_first():
    """
    First test: register a custom operation.

    This should not leak to the next test.
    """
    # Import registry module to get current instance
    from quant_evaluator.backends import registry

    # Registry should start empty (or with defaults only)
    initial_ops = registry._global_registry.list_operations()

    # Register a test-specific operation directly
    def test_impl_first(x):
        return x * 2

    registry._global_registry.register("test_op_first", BackendType.NUMPY, test_impl_first)

    # Verify registration
    assert "test_op_first" in registry._global_registry.list_operations()
    impl = registry._global_registry.get("test_op_first", BackendType.NUMPY)
    assert impl is not None
    assert impl(5) == 10


def test_registry_isolation_second():
    """
    Second test: verify previous registration is gone.

    This test should not see the operation registered in the first test.
    """
    # Import registry module to get current instance
    from quant_evaluator.backends import registry

    # Registry should be clean - no test_op_first from previous test
    ops = registry._global_registry.list_operations()
    assert "test_op_first" not in ops

    # Register a different operation
    def test_impl_second(x):
        return x * 3

    registry._global_registry.register("test_op_second", BackendType.NUMPY, test_impl_second)

    # Verify only this operation exists (not the first one)
    assert "test_op_second" in registry._global_registry.list_operations()
    assert "test_op_first" not in registry._global_registry.list_operations()

    impl = registry._global_registry.get("test_op_second", BackendType.NUMPY)
    assert impl is not None
    assert impl(5) == 15


def test_registry_isolation_third():
    """
    Third test: verify both previous registrations are gone.

    Registry should be completely fresh.
    """
    # Import registry module to get current instance
    from quant_evaluator.backends import registry

    # Registry should be clean - neither previous operation should exist
    ops = registry._global_registry.list_operations()
    assert "test_op_first" not in ops
    assert "test_op_second" not in ops
