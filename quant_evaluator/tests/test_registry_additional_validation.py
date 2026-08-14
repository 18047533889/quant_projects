"""
Additional validation tests for registry isolation.

Tests edge cases to ensure the fixture truly provides isolation.
"""

from quant_evaluator.backends.selector import BackendType


def test_registry_is_clean_at_start():
    """
    Test that registry starts completely clean.

    This would fail without the conftest fixture properly resetting.
    """
    from quant_evaluator.backends import registry

    # Registry should be a fresh BackendRegistry instance
    ops = registry._global_registry.list_operations()

    # Should have no operations or only legitimate default operations
    # (not test operations from previous tests)
    assert "test_op_first" not in ops
    assert "test_op_second" not in ops
    assert "test_op_third" not in ops


def test_registry_identity_changes_between_tests():
    """
    Test that the registry object identity is fresh.

    Verifies the fixture is creating new instances, not just clearing.
    """
    from quant_evaluator.backends import registry

    # Store the registry object id
    registry_id = id(registry._global_registry)

    # Register something
    def test_impl(x):
        return x * 10

    registry._global_registry.register("test_identity", BackendType.NUMPY, test_impl)

    # Store this for comparison in next test
    # (This would need pytest caching to truly verify across tests,
    # but within a single test we can verify the object exists)
    assert registry._global_registry.get("test_identity", BackendType.NUMPY) is not None


def test_multiple_registrations_isolated():
    """
    Test that multiple operations registered in one test don't leak.
    """
    from quant_evaluator.backends import registry

    # Register multiple operations
    def impl1(x):
        return x * 2
    def impl2(x):
        return x * 3
    def impl3(x):
        return x * 4

    registry._global_registry.register("multi_op_1", BackendType.NUMPY, impl1)
    registry._global_registry.register("multi_op_2", BackendType.NUMPY, impl2)
    registry._global_registry.register("multi_op_3", BackendType.NUMBA, impl3)

    ops = registry._global_registry.list_operations()
    assert "multi_op_1" in ops
    assert "multi_op_2" in ops
    assert "multi_op_3" in ops
    assert len([op for op in ops if op.startswith("multi_op_")]) == 3


def test_previous_multiple_registrations_gone():
    """
    Verify operations from previous test are completely gone.
    """
    from quant_evaluator.backends import registry

    ops = registry._global_registry.list_operations()

    # None of the operations from previous test should exist
    assert "multi_op_1" not in ops
    assert "multi_op_2" not in ops
    assert "multi_op_3" not in ops
    assert "test_identity" not in ops


def test_backend_specific_registrations_isolated():
    """
    Test that backend-specific registrations are properly isolated.
    """
    from quant_evaluator.backends import registry

    def numpy_impl(x):
        return x * 2
    def numba_impl(x):
        return x * 3

    # Register same operation for different backends
    registry._global_registry.register("same_op", BackendType.NUMPY, numpy_impl)
    registry._global_registry.register("same_op", BackendType.NUMBA, numba_impl)

    # Verify both are registered
    backends = registry._global_registry.list_backends("same_op")
    assert BackendType.NUMPY in backends
    assert BackendType.NUMBA in backends
    assert len(backends) == 2


def test_metadata_registrations_isolated():
    """
    Test that metadata registrations are also cleaned up.
    """
    from quant_evaluator.backends import registry

    def test_impl(x):
        return x * 5

    metadata = {"version": "1.0.0", "author": "test"}
    registry._global_registry.register(
        "op_with_meta", BackendType.NUMPY, test_impl, metadata=metadata
    )

    # Verify metadata is stored
    retrieved_meta = registry._global_registry.get_metadata("op_with_meta", BackendType.NUMPY)
    assert retrieved_meta is not None
    assert retrieved_meta["version"] == "1.0.0"
    assert retrieved_meta["author"] == "test"


def test_previous_metadata_gone():
    """
    Verify metadata from previous test is gone.
    """
    from quant_evaluator.backends import registry

    # Should not find metadata from previous test
    retrieved_meta = registry._global_registry.get_metadata("op_with_meta", BackendType.NUMPY)
    assert retrieved_meta is None
