"""
Test that transform registry isolation works correctly across tests.

Verifies that the conftest.py fixture properly resets the global registry
to prevent test contamination.
"""

from factor_preprocess.registry.transforms import (
    get_default_registry,
    TransformCategory,
    _default_registry,
)


def test_transform_registry_isolation_first():
    """
    First test: register a custom transform.

    This should not leak to the next test.
    """
    registry = get_default_registry()

    # Get baseline count
    initial_transforms = registry.all_transforms()
    initial_count = len(initial_transforms)

    # Register a test-specific transform
    def test_transform_first(df):
        return df * 2

    registry.register(
        name="test_transform_first",
        func=test_transform_first,
        category=TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Test transform for isolation test",
    )

    # Verify registration
    assert registry.get("test_transform_first") is not None
    assert len(registry.all_transforms()) == initial_count + 1


def test_transform_registry_isolation_second():
    """
    Second test: verify previous registration is gone.

    Registry should be fresh with only default transforms.
    """
    registry = get_default_registry()

    # Should not find the transform from the first test
    assert registry.get("test_transform_first") is None

    # Register a different transform
    def test_transform_second(df):
        return df * 3

    registry.register(
        name="test_transform_second",
        func=test_transform_second,
        category=TransformCategory.TEMPORAL,
        version="1.0.0",
        description="Second test transform",
    )

    # Verify only default transforms plus this one exist
    assert registry.get("test_transform_second") is not None
    assert registry.get("test_transform_first") is None


def test_transform_registry_isolation_third():
    """
    Third test: verify both previous registrations are gone.

    Registry should be completely fresh with only defaults.
    """
    registry = get_default_registry()

    # Should not find either previous test transform
    assert registry.get("test_transform_first") is None
    assert registry.get("test_transform_second") is None

    # Should still have all default transforms
    all_transforms = registry.all_transforms()
    assert len(all_transforms) > 0

    # Verify we can still access standard transforms
    assert registry.get("cs_rank") is not None
    assert registry.get("rolling_mean") is not None
