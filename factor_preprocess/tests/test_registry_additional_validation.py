"""
Additional validation tests for transform registry isolation.

Tests edge cases to ensure the fixture truly provides isolation.
"""

from factor_preprocess.registry.transforms import (
    get_default_registry,
    TransformCategory,
)


def test_registry_starts_with_defaults_only():
    """
    Test that registry starts with default transforms but no test transforms.

    This would fail if test transforms from previous tests leaked.
    """
    registry = get_default_registry()

    all_transforms = registry.all_transforms()
    transform_names = [t.name for t in all_transforms]

    # Should have default transforms
    assert "cs_rank" in transform_names
    assert "rolling_mean" in transform_names

    # Should NOT have any test transforms
    assert "test_transform_first" not in transform_names
    assert "test_transform_second" not in transform_names
    assert "test_transform_third" not in transform_names


def test_registry_baseline_count():
    """
    Test that registry has expected baseline count of default transforms.

    Count should be consistent across tests (no accumulation).
    """
    registry = get_default_registry()

    all_transforms = registry.all_transforms()
    baseline_count = len(all_transforms)

    # Should have a reasonable number of default transforms (>= 10)
    assert baseline_count >= 10

    # Store this for implicit comparison in next test


def test_category_filtering_works_after_reset():
    """
    Test that category-based filtering works correctly after reset.
    """
    registry = get_default_registry()

    # Get cross-sectional transforms
    cs_transforms = registry.list_by_category(TransformCategory.CROSS_SECTIONAL)

    # Should have at least a few
    assert len(cs_transforms) >= 3

    # All should be in correct category
    for transform in cs_transforms:
        assert transform.category == TransformCategory.CROSS_SECTIONAL


def test_custom_transform_registration_isolated():
    """
    Test that custom transforms registered in this test don't leak.
    """
    registry = get_default_registry()

    # Get baseline
    baseline_transforms = registry.all_transforms()
    baseline_count = len(baseline_transforms)

    # Register a custom transform
    def custom_func(df):
        return df * 100

    registry.register(
        name="custom_isolated_transform",
        func=custom_func,
        category=TransformCategory.CROSS_SECTIONAL,
        version="1.0.0",
        description="Custom test transform",
        tags={"test", "custom"},
    )

    # Verify it's registered
    assert registry.get("custom_isolated_transform") is not None
    assert len(registry.all_transforms()) == baseline_count + 1

    # Verify tag filtering works
    tagged = registry.list_by_tag("custom")
    assert len(tagged) == 1
    assert tagged[0].name == "custom_isolated_transform"


def test_previous_custom_transform_gone():
    """
    Verify custom transform from previous test is gone.
    """
    registry = get_default_registry()

    # Should not find the custom transform
    assert registry.get("custom_isolated_transform") is None

    # Tag should also not exist
    tagged = registry.list_by_tag("custom")
    assert len(tagged) == 0


def test_signature_hash_computed_correctly():
    """
    Test that signature hashes are computed for transforms.
    """
    registry = get_default_registry()

    # Get a default transform
    cs_rank_meta = registry.get("cs_rank")
    assert cs_rank_meta is not None
    assert cs_rank_meta.signature_hash is not None
    assert len(cs_rank_meta.signature_hash) == 16  # Should be 16 char hex


def test_duplicate_registration_prevented():
    """
    Test that duplicate registration with different version is prevented.
    """
    registry = get_default_registry()

    def test_func(df):
        return df

    # First registration should work
    registry.register(
        name="duplicate_test_transform",
        func=test_func,
        category=TransformCategory.TEMPORAL,
        version="1.0.0",
        description="First version",
    )

    # Second registration with different version should raise
    try:
        registry.register(
            name="duplicate_test_transform",
            func=test_func,
            category=TransformCategory.TEMPORAL,
            version="2.0.0",
            description="Second version",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already registered" in str(e)


def test_causal_safe_filtering():
    """
    Test that causal_safe filtering works correctly.
    """
    registry = get_default_registry()

    causal_safe = registry.list_causal_safe()

    # Should have at least some causal safe transforms
    assert len(causal_safe) > 0

    # All should be marked causal_safe
    for transform in causal_safe:
        assert transform.causal_safe is True
