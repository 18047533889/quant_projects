"""Tests for transform registry."""
import pytest
import numpy as np
from factor_preprocess.registry import (
    TransformRegistry,
    TransformCategory,
    TransformMetadata,
    get_default_registry,
)


def dummy_transform(x: np.ndarray) -> np.ndarray:
    """Dummy transform for testing."""
    return x * 2


def another_transform(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Another dummy transform."""
    return x * scale


class TestTransformRegistry:
    """Test TransformRegistry basic functionality."""

    def test_register_and_get(self):
        """Test registering and retrieving transforms."""
        registry = TransformRegistry()

        registry.register(
            name="dummy",
            func=dummy_transform,
            category=TransformCategory.CROSS_SECTIONAL,
            version="1.0.0",
            description="Test transform",
        )

        meta = registry.get("dummy")
        assert meta is not None
        assert meta.name == "dummy"
        assert meta.func is dummy_transform
        assert meta.category == TransformCategory.CROSS_SECTIONAL
        assert meta.version == "1.0.0"
        assert meta.causal_safe is True

    def test_get_function(self):
        """Test retrieving function directly."""
        registry = TransformRegistry()
        registry.register(
            "dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL
        )

        func = registry.get_function("dummy")
        assert func is dummy_transform

        # Test it works
        result = func(np.array([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))

    def test_duplicate_registration_same_version(self):
        """Test registering same transform twice with same version is idempotent."""
        registry = TransformRegistry()

        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")

        # Should not raise, just idempotent
        assert registry.get("dummy") is not None

    def test_duplicate_registration_different_version(self):
        """Test registering same name with different version raises."""
        registry = TransformRegistry()

        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")

        with pytest.raises(ValueError, match="already registered with version"):
            registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="2.0.0")

    def test_list_by_category(self):
        """Test listing transforms by category."""
        registry = TransformRegistry()

        registry.register("cs1", dummy_transform, TransformCategory.CROSS_SECTIONAL)
        registry.register("cs2", another_transform, TransformCategory.CROSS_SECTIONAL)
        registry.register("temp1", dummy_transform, TransformCategory.TEMPORAL)

        cs_transforms = registry.list_by_category(TransformCategory.CROSS_SECTIONAL)
        assert len(cs_transforms) == 2
        assert set(t.name for t in cs_transforms) == {"cs1", "cs2"}

        temp_transforms = registry.list_by_category(TransformCategory.TEMPORAL)
        assert len(temp_transforms) == 1
        assert temp_transforms[0].name == "temp1"

    def test_list_by_tag(self):
        """Test listing transforms by tag."""
        registry = TransformRegistry()

        registry.register(
            "t1", dummy_transform, TransformCategory.CROSS_SECTIONAL,
            tags={"rank", "normalization"}
        )
        registry.register(
            "t2", another_transform, TransformCategory.CROSS_SECTIONAL,
            tags={"rank", "outlier"}
        )
        registry.register(
            "t3", dummy_transform, TransformCategory.TEMPORAL,
            tags={"smoothing"}
        )

        rank_transforms = registry.list_by_tag("rank")
        assert len(rank_transforms) == 2
        assert set(t.name for t in rank_transforms) == {"t1", "t2"}

        smoothing_transforms = registry.list_by_tag("smoothing")
        assert len(smoothing_transforms) == 1
        assert smoothing_transforms[0].name == "t3"

    def test_list_causal_safe(self):
        """Test filtering causal-safe transforms."""
        registry = TransformRegistry()

        registry.register(
            "safe1", dummy_transform, TransformCategory.CROSS_SECTIONAL,
            causal_safe=True
        )
        registry.register(
            "safe2", another_transform, TransformCategory.TEMPORAL,
            causal_safe=True
        )
        registry.register(
            "unsafe", dummy_transform, TransformCategory.CROSS_SECTIONAL,
            causal_safe=False
        )

        safe = registry.list_causal_safe()
        assert len(safe) == 2
        assert set(t.name for t in safe) == {"safe1", "safe2"}
        assert all(t.causal_safe for t in safe)

    def test_signature_hash_generation(self):
        """Test that signature hashes are generated."""
        registry = TransformRegistry()
        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL)

        hash_val = registry.get_signature_hash("dummy")
        assert hash_val is not None
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # 16 hex chars

    def test_signature_hash_stability(self):
        """Test that signature hashes are stable across registrations."""
        registry1 = TransformRegistry()
        registry1.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL)
        hash1 = registry1.get_signature_hash("dummy")

        registry2 = TransformRegistry()
        registry2.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL)
        hash2 = registry2.get_signature_hash("dummy")

        assert hash1 == hash2

    def test_all_transforms(self):
        """Test getting all transforms."""
        registry = TransformRegistry()
        registry.register("t1", dummy_transform, TransformCategory.CROSS_SECTIONAL)
        registry.register("t2", another_transform, TransformCategory.TEMPORAL)

        all_transforms = registry.all_transforms()
        assert len(all_transforms) == 2
        assert set(t.name for t in all_transforms) == {"t1", "t2"}


class TestDefaultRegistry:
    """Test default registry population."""

    def test_default_registry_exists(self):
        """Test that default registry can be retrieved."""
        registry = get_default_registry()
        assert registry is not None
        assert isinstance(registry, TransformRegistry)

    def test_default_registry_singleton(self):
        """Test that default registry is a singleton."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2

    def test_default_registry_has_cross_sectional(self):
        """Test default registry includes cross-sectional transforms."""
        registry = get_default_registry()

        cs_transforms = registry.list_by_category(TransformCategory.CROSS_SECTIONAL)
        assert len(cs_transforms) > 0

        # Check specific transforms exist
        assert registry.get("cs_rank") is not None
        assert registry.get("cs_zscore") is not None
        assert registry.get("cs_demean") is not None
        assert registry.get("cs_winsor") is not None
        assert registry.get("cs_scale") is not None

    def test_default_registry_has_temporal(self):
        """Test default registry includes temporal transforms."""
        registry = get_default_registry()

        temp_transforms = registry.list_by_category(TransformCategory.TEMPORAL)
        assert len(temp_transforms) > 0

        assert registry.get("rolling_mean") is not None
        assert registry.get("rolling_std") is not None
        assert registry.get("ewma") is not None

    def test_default_registry_has_volatility(self):
        """Test default registry includes volatility transforms."""
        registry = get_default_registry()

        vol_transforms = registry.list_by_category(TransformCategory.VOLATILITY)
        assert len(vol_transforms) > 0

        assert registry.get("volatility_scale") is not None
        assert registry.get("realized_volatility") is not None

    def test_default_registry_has_missingness(self):
        """Test default registry includes missingness transforms."""
        registry = get_default_registry()

        miss_transforms = registry.list_by_category(TransformCategory.MISSINGNESS)
        assert len(miss_transforms) > 0

        assert registry.get("forward_fill") is not None
        assert registry.get("missing_indicator") is not None

    def test_default_registry_has_freshness(self):
        """Test default registry includes freshness transforms."""
        registry = get_default_registry()

        fresh_transforms = registry.list_by_category(TransformCategory.FRESHNESS)
        assert len(fresh_transforms) > 0

        assert registry.get("days_since_update") is not None
        assert registry.get("freshness_score") is not None

    def test_default_registry_has_neutralization(self):
        """Test default registry includes neutralization transforms."""
        registry = get_default_registry()

        neut_transforms = registry.list_by_category(TransformCategory.NEUTRALIZATION)
        assert len(neut_transforms) > 0

        assert registry.get("ols_neutralize") is not None
        assert registry.get("compute_exposures") is not None

    def test_all_default_transforms_causal_safe(self):
        """Test that all default transforms are marked causal safe."""
        registry = get_default_registry()

        all_transforms = registry.all_transforms()
        assert len(all_transforms) > 0

        for meta in all_transforms:
            assert meta.causal_safe is True, f"{meta.name} should be causal_safe"

    def test_default_registry_functions_callable(self):
        """Test that registered functions are actually callable."""
        registry = get_default_registry()

        # Sample a few transforms and verify they're callable
        for name in ["cs_rank", "cs_zscore", "rolling_mean"]:
            func = registry.get_function(name)
            assert func is not None
            assert callable(func)


class TestTransformMetadata:
    """Test TransformMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata directly."""
        meta = TransformMetadata(
            name="test",
            func=dummy_transform,
            category=TransformCategory.CROSS_SECTIONAL,
            version="1.0.0",
            description="Test",
        )

        assert meta.name == "test"
        assert meta.func is dummy_transform
        assert meta.signature_hash is not None

    def test_metadata_with_parameters(self):
        """Test metadata with default parameters."""
        meta = TransformMetadata(
            name="test",
            func=another_transform,
            category=TransformCategory.CROSS_SECTIONAL,
            version="1.0.0",
            description="Test",
            parameters={"scale": 2.0, "offset": 0.1},
        )

        assert meta.parameters == {"scale": 2.0, "offset": 0.1}

    def test_metadata_with_tags(self):
        """Test metadata with tags."""
        meta = TransformMetadata(
            name="test",
            func=dummy_transform,
            category=TransformCategory.CROSS_SECTIONAL,
            version="1.0.0",
            description="Test",
            tags={"rank", "normalization", "production"},
        )

        assert meta.tags == {"rank", "normalization", "production"}
