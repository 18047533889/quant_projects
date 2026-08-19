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
        """Test registering the exact same transform and metadata is idempotent."""
        registry = TransformRegistry()

        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")
        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")

        assert registry.get("dummy") is not None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"func": another_transform},
            {"category": TransformCategory.TEMPORAL},
            {"description": "Conflicting description"},
            {"parameters": {"scale": 2.0}},
            {"tags": {"conflicting"}},
            {"causal_safe": False, "admission": "OFFLINE_ONLY"},
            {"admission": "RESEARCH_ONLY"},
        ],
    )
    def test_duplicate_registration_same_version_rejects_conflicts(self, overrides):
        registry = TransformRegistry()
        registration = {
            "name": "dummy",
            "func": dummy_transform,
            "category": TransformCategory.CROSS_SECTIONAL,
            "version": "1.0.0",
            "description": "Original description",
            "parameters": {},
            "tags": {"original"},
            "causal_safe": True,
            "admission": "PRODUCTION",
        }
        registry.register(**registration)

        conflicting = {**registration, **overrides}
        with pytest.raises(ValueError, match="conflicting metadata or implementation"):
            registry.register(**conflicting)

        metadata = registry.get("dummy")
        assert metadata is not None
        assert metadata.func is dummy_transform
        assert metadata.category == TransformCategory.CROSS_SECTIONAL
        assert metadata.admission == "PRODUCTION"

    def test_registration_copies_mutable_metadata_inputs(self):
        registry = TransformRegistry()
        parameters = {"scale": 1.0, "nested": {"limit": 1}}
        tags = {"original"}
        registry.register(
            "dummy",
            dummy_transform,
            TransformCategory.CROSS_SECTIONAL,
            parameters=parameters,
            tags=tags,
        )

        parameters["scale"] = 2.0
        parameters["nested"]["limit"] = 9
        tags.add("mutated")

        metadata = registry.get("dummy")
        assert metadata is not None
        assert metadata.parameters == {"scale": 1.0, "nested": {"limit": 1}}
        assert metadata.tags == {"original"}

    @pytest.mark.parametrize(
        "retrieve",
        [
            lambda registry: registry.get("dummy"),
            lambda registry: registry.validate_production("dummy"),
            lambda registry: registry.list_by_category(TransformCategory.CROSS_SECTIONAL)[0],
            lambda registry: registry.list_by_tag("original")[0],
            lambda registry: registry.list_causal_safe()[0],
            lambda registry: registry.all_transforms()[0],
        ],
    )
    def test_retrieved_metadata_mutation_does_not_change_registry(self, retrieve):
        registry = TransformRegistry()
        registry.register(
            "dummy",
            dummy_transform,
            TransformCategory.CROSS_SECTIONAL,
            tags={"original"},
            admission="PRODUCTION",
        )

        exposed = retrieve(registry)
        assert exposed is not None
        exposed.category = TransformCategory.TEMPORAL
        exposed.admission = "OFFLINE_ONLY"
        exposed.causal_safe = False
        exposed.parameters["mutated"] = True
        exposed.tags.add("mutated")

        retained = registry.get("dummy")
        assert retained is not None
        assert retained.category == TransformCategory.CROSS_SECTIONAL
        assert retained.admission == "PRODUCTION"
        assert retained.causal_safe is True
        assert retained.parameters == {}
        assert retained.tags == {"original"}
        assert [item.name for item in registry.list_by_category(
            TransformCategory.CROSS_SECTIONAL
        )] == ["dummy"]
        assert registry.list_by_category(TransformCategory.TEMPORAL) == []
        assert registry.list_by_tag("mutated") == []

    def test_offline_metadata_snapshot_cannot_bypass_production_validation(self):
        registry = TransformRegistry()
        registry.register(
            "offline",
            dummy_transform,
            TransformCategory.TEMPORAL,
            causal_safe=False,
            admission="OFFLINE_ONLY",
        )

        exposed = registry.get("offline")
        assert exposed is not None
        exposed.admission = "PRODUCTION"
        exposed.causal_safe = True

        with pytest.raises(ValueError, match="OFFLINE_ONLY"):
            registry.validate_production("offline")

    def test_duplicate_registration_different_version(self):
        """Test registering same name with different version raises."""
        registry = TransformRegistry()

        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="1.0.0")

        with pytest.raises(ValueError, match="already registered with version"):
            registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL, version="2.0.0")

    def test_omitted_admission_is_unverified_and_rejected_for_production(self):
        registry = TransformRegistry()
        registry.register("dummy", dummy_transform, TransformCategory.CROSS_SECTIONAL)

        metadata = registry.get("dummy")
        assert metadata is not None
        assert metadata.admission == "UNKNOWN"
        with pytest.raises(ValueError, match="UNKNOWN"):
            registry.validate_production("dummy")

    def test_explicit_production_admission_passes_validation(self):
        registry = TransformRegistry()
        registry.register(
            "dummy",
            dummy_transform,
            TransformCategory.CROSS_SECTIONAL,
            admission="PRODUCTION",
        )

        metadata = registry.validate_production("dummy")
        assert metadata.name == "dummy"

    def test_invalid_admission_is_rejected(self):
        with pytest.raises(ValueError, match="admission must be"):
            TransformRegistry().register(
                "dummy",
                dummy_transform,
                TransformCategory.CROSS_SECTIONAL,
                admission="UNKNOWN",
            )

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
            causal_safe=False,
            admission="RESEARCH_ONLY",
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
        """Test that every production-admitted transform is marked causal safe."""
        registry = get_default_registry()

        all_transforms = registry.all_transforms()
        assert len(all_transforms) > 0

        for meta in all_transforms:
            if meta.admission == "PRODUCTION":
                assert meta.causal_safe is True, f"{meta.name} should be causal_safe"

    def test_full_series_transforms_are_offline_only(self):
        registry = get_default_registry()
        for name in [
            "bandpass_filter",
            "extract_cycle",
            "christiano_fitzgerald_filter",
            "wavelet_decompose",
            "wavelet_smooth",
            "wavelet_denoise",
        ]:
            metadata = registry.get(name)
            assert metadata is not None
            assert metadata.causal_safe is False
            assert metadata.admission == "OFFLINE_ONLY"
            with pytest.raises(ValueError, match="OFFLINE_ONLY"):
                registry.validate_production(name)

    def test_full_sample_transforms_are_research_only(self):
        registry = get_default_registry()
        for name in ["impute_with_fallback", "detect_correlation_regime"]:
            metadata = registry.get(name)
            assert metadata is not None
            assert metadata.causal_safe is False
            assert metadata.admission == "RESEARCH_ONLY"
            with pytest.raises(ValueError, match="RESEARCH_ONLY"):
                registry.validate_production(name)

    def test_production_validation_rejects_unknown_transform(self):
        with pytest.raises(ValueError, match="not registered"):
            TransformRegistry().validate_production("unregistered_transform")

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
