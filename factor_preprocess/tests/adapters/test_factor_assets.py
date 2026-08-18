"""
Tests for factor_assets adapter.

Tests protocol compliance, mock providers, and graceful failure.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

from factor_preprocess.adapters.factor_assets import (
    FactorSetProvider,
    FactorAssetsAdapter,
    OptionalDependencyMissing,
    check_factor_assets_available,
    create_adapter,
)


class MockFactorSet:
    """Mock FactorSet for testing."""

    def __init__(
        self,
        set_id: str,
        name: str,
        factor_ids: tuple[str, ...],
        created_at: str,
        universe_ref: Optional[str] = None,
        frequency: Optional[str] = None,
    ):
        self.set_id = set_id
        self.name = name
        self.factor_ids = factor_ids
        self.created_at = created_at
        self.universe_ref = universe_ref
        self.frequency = frequency
        self.size = len(factor_ids)


class MockFactorSetProvider:
    """Mock provider for testing."""

    def __init__(self):
        self.call_log = []
        # Mock data: 10 dates x 5 assets x 3 factors
        self.n_dates = 10
        self.n_assets = 5
        self.dates = np.array([f"2024-01-{i+1:02d}" for i in range(self.n_dates)])
        self.assets = np.array([f"ASSET_{i}" for i in range(self.n_assets)])
        self.factors = {
            "FACTOR_A": np.random.randn(self.n_dates, self.n_assets),
            "FACTOR_B": np.random.randn(self.n_dates, self.n_assets),
            "FACTOR_C": np.random.randn(self.n_dates, self.n_assets),
        }

    def get_factor_values(
        self,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.call_log.append(("get_factor_values", factor_id))

        if factor_id not in self.factors:
            raise ValueError(f"Factor {factor_id} not found")

        return {
            "values": self.factors[factor_id],
            "dates": self.dates,
            "assets": self.assets,
            "metadata": {
                "factor_id": factor_id,
                "source": "mock",
            },
        }

    def get_factor_batch(
        self,
        factor_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.call_log.append(("get_factor_batch", tuple(factor_ids)))

        # Stack factors along axis 2
        values_list = []
        metadata = {}
        for fid in factor_ids:
            if fid not in self.factors:
                raise ValueError(f"Factor {fid} not found")
            values_list.append(self.factors[fid])
            metadata[fid] = {"factor_id": fid, "source": "mock"}

        # Handle empty factor list
        if len(values_list) == 0:
            values = np.empty((self.n_dates, self.n_assets, 0))
        else:
            values = np.stack(values_list, axis=2)  # (n_dates, n_assets, n_factors)

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "factor_ids": factor_ids,
            "metadata": metadata,
        }

    def validate_factor_set(self, factor_set: Any) -> bool:
        self.call_log.append(("validate_factor_set", factor_set.set_id))

        # Check all factors exist
        for fid in factor_set.factor_ids:
            if fid not in self.factors:
                raise ValueError(f"Factor {fid} in set but not found in provider")
        return True


class TestFactorSetProvider:
    """Test the protocol definition."""

    def test_mock_provider_implements_protocol(self):
        """Mock provider should satisfy the protocol."""
        provider = MockFactorSetProvider()

        # Should have required methods
        assert hasattr(provider, "get_factor_values")
        assert hasattr(provider, "get_factor_batch")
        assert hasattr(provider, "validate_factor_set")

    def test_mock_provider_get_factor_values(self):
        """Test single factor retrieval."""
        provider = MockFactorSetProvider()
        result = provider.get_factor_values("FACTOR_A")

        assert "values" in result
        assert "dates" in result
        assert "assets" in result
        assert "metadata" in result
        assert result["values"].shape == (10, 5)

    def test_mock_provider_get_factor_batch(self):
        """Test batch factor retrieval."""
        provider = MockFactorSetProvider()
        result = provider.get_factor_batch(["FACTOR_A", "FACTOR_B"])

        assert "values" in result
        assert result["values"].shape == (10, 5, 2)
        assert "factor_ids" in result
        assert result["factor_ids"] == ["FACTOR_A", "FACTOR_B"]

    def test_mock_provider_validate_factor_set(self):
        """Test factor set validation."""
        provider = MockFactorSetProvider()
        factor_set = MockFactorSet(
            set_id="test_set",
            name="Test Set",
            factor_ids=("FACTOR_A", "FACTOR_B"),
            created_at="2024-01-01T00:00:00",
        )

        assert provider.validate_factor_set(factor_set) is True

    def test_mock_provider_validation_failure(self):
        """Test validation with missing factor."""
        provider = MockFactorSetProvider()
        factor_set = MockFactorSet(
            set_id="test_set",
            name="Test Set",
            factor_ids=("FACTOR_A", "FACTOR_MISSING"),
            created_at="2024-01-01T00:00:00",
        )

        with pytest.raises(ValueError, match="not found in provider"):
            provider.validate_factor_set(factor_set)


class TestFactorAssetsAdapter:
    """Test the adapter implementation."""

    def test_adapter_initialization(self):
        """Adapter should accept a provider."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)
        assert adapter is not None

    def test_load_factor_set(self):
        """Test loading a FactorSet."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        factor_set = MockFactorSet(
            set_id="test_set",
            name="Test Set",
            factor_ids=("FACTOR_A", "FACTOR_B", "FACTOR_C"),
            created_at="2024-01-01T00:00:00",
            universe_ref="UNIVERSE_1",
            frequency="daily",
        )

        result = adapter.load_factor_set(factor_set)

        # Check structure
        assert "values" in result
        assert "dates" in result
        assert "assets" in result
        assert "factor_ids" in result
        assert "metadata" in result
        assert "set_metadata" in result

        # Check shape
        assert result["values"].shape == (10, 5, 3)

        # Check set metadata
        set_meta = result["set_metadata"]
        assert set_meta["set_id"] == "test_set"
        assert set_meta["set_name"] == "Test Set"
        assert set_meta["n_factors"] == 3
        assert set_meta["universe"] == "UNIVERSE_1"
        assert set_meta["frequency"] == "daily"

    def test_load_factor_set_with_filters(self):
        """Test loading with date filters."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        factor_set = MockFactorSet(
            set_id="test_set",
            name="Test Set",
            factor_ids=("FACTOR_A",),
            created_at="2024-01-01T00:00:00",
        )

        result = adapter.load_factor_set(
            factor_set,
            start_date="2024-01-01",
            end_date="2024-01-10",
        )

        assert result is not None
        # Provider should have been called with batch method
        assert ("get_factor_batch", ("FACTOR_A",)) in provider.call_log

    def test_load_single_factor(self):
        """Test loading a single factor by ID."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        result = adapter.load_single_factor("FACTOR_A")

        assert "values" in result
        assert "dates" in result
        assert "assets" in result
        assert result["values"].shape == (10, 5)

    def test_load_single_factor_not_found(self):
        """Test error when factor doesn't exist."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        with pytest.raises(ValueError, match="not found"):
            adapter.load_single_factor("FACTOR_MISSING")

    def test_invalid_factor_set(self):
        """Test validation failure."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        factor_set = MockFactorSet(
            set_id="bad_set",
            name="Bad Set",
            factor_ids=("FACTOR_MISSING",),
            created_at="2024-01-01T00:00:00",
        )

        with pytest.raises(ValueError, match="not found"):
            adapter.load_factor_set(factor_set)


class TestOptionalDependency:
    """Test optional dependency handling."""

    def test_check_factor_assets_available(self):
        """Test availability check."""
        # This will be False unless factor_assets is installed
        available = check_factor_assets_available()
        assert isinstance(available, bool)

    def test_optional_dependency_missing_exception(self):
        """Test exception structure."""
        exc = OptionalDependencyMissing("test_package", "test_feature")

        assert exc.package_name == "test_package"
        assert exc.feature_name == "test_feature"
        assert "test_package" in str(exc)
        assert "test_feature" in str(exc)
        assert "pip install" in str(exc)

    def test_create_adapter_with_provider(self):
        """Test create_adapter with explicit provider."""
        provider = MockFactorSetProvider()
        adapter = create_adapter(provider=provider)

        assert isinstance(adapter, FactorAssetsAdapter)

    def test_create_adapter_without_factor_assets(self):
        """Test create_adapter fails gracefully without factor_assets."""
        # Only test if factor_assets is NOT installed
        if not check_factor_assets_available():
            with pytest.raises(OptionalDependencyMissing) as exc_info:
                create_adapter()

            assert exc_info.value.package_name == "factor_assets"

    def test_availability_requires_default_provider(self):
        """An importable package alone must not advertise default integration."""
        try:
            import factor_assets  # noqa: F401
        except ImportError:
            pytest.skip("factor_assets is not installed")

        try:
            from factor_preprocess.adapters._factor_assets_impl import (  # noqa: F401
                DefaultFactorSetProvider,
            )
        except ImportError:
            assert not check_factor_assets_available()
            with pytest.raises(OptionalDependencyMissing):
                create_adapter()
        else:
            assert check_factor_assets_available()


class TestProtocolCompliance:
    """Test that mock provider matches the protocol."""

    def test_provider_signature_matches_protocol(self):
        """Verify all protocol methods are implemented."""
        provider = MockFactorSetProvider()

        # Test get_factor_values signature
        result = provider.get_factor_values(
            factor_id="FACTOR_A",
            start_date=None,
            end_date=None,
            universe=None,
        )
        assert isinstance(result, dict)

        # Test get_factor_batch signature
        result = provider.get_factor_batch(
            factor_ids=["FACTOR_A"],
            start_date=None,
            end_date=None,
            universe=None,
        )
        assert isinstance(result, dict)

        # Test validate_factor_set signature
        factor_set = MockFactorSet(
            set_id="test", name="Test", factor_ids=("FACTOR_A",), created_at="2024-01-01"
        )
        result = provider.validate_factor_set(factor_set)
        assert isinstance(result, bool)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_factor_ids(self):
        """Test handling of empty factor list."""
        provider = MockFactorSetProvider()
        result = provider.get_factor_batch(factor_ids=[])

        # Should return empty values
        assert result["values"].shape == (10, 5, 0)
        assert result["factor_ids"] == []

    def test_duplicate_factor_ids(self):
        """Test handling of duplicate factors in set."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        factor_set = MockFactorSet(
            set_id="dup_set",
            name="Duplicate Set",
            factor_ids=("FACTOR_A", "FACTOR_A"),  # Duplicate
            created_at="2024-01-01T00:00:00",
        )

        result = adapter.load_factor_set(factor_set)
        # Should still work, returns requested shape
        assert result["values"].shape == (10, 5, 2)

    def test_missing_optional_attributes(self):
        """Test FactorSet without optional attributes."""
        provider = MockFactorSetProvider()
        adapter = FactorAssetsAdapter(provider)

        # Create minimal FactorSet
        factor_set = MockFactorSet(
            set_id="minimal",
            name="Minimal",
            factor_ids=("FACTOR_A",),
            created_at="2024-01-01T00:00:00",
            # No universe_ref or frequency
        )

        result = adapter.load_factor_set(factor_set)
        assert result["set_metadata"]["universe"] is None
        assert result["set_metadata"]["frequency"] is None
