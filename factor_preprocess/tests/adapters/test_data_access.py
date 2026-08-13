"""
Tests for data_access adapter.

Tests protocol compliance, mock providers, and graceful failure.
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional, List

from factor_preprocess.adapters.data_access import (
    ExposureProvider,
    DataAccessAdapter,
    OptionalDependencyMissing,
    check_data_access_available,
    create_adapter,
)


class MockExposureProvider:
    """Mock provider for testing."""

    def __init__(self):
        self.call_log = []
        # Mock data: 10 dates x 5 assets
        self.n_dates = 10
        self.n_assets = 5
        self.dates = np.array([f"2024-01-{i+1:02d}" for i in range(self.n_dates)])
        self.assets = np.array([f"ASSET_{i}" for i in range(self.n_assets)])

    def get_industry_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        industry_classification: str = "default",
    ) -> Dict[str, Any]:
        self.call_log.append(("get_industry_exposure", market, industry_classification))

        # Mock industry codes: repeating pattern
        industry_codes = np.array([f"IND_{i % 3}" for i in range(self.n_assets)])
        values = np.tile(industry_codes, (self.n_dates, 1))

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "classification": industry_classification,
            "categories": ["IND_0", "IND_1", "IND_2"],
            "metadata": {
                "source": "mock",
                "market": market,
            },
        }

    def get_size_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        size_metric: str = "market_cap",
    ) -> Dict[str, Any]:
        self.call_log.append(("get_size_exposure", market, size_metric))

        # Mock size values: random market caps
        values = np.random.lognormal(mean=20, sigma=2, size=(self.n_dates, self.n_assets))

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "metric": size_metric,
            "metadata": {
                "source": "mock",
                "market": market,
            },
        }

    def get_sector_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        sector_classification: str = "default",
    ) -> Dict[str, Any]:
        self.call_log.append(("get_sector_exposure", market, sector_classification))

        # Mock sector codes
        sector_codes = np.array([f"SEC_{i % 2}" for i in range(self.n_assets)])
        values = np.tile(sector_codes, (self.n_dates, 1))

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "classification": sector_classification,
            "categories": ["SEC_0", "SEC_1"],
            "metadata": {
                "source": "mock",
                "market": market,
            },
        }

    def get_beta_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        window_days: int = 252,
    ) -> Dict[str, Any]:
        self.call_log.append(("get_beta_exposure", market, window_days))

        # Mock beta values: centered around 1.0
        values = np.random.normal(loc=1.0, scale=0.3, size=(self.n_dates, self.n_assets))

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "window_days": window_days,
            "metadata": {
                "source": "mock",
                "market": market,
            },
        }

    def get_custom_exposure(
        self,
        exposure_name: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.call_log.append(("get_custom_exposure", exposure_name, market))

        # Mock custom exposure
        values = np.random.randn(self.n_dates, self.n_assets)

        return {
            "values": values,
            "dates": self.dates,
            "assets": self.assets,
            "exposure_name": exposure_name,
            "metadata": {
                "source": "mock",
                "market": market,
                **kwargs,
            },
        }


class TestExposureProvider:
    """Test the protocol definition."""

    def test_mock_provider_implements_protocol(self):
        """Mock provider should satisfy the protocol."""
        provider = MockExposureProvider()

        # Should have required methods
        assert hasattr(provider, "get_industry_exposure")
        assert hasattr(provider, "get_size_exposure")
        assert hasattr(provider, "get_sector_exposure")
        assert hasattr(provider, "get_beta_exposure")
        assert hasattr(provider, "get_custom_exposure")

    def test_mock_provider_get_industry_exposure(self):
        """Test industry exposure retrieval."""
        provider = MockExposureProvider()
        result = provider.get_industry_exposure(market="ashare")

        assert "values" in result
        assert "dates" in result
        assert "assets" in result
        assert "classification" in result
        assert "categories" in result
        assert "metadata" in result
        assert result["values"].shape == (10, 5)

    def test_mock_provider_get_size_exposure(self):
        """Test size exposure retrieval."""
        provider = MockExposureProvider()
        result = provider.get_size_exposure(market="ashare", size_metric="log_market_cap")

        assert "values" in result
        assert "metric" in result
        assert result["metric"] == "log_market_cap"
        assert result["values"].shape == (10, 5)

    def test_mock_provider_get_sector_exposure(self):
        """Test sector exposure retrieval."""
        provider = MockExposureProvider()
        result = provider.get_sector_exposure(market="us")

        assert "values" in result
        assert "categories" in result
        assert len(result["categories"]) > 0

    def test_mock_provider_get_beta_exposure(self):
        """Test beta exposure retrieval."""
        provider = MockExposureProvider()
        result = provider.get_beta_exposure(market="us", window_days=126)

        assert "values" in result
        assert "window_days" in result
        assert result["window_days"] == 126

    def test_mock_provider_get_custom_exposure(self):
        """Test custom exposure retrieval."""
        provider = MockExposureProvider()
        result = provider.get_custom_exposure(
            exposure_name="momentum",
            market="ashare",
            lookback_days=20,
        )

        assert "values" in result
        assert "exposure_name" in result
        assert result["exposure_name"] == "momentum"
        assert "lookback_days" in result["metadata"]


class TestDataAccessAdapter:
    """Test the adapter implementation."""

    def test_adapter_initialization(self):
        """Adapter should accept a provider."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)
        assert adapter is not None

    def test_fetch_industry_exposure(self):
        """Test fetching industry exposure."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_industry_exposure(
            market="ashare",
            start_date="2024-01-01",
            end_date="2024-01-10",
            industry_classification="SW_L1",
        )

        assert "values" in result
        assert "classification" in result
        assert result["classification"] == "SW_L1"
        assert ("get_industry_exposure", "ashare", "SW_L1") in provider.call_log

    def test_fetch_size_exposure(self):
        """Test fetching size exposure."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_size_exposure(
            market="us",
            size_metric="log_market_cap",
        )

        assert "values" in result
        assert "metric" in result
        assert result["metric"] == "log_market_cap"

    def test_fetch_multi_exposure(self):
        """Test fetching multiple exposures."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="ashare",
            exposure_types=["industry", "size", "beta"],
        )

        assert "industry" in result
        assert "size" in result
        assert "beta" in result

        # Verify each exposure has correct structure
        assert "values" in result["industry"]
        assert "values" in result["size"]
        assert "values" in result["beta"]

    def test_fetch_multi_exposure_with_custom(self):
        """Test multi-exposure fetch with custom exposure."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="us",
            exposure_types=["industry", "momentum"],  # momentum is custom
        )

        assert "industry" in result
        assert "momentum" in result
        assert result["momentum"]["exposure_name"] == "momentum"

    def test_fetch_sector_exposure_through_multi(self):
        """Test sector exposure via multi-fetch."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="us",
            exposure_types=["sector"],
        )

        assert "sector" in result
        assert "categories" in result["sector"]


class TestOptionalDependency:
    """Test optional dependency handling."""

    def test_check_data_access_available(self):
        """Test availability check."""
        # This will be False unless dataaccess is installed
        available = check_data_access_available()
        assert isinstance(available, bool)

    def test_optional_dependency_missing_exception(self):
        """Test exception structure."""
        exc = OptionalDependencyMissing("dataaccess", "Exposure context")

        assert exc.package_name == "dataaccess"
        assert exc.feature_name == "Exposure context"
        assert "dataaccess" in str(exc)
        assert "pip install" in str(exc)

    def test_create_adapter_with_provider(self):
        """Test create_adapter with explicit provider."""
        provider = MockExposureProvider()
        adapter = create_adapter(provider=provider)

        assert isinstance(adapter, DataAccessAdapter)

    def test_create_adapter_without_dataaccess(self):
        """Test create_adapter fails gracefully without dataaccess."""
        # Only test if dataaccess is NOT installed
        if not check_data_access_available():
            with pytest.raises(OptionalDependencyMissing) as exc_info:
                create_adapter()

            assert exc_info.value.package_name == "dataaccess"


class TestProtocolCompliance:
    """Test that mock provider matches the protocol."""

    def test_provider_signature_matches_protocol(self):
        """Verify all protocol methods are implemented."""
        provider = MockExposureProvider()

        # Test all method signatures
        result = provider.get_industry_exposure(
            market="ashare",
            start_date=None,
            end_date=None,
            assets=None,
            industry_classification="default",
        )
        assert isinstance(result, dict)

        result = provider.get_size_exposure(
            market="ashare",
            start_date=None,
            end_date=None,
            assets=None,
            size_metric="market_cap",
        )
        assert isinstance(result, dict)

        result = provider.get_sector_exposure(
            market="us",
            start_date=None,
            end_date=None,
            assets=None,
            sector_classification="default",
        )
        assert isinstance(result, dict)

        result = provider.get_beta_exposure(
            market="us",
            start_date=None,
            end_date=None,
            assets=None,
            window_days=252,
        )
        assert isinstance(result, dict)

        result = provider.get_custom_exposure(
            exposure_name="custom",
            market="ashare",
            start_date=None,
            end_date=None,
            assets=None,
        )
        assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_exposure_types_list(self):
        """Test multi-fetch with empty list."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="ashare",
            exposure_types=[],
        )

        assert result == {}

    def test_unknown_market(self):
        """Test with unknown market identifier."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        # Should still work, provider decides behavior
        result = adapter.fetch_industry_exposure(market="unknown")
        assert result["metadata"]["market"] == "unknown"

    def test_asset_filtering(self):
        """Test with asset filter."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_industry_exposure(
            market="ashare",
            assets=["ASSET_0", "ASSET_1"],
        )

        # Provider received the call (behavior is provider-specific)
        assert ("get_industry_exposure", "ashare", "default") in provider.call_log

    def test_date_range_filtering(self):
        """Test with date range."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_size_exposure(
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-05",
        )

        assert result is not None

    def test_custom_kwargs_passed_through(self):
        """Test that custom kwargs are passed to provider."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="ashare",
            exposure_types=["custom_exp"],
            custom_param="value",
        )

        # Custom exposure should have received kwargs
        assert "custom_exp" in result
        assert "custom_param" in result["custom_exp"]["metadata"]


class TestMultipleMarkets:
    """Test adapter behavior across different markets."""

    def test_ashare_market(self):
        """Test A-share market exposures."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="ashare",
            exposure_types=["industry", "size"],
        )

        assert result["industry"]["metadata"]["market"] == "ashare"
        assert result["size"]["metadata"]["market"] == "ashare"

    def test_us_market(self):
        """Test US market exposures."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        result = adapter.fetch_multi_exposure(
            market="us",
            exposure_types=["sector", "beta"],
        )

        assert result["sector"]["metadata"]["market"] == "us"
        assert result["beta"]["metadata"]["market"] == "us"

    def test_different_classification_schemes(self):
        """Test different industry classification schemes."""
        provider = MockExposureProvider()
        adapter = DataAccessAdapter(provider)

        # Test SW_L1 (A-share)
        result_sw = adapter.fetch_industry_exposure(
            market="ashare",
            industry_classification="SW_L1",
        )
        assert result_sw["classification"] == "SW_L1"

        # Test GICS (US)
        result_gics = adapter.fetch_industry_exposure(
            market="us",
            industry_classification="GICS",
        )
        assert result_gics["classification"] == "GICS"
