"""
Data Access adapter - optional integration with the data_access package.

Provides protocol-based boundary for fetching exposure context (industry, sector, size, etc.)
used in neutralization and feature engineering. Fails gracefully if not available.
"""

from copy import deepcopy
from collections.abc import Mapping
from types import MappingProxyType
import pandas as pd
from typing import Protocol, Dict, Any, Optional, List
import numpy as np


class OptionalDependencyMissing(Exception):
    """Raised when an optional dependency is required but not available."""

    def __init__(self, package_name: str, feature_name: str):
        self.package_name = package_name
        self.feature_name = feature_name
        super().__init__(
            f"Optional dependency '{package_name}' is required for {feature_name}. "
            f"Install it with: pip install {package_name}"
        )


# Exposure types the adapter understands natively. Anything outside this set is
# rejected (fail-closed) rather than silently routed to a custom fetch.
SUPPORTED_EXPOSURE_TYPES = frozenset({"industry", "size", "sector", "beta", "custom"})

# Provenance keys that MUST be present in every exposure bundle's metadata.
# Exposure provenance is mandatory, not optional: without it a bundle cannot be
# trusted for point-in-time neutralization.
REQUIRED_PROVENANCE_KEYS = (
    "knowledge_time",
    "effective_time",
    "snapshot_ref",
    "classification_version",
    "market",
    "universe_ref",
)


def _freeze_provenance(value):
    """Detach nested provider state; never retain a live metadata alias."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_provenance(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_provenance(item) for item in value)
    if isinstance(value, np.ndarray):
        return _freeze_provenance(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return value.item() if isinstance(value, np.generic) else value
    raise ValueError(f"Unsupported mutable exposure metadata type: {type(value).__name__}")


def _freeze_array(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        # Provider object arrays commonly contain string axes/categories.
        # Arbitrary object payloads cannot be certified immutable by NumPy flags.
        if all(isinstance(item, str) for item in value.flat):
            value = value.astype(str)
        else:
            raise ValueError("Exposure object arrays require scalar string values")
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _market_identity(value):
    normalized = str(value).strip().lower()
    return {"cn": "ashare", "a-share": "ashare", "a_share": "ashare"}.get(normalized, normalized)


def _utc_dates(value, label):
    try:
        dates = pd.to_datetime(value, utc=True, errors="raise")
        if np.asarray(pd.isna(dates)).any():
            raise ValueError("NaT")
        return dates
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid exposure {label}") from exc


def _validate_exposure_bundle(bundle: Any, exposure_type: str, *,
                              market=None, start_date=None, end_date=None,
                              assets=None, classification=None) -> Dict[str, Any]:
    """Validate an exposure bundle carries mandatory provenance metadata.

    Fail-closed: a bundle without the required provenance keys is rejected
    rather than silently passed through.

    Raises:
        ValueError: if the bundle is malformed or missing provenance.
    """
    request_assets = assets
    if not isinstance(bundle, dict):
        raise ValueError(
            f"Exposure provider returned non-dict for '{exposure_type}': {type(bundle).__name__}"
        )
    metadata = bundle.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Exposure bundle for '{exposure_type}' missing 'metadata' dict"
        )
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in metadata]
    if missing:
        raise ValueError(
            f"Exposure bundle for '{exposure_type}' missing required provenance "
            f"keys: {missing}"
        )
    for key in REQUIRED_PROVENANCE_KEYS:
        if not isinstance(metadata[key], str) or metadata[key].strip().lower() in ("", "unknown"):
            raise ValueError(
                f"Exposure bundle for '{exposure_type}' has unresolved provenance {key!r}"
            )
    values = bundle.get("values")
    dates = bundle.get("dates")
    assets = bundle.get("assets")
    if not all(isinstance(v, np.ndarray) for v in (values, dates, assets)):
        raise ValueError(f"Exposure bundle for '{exposure_type}' requires ndarray values/dates/assets")
    if values.ndim != 2 or dates.ndim != 1 or assets.ndim != 1:
        raise ValueError(f"Exposure bundle for '{exposure_type}' requires named TN axes")
    if values.shape != (len(dates), len(assets)):
        raise ValueError(
            f"Exposure bundle for '{exposure_type}' shape {values.shape} != "
            f"(dates={len(dates)}, assets={len(assets)})"
        )
    if len(set(dates.tolist())) != len(dates) or len(set(assets.tolist())) != len(assets):
        raise ValueError(f"Exposure bundle for '{exposure_type}' axes must be unique")
    # Request identity is checked before copying or reordering any values.
    if market is not None and _market_identity(metadata["market"]) != _market_identity(market):
        raise ValueError("Exposure market does not match request")
    if classification not in (None, "default"):
        if bundle.get("classification") != classification:
            raise ValueError("Exposure classification does not match request")
    requested_assets = request_assets
    if requested_assets is not None and list(assets) != list(requested_assets):
        raise ValueError("Exposure assets/order do not match request; explicit coverage alignment required")
    if market is not None or start_date is not None or end_date is not None:
        actual_dates = _utc_dates(dates, "dates")
        if not actual_dates.is_monotonic_increasing:
            raise ValueError("Exposure dates must be ordered")
        if start_date is not None and (actual_dates < _utc_dates(start_date, "start_date")).any():
            raise ValueError("Exposure dates precede requested start_date")
        if end_date is not None and (actual_dates > _utc_dates(end_date, "end_date")).any():
            raise ValueError("Exposure dates exceed requested end_date")
        # Scalar times are only dataset-level bounds, NOT row-level PIT proof.
        # Reject known-future bundles; a provider must supply per-cell authority
        # before historical point-in-time certification can be claimed.
        if len(actual_dates):
            latest = actual_dates[-1]
            for key in ("knowledge_time", "effective_time"):
                observed_time = _utc_dates(metadata[key], key)
                if not isinstance(observed_time, pd.Timestamp):
                    raise ValueError(f"Exposure {key} requires declared scalar-bound schema")
                if observed_time > latest:
                    raise ValueError(f"Exposure {key} is later than returned date range")
    # Snapshot at the boundary: callers cannot mutate provider buffers after validation.
    result = {key: deepcopy(value) for key, value in bundle.items() if key != "metadata"}
    result["metadata"] = _freeze_provenance(metadata)
    for key in ("values", "dates", "assets"):
        result[key] = _freeze_array(bundle[key])
    result["layout"] = "TN"
    result["dtype"] = str(result["values"].dtype)
    return result


class ExposureProvider(Protocol):
    """
    Protocol for exposure data providers.

    Defines the interface for fetching exposure context needed for neutralization.
    Implementations must provide exposure values aligned with factor data.

    Every returned exposure bundle is a dict that MUST contain, at minimum:
        - 'values': np.ndarray of shape (n_times, n_assets)
        - 'dates': np.ndarray of date strings
        - 'assets': np.ndarray of asset identifiers
        - 'metadata': dict with provenance. The metadata dict MUST contain the
          keys ``knowledge_time``, ``effective_time``, ``snapshot_ref``,
          ``classification_version``, ``market``, ``universe_ref``. Exposure
          provenance is mandatory, not optional: bundles missing these keys are
          rejected (fail-closed) by the adapter.
    """

    def get_industry_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        industry_classification: str = "default",
    ) -> Dict[str, Any]:
        """
        Retrieve industry classification exposures.

        Args:
            market: Market identifier (e.g., "ashare", "us")
            start_date: Optional start date filter
            end_date: Optional end date filter
            assets: Optional list of assets to filter
            industry_classification: Classification scheme (e.g., "GICS", "SW_L1")

        Returns:
            Dictionary with:
                - 'values': np.ndarray of shape (n_times, n_assets) with industry codes
                - 'dates': np.ndarray of date strings
                - 'assets': np.ndarray of asset identifiers
                - 'classification': str, the classification used
                - 'categories': list of unique industry codes
                - 'metadata': dict with source info
        """
        ...

    def get_size_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        size_metric: str = "market_cap",
    ) -> Dict[str, Any]:
        """
        Retrieve size exposures (market cap, log market cap, etc.).

        Args:
            market: Market identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            assets: Optional list of assets to filter
            size_metric: Metric to use ("market_cap", "log_market_cap", "total_assets")

        Returns:
            Dictionary with:
                - 'values': np.ndarray of shape (n_times, n_assets) with size values
                - 'dates': np.ndarray of date strings
                - 'assets': np.ndarray of asset identifiers
                - 'metric': str, the metric used
                - 'metadata': dict with source info
        """
        ...

    def get_sector_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        sector_classification: str = "default",
    ) -> Dict[str, Any]:
        """
        Retrieve sector classification exposures (broader than industry).

        Args:
            market: Market identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            assets: Optional list of assets to filter
            sector_classification: Classification scheme

        Returns:
            Dictionary with:
                - 'values': np.ndarray of shape (n_times, n_assets) with sector codes
                - 'dates': np.ndarray of date strings
                - 'assets': np.ndarray of asset identifiers
                - 'classification': str, the classification used
                - 'categories': list of unique sector codes
                - 'metadata': dict with source info
        """
        ...

    def get_beta_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        window_days: int = 252,
    ) -> Dict[str, Any]:
        """
        Retrieve market beta exposures.

        Args:
            market: Market identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            assets: Optional list of assets to filter
            window_days: Rolling window for beta calculation

        Returns:
            Dictionary with:
                - 'values': np.ndarray of shape (n_times, n_assets) with beta values
                - 'dates': np.ndarray of date strings
                - 'assets': np.ndarray of asset identifiers
                - 'window_days': int, the window used
                - 'metadata': dict with source info
        """
        ...

    def get_custom_exposure(
        self,
        exposure_name: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Retrieve custom exposure by name.

        Args:
            exposure_name: Name of exposure to fetch
            market: Market identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            assets: Optional list of assets to filter
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with:
                - 'values': np.ndarray of shape (n_times, n_assets)
                - 'dates': np.ndarray of date strings
                - 'assets': np.ndarray of asset identifiers
                - 'exposure_name': str
                - 'metadata': dict with source info
        """
        ...


class DataAccessAdapter:
    """
    Adapter for data_access package integration.

    Provides exposure context for neutralization and feature engineering.
    Raises OptionalDependencyMissing if data_access is not installed.
    """

    def __init__(self, provider: ExposureProvider):
        """
        Initialize adapter with a provider.

        Args:
            provider: Implementation of ExposureProvider protocol
        """
        self._provider = provider

    def fetch_industry_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        industry_classification: str = "default",
    ) -> Dict[str, Any]:
        """
        Fetch industry exposure for neutralization.

        Args:
            market: Market identifier (e.g., "ashare", "us")
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            assets: Optional asset filter
            industry_classification: Classification scheme

        Returns:
            Dictionary with industry exposure data

        Raises:
            OptionalDependencyMissing: If data_access not available
            ValueError: If the provider returns a bundle without mandatory
                provenance metadata (fail-closed).
        """
        assets = tuple(assets) if assets is not None else None
        bundle = self._provider.get_industry_exposure(
            market=market,
            start_date=start_date,
            end_date=end_date,
            assets=assets,
            industry_classification=industry_classification,
        )
        return _validate_exposure_bundle(bundle, "industry", market=market,
            start_date=start_date, end_date=end_date, assets=assets,
            classification=industry_classification)

    def fetch_size_exposure(
        self,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        size_metric: str = "market_cap",
    ) -> Dict[str, Any]:
        """
        Fetch size exposure for neutralization.

        Args:
            market: Market identifier
            start_date: Start date
            end_date: End date
            assets: Optional asset filter
            size_metric: Size metric to use

        Returns:
            Dictionary with size exposure data

        Raises:
            OptionalDependencyMissing: If data_access not available
            ValueError: If the provider returns a bundle without mandatory
                provenance metadata (fail-closed).
        """
        assets = tuple(assets) if assets is not None else None
        bundle = self._provider.get_size_exposure(
            market=market,
            start_date=start_date,
            end_date=end_date,
            assets=assets,
            size_metric=size_metric,
        )
        return _validate_exposure_bundle(bundle, "size", market=market,
            start_date=start_date, end_date=end_date, assets=assets)

    def fetch_multi_exposure(
        self,
        market: str,
        exposure_types: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple exposures in one call for efficiency.

        Args:
            market: Market identifier
            exposure_types: List of exposure names (e.g., ["industry", "size", "beta"])
            start_date: Start date
            end_date: End date
            assets: Optional asset filter
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary mapping exposure_type -> exposure data dict

        Raises:
            OptionalDependencyMissing: If data_access not available
            ValueError: If an exposure_type is not in the supported set, or a
                provider returns a bundle without mandatory provenance metadata
                (fail-closed).
        """
        assets = tuple(assets) if assets is not None else None
        result = {}

        for exp_type in exposure_types:
            if exp_type not in SUPPORTED_EXPOSURE_TYPES:
                raise ValueError(
                    f"Unknown exposure type '{exp_type}'. Supported types: "
                    f"{sorted(SUPPORTED_EXPOSURE_TYPES)}"
                )
            if exp_type == "industry":
                result["industry"] = self.fetch_industry_exposure(
                    market, start_date, end_date, assets
                )
            elif exp_type == "size":
                result["size"] = self.fetch_size_exposure(
                    market, start_date, end_date, assets
                )
            elif exp_type == "sector":
                result["sector"] = _validate_exposure_bundle(
                    self._provider.get_sector_exposure(
                        market, start_date, end_date, assets
                    ),
                    "sector",
                    market=market, start_date=start_date, end_date=end_date, assets=assets,
                )
            elif exp_type == "beta":
                result["beta"] = _validate_exposure_bundle(
                    self._provider.get_beta_exposure(
                        market, start_date, end_date, assets
                    ),
                    "beta",
                    market=market, start_date=start_date, end_date=end_date, assets=assets,
                )
            else:
                # Custom exposure
                result[exp_type] = _validate_exposure_bundle(
                    self._provider.get_custom_exposure(
                        exposure_name=exp_type,
                        market=market,
                        start_date=start_date,
                        end_date=end_date,
                        assets=assets,
                        **kwargs,
                    ),
                    exp_type,
                    market=market, start_date=start_date, end_date=end_date, assets=assets,
                )

        return result


def check_data_access_available() -> bool:
    """Check if the data_access package is available."""
    try:
        import data_access
        return True
    except ImportError:
        return False


def create_adapter(provider: Optional[ExposureProvider] = None) -> DataAccessAdapter:
    """
    Create a DataAccessAdapter instance.

    Args:
        provider: Optional custom provider; if None, tries to create default

    Returns:
        DataAccessAdapter instance

    Raises:
        OptionalDependencyMissing: If data_access not available and no provider given
    """
    if provider is not None:
        return DataAccessAdapter(provider)

    if not check_data_access_available():
        raise OptionalDependencyMissing(
            package_name="data_access",
            feature_name="Exposure context for neutralization"
        )

    # Import and create default provider
    from factor_preprocess.adapters._data_access_impl import DefaultExposureProvider
    provider = DefaultExposureProvider()
    return DataAccessAdapter(provider)


__all__ = [
    "ExposureProvider",
    "DataAccessAdapter",
    "OptionalDependencyMissing",
    "check_data_access_available",
    "create_adapter",
    "SUPPORTED_EXPOSURE_TYPES",
    "REQUIRED_PROVENANCE_KEYS",
    "_validate_exposure_bundle",
]
