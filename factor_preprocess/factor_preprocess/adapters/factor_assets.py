"""
Factor Assets adapter - optional integration with factor_assets package.

Provides protocol-based boundary for converting FactorSet to preprocessing input.
Fails gracefully if factor_assets is not available.
"""

from typing import Protocol, Dict, Any, Optional
from datetime import datetime
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


class FactorSetProvider(Protocol):
    """
    Protocol for factor set providers.

    Defines the interface for converting a FactorSet into preprocessing input.
    Implementations must provide factor values as arrays ready for transformation.
    """

    def get_factor_values(
        self,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve factor values for a single factor.

        Returns:
            Dictionary with keys:
                - 'values': np.ndarray of shape (n_times, n_assets)
                - 'dates': np.ndarray of date strings/timestamps
                - 'assets': np.ndarray of asset identifiers
                - 'metadata': dict with factor metadata
        """
        ...

    def get_factor_batch(
        self,
        factor_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve factor values for multiple factors efficiently.

        Returns:
            Dictionary with keys:
                - 'values': np.ndarray of shape (n_times, n_assets, n_factors)
                - 'dates': np.ndarray of date strings/timestamps
                - 'assets': np.ndarray of asset identifiers
                - 'factor_ids': list of factor IDs (same order as values axis 2)
                - 'metadata': dict mapping factor_id -> metadata
        """
        ...

    def validate_factor_set(self, factor_set: Any) -> bool:
        """
        Validate that a FactorSet is well-formed and factors exist.

        Args:
            factor_set: FactorSet object from factor_assets

        Returns:
            True if valid, raises ValueError otherwise
        """
        ...


class FactorAssetsAdapter:
    """
    Adapter for factor_assets package integration.

    Converts FactorSet objects to preprocessing-ready input arrays.
    Raises OptionalDependencyMissing if factor_assets is not installed.
    """

    def __init__(self, provider: FactorSetProvider):
        """
        Initialize adapter with a provider.

        Args:
            provider: Implementation of FactorSetProvider protocol
        """
        self._provider = provider

    def load_factor_set(
        self,
        factor_set: Any,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load factor values from a FactorSet.

        Args:
            factor_set: FactorSet object from factor_assets
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)

        Returns:
            Dictionary with:
                - 'values': np.ndarray (n_times, n_assets, n_factors)
                - 'dates': np.ndarray of dates
                - 'assets': np.ndarray of asset IDs
                - 'factor_ids': list of factor IDs
                - 'metadata': dict of factor metadata
                - 'set_metadata': FactorSet metadata

        Raises:
            OptionalDependencyMissing: If factor_assets not available
            ValueError: If factor_set is invalid
        """
        # Validate the factor set
        if not self._provider.validate_factor_set(factor_set):
            raise ValueError(f"Invalid factor set: {factor_set.set_id}")

        # Extract factor IDs from the set
        factor_ids = list(factor_set.factor_ids)

        # Use universe and frequency from the set if available
        universe = getattr(factor_set, 'universe_ref', None)

        # Load all factors in batch
        result = self._provider.get_factor_batch(
            factor_ids=factor_ids,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
        )

        # Add set-level metadata
        result['set_metadata'] = {
            'set_id': factor_set.set_id,
            'set_name': factor_set.name,
            'created_at': factor_set.created_at,
            'frequency': getattr(factor_set, 'frequency', None),
            'universe': universe,
            'n_factors': factor_set.size,
        }

        return result

    def load_single_factor(
        self,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a single factor by ID.

        Args:
            factor_id: Factor identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            universe: Optional universe filter

        Returns:
            Dictionary with factor values and metadata

        Raises:
            OptionalDependencyMissing: If factor_assets not available
        """
        return self._provider.get_factor_values(
            factor_id=factor_id,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
        )


def check_factor_assets_available() -> bool:
    """Check whether the factor_assets integration can construct its default provider."""
    try:
        import factor_assets
        from factor_preprocess.adapters._factor_assets_impl import DefaultFactorSetProvider
    except ImportError:
        return False
    return True


def create_adapter(provider: Optional[FactorSetProvider] = None) -> FactorAssetsAdapter:
    """
    Create a FactorAssetsAdapter instance.

    Args:
        provider: Optional custom provider; if None, tries to create default

    Returns:
        FactorAssetsAdapter instance

    Raises:
        OptionalDependencyMissing: If factor_assets not available and no provider given
    """
    if provider is not None:
        return FactorAssetsAdapter(provider)

    if not check_factor_assets_available():
        raise OptionalDependencyMissing(
            package_name="factor_assets",
            feature_name="FactorSet integration"
        )

    # Import and create default provider
    from factor_preprocess.adapters._factor_assets_impl import DefaultFactorSetProvider
    provider = DefaultFactorSetProvider()
    return FactorAssetsAdapter(provider)


__all__ = [
    "FactorSetProvider",
    "FactorAssetsAdapter",
    "OptionalDependencyMissing",
    "check_factor_assets_available",
    "create_adapter",
]
