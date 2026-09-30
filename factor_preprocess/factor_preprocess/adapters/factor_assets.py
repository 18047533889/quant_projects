"""
Factor Assets adapter - optional integration with factor_assets package.

Provides protocol-based boundary for converting FactorSet to preprocessing input.
The provider is injected by the caller; FactorAssets does not define a default
provider or own the factor-value read contract.
"""

from collections.abc import Sequence
from typing import Protocol, Dict, Any, Optional
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


def _validated_factor_ids(value: Any, source: str) -> list[str]:
    """Require an ordered sequence of usable factor ID strings."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{source} factor_ids must be a sequence of strings")
    factor_ids = list(value)
    if any(not isinstance(factor_id, str) for factor_id in factor_ids):
        raise ValueError(f"{source} factor_ids must contain only strings")
    if any(not factor_id for factor_id in factor_ids):
        raise ValueError(f"{source} factor_ids cannot contain empty strings")
    return factor_ids


class FactorAssetsAdapter:
    """
    Adapter for factor_assets package integration.

    Converts FactorSet objects to preprocessing-ready input arrays.
    Requires a caller-supplied provider; use :func:`create_adapter` for checks.
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
            set_id = getattr(factor_set, "set_id", "<unknown>")
            raise ValueError(f"Invalid factor set: {set_id}")
        factor_ids = _validated_factor_ids(
            getattr(factor_set, "factor_ids", None), "FactorSet"
        )
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("FactorSet contains duplicate factor_ids")

        # Extract factor IDs from the set
        # Use universe and frequency from the set if available
        universe = getattr(factor_set, 'universe_ref', None)

        # Load all factors in batch
        result = self._provider.get_factor_batch(
            factor_ids=factor_ids,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
        )

        # The third values axis is positional: a provider returning the right
        # shape with IDs in a different order would silently attach each
        # factor's values to the wrong identity downstream.
        if not isinstance(result, dict):
            raise ValueError("factor batch result must be a dictionary")
        if "factor_ids" not in result:
            raise ValueError("factor batch result is missing factor_ids")
        returned_ids = _validated_factor_ids(result["factor_ids"], "factor batch result")
        if len(set(returned_ids)) != len(returned_ids):
            raise ValueError("factor batch result contains duplicate factor_ids")
        if returned_ids != factor_ids:
            missing = [factor_id for factor_id in factor_ids if factor_id not in returned_ids]
            unexpected = [factor_id for factor_id in returned_ids if factor_id not in factor_ids]
            if missing or unexpected:
                raise ValueError(
                    "factor batch result factor_ids mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            raise ValueError(
                "factor batch result factor_ids order does not match requested order"
            )

        values = np.asarray(result.get("values"))
        if values.ndim != 3:
            raise ValueError(
                "factor batch values must have shape (n_times, n_assets, n_factors)"
            )
        if values.shape[2] != len(factor_ids):
            raise ValueError(
                "factor batch values factor axis does not match factor_ids: "
                f"{values.shape[2]} != {len(factor_ids)}"
            )
        for axis, key in ((0, "dates"), (1, "assets")):
            if key not in result:
                raise ValueError(f"factor batch result is missing {key}")
            if values.shape[axis] != len(result[key]):
                raise ValueError(
                    f"factor batch values axis {axis} does not match {key}: "
                    f"{values.shape[axis]} != {len(result[key])}"
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
    """Return whether the optional ``factor_assets`` package is importable."""
    try:
        import factor_assets
    except ImportError:
        return False
    return True


def create_adapter(provider: Optional[FactorSetProvider] = None) -> FactorAssetsAdapter:
    """
    Create an adapter from an injected provider; there is no default provider.

    Args:
        provider: Provider implementing :class:`FactorSetProvider`.

    Returns:
        FactorAssetsAdapter instance

    Raises:
        OptionalDependencyMissing: If ``factor_assets`` is not importable and
            no provider is supplied.
        ValueError: If ``factor_assets`` is available but no provider is supplied.
    """
    if provider is not None:
        return FactorAssetsAdapter(provider)

    if not check_factor_assets_available():
        raise OptionalDependencyMissing(
            package_name="factor_assets",
            feature_name="FactorSet integration"
        )

    raise ValueError(
        "A FactorSetProvider must be supplied; factor_assets does not expose a "
        "default batch-value provider. Configure and inject a provider explicitly."
    )


__all__ = [
    "FactorSetProvider",
    "FactorAssetsAdapter",
    "OptionalDependencyMissing",
    "check_factor_assets_available",
    "create_adapter",
]
