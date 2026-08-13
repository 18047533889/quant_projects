"""
DataAccess adapter for factor value and catalog integration.

Provides optional integration with DataAccess for reading factor values
and catalog metadata. This is NOT required for FA core functionality.

FA stores identity, lineage, and evidence references — not raw factor values.
This adapter enables optional reads when factor values are needed.

This is an OPTIONAL adapter — FA core does not depend on DA.
"""

from typing import Protocol, Optional, Dict, Any, Tuple
from datetime import date

from factor_assets.adapters import OptionalDependencyMissing


# DataAccess is not yet available, so we define the protocol
# Real implementation would import from data_access when available
DA_AVAILABLE = False


class FactorValueReader(Protocol):
    """
    Protocol for reading factor values from DataAccess.

    FA defines this protocol; adapters implement it for DA integration.
    FA never materializes or owns factor values — this is for optional reads.
    """

    def read_factor_values(
        self,
        factor_id: str,
        start_date: date,
        end_date: date,
        universe: Optional[str] = None,
    ) -> Any:
        """
        Read factor values for a time range.

        Args:
            factor_id: Factor identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            universe: Optional universe filter

        Returns:
            Factor values (implementation-specific type)
        """
        ...

    def check_factor_availability(
        self,
        factor_id: str,
        as_of_date: Optional[date] = None,
    ) -> bool:
        """
        Check if factor values are available.

        Args:
            factor_id: Factor identifier
            as_of_date: Optional as-of date for point-in-time check

        Returns:
            True if factor is available
        """
        ...


class CatalogReader(Protocol):
    """
    Protocol for reading catalog metadata from DataAccess.

    FA may use this to validate factor availability and metadata consistency.
    """

    def get_catalog_entry(
        self,
        factor_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get catalog entry for a factor.

        Args:
            factor_id: Factor identifier

        Returns:
            Catalog entry dict or None if not found
        """
        ...

    def list_available_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        """
        List available factors matching filters.

        Args:
            filters: Optional filter criteria

        Returns:
            Tuple of factor IDs
        """
        ...


class DAFactorValueReader:
    """
    FactorValueReader implementation for DataAccess.

    This adapter is OPTIONAL and only needed when FA needs to read
    factor values. Core FA functionality does not require this.
    """

    def __init__(self):
        """
        Initialize DA factor value reader.

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DAFactorValueReader")

        # Real implementation would initialize DA connection
        raise NotImplementedError(
            "DAFactorValueReader not yet implemented. "
            "Waiting for DataAccess package to be available."
        )

    def read_factor_values(
        self,
        factor_id: str,
        start_date: date,
        end_date: date,
        universe: Optional[str] = None,
    ) -> Any:
        """
        Read factor values through DataAccess.

        Args:
            factor_id: Factor identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            universe: Optional universe filter

        Returns:
            Factor values from DA

        Raises:
            OptionalDependencyMissing: If DA is not available
            NotImplementedError: DA integration not yet complete
        """
        raise NotImplementedError("DA integration pending")

    def check_factor_availability(
        self,
        factor_id: str,
        as_of_date: Optional[date] = None,
    ) -> bool:
        """
        Check factor availability through DataAccess.

        Args:
            factor_id: Factor identifier
            as_of_date: Optional as-of date

        Returns:
            True if available

        Raises:
            OptionalDependencyMissing: If DA is not available
            NotImplementedError: DA integration not yet complete
        """
        raise NotImplementedError("DA integration pending")


class DACatalogReader:
    """
    CatalogReader implementation for DataAccess.

    This adapter is OPTIONAL and only needed when FA needs to read
    catalog metadata. Core FA functionality does not require this.
    """

    def __init__(self):
        """
        Initialize DA catalog reader.

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DACatalogReader")

        raise NotImplementedError(
            "DACatalogReader not yet implemented. "
            "Waiting for DataAccess package to be available."
        )

    def get_catalog_entry(
        self,
        factor_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get catalog entry through DataAccess.

        Args:
            factor_id: Factor identifier

        Returns:
            Catalog entry or None

        Raises:
            OptionalDependencyMissing: If DA is not available
            NotImplementedError: DA integration not yet complete
        """
        raise NotImplementedError("DA integration pending")

    def list_available_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        """
        List available factors through DataAccess.

        Args:
            filters: Optional filter criteria

        Returns:
            Tuple of factor IDs

        Raises:
            OptionalDependencyMissing: If DA is not available
            NotImplementedError: DA integration not yet complete
        """
        raise NotImplementedError("DA integration pending")


__all__ = [
    "FactorValueReader",
    "CatalogReader",
    "DAFactorValueReader",
    "DACatalogReader",
]
