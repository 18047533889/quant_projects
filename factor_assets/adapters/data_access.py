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


# Try to import DataAccess - use lazy import to avoid module conflicts
DA_AVAILABLE = False
_DataAccessStore = None
_get_store = None
_ReadHandle = None
_DataRequest = None

def _try_import_da():
    """Lazy import DataAccess to avoid conflicts."""
    global DA_AVAILABLE, _DataAccessStore, _get_store, _ReadHandle, _DataRequest
    if DA_AVAILABLE or _DataAccessStore is not None:
        return
    try:
        from dataaccess import DataAccessStore, get_store, ReadHandle, DataRequest
        DA_AVAILABLE = True
        _DataAccessStore = DataAccessStore
        _get_store = get_store
        _ReadHandle = ReadHandle
        _DataRequest = DataRequest
    except (ImportError, TypeError):
        # TypeError can occur due to dataclass conflicts
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

    def __init__(self, store: Optional["DataAccessStore"] = None):
        """
        Initialize DA factor value reader.

        Args:
            store: Optional DataAccessStore instance (uses get_store() if None)

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        _try_import_da()
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DAFactorValueReader")

        self._store = store if store is not None else _get_store()

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
            ValueError: If factor not found or request invalid
        """
        # Create read request for factor
        request = _DataRequest(
            source=factor_id,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
        )

        # Execute read through DA store
        handle = self._store.read(request)
        return handle.get_result()

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
        """
        try:
            # Attempt minimal read to check availability
            request = _DataRequest(
                source=factor_id,
                start_date=as_of_date or date.today(),
                end_date=as_of_date or date.today(),
            )
            handle = self._store.read(request)
            # If we can get a handle, factor is available
            return True
        except Exception:
            return False


class DACatalogReader:
    """
    CatalogReader implementation for DataAccess.

    This adapter is OPTIONAL and only needed when FA needs to read
    catalog metadata. Core FA functionality does not require this.
    """

    def __init__(self, store: Optional["DataAccessStore"] = None):
        """
        Initialize DA catalog reader.

        Args:
            store: Optional DataAccessStore instance (uses get_store() if None)

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        _try_import_da()
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DACatalogReader")

        self._store = store if store is not None else _get_store()

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
        """
        try:
            # Query DA semantic catalog or registry for factor metadata
            # For now, return basic availability info
            request = _DataRequest(source=factor_id, start_date=date.today(), end_date=date.today())
            handle = self._store.read(request)
            return {
                "factor_id": factor_id,
                "available": True,
                "source": factor_id,
            }
        except Exception:
            return None

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
        """
        # DataAccess doesn't expose a direct catalog listing API
        # This would need to query the semantic catalog or registry
        # For now, return empty tuple as we don't have catalog enumeration
        return ()


__all__ = [
    "FactorValueReader",
    "CatalogReader",
    "DAFactorValueReader",
    "DACatalogReader",
]
