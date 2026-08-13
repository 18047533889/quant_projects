"""
DataAccess adapter for quant_evaluator (optional dependency).

This adapter provides Protocol-based boundaries for DataAccess integration.
Core quant_evaluator modules MUST NOT import this module.

When to use:
- Converting DA reads to FactorBatch/LabelBundle
- Accessing DA calendar/universe through standard protocols
- Optional DA get_store() integration

Lazy import pattern ensures core package remains usable without DA installed.
"""

from typing import Protocol, Any, Optional, Tuple, List
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


class ContextProvider(Protocol):
    """
    Protocol for providing calendar/timing context.

    Implementations may delegate to DA calendar or provide custom logic.
    """

    def get_trading_dates(
        self,
        start: Any,
        end: Any,
        market: str = "CN"
    ) -> Tuple[Any, ...]:
        """Return trading dates in [start, end]."""
        ...

    def is_trading_day(self, date: Any, market: str = "CN") -> bool:
        """Check if date is a trading day."""
        ...

    def next_trading_day(self, date: Any, market: str = "CN") -> Any:
        """Get next trading day after date."""
        ...


class UniverseProvider(Protocol):
    """
    Protocol for providing universe/asset lists.

    Implementations may delegate to DA universe or provide custom logic.
    """

    def get_universe(
        self,
        date: Any,
        universe_id: str,
        market: str = "CN"
    ) -> Tuple[str, ...]:
        """Return asset codes in universe at date."""
        ...

    def is_in_universe(
        self,
        asset: str,
        date: Any,
        universe_id: str,
        market: str = "CN"
    ) -> bool:
        """Check if asset is in universe at date."""
        ...


class DataAccessAdapter:
    """
    Adapter for DataAccess integration (optional dependency).

    Provides conversion from DA reads to QE contracts (FactorBatch, LabelBundle).
    Fails with OptionalDependencyMissing if DataAccess is not installed.

    Design principles:
    - Protocol-based boundaries; no direct DA type dependencies
    - Lazy import; core QE never imports this
    - Explicit timing; never infers dates from data
    - Fail-closed; raises on ambiguous conversions

    Usage:
        adapter = DataAccessAdapter()
        batch = adapter.da_frame_to_factor_batch(
            df,
            factor_ids=["momentum", "value"],
            time_col="date",
            asset_col="code"
        )
    """

    def __init__(self):
        """
        Initialize adapter and verify DataAccess is available.

        Raises:
            OptionalDependencyMissing: If dataaccess package not installed
        """
        try:
            import dataaccess  # noqa: F401
            self._da_module = dataaccess
        except ImportError as e:
            raise OptionalDependencyMissing(
                "DataAccess is not installed. "
                "Install dataaccess package to use this adapter."
            ) from e

    def da_frame_to_factor_batch(
        self,
        df: Any,
        factor_ids: List[str],
        time_col: str = "date",
        asset_col: str = "code",
        validity_col: Optional[str] = None,
        context_refs: Optional[dict] = None,
    ) -> FactorBatch:
        """
        Convert DataAccess DataFrame to FactorBatch.

        Args:
            df: DA DataFrame with columns [time_col, asset_col, *factor_ids]
            factor_ids: Factor column names to extract
            time_col: Time axis column name
            asset_col: Asset axis column name
            validity_col: Optional validity mask column
            context_refs: Optional context metadata

        Returns:
            FactorBatch with explicit axes and values

        Raises:
            ValueError: If required columns missing or shape invalid
            OptionalDependencyMissing: If DA not available

        Notes:
            - Assumes df is already pivoted or in long format
            - Time and asset axes are extracted and sorted
            - Missing values become NaN in values array
            - Validity mask is optional; if None, derived from finite values
        """
        # This is a reference implementation stub
        # Full implementation requires understanding DA DataFrame contract
        raise NotImplementedError(
            "da_frame_to_factor_batch requires DA DataFrame contract freeze. "
            "Use pandas adapter for reference/debug workflows."
        )

    def da_frame_to_label_bundle(
        self,
        df: Any,
        target_id: str,
        horizon: int,
        execution_delay: int = 0,
        decision_time_col: str = "decision_time",
        label_start_col: str = "label_start",
        label_end_col: str = "label_end",
        value_col: str = "label_value",
        validity_col: Optional[str] = None,
        calendar_ref: Optional[str] = None,
    ) -> LabelBundle:
        """
        Convert DataAccess DataFrame to LabelBundle with explicit timing.

        Args:
            df: DA DataFrame with timing columns and label values
            target_id: Label identifier (e.g., "ret_5d_vwap")
            horizon: Forward horizon in calendar units
            execution_delay: Execution delay after decision
            decision_time_col: Column for decision timestamp
            label_start_col: Column for label window start
            label_end_col: Column for label window end
            value_col: Column for label values
            validity_col: Optional validity mask column
            calendar_ref: Optional calendar reference

        Returns:
            LabelBundle with explicit timing vectors

        Raises:
            ValueError: If timing columns missing or horizon invalid
            OptionalDependencyMissing: If DA not available

        Notes:
            - All timing is explicit; never inferred from data
            - Horizon must be positive
            - Execution delay must be non-negative
            - Timing columns must be sortable and aligned
        """
        # This is a reference implementation stub
        # Full implementation requires understanding DA timing contract
        raise NotImplementedError(
            "da_frame_to_label_bundle requires DA timing contract freeze. "
            "Use explicit LabelBundle construction for now."
        )

    def get_context_provider(self) -> ContextProvider:
        """
        Get DA-backed ContextProvider for calendar/timing queries.

        Returns:
            ContextProvider delegating to DA calendar

        Raises:
            OptionalDependencyMissing: If DA not available
            NotImplementedError: If DA calendar API not frozen
        """
        raise NotImplementedError(
            "DA ContextProvider requires DA calendar API freeze."
        )

    def get_universe_provider(self) -> UniverseProvider:
        """
        Get DA-backed UniverseProvider for universe queries.

        Returns:
            UniverseProvider delegating to DA universe

        Raises:
            OptionalDependencyMissing: If DA not available
            NotImplementedError: If DA universe API not frozen
        """
        raise NotImplementedError(
            "DA UniverseProvider requires DA universe API freeze."
        )
