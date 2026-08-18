"""
FactorEngine adapter for quant_evaluator (optional dependency).

This adapter provides Protocol-based boundaries for FactorEngine integration.
Core quant_evaluator modules MUST NOT import this module.

When to use:
- Converting FE execution results to FactorBatch
- Extracting canonical factor identity from FE Factor/Expr
- Obtaining factor complexity/cost estimates

Lazy import pattern ensures core package remains usable without FE installed.
"""

from typing import Protocol, Any, Optional, List, Tuple
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef


class FactorBatchProvider(Protocol):
    """
    Protocol for providing factor values in batch.

    Implementations may delegate to FE execution or provide pre-computed values.
    """

    def compute_batch(
        self,
        factor_exprs: List[Any],
        start_date: Any,
        end_date: Any,
        universe: Tuple[str, ...],
        market: str = "CN"
    ) -> FactorBatch:
        """
        Compute factor values for date range and universe.

        Returns:
            FactorBatch with computed values and validity
        """
        ...


class FactorIdentityProvider(Protocol):
    """
    Protocol for extracting canonical factor identity.

    Implementations delegate to FE canonical identity system.
    """

    def get_factor_id(self, factor_expr: Any) -> str:
        """
        Get canonical factor ID from FE expression.

        Returns:
            Canonical factor ID string (stable hash or semantic digest)
        """
        ...

    def get_structural_id(self, factor_expr: Any) -> str:
        """
        Get structural factor ID (ignoring parameter values).

        Returns:
            Structural ID for grouping factor families
        """
        ...


class FactorEngineAdapter:
    """
    Adapter for FactorEngine integration (optional dependency).

    Provides conversion from FE execution results to QE FactorBatch contract.
    Fails with OptionalDependencyMissing if FactorEngine is not installed.

    Design principles:
    - Protocol-based boundaries; no direct FE type dependencies in signatures
    - Lazy import; core QE never imports this
    - Delegates to FE public API only (api, runtime, ir)
    - Never duplicates FE kernels or identity logic

    Usage:
        adapter = FactorEngineAdapter()
        batch = adapter.fe_result_to_factor_batch(
            fe_result,
            factor_ids=["momentum_20d", "value_bm"]
        )
    """

    def __init__(self):
        """
        Initialize adapter and verify FactorEngine is available.

        Raises:
            OptionalDependencyMissing: If factor_engine package not installed
        """
        try:
            # Import FE public API modules only
            from api import factor as fe_factor  # noqa: F401
            from runtime import FactorEngine as fe_engine  # noqa: F401

            self._fe_factor = fe_factor
            self._fe_engine = fe_engine
        except ImportError as e:
            raise OptionalDependencyMissing(
                "FactorEngine is not installed. "
                "Install factor_engine packages to use this adapter."
            ) from e

    def fe_result_to_factor_batch(
        self,
        fe_result: Any,
        factor_ids: Optional[List[str]] = None,
        extract_validity: bool = True,
        context_refs: Optional[dict] = None,
    ) -> FactorBatch:
        """
        Convert FactorEngine execution result to FactorBatch.

        Args:
            fe_result: FE execution result (DataFrame or MaterializedResult)
            factor_ids: Explicit factor IDs; if None, extracted from result
            extract_validity: Whether to extract validity mask from result
            context_refs: Optional context metadata

        Returns:
            FactorBatch with explicit axes and values

        Raises:
            ValueError: If result format invalid or axes missing
            OptionalDependencyMissing: If FE not available

        Notes:
            - Assumes fe_result has time/asset axes and factor columns
            - Time axis extracted from result index or explicit column
            - Asset axis extracted from result columns or multi-index
            - Validity derived from FE validity mask or finite value check
            - Factor IDs may be canonical hashes or user-provided names
        """
        # This is a reference implementation stub
        # Full implementation requires FE MaterializedResult contract freeze
        raise NotImplementedError(
            "fe_result_to_factor_batch requires FE MaterializedResult contract freeze. "
            "Use pandas adapter for reference/debug workflows."
        )

    def fe_factors_to_factor_batch(
        self,
        factor_exprs: List[Any],
        start_date: Any,
        end_date: Any,
        universe: Tuple[str, ...],
        market: str = "CN",
        extract_ids: bool = True,
        context_refs: Optional[dict] = None,
    ) -> FactorBatch:
        """
        Execute FE factors and convert to FactorBatch.

        Args:
            factor_exprs: List of FE Factor/Expr objects
            start_date: Start date for evaluation
            end_date: End date for evaluation
            universe: Asset universe tuple
            market: Market identifier
            extract_ids: Whether to extract canonical IDs from exprs
            context_refs: Optional context metadata

        Returns:
            FactorBatch with executed values

        Raises:
            ValueError: If execution fails or result invalid
            OptionalDependencyMissing: If FE not available

        Notes:
            - Delegates to FE execution runtime
            - Does not duplicate FE operator kernels
            - Timing context must be explicit
            - Universe must be pre-resolved (no implicit universe logic)
        """
        # This is a reference implementation stub
        # Full implementation requires FE execution API freeze
        raise NotImplementedError(
            "fe_factors_to_factor_batch requires FE execution API freeze. "
            "Use FE runtime directly and manual FactorBatch construction for now."
        )

    def extract_factor_id(self, factor_expr: Any) -> str:
        """
        Extract canonical factor ID from FE Factor/Expr.

        Args:
            factor_expr: FE Factor or Expr object

        Returns:
            Canonical factor ID (stable hash or semantic digest)

        Raises:
            ValueError: If factor_expr is not a valid FE expression
            OptionalDependencyMissing: If FE not available

        Notes:
            - Delegates to FE canonical identity system
            - Never duplicates FE identity logic
            - Returns stable hash across FE versions (when possible)
            - Structural vs semantic ID distinction handled by FE
        """
        # This is a reference implementation stub
        # Full implementation requires FE identity API freeze
        raise NotImplementedError(
            "extract_factor_id requires FE canonical identity API freeze. "
            "Use FE identity module directly for now."
        )

    def extract_structural_id(self, factor_expr: Any) -> str:
        """
        Extract structural factor ID (ignoring parameter values).

        Args:
            factor_expr: FE Factor or Expr object

        Returns:
            Structural ID for grouping factor families

        Raises:
            ValueError: If factor_expr is not a valid FE expression
            OptionalDependencyMissing: If FE not available

        Notes:
            - Structural ID groups factors with same operator DAG structure
            - Parameter values (window lengths, alphas, etc.) are ignored
            - Used for factor family analysis and structural similarity
            - Delegates to FE structural identity system
        """
        # This is a reference implementation stub
        raise NotImplementedError(
            "extract_structural_id requires FE structural identity API freeze."
        )

    def get_batch_provider(self) -> FactorBatchProvider:
        """
        Get FE-backed FactorBatchProvider for execution.

        Returns:
            FactorBatchProvider delegating to FE execution

        Raises:
            OptionalDependencyMissing: If FE not available
            NotImplementedError: If FE execution API not frozen
        """
        raise NotImplementedError(
            "FE FactorBatchProvider requires FE execution API freeze."
        )

    def get_identity_provider(self) -> FactorIdentityProvider:
        """
        Get FE-backed FactorIdentityProvider for canonical IDs.

        Returns:
            FactorIdentityProvider delegating to FE identity system

        Raises:
            OptionalDependencyMissing: If FE not available
            NotImplementedError: If FE identity API not frozen
        """
        raise NotImplementedError(
            "FE FactorIdentityProvider requires FE identity API freeze."
        )
