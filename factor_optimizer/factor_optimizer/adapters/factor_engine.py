"""FactorEngineAdapter: protocol for FE integration (optional dependency)."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class FactorEngineAdapter(Protocol):
    """
    Protocol for FactorEngine integration.

    This is a protocol/interface, not an implementation. FO does NOT directly import
    or depend on FE. Concrete adapters are provided when FE is available.

    Required capabilities:
    - Compute canonical identity hash for deduplication
    - Validate mutation legality (operator existence, parameter bounds)
    - Estimate complexity (operator count, depth, lookback)
    - Get operator metadata for mutation grammar
    """

    def compute_canonical_hash(self, factor_definition: Any) -> str:
        """
        Compute FE canonical identity hash.

        Args:
            factor_definition: Factor definition (FE-specific format: Factor or Expr)

        Returns:
            Canonical identity hash string (SHA256 hex)

        Raises:
            Exception: If factor is invalid or canonicalization fails
        """
        ...

    def validate_mutation(self, mutation: Any, spec: Any) -> Dict[str, Any]:
        """
        Validate mutation legality through FE.

        Args:
            mutation: CandidateMutation object
            spec: MutationSpec object

        Returns:
            Dictionary with:
                - is_legal (bool): Whether mutation is legal
                - reason (str): Reason if illegal
                - metadata (dict): Additional validation info
        """
        ...

    def estimate_complexity(self, factor_definition: Any) -> Dict[str, Any]:
        """
        Estimate factor complexity through FE analyzer.

        Args:
            factor_definition: Factor definition (FE-specific format: Factor or Expr)

        Returns:
            Dictionary with complexity metrics:
                - operator_count (int): Number of operators in AST
                - max_depth (int): Maximum AST depth
                - lookback_periods (int): Maximum lookback window
                - estimated_cost (float): Relative compute cost
                - domains (list): Data domains required
                - sources (list): Data sources required
        """
        ...

    def get_operator_metadata(self, operator_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get operator metadata for mutation grammar.

        Args:
            operator_names: Optional list of specific operators (None = all)

        Returns:
            Dictionary mapping operator_name to metadata:
                - parameters (list): Parameter specifications
                - backends (list): Available backends
                - timing (str): Timing classification
                - role (str): Parameter role
                - status (str): Implementation status
        """
        ...


class OptionalDependencyMissing(Exception):
    """Raised when optional FE dependency is not available."""

    pass


def _count_operators(expr: Any) -> int:
    """Count operators in an expression tree."""
    from expr.base import Expr
    from expr.cleaned_call import CleanedCall

    if not isinstance(expr, Expr):
        return 0

    if isinstance(expr, CleanedCall):
        # Count this operator plus recursively count in args
        return 1 + sum(_count_operators(arg) for arg in expr.args)

    return 0


def _compute_max_depth(expr: Any) -> int:
    """Compute maximum depth of expression tree."""
    from expr.base import Expr
    from expr.cleaned_call import CleanedCall

    if not isinstance(expr, Expr):
        return 0

    if isinstance(expr, CleanedCall):
        if not expr.args:
            return 1
        return 1 + max(_compute_max_depth(arg) for arg in expr.args)

    return 1


def _extract_lookback(expr: Any) -> int:
    """Extract maximum lookback period from expression."""
    from expr.base import Expr
    from expr.cleaned_call import CleanedCall

    if not isinstance(expr, Expr):
        return 0

    if isinstance(expr, CleanedCall):
        # Check for window parameter
        max_lookback = 0

        # Common window parameter names
        for param_name in ["window", "period", "span", "lookback"]:
            if param_name in expr.kwargs:
                value = expr.kwargs[param_name]
                if isinstance(value, int) and value > 0:
                    max_lookback = max(max_lookback, value)

        # Recursively check args
        for arg in expr.args:
            max_lookback = max(max_lookback, _extract_lookback(arg))

        return max_lookback

    return 0


def create_fe_adapter() -> FactorEngineAdapter:
    """
    Create FE adapter if factor-engine is installed.

    Returns:
        FactorEngineAdapter implementation

    Raises:
        OptionalDependencyMissing: If factor-engine not installed
    """
    try:
        # Try to import FE - this will fail if not installed
        import sys
        import os

        # Add factor_engine to path if in expected location
        fe_path = os.path.join(os.path.dirname(__file__), "../../../factor_engine")
        if os.path.exists(fe_path) and fe_path not in sys.path:
            sys.path.insert(0, os.path.abspath(fe_path))

        import api  # FE top-level public API
        from mining.campaign import candidate_semantic_hash  # For canonical identity
        from cleaned_operators.registry import OperatorRegistry
        from expr.base import Expr

        # Concrete adapter implementation
        class ConcreteFEAdapter:
            """Concrete FE adapter implementation."""

            def __init__(self):
                """Initialize with FE operator registry."""
                self.operator_registry = OperatorRegistry()

            def compute_canonical_hash(self, factor_definition: Any) -> str:
                """Compute canonical hash using FE's mining module."""
                # factor_definition should be an api.Factor or expr.Expr
                return candidate_semantic_hash(factor_definition)

            def validate_mutation(self, mutation: Any, spec: Any) -> Dict[str, Any]:
                """Validate mutation through FE operator catalog."""
                # Check if mutation involves operators that exist in catalog
                metadata = {}

                # Extract operator names from mutation
                # Support both dict format and object format with .parameters
                if hasattr(mutation, 'parameters'):
                    target_operator = mutation.parameters.get("target_operator")
                elif isinstance(mutation, dict):
                    target_operator = mutation.get("operator") or mutation.get("target_operator")
                else:
                    target_operator = None

                if target_operator:
                    # Check if operator exists
                    catalog = self.operator_registry.catalog()
                    if target_operator not in catalog:
                        return {
                            "is_legal": False,
                            "reason": f"Operator '{target_operator}' not found in FE catalog",
                            "metadata": metadata,
                        }

                    # Check operator status
                    op_meta = catalog[target_operator]
                    if op_meta.get("status") != "implemented":
                        return {
                            "is_legal": False,
                            "reason": f"Operator '{target_operator}' status: {op_meta.get('status')}",
                            "metadata": metadata,
                        }

                    metadata["operator_meta"] = op_meta

                # All checks passed
                return {
                    "is_legal": True,
                    "reason": "Mutation passed FE validation",
                    "metadata": metadata,
                }

            def estimate_complexity(self, factor_definition: Any) -> Dict[str, Any]:
                """Estimate complexity through FE analyzer."""
                # Extract expression from Factor if needed
                if hasattr(factor_definition, "expr"):
                    expr = factor_definition.expr
                else:
                    expr = factor_definition

                # Compute complexity metrics
                operator_count = _count_operators(expr)
                max_depth = _compute_max_depth(expr)
                lookback_periods = _extract_lookback(expr)

                # Simple cost heuristic: combine operator count and lookback
                estimated_cost = float(operator_count) * (1.0 + lookback_periods / 20.0)

                return {
                    "operator_count": operator_count,
                    "max_depth": max_depth,
                    "lookback_periods": lookback_periods,
                    "estimated_cost": estimated_cost,
                    "domains": [],  # Would require deeper analysis
                    "sources": [],  # Would require deeper analysis
                }

            def get_operator_metadata(
                self, operator_names: Optional[List[str]] = None
            ) -> Dict[str, Dict[str, Any]]:
                """Get operator metadata from FE catalog."""
                catalog = self.operator_registry.catalog()

                if operator_names is None:
                    # Return all operators
                    return catalog

                # Return specific operators
                result = {}
                for name in operator_names:
                    if name in catalog:
                        result[name] = catalog[name]

                return result

        return ConcreteFEAdapter()

    except ImportError as e:
        raise OptionalDependencyMissing(
            "factor-engine not installed or not in Python path. "
            "Ensure factor_engine is available in the parent directory."
        ) from e
