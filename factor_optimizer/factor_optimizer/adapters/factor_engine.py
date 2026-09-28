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
    from factor_engine.expr.base import Expr
    from factor_engine.expr.cleaned_call import CleanedCall

    if not isinstance(expr, Expr):
        return 0

    if isinstance(expr, CleanedCall):
        # Count this operator plus recursively count in args
        return 1 + sum(_count_operators(arg) for arg in expr.args)

    return 0


def _compute_max_depth(expr: Any) -> int:
    """Compute maximum depth of expression tree."""
    from factor_engine.expr.base import Expr
    from factor_engine.expr.cleaned_call import CleanedCall

    if not isinstance(expr, Expr):
        return 0

    if isinstance(expr, CleanedCall):
        if not expr.args:
            return 1
        return 1 + max(_compute_max_depth(arg) for arg in expr.args)

    return 1


def _extract_lookback(expr: Any) -> int:
    """Extract maximum lookback period from expression."""
    from factor_engine.expr.base import Expr
    from factor_engine.expr.cleaned_call import CleanedCall

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

        import factor_engine.api  # FE top-level public API
        from factor_engine.mining.campaign import candidate_semantic_hash  # For canonical identity
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.backend.parameter_aliases import normalize_parameter_aliases
        from factor_engine.expr.base import Expr
        from factor_engine.ir.analyzer import Analyzer
        import pandas as pd

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
                """Validate referenced operators and any supplied FE parameters."""
                metadata = {}
                parameters = getattr(mutation, "parameters", None)
                if parameters is None and isinstance(mutation, dict):
                    parameters = mutation
                if parameters is None:
                    parameters = {}
                elif not isinstance(parameters, dict):
                    return {"is_legal": False,
                            "reason": "mutation parameters must be a mapping",
                            "metadata": metadata}

                operator_refs = []
                for key in ("target_operator", "replacement_operator", "operator_name", "operator"):
                    if key not in parameters or parameters[key] is None:
                        continue
                    name = parameters[key]
                    if isinstance(name, str) and not name:
                        continue
                    if not isinstance(name, str):
                        return {"is_legal": False,
                                "reason": f"{key} must be a string",
                                "metadata": metadata}
                    operator_refs.append((key, name))
                if not operator_refs:
                    metadata["operator_check"] = "not_applicable"
                    return {"is_legal": True,
                            "reason": "No FE operator reference; operator check not applicable",
                            "metadata": metadata}

                catalog = self.operator_registry.catalog()
                canonical_operators, resolved = {}, {}
                for key, name in operator_refs:
                    if not isinstance(name, str):
                        return {"is_legal": False, "reason": f"{key} must be a string", "metadata": metadata}
                    try:
                        canonical = OperatorRegistry.resolve_canonical_strict(name)
                    except (KeyError, ValueError) as exc:
                        return {"is_legal": False,
                                "reason": f"Operator {name!r} not found in FE catalog: {exc}",
                                "metadata": metadata}
                    op_meta = catalog.get(canonical)
                    if not op_meta or op_meta.get("status") not in {"implemented", "production"}:
                        status = op_meta.get("status") if op_meta else "unknown"
                        return {"is_legal": False,
                                "reason": f"Operator {name!r} resolves to {canonical!r} with status {status!r}",
                                "metadata": metadata}
                    canonical_operators[key] = canonical
                    resolved[key] = op_meta

                metadata["canonical_operators"] = canonical_operators
                metadata["operator_meta"] = resolved[operator_refs[-1][0]]
                normalized_parameters = {}
                for key, canonical in canonical_operators.items():
                    kwargs = parameters.get(f"{key}_parameters", {})
                    if not isinstance(kwargs, dict):
                        return {"is_legal": False,
                                "reason": f"Parameters for {canonical!r} must be a mapping",
                                "metadata": metadata}
                    if (key == "replacement_operator" or len(canonical_operators) == 1) and not kwargs:
                        kwargs = parameters.get("operator_parameters", {})
                    if not isinstance(kwargs, dict):
                        return {"is_legal": False,
                                "reason": f"Parameters for {canonical!r} must be a mapping",
                                "metadata": metadata}
                    if key == "operator_name" and parameters.get("parameter_name") is not None:
                        parameter_name = parameters["parameter_name"]
                        if not isinstance(parameter_name, str) or not parameter_name:
                            return {"is_legal": False,
                                    "reason": "parameter_name must be a nonempty string",
                                    "metadata": metadata}
                        kwargs = {**kwargs, parameter_name: parameters.get("new_value")}
                    if not kwargs:
                        continue
                    try:
                        normalized = normalize_parameter_aliases(canonical, kwargs)
                        self._validate_operator_parameters(canonical, normalized)
                    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                        return {"is_legal": False,
                                "reason": f"Invalid parameters for FE operator {canonical!r}: {exc}",
                                "metadata": metadata}
                    normalized_parameters[key] = normalized
                if normalized_parameters:
                    metadata["normalized_operator_parameters"] = normalized_parameters
                return {
                    "is_legal": True,
                    "reason": "Mutation passed FE operator validation",
                    "metadata": metadata,
                }

            def _validate_operator_parameters(self, canonical: str, kwargs: Dict[str, Any]) -> None:
                """Run FE's registered call-contract validator without executing a kernel."""
                backends = self.operator_registry.backends_for(canonical)
                operator = None
                for backend in ("pandas_numpy", *backends):
                    if backend in backends:
                        operator = self.operator_registry.get(canonical, backend=backend, mode="any")
                        if operator is not None:
                            break
                if operator is None:
                    raise ValueError("no registered FE operator implementation is available")
                meta = operator.metadata
                names = list(getattr(meta, "param_names", ()) or ())
                specs = getattr(meta, "param_specs", {}) or {}
                inputs = list(getattr(meta, "panel_params", ()) or ())
                if not inputs:
                    scalar_names = set(getattr(meta, "scalar_params", ()) or ())
                    inputs = [name for name in names if name not in specs and name not in scalar_names]
                inputs = [name for name in inputs if name not in kwargs]
                placeholder = pd.DataFrame([[0.0]])
                operator._prepare_call(tuple(placeholder for _ in inputs), kwargs)

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

                # Preserve the legacy score: FE Analyzer history has different
                # semantics, so expose its typed evidence additively.
                fe_analysis = None
                if isinstance(expr, Expr):
                    try:
                        analysis = Analyzer().lower(expr)
                        fe_analysis = {
                            "available": True,
                            "lookback": analysis.lookback,
                            "has_ts_op": analysis.has_ts_op,
                            "has_cs_op": analysis.has_cs_op,
                            "referenced_columns": sorted(analysis.referenced_columns),
                            "requires_full_history": analysis.requires_full_history,
                        }
                    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
                        fe_analysis = {"available": False, "reason": str(exc)}

                return {
                    "operator_count": operator_count,
                    "max_depth": max_depth,
                    "lookback_periods": lookback_periods,
                    "estimated_cost": estimated_cost,
                    "domains": [],  # Would require deeper analysis
                    "sources": [],  # Would require deeper analysis
                    "fe_analysis": fe_analysis,
                }

            def get_operator_metadata(
                self, operator_names: Optional[List[str]] = None
            ) -> Dict[str, Dict[str, Any]]:
                """Get operator metadata from FE catalog."""
                catalog = self.operator_registry.catalog()

                if operator_names is None:
                    # Return all operators
                    return catalog

                # Resolve aliases to the same canonical FE names used by
                # mutation validation. Unknown names retain the old omission
                # behavior; result keys are always canonical names.
                result = {}
                for name in operator_names:
                    if not isinstance(name, str):
                        continue
                    try:
                        canonical = OperatorRegistry.resolve_canonical_strict(name)
                    except (KeyError, ValueError):
                        continue
                    if canonical in catalog:
                        result[canonical] = catalog[canonical]

                return result

        return ConcreteFEAdapter()

    except ImportError as e:
        raise OptionalDependencyMissing(
            "factor-engine not installed or not in Python path. "
            "Ensure factor_engine is available in the parent directory."
        ) from e
