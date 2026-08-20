"""MutationRegistry: catalog of available mutation operations."""

from typing import Dict, List, Optional
from .mutation_spec import MutationSpec, ParameterSpec, ParameterKind, ParameterRole


class MutationRegistry:
    """Registry of available mutation operation specifications."""

    def __init__(self):
        self._specs: Dict[str, MutationSpec] = {}
        self._version = "0.1.0"

    def register(self, spec: MutationSpec) -> None:
        """Register a mutation specification."""
        if spec.mutation_type in self._specs:
            raise ValueError(f"Mutation type already registered: {spec.mutation_type}")
        self._specs[spec.mutation_type] = spec

    def get(self, mutation_type: str) -> Optional[MutationSpec]:
        """Get mutation specification by type."""
        return self._specs.get(mutation_type)

    def list_mutation_types(self) -> List[str]:
        """List all registered mutation types."""
        return sorted(self._specs.keys())

    def list_specs(self) -> List[MutationSpec]:
        """List all registered mutation specifications."""
        return list(self._specs.values())

    def version(self) -> str:
        """Get registry version."""
        return self._version


# Global registry instance
_GLOBAL_REGISTRY: Optional[MutationRegistry] = None


def get_mutation_registry() -> MutationRegistry:
    """Get the global mutation registry, creating with defaults if needed."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = MutationRegistry()
        _register_default_mutations(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def _register_default_mutations(registry: MutationRegistry) -> None:
    """Register default mutation operations."""

    # Parameter tuning: adjust numeric parameters
    registry.register(
        MutationSpec(
            mutation_type="parameter_tune",
            version="0.1.0",
            description="Adjust a numeric parameter of an existing operator",
            parameters=[
                ParameterSpec(
                    name="operator_name",
                    kind=ParameterKind.STRING,
                    role=ParameterRole.STRUCTURAL,
                    description="Name of operator to tune",
                ),
                ParameterSpec(
                    name="parameter_name",
                    kind=ParameterKind.STRING,
                    role=ParameterRole.STRUCTURAL,
                    description="Name of parameter to adjust",
                ),
                ParameterSpec(
                    name="new_value",
                    kind=ParameterKind.FLOAT,
                    role=ParameterRole.SCALAR,
                    description="New parameter value",
                ),
            ],
            requires_single_parent=True,
        )
    )

    # Window adjustment: change lookback window
    registry.register(
        MutationSpec(
            mutation_type="window_adjust",
            version="0.1.0",
            description="Adjust the lookback window of a temporal operator",
            parameters=[
                ParameterSpec(
                    name="new_window",
                    kind=ParameterKind.INTEGER,
                    role=ParameterRole.WINDOW,
                    min_value=1,
                    max_value=252,
                    description="New window length in periods",
                ),
            ],
            requires_single_parent=True,
        )
    )

    # Operator swap: replace an operator with similar one
    registry.register(
        MutationSpec(
            mutation_type="operator_swap",
            version="0.1.0",
            description="Replace an operator with a semantically similar one",
            parameters=[
                ParameterSpec(
                    name="target_operator",
                    kind=ParameterKind.STRING,
                    role=ParameterRole.STRUCTURAL,
                    description="Operator to replace",
                ),
                ParameterSpec(
                    name="replacement_operator",
                    kind=ParameterKind.STRING,
                    role=ParameterRole.STRUCTURAL,
                    description="New operator",
                ),
            ],
            requires_single_parent=True,
        )
    )

    # Decay adjustment: change decay/smoothing parameter
    registry.register(
        MutationSpec(
            mutation_type="decay_adjust",
            version="0.1.0",
            description="Adjust decay or smoothing parameter",
            parameters=[
                ParameterSpec(
                    name="new_decay",
                    kind=ParameterKind.FLOAT,
                    role=ParameterRole.DECAY,
                    min_value=0.0,
                    max_value=1.0,
                    description="New decay factor",
                ),
            ],
            requires_single_parent=True,
        )
    )

    # Simple composition: combine two factors
    registry.register(
        MutationSpec(
            mutation_type="linear_combination",
            version="0.1.0",
            description="Linearly combine two factors",
            parameters=[
                ParameterSpec(
                    name="weight_a",
                    kind=ParameterKind.FLOAT,
                    role=ParameterRole.SCALAR,
                    min_value=0.0,
                    max_value=1.0,
                    description="Weight for first factor",
                ),
                ParameterSpec(
                    name="weight_b",
                    kind=ParameterKind.FLOAT,
                    role=ParameterRole.SCALAR,
                    min_value=0.0,
                    max_value=1.0,
                    description="Weight for second factor",
                ),
            ],
            requires_single_parent=False,
            requires_multiple_parents=True,
        )
    )

    # Threshold adjustment
    registry.register(
        MutationSpec(
            mutation_type="threshold_adjust",
            version="0.1.0",
            description="Adjust a threshold parameter",
            parameters=[
                ParameterSpec(
                    name="new_threshold",
                    kind=ParameterKind.FLOAT,
                    role=ParameterRole.THRESHOLD,
                    description="New threshold value",
                ),
            ],
            requires_single_parent=True,
        )
    )
