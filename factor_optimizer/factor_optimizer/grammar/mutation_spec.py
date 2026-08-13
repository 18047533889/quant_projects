"""MutationSpec: typed mutation operation with parameter constraints."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ParameterKind(Enum):
    """Type of parameter value."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    ENUM = "enum"
    INTEGER_LIST = "integer_list"
    FLOAT_LIST = "float_list"
    STRING_LIST = "string_list"


class ParameterRole(Enum):
    """Semantic role of parameter in factor computation."""

    WINDOW = "window"  # Lookback window length
    DECAY = "decay"  # Decay/smoothing parameter
    THRESHOLD = "threshold"  # Numeric threshold
    QUANTILE = "quantile"  # Quantile cutoff [0, 1]
    SCALAR = "scalar"  # Generic numeric scalar
    CATEGORICAL = "categorical"  # Categorical choice
    STRUCTURAL = "structural"  # Changes computation structure
    TIMING = "timing"  # Affects temporal alignment
    CAUSAL = "causal"  # Impacts causal validity


@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for a mutation parameter.

    Attributes:
        name: Parameter name
        kind: Value type
        role: Semantic role
        required: Whether parameter is required
        default: Default value if not required
        min_value: Minimum allowed value (numeric types)
        max_value: Maximum allowed value (numeric types)
        allowed_values: Allowed values (enum type)
        description: Human-readable description
    """

    name: str
    kind: ParameterKind
    role: ParameterRole
    required: bool = True
    default: Optional[Any] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter spec."""
        if not self.name:
            raise ValueError("Parameter name is required")

        # Enum kind requires allowed_values
        if self.kind == ParameterKind.ENUM and not self.allowed_values:
            raise ValueError(f"Parameter {self.name}: ENUM kind requires allowed_values")

        # Numeric constraints only for numeric types
        numeric_kinds = {
            ParameterKind.INTEGER,
            ParameterKind.FLOAT,
            ParameterKind.INTEGER_LIST,
            ParameterKind.FLOAT_LIST,
        }
        if (self.min_value is not None or self.max_value is not None) and self.kind not in numeric_kinds:
            raise ValueError(f"Parameter {self.name}: min/max only valid for numeric kinds")

    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a parameter value.

        Returns:
            (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"{self.name} is required"
            return True, None

        # Type validation
        if self.kind == ParameterKind.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"{self.name} must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name} must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name} must be <= {self.max_value}"

        elif self.kind == ParameterKind.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"{self.name} must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name} must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name} must be <= {self.max_value}"

        elif self.kind == ParameterKind.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"{self.name} must be a boolean"

        elif self.kind == ParameterKind.STRING:
            if not isinstance(value, str):
                return False, f"{self.name} must be a string"

        elif self.kind == ParameterKind.ENUM:
            if value not in self.allowed_values:
                return False, f"{self.name} must be one of {self.allowed_values}"

        elif self.kind == ParameterKind.INTEGER_LIST:
            if not isinstance(value, list):
                return False, f"{self.name} must be a list"
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                return False, f"{self.name} must contain only integers"

        elif self.kind == ParameterKind.FLOAT_LIST:
            if not isinstance(value, list):
                return False, f"{self.name} must be a list"
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                return False, f"{self.name} must contain only numbers"

        elif self.kind == ParameterKind.STRING_LIST:
            if not isinstance(value, list):
                return False, f"{self.name} must be a list"
            if not all(isinstance(v, str) for v in value):
                return False, f"{self.name} must contain only strings"

        return True, None


@dataclass(frozen=True)
class MutationSpec:
    """
    Specification for a mutation operation type.

    Attributes:
        mutation_type: Unique mutation type identifier
        version: Grammar version
        description: Human-readable description
        parameters: Parameter specifications
        requires_single_parent: Whether mutation requires exactly one parent
        requires_multiple_parents: Whether mutation requires 2+ parents
        domains: Allowed factor domains (empty = all)
        sources: Allowed data sources (empty = all)
    """

    mutation_type: str
    version: str
    description: str
    parameters: List[ParameterSpec] = field(default_factory=list)
    requires_single_parent: bool = True
    requires_multiple_parents: bool = False
    domains: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate mutation spec."""
        if not self.mutation_type:
            raise ValueError("mutation_type is required")
        if not self.version:
            raise ValueError("version is required")
        if self.requires_single_parent and self.requires_multiple_parents:
            raise ValueError("Cannot require both single and multiple parents")

        # Check for duplicate parameter names
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            raise ValueError(f"Duplicate parameter names in {self.mutation_type}")

    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Get parameter spec by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a parameter dictionary.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check required parameters
        for param_spec in self.parameters:
            value = params.get(param_spec.name)
            is_valid, error_msg = param_spec.validate_value(value)
            if not is_valid:
                errors.append(error_msg)

        # Check for unknown parameters
        known_names = {p.name for p in self.parameters}
        for name in params.keys():
            if name not in known_names:
                errors.append(f"Unknown parameter: {name}")

        return len(errors) == 0, errors
