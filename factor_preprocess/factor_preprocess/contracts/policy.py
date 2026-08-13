"""
Preprocessing policy and transform specification contracts.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


class TransformKind(Enum):
    """Type of transform operation."""
    CROSS_SECTIONAL = "cross_sectional"
    ROLLING = "rolling"
    NEUTRALIZATION = "neutralization"
    REPRESENTATION = "representation"


class TransformMode(Enum):
    """Whether transform requires fitting."""
    STATELESS = "stateless"
    FITTED = "fitted"


@dataclass(frozen=True)
class TransformSpec:
    """
    Specification for a single transform operation.

    Immutable and serializable.
    """
    name: str
    kind: TransformKind
    mode: TransformMode
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    requires_exposure: bool = False
    produces_channels: List[str] = field(default_factory=lambda: ["transformed"])
    config_hash: Optional[str] = None

    def __post_init__(self):
        """Validate spec on construction."""
        if not self.name:
            raise ValueError("TransformSpec.name cannot be empty")
        if not self.version:
            raise ValueError("TransformSpec.version cannot be empty")

        # Ensure parameters is immutable
        if self.parameters and not isinstance(self.parameters, dict):
            raise TypeError("parameters must be dict")


@dataclass(frozen=True)
class PreprocessingPolicy:
    """
    Ordered sequence of transforms to apply.

    This is the top-level contract for preprocessing pipeline.
    """
    policy_id: str
    transforms: List[TransformSpec]
    version: str = "0.1.0"

    # Context requirements
    requires_universe: bool = False
    requires_industry: bool = False
    requires_size: bool = False

    # Output specification
    output_channels: List[str] = field(default_factory=lambda: ["features"])
    missing_indicator: bool = True
    freshness_channel: bool = False

    def __post_init__(self):
        """Validate policy on construction."""
        if not self.policy_id:
            raise ValueError("PreprocessingPolicy.policy_id cannot be empty")
        if not self.transforms:
            raise ValueError("PreprocessingPolicy.transforms cannot be empty")

        # Check for duplicate transform names
        names = [t.name for t in self.transforms]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate transform names in policy")

    def has_fitted_transforms(self) -> bool:
        """Check if any transform requires fitting."""
        return any(t.mode == TransformMode.FITTED for t in self.transforms)
