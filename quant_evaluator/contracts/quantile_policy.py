"""
Quantile tie-breaking policy contract.

QE-Q-P0-001: Formal tie-breaking policy for quantile assignment.
"""

from enum import Enum
from typing import Literal


class QuantileTiePolicy(str, Enum):
    """
    Tie-breaking policy for quantile assignment.

    When multiple values are equal at quantile boundaries, this policy
    determines how they are assigned to bins.
    """
    AVERAGE = "average"  # Tied values get average quantile (NOT IMPLEMENTED - requires rank-based algorithm)
    MIN = "min"          # Tied values get minimum quantile
    MAX = "max"          # Tied values get maximum quantile
    FIRST = "first"      # Tie-break by original order (NOT IMPLEMENTED - order-dependent)


# Type alias for static type checking
QuantileTiePolicyLiteral = Literal["average", "min", "max", "first"]


def validate_tie_policy(policy: str) -> QuantileTiePolicy:
    """
    Validate and normalize tie policy string.

    Args:
        policy: Policy string

    Returns:
        Validated QuantileTiePolicy enum

    Raises:
        ValueError: If policy is invalid or not implemented
    """
    try:
        result = QuantileTiePolicy(policy)
    except ValueError:
        valid = ", ".join(p.value for p in QuantileTiePolicy)
        raise ValueError(f"Invalid tie policy '{policy}'. Must be one of: {valid}")

    # Check for unimplemented policies
    if result in (QuantileTiePolicy.AVERAGE, QuantileTiePolicy.FIRST):
        raise NotImplementedError(
            f"Tie policy '{result.value}' is not implemented. "
            f"Current implementation uses boundary comparison which is equivalent to 'max'. "
            f"Use 'min' or 'max' for now."
        )

    return result
