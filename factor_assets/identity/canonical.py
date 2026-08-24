"""
Identity management through FE protocol/adapter boundary.

FA does NOT duplicate FE's parser or canonical hash computation.
Identity is obtained through a protocol boundary from FE adapters.
"""

from dataclasses import dataclass
from typing import Protocol, Optional


class FactorIdentityProvider(Protocol):
    """
    Protocol for obtaining factor identity from FactorEngine.

    FA does not implement this — it is provided by an optional FE adapter.
    This protocol defines the boundary without creating a hard dependency.
    """

    def get_canonical_hash(self, expression: str) -> str:
        """
        Get canonical hash for a factor expression.

        Args:
            expression: Factor expression string

        Returns:
            Canonical hash from FE

        Raises:
            ValueError: If expression is invalid
        """
        ...

    def get_canonical_repr(self, expression: str) -> str:
        """
        Get canonical representation of a factor expression.

        Args:
            expression: Factor expression string

        Returns:
            Canonical representation from FE
        """
        ...

    def get_identity_ref(self, expression: str) -> str:
        """
        Get FE identity reference for a factor.

        Args:
            expression: Factor expression string

        Returns:
            FE identity reference (e.g., plan hash, IR digest)
        """
        ...


@dataclass(frozen=True)
class FactorIdentity:
    """
    Complete factor identity from FE.

    Obtained through FactorIdentityProvider, not computed locally.
    """
    canonical_repr: str
    canonical_hash: str
    fe_identity_ref: Optional[str] = None
    fe_compiler_generation: Optional[str] = None
    complexity_score: Optional[float] = None

    def __post_init__(self):
        if not self.canonical_repr:
            raise ValueError("canonical_repr is required")
        if not self.canonical_hash:
            raise ValueError("canonical_hash is required")


def create_factor_id(canonical_hash: str, prefix: str = "F") -> str:
    """
    Create a factor ID from canonical hash.

    Simple deterministic mapping: prefix + first 16 chars of hash.
    Real production may use a more sophisticated scheme.

    Args:
        canonical_hash: Canonical hash from FE
        prefix: ID prefix (default "F")

    Returns:
        Factor ID string
    """
    if not canonical_hash:
        raise ValueError("canonical_hash is required")
    if len(canonical_hash) < 16:
        raise ValueError("canonical_hash must be at least 16 characters")

    return f"{prefix}{canonical_hash[:16]}"
