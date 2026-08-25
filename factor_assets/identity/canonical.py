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


@dataclass(frozen=True)
class FactorDefinitionIdentity:
    """Identity of the *factor definition* itself.

    Distinct from :class:`FactorCompilerIdentity` (which compiler/version
    produced the representation) and :class:`FactorValueIdentity` (which
    materialization/instance of the factor's values).  The definition identity
    is version-qualified: ``factor_version`` is MANDATORY in production and
    MUST NOT fall back to an FE compiler generation — the compiler is a
    different identity axis.
    """

    factor_id: str
    factor_version: str
    canonical_hash: Optional[str] = None
    canonical_repr: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.factor_version:
            raise ValueError(
                "FactorDefinitionIdentity requires a factor_version; it must "
                "not fall back to an FE compiler generation"
            )


@dataclass(frozen=True)
class FactorCompilerIdentity:
    """Identity of the FE compiler / toolchain that produced the representation.

    This is a separate identity axis from the factor definition: two factors
    with the same canonical expression may be compiled by different compiler
    generations.  Recording it is provenance, never a substitute for
    ``FactorDefinitionIdentity.factor_version``.
    """

    compiler_generation: Optional[str] = None
    fe_identity_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.compiler_generation and not self.fe_identity_ref:
            raise ValueError(
                "FactorCompilerIdentity requires at least one of "
                "compiler_generation or fe_identity_ref"
            )


@dataclass(frozen=True)
class FactorValueIdentity:
    """Identity of a specific materialized instance of a factor's values.

    Distinguishes one realization of a factor (e.g. by snapshot/universe/run)
    from another.  It is orthogonal to the definition and compiler identity.
    """

    factor_id: str
    snapshot_ref: Optional[str] = None
    universe_ref: Optional[str] = None
    run_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.snapshot_ref and not self.universe_ref and not self.run_ref:
            raise ValueError(
                "FactorValueIdentity requires at least one of "
                "snapshot_ref/universe_ref/run_ref"
            )


__all__ = [
    "FactorIdentity",
    "FactorIdentityProvider",
    "FactorDefinitionIdentity",
    "FactorCompilerIdentity",
    "FactorValueIdentity",
    "create_factor_id",
]
