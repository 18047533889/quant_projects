"""
FactorEngine adapter for factor identity integration.

Provides FactorIdentityProvider implementation that connects to FE's
canonical hash computation and expression representation.

This is an OPTIONAL adapter — FA core does not depend on FE.
"""

from typing import Optional
import hashlib
import json

try:
    from factor_engine.expr import canonical_expression, expression_payload, Expr
    from factor_engine.expr.base import ensure_expr
    FE_AVAILABLE = True
except ImportError:
    FE_AVAILABLE = False
    canonical_expression = None
    expression_payload = None
    Expr = None
    ensure_expr = None

from factor_assets.identity.canonical import FactorIdentity
from factor_assets.adapters import OptionalDependencyMissing


class FEIdentityProvider:
    """
    FactorIdentityProvider implementation for FactorEngine.

    Connects to FE's canonical expression system to obtain:
    - Canonical hash (from FE's expression payload)
    - Canonical representation (from FE's expression tree)
    - FE identity references (compiler generation, IR digest)

    FA does NOT duplicate FE's parser or hash logic — it delegates through this adapter.
    """

    def __init__(self, compiler_generation: Optional[str] = None, *,
                 surface: str = "daily", dialect: str = "native",
                 dialect_version: Optional[str] = None):
        """
        Initialize FE identity provider.

        Args:
            compiler_generation: Optional FE compiler generation tag

        Raises:
            OptionalDependencyMissing: If FE is not available
        """
        if not FE_AVAILABLE:
            raise OptionalDependencyMissing("factor_engine", "FEIdentityProvider")

        self.compiler_generation = compiler_generation or "fe-0.9.x"
        self.surface = surface
        self.dialect = dialect
        self.dialect_version = dialect_version

    def get_canonical_hash(self, expression: str) -> str:
        """
        Get canonical hash for a factor expression from FE.

        Args:
            expression: Factor expression string

        Returns:
            Canonical hash (hex digest)

        Raises:
            OptionalDependencyMissing: If FE is not available
            ValueError: If expression is invalid
        """
        # Parse and normalize expression through FE
        try:
            expr_node = self._parse_expression(expression)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            raise ValueError(f"Invalid factor expression: {e}") from e

        # Get FE canonical representation
        canonical_repr = canonical_expression(expr_node)

        # Hash the canonical representation (deterministic)
        hash_bytes = hashlib.sha256(canonical_repr.encode("utf-8")).digest()
        return hash_bytes.hex()

    def get_canonical_repr(self, expression: str) -> str:
        """
        Get canonical representation of a factor expression from FE.

        Args:
            expression: Factor expression string

        Returns:
            Canonical JSON representation from FE

        Raises:
            OptionalDependencyMissing: If FE is not available
            ValueError: If expression is invalid
        """
        try:
            expr_node = self._parse_expression(expression)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            raise ValueError(f"Invalid factor expression: {e}") from e

        return canonical_expression(expr_node)

    def get_identity_ref(self, expression: str) -> str:
        """
        Get FE identity reference for a factor.

        For now, this returns the canonical hash. In production, this could
        return FE plan hash, IR digest, or other FE-specific identity.

        Args:
            expression: Factor expression string

        Returns:
            FE identity reference

        Raises:
            OptionalDependencyMissing: If FE is not available
            ValueError: If expression is invalid
        """
        # For now, use canonical hash as identity ref
        # Production may use FE plan hash or IR digest
        return self.get_canonical_hash(expression)

    def get_full_identity(
        self,
        expression: str,
        complexity_score: Optional[float] = None,
    ) -> FactorIdentity:
        """
        Get complete factor identity from FE.

        Args:
            expression: Factor expression string
            complexity_score: Optional complexity score (computed separately)

        Returns:
            Complete FactorIdentity

        Raises:
            OptionalDependencyMissing: If FE is not available
            ValueError: If expression is invalid
        """
        canonical_repr = self.get_canonical_repr(expression)
        # Parsing/parameter certification belongs to FE and may be expensive.
        # Resolve once so all three identity fields describe the same tree.
        canonical_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
        fe_identity_ref = canonical_hash

        return FactorIdentity(
            canonical_repr=canonical_repr,
            canonical_hash=canonical_hash,
            fe_identity_ref=fe_identity_ref,
            fe_compiler_generation=self.compiler_generation,
            complexity_score=complexity_score,
        )

    def _parse_expression(self, expression: str) -> "Expr":
        """
        Parse expression string through FE.

        Uses FE's existing budgeted/allowlisted DSL parser. ``ensure_expr`` is
        a scalar conversion API: passing a field name to it creates a string
        literal and therefore the wrong factor identity.

        Args:
            expression: Expression string or Expr node

        Returns:
            Expr node

        Raises:
            ValueError: If expression cannot be parsed
        """
        if isinstance(expression, Expr):
            return expression

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a nonempty DSL string or FE Expr")
        try:
            from factor_engine.api.dsl_parser import parse_expr
            return parse_expr(expression, surface=self.surface, dialect=self.dialect,
                              dialect_version=self.dialect_version)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            raise ValueError(f"Failed to parse expression '{expression}': {e}") from e

    def validate_expression(self, expression: str) -> bool:
        """
        Validate that an expression is well-formed according to FE.

        Args:
            expression: Expression to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self._parse_expression(expression)
            return True
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):
            return False


class FEIdentityProviderFromExpr(FEIdentityProvider):
    """
    FE identity provider that accepts Expr nodes directly.

    Use this when you already have parsed Expr nodes from FE
    and don't need string parsing.
    """

    def _parse_expression(self, expression) -> "Expr":
        """
        Accept Expr node directly without string parsing.

        Args:
            expression: Expr node from FE

        Returns:
            The same Expr node

        Raises:
            ValueError: If not an Expr node
        """
        if not isinstance(expression, Expr):
            raise ValueError(
                f"FEIdentityProviderFromExpr requires Expr node, got {type(expression).__name__}"
            )
        return expression


__all__ = [
    "FEIdentityProvider",
    "FEIdentityProviderFromExpr",
]
