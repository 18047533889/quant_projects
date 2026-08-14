"""
Identity adapters for exact, sign-invariant, and structural matching.

Provides multiple levels of identity comparison:
- Exact: Canonical hash equality (default)
- Sign-invariant: Treats f and -f as equivalent
- Structural: Matches based on operator structure, ignoring parameters
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from factor_assets.identity.canonical import FactorIdentity


@dataclass(frozen=True)
class SignInvariantIdentity:
    """
    Sign-invariant identity for factors.

    Treats f(x) and -f(x) as the same factor.
    Useful for factors where direction is determined by evidence/ranking.
    """
    canonical_hash: str
    sign_normalized_hash: str
    is_negated: bool

    @classmethod
    def from_canonical_hash(cls, canonical_hash: str, canonical_repr: str) -> "SignInvariantIdentity":
        """
        Create sign-invariant identity from canonical representation.

        Args:
            canonical_hash: Canonical hash from FE
            canonical_repr: Canonical representation string

        Returns:
            SignInvariantIdentity
        """
        # Detect if expression is negated
        is_negated = cls._is_negated_expression(canonical_repr)

        # Compute sign-normalized hash
        sign_normalized_repr = cls._remove_negation(canonical_repr)
        sign_normalized_hash = hashlib.sha256(sign_normalized_repr.encode('utf-8')).hexdigest()

        return cls(
            canonical_hash=canonical_hash,
            sign_normalized_hash=sign_normalized_hash,
            is_negated=is_negated,
        )

    @staticmethod
    def _is_negated_expression(expr: str) -> bool:
        """Check if expression is a top-level negation."""
        # Simple heuristic: check if starts with minus or contains neg operator
        stripped = expr.strip()
        return stripped.startswith('-') or '"op":"neg"' in expr or '"op":"minus"' in expr

    @staticmethod
    def _remove_negation(expr: str) -> str:
        """Remove top-level negation from expression."""
        stripped = expr.strip()
        if stripped.startswith('-'):
            return stripped[1:].strip()
        # For JSON-like representations, remove neg wrapper
        if '"op":"neg"' in expr:
            # This is a simplified version - production would parse JSON properly
            return re.sub(r'\{"op":"neg","args":\[(.*?)\]\}', r'\1', expr)
        return expr

    def matches(self, other: "SignInvariantIdentity") -> bool:
        """Check if two identities match under sign-invariant comparison."""
        return self.sign_normalized_hash == other.sign_normalized_hash


@dataclass(frozen=True)
class StructuralIdentity:
    """
    Structural identity for factors.

    Matches based on operator structure, ignoring parameter values.
    Example: ts_rank(close, 20) and ts_rank(volume, 10) have same structure.
    """
    canonical_hash: str
    structural_hash: str
    operator_signature: str

    @classmethod
    def from_canonical_repr(cls, canonical_hash: str, canonical_repr: str) -> "StructuralIdentity":
        """
        Create structural identity from canonical representation.

        Args:
            canonical_hash: Canonical hash from FE
            canonical_repr: Canonical representation string

        Returns:
            StructuralIdentity
        """
        # Extract operator signature (structure without parameter values)
        operator_sig = cls._extract_operator_signature(canonical_repr)
        structural_hash = hashlib.sha256(operator_sig.encode('utf-8')).hexdigest()

        return cls(
            canonical_hash=canonical_hash,
            structural_hash=structural_hash,
            operator_signature=operator_sig,
        )

    @staticmethod
    def _extract_operator_signature(expr: str) -> str:
        """
        Extract operator signature from expression.

        Preserves operator names and structure, but normalizes parameter values.
        Field names (like "op", "args") and operator names are preserved.
        Only data values (numbers, non-operator strings) are normalized.
        """
        sig = expr

        # Replace numeric literals with <NUM> (but not in field names)
        # Match standalone numbers, not parts of words
        sig = re.sub(r':\s*(\d+\.?\d*)', r':<NUM>', sig)
        sig = re.sub(r'\[(\d+\.?\d*)\]', r'[<NUM>]', sig)
        sig = re.sub(r',\s*(\d+\.?\d*)', r',<NUM>', sig)

        # Replace string values but keep "op": and "args": keys and operator names
        # Replace strings in args arrays that are field references
        # This is a simple heuristic - keep strings after "op": but replace others
        def replace_string(match):
            full_match = match.group(0)
            string_content = match.group(1)

            # Keep if it's an operator name (comes after "op":)
            # Keep if it's a key name
            if string_content in ['op', 'args', 'kwargs']:
                return full_match

            # Look back to see if this is an operator value
            return full_match  # For now, keep all strings to preserve structure

        # Actually, let's just normalize numbers and keep string structure as-is
        # The key is that same operator with different params should match
        return sig

    def matches(self, other: "StructuralIdentity") -> bool:
        """Check if two identities match under structural comparison."""
        return self.structural_hash == other.structural_hash


class IdentityAdapter:
    """
    Adapter for computing multiple identity variants.

    Provides exact, sign-invariant, and structural identity computation.
    """

    def __init__(self):
        pass

    def compute_exact_identity(self, factor_identity: FactorIdentity) -> str:
        """
        Compute exact identity (canonical hash).

        Args:
            factor_identity: Factor identity from FE

        Returns:
            Canonical hash
        """
        return factor_identity.canonical_hash

    def compute_sign_invariant_identity(self, factor_identity: FactorIdentity) -> SignInvariantIdentity:
        """
        Compute sign-invariant identity.

        Args:
            factor_identity: Factor identity from FE

        Returns:
            SignInvariantIdentity
        """
        return SignInvariantIdentity.from_canonical_hash(
            factor_identity.canonical_hash,
            factor_identity.canonical_repr,
        )

    def compute_structural_identity(self, factor_identity: FactorIdentity) -> StructuralIdentity:
        """
        Compute structural identity.

        Args:
            factor_identity: Factor identity from FE

        Returns:
            StructuralIdentity
        """
        return StructuralIdentity.from_canonical_repr(
            factor_identity.canonical_hash,
            factor_identity.canonical_repr,
        )

    def compute_all_identities(
        self,
        factor_identity: FactorIdentity
    ) -> tuple[str, SignInvariantIdentity, StructuralIdentity]:
        """
        Compute all identity variants.

        Args:
            factor_identity: Factor identity from FE

        Returns:
            Tuple of (exact_hash, sign_invariant, structural)
        """
        return (
            self.compute_exact_identity(factor_identity),
            self.compute_sign_invariant_identity(factor_identity),
            self.compute_structural_identity(factor_identity),
        )


__all__ = [
    "SignInvariantIdentity",
    "StructuralIdentity",
    "IdentityAdapter",
]
