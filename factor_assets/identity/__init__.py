"""
Identity package — factor identity management through FE protocol boundary.
"""

from factor_assets.identity.canonical import (
    FactorIdentityProvider,
    FactorIdentity,
    create_factor_id,
)
from factor_assets.identity.adapters import (
    SignInvariantIdentity,
    StructuralIdentity,
    IdentityAdapter,
)

__all__ = [
    "FactorIdentityProvider",
    "FactorIdentity",
    "create_factor_id",
    "SignInvariantIdentity",
    "StructuralIdentity",
    "IdentityAdapter",
]
