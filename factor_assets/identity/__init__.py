"""
Identity package — factor identity management through FE protocol boundary.
"""

from factor_assets.identity.canonical import (
    FactorIdentityProvider,
    FactorIdentity,
    create_factor_id,
)

__all__ = [
    "FactorIdentityProvider",
    "FactorIdentity",
    "create_factor_id",
]
