"""Platform HTTP API layer.

QRP-P1. Thin FastAPI skeleton: auth (login/logout/me) + a protected health
route. No domain logic. Auth dependency injects the Db + RBAC.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
