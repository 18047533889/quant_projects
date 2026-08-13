"""Sync subpackage for idempotency and deduplication."""

from .idempotency import IdempotentSyncEngine

# Backward compatibility - re-export legacy IdempotentSync
from ..sync_legacy import IdempotentSync

__all__ = ["IdempotentSyncEngine", "IdempotentSync"]
