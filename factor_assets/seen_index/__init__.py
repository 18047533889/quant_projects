"""
Seen index package — exact history tracking for deduplication.
"""

from factor_assets.seen_index.exact import (
    SeenRecord,
    SeenIndex,
)
from factor_assets.seen_index.persistent import PersistentSeenIndex

__all__ = [
    "SeenRecord",
    "SeenIndex",
    "PersistentSeenIndex",
]
