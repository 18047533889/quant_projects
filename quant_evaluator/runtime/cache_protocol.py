"""
Cache protocol for intermediate metric results.

Single behavioural contract for intermediate-result caches.  The production
implementation is :class:`V2IntermediateCache` (backed by the cache_v2
``MultiLevelCache``); the legacy :class:`IntermediateCache` is deprecated and
retained only for backward compatibility.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):
    """Behavioural contract for intermediate metric caches.

    Keys are :class:`quant_evaluator.runtime.intermediates.CacheKey` values.
    Implementations must be safe to share across evaluations of the same
    ``Evaluator`` and must treat an unreadable entry exactly like a miss.
    """

    def get(self, key: Any) -> Optional[Any]:
        """Return the cached value for ``key`` or None on miss."""
        ...

    def put(self, key: Any, value: Any) -> bool:
        """Store ``value`` under ``key``; return True when stored."""
        ...

    def has(self, key: Any) -> bool:
        """Return whether ``key`` is present (without promoting it)."""
        ...

    def invalidate(self, key: Any) -> bool:
        """Remove one entry; return True when something was removed."""
        ...

    def invalidate_metric(self, metric_id: str) -> int:
        """Remove every entry for ``metric_id``; return the count removed."""
        ...

    def clear(self) -> None:
        """Remove every entry."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics including ``num_entries``/``size_mb``."""
        ...


__all__ = ["CacheProtocol"]
