"""LocalArtifactCache — on-disk LRU cache keyed by content_hash (spec §22).

Implements the ``LocalArtifactCache`` port declared in
:mod:`quant_platform.app.contracts.storage`:

- ``max_bytes`` capacity with LRU eviction (evict the least-recently-used entry
  while the index exceeds ``max_bytes``).
- ``store`` recomputes the sha256 of ``data`` and fails closed (raises
  ``ChecksumMismatchError``) if it does not equal the ``content_hash`` key —
  corrupt bytes never land on disk under that key.
- ``lookup`` verifies the stored bytes against the key before returning them and
  treats a verification failure as a cache miss (the corrupt entry is removed
  and the miss is counted); a missing/malformed file is also a fail-closed miss
  rather than returning bytes under the wrong hash.
- ``hit_rate`` tracks hits and misses across both ``lookup`` and ``store``
  (a ``store`` for an already-present hash is counted as a hit).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path

from quant_platform.app.contracts import CacheEvictionPolicy

__all__ = ["LocalArtifactCacheImpl", "ChecksumMismatchError"]


class ChecksumMismatchError(ValueError):
    """Raised when ``store`` data does not match the ``content_hash`` key."""


_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


def _is_content_hash(name: str) -> bool:
    return (
        len(name) == 64
        and all(ch in _SHA256_HEX_CHARS for ch in name)
    )


class LocalArtifactCacheImpl:
    """On-disk LRU ``LocalArtifactCache`` keyed by ``content_hash``."""

    eviction_policy = CacheEvictionPolicy.LRU

    def __init__(self, root: str | Path, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        # content_hash -> (size, last_access_ns), insertion order == recency order.
        self._index: "OrderedDict[str, tuple[int, int]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._reindex()

    # ---- LocalArtifactCache port -----------------------------------------

    def lookup(self, content_hash: str) -> bytes | None:
        with self._lock:
            if content_hash not in self._index:
                self._misses += 1
                return None
            path = self.root / content_hash
            try:
                data = path.read_bytes()
            except OSError:
                # Missing/unreadable → fail-closed miss (never return bytes
                # under a hash we cannot verify).
                self._index.pop(content_hash, None)
                self._misses += 1
                return None
            if hashlib.sha256(data).hexdigest() != content_hash:
                # Verification failed → corrupt entry: remove and count as miss
                # (fail-closed, never serve the wrong bytes).
                self._index.pop(content_hash, None)
                try:
                    path.unlink()
                except OSError:
                    pass
                self._misses += 1
                return None
            self._hits += 1
            self._index.move_to_end(content_hash)
            self._index[content_hash] = (len(data), time.time_ns())
            return data

    def store(self, content_hash: str, data: bytes) -> None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != content_hash:
            raise ChecksumMismatchError(
                "cache store failed: data content_hash "
                f"{actual!r} != key {content_hash!r}"
            )
        with self._lock:
            if content_hash in self._index and (self.root / content_hash).exists():
                self._hits += 1  # idempotent store for a present entry is a hit
                self._index[content_hash] = (len(data), time.time_ns())
                self._index.move_to_end(content_hash)
                return
            path = self.root / content_hash
            path.write_bytes(data)
            self._index[content_hash] = (len(data), time.time_ns())
            self._index.move_to_end(content_hash)
            self._evict_locked()

    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return float(self._hits / total) if total else 0.0

    def contains(self, content_hash: str) -> bool:
        """Convenience: True iff ``content_hash`` is currently cached."""
        with self._lock:
            return content_hash in self._index and (self.root / content_hash).exists()

    # ---- internals --------------------------------------------------------

    def _reindex(self) -> None:
        """Scan existing cache files into the LRU index (by mtime, oldest first)."""
        items: list[tuple[int, Path, int]] = []
        try:
            candidates = list(self.root.iterdir())
        except OSError:
            candidates = []
        for f in candidates:
            if f.is_file() and _is_content_hash(f.name):
                try:
                    st = f.stat()
                    items.append((st.st_mtime_ns, f, st.st_size))
                except OSError:
                    continue
        items.sort(key=lambda t: t[0])
        for _, path, size in items:
            self._index[path.name] = (int(size), int(path.stat().st_mtime_ns))

    def _evict_locked(self) -> None:
        while self._total_bytes_locked() > self.max_bytes and self._index:
            oldest_hash, _ = next(iter(self._index.items()))
            self._index.pop(oldest_hash, None)
            try:
                (self.root / oldest_hash).unlink()
            except OSError:
                pass

    def _total_bytes_locked(self) -> int:
        return sum(sz for sz, _ in self._index.values())