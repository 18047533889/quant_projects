"""QRP-P2 — LocalArtifactCacheImpl tests (max_bytes + LRU + hit_rate).

Cover:
(a) hit/miss + hit_rate accounting;
(b) LRU eviction past ``max_bytes`` (oldest entries evicted);
(c) ``store`` computes sha256 and fails closed (ChecksumMismatchError) when
    data does not match the key — nothing lands on disk;
(d) ``lookup`` verifies bytes; a corrupt/verification-failing entry is treated
    as a fail-closed miss and removed (never serves wrong bytes);
(e) reindex picks up pre-existing cache files and serves them.
"""

from __future__ import annotations

import hashlib

import pytest

from quant_platform.app.storage.cache import (
    ChecksumMismatchError,
    LocalArtifactCacheImpl,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- (a) hit_rate accounting ----------------------------------------------
def test_hit_rate_accounting(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache")
    c1 = sha256_bytes(b"alpha")
    assert cache.lookup(c1) is None
    assert cache.hit_rate() == 0.0

    cache.store(c1, b"alpha")
    assert cache.lookup(c1) == b"alpha"
    # 1 hit / 2 total
    assert cache.hit_rate() == 0.5

    assert cache.lookup(c1) == b"alpha"
    # 2 hits / 3 total
    assert cache.hit_rate() == pytest.approx(2 / 3)


# ---- (b) LRU eviction -----------------------------------------------------
def test_lru_eviction_past_max_bytes(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache", max_bytes=40)
    # "first" = 5 bytes, "second" = 6 bytes, filler = 35 bytes → each store fills
    # up to 46 after the second, past 40, evicting the least-recently-used.
    c_first = sha256_bytes(b"first")
    c_second = sha256_bytes(b"second")
    c_fill = sha256_bytes(b"x" * 35)
    cache.store(c_first, b"first")
    cache.store(c_second, b"second")
    cache.store(c_fill, b"x" * 35)

    assert cache.lookup(c_first) is None  # LRU evicted
    assert cache.lookup(c_second) is None  # evicted
    assert cache.lookup(c_fill) == b"x" * 35  # still resident

    # Accessing c_fill refreshes it; storing a new big entry evicts c_fill later.
    cache.store(sha256_bytes(b"z" * 38), b"z" * 38)
    assert cache.lookup(c_fill) is None


def test_lru_recency_respected(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache", max_bytes=60)
    c_a = sha256_bytes(b"a" * 20)
    c_b = sha256_bytes(b"b" * 20)
    c_c = sha256_bytes(b"c" * 20)
    cache.store(c_a, b"a" * 20)
    cache.store(c_b, b"b" * 20)
    cache.store(c_c, b"c" * 20)  # 60 == max_bytes, nothing evicted yet
    assert cache.lookup(c_a) == b"a" * 20  # refresh a → a is most recent

    # Next store (20 bytes) exceeds 60 → evict least-recently-used = b.
    cache.store(sha256_bytes(b"d" * 20), b"d" * 20)
    assert cache.lookup(c_b) is None  # evicted
    assert cache.lookup(c_a) == b"a" * 20  # recency saved it


# ---- (c) store checksum fail-closed ---------------------------------------
def test_store_checksum_fail_closed(tmp_path):
    cache = LocalArtifactCacheImpl(tmp_path / "cache")
    good_hash = sha256_bytes(b"x")
    with pytest.raises(ChecksumMismatchError):
        cache.store(good_hash, b"does-not-match")
    # Nothing was persisted under the mismatched key.
    assert cache.lookup(good_hash) is None


def test_chechksum_error_is_value_error():
    # A ValueError subclass so callers can catch broadly (baseline contract).
    assert issubclass(ChecksumMismatchError, ValueError)


# ---- (d) lookup verifies; corrupt entry fail-closed ------------------------
def test_lookup_corrupt_entry_fail_closed(tmp_path):
    root = tmp_path / "cache"
    cache = LocalArtifactCacheImpl(root)
    c1 = sha256_bytes(b"verifiable-payload")
    cache.store(c1, b"verifiable-payload")
    assert cache.lookup(c1) == b"verifiable-payload"

    # Tamper the on-disk file behind the cache's back.
    (root / c1).write_bytes(b"EVIL TAMPERED BYTES")

    # Verification fails → treated as a miss, never served.
    assert cache.lookup(c1) is None
    # Corrupt entry is removed from the index and disk.
    assert cache.contains(c1) is False
    assert not (root / c1).exists()

    # One hit counted (before tamper) + one miss (corrupt lookup) → 1/2.
    assert cache.hit_rate() == pytest.approx(0.5)


# ---- (e) reindex picks up pre-existing files -------------------------------
def test_reindex_recovers_existing_files(tmp_path):
    root = tmp_path / "cache"
    c1 = sha256_bytes(b"pre-existing")
    root.mkdir(parents=True, exist_ok=True)
    (root / c1).write_bytes(b"pre-existing")

    cache = LocalArtifactCacheImpl(root)
    assert cache.lookup(c1) == b"pre-existing"
    assert cache.hit_rate() == 1.0