"""Tests for SeenCache."""

import pytest
from datetime import datetime
from factor_optimizer.seen.identity import SeenCache, SeenRecord


def test_seen_record_creation():
    """Test SeenRecord creation."""
    record = SeenRecord(
        canonical_hash="hash123",
        first_seen_at=datetime.now(),
        trial_id="trial_001",
        factor_id="factor_001",
    )

    assert record.canonical_hash == "hash123"
    assert record.trial_id == "trial_001"
    assert record.factor_id == "factor_001"


def test_seen_record_serialization():
    """Test SeenRecord serialization."""
    record = SeenRecord(
        canonical_hash="hash123",
        first_seen_at=datetime(2026, 8, 13, 10, 0, 0),
        trial_id="t1",
        metadata={"note": "test"},
    )

    serialized = record.to_dict()
    assert serialized["canonical_hash"] == "hash123"
    assert isinstance(serialized["first_seen_at"], str)

    restored = SeenRecord.from_dict(serialized)
    assert restored.canonical_hash == record.canonical_hash
    assert restored.trial_id == record.trial_id


def test_seen_cache_creation():
    """Test SeenCache creation."""
    cache = SeenCache()
    assert cache.size() == 0


def test_seen_cache_mark_seen():
    """Test marking a factor as seen."""
    cache = SeenCache()

    record = cache.mark_seen("hash1", "trial1", factor_id="f1")
    assert record.canonical_hash == "hash1"
    assert record.trial_id == "trial1"
    assert record.factor_id == "f1"
    assert cache.size() == 1


def test_seen_cache_is_seen():
    """Test checking if hash is seen."""
    cache = SeenCache()

    assert not cache.is_seen("hash1")

    cache.mark_seen("hash1", "trial1")
    assert cache.is_seen("hash1")


def test_seen_cache_get_record():
    """Test retrieving seen record."""
    cache = SeenCache()

    cache.mark_seen("hash1", "trial1", factor_id="f1")

    record = cache.get_record("hash1")
    assert record is not None
    assert record.canonical_hash == "hash1"

    missing = cache.get_record("nonexistent")
    assert missing is None


def test_seen_cache_get_by_trial():
    """Test retrieving record by trial ID."""
    cache = SeenCache()

    cache.mark_seen("hash1", "trial1", factor_id="f1")

    record = cache.get_by_trial("trial1")
    assert record is not None
    assert record.canonical_hash == "hash1"

    missing = cache.get_by_trial("nonexistent")
    assert missing is None


def test_seen_cache_duplicate_hash():
    """Test that duplicate hashes return existing record."""
    cache = SeenCache()

    record1 = cache.mark_seen("hash1", "trial1", factor_id="f1")
    record2 = cache.mark_seen("hash1", "trial2")

    # Should return same record
    assert record1 is record2
    assert cache.size() == 1


def test_seen_cache_update_factor_id():
    """Test updating factor_id on existing record."""
    cache = SeenCache()

    record1 = cache.mark_seen("hash1", "trial1")
    assert record1.factor_id is None

    record2 = cache.mark_seen("hash1", "trial2", factor_id="f1")
    assert record2.factor_id == "f1"
    assert record1 is record2


def test_seen_cache_clear():
    """Test clearing the cache."""
    cache = SeenCache()

    cache.mark_seen("hash1", "trial1")
    cache.mark_seen("hash2", "trial2")
    assert cache.size() == 2

    cache.clear()
    assert cache.size() == 0
    assert not cache.is_seen("hash1")


def test_seen_cache_export_import():
    """Test exporting and importing records."""
    cache1 = SeenCache()
    cache1.mark_seen("hash1", "trial1", factor_id="f1")
    cache1.mark_seen("hash2", "trial2")

    exported = cache1.export_records()
    assert len(exported) == 2

    cache2 = SeenCache()
    cache2.import_records(exported)
    assert cache2.size() == 2
    assert cache2.is_seen("hash1")
    assert cache2.is_seen("hash2")


def test_seen_cache_compute_hash_no_adapter():
    """Test that compute_canonical_hash fails without adapter."""
    cache = SeenCache()

    with pytest.raises(RuntimeError, match="FE adapter required"):
        cache.compute_canonical_hash("some_factor")


def test_seen_cache_compute_hash_with_adapter():
    """Test compute_canonical_hash with mock adapter."""

    class MockAdapter:
        def compute_canonical_hash(self, factor_def):
            return f"hash_{factor_def}"

    cache = SeenCache(fe_adapter=MockAdapter())
    hash_value = cache.compute_canonical_hash("factor123")

    assert hash_value == "hash_factor123"


def test_seen_cache_check_and_mark():
    """Test check_and_mark workflow."""

    class MockAdapter:
        def compute_canonical_hash(self, factor_def):
            return f"hash_{factor_def}"

    cache = SeenCache(fe_adapter=MockAdapter())

    # First time: not seen, should mark
    was_seen, record = cache.check_and_mark("factor1", "trial1")
    assert not was_seen
    assert record.canonical_hash == "hash_factor1"
    assert cache.size() == 1

    # Second time: seen
    was_seen, record = cache.check_and_mark("factor1", "trial2")
    assert was_seen
    assert record.trial_id == "trial1"  # Original trial
    assert cache.size() == 1  # No new record


def test_seen_cache_multiple_trials():
    """Test tracking multiple distinct factors."""
    cache = SeenCache()

    cache.mark_seen("hash1", "trial1")
    cache.mark_seen("hash2", "trial2")
    cache.mark_seen("hash3", "trial3")

    assert cache.size() == 3
    assert cache.is_seen("hash1")
    assert cache.is_seen("hash2")
    assert cache.is_seen("hash3")
