"""
Test seen index for exact history tracking.
"""

import pytest

from factor_assets.seen_index import SeenRecord, SeenIndex


def test_seen_record_creation():
    """Test SeenRecord creation."""
    record = SeenRecord(
        canonical_hash="abc123",
        factor_id="F001",
        first_seen_at="2024-01-01T00:00:00Z",
        origin="manual",
        origin_ref="user_123",
        structural_hash="struct_xyz",
    )

    assert record.canonical_hash == "abc123"
    assert record.factor_id == "F001"
    assert record.origin == "manual"


def test_seen_record_requires_fields():
    """SeenRecord must have canonical_hash, factor_id, first_seen_at."""
    with pytest.raises(ValueError, match="canonical_hash"):
        SeenRecord(
            canonical_hash="",
            factor_id="F001",
            first_seen_at="2024-01-01T00:00:00Z",
            origin="manual",
        )

    with pytest.raises(ValueError, match="factor_id"):
        SeenRecord(
            canonical_hash="abc123",
            factor_id="",
            first_seen_at="2024-01-01T00:00:00Z",
            origin="manual",
        )


def test_seen_index_record():
    """Test recording a factor as seen."""
    index = SeenIndex()

    record = index.record(
        canonical_hash="abc123",
        factor_id="F001",
        origin="manual",
    )

    assert record.canonical_hash == "abc123"
    assert record.factor_id == "F001"
    assert record.first_seen_at
    assert index.count() == 1


def test_seen_index_is_seen():
    """Test checking if a factor has been seen."""
    index = SeenIndex()

    assert not index.is_seen("abc123")

    index.record(canonical_hash="abc123", factor_id="F001")

    assert index.is_seen("abc123")


def test_seen_index_preserves_first_seen():
    """SeenIndex should preserve first-seen timestamp."""
    index = SeenIndex()

    # Record first time
    first_record = index.record(
        canonical_hash="abc123",
        factor_id="F001",
        origin="manual",
    )
    first_timestamp = first_record.first_seen_at

    # Try to record again with different info
    second_record = index.record(
        canonical_hash="abc123",
        factor_id="F999",  # Different ID (shouldn't matter)
        origin="llm",      # Different origin (shouldn't matter)
    )

    # Should return the original record
    assert second_record.first_seen_at == first_timestamp
    assert second_record.factor_id == "F001"  # Original ID preserved
    assert second_record.origin == "manual"   # Original origin preserved


def test_seen_index_get():
    """Test retrieving seen records."""
    index = SeenIndex()

    index.record(canonical_hash="abc123", factor_id="F001")

    record = index.get("abc123")
    assert record is not None
    assert record.factor_id == "F001"

    not_found = index.get("xyz999")
    assert not_found is None


def test_seen_index_get_all():
    """Test retrieving all seen records."""
    index = SeenIndex()

    index.record(canonical_hash="abc123", factor_id="F001")
    index.record(canonical_hash="def456", factor_id="F002")
    index.record(canonical_hash="ghi789", factor_id="F003")

    all_records = index.get_all()
    assert len(all_records) == 3

    hashes = {r.canonical_hash for r in all_records}
    assert hashes == {"abc123", "def456", "ghi789"}


def test_seen_index_count():
    """Test counting seen factors."""
    index = SeenIndex()

    assert index.count() == 0

    index.record(canonical_hash="abc123", factor_id="F001")
    assert index.count() == 1

    index.record(canonical_hash="def456", factor_id="F002")
    assert index.count() == 2

    # Recording same hash again doesn't increase count
    index.record(canonical_hash="abc123", factor_id="F001")
    assert index.count() == 2


def test_seen_index_multiple_origins():
    """Test tracking factors from different origins."""
    index = SeenIndex()

    index.record(canonical_hash="manual_001", factor_id="F001", origin="manual")
    index.record(canonical_hash="llm_001", factor_id="F002", origin="llm")
    index.record(canonical_hash="search_001", factor_id="F003", origin="search")
    index.record(canonical_hash="corpus_001", factor_id="F004", origin="corpus")

    assert index.count() == 4

    manual_record = index.get("manual_001")
    assert manual_record.origin == "manual"

    llm_record = index.get("llm_001")
    assert llm_record.origin == "llm"


def test_seen_index_immutable_records():
    """Seen records should be immutable."""
    index = SeenIndex()

    record = index.record(canonical_hash="abc123", factor_id="F001")

    with pytest.raises(Exception):  # FrozenInstanceError
        record.factor_id = "F999"  # type: ignore


def test_seen_index_with_structural_hash():
    """Test seen index with structural hash."""
    index = SeenIndex()

    record = index.record(
        canonical_hash="abc123",
        factor_id="F001",
        structural_hash="struct_xyz",
    )

    assert record.structural_hash == "struct_xyz"

    retrieved = index.get("abc123")
    assert retrieved.structural_hash == "struct_xyz"
