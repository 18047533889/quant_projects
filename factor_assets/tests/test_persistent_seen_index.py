"""
Tests for persistent seen index.
"""

import pytest
from pathlib import Path
import tempfile

from factor_assets.seen_index.persistent import PersistentSeenIndex
from factor_assets.seen_index.exact import SeenRecord


class TestPersistentSeenIndex:
    """Tests for PersistentSeenIndex."""

    def test_in_memory_basic(self):
        """Test basic in-memory operations."""
        index = PersistentSeenIndex(":memory:")

        # Record a factor
        record = index.record(
            canonical_hash="hash1",
            factor_id="factor1",
            origin="manual",
        )

        assert record.canonical_hash == "hash1"
        assert record.factor_id == "factor1"
        assert record.origin == "manual"
        assert index.is_seen("hash1")
        assert index.count() == 1

    def test_persistent_storage(self):
        """Test persistent storage to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seen.db"

            # Create index and record factor
            index1 = PersistentSeenIndex(str(db_path))
            index1.record("hash1", "factor1", "manual")
            index1.close()

            # Reopen and verify persistence
            index2 = PersistentSeenIndex(str(db_path))
            assert index2.is_seen("hash1")
            assert index2.count() == 1
            record = index2.get("hash1")
            assert record.factor_id == "factor1"
            index2.close()

    def test_duplicate_record_returns_existing(self):
        """Test that duplicate records return existing record."""
        index = PersistentSeenIndex(":memory:")

        record1 = index.record("hash1", "factor1", "manual")
        record2 = index.record("hash1", "factor2", "llm")  # Different factor_id

        # Should return original record
        assert record1.factor_id == record2.factor_id == "factor1"
        assert record1.origin == record2.origin == "manual"
        assert index.count() == 1

    def test_get_all_ordered(self):
        """Test get_all returns records ordered by first_seen_at."""
        index = PersistentSeenIndex(":memory:")

        index.record("hash1", "factor1", "manual")
        index.record("hash2", "factor2", "llm")
        index.record("hash3", "factor3", "search")

        records = index.get_all()
        assert len(records) == 3
        assert records[0].factor_id == "factor1"
        assert records[1].factor_id == "factor2"
        assert records[2].factor_id == "factor3"

    def test_structural_hash_optional(self):
        """Test that structural_hash is optional."""
        index = PersistentSeenIndex(":memory:")

        record = index.record(
            canonical_hash="hash1",
            factor_id="factor1",
            origin="manual",
            structural_hash="struct1",
        )

        assert record.structural_hash == "struct1"

        record2 = index.record(
            canonical_hash="hash2",
            factor_id="factor2",
            origin="manual",
        )

        assert record2.structural_hash is None

    def test_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seen.db"

            with PersistentSeenIndex(str(db_path)) as index:
                index.record("hash1", "factor1", "manual")
                assert index.is_seen("hash1")

            # Should be closed after exiting context
            # Reopen to verify persistence
            with PersistentSeenIndex(str(db_path)) as index:
                assert index.is_seen("hash1")

    def test_empty_canonical_hash_raises(self):
        """Test that empty canonical_hash raises ValueError."""
        index = PersistentSeenIndex(":memory:")

        with pytest.raises(ValueError, match="canonical_hash is required"):
            index.record("", "factor1", "manual")

    def test_empty_factor_id_raises(self):
        """Test that empty factor_id raises ValueError."""
        index = PersistentSeenIndex(":memory:")

        with pytest.raises(ValueError, match="factor_id is required"):
            index.record("hash1", "", "manual")

    def test_get_nonexistent(self):
        """Test getting nonexistent record returns None."""
        index = PersistentSeenIndex(":memory:")

        assert index.get("nonexistent") is None
        assert not index.is_seen("nonexistent")

    def test_multiple_factors_same_origin(self):
        """Test recording multiple factors with same origin."""
        index = PersistentSeenIndex(":memory:")

        index.record("hash1", "factor1", "llm", "run-001")
        index.record("hash2", "factor2", "llm", "run-001")
        index.record("hash3", "factor3", "llm", "run-002")

        assert index.count() == 3

        records = index.get_all()
        llm_records = [r for r in records if r.origin == "llm"]
        assert len(llm_records) == 3
