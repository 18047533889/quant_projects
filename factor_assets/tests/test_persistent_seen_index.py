"""
Tests for persistent seen index.
"""

import pytest
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

from factor_assets.seen_index.persistent import PersistentSeenIndex
from factor_assets.seen_index.exact import SeenRecord


class _AttemptTrackingLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.writer_attempted = threading.Event()

    def __enter__(self):
        if threading.current_thread().name in {"writer", "closer"}:
            self.writer_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


class _BlockingSelectConnection:
    def __init__(self, connection):
        self.connection = connection
        self.select_entered = threading.Event()
        self.release_select = threading.Event()

    def execute(self, sql, parameters=()):
        if sql.lstrip().upper().startswith("SELECT") and not self.select_entered.is_set():
            self.select_entered.set()
            if not self.release_select.wait(timeout=5):
                raise TimeoutError("test did not release blocked SELECT")
        return self.connection.execute(sql, parameters)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.connection.__exit__(exc_type, exc_value, traceback)

    def close(self):
        return self.connection.close()



class TestPersistentSeenIndex:
    """Tests for PersistentSeenIndex."""
    @pytest.mark.parametrize("read_name", ["is_seen", "get", "get_all", "count"])
    def test_reads_serialize_with_record_on_shared_connection(self, read_name):

        index = PersistentSeenIndex(":memory:")
        index.record("seed", "seed-factor")
        lock = _AttemptTrackingLock()
        connection = _BlockingSelectConnection(index._conn)
        index._conn = connection
        index._lock = lock
        results, errors = {}, []
        reader_done = threading.Event()
        writer_done = threading.Event()

        def read():
            try:
                readers = {
                    "is_seen": lambda: index.is_seen("seed"),
                    "get": lambda: index.get("seed"),
                    "get_all": index.get_all,
                    "count": index.count,
                }
                results["read"] = readers[read_name]()
            except BaseException as exc:
                errors.append(exc)
            finally:
                reader_done.set()

        def write():
            try:
                results["write"] = index.record("new", "new-factor")
            except BaseException as exc:
                errors.append(exc)
            finally:
                writer_done.set()

        reader = threading.Thread(target=read, name="reader")
        reader.start()
        assert connection.select_entered.wait(timeout=2)
        writer = threading.Thread(target=write, name="writer")
        writer.start()
        assert lock.writer_attempted.wait(timeout=2)
        assert not writer_done.wait(timeout=0.05)
        connection.release_select.set()
        reader.join(timeout=2)
        writer.join(timeout=2)

        assert reader_done.is_set() and writer_done.is_set()
        assert errors == []

        if read_name == "is_seen":
            assert results["read"] is True
        elif read_name == "get":
            assert results["read"].canonical_hash == "seed"
        elif read_name == "get_all":
            assert [record.canonical_hash for record in results["read"]] == ["seed"]
        else:
            assert results["read"] == 1
        assert results["write"].canonical_hash == "new"
        assert index.count() == 2
        index.close()

    def test_close_waits_for_active_read_on_shared_connection(self):
        index = PersistentSeenIndex(":memory:")
        index.record("seed", "seed-factor")
        lock = _AttemptTrackingLock()
        connection = _BlockingSelectConnection(index._conn)
        index._conn = connection
        index._lock = lock
        results, errors = [], []
        close_done = threading.Event()

        def read():
            try:
                results.append(index.count())
            except BaseException as exc:
                errors.append(exc)

        def close():
            try:
                index.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                close_done.set()

        reader = threading.Thread(target=read, name="reader")
        reader.start()
        assert connection.select_entered.wait(timeout=2)
        closer = threading.Thread(target=close, name="closer")
        closer.start()
        assert lock.writer_attempted.wait(timeout=2)
        assert not close_done.wait(timeout=0.05)
        connection.release_select.set()
        reader.join(timeout=2)
        closer.join(timeout=2)

        assert results == [1]
        assert errors == []
        assert close_done.is_set()
        assert index._conn is None

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

    def test_factor_id_collision_is_rejected(self):
        index = PersistentSeenIndex(":memory:")
        index.record("hash1", "factor1")

        with pytest.raises(ValueError, match="factor_id already recorded with canonical_hash hash1"):
            index.record("hash2", "factor1")

        assert index.count() == 1
        assert index.get("hash1").factor_id == "factor1"
        assert index.get("hash2") is None

    def test_legacy_duplicate_factor_ids_fail_closed_without_data_loss(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                CREATE TABLE seen_factors (
                    canonical_hash TEXT PRIMARY KEY,
                    factor_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    origin_ref TEXT,
                    structural_hash TEXT
                )
                """
            )
            connection.executemany(
                "INSERT INTO seen_factors VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("hash1", "factor1", "2024-01-01T00:00:00+00:00", "manual", None, None),
                    ("hash2", "factor1", "2024-01-02T00:00:00+00:00", "manual", None, None),
                ],
            )

        with pytest.raises(sqlite3.IntegrityError, match="seen_factors.factor_id"):
            PersistentSeenIndex(str(db_path))

        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "SELECT canonical_hash, factor_id FROM seen_factors ORDER BY canonical_hash"
            ).fetchall()
            unique_index = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'uq_factor_id'"
            ).fetchone()
        assert rows == [("hash1", "factor1"), ("hash2", "factor1")]
        assert unique_index is None

    def test_concurrent_writers_same_hash_are_idempotent(self, tmp_path):
        db_path = tmp_path / "seen.db"
        barrier = threading.Barrier(4)

        def write(writer_id):
            try:
                index = PersistentSeenIndex(str(db_path))
            except Exception:
                barrier.abort()
                raise
            with index:
                barrier.wait(timeout=5)
                return index.record("shared", f"factor-{writer_id}", f"origin-{writer_id}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(write, range(4)))

        assert len({record.first_seen_at for record in records}) == 1
        assert len({record.factor_id for record in records}) == 1
        with PersistentSeenIndex(str(db_path)) as index:
            assert index.count() == 1

    def test_record_retries_busy_error(self, monkeypatch):
        index = PersistentSeenIndex(":memory:", busy_timeout_ms=0, max_busy_retries=2)
        class ConnectionProxy:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, parameters=()):
                return self.connection.execute(sql, parameters)

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.connection.__exit__(exc_type, exc_val, exc_tb)

        proxy = ConnectionProxy(index._conn)
        original_execute = proxy.execute
        attempts = 0

        def flaky_execute(sql, parameters=()):
            nonlocal attempts
            if "INSERT INTO seen_factors" in sql and attempts == 0:
                attempts += 1
                raise sqlite3.OperationalError("database is locked")
            return original_execute(sql, parameters)

        proxy.execute = flaky_execute
        index._conn = proxy
        record = index.record("hash1", "factor1")

        assert attempts == 1
        assert record.factor_id == "factor1"
        assert index.count() == 1

    def test_record_rolls_back_failed_insert(self):
        index = PersistentSeenIndex(":memory:")

        class ConnectionProxy:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, parameters=()):
                cursor = self.connection.execute(sql, parameters)
                if "INSERT INTO seen_factors" in sql:
                    raise RuntimeError("injected failure")
                return cursor

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.connection.__exit__(exc_type, exc_val, exc_tb)

        index._conn = ConnectionProxy(index._conn)
        with pytest.raises(RuntimeError, match="injected failure"):
            index.record("hash1", "factor1")

        assert index.count() == 0

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
