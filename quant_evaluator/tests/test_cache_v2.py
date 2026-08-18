"""
Tests for advanced cache system.

Tests multi-level caching, compression, invalidation, and warming strategies.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
import pytest

import quant_evaluator.runtime.cache_v2 as cache_v2_module
from quant_evaluator.runtime.cache_v2 import (
    MultiLevelCache,
    MemoryCacheLayer,
    DiskCacheLayer,
    RedisCacheLayer,
    create_compressor,
    ZlibCompressor,
    NoCompressor,
    CacheMetadata,
    CacheConfigurationError,
    CacheV2Config,
    create_cache,
    MostRecentKeysStrategy,
)


class TestCompression:
    """Test compression functionality."""

    def test_zlib_compression(self):
        """Test zlib compression and decompression."""
        compressor = ZlibCompressor(level=6)
        data = b"hello world" * 1000

        compressed, stats = compressor.compress(data)
        assert len(compressed) < len(data)
        assert stats.ratio > 1.0
        assert stats.savings_pct > 0

        decompressed = compressor.decompress(compressed)
        assert decompressed == data

    def test_no_compression(self):
        """Test no-op compressor."""
        compressor = NoCompressor()
        data = b"hello world"

        compressed, stats = compressor.compress(data)
        assert compressed == data
        assert stats.ratio == 1.0
        assert stats.savings_pct == 0.0

    def test_create_compressor(self):
        """Test compressor factory."""
        comp = create_compressor("zlib")
        assert isinstance(comp, ZlibCompressor)

        comp = create_compressor("none")
        assert isinstance(comp, NoCompressor)


    def test_no_compression_round_trip(self):
        cache = MultiLevelCache(
            compression="none",
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )
        assert cache.put("key", "value")
        assert cache.get("key") == "value"

    def test_basic_operations(self):
        """Test get, put, and invalidate."""
        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        # Put and get
        data = np.random.randn(1000)
        assert cache.put("key1", data)

        result = cache.get("key1")
        assert result is not None
        value, metadata = result
        np.testing.assert_array_equal(value, data)
        assert metadata.key == "key1"
        assert metadata.access_count == 1

        # Second get updates access count
        result = cache.get("key1")
        assert result[1].access_count == 2

        # Invalidate
        assert cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        data = {"value": 42}
        cache.put("key1", data, ttl_seconds=0.1)

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_ttl_expiration_at_exact_boundary(self, monkeypatch):
        """An entry expires exactly when its absolute TTL is reached."""
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=create_compressor("none"),
            enable_compression=False,
        )

        assert cache.put("key1", "value", ttl_seconds=10.0)
        clock.now = 1009.999
        assert cache.get("key1") is not None
        clock.now = 1010.0
        assert cache.get("key1") is None

    def test_direct_put_captures_origin_before_slow_serialization(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def slow_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 5.0
            return payload

        monkeypatch.setattr(cache_v2_module.pickle, "dumps", slow_dumps)
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=create_compressor("none"),
            enable_compression=False,
        )

        assert cache.put("key", "value", ttl_seconds=10.0)
        result = cache.get("key")
        assert result is not None
        assert result[1].created_at == 1000.0
        assert result[1].is_expired(1010.0)

    def test_direct_put_rejects_ttl_exhausted_during_serialization(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def slow_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 10.0
            return payload

        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=create_compressor("none"),
            enable_compression=False,
        )
        assert cache.put("key", "old")

        monkeypatch.setattr(cache_v2_module.pickle, "dumps", slow_dumps)
        assert cache.put("key", "new", ttl_seconds=10.0) is False
        result = cache.get("key")
        assert result is not None
        assert result[0] == "old"

    @pytest.mark.parametrize("ttl", [0.0, -1.0, float("nan"), float("inf"), True, False])
    def test_nonpositive_or_nonfinite_ttl_is_rejected(self, ttl):
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=create_compressor("none"),
            enable_compression=False,
        )

        assert cache.put("key1", "value", ttl_seconds=ttl) is False
        assert cache.get("key1") is None
    @pytest.mark.parametrize("created_at", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_creation_time_expires_fail_closed(self, created_at):
        metadata = CacheMetadata(
            key="key1",
            created_at=created_at,
            last_accessed=created_at,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=10.0,
            dependencies=set(),
        )

        assert metadata.is_expired(1000.0) is True

    def test_future_creation_time_expires_fail_closed(self):
        metadata = CacheMetadata(
            key="key1",
            created_at=1001.0,
            last_accessed=1001.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=10.0,
            dependencies=set(),
        )

        assert metadata.is_expired(1000.0) is True

    def test_from_dict_rejects_retrograde_last_accessed(self):
        data = CacheMetadata(
            key="key1", created_at=100.0, last_accessed=100.0,
            access_count=0, size_bytes=1, compressed_size=1,
            ttl_seconds=10.0, dependencies=set(),
        ).to_dict()
        data["last_accessed"] = 99.0

        with pytest.raises(ValueError, match="last_accessed precedes created_at"):
            CacheMetadata.from_dict(data)

    def test_from_dict_accepts_equal_timestamps(self):
        data = CacheMetadata(
            key="key1", created_at=100.0, last_accessed=100.0,
            access_count=0, size_bytes=1, compressed_size=1,
            ttl_seconds=10.0, dependencies=set(),
        ).to_dict()

        metadata = CacheMetadata.from_dict(data)
        assert metadata.created_at == metadata.last_accessed == 100.0

    def test_lru_eviction(self):
        """Test LRU eviction under memory pressure."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=500,  # Very small cache to force eviction
            compressor=compressor,
            enable_compression=False,
        )

        # Fill cache with items
        for i in range(20):
            cache.put(f"key_{i}", f"value_{i}" * 10)

        stats = cache.get_stats()
        # Should have evicted some entries
        assert stats["num_entries"] < 20
        assert stats["utilization"] <= 1.0

    def test_dependency_invalidation(self):
        """Test invalidating entries by dependency."""
        compressor = create_compressor("none")
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        # Put entries with dependencies
        cache.put("key1", "value1", dependencies={"dep_a"})
        cache.put("key2", "value2", dependencies={"dep_a", "dep_b"})
        cache.put("key3", "value3", dependencies={"dep_b"})
        cache.put("key4", "value4", dependencies=set())

        # Invalidate dep_a
        count = cache.invalidate_dependencies("dep_a")
        assert count == 2  # key1 and key2

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_dependency_invalidation_snapshots_caller_set(self):
        """Caller mutation must not remove stored dependency edges."""
        dependencies = {"source"}
        cache = MemoryCacheLayer(
            max_size_bytes=1024 * 1024,
            compressor=create_compressor("none"),
            enable_compression=False,
        )

        assert cache.put("key", "value", dependencies=dependencies)
        dependencies.clear()

        assert cache.invalidate_dependencies("source") == 1
        assert cache.get("key") is None

        """Test that compression reduces memory usage."""
        compressor = create_compressor("zlib")

        cache_compressed = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        cache_uncompressed = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=False,
        )

        # Create compressible data
        data = np.zeros(10000)  # Highly compressible

        cache_compressed.put("key1", data)
        cache_uncompressed.put("key1", data)

        stats_compressed = cache_compressed.get_stats()
        stats_uncompressed = cache_uncompressed.get_stats()

        # Compressed should use less memory
        assert stats_compressed["current_size_bytes"] < stats_uncompressed["current_size_bytes"]
        assert stats_compressed["memory_savings_pct"] > 50  # Should be >50% savings


class TestDiskCacheLayer:
    """Test L2 disk cache layer."""

    def test_basic_operations(self):
        """Test disk cache put and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            data = np.random.randn(1000)
            assert cache.put("key1", data)

            result = cache.get("key1")
            assert result is not None
            value, metadata = result
            np.testing.assert_array_equal(value, data)

    def test_malformed_record_is_quarantined(self):
        """Malformed single-file records must not remain readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "value")
            cache_path = cache._key_path("key1")
            cache_path.write_bytes(b"{not-json")

            assert cache.get("key1") is None
            assert not cache_path.exists()

    def test_dependency_invalidation_uses_physical_record_when_metadata_key_is_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "value", dependencies={"dep"})
            cache_path = cache._key_path("key1")
            metadata, compressed = cache._decode_record(cache_path.read_bytes())
            metadata.key = "forged-key"
            cache_path.write_bytes(cache._encode_record(metadata, compressed))

            assert cache.invalidate_dependencies("dep") == 1
            assert not cache_path.exists()
            assert cache.get("key1") is None

    def test_dependency_invalidation_preserves_replacement_after_scan(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "old", dependencies={"dep"})
            cache_path = cache._key_path("key1")
            scanned = threading.Event()
            resume = threading.Event()
            original = cache._decode_record
            calls = [0]

            def pause_after_scan(raw):
                decoded = original(raw)
                if calls[0] == 0:
                    calls[0] += 1
                    scanned.set()
                    assert resume.wait(timeout=2.0)
                return decoded

            monkeypatch.setattr(cache, "_decode_record", pause_after_scan)
            results = []
            worker = threading.Thread(target=lambda: results.append(
                cache._invalidate_dependencies_unlocked("dep")))
            worker.start()
            assert scanned.wait(timeout=2.0)
            assert cache.put("key1", "new", dependencies={"other"})
            resume.set()
            worker.join(timeout=2.0)

            assert not worker.is_alive()
            assert results == [0]
            assert cache.get("key1")[0] == "new"

    def test_malformed_quarantine_preserves_replacement_after_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "old")
            cache_path = cache._key_path("key1")
            cache_path.write_bytes(b"{not-json")
            assert cache.get("key1") is None
            assert not cache_path.exists()
            assert cache.put("key1", "new")
            assert cache.get("key1")[0] == "new"

    def test_dependency_invalidation_preserves_value_only_replacement(self, monkeypatch):
        """A changed immutable record must survive cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "old", dependencies={"dep"})
            cache_path = cache._key_path("key1")
            scanned = threading.Event()
            resume = threading.Event()
            original = cache._decode_record
            calls = [0]

            def pause_after_scan(raw):
                decoded = original(raw)
                if calls[0] == 0:
                    calls[0] += 1
                    scanned.set()
                    assert resume.wait(timeout=2.0)
                return decoded

            monkeypatch.setattr(cache, "_decode_record", pause_after_scan)
            results = []
            worker = threading.Thread(target=lambda: results.append(
                cache._invalidate_dependencies_unlocked("dep")))
            worker.start()
            assert scanned.wait(timeout=2.0)
            cache_path.write_bytes(b"replacement")
            resume.set()
            worker.join(timeout=2.0)

            assert not worker.is_alive()
            assert results == [0]
            assert cache_path.read_bytes() == b"replacement"

    def test_malformed_quarantine_preserves_value_only_replacement(self):
        """Malformed immutable records must not delete a replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key1", "old", dependencies={"dep"})
            cache_path = cache._key_path("key1")
            cache_path.write_bytes(b"{not-json")
            assert cache.invalidate_dependencies("dep") == 0
            cache_path.write_bytes(b"replacement")
            assert cache_path.read_bytes() == b"replacement"


    def test_direct_put_captures_origin_before_slow_serialization(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def slow_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 5.0
            return payload

        monkeypatch.setattr(cache_v2_module.pickle, "dumps", slow_dumps)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key", "value", ttl_seconds=10.0)
            result = cache.get("key")
            assert result is not None
            assert result[1].created_at == 1000.0
            assert result[1].is_expired(1010.0)

    def test_direct_put_rejects_ttl_exhausted_during_serialization(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def slow_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 10.0
            return payload

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert cache.put("key", "old")
            monkeypatch.setattr(cache_v2_module.pickle, "dumps", slow_dumps)
            assert cache.put("key", "new", ttl_seconds=10.0) is False
            result = cache.get("key")
            assert result is not None
            assert result[0] == "old"

    def test_equivalent_roots_share_entry_lock(self, tmp_path):
        real_root = tmp_path / "cache"
        real_root.mkdir()
        alias_root = tmp_path / "cache-alias"
        alias_root.symlink_to(real_root, target_is_directory=True)
        first = DiskCacheLayer(real_root, create_compressor("none"))
        second = DiskCacheLayer(alias_root, create_compressor("none"))

        assert first._entry_lock(first._key_path("key")) is second._entry_lock(
            second._key_path("key")
        )

    def test_clear_waits_for_same_process_writer(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            entered = threading.Event()
            release = threading.Event()
            original_compress = cache.compressor.compress

            def pause_compress(data):
                entered.set()
                assert release.wait(timeout=2.0)
                return original_compress(data)

            monkeypatch.setattr(cache.compressor, "compress", pause_compress)
            writer_result = []
            writer = threading.Thread(
                target=lambda: writer_result.append(cache.put("key", "value"))
            )
            writer.start()
            assert entered.wait(timeout=2.0)

            clear_done = threading.Event()
            clearer = threading.Thread(
                target=lambda: (cache.clear(), clear_done.set())
            )
            clearer.start()
            assert not clear_done.wait(timeout=0.1)

            release.set()
            writer.join(timeout=2.0)
            clearer.join(timeout=2.0)

            assert not writer.is_alive()
            assert not clearer.is_alive()
            assert writer_result == [True]
            assert cache.get("key") is None

    @pytest.mark.skipif(cache_v2_module.fcntl is None, reason="requires POSIX flock")
    def test_clear_preserves_process_lock_inode_and_blocks_other_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = DiskCacheLayer(root, create_compressor("none"))
            assert cache.put("key", "value")
            lock_path = root / ".cache.lock"
            before = lock_path.stat().st_ino
            ready = root / "ready"
            release = root / "release"
            code = r'''
import sys, time
from pathlib import Path
from quant_evaluator.runtime.cache_v2 import DiskCacheLayer, create_compressor
root, ready, release = map(Path, sys.argv[1:])
cache = DiskCacheLayer(root, create_compressor("none"))
with cache._process_lock():
    ready.write_text("ready", encoding="utf-8")
    while not release.exists():
        time.sleep(0.002)
'''
            holder = subprocess.Popen(
                [sys.executable, "-c", code, str(root), str(ready), str(release)],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 5.0
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.002)
            assert ready.exists(), holder.stderr.read() if holder.stderr else "lock holder failed"

            clear_done = threading.Event()
            clearer = threading.Thread(target=lambda: (cache.clear(), clear_done.set()))
            clearer.start()
            assert not clear_done.wait(timeout=0.1)
            release.write_text("release", encoding="utf-8")
            clearer.join(timeout=5.0)
            stdout, stderr = holder.communicate(timeout=5.0)

            assert holder.returncode == 0, f"{stdout}\n{stderr}"
            assert not clearer.is_alive()
            assert clear_done.is_set()
            assert lock_path.stat().st_ino == before
            assert cache.get("key") is None

    @pytest.mark.skipif(cache_v2_module.fcntl is None, reason="requires POSIX flock")
    def test_process_lock_is_reentrant_for_same_thread_and_root(self, tmp_path):
        first = DiskCacheLayer(tmp_path, create_compressor("none"))
        second = DiskCacheLayer(tmp_path, create_compressor("none"))

        with first._process_lock():
            with second._process_lock():
                assert first.put("key", "value")

        assert first.get("key")[0] == "value"

    def test_same_root_instances_serialize_stale_cleanup_and_replacement(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        with tempfile.TemporaryDirectory() as tmpdir:
            first = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            second = DiskCacheLayer(Path(tmpdir), create_compressor("none"))
            assert first.put("key", "old", ttl_seconds=1.0)
            clock.now = 1002.0

            reader_started = threading.Event()
            release_reader = threading.Event()
            original_decode = first._decode_record

            def pause_decode(raw):
                decoded = original_decode(raw)
                if not reader_started.is_set():
                    reader_started.set()
                    assert release_reader.wait(timeout=2.0)
                return decoded

            monkeypatch.setattr(first, "_decode_record", pause_decode)
            reader_result = []
            reader = threading.Thread(target=lambda: reader_result.append(first.get("key")))
            writer_result = []
            writer = threading.Thread(target=lambda: writer_result.append(second.put("key", "new")))
            reader.start()
            assert reader_started.wait(timeout=2.0)
            writer.start()
            release_reader.set()
            reader.join(timeout=2.0)
            writer.join(timeout=2.0)

            assert not reader.is_alive()
            assert not writer.is_alive()
            assert writer_result == [True]
            assert first.get("key")[0] == "new"

    @pytest.mark.parametrize("created_at", [989.0, 1001.0, float("nan"), True, False])
    def test_invalid_origin_is_rejected(self, monkeypatch, created_at):
        monkeypatch.setattr(cache_v2_module.time, "time", FrozenClock(1000.0).time)
        cache = DiskCacheLayer(Path(tempfile.mkdtemp()), create_compressor("none"))
        try:
            assert cache.put("key", "value", ttl_seconds=10.0, created_at=created_at) is False
            assert cache.get("key") is None
        finally:
            cache.clear()

    def test_checksum_verification(self):
        """Test that corrupted cache files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            data = {"value": 42}
            cache.put("key1", data)

            # Corrupt the cache file
            cache_path = cache._key_path("key1")
            cache_path.write_bytes(b"corrupted data")

            # Should return None due to checksum mismatch
            result = cache.get("key1")
            assert result is None

    def test_atomic_writes(self):
        """Test that writes are atomic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("zlib")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            data = {"value": 42}
            cache.put("key1", data)

            # Verify no temp files left behind
            temp_files = list(Path(tmpdir).rglob(".tmp.*"))
            assert len(temp_files) == 0

    def test_restart_removes_record_temp_orphan(self, tmp_path):
        code = r"""
import os
import sys
from pathlib import Path
from quant_evaluator.runtime.cache_v2 import DiskCacheLayer, create_compressor

root = Path(sys.argv[1])
cache = DiskCacheLayer(root, create_compressor("none"))
original_replace = Path.replace

def crash_before_publish(path, target):
    if path.name.startswith(".tmp.") and path.suffix == ".cache":
        os._exit(17)
    return original_replace(path, target)

Path.replace = crash_before_publish
cache.put("key", "value")
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            },
        )
        assert result.returncode == 17
        assert len(list(tmp_path.rglob(".tmp.*.cache"))) == 1

        DiskCacheLayer(tmp_path, create_compressor("none"))

        assert list(tmp_path.rglob(".tmp.*.cache")) == []

    def test_failed_record_replace_preserves_previous_record(self, monkeypatch, tmp_path):
        cache = DiskCacheLayer(tmp_path, create_compressor("none"))
        assert cache.put("key", "old")
        original_replace = Path.replace

        def fail_record_replace(path, target):
            if path.name.startswith(".tmp.") and path.suffix == ".cache":
                raise OSError("injected replace failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_record_replace)
        assert cache.put("key", "new") is False
        assert cache.get("key")[0] == "old"
        assert list(tmp_path.rglob(".tmp.*.cache")) == []

    def test_failed_record_fsync_preserves_previous_record(self, monkeypatch, tmp_path):
        cache = DiskCacheLayer(tmp_path, create_compressor("none"))
        assert cache.put("key", "old")
        original_fsync = cache_v2_module.os.fsync
        calls = [0]

        def fail_first_fsync(fd):
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("injected file fsync failure")
            return original_fsync(fd)

        monkeypatch.setattr(cache_v2_module.os, "fsync", fail_first_fsync)
        assert cache.put("key", "new") is False
        assert cache.get("key")[0] == "old"
        assert list(tmp_path.rglob(".tmp.*.cache")) == []

    def test_directory_fsync_unavailable_keeps_published_record(self, monkeypatch, tmp_path):
        cache = DiskCacheLayer(tmp_path, create_compressor("none"))
        assert cache.put("key", "old")
        original_fsync = cache_v2_module.os.fsync
        calls = [0]

        def fail_directory_fsync(fd):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("injected directory fsync failure")
            return original_fsync(fd)

        monkeypatch.setattr(cache_v2_module.os, "fsync", fail_directory_fsync)
        assert cache.put("key", "new") is True
        assert cache.get("key")[0] == "new"
        assert list(tmp_path.rglob(".tmp.*.cache")) == []


class FrozenClock:
    def __init__(self, now: float):
        self.now = now

    def time(self) -> float:
        return self.now


class FakeRedisLayer:
    def __init__(self):
        self.entries = {}

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        value, metadata = entry
        if metadata.is_expired(cache_v2_module.time.time()):
            self.invalidate(key)
            return None
        return value, metadata

    def put(self, key, value, ttl_seconds=None, dependencies=None, created_at=None):
        now = cache_v2_module.time.time()
        self.entries[key] = (
            value,
            CacheMetadata(
                key=key,
                created_at=now if created_at is None else created_at,
                last_accessed=now,
                access_count=0,
                size_bytes=0,
                compressed_size=0,
                ttl_seconds=ttl_seconds,
                dependencies=dependencies or set(),
            ),
        )
        return True

    def invalidate(self, key):
        return self.entries.pop(key, None) is not None

    def invalidate_dependencies(self, dependency, *, return_keys=False):
        keys = {
            key for key, (_, metadata) in self.entries.items()
            if dependency in metadata.dependencies
        }
        for key in keys:
            self.invalidate(key)
        return keys if return_keys else len(keys)

    def clear(self):
        self.entries.clear()


class FakeRedisClient:
    def __init__(self, hashes, values):
        self.hashes = dict(hashes)
        self.values = dict(values)
        self._pipeline_commands = []

    def scan(self, cursor, match=None, count=None):
        return 0, list(self.hashes)

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def delete(self, *keys):
        removed = 0
        for key in keys:
            variants = {key}
            if isinstance(key, bytes):
                variants.add(key.decode("utf-8"))
            elif isinstance(key, str):
                variants.add(key.encode("utf-8"))
            for variant in variants:
                if variant in self.hashes:
                    del self.hashes[variant]
                    removed += 1
                if variant in self.values:
                    del self.values[variant]
                    removed += 1
        return removed

    def eval(self, script, numkeys, meta_key, value_key, *args):
        meta = self.hgetall(meta_key)
        raw_data = meta.get(b"data")
        if raw_data is None:
            raw_data = meta.get("data")
        if "generation" in script:
            if not raw_data:
                return 0
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")
            current = json.loads(raw_data).get("generation", "")
            if not args or current != args[0]:
                return 0
        else:
            mode, expected = args
            if mode == "missing":
                if raw_data is not None:
                    return 0
            elif raw_data != expected and (
                not isinstance(raw_data, bytes) or raw_data.decode("utf-8") != expected
            ):
                return 0
        return self.delete(meta_key, value_key)

    def pipeline(self):
        self._pipeline_commands = []
        return self

    def execute(self):
        results = []
        for command, keys in self._pipeline_commands:
            if command == "delete":
                results.append(self.delete(*keys))
        return results

    def __getattr__(self, name):
        if name == "delete":
            return self.delete
        raise AttributeError(name)

    def queue_delete(self, *keys):
        self._pipeline_commands.append(("delete", keys))
        return self


class FakeRedisPipelineClient(FakeRedisClient):
    def pipeline(self):
        client = self

        class Pipeline:
            def __init__(self):
                self.commands = []

            def delete(self, *keys):
                self.commands.append(keys)
                return self

            def execute(self):
                return [client.delete(*keys) for keys in self.commands]

        return Pipeline()


class RecordingRedisClient:
    def __init__(self):
        self.commands = []

    def pipeline(self):
        client = self

        class Pipeline:
            def set(self, key, value):
                client.commands.append(("set", key, value))
                return self

            def setex(self, key, ttl, value):
                client.commands.append(("setex", key, ttl, value))
                return self

            def hset(self, key, mapping):
                client.commands.append(("hset", key, mapping))
                return self

            def expire(self, key, ttl):
                client.commands.append(("expire", key, ttl))
                return self

            def execute(self):
                return [True] * len(client.commands)

        return Pipeline()


class TestRedisCacheLayer:
    def _layer(self):
        layer = object.__new__(RedisCacheLayer)
        layer.client = RecordingRedisClient()
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        return layer

    def test_no_ttl_remains_non_expiring_in_redis_metadata_and_storage(self, monkeypatch):
        monkeypatch.setattr(cache_v2_module.time, "time", FrozenClock(1000.0).time)
        layer = self._layer()

        assert layer.put("key", "value", ttl_seconds=None)
        assert [command[0] for command in layer.client.commands] == ["set", "hset"]
        metadata = json.loads(layer.client.commands[1][2]["data"])
        assert metadata["ttl_seconds"] is None

    def test_fractional_ttl_is_rounded_up_for_redis(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        layer = self._layer()

        assert layer.put("key", "value", ttl_seconds=0.25)
        assert [(command[0], command[2]) for command in layer.client.commands if command[0] in {"setex", "expire"}] == [
            ("setex", 1),
            ("expire", 1),
        ]

    def test_direct_put_captures_origin_before_slow_serialization(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def slow_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 5.0
            return payload

        monkeypatch.setattr(cache_v2_module.pickle, "dumps", slow_dumps)
        layer = self._layer()
        assert layer.put("key", "value", ttl_seconds=10.0)
        metadata = json.loads(layer.client.commands[1][2]["data"])
        assert metadata["created_at"] == 1000.0
        assert [command[2] for command in layer.client.commands if command[0] == "setex"] == [5]

    def test_put_rejects_when_serialization_consumes_ttl(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        original_dumps = cache_v2_module.pickle.dumps

        def exhausted_dumps(value, protocol=None):
            payload = original_dumps(value, protocol=protocol)
            clock.now += 10.0
            return payload

        monkeypatch.setattr(cache_v2_module.pickle, "dumps", exhausted_dumps)
        layer = self._layer()
        assert layer.put("key", "value", ttl_seconds=10.0) is False
        assert layer.client.commands == []

    def test_promotion_uses_remaining_absolute_ttl(self, monkeypatch):
        clock = FrozenClock(1009.2)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        layer = self._layer()

        assert layer.put("key", "value", ttl_seconds=10.0, created_at=1000.0)
        assert layer.client.commands[0][2] == 1
        metadata = json.loads(layer.client.commands[1][2]["data"])
        assert metadata["created_at"] == 1000.0
        assert metadata["ttl_seconds"] == 10.0

    @pytest.mark.parametrize("ttl", [0.0, -1.0, float("nan"), float("inf"), True, False])
    def test_nonpositive_or_nonfinite_ttl_is_rejected(self, monkeypatch, ttl):
        monkeypatch.setattr(cache_v2_module.time, "time", FrozenClock(1000.0).time)
        layer = self._layer()

        assert layer.put("key", "value", ttl_seconds=ttl) is False
        assert layer.client.commands == []

    def test_expired_promotion_is_rejected(self, monkeypatch):
        monkeypatch.setattr(cache_v2_module.time, "time", FrozenClock(1010.0).time)
        layer = self._layer()

        assert layer.put("key", "value", ttl_seconds=10.0, created_at=1000.0) is False
        assert layer.client.commands == []


class TestMultiLevelCache:
    """Test multi-level cache coordinator."""

    def test_production_config_factory_l1_only_works(self):
        cache = create_cache(CacheV2Config(runtime_mode="production"))
        assert cache.runtime_mode == "production"
        assert cache.l1 is not None
        assert cache.l2 is None
        assert cache.l3 is None
        assert cache.put("key", "value")
        assert cache.get("key") == "value"

    @pytest.mark.parametrize("layer", ["l2", "l3"])
    def test_production_config_factory_rejects_durable_layers(self, tmp_path, layer):
        kwargs = {"enable_l2": layer == "l2", "enable_l3": layer == "l3"}
        if layer == "l2":
            kwargs["disk_root"] = tmp_path
        else:
            kwargs["redis_url"] = "redis://localhost:6379/0"
        with pytest.raises(CacheConfigurationError):
            create_cache(CacheV2Config(runtime_mode="production", **kwargs))

    def test_research_explicit_l2_opt_in_is_advisory(self, tmp_path):
        cache = create_cache(CacheV2Config(
            runtime_mode="research", disk_root=tmp_path, enable_l2=True
        ))
        assert cache.runtime_mode == "research"
        assert cache.l2 is not None

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"enable_l2": True}, "disk_root"),
            ({"enable_l3": True}, "redis_url"),
        ],
    )
    def test_requested_durable_layer_unavailable_never_silently_falls_back(
        self, kwargs, message
    ):
        with pytest.raises(CacheConfigurationError, match=message):
            create_cache(CacheV2Config(runtime_mode="research", **kwargs))

    def test_requested_redis_dependency_unavailable_never_silently_falls_back(
        self, monkeypatch
    ):
        monkeypatch.setattr(cache_v2_module, "HAS_REDIS", False)
        with pytest.raises(CacheConfigurationError, match="redis-py"):
            create_cache(CacheV2Config(
                runtime_mode="test",
                redis_url="redis://localhost:6379/0",
                enable_l3=True,
            ))

    def test_production_rejects_durable_layers(self, tmp_path):
        with pytest.raises(CacheConfigurationError, match="L2 disk"):
            MultiLevelCache(
                disk_root=tmp_path,
                enable_l2=True,
                production_mode=True,
            )

        with pytest.raises(CacheConfigurationError, match="L3 Redis"):
            MultiLevelCache(
                redis_url="redis://localhost:6379/0",
                enable_l3=True,
                production_mode=True,
            )

    def test_production_defaults_to_l1_only(self, tmp_path):
        cache = MultiLevelCache(disk_root=tmp_path, production_mode=True)
        assert cache.l1 is not None
        assert cache.l2 is None
        assert cache.l3 is None
        assert cache.put("key", "value")
        assert cache.get("key") == "value"

    def test_research_mode_requires_explicit_durable_opt_in(self, tmp_path):
        default_cache = MultiLevelCache(disk_root=tmp_path)
        assert default_cache.l2 is None

        research_cache = MultiLevelCache(
            disk_root=tmp_path,
            enable_l2=True,
            production_mode=False,
        )
        assert research_cache.l2 is not None

    def test_l1_hit(self):
        """Test L1 cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = np.random.randn(100)
            cache.put("key1", data)

            # Should hit L1
            result = cache.get("key1")
            np.testing.assert_array_equal(result, data)

            stats = cache.get_stats()
            assert stats["l1_hits"] == 1
            assert stats["l2_hits"] == 0
            assert stats["misses"] == 0

    def test_l2_promotion(self):
        """Test L2 hit promotes to L1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=10.0,  # Large enough for testing
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = np.random.randn(100)
            cache.put("key1", data, write_through=True)

            # Verify it's in L2
            assert cache.l2.get("key1") is not None

            # Clear L1 to force L2 hit
            cache.l1.clear()

            # Reset stats after clear
            cache.stats.l1_hits = 0
            cache.stats.l2_hits = 0

            # First get should hit L2 and promote to L1
            result = cache.get("key1")
            assert result is not None
            np.testing.assert_array_equal(result, data)

            stats = cache.get_stats()
            assert stats["l2_hits"] == 1  # L2 hit

            # Verify promoted to L1
            assert cache.l1.get("key1") is not None

            # Next get should hit L1
            result2 = cache.get("key1")
            stats = cache.get_stats()
            assert stats["l1_hits"] == 1  # L1 hit after promotion

    def test_l2_promotion_preserves_absolute_expiry(self, monkeypatch):
        """L2 promotion must not restart the entry TTL in L1."""
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )
            cache.put("key1", "value", ttl_seconds=10.0)
            cache.l1.clear()

            clock.now = 1006.0
            assert cache.get("key1") == "value"
            promoted = cache.l1.get("key1")
            assert promoted is not None
            assert promoted[1].created_at == 1000.0

            clock.now = 1010.1
            assert cache.get("key1") is None
            assert cache.l1.get("key1") is None
            assert cache.l2.get("key1") is None

    def test_l2_promotion_rechecks_expiry_after_slow_promotion(self, monkeypatch):
        clock = FrozenClock(1000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )
            cache.put("key1", "value", ttl_seconds=10.0)
            cache.l1.clear()
            clock.now = 1009.0
            original_put = cache.l1.put

            def slow_promotion(*args, **kwargs):
                clock.now = 1010.0
                return original_put(*args, **kwargs)

            monkeypatch.setattr(cache.l1, "put", slow_promotion)
            assert cache.get("key1") is None
            assert cache.l1.get("key1") is None
            assert cache.l2.get("key1") is None
            assert cache.get_stats()["l2_hits"] == 0
            assert cache.get_stats()["misses"] == 1

    def test_l3_promotion_preserves_absolute_expiry(self, monkeypatch):
        """L3 promotion into L2/L1 must preserve the original expiry."""
        clock = FrozenClock(2000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )
            cache.l3 = FakeRedisLayer()
            cache.l3.put("key1", "value", ttl_seconds=10.0)

            clock.now = 2006.0
            assert cache.get("key1") == "value"
            assert cache.l1.get("key1")[1].created_at == 2000.0
            assert cache.l2.get("key1")[1].created_at == 2000.0

            clock.now = 2010.1
            assert cache.get("key1") is None
            assert cache.l1.get("key1") is None
            assert cache.l2.get("key1") is None
            assert cache.l3.get("key1") is None

    def test_l3_promotion_rechecks_expiry_after_slow_promotion(self, monkeypatch):
        clock = FrozenClock(2000.0)
        monkeypatch.setattr(cache_v2_module.time, "time", clock.time)
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            compression="zlib",
            enable_l1=True,
            enable_l2=False,
        )
        cache.l3 = FakeRedisLayer()
        cache.l3.put("key1", "value", ttl_seconds=10.0)
        clock.now = 2009.0
        original_put = cache.l1.put

        def slow_promotion(*args, **kwargs):
            clock.now = 2010.0
            return original_put(*args, **kwargs)

        monkeypatch.setattr(cache.l1, "put", slow_promotion)
        assert cache.get("key1") is None
        assert cache.l1.get("key1") is None
        assert cache.l3.get("key1") is None
        assert cache.get_stats()["l3_hits"] == 0
        assert cache.get_stats()["misses"] == 1

    def test_dependency_invalidation_transitively_removes_derived_entries(self):
        """Invalidating a raw key also removes dependent derived keys."""
        cache = MultiLevelCache(
            compression="none",
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )
        assert cache.put("d1", "derived-one", dependencies={"raw"})
        assert cache.put("d2", "derived-two", dependencies={"d1"})

        assert cache.invalidate_dependency("raw") == 2
        assert cache.get("d1") is None
        assert cache.get("d2") is None

    def test_dependency_invalidation_counts_logical_keys_once(self):
        """The coordinator counts a key once across L1 and L2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="none",
                enable_l1=True,
                enable_l2=True,
            )
            assert cache.put("key1", "stale", dependencies={"source"})
            assert cache.invalidate_dependency("source") == 1
            assert cache.get_stats()["invalidations"] == 1

    def test_shared_l3_invalidation_expires_other_coordinator_l1(self):
        """A shared Redis epoch prevents stale sibling L1 hits without L2."""
        class SharedRedis:
            def __init__(self):
                self.entries = {}
                self.epoch = ""

        class Layer:
            def __init__(self, shared):
                self.shared = shared

            def invalidation_epoch(self):
                return self.shared.epoch

            def bump_invalidation_epoch(self):
                self.shared.epoch = f"epoch-{len(self.shared.epoch) + 1}"
                return self.shared.epoch

            def put(self, key, value, ttl_seconds=None, dependencies=None, created_at=None):
                self.shared.entries[key] = (value, set(dependencies or ()))
                return True

            def get(self, key):
                entry = self.shared.entries.get(key)
                if entry is None:
                    return None
                value, dependencies = entry
                return value, CacheMetadata(
                    key=key,
                    created_at=0.0,
                    last_accessed=0.0,
                    access_count=0,
                    size_bytes=0,
                    compressed_size=0,
                    ttl_seconds=None,
                    dependencies=dependencies,
                )

            def invalidate_dependencies(self, dependency, *, return_keys=False):
                keys = {
                    key for key, (_, dependencies) in self.shared.entries.items()
                    if dependency in dependencies
                }
                for key in keys:
                    self.shared.entries.pop(key, None)
                return keys if return_keys else len(keys)

        shared = SharedRedis()
        first = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        second = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        first.l3 = Layer(shared)
        second.l3 = Layer(shared)
        first._l1_invalidation_epoch = first.l3.invalidation_epoch()
        second._l1_invalidation_epoch = second.l3.invalidation_epoch()

        assert first.put("key1", "stale", dependencies={"source"})
        assert second.get("key1") == "stale"
        assert second.l1.get("key1")[0] == "stale"

        assert first.invalidate_dependency("source") == 1
        assert second.get("key1") is None
        assert second.l1.get("key1") is None

    def test_l3_epoch_read_failure_bypasses_l1_with_empty_marker(self):
        """A failed shared epoch read must not serve an L1 value."""
        class FailingEpochLayer:
            def invalidation_epoch(self):
                raise RuntimeError("redis unavailable")

            def get(self, key):
                return None

        cache = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        cache.l3 = FailingEpochLayer()
        assert cache.put("key", "stale", write_through=False)

        assert cache.get("key") is None
        assert cache.l1.get("key") is None
        assert cache._l1_invalidation_epoch is None

    def test_l3_epoch_read_failure_bypasses_populated_l1(self):
        """A failed shared epoch read must invalidate a populated marker."""
        class FailingEpochLayer:
            def invalidation_epoch(self):
                raise RuntimeError("redis unavailable")

            def get(self, key):
                return None

        cache = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        cache.l3 = FailingEpochLayer()
        cache._l1_invalidation_epoch = "known-epoch"
        assert cache.put("key", "stale", write_through=False)

        assert cache.get("key") is None
        assert cache.l1.get("key") is None
        assert cache._l1_invalidation_epoch is None

    def test_l3_epoch_write_failure_does_not_publish_local_marker(self):
        """A failed shared epoch write must leave the local marker unknown."""
        class FailingEpochLayer:
            def invalidation_epoch(self):
                return "known-epoch"

            def bump_invalidation_epoch(self):
                raise RuntimeError("redis unavailable")

            def invalidate_dependencies(self, dependency, *, return_keys=False):
                return set() if return_keys else 0

        cache = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        cache.l3 = FailingEpochLayer()
        cache._l1_invalidation_epoch = "known-epoch"
        assert cache.put("key", "stale", write_through=False)

        assert cache.invalidate_dependency("source") == 0
        assert cache._l1_invalidation_epoch is None

    def test_shared_l2_direct_invalidation_expires_sibling_l1(self, tmp_path):
        """A shared L2 epoch prevents stale sibling L1 direct-key hits."""
        first = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        second = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        assert first.put("key1", "stale")
        assert second.get("key1") == "stale"
        assert second.l1.get("key1")[0] == "stale"

        assert first.invalidate("key1")

        assert second.get("key1") is None
        assert second.l1.get("key1") is None

    def test_shared_l2_invalidation_expires_other_coordinator_l1(self, tmp_path):
        """A durable invalidation epoch prevents stale sibling L1 hits."""
        first = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        second = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        assert first.put("key1", "stale", dependencies={"source"})
        assert second.get("key1") == "stale"
        assert second.l1.get("key1")[0] == "stale"

        assert first.invalidate_dependency("source") == 1

        assert second.get("key1") is None
        assert second.l1.get("key1") is None

    def test_sibling_invalidation_fences_blocked_l2_read(self, tmp_path):
        """A sibling invalidation cannot be bypassed by an already blocked L2 read."""
        class SharedLayer:
            def __init__(self):
                self.entries = {}
                self.epoch = "epoch-0"

            def invalidation_epoch(self):
                return self.epoch

            def bump_invalidation_epoch(self):
                self.epoch = f"epoch-{int(self.epoch.rsplit('-', 1)[1]) + 1}"
                return self.epoch

            def put(self, key, value, ttl_seconds=None, dependencies=None, created_at=None):
                self.entries[key] = (value, set(dependencies or ()))
                return True

            def get(self, key):
                entry = self.entries.get(key)
                if entry is None:
                    return None
                value, dependencies = entry
                return value, CacheMetadata(
                    key=key, created_at=0.0, last_accessed=0.0,
                    access_count=0, size_bytes=0, compressed_size=0,
                    ttl_seconds=None, dependencies=dependencies,
                )

            def invalidate_dependencies(self, dependency, *, return_keys=False):
                keys = {
                    key for key, (_, dependencies) in self.entries.items()
                    if dependency in dependencies
                }
                for key in keys:
                    self.entries.pop(key, None)
                return keys if return_keys else len(keys)

        shared = SharedLayer()
        first = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        second = MultiLevelCache(
            compression="none", enable_l1=True, enable_l2=False, enable_l3=False
        )
        first.l3 = shared
        second.l3 = shared
        first._l1_invalidation_epoch = shared.invalidation_epoch()
        second._l1_invalidation_epoch = shared.invalidation_epoch()
        assert shared.put("key1", "stale", dependencies={"source"})

        read_entered = threading.Event()
        release_read = threading.Event()
        original_get = shared.get

        def blocking_get(key):
            result = original_get(key)
            read_entered.set()
            if not release_read.wait(timeout=5.0):
                raise TimeoutError("blocked lower-layer read was not released")
            return result

        shared.get = blocking_get
        read_result = []
        invalidation_result = []
        thread_errors = []

        def capture(operation, results):
            try:
                results.append(operation())
            except BaseException as exc:
                thread_errors.append(exc)

        reader = threading.Thread(target=capture, args=(lambda: first.get("key1"), read_result))
        reader.start()
        assert read_entered.wait(timeout=5.0)

        invalidator = threading.Thread(
            target=capture,
            args=(lambda: second.invalidate_dependency("source"), invalidation_result),
        )
        invalidator.start()
        invalidator.join(timeout=5.0)
        assert not invalidator.is_alive()
        assert invalidation_result == [1]

        release_read.set()
        reader.join(timeout=5.0)
        assert not reader.is_alive()
        assert thread_errors == []
        assert read_result == [None]
        assert first.l1.get("key1") is None

    @pytest.mark.skipif(cache_v2_module.fcntl is None, reason="requires POSIX flock")
    def test_shared_l2_invalidation_expires_parent_l1_across_processes(self, tmp_path):
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        assert cache.put("key1", "stale", dependencies={"source"})
        assert cache.get("key1") == "stale"
        assert cache.l1.get("key1")[0] == "stale"

        code = r'''
import sys
from pathlib import Path
from quant_evaluator.runtime.cache_v2 import MultiLevelCache
cache = MultiLevelCache(
    memory_size_mb=1.0,
    disk_root=Path(sys.argv[1]),
    compression="none",
    enable_l1=True,
    enable_l2=True,
)
removed = cache.invalidate_dependency("source")
raise SystemExit(0 if removed == 1 else 2)
'''
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[2],
            env={**cache_v2_module.os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            text=True,
            capture_output=True,
            timeout=10.0,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert cache.get("key1") is None
        assert cache.l1.get("key1") is None

    def test_forged_l2_metadata_key_cannot_redirect_dependency_walk(self, tmp_path):
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        assert cache.l1.put("victim", "must-survive", dependencies={"forged"})
        assert cache.l2.put("actual", "corrupt", dependencies={"source"})
        value_path = cache.l2._key_path("actual")
        metadata, compressed = cache.l2._decode_record(value_path.read_bytes())
        metadata.key = "forged"
        value_path.write_bytes(cache.l2._encode_record(metadata, compressed))

        assert cache.invalidate_dependency("source") == 0
        assert not value_path.exists()
        assert cache.l1.get("victim")[0] == "must-survive"
        assert cache.get_stats()["invalidations"] == 0

    def test_legacy_metadata_key_cannot_redirect_dependency_walk(self, tmp_path):
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="none",
            enable_l1=True,
            enable_l2=True,
        )
        assert cache.l1.put("victim", "must-survive", dependencies={"forged"})
        assert cache.l2.put("actual", "value")
        value_path = cache.l2._key_path("actual")
        metadata, compressed = cache.l2._decode_record(value_path.read_bytes())
        value_path.write_bytes(cache.l2._encode_record(metadata, compressed))
        metadata.key = "victim"
        metadata.dependencies = {"source"}
        cache.l2._meta_path(value_path).write_text(
            json.dumps({**metadata.to_dict(), "checksum": "unused"}),
            encoding="utf-8",
        )

        assert cache.invalidate_dependency("source") == 0
        assert not value_path.exists()
        assert not cache.l2._meta_path(value_path).exists()
        assert cache.l1.get("victim")[0] == "must-survive"
        assert cache.get_stats()["invalidations"] == 0

        cache = MultiLevelCache(
            memory_size_mb=1.0,
            compression="none",
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )
        assert cache.l1.put("victim", "must-survive", dependencies={"forged"})
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:actual:meta"
        metadata = CacheMetadata(
            key="forged",
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=None,
            dependencies={"source"},
        )
        layer.client = FakeRedisPipelineClient(
            hashes={physical_meta: {b"data": json.dumps(metadata.to_dict()).encode()}},
            values={b"cache:actual": b"corrupt"},
        )
        cache.l3 = layer

        assert cache.invalidate_dependency("source") == 0
        assert physical_meta not in layer.client.hashes
        assert b"cache:actual" not in layer.client.values
        assert cache.l1.get("victim")[0] == "must-survive"
        assert cache.get_stats()["invalidations"] == 0

    def test_dependency_invalidation_removes_l2_only_entry(self):
        """An L2-only entry is removed and cannot be promoted again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )
            cache.put("key1", "stale", dependencies={"source"})
            cache.l1.clear()

            assert cache.invalidate_dependency("source") == 1
            assert cache.l2.get("key1") is None
            assert cache.get("key1") is None
            assert cache.l1.get("key1") is None

    @pytest.mark.parametrize("source_layer", ["l2", "l3"])
    def test_dependency_invalidation_does_not_race_with_promotion(
        self, tmp_path, source_layer
    ):
        """A captured lower-layer read must not survive later invalidation."""
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            disk_root=tmp_path,
            compression="zlib",
            enable_l1=True,
            enable_l2=source_layer == "l2",
        )
        if source_layer == "l3":
            cache.l3 = FakeRedisLayer()
        source = cache.l2 if source_layer == "l2" else cache.l3
        source.put("key1", "stale", dependencies={"source"})

        invalidator_attempted = threading.Event()
        read_captured = threading.Event()
        release_read = threading.Event()
        original_lock = cache._coordination_lock
        original_get = source.get

        class ObservedCoordinationLock:
            def __enter__(self):
                if threading.current_thread().name == "cache-invalidator":
                    invalidator_attempted.set()
                original_lock.acquire()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                original_lock.release()

        def blocking_get(key):
            result = original_get(key)
            read_captured.set()
            if not release_read.wait(timeout=5.0):
                raise TimeoutError("reader was not released")
            return result

        cache._coordination_lock = ObservedCoordinationLock()
        source.get = blocking_get
        read_result = []
        invalidation_result = []
        thread_errors = []

        def capture_result(operation, results):
            try:
                results.append(operation())
            except BaseException as exc:
                thread_errors.append(exc)

        reader = threading.Thread(
            name="cache-reader",
            target=capture_result,
            args=(lambda: cache.get("key1"), read_result),
        )
        invalidator = threading.Thread(
            name="cache-invalidator",
            target=capture_result,
            args=(
                lambda: cache.invalidate_dependency("source"),
                invalidation_result,
            ),
        )

        reader.start()
        assert read_captured.wait(timeout=5.0)
        invalidator.start()
        assert invalidator_attempted.wait(timeout=5.0)
        release_read.set()
        reader.join(timeout=5.0)
        invalidator.join(timeout=5.0)

        assert not reader.is_alive()
        assert not invalidator.is_alive()
        assert thread_errors == []
        assert read_result == ["stale"]
        assert invalidation_result and invalidation_result[0] >= 1
        assert cache.l1.get("key1") is None
        assert original_get("key1") is None
        assert cache.get("key1") is None

    def test_dependency_invalidation_removes_l3_only_entry(self):
        """An invalidated L3-only entry must never be promoted back."""
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )
        cache.l3 = FakeRedisLayer()
        cache.l3.put("key1", "stale", dependencies={"source"})

        assert cache.invalidate_dependency("source") == 1
        assert cache.l3.get("key1") is None
        assert cache.get("key1") is None
        assert cache.l1.get("key1") is None

    def test_dependency_invalidation_quarantines_empty_l3_metadata(self):
        """Empty Redis metadata must quarantine its physical value pair."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        empty_meta = b"cache:empty:meta"
        matching_meta = b"cache:matching:meta"
        matching = CacheMetadata(
            key="matching",
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=None,
            dependencies={"source"},
        )
        layer.client = FakeRedisPipelineClient(
            hashes={
                empty_meta: {},
                matching_meta: {b"data": json.dumps(matching.to_dict()).encode()},
            },
            values={
                b"cache:empty": b"orphan",
                b"cache:matching": b"stale",
            },
        )

        assert layer.invalidate_dependencies("source") == 1
        assert empty_meta not in layer.client.hashes
        assert b"cache:empty" not in layer.client.values
        assert matching_meta not in layer.client.hashes
        assert b"cache:matching" not in layer.client.values

    def test_dependency_invalidation_quarantines_empty_data_field(self):
        """An empty Redis metadata payload must quarantine its physical pair."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:empty-data:meta"
        layer.client = FakeRedisPipelineClient(
            hashes={physical_meta: {b"data": b""}},
            values={b"cache:empty-data": b"orphan"},
        )

        assert layer.invalidate_dependencies("source") == 0
        assert physical_meta not in layer.client.hashes
        assert b"cache:empty-data" not in layer.client.values

        """A replacement after scan must not be deleted by stale invalidation."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:actual:meta"
        old = CacheMetadata(
            key="actual", created_at=1000.0, last_accessed=1000.0,
            access_count=0, size_bytes=1, compressed_size=1, ttl_seconds=None,
            dependencies={"source"}, generation="old",
        )
        new = CacheMetadata(
            key="actual", created_at=1001.0, last_accessed=1001.0,
            access_count=0, size_bytes=1, compressed_size=1, ttl_seconds=None,
            dependencies={"other"}, generation="new",
        )
        client = FakeRedisPipelineClient(
            hashes={physical_meta: {b"data": json.dumps(old.to_dict()).encode()}},
            values={b"cache:actual": b"old"},
        )
        layer.client = client
        original_hgetall = client.hgetall

        def hgetall_then_replace(key):
            result = original_hgetall(key)
            client.hashes[physical_meta] = {b"data": json.dumps(new.to_dict()).encode()}
            client.values[b"cache:actual"] = b"new"
            return result

        client.hgetall = hgetall_then_replace
        assert layer.invalidate_dependencies("source") == 0
        assert physical_meta in client.hashes
        assert client.values[b"cache:actual"] == b"new"

    def test_dependency_invalidation_uses_physical_key_for_forged_redis_metadata(self):
        """A forged logical key must not redirect physical deletion."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:actual:meta"
        metadata = CacheMetadata(
            key="forged",
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=None,
            dependencies={"source"},
        )
        layer.client = FakeRedisPipelineClient(
            hashes={physical_meta: {b"data": json.dumps(metadata.to_dict()).encode()}},
            values={
                b"cache:actual": b"value",
                b"cache:forged": b"must-survive",
            },
        )

        assert layer.invalidate_dependencies("source") == 1
        assert physical_meta not in layer.client.hashes
        assert b"cache:actual" not in layer.client.values
        assert b"cache:forged" in layer.client.values

    def test_corrupt_redis_metadata_is_quarantined(self):
        """Corrupt Redis metadata is quarantined without blocking later matches."""
        compressor = create_compressor("zlib")
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = compressor
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600

        matching = CacheMetadata(
            key="matching",
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=None,
            dependencies={"source"},
        )
        unrelated = CacheMetadata(
            key="unrelated",
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=0,
            size_bytes=1,
            compressed_size=1,
            ttl_seconds=None,
            dependencies={"other"},
        )
        corrupt_meta = b"cache:corrupt:meta"
        matching_meta = b"cache:matching:meta"
        unrelated_meta = b"cache:unrelated:meta"
        layer.client = FakeRedisPipelineClient(
            hashes={
                corrupt_meta: {b"data": b"{not-json"},
                matching_meta: {b"data": json.dumps(matching.to_dict()).encode()},
                unrelated_meta: {b"data": json.dumps(unrelated.to_dict()).encode()},
            },
            values={
                b"cache:corrupt": b"stale-corrupt",
                "cache:matching": b"stale-matching",
                b"cache:unrelated": b"valid-unrelated",
            },
        )

        assert layer.invalidate_dependencies("source") == 1
        assert corrupt_meta not in layer.client.hashes
        assert b"cache:corrupt" not in layer.client.values
        assert matching_meta not in layer.client.hashes
        assert "cache:matching" not in layer.client.values
        assert unrelated_meta in layer.client.hashes
        assert b"cache:unrelated" in layer.client.values

    def test_corrupt_redis_metadata_preserves_replacement_race(self):
        """A replacement after corrupt metadata inspection must survive quarantine."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:corrupt:meta"
        replacement = CacheMetadata(
            key="corrupt", created_at=1001.0, last_accessed=1001.0,
            access_count=0, size_bytes=1, compressed_size=1, ttl_seconds=None,
            dependencies={"other"}, generation="replacement",
        )
        client = FakeRedisPipelineClient(
            hashes={physical_meta: {b"data": b"{not-json"}},
            values={b"cache:corrupt": b"replacement"},
        )
        layer.client = client
        original_hgetall = client.hgetall

        def hgetall_then_replace(key):
            result = original_hgetall(key)
            client.hashes[physical_meta] = {
                b"data": json.dumps(replacement.to_dict()).encode()
            }
            return result

        client.hgetall = hgetall_then_replace
        assert layer.invalidate_dependencies("source") == 0
        assert physical_meta in client.hashes
        assert client.values[b"cache:corrupt"] == b"replacement"

    def test_empty_redis_metadata_preserves_replacement_race(self):
        """A replacement after empty metadata inspection must survive quarantine."""
        layer = object.__new__(RedisCacheLayer)
        layer.compressor = create_compressor("none")
        layer.key_prefix = "cache:"
        layer.default_ttl = 3600
        physical_meta = b"cache:empty-race:meta"
        replacement = CacheMetadata(
            key="empty-race", created_at=1001.0, last_accessed=1001.0,
            access_count=0, size_bytes=1, compressed_size=1, ttl_seconds=None,
            dependencies={"other"}, generation="replacement",
        )
        client = FakeRedisPipelineClient(
            hashes={physical_meta: {}},
            values={b"cache:empty-race": b"replacement"},
        )
        layer.client = client
        original_hgetall = client.hgetall

        def hgetall_then_replace(key):
            result = original_hgetall(key)
            client.hashes[physical_meta] = {
                b"data": json.dumps(replacement.to_dict()).encode()
            }
            return result

        client.hgetall = hgetall_then_replace
        assert layer.invalidate_dependencies("source") == 0
        assert physical_meta in client.hashes
        assert client.values[b"cache:empty-race"] == b"replacement"

        """Test cache miss across all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            result = cache.get("nonexistent")
            assert result is None

            stats = cache.get_stats()
            assert stats["misses"] == 1

    def test_invalidation_cascade(self):
        """Test invalidation across all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = {"value": 42}
            cache.put("key1", data)

            # Verify in both layers
            assert cache.l1.get("key1") is not None
            assert cache.l2.get("key1") is not None

            # Invalidate
            cache.invalidate("key1")

            # Should be gone from both layers
            assert cache.l1.get("key1") is None
            assert cache.l2.get("key1") is None

    def test_cache_warming(self):
        """Test cache warming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
            )

            # Loader function
            def loader(key: str):
                return f"value_for_{key}"

            # Warm cache
            keys = ["key1", "key2", "key3"]
            cache.warm(keys, loader)

            # All keys should be cached
            for key in keys:
                assert cache.get(key) == f"value_for_{key}"

            stats = cache.get_stats()
            assert stats["l1_hits"] == 3

    def test_concurrent_warm_loads_each_key_once(self):
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
            runtime_mode="test",
        )
        loader_entered = threading.Event()
        release_loader = threading.Event()
        loader_calls = []
        errors = []

        def loader(key):
            loader_calls.append(key)
            loader_entered.set()
            if not release_loader.wait(timeout=5.0):
                raise TimeoutError("loader was not released")
            return "value"

        def warm():
            try:
                cache.warm(["key"], loader)
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=warm)
        second = threading.Thread(target=warm)
        first.start()
        assert loader_entered.wait(timeout=5.0)
        second.start()
        release_loader.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert loader_calls == ["key"]
        assert cache.get("key") == "value"

    def test_write_through(self):
        """Test write-through to all layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=1.0,
                disk_root=Path(tmpdir),
                compression="zlib",
                enable_l1=True,
                enable_l2=True,
            )

            data = {"value": 42}
            cache.put("key1", data, write_through=True)

            # Should be in both layers
            assert cache.l1.get("key1") is not None
            assert cache.l2.get("key1") is not None

    def test_memory_only_mode(self):
        """Test cache with L1 only."""
        cache = MultiLevelCache(
            memory_size_mb=1.0,
            compression="zlib",
            enable_l1=True,
            enable_l2=False,
            enable_l3=False,
        )

        data = {"value": 42}
        cache.put("key1", data)

        result = cache.get("key1")
        assert result == data

        assert cache.l2 is None
        assert cache.l3 is None


class TestCacheWarmingStrategies:
    """Test cache warming strategies."""

    def test_most_recent_keys_strategy(self):
        """Test most recent keys warming strategy."""
        history = ["key1", "key2", "key3", "key1", "key4", "key2"]
        strategy = MostRecentKeysStrategy(history, top_n=3)

        keys = strategy.get_keys_to_warm()

        # Should return 3 most recent unique keys in reverse order
        assert len(keys) == 3
        assert "key2" in keys  # Most recent
        assert "key4" in keys
        assert "key1" in keys


class TestConcurrency:
    """Test thread safety of cache operations."""

    def test_concurrent_puts(self):
        """Test concurrent put operations."""
        import threading

        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        def worker(thread_id: int):
            for i in range(100):
                cache.put(f"thread{thread_id}_key{i}", f"value_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cache should be in consistent state
        stats = cache.get_stats()
        assert stats["num_entries"] > 0
        assert stats["current_size_bytes"] <= cache.max_size_bytes

    def test_concurrent_get_put(self):
        """Test concurrent get and put operations."""
        import threading

        compressor = create_compressor("zlib")
        cache = MemoryCacheLayer(
            max_size_bytes=1 * 1024 * 1024,
            compressor=compressor,
            enable_compression=True,
        )

        # Pre-populate
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")

        def reader():
            for i in range(100):
                cache.get(f"key_{i % 50}")

        def writer():
            for i in range(100):
                cache.put(f"key_{i % 50}", f"new_value_{i}")

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crashes = success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
