"""并发安全测试：验证 DataAccessSource 在多线程场景下的正确性。

2026-08-13：P0 修复验证 - 并发缓存失效、enable_lazy_scan、refresh_snapshot。
"""
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from factor_engine.storage.sources.data_access_source import DataAccessSource


class TestConcurrentCacheSafety:
    """测试并发缓存访问的安全性和一致性。"""

    def test_concurrent_refresh_snapshot_single_flight(self):
        """验证 refresh_snapshot 的 single-flight 行为：多线程同时刷新时只有一个执行。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.manifest_version.return_value = {
                "has_manifest": False,
                "fresh": False,
            }
            mock_store.describe_dataset.return_value = MagicMock(snapshot_id="snap_v1")
            mock_get_store.return_value = mock_store

            source = DataAccessSource(dataset="test_dataset")
            source._snapshot_ttl_seconds = 0.01  # 10ms TTL，强制刷新

            # 10 个线程同时刷新快照
            call_count = []

            def refresh_and_count():
                result = source.refresh_snapshot()
                with threading.Lock():
                    call_count.append(1)
                return result

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(refresh_and_count) for _ in range(10)]
                results = [f.result() for f in futures]

            # 验证：所有线程看到同一个 snapshot_id
            assert all(r == "snap_v1" for r in results)
            assert len(set(results)) == 1

            # 验证：describe_dataset 只被调用 1 次（single-flight 生效）
            # 注意：由于 double-check，可能是 1-2 次，但不应该是 10 次
            assert mock_store.describe_dataset.call_count <= 2, \
                f"Expected ≤2 calls, got {mock_store.describe_dataset.call_count}"

    def test_concurrent_load_columns_cache_bytes_accurate(self):
        """验证并发 load_columns 时 _cache_bytes 计数准确。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_dataset.return_value = MagicMock(
                time_column="date",
                instrument_column="instrument",
            )
            mock_store.read.return_value = MagicMock(
                snapshot=MagicMock(snapshot_id="snap_v1"),
                to_arrow=lambda: MagicMock(),
            )
            mock_get_store.return_value = mock_store

            with patch("data_access.read.adapters.arrow_table_to_multiindex_columns") as mock_arrow:
                import pandas as pd
                # Mock 返回固定大小的 Series
                def make_series(name):
                    return pd.Series([1.0] * 100, name=name)

                mock_arrow.return_value = {
                    "close": make_series("close"),
                    "open": make_series("open"),
                    "high": make_series("high"),
                    "low": make_series("low"),
                }

                source = DataAccessSource(dataset="test_dataset")

                # 10 个线程并发加载不同列
                columns = ["close", "open", "high", "low"]
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(source.load_column, columns[i % len(columns)])
                        for i in range(40)
                    ]
                    results = [f.result() for f in futures]

                # 验证：_cache_bytes 与实际驻留总和一致
                actual_bytes = sum(
                    source._series_bytes(v)
                    for v in source._column_cache.values()
                )
                assert source._cache_bytes == actual_bytes, \
                    f"cache_bytes={source._cache_bytes} != actual={actual_bytes}"

                # 验证：所有列都成功缓存
                assert len(source._column_cache) == len(columns)

    def test_concurrent_enable_lazy_scan_no_cache_clear(self):
        """验证并发 enable_lazy_scan 不会清空共享 cache。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_dataset.return_value = MagicMock(
                time_column="date",
                instrument_column="instrument",
            )
            mock_store.read.return_value = MagicMock(
                snapshot=MagicMock(snapshot_id="snap_v1"),
                to_arrow=lambda: MagicMock(),
            )
            mock_get_store.return_value = mock_store

            with patch("data_access.read.adapters.arrow_table_to_multiindex_columns") as mock_arrow:
                import pandas as pd
                mock_arrow.return_value = {
                    "close": pd.Series([1.0] * 100, name="close"),
                }

                source = DataAccessSource(dataset="test_dataset")

                # 主线程缓存一些数据
                source.load_column("close")
                assert "close" in source._column_cache
                initial_cache_size = len(source._column_cache)

                # 10 个 worker 线程同时 enable_lazy_scan
                def worker():
                    source.enable_lazy_scan(True)
                    # 返回 cache 是否仍有数据
                    return len(source._column_cache) > 0

                with ThreadPoolExecutor(max_workers=10) as executor:
                    results = list(executor.map(lambda _: worker(), range(10)))

                # 验证：cache 没有被清空
                assert all(results), "Some threads saw empty cache"
                assert "close" in source._column_cache
                assert len(source._column_cache) == initial_cache_size

    def test_concurrent_put_cache_no_corruption(self):
        """验证并发 _put_cache 不会破坏 cache 结构。"""
        source = DataAccessSource(dataset="test_dataset")

        import pandas as pd

        def put_series(name):
            series = pd.Series([1.0] * 100, name=name)
            source._put_cache(source._column_cache, name, series)
            return name

        # 50 个线程并发插入不同列
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(put_series, f"col_{i}")
                for i in range(100)
            ]
            results = [f.result() for f in as_completed(futures)]

        # 验证：所有列都成功插入（可能部分被淘汰，但不应崩溃）
        assert len(results) == 100
        # 验证：cache 结构完整（不会因为并发而损坏）
        assert isinstance(source._column_cache, dict)
        assert source._cache_bytes >= 0


class TestConcurrentResourceManagement:
    """测试并发场景下的资源管理。"""

    def test_lazy_bundle_lifecycle_concurrent(self):
        """验证并发场景下 lazy_bundle 的生命周期管理。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_dataset.return_value = MagicMock(
                time_column="date",
                instrument_column="instrument",
            )
            mock_get_store.return_value = mock_store

            with patch("factor_engine.backend.polars_lazy.build_lazy_column_bundle") as mock_build:
                # Mock bundle with close() method
                closed_bundles = []

                def make_bundle():
                    bundle = MagicMock()
                    bundle.physical_columns = []
                    bundle.output_names = {}
                    bundle.snapshot_id = "snap_v1"

                    def close_fn():
                        closed_bundles.append(bundle)

                    bundle.close = close_fn
                    return bundle

                mock_build.side_effect = lambda *args, **kwargs: make_bundle()

                source = DataAccessSource(dataset="test_dataset")
                source.enable_lazy_scan(True)

                # 10 个线程并发 prefetch（会创建多个 bundle）
                def prefetch_worker(i):
                    source._prefetch_lazy_bundle([f"col_{i}"])

                with ThreadPoolExecutor(max_workers=10) as executor:
                    list(executor.map(prefetch_worker, range(10)))

                # 验证：旧 bundle 被关闭（closed_bundles 非空）
                assert len(closed_bundles) > 0, "Old bundles should be closed"

    def test_context_manager_cleanup(self):
        """验证 context manager 正确清理资源。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store"):
            with DataAccessSource(dataset="test_dataset") as source:
                source._closed = False
                assert not source._closed

            # 退出 context manager 后应该被关闭
            assert source._closed

    def test_close_idempotent(self):
        """验证 close() 可以安全地多次调用。"""
        source = DataAccessSource(dataset="test_dataset")

        # 多次调用 close() 不应崩溃
        source.close()
        source.close()
        source.close()

        assert source._closed


class TestConcurrentEdgeCases:
    """测试并发场景的边界条件。"""

    def test_concurrent_clear_cache_and_read(self):
        """验证 clear_cache 与并发读取不会导致崩溃。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_dataset.return_value = MagicMock(
                time_column="date",
                instrument_column="instrument",
            )
            mock_store.read.return_value = MagicMock(
                snapshot=MagicMock(snapshot_id="snap_v1"),
                to_arrow=lambda: MagicMock(),
            )
            mock_get_store.return_value = mock_store

            with patch("data_access.read.adapters.arrow_table_to_multiindex_columns") as mock_arrow:
                import pandas as pd
                mock_arrow.return_value = {
                    "close": pd.Series([1.0] * 100, name="close"),
                }

                source = DataAccessSource(dataset="test_dataset")

                # 一个线程不断读取，另一个线程不断清缓存
                errors = []

                def reader():
                    for _ in range(20):
                        try:
                            source.load_column("close")
                        except Exception as e:
                            errors.append(e)
                        time.sleep(0.001)

                def clearer():
                    for _ in range(20):
                        try:
                            source.clear_cache(reset_snapshot=False)
                        except Exception as e:
                            errors.append(e)
                        time.sleep(0.001)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    f1 = executor.submit(reader)
                    f2 = executor.submit(clearer)
                    f1.result()
                    f2.result()

                # 验证：没有崩溃或异常
                assert len(errors) == 0, f"Concurrent ops raised errors: {errors}"

    def test_concurrent_snapshot_change_during_read(self):
        """验证快照变化时并发读取的行为。"""
        with patch("factor_engine.storage.sources.data_access_source._get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.manifest_version.return_value = {
                "has_manifest": True,
                "fresh": True,
                "dataset_version": "v1",
                "partition_version": "p1",
            }
            mock_store.get_dataset.return_value = MagicMock(
                time_column="date",
                instrument_column="instrument",
            )
            mock_store.read.return_value = MagicMock(
                snapshot=MagicMock(snapshot_id="snap_v1"),
                to_arrow=lambda: MagicMock(),
            )
            mock_get_store.return_value = mock_store

            with patch("data_access.read.adapters.arrow_table_to_multiindex_columns") as mock_arrow:
                import pandas as pd
                mock_arrow.return_value = {
                    "close": pd.Series([1.0] * 100, name="close"),
                }

                source = DataAccessSource(dataset="test_dataset")
                source._snapshot_ttl_seconds = 0.01  # 10ms TTL

                # 第一次读取
                source.load_column("close")
                assert source._manifest_token == "manifest:v1:p1"

                # 模拟快照变化
                mock_store.manifest_version.return_value = {
                    "has_manifest": True,
                    "fresh": True,
                    "dataset_version": "v2",
                    "partition_version": "p2",
                }

                # 并发读取（会触发 refresh_snapshot）
                time.sleep(0.02)  # 等待 TTL 过期
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(source.load_column, "close") for _ in range(5)]
                    results = [f.result() for f in futures]

                # 验证：快照已更新
                assert source._manifest_token == "manifest:v2:p2"
                # 验证：所有读取都成功
                assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
