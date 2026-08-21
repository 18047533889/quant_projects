"""
Benchmark suite for cache system performance testing.

Tests cache operations, compression ratios, and memory usage across different scenarios.
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path
import tempfile
import shutil

from quant_evaluator.runtime.cache_v2 import (
    MultiLevelCache,
    create_compressor,
    MemoryCacheLayer,
    DiskCacheLayer,
)


class CacheBenchmark:
    """Benchmark suite for cache performance."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def generate_test_data(self, size_kb: int, data_type: str = "numeric") -> Any:
        """
        Generate test data of specified size and type.

        Args:
            size_kb: Target size in kilobytes
            data_type: "numeric", "sparse", "text", "mixed"

        Returns:
            Test data
        """
        if data_type == "numeric":
            # Dense numeric array
            n_elements = (size_kb * 1024) // 8  # 8 bytes per float64
            return np.random.randn(n_elements)

        elif data_type == "sparse":
            # Sparse array (90% zeros)
            n_elements = (size_kb * 1024) // 8
            data = np.random.randn(n_elements)
            mask = np.random.rand(n_elements) < 0.9
            data[mask] = 0.0
            return data

        elif data_type == "text":
            # Text data (repeated strings)
            text = "The quick brown fox jumps over the lazy dog. " * 100
            n_repeats = (size_kb * 1024) // len(text)
            return text * n_repeats

        elif data_type == "mixed":
            # Mixed DataFrame
            n_rows = (size_kb * 1024) // 100  # ~100 bytes per row
            return pd.DataFrame({
                "id": np.arange(n_rows),
                "value": np.random.randn(n_rows),
                "category": np.random.choice(["A", "B", "C"], n_rows),
                "timestamp": pd.date_range("2020-01-01", periods=n_rows),
                "text": ["sample_text"] * n_rows,
            })

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def benchmark_compression(self):
        """Benchmark compression ratios and speed for different data types."""
        print("Benchmarking compression...")

        data_types = ["numeric", "sparse", "text", "mixed"]
        sizes_kb = [10, 100, 1000]
        methods = ["lz4", "zlib"]

        for data_type in data_types:
            for size_kb in sizes_kb:
                data = self.generate_test_data(size_kb, data_type)

                for method in methods:
                    try:
                        compressor = create_compressor(method)

                        # Measure compression
                        import pickle
                        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                        original_size = len(serialized)

                        start = time.perf_counter()
                        compressed, stats = compressor.compress(serialized)
                        compress_time = time.perf_counter() - start

                        # Measure decompression
                        start = time.perf_counter()
                        decompressed = compressor.decompress(compressed)
                        decompress_time = time.perf_counter() - start

                        result = {
                            "benchmark": "compression",
                            "data_type": data_type,
                            "size_kb": size_kb,
                            "method": method,
                            "original_bytes": original_size,
                            "compressed_bytes": len(compressed),
                            "compression_ratio": stats.ratio,
                            "savings_pct": stats.savings_pct,
                            "compress_time_ms": compress_time * 1000,
                            "decompress_time_ms": decompress_time * 1000,
                            "throughput_mb_s": (original_size / (1024 * 1024)) / compress_time,
                        }
                        self.results.append(result)
                        print(f"  {data_type:8s} {size_kb:5d}KB {method:4s}: "
                              f"{stats.ratio:.2f}x ratio, {stats.savings_pct:.1f}% savings")

                    except Exception as e:
                        print(f"  Failed: {data_type} {size_kb}KB {method}: {e}")

    def benchmark_memory_cache(self):
        """Benchmark L1 memory cache operations."""
        print("\nBenchmarking L1 memory cache...")

        compressor = create_compressor("lz4")
        cache = MemoryCacheLayer(
            max_size_bytes=10 * 1024 * 1024,  # 10MB
            compressor=compressor,
            enable_compression=True,
        )

        # Test 1: Put performance
        n_items = 1000
        item_size_kb = 10
        data = self.generate_test_data(item_size_kb, "numeric")

        start = time.perf_counter()
        for i in range(n_items):
            cache.put(f"key_{i}", data, ttl_seconds=300)
        put_time = time.perf_counter() - start

        # Test 2: Get performance (hits)
        start = time.perf_counter()
        for i in range(n_items):
            cache.get(f"key_{i}")
        get_time = time.perf_counter() - start

        # Test 3: Get performance (misses)
        start = time.perf_counter()
        for i in range(n_items):
            cache.get(f"missing_key_{i}")
        miss_time = time.perf_counter() - start

        stats = cache.get_stats()

        result = {
            "benchmark": "l1_memory_cache",
            "n_items": n_items,
            "item_size_kb": item_size_kb,
            "put_time_ms": put_time * 1000,
            "get_time_ms": get_time * 1000,
            "miss_time_ms": miss_time * 1000,
            "put_ops_per_sec": n_items / put_time,
            "get_ops_per_sec": n_items / get_time,
            "memory_savings_pct": stats["memory_savings_pct"],
            "utilization": stats["utilization"],
        }
        self.results.append(result)

        print(f"  Put: {result['put_ops_per_sec']:.0f} ops/s")
        print(f"  Get: {result['get_ops_per_sec']:.0f} ops/s")
        print(f"  Memory savings: {result['memory_savings_pct']:.1f}%")

    def benchmark_disk_cache(self):
        """Benchmark L2 disk cache operations."""
        print("\nBenchmarking L2 disk cache...")

        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = create_compressor("lz4")
            cache = DiskCacheLayer(
                root_dir=Path(tmpdir),
                compressor=compressor,
            )

            # Test: Put and get performance
            n_items = 100
            item_size_kb = 100
            data = self.generate_test_data(item_size_kb, "mixed")

            start = time.perf_counter()
            for i in range(n_items):
                cache.put(f"key_{i}", data, ttl_seconds=3600)
            put_time = time.perf_counter() - start

            start = time.perf_counter()
            for i in range(n_items):
                cache.get(f"key_{i}")
            get_time = time.perf_counter() - start

            result = {
                "benchmark": "l2_disk_cache",
                "n_items": n_items,
                "item_size_kb": item_size_kb,
                "put_time_ms": put_time * 1000,
                "get_time_ms": get_time * 1000,
                "put_ops_per_sec": n_items / put_time,
                "get_ops_per_sec": n_items / get_time,
            }
            self.results.append(result)

            print(f"  Put: {result['put_ops_per_sec']:.1f} ops/s")
            print(f"  Get: {result['get_ops_per_sec']:.1f} ops/s")

    def benchmark_multilevel_cache(self):
        """Benchmark multi-level cache with promotion."""
        print("\nBenchmarking multi-level cache...")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                memory_size_mb=5.0,
                disk_root=Path(tmpdir),
                compression="lz4",
                enable_l1=True,
                enable_l2=True,
                enable_l3=False,
            )

            n_items = 200
            item_size_kb = 50
            data = self.generate_test_data(item_size_kb, "numeric")

            # Write to cache
            start = time.perf_counter()
            for i in range(n_items):
                cache.put(f"key_{i}", data, ttl_seconds=3600)
            write_time = time.perf_counter() - start

            # Test L1 hits (recent items)
            start = time.perf_counter()
            for i in range(n_items - 50, n_items):
                cache.get(f"key_{i}")
            l1_time = time.perf_counter() - start

            # Test L2 hits (evicted from L1)
            start = time.perf_counter()
            for i in range(50):
                cache.get(f"key_{i}")
            l2_time = time.perf_counter() - start

            # Test misses
            start = time.perf_counter()
            for i in range(50):
                cache.get(f"missing_{i}")
            miss_time = time.perf_counter() - start

            stats = cache.get_stats()

            result = {
                "benchmark": "multilevel_cache",
                "n_items": n_items,
                "item_size_kb": item_size_kb,
                "write_time_ms": write_time * 1000,
                "l1_hit_time_ms": l1_time * 1000,
                "l2_hit_time_ms": l2_time * 1000,
                "miss_time_ms": miss_time * 1000,
                "l1_hits": stats["l1_hits"],
                "l2_hits": stats["l2_hits"],
                "misses": stats["misses"],
                "hit_rate": stats["hit_rate"],
                "memory_savings_pct": stats["l1"]["memory_savings_pct"],
            }
            self.results.append(result)

            print(f"  L1 hits: {stats['l1_hits']}, L2 hits: {stats['l2_hits']}, "
                  f"Misses: {stats['misses']}")
            print(f"  Hit rate: {stats['hit_rate']:.1%}")
            print(f"  Memory savings: {result['memory_savings_pct']:.1f}%")

    def benchmark_eviction(self):
        """Benchmark LRU eviction under memory pressure."""
        print("\nBenchmarking LRU eviction...")

        compressor = create_compressor("lz4")
        cache = MemoryCacheLayer(
            max_size_bytes=1 * 1024 * 1024,  # 1MB limit
            compressor=compressor,
            enable_compression=True,
        )

        # Fill cache beyond capacity
        n_items = 500
        item_size_kb = 10
        data = self.generate_test_data(item_size_kb, "numeric")

        start = time.perf_counter()
        for i in range(n_items):
            cache.put(f"key_{i}", data)
        fill_time = time.perf_counter() - start

        stats = cache.get_stats()

        result = {
            "benchmark": "eviction",
            "n_items_written": n_items,
            "n_items_retained": stats["num_entries"],
            "item_size_kb": item_size_kb,
            "fill_time_ms": fill_time * 1000,
            "utilization": stats["utilization"],
            "eviction_rate": 1.0 - (stats["num_entries"] / n_items),
        }
        self.results.append(result)

        print(f"  Written: {n_items}, Retained: {stats['num_entries']}")
        print(f"  Eviction rate: {result['eviction_rate']:.1%}")

    def run_all(self):
        """Run all benchmarks."""
        print("=" * 60)
        print("Cache System Benchmark Suite")
        print("=" * 60)

        self.benchmark_compression()
        self.benchmark_memory_cache()
        self.benchmark_disk_cache()
        self.benchmark_multilevel_cache()
        self.benchmark_eviction()

        # Save results
        df = pd.DataFrame(self.results)
        output_file = self.output_dir / "cache_benchmark_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {output_file}")
        print(f"{'=' * 60}")

        return df

    def print_summary(self):
        """Print summary of benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Compression summary
        compression_df = df[df["benchmark"] == "compression"]
        if not compression_df.empty:
            print("\nCompression (average across all data types):")
            for method in compression_df["method"].unique():
                method_df = compression_df[compression_df["method"] == method]
                avg_ratio = method_df["compression_ratio"].mean()
                avg_savings = method_df["savings_pct"].mean()
                avg_throughput = method_df["throughput_mb_s"].mean()
                print(f"  {method}: {avg_ratio:.2f}x ratio, {avg_savings:.1f}% savings, "
                      f"{avg_throughput:.1f} MB/s")

        # Memory cache summary
        l1_df = df[df["benchmark"] == "l1_memory_cache"]
        if not l1_df.empty:
            row = l1_df.iloc[0]
            print(f"\nL1 Memory Cache:")
            print(f"  Put: {row['put_ops_per_sec']:.0f} ops/s")
            print(f"  Get: {row['get_ops_per_sec']:.0f} ops/s")
            print(f"  Memory savings: {row['memory_savings_pct']:.1f}%")

        # Multilevel cache summary
        ml_df = df[df["benchmark"] == "multilevel_cache"]
        if not ml_df.empty:
            row = ml_df.iloc[0]
            print(f"\nMulti-level Cache:")
            print(f"  Hit rate: {row['hit_rate']:.1%}")
            print(f"  Memory savings: {row['memory_savings_pct']:.1f}%")
            print(f"  L1 hit time: {row['l1_hit_time_ms']/50:.2f} ms/op")
            print(f"  L2 hit time: {row['l2_hit_time_ms']/50:.2f} ms/op")


def main():
    """Run benchmarks."""
    output_dir = Path(__file__).parent.parent / "benchmarks" / "cache"
    benchmark = CacheBenchmark(output_dir)

    df = benchmark.run_all()
    benchmark.print_summary()

    return df


if __name__ == "__main__":
    main()
