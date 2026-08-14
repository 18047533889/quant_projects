#!/usr/bin/env python3
"""
Research Control Write/Query Performance Benchmarks

Tests ledger write and query performance:
- Campaign creation and retrieval
- Trial logging (batch and streaming)
- Query performance (filters, aggregations)
- Concurrent write scenarios

Measures:
- Write throughput (records/sec)
- Query latency
- Index performance
- Concurrent access patterns
"""

import gc
import time
import traceback
import tempfile
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

try:
    from research_control.ledger.campaign import CampaignLedger
    from research_control.ledger.trial import TrialLedger
    from research_control.ledger.query import QueryLedger
    RC_AVAILABLE = True
except ImportError:
    RC_AVAILABLE = False


def benchmark_campaign_writes(n_campaigns: int, db_path: Path) -> Dict[str, Any]:
    """Benchmark campaign creation."""
    try:
        if db_path.exists():
            db_path.unlink()

        ledger = CampaignLedger(str(db_path))

        gc.collect()
        start = time.perf_counter()

        campaign_ids = []
        for i in range(n_campaigns):
            campaign_id = f"campaign_{i:06d}"
            ledger.create_campaign(
                campaign_id=campaign_id,
                name=f"Benchmark Campaign {i}",
                description=f"Test campaign for benchmarking - {i}",
                config={"seed": i, "iterations": 1000, "optimizer": "bayesian"}
            )
            campaign_ids.append(campaign_id)

        elapsed = time.perf_counter() - start
        throughput = n_campaigns / elapsed

        return {
            "n_campaigns": n_campaigns,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_write": round(elapsed / n_campaigns * 1000, 3),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def benchmark_trial_writes(n_trials: int, db_path: Path, batch_size: int = 100) -> Dict[str, Any]:
    """Benchmark trial logging."""
    try:
        if db_path.exists():
            db_path.unlink()

        ledger = TrialLedger(str(db_path))

        # Create a campaign first
        campaign_ledger = CampaignLedger(str(db_path))
        campaign_ledger.create_campaign(
            campaign_id="benchmark_campaign",
            name="Benchmark",
            description="For trial writes"
        )

        gc.collect()
        start = time.perf_counter()

        for i in range(n_trials):
            ledger.log_trial(
                campaign_id="benchmark_campaign",
                trial_id=f"trial_{i:08d}",
                factor_expression=f"(close / mean(close, {i % 100 + 5}))",
                metrics={
                    "rank_ic": np.random.randn() * 0.1 + 0.05,
                    "rank_icir": np.random.randn() * 0.5 + 1.5,
                    "sharpe": np.random.randn() * 0.3 + 0.8,
                },
                metadata={"iteration": i, "seed": i * 42}
            )

        elapsed = time.perf_counter() - start
        throughput = n_trials / elapsed

        return {
            "n_trials": n_trials,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_write": round(elapsed / n_trials * 1000, 3),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def benchmark_query_performance(n_trials: int, db_path: Path) -> Dict[str, Any]:
    """Benchmark query operations."""
    try:
        if db_path.exists():
            db_path.unlink()

        # Populate database
        trial_ledger = TrialLedger(str(db_path))
        campaign_ledger = CampaignLedger(str(db_path))

        campaign_ledger.create_campaign(
            campaign_id="benchmark_campaign",
            name="Benchmark",
            description="For queries"
        )

        print(f"    Populating {n_trials} trials...")
        for i in range(n_trials):
            trial_ledger.log_trial(
                campaign_id="benchmark_campaign",
                trial_id=f"trial_{i:08d}",
                factor_expression=f"(close / mean(close, {i % 100 + 5}))",
                metrics={
                    "rank_ic": np.random.randn() * 0.1 + 0.05,
                    "rank_icir": np.random.randn() * 0.5 + 1.5,
                    "sharpe": np.random.randn() * 0.3 + 0.8,
                },
                metadata={"iteration": i, "category": f"cat_{i % 10}"}
            )

        query_ledger = QueryLedger(str(db_path))

        queries = {
            "full_scan": lambda: query_ledger.get_all_trials("benchmark_campaign"),
            "top_k_ic": lambda: query_ledger.get_top_trials(
                "benchmark_campaign", metric="rank_ic", limit=100
            ),
            "filter_range": lambda: query_ledger.get_trials_by_metric_range(
                "benchmark_campaign", metric="rank_ic", min_value=0.0, max_value=0.2
            ),
        }

        results = {}
        for query_name, query_func in queries.items():
            gc.collect()
            times = []

            # Run 5 times and take median
            for _ in range(5):
                start = time.perf_counter()
                result = query_func()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            median_time = np.median(times)
            results[query_name] = {
                "median_latency_ms": round(median_time * 1000, 3),
                "min_latency_ms": round(min(times) * 1000, 3),
                "max_latency_ms": round(max(times) * 1000, 3),
            }

        return {
            "n_trials": n_trials,
            "queries": results,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def benchmark_concurrent_writes(n_writers: int, writes_per_writer: int, db_path: Path) -> Dict[str, Any]:
    """Benchmark concurrent write scenario (sequential simulation)."""
    try:
        if db_path.exists():
            db_path.unlink()

        ledger = TrialLedger(str(db_path))
        campaign_ledger = CampaignLedger(str(db_path))

        campaign_ledger.create_campaign(
            campaign_id="concurrent_benchmark",
            name="Concurrent",
            description="Concurrent writes"
        )

        total_writes = n_writers * writes_per_writer

        gc.collect()
        start = time.perf_counter()

        for writer_id in range(n_writers):
            for write_id in range(writes_per_writer):
                trial_id = f"writer_{writer_id:03d}_trial_{write_id:06d}"
                ledger.log_trial(
                    campaign_id="concurrent_benchmark",
                    trial_id=trial_id,
                    factor_expression=f"test_expr_{writer_id}_{write_id}",
                    metrics={"rank_ic": np.random.randn() * 0.1},
                    metadata={"writer": writer_id}
                )

        elapsed = time.perf_counter() - start
        throughput = total_writes / elapsed

        return {
            "n_writers": n_writers,
            "writes_per_writer": writes_per_writer,
            "total_writes": total_writes,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def main():
    """Run all research control benchmarks."""
    print("=" * 70)
    print("RESEARCH CONTROL PERFORMANCE BENCHMARKS")
    print("=" * 70)

    if not RC_AVAILABLE:
        print("ERROR: research_control package not available")
        return {"error": "research_control not available"}

    temp_dir = Path(tempfile.mkdtemp(prefix="rc_bench_"))

    try:
        results = {}
        total_start = time.perf_counter()

        # Campaign writes
        print(f"\n{'='*70}")
        print("CAMPAIGN WRITES")
        print(f"{'='*70}")
        for scale_name, n_campaigns in [("small", 100), ("medium", 1000), ("large", 10000)]:
            print(f"\n  Scale: {scale_name} ({n_campaigns} campaigns)")
            db_path = temp_dir / f"campaigns_{scale_name}.db"
            result = benchmark_campaign_writes(n_campaigns, db_path)
            results[f"campaign_writes_{scale_name}"] = result
            if "error" not in result:
                print(f"    ✓ {result['throughput_per_sec']:.1f} campaigns/s")

        # Trial writes
        print(f"\n{'='*70}")
        print("TRIAL WRITES")
        print(f"{'='*70}")
        for scale_name, n_trials in [("small", 1000), ("medium", 10000), ("large", 100000)]:
            print(f"\n  Scale: {scale_name} ({n_trials} trials)")
            db_path = temp_dir / f"trials_{scale_name}.db"
            result = benchmark_trial_writes(n_trials, db_path)
            results[f"trial_writes_{scale_name}"] = result
            if "error" not in result:
                print(f"    ✓ {result['throughput_per_sec']:.1f} trials/s")

        # Query performance
        print(f"\n{'='*70}")
        print("QUERY PERFORMANCE")
        print(f"{'='*70}")
        for scale_name, n_trials in [("small", 1000), ("medium", 10000), ("large", 100000)]:
            print(f"\n  Scale: {scale_name} ({n_trials} trials)")
            db_path = temp_dir / f"query_{scale_name}.db"
            result = benchmark_query_performance(n_trials, db_path)
            results[f"query_{scale_name}"] = result
            if "error" not in result:
                print(f"    ✓ Queries completed")

        # Concurrent writes
        print(f"\n{'='*70}")
        print("CONCURRENT WRITES")
        print(f"{'='*70}")
        for n_writers, writes_each in [(10, 1000), (100, 1000)]:
            print(f"\n  {n_writers} writers × {writes_each} writes")
            db_path = temp_dir / f"concurrent_{n_writers}x{writes_each}.db"
            result = benchmark_concurrent_writes(n_writers, writes_each, db_path)
            results[f"concurrent_{n_writers}x{writes_each}"] = result
            if "error" not in result:
                print(f"    ✓ {result['throughput_per_sec']:.1f} writes/s")

        total_elapsed = time.perf_counter() - total_start

        return {
            "benchmark": "research_control",
            "description": "Ledger write and query performance",
            "results": results,
            "total_time_s": round(total_elapsed, 2)
        }

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import json
    result = main()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))
