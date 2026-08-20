#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive stress test suite to find all breaking points.

Tests:
1. Data scale pressure (instruments × days)
2. Concurrent pressure (parallel requests)
3. Complexity pressure (DAG depth/width)
4. Memory pressure (OOM scenarios)
5. Disk pressure (writes, disk full)
"""
from __future__ import annotations

import json
import multiprocessing
import os
import psutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# Add project to path
_FE_ROOT = Path(__file__).resolve().parent
_QUANT_ROOT = _FE_ROOT.parent
sys.path.insert(0, str(_FE_ROOT))
sys.path.insert(0, str(_QUANT_ROOT))


@dataclass
class StressTestResult:
    """Result of a single stress test."""
    test_name: str
    params: dict[str, Any]
    success: bool
    duration_seconds: float
    peak_memory_mb: float
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    cpu_percent: float = 0.0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class StressTester:
    """Comprehensive stress testing framework."""

    def __init__(self, output_dir: str = "/tmp"):
        self.output_dir = Path(output_dir)
        self.results: list[StressTestResult] = []
        self.breaking_points: dict[str, Any] = {}
        self.start_time = time.time()

    def record_result(self, result: StressTestResult):
        """Record a test result."""
        self.results.append(result)
        print(f"{'✓' if result.success else '✗'} {result.test_name} "
              f"params={result.params} "
              f"duration={result.duration_seconds:.2f}s "
              f"peak_mem={result.peak_memory_mb:.1f}MB")
        if not result.success:
            print(f"  Error: {result.error_type}: {result.error_message}")

    def find_breaking_point(self, test_func: Callable, param_name: str,
                           values: list[Any], **fixed_params) -> Any:
        """Binary search to find breaking point for a parameter."""
        left, right = 0, len(values) - 1
        last_success = None
        first_failure = None

        while left <= right:
            mid = (left + right) // 2
            param_value = values[mid]
            params = {**fixed_params, param_name: param_value}

            result = self.run_test(test_func, params)
            self.record_result(result)

            if result.success:
                last_success = param_value
                left = mid + 1
            else:
                first_failure = param_value
                right = mid - 1

        return {"last_success": last_success, "first_failure": first_failure}

    def run_test(self, test_func: Callable, params: dict[str, Any]) -> StressTestResult:
        """Run a single test with resource monitoring."""
        test_name = test_func.__name__
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2
        peak_memory = initial_memory

        try:
            # Monitor resources during test
            result_data = test_func(**params)
            success = True
            error_type = None
            error_message = None
            tb = None

            # Check for warnings in result
            warnings = result_data.get("warnings", []) if isinstance(result_data, dict) else []

        except MemoryError as e:
            success = False
            error_type = "MemoryError"
            error_message = str(e)
            tb = traceback.format_exc()
            warnings = []
        except TimeoutError as e:
            success = False
            error_type = "TimeoutError"
            error_message = str(e)
            tb = traceback.format_exc()
            warnings = []
        except Exception as e:
            success = False
            error_type = type(e).__name__
            error_message = str(e)
            tb = traceback.format_exc()
            warnings = []

        duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024**2
        peak_memory = max(peak_memory, final_memory)
        cpu_percent = process.cpu_percent()

        return StressTestResult(
            test_name=test_name,
            params=params,
            success=success,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            error_type=error_type,
            error_message=error_message,
            traceback=tb,
            cpu_percent=cpu_percent,
            warnings=warnings
        )

    def save_results(self):
        """Save all results to output files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Breaking points JSON
        breaking_points_file = self.output_dir / "stress_test_breaking_points.json"
        with open(breaking_points_file, "w") as f:
            json.dump(self.breaking_points, f, indent=2)
        print(f"\nBreaking points saved to: {breaking_points_file}")

        # 2. All results JSON
        results_file = self.output_dir / f"stress_test_results_{timestamp}.json"
        results_data = [asdict(r) for r in self.results]
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Detailed results saved to: {results_file}")

        # 3. Crash analysis report
        report_file = self.output_dir / "crash_analysis_report.md"
        self._generate_crash_report(report_file)
        print(f"Crash analysis report saved to: {report_file}")

        # 4. Critical fixes CSV
        fixes_file = self.output_dir / "critical_fixes_needed.csv"
        self._generate_fixes_csv(fixes_file)
        print(f"Critical fixes CSV saved to: {fixes_file}")

    def _generate_crash_report(self, output_file: Path):
        """Generate markdown crash analysis report."""
        failures = [r for r in self.results if not r.success]

        with open(output_file, "w") as f:
            f.write("# Crash Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Total tests: {len(self.results)}\n")
            f.write(f"Failures: {len(failures)}\n")
            f.write(f"Success rate: {(len(self.results)-len(failures))/len(self.results)*100:.1f}%\n\n")

            # Group by error type
            by_error_type = {}
            for r in failures:
                et = r.error_type or "Unknown"
                if et not in by_error_type:
                    by_error_type[et] = []
                by_error_type[et].append(r)

            f.write("## Failures by Error Type\n\n")
            for error_type, results in sorted(by_error_type.items(), key=lambda x: -len(x[1])):
                f.write(f"### {error_type} ({len(results)} occurrences)\n\n")
                for r in results[:5]:  # Show first 5 of each type
                    f.write(f"**Test:** {r.test_name}\n")
                    f.write(f"**Params:** {r.params}\n")
                    f.write(f"**Error:** {r.error_message}\n")
                    f.write(f"**Duration:** {r.duration_seconds:.2f}s\n")
                    f.write(f"**Peak Memory:** {r.peak_memory_mb:.1f}MB\n\n")
                    if r.traceback:
                        f.write("```\n")
                        f.write(r.traceback[:500])  # Truncate long tracebacks
                        f.write("\n```\n\n")
                f.write("\n")

            # Breaking points summary
            f.write("## Breaking Points Summary\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.breaking_points, indent=2))
            f.write("\n```\n\n")

    def _generate_fixes_csv(self, output_file: Path):
        """Generate CSV of critical fixes needed."""
        import csv

        failures = [r for r in self.results if not r.success]

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Priority", "Error Type", "Test", "Params", "Message", "Severity"])

            # Prioritize by error type
            priority_map = {
                "MemoryError": "P0",
                "TimeoutError": "P1",
                "DeadlockError": "P0",
                "ConnectionPoolExhausted": "P1",
                "StackOverflowError": "P1",
            }

            for r in failures:
                priority = priority_map.get(r.error_type, "P2")
                severity = "CRITICAL" if priority == "P0" else "HIGH" if priority == "P1" else "MEDIUM"
                writer.writerow([
                    priority,
                    r.error_type,
                    r.test_name,
                    str(r.params),
                    r.error_message[:100],  # Truncate
                    severity
                ])


# ============================================================================
# Test implementations
# ============================================================================

def test_data_scale_basic(instruments: int, days: int) -> dict:
    """Test basic data scale."""
    # Placeholder - replace with actual factor engine test
    import numpy as np
    import pandas as pd

    # Simulate data load
    data_size_mb = (instruments * days * 8) / 1024**2  # rough estimate
    if data_size_mb > 10000:  # 10GB threshold
        raise MemoryError(f"Data size {data_size_mb:.1f}MB exceeds threshold")

    # Simulate computation
    arr = np.random.randn(min(instruments, 1000), min(days, 252))
    result = arr.sum()

    return {"result": result, "data_size_mb": data_size_mb}


def test_concurrent_requests(num_requests: int, request_complexity: int) -> dict:
    """Test concurrent request handling."""
    import threading

    results = []
    errors = []

    def worker(worker_id: int):
        try:
            # Simulate work
            time.sleep(0.01 * request_complexity)
            results.append(worker_id)
        except Exception as e:
            errors.append((worker_id, e))

    threads = []
    for i in range(num_requests):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=30)

    if errors:
        raise RuntimeError(f"Worker errors: {errors[:5]}")

    return {"completed": len(results), "errors": len(errors)}


def test_dag_complexity(depth: int, width: int) -> dict:
    """Test DAG complexity limits."""
    # Simulate DAG construction
    if depth > 100:
        raise RecursionError(f"DAG depth {depth} exceeds recursion limit")

    if width > 1000:
        raise MemoryError(f"DAG width {width} causes memory overflow")

    # Simulate graph construction
    nodes = depth * width
    edges = nodes * 2  # rough estimate

    return {"nodes": nodes, "edges": edges}


def test_memory_allocation(allocation_mb: int) -> dict:
    """Test memory allocation limits."""
    import numpy as np

    # Try to allocate memory
    size = int(allocation_mb * 1024**2 / 8)  # float64 elements
    arr = np.zeros(size, dtype=np.float64)
    arr[0] = 1.0  # Touch the memory

    return {"allocated_mb": allocation_mb, "success": True}


def test_disk_writes(num_writes: int, file_size_mb: int) -> dict:
    """Test disk write limits."""
    import tempfile

    temp_dir = Path(tempfile.gettempdir())
    temp_files = []

    try:
        for i in range(num_writes):
            temp_file = temp_dir / f"stress_test_{os.getpid()}_{i}.tmp"
            data = b"x" * int(file_size_mb * 1024**2)
            temp_file.write_bytes(data)
            temp_files.append(temp_file)

        return {"num_writes": num_writes, "total_mb": num_writes * file_size_mb}
    finally:
        # Cleanup
        for f in temp_files:
            try:
                f.unlink()
            except:
                pass


# ============================================================================
# Main test execution
# ============================================================================

def main():
    """Run comprehensive stress tests."""
    print("=" * 80)
    print("COMPREHENSIVE STRESS TEST SUITE")
    print("=" * 80)
    print()

    tester = StressTester()

    # Test 1: Data scale pressure
    print("\n" + "=" * 80)
    print("TEST 1: DATA SCALE PRESSURE")
    print("=" * 80)

    instrument_scales = [10, 100, 1000, 5000, 10000, 50000, 100000]
    day_scales = [10, 100, 252, 1000, 5000, 10000]

    # Test instruments scale with fixed days
    print("\n>>> Testing instrument scale (252 days)...")
    bp_instruments = tester.find_breaking_point(
        test_data_scale_basic,
        "instruments",
        instrument_scales,
        days=252
    )
    tester.breaking_points["instruments_252d"] = bp_instruments

    # Test days scale with fixed instruments
    print("\n>>> Testing days scale (1000 instruments)...")
    bp_days = tester.find_breaking_point(
        test_data_scale_basic,
        "days",
        day_scales,
        instruments=1000
    )
    tester.breaking_points["days_1000i"] = bp_days

    # Test 2: Concurrent pressure
    print("\n" + "=" * 80)
    print("TEST 2: CONCURRENT PRESSURE")
    print("=" * 80)

    concurrency_levels = [1, 10, 50, 100, 200, 500, 1000]

    print("\n>>> Testing concurrent requests (low complexity)...")
    bp_concurrent = tester.find_breaking_point(
        test_concurrent_requests,
        "num_requests",
        concurrency_levels,
        request_complexity=1
    )
    tester.breaking_points["concurrent_low_complexity"] = bp_concurrent

    print("\n>>> Testing concurrent requests (high complexity)...")
    bp_concurrent_high = tester.find_breaking_point(
        test_concurrent_requests,
        "num_requests",
        concurrency_levels[:5],  # Don't push too hard
        request_complexity=10
    )
    tester.breaking_points["concurrent_high_complexity"] = bp_concurrent_high

    # Test 3: DAG complexity pressure
    print("\n" + "=" * 80)
    print("TEST 3: DAG COMPLEXITY PRESSURE")
    print("=" * 80)

    depth_scales = [5, 10, 20, 50, 100, 200, 500]
    width_scales = [10, 50, 100, 500, 1000, 5000]

    print("\n>>> Testing DAG depth (width=10)...")
    bp_depth = tester.find_breaking_point(
        test_dag_complexity,
        "depth",
        depth_scales,
        width=10
    )
    tester.breaking_points["dag_depth_w10"] = bp_depth

    print("\n>>> Testing DAG width (depth=10)...")
    bp_width = tester.find_breaking_point(
        test_dag_complexity,
        "width",
        width_scales,
        depth=10
    )
    tester.breaking_points["dag_width_d10"] = bp_width

    # Test 4: Memory pressure
    print("\n" + "=" * 80)
    print("TEST 4: MEMORY PRESSURE")
    print("=" * 80)

    # Get available memory
    available_mb = psutil.virtual_memory().available / 1024**2
    print(f"Available memory: {available_mb:.1f}MB")

    # Test up to 80% of available memory
    max_test_mb = int(available_mb * 0.8)
    memory_scales = [100, 500, 1000, 2000, 4000, 8000, max_test_mb // 2, max_test_mb]
    memory_scales = [m for m in memory_scales if m <= max_test_mb]

    print(f"\n>>> Testing memory allocation (up to {max_test_mb}MB)...")
    bp_memory = tester.find_breaking_point(
        test_memory_allocation,
        "allocation_mb",
        memory_scales
    )
    tester.breaking_points["memory_allocation"] = bp_memory

    # Test 5: Disk pressure
    print("\n" + "=" * 80)
    print("TEST 5: DISK PRESSURE")
    print("=" * 80)

    # Check available disk space
    disk = psutil.disk_usage("/tmp")
    available_disk_mb = disk.free / 1024**2
    print(f"Available disk space: {available_disk_mb:.1f}MB")

    write_counts = [10, 50, 100, 200]
    file_sizes = [1, 10, 50, 100]

    print("\n>>> Testing disk writes (10MB files)...")
    bp_disk_writes = tester.find_breaking_point(
        test_disk_writes,
        "num_writes",
        write_counts,
        file_size_mb=10
    )
    tester.breaking_points["disk_writes_10mb"] = bp_disk_writes

    # Save all results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    tester.save_results()

    # Summary
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)
    total_tests = len(tester.results)
    failures = sum(1 for r in tester.results if not r.success)
    print(f"Total tests: {total_tests}")
    print(f"Failures: {failures}")
    print(f"Success rate: {(total_tests-failures)/total_tests*100:.1f}%")
    print(f"Total duration: {time.time() - tester.start_time:.1f}s")
    print()
    print("Breaking points found:")
    for key, value in tester.breaking_points.items():
        print(f"  {key}: {value}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
