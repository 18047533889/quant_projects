#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Factor Engine Stress Tests - Find ALL Breaking Points.

Exercises actual factor engine components:
- FactorEngine compilation and execution
- Resource management (memory, CPU, disk)
- Concurrency (parallel factor execution)
- DAG complexity (deep/wide factor graphs)
- Backend stress (Pandas/Polars/DuckDB)
"""
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import psutil
import resource
import signal
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, asdict, field
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
    category: str
    params: dict[str, Any]
    success: bool
    duration_seconds: float
    peak_memory_mb: float
    cpu_percent: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    traceback_summary: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class ResourceMonitor:
    """Monitor system resources during test execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.peak_cpu = 0
        self.monitoring = False
        self._thread = None

    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.peak_memory = self.process.memory_info().rss / 1024**2
        self.peak_cpu = 0

        def monitor():
            while self.monitoring:
                try:
                    mem = self.process.memory_info().rss / 1024**2
                    self.peak_memory = max(self.peak_memory, mem)
                    cpu = self.process.cpu_percent(interval=0.1)
                    self.peak_cpu = max(self.peak_cpu, cpu)
                except:
                    pass
                time.sleep(0.1)

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.peak_memory, self.peak_cpu


class FactorEngineStressTester:
    """Comprehensive Factor Engine stress testing."""

    def __init__(self, output_dir: str = "/tmp"):
        self.output_dir = Path(output_dir)
        self.results: list[StressTestResult] = []
        self.breaking_points: dict[str, Any] = {}
        self.start_time = time.time()
        self.critical_failures: list[StressTestResult] = []

    def run_test_with_timeout(self, test_func: Callable, params: dict, timeout: int = 300) -> StressTestResult:
        """Run test with timeout and resource monitoring."""
        test_name = test_func.__name__
        category = params.pop("_category", "general")

        monitor = ResourceMonitor()
        monitor.start()
        start_time = time.time()

        success = False
        error_type = None
        error_message = None
        tb_summary = None
        warnings = []
        metrics = {}

        try:
            # Run test with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(test_func, **params)
                try:
                    result = future.result(timeout=timeout)
                    success = True
                    if isinstance(result, dict):
                        warnings = result.get("warnings", [])
                        metrics = result.get("metrics", {})
                except FuturesTimeoutError:
                    success = False
                    error_type = "TimeoutError"
                    error_message = f"Test exceeded {timeout}s timeout"
                    future.cancel()
                except Exception as e:
                    success = False
                    error_type = type(e).__name__
                    error_message = str(e)
                    tb_summary = "\n".join(traceback.format_exc().split("\n")[:20])
        except Exception as e:
            success = False
            error_type = "TestSetupError"
            error_message = str(e)
            tb_summary = traceback.format_exc()

        duration = time.time() - start_time
        peak_memory, peak_cpu = monitor.stop()

        result = StressTestResult(
            test_name=test_name,
            category=category,
            params=params,
            success=success,
            duration_seconds=duration,
            peak_memory_mb=peak_memory,
            cpu_percent=peak_cpu,
            error_type=error_type,
            error_message=error_message,
            traceback_summary=tb_summary,
            warnings=warnings,
            metrics=metrics
        )

        self.record_result(result)
        return result

    def record_result(self, result: StressTestResult):
        """Record test result and print summary."""
        self.results.append(result)

        status = "✓" if result.success else "✗"
        print(f"{status} [{result.category}] {result.test_name} "
              f"params={result.params} "
              f"time={result.duration_seconds:.2f}s "
              f"mem={result.peak_memory_mb:.0f}MB")

        if not result.success:
            print(f"  ERROR: {result.error_type}: {result.error_message}")
            if result.error_type in ["MemoryError", "DeadlockError", "SegmentationFault"]:
                self.critical_failures.append(result)

        if result.warnings:
            for w in result.warnings[:3]:
                print(f"  WARN: {w}")

    def find_breaking_point_binary(self, test_func: Callable, param_name: str,
                                   values: list[Any], category: str, **fixed_params) -> dict:
        """Binary search to find breaking point."""
        left, right = 0, len(values) - 1
        last_success = None
        first_failure = None

        while left <= right:
            mid = (left + right) // 2
            param_value = values[mid]
            params = {**fixed_params, param_name: param_value, "_category": category}

            result = self.run_test_with_timeout(test_func, params)

            if result.success:
                last_success = param_value
                left = mid + 1
            else:
                first_failure = param_value
                right = mid - 1

        return {
            "param": param_name,
            "last_success": last_success,
            "first_failure": first_failure,
            "fixed_params": fixed_params
        }

    def find_breaking_point_linear(self, test_func: Callable, param_name: str,
                                   values: list[Any], category: str, **fixed_params) -> dict:
        """Linear search to find breaking point (for non-monotonic failures)."""
        last_success = None
        first_failure = None

        for param_value in values:
            params = {**fixed_params, param_name: param_value, "_category": category}
            result = self.run_test_with_timeout(test_func, params, timeout=60)

            if result.success:
                last_success = param_value
            else:
                if first_failure is None:
                    first_failure = param_value
                # Continue to find if it recovers

        return {
            "param": param_name,
            "last_success": last_success,
            "first_failure": first_failure,
            "fixed_params": fixed_params
        }

    def save_results(self):
        """Save comprehensive test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Breaking points JSON
        breaking_file = self.output_dir / "stress_test_breaking_points.json"
        with open(breaking_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "breaking_points": self.breaking_points,
                "critical_failures_count": len(self.critical_failures),
                "total_tests": len(self.results),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "success_rate": (len(self.results) - sum(1 for r in self.results if not r.success)) / len(self.results) * 100 if self.results else 0
            }, f, indent=2)
        print(f"\n✓ Breaking points saved: {breaking_file}")

        # 2. Detailed results JSON
        results_file = self.output_dir / f"stress_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"✓ Detailed results saved: {results_file}")

        # 3. Crash analysis report
        report_file = self.output_dir / "crash_analysis_report.md"
        self._generate_crash_report(report_file)
        print(f"✓ Crash analysis saved: {report_file}")

        # 4. Critical fixes CSV
        fixes_file = self.output_dir / "critical_fixes_needed.csv"
        self._generate_fixes_csv(fixes_file)
        print(f"✓ Critical fixes CSV saved: {fixes_file}")

    def _generate_crash_report(self, output_file: Path):
        """Generate detailed crash analysis report."""
        failures = [r for r in self.results if not r.success]

        with open(output_file, "w") as f:
            f.write("# Factor Engine Stress Test - Crash Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Duration:** {time.time() - self.start_time:.1f}s\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tests:** {len(self.results)}\n")
            f.write(f"- **Passed:** {len(self.results) - len(failures)}\n")
            f.write(f"- **Failed:** {len(failures)}\n")
            f.write(f"- **Success Rate:** {(len(self.results)-len(failures))/len(self.results)*100:.1f}%\n")
            f.write(f"- **Critical Failures:** {len(self.critical_failures)}\n\n")

            # Group by category
            by_category = {}
            for r in self.results:
                if r.category not in by_category:
                    by_category[r.category] = {"passed": 0, "failed": 0}
                if r.success:
                    by_category[r.category]["passed"] += 1
                else:
                    by_category[r.category]["failed"] += 1

            f.write("## Results by Category\n\n")
            f.write("| Category | Passed | Failed | Success Rate |\n")
            f.write("|----------|--------|--------|-------------|\n")
            for cat, counts in sorted(by_category.items()):
                total = counts["passed"] + counts["failed"]
                rate = counts["passed"] / total * 100 if total > 0 else 0
                f.write(f"| {cat} | {counts['passed']} | {counts['failed']} | {rate:.1f}% |\n")
            f.write("\n")

            # Group failures by error type
            by_error = {}
            for r in failures:
                et = r.error_type or "Unknown"
                if et not in by_error:
                    by_error[et] = []
                by_error[et].append(r)

            f.write("## Failures by Error Type\n\n")
            for error_type, results in sorted(by_error.items(), key=lambda x: -len(x[1])):
                f.write(f"### {error_type} ({len(results)} occurrences)\n\n")

                for i, r in enumerate(results[:5], 1):
                    f.write(f"#### {i}. {r.test_name}\n\n")
                    f.write(f"**Category:** {r.category}\n\n")
                    f.write(f"**Parameters:**\n```json\n{json.dumps(r.params, indent=2)}\n```\n\n")
                    f.write(f"**Error Message:** {r.error_message}\n\n")
                    f.write(f"**Duration:** {r.duration_seconds:.2f}s | ")
                    f.write(f"**Peak Memory:** {r.peak_memory_mb:.0f}MB | ")
                    f.write(f"**CPU:** {r.cpu_percent:.0f}%\n\n")

                    if r.traceback_summary:
                        f.write(f"**Traceback:**\n```python\n{r.traceback_summary}\n```\n\n")

                if len(results) > 5:
                    f.write(f"*... and {len(results)-5} more*\n\n")

            # Breaking points summary
            f.write("## Breaking Points Summary\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.breaking_points, indent=2))
            f.write("\n```\n\n")

            # Critical failures
            if self.critical_failures:
                f.write("## 🚨 CRITICAL FAILURES (Require Immediate Fix)\n\n")
                for r in self.critical_failures:
                    f.write(f"### {r.error_type}: {r.test_name}\n\n")
                    f.write(f"**Params:** {r.params}\n\n")
                    f.write(f"**Message:** {r.error_message}\n\n")
                    f.write("---\n\n")

    def _generate_fixes_csv(self, output_file: Path):
        """Generate prioritized fixes CSV."""
        import csv

        failures = [r for r in self.results if not r.success]

        # Priority mapping
        priority_map = {
            "MemoryError": ("P0", "CRITICAL"),
            "SegmentationFault": ("P0", "CRITICAL"),
            "DeadlockError": ("P0", "CRITICAL"),
            "TimeoutError": ("P1", "HIGH"),
            "RecursionError": ("P1", "HIGH"),
            "ConnectionPoolExhausted": ("P1", "HIGH"),
            "ResourceExhausted": ("P1", "HIGH"),
            "DatabaseLocked": ("P2", "MEDIUM"),
            "ValueError": ("P2", "MEDIUM"),
            "AssertionError": ("P2", "MEDIUM"),
        }

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Priority", "Severity", "Category", "Error Type",
                "Test", "Params", "Message", "Duration(s)", "Memory(MB)"
            ])

            for r in sorted(failures, key=lambda x: priority_map.get(x.error_type, ("P3", "LOW"))[0]):
                priority, severity = priority_map.get(r.error_type, ("P3", "LOW"))
                writer.writerow([
                    priority,
                    severity,
                    r.category,
                    r.error_type,
                    r.test_name,
                    str(r.params)[:100],
                    (r.error_message or "")[:150],
                    f"{r.duration_seconds:.2f}",
                    f"{r.peak_memory_mb:.0f}"
                ])


# ============================================================================
# Test Implementations - Factor Engine Specific
# ============================================================================

def test_simple_factor_compilation(num_factors: int) -> dict:
    """Test compilation of simple factors."""
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.api import rank, ts_mean
    from factor_engine.backend.debug_backend import DebugBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.datasource import DataSource

    class DummyDataSource(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DummyDataSource())

    factors = []
    for i in range(num_factors):
        expr = ts_mean(col("close"), 20 + i)
        f = Factor(name=f"factor_{i}", expr=expr, freq="1d", universe="test")
        factors.append(f)

    # Compile all factors
    dag = engine.compile_many(factors)

    return {
        "compiled": len(factors),
        "dag_nodes": len(dag.roots) if hasattr(dag, "roots") else 0,
        "warnings": []
    }


def test_deep_dag_compilation(depth: int) -> dict:
    """Test compilation of deeply nested factor DAG."""
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.api import ts_mean
    from factor_engine.backend.debug_backend import DebugBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.datasource import DataSource

    class DummyDataSource(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DummyDataSource())

    # Build deeply nested expression
    expr = col("close")
    for i in range(depth):
        expr = ts_mean(expr, 5)

    factor = Factor(name="deep_factor", expr=expr, freq="1d", universe="test")
    dag = engine.compile(factor)

    return {"depth": depth, "compiled": True}


def test_wide_dag_compilation(width: int) -> dict:
    """Test compilation of wide factor DAG (many parallel factors)."""
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.api import ts_mean, rank, ts_std
    from factor_engine.backend.debug_backend import DebugBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.datasource import DataSource

    class DummyDataSource(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DummyDataSource())

    # Create wide DAG with different operations
    factors = []
    for i in range(width):
        if i % 3 == 0:
            expr = ts_mean(col("close"), 20)
        elif i % 3 == 1:
            expr = ts_std(col("close"), 20)
        else:
            expr = rank(col("close"))

        factor = Factor(name=f"wide_factor_{i}", expr=expr, freq="1d", universe="test")
        factors.append(factor)

    dag = engine.compile_many(factors)

    return {"width": width, "compiled": len(factors)}


def test_memory_allocation_limit(size_mb: int) -> dict:
    """Test memory allocation limits."""
    import numpy as np

    # Try to allocate specified memory
    size_elements = int(size_mb * 1024**2 / 8)
    arr = np.zeros(size_elements, dtype=np.float64)
    arr[0] = 1.0  # Touch memory to ensure allocation
    arr[-1] = 1.0

    # Force calculation to prevent optimization
    checksum = arr.sum()

    del arr
    gc.collect()

    return {"allocated_mb": size_mb, "checksum": checksum}


def test_concurrent_factor_requests(num_concurrent: int, complexity: int) -> dict:
    """Test concurrent factor compilation/execution."""
    import threading
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.api import ts_mean
    from factor_engine.backend.debug_backend import DebugBackend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.datasource import DataSource

    class DummyDataSource(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    results = []
    errors = []
    lock = threading.Lock()

    def worker(worker_id: int):
        try:
            engine = FactorEngine(backend=DebugBackend(), data_source=DummyDataSource())
            expr = col("close")
            for _ in range(complexity):
                expr = ts_mean(expr, 10)

            factor = Factor(name=f"factor_{worker_id}", expr=expr, freq="1d", universe="test")
            dag = engine.compile(factor)

            with lock:
                results.append(worker_id)
        except Exception as e:
            with lock:
                errors.append((worker_id, str(e)))

    threads = []
    for i in range(num_concurrent):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=60)

    if errors:
        raise RuntimeError(f"Worker errors: {len(errors)} failures")

    return {"completed": len(results), "errors": len(errors)}


def test_process_pool_factor_execution(num_processes: int) -> dict:
    """Test factor execution in process pool."""
    from concurrent.futures import ProcessPoolExecutor

    def compile_factor(factor_id: int):
        # Import inside worker
        from factor_engine.api.columns import col
        from factor_engine.api.factor import Factor
        from factor_engine.api import ts_mean
        from factor_engine.backend.debug_backend import DebugBackend
        from factor_engine.runtime.engine import FactorEngine
        from factor_engine.storage.datasource import DataSource

        class DummyDataSource(DataSource):
            def load_column(self, name: str):
                raise NotImplementedError()

        engine = FactorEngine(backend=DebugBackend(), data_source=DummyDataSource())
        expr = ts_mean(col("close"), 20 + factor_id)
        factor = Factor(name=f"factor_{factor_id}", expr=expr, freq="1d", universe="test")
        dag = engine.compile(factor)
        return factor_id

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(compile_factor, i) for i in range(num_processes * 2)]
        results = [f.result(timeout=30) for f in futures]

    return {"processes": num_processes, "completed": len(results)}


def test_disk_io_stress(num_files: int, file_size_mb: int) -> dict:
    """Test disk I/O stress."""
    temp_dir = Path(tempfile.gettempdir()) / f"fe_stress_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)

    try:
        files_created = []
        for i in range(num_files):
            file_path = temp_dir / f"test_{i}.dat"
            data = os.urandom(int(file_size_mb * 1024**2))
            file_path.write_bytes(data)
            files_created.append(file_path)

        # Read back to verify
        for file_path in files_created:
            size = file_path.stat().st_size

        return {"files": num_files, "total_mb": num_files * file_size_mb}
    finally:
        # Cleanup
        for file_path in files_created:
            try:
                file_path.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


# ============================================================================
# Main Test Execution
# ============================================================================

def main():
    """Execute comprehensive factor engine stress tests."""
    print("=" * 80)
    print("FACTOR ENGINE COMPREHENSIVE STRESS TESTING")
    print("=" * 80)
    print(f"PID: {os.getpid()}")
    print(f"CPU Count: {multiprocessing.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"Available Disk: {psutil.disk_usage('/tmp').free / 1024**3:.1f} GB")
    print("=" * 80)
    print()

    tester = FactorEngineStressTester()

    # ========================================================================
    # TEST CATEGORY 1: COMPILATION SCALE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST CATEGORY 1: COMPILATION SCALE")
    print("=" * 80)

    factor_counts = [1, 10, 50, 100, 500, 1000, 2000, 5000]
    print("\n>>> Finding breaking point: Number of factors...")
    bp_factors = tester.find_breaking_point_binary(
        test_simple_factor_compilation,
        "num_factors",
        factor_counts,
        "compilation_scale"
    )
    tester.breaking_points["max_factors"] = bp_factors

    # ========================================================================
    # TEST CATEGORY 2: DAG COMPLEXITY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST CATEGORY 2: DAG COMPLEXITY")
    print("=" * 80)

    depth_levels = [5, 10, 20, 50, 100, 200, 500, 1000]
    print("\n>>> Finding breaking point: DAG depth...")
    bp_depth = tester.find_breaking_point_binary(
        test_deep_dag_compilation,
        "depth",
        depth_levels,
        "dag_complexity"
    )
    tester.breaking_points["max_dag_depth"] = bp_depth

    width_levels = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    print("\n>>> Finding breaking point: DAG width...")
    bp_width = tester.find_breaking_point_binary(
        test_wide_dag_compilation,
        "width",
        width_levels,
        "dag_complexity"
    )
    tester.breaking_points["max_dag_width"] = bp_width

    # ========================================================================
    # TEST CATEGORY 3: CONCURRENCY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST CATEGORY 3: CONCURRENCY")
    print("=" * 80)

    concurrency_levels = [1, 10, 50, 100, 200, 500, 1000]
    print("\n>>> Finding breaking point: Concurrent threads (low complexity)...")
    bp_threads_low = tester.find_breaking_point_linear(
        test_concurrent_factor_requests,
        "num_concurrent",
        concurrency_levels,
        "concurrency",
        complexity=1
    )
    tester.breaking_points["max_concurrent_threads_low"] = bp_threads_low

    print("\n>>> Finding breaking point: Concurrent threads (high complexity)...")
    bp_threads_high = tester.find_breaking_point_linear(
        test_concurrent_factor_requests,
        "num_concurrent",
        concurrency_levels[:5],
        "concurrency",
        complexity=10
    )
    tester.breaking_points["max_concurrent_threads_high"] = bp_threads_high

    # Process pool concurrency
    process_levels = [2, 4, 8, 16, 32, 64]
    cpu_count = multiprocessing.cpu_count()
    process_levels = [p for p in process_levels if p <= cpu_count * 4]

    print(f"\n>>> Finding breaking point: Process pool ({cpu_count} CPUs)...")
    bp_processes = tester.find_breaking_point_linear(
        test_process_pool_factor_execution,
        "num_processes",
        process_levels,
        "concurrency"
    )
    tester.breaking_points["max_process_pool"] = bp_processes

    # ========================================================================
    # TEST CATEGORY 4: MEMORY PRESSURE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST CATEGORY 4: MEMORY PRESSURE")
    print("=" * 80)

    available_mb = psutil.virtual_memory().available / 1024**2
    print(f"Available memory: {available_mb:.0f} MB")

    # Test up to 70% of available memory
    max_test_mb = int(available_mb * 0.7)
    memory_levels = [100, 500, 1000, 2000, 4000, 8000, max_test_mb // 2, max_test_mb]
    memory_levels = [m for m in memory_levels if m <= max_test_mb]

    print(f"\n>>> Finding breaking point: Memory allocation (max {max_test_mb}MB)...")
    bp_memory = tester.find_breaking_point_binary(
        test_memory_allocation_limit,
        "size_mb",
        memory_levels,
        "memory_pressure"
    )
    tester.breaking_points["max_memory_allocation"] = bp_memory

    # ========================================================================
    # TEST CATEGORY 5: DISK I/O PRESSURE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST CATEGORY 5: DISK I/O PRESSURE")
    print("=" * 80)

    available_disk_mb = psutil.disk_usage("/tmp").free / 1024**2
    print(f"Available disk: {available_disk_mb:.0f} MB")

    # Test small files
    file_counts = [10, 50, 100, 500, 1000, 2000]
    print("\n>>> Finding breaking point: Many small files (1MB each)...")
    bp_small_files = tester.find_breaking_point_linear(
        test_disk_io_stress,
        "num_files",
        file_counts,
        "disk_io",
        file_size_mb=1
    )
    tester.breaking_points["max_small_files"] = bp_small_files

    # Test large files
    max_test_disk_mb = min(5000, int(available_disk_mb * 0.5))
    large_file_counts = [5, 10, 20, 50, 100]
    print(f"\n>>> Finding breaking point: Large files (100MB each, max {max_test_disk_mb}MB)...")
    bp_large_files = tester.find_breaking_point_linear(
        test_disk_io_stress,
        "num_files",
        [n for n in large_file_counts if n * 100 <= max_test_disk_mb],
        "disk_io",
        file_size_mb=100
    )
    tester.breaking_points["max_large_files"] = bp_large_files

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    tester.save_results()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY")
    print("=" * 80)

    total = len(tester.results)
    failures = sum(1 for r in tester.results if not r.success)
    success_rate = (total - failures) / total * 100 if total > 0 else 0

    print(f"Total Tests: {total}")
    print(f"Passed: {total - failures}")
    print(f"Failed: {failures}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Critical Failures: {len(tester.critical_failures)}")
    print(f"Total Duration: {time.time() - tester.start_time:.1f}s")
    print()

    print("Breaking Points Found:")
    for key, value in tester.breaking_points.items():
        print(f"  {key}:")
        print(f"    Last Success: {value.get('last_success')}")
        print(f"    First Failure: {value.get('first_failure')}")

    print("\n" + "=" * 80)

    # Return exit code
    return 1 if tester.critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
