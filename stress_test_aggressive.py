#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggressive parallel stress testing - find breaking points FAST.

Runs multiple stress test categories in parallel to quickly identify:
1. Memory limits (OOM)
2. Concurrency limits (deadlocks, race conditions)
3. Resource exhaustion (file handles, connections)
4. Performance cliffs (exponential slowdown)
"""
from __future__ import annotations

import concurrent.futures
import gc
import json
import multiprocessing
import os
import psutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_FE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_FE_ROOT))
sys.path.insert(0, str(_FE_ROOT.parent))


@dataclass
class CrashReport:
    """Crash/failure report."""
    test_name: str
    params: dict
    error_type: str
    error_msg: str
    traceback: str
    memory_mb: float
    duration_s: float
    timestamp: str


class AggressiveStressTester:
    """Fast, aggressive stress testing."""

    def __init__(self):
        self.crashes: list[CrashReport] = []
        self.lock = threading.Lock()
        self.start_time = time.time()

    def record_crash(self, crash: CrashReport):
        """Record a crash."""
        with self.lock:
            self.crashes.append(crash)
            print(f"💥 CRASH: {crash.test_name} - {crash.error_type}: {crash.error_msg[:80]}")

    def run_until_crash(self, test_func: Callable, param_name: str,
                       start_value: Any, multiplier: float = 2.0, timeout: int = 30) -> Any:
        """Exponentially increase parameter until crash."""
        value = start_value
        last_success = None

        while True:
            try:
                print(f"  Testing {param_name}={value}...")
                start = time.time()
                proc = psutil.Process()
                initial_mem = proc.memory_info().rss / 1024**2

                # Run with timeout
                result = self._run_with_timeout(test_func, {param_name: value}, timeout)

                duration = time.time() - start
                peak_mem = proc.memory_info().rss / 1024**2

                print(f"    ✓ Success: {duration:.2f}s, {peak_mem:.0f}MB")
                last_success = value
                value = int(value * multiplier)

            except Exception as e:
                duration = time.time() - start
                mem = proc.memory_info().rss / 1024**2

                crash = CrashReport(
                    test_name=test_func.__name__,
                    params={param_name: value},
                    error_type=type(e).__name__,
                    error_msg=str(e),
                    traceback=traceback.format_exc(),
                    memory_mb=mem,
                    duration_s=duration,
                    timestamp=datetime.now().isoformat()
                )
                self.record_crash(crash)
                return last_success

    def _run_with_timeout(self, func: Callable, kwargs: dict, timeout: int):
        """Run function with timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **kwargs)
            return future.result(timeout=timeout)

    def save_results(self, output_dir: Path):
        """Save crash reports."""
        output_dir.mkdir(exist_ok=True)

        # Crashes JSON
        crashes_file = output_dir / "aggressive_crashes.json"
        with open(crashes_file, "w") as f:
            json.dump([asdict(c) for c in self.crashes], f, indent=2)
        print(f"\n✓ Saved {len(self.crashes)} crashes to: {crashes_file}")

        # Summary report
        report_file = output_dir / "aggressive_summary.md"
        with open(report_file, "w") as f:
            f.write("# Aggressive Stress Test - Crash Summary\n\n")
            f.write(f"**Duration:** {time.time() - self.start_time:.1f}s\n\n")
            f.write(f"**Total Crashes:** {len(self.crashes)}\n\n")

            # Group by error type
            by_type = {}
            for c in self.crashes:
                if c.error_type not in by_type:
                    by_type[c.error_type] = []
                by_type[c.error_type].append(c)

            f.write("## Crashes by Type\n\n")
            for error_type, crashes in sorted(by_type.items(), key=lambda x: -len(x[1])):
                f.write(f"### {error_type} ({len(crashes)})\n\n")
                for c in crashes:
                    f.write(f"- **{c.test_name}** params={c.params}\n")
                    f.write(f"  - Error: {c.error_msg[:150]}\n")
                    f.write(f"  - Memory: {c.memory_mb:.0f}MB, Duration: {c.duration_s:.2f}s\n\n")

        print(f"✓ Saved report to: {report_file}")


# ============================================================================
# Aggressive test implementations
# ============================================================================

def stress_memory_rapid(size_mb: int):
    """Rapidly allocate memory."""
    import numpy as np
    arrays = []
    chunk_mb = 100
    for _ in range(size_mb // chunk_mb):
        arr = np.zeros(int(chunk_mb * 1024**2 / 8), dtype=np.float64)
        arr[0] = 1.0
        arrays.append(arr)
    return sum(a.sum() for a in arrays)


def stress_threads_rapid(num_threads: int):
    """Create many threads rapidly."""
    import threading

    def worker():
        time.sleep(0.1)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=10)

    return len(threads)


def stress_file_handles(num_files: int):
    """Open many file handles."""
    temp_dir = Path(tempfile.gettempdir()) / f"stress_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)

    try:
        files = []
        for i in range(num_files):
            f = open(temp_dir / f"file_{i}.tmp", "w")
            f.write("test")
            files.append(f)

        # Keep all open
        time.sleep(0.1)

        for f in files:
            f.close()

        return len(files)
    finally:
        for f in files:
            try:
                f.close()
            except:
                pass
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def stress_recursion(depth: int):
    """Test recursion depth."""
    def recurse(n):
        if n <= 0:
            return 1
        return recurse(n - 1) + 1

    return recurse(depth)


def stress_subprocess_spawn(num_procs: int):
    """Spawn many subprocesses."""
    import subprocess

    procs = []
    for _ in range(num_procs):
        p = subprocess.Popen(["sleep", "0.1"])
        procs.append(p)

    for p in procs:
        p.wait(timeout=5)

    return len(procs)


def stress_factor_compilation_rapid(num_factors: int):
    """Rapidly compile many factors."""
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DummyDS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DummyDS())

    factors = []
    for i in range(num_factors):
        expr = ts_mean(col("close"), 10 + (i % 100))
        f = Factor(name=f"f_{i}", expr=expr, freq="1d", universe="test")
        factors.append(f)

    dag = engine.compile_many(factors)
    return len(factors)


def stress_nested_dag(depth: int):
    """Deeply nested DAG."""
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DummyDS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DummyDS())

    expr = col("close")
    for _ in range(depth):
        expr = ts_mean(expr, 5)

    f = Factor(name="deep", expr=expr, freq="1d", universe="test")
    dag = engine.compile(f)
    return depth


def stress_concurrent_compilation(num_concurrent: int):
    """Concurrent factor compilation."""
    import threading
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DummyDS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    results = []
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            engine = FactorEngine(backend=DebugBackend(), data_source=DummyDS())
            expr = ts_mean(col("close"), 20)
            f = Factor(name=f"f_{i}", expr=expr, freq="1d", universe="test")
            dag = engine.compile(f)
            with lock:
                results.append(i)
        except Exception as e:
            with lock:
                errors.append((i, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    if errors:
        raise RuntimeError(f"{len(errors)} workers failed")

    return len(results)


# ============================================================================
# Main execution
# ============================================================================

def main():
    print("=" * 80)
    print("AGGRESSIVE PARALLEL STRESS TESTING")
    print("=" * 80)
    print(f"Finding breaking points FAST...")
    print(f"System: {multiprocessing.cpu_count()} CPUs, "
          f"{psutil.virtual_memory().total / 1024**3:.1f}GB RAM")
    print("=" * 80)
    print()

    tester = AggressiveStressTester()

    # Run tests in parallel
    tests = [
        ("MEMORY", stress_memory_rapid, "size_mb", 100, 1.5),
        ("THREADS", stress_threads_rapid, "num_threads", 10, 2.0),
        ("FILE_HANDLES", stress_file_handles, "num_files", 100, 2.0),
        ("RECURSION", stress_recursion, "depth", 100, 1.5),
        ("SUBPROCESSES", stress_subprocess_spawn, "num_procs", 10, 2.0),
        ("FACTOR_COMPILATION", stress_factor_compilation_rapid, "num_factors", 10, 2.0),
        ("NESTED_DAG", stress_nested_dag, "depth", 10, 1.5),
        ("CONCURRENT_COMPILATION", stress_concurrent_compilation, "num_concurrent", 10, 2.0),
    ]

    def run_test_category(name, func, param, start, mult):
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"{'='*60}")
        last_success = tester.run_until_crash(func, param, start, mult, timeout=60)
        print(f"✓ {name} last success: {param}={last_success}")
        return (name, last_success)

    # Run all tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_test_category, name, func, param, start, mult): name
            for name, func, param, start, mult in tests
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                test_name, last_success = future.result()
                results[test_name] = last_success
            except Exception as e:
                print(f"✗ Test category {name} failed: {e}")
                results[name] = "FAILED"

    # Save results
    output_dir = Path("/tmp")
    tester.save_results(output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("BREAKING POINTS FOUND")
    print("=" * 80)
    for name, value in results.items():
        print(f"{name:30s}: {value}")

    print(f"\nTotal crashes recorded: {len(tester.crashes)}")
    print(f"Duration: {time.time() - tester.start_time:.1f}s")

    # Create critical fixes list
    critical_types = {"MemoryError", "RecursionError", "OSError", "ResourceWarning",
                     "DeadlockError", "SegmentationFault"}
    critical_crashes = [c for c in tester.crashes if c.error_type in critical_types]

    if critical_crashes:
        print(f"\n🚨 {len(critical_crashes)} CRITICAL CRASHES REQUIRE IMMEDIATE FIXES")
        for c in critical_crashes[:5]:
            print(f"  - {c.error_type}: {c.test_name} {c.params}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
