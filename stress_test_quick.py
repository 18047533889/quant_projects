#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick stress test - identify critical breaking points in <60 seconds."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results = {"tests": [], "crashes": [], "breaking_points": {}}


def test(name, func, *args, **kwargs):
    """Run a test and record result."""
    print(f"  Testing {name}...", end=" ", flush=True)
    start = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"✓ {duration:.2f}s")
        results["tests"].append({"name": name, "success": True, "duration": duration})
        return True
    except Exception as e:
        duration = time.time() - start
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"✗ {error_type}: {error_msg[:60]}")
        results["tests"].append({
            "name": name,
            "success": False,
            "duration": duration,
            "error": error_type,
            "message": error_msg[:200]
        })
        results["crashes"].append({
            "test": name,
            "error_type": error_type,
            "error_msg": error_msg,
            "traceback": traceback.format_exc()[:500]
        })
        return False


def find_limit(name, func, start, max_val, step_mult=2.0):
    """Find breaking point by exponential search."""
    print(f"\n[{name}]")
    val = start
    last_success = None

    while val <= max_val:
        if test(f"{name}({val})", func, val):
            last_success = val
            val = int(val * step_mult)
        else:
            results["breaking_points"][name] = {
                "last_success": last_success,
                "first_failure": val
            }
            return last_success

    results["breaking_points"][name] = {
        "last_success": last_success,
        "first_failure": None
    }
    return last_success


# ============================================================================
# Quick tests
# ============================================================================

def simple_factor_compile(n):
    """Compile n simple factors."""
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DS())
    factors = [Factor(name=f"f{i}", expr=ts_mean(col("close"), 20), freq="1d", universe="test")
               for i in range(n)]
    dag = engine.compile_many(factors)
    return len(factors)


def deep_dag(depth):
    """Compile deeply nested factor."""
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DS())
    expr = col("close")
    for _ in range(depth):
        expr = ts_mean(expr, 5)
    f = Factor(name="deep", expr=expr, freq="1d", universe="test")
    dag = engine.compile(f)
    return depth


def wide_dag(width):
    """Compile wide DAG."""
    from api.columns import col
    from api.factor import Factor
    from api import ts_mean, ts_std, rank
    from backend.debug_backend import DebugBackend
    from runtime.engine import FactorEngine
    from storage.datasource import DataSource

    class DS(DataSource):
        def load_column(self, name: str):
            raise NotImplementedError()

    engine = FactorEngine(backend=DebugBackend(), data_source=DS())
    ops = [ts_mean, ts_std, rank]
    factors = [Factor(name=f"f{i}", expr=ops[i % 3](col("close"), 20 if i % 3 < 2 else None),
                     freq="1d", universe="test")
               for i in range(width)]
    dag = engine.compile_many(factors)
    return width


def memory_alloc(mb):
    """Allocate memory."""
    import numpy as np
    arr = np.zeros(int(mb * 1024**2 / 8), dtype=np.float64)
    arr[0] = 1.0
    arr[-1] = 1.0
    return arr.sum()


def thread_spawn(n):
    """Spawn threads."""
    import threading
    threads = []

    def worker():
        time.sleep(0.05)

    for _ in range(n):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=5)

    return len(threads)


def file_handles(n):
    """Open file handles."""
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / f"quick_stress_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)

    try:
        files = []
        for i in range(n):
            f = open(temp_dir / f"f{i}.tmp", "w")
            f.write("x")
            files.append(f)

        result = len(files)

        for f in files:
            f.close()

        return result
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("QUICK STRESS TEST - Finding Critical Breaking Points")
    print("=" * 70)

    start_time = time.time()

    # Test 1: Factor compilation scale
    find_limit("simple_factors", simple_factor_compile, 10, 10000, 2.0)

    # Test 2: DAG depth
    find_limit("dag_depth", deep_dag, 10, 1000, 1.5)

    # Test 3: DAG width
    find_limit("dag_width", wide_dag, 10, 5000, 2.0)

    # Test 4: Memory
    find_limit("memory_mb", memory_alloc, 100, 10000, 1.5)

    # Test 5: Threads
    find_limit("threads", thread_spawn, 10, 10000, 2.0)

    # Test 6: File handles
    find_limit("file_handles", file_handles, 100, 10000, 2.0)

    # Save results
    duration = time.time() - start_time

    output_file = Path("/tmp/stress_test_breaking_points.json")
    results["summary"] = {
        "duration_seconds": duration,
        "total_tests": len(results["tests"]),
        "crashes": len(results["crashes"]),
        "success_rate": sum(1 for t in results["tests"] if t["success"]) / len(results["tests"]) * 100
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s")
    print(f"Tests: {len(results['tests'])} ({results['summary']['success_rate']:.1f}% success)")
    print(f"Crashes: {len(results['crashes'])}")
    print()
    print("Breaking Points:")
    for name, bp in results["breaking_points"].items():
        print(f"  {name:20s}: last_success={bp['last_success']}, first_failure={bp['first_failure']}")

    print(f"\n✓ Results saved to: {output_file}")

    # Create crash report
    if results["crashes"]:
        crash_file = Path("/tmp/crash_analysis_report.md")
        with open(crash_file, "w") as f:
            f.write("# Crash Analysis Report\n\n")
            f.write(f"**Crashes:** {len(results['crashes'])}\n\n")
            for crash in results["crashes"]:
                f.write(f"## {crash['test']}\n\n")
                f.write(f"**Error:** {crash['error_type']}\n\n")
                f.write(f"**Message:** {crash['error_msg']}\n\n")
                f.write(f"```\n{crash['traceback']}\n```\n\n")
        print(f"✓ Crash report saved to: {crash_file}")

    # Create fixes CSV
    if results["crashes"]:
        fixes_file = Path("/tmp/critical_fixes_needed.csv")
        with open(fixes_file, "w") as f:
            f.write("Priority,Test,Error Type,Message\n")
            priority_map = {
                "MemoryError": "P0",
                "RecursionError": "P0",
                "OSError": "P1",
                "RuntimeError": "P2",
            }
            for crash in results["crashes"]:
                priority = priority_map.get(crash["error_type"], "P2")
                msg = crash["error_msg"].replace(",", ";")[:100]
                f.write(f"{priority},{crash['test']},{crash['error_type']},{msg}\n")
        print(f"✓ Fixes CSV saved to: {fixes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
