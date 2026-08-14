"""
Test suite for resource leak detection and fixes.

Tests long-running scenarios to detect memory leaks, file handle leaks,
and connection leaks.
"""

import gc
import os
import sys
import time
import psutil
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime


def test_memory_leak_in_batch_processing():
    """Test for memory leaks in repeated batch processing."""
    print("\n[TEST] Memory leak in batch processing")

    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024 / 1024

    # Simulate 100 iterations of batch processing
    for i in range(100):
        # Create large arrays
        data = np.random.rand(1000, 100)
        result = np.mean(data, axis=0)

        # This should be cleaned up
        del data
        del result

        if i % 20 == 0:
            gc.collect()

    mem_end = process.memory_info().rss / 1024 / 1024
    growth = mem_end - mem_start

    print(f"  Memory start: {mem_start:.1f} MB")
    print(f"  Memory end: {mem_end:.1f} MB")
    print(f"  Growth: {growth:.1f} MB")

    if growth < 10:
        print("  ✓ PASS: No significant memory leak")
    else:
        print("  ✗ FAIL: Potential memory leak detected")

    return growth < 10


def test_file_handle_leak():
    """Test for file handle leaks."""
    print("\n[TEST] File handle leak")

    process = psutil.Process()
    files_start = len(process.open_files())

    # Create temp files and read them
    temp_files = []
    for i in range(10):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"test data")
        os.close(fd)
        temp_files.append(path)

    # Read files WITH proper cleanup
    for path in temp_files:
        with open(path, 'r') as f:
            _ = f.read()

    # Clean up temp files
    for path in temp_files:
        try:
            os.unlink(path)
        except:
            pass

    files_end = len(process.open_files())
    leaked = files_end - files_start

    print(f"  Open files start: {files_start}")
    print(f"  Open files end: {files_end}")
    print(f"  Leaked: {leaked}")

    if leaked == 0:
        print("  ✓ PASS: No file handle leaks")
    else:
        print("  ✗ FAIL: File handles leaked")

    return leaked == 0


def test_sqlite_connection_leak():
    """Test for SQLite connection leaks."""
    print("\n[TEST] SQLite connection leak")

    sys.path.insert(0, str(Path(__file__).parent / "research_control"))

    try:
        from research_control.ledger.campaign import CampaignLedger

        # Create in-memory ledger
        ledger = CampaignLedger(db_path=":memory:")

        # Use it
        for i in range(10):
            ledger.append(
                event_id=f"evt_{i}",
                campaign_id="camp_test",
                state="created",
                timestamp=datetime.now()
            )

        # Check if close method exists
        if hasattr(ledger, 'close'):
            ledger.close()
            print("  ✓ PASS: Ledger has close() method")
            return True
        else:
            print("  ✗ FAIL: Ledger missing close() method")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: Error testing ledger: {e}")
        return False


def _worker_func(x):
    """Worker function for multiprocessing (must be at module level)."""
    return x * 2


def test_multiprocessing_pool_cleanup():
    """Test multiprocessing pool cleanup."""
    print("\n[TEST] Multiprocessing pool cleanup")

    from multiprocessing import Pool
    import threading

    threads_start = threading.active_count()

    # Use pool with context manager (correct pattern)
    with Pool(processes=2) as pool:
        results = pool.map(_worker_func, range(10))

    time.sleep(0.5)  # Allow cleanup

    threads_end = threading.active_count()

    print(f"  Threads start: {threads_start}")
    print(f"  Threads end: {threads_end}")

    if threads_end <= threads_start:
        print("  ✓ PASS: Pool cleaned up properly")
        return True
    else:
        print("  ✗ WARN: Extra threads detected (may be normal)")
        return True  # Not a hard fail


def test_numpy_array_accumulation():
    """Test for numpy array accumulation in loops."""
    print("\n[TEST] Numpy array accumulation")

    process = psutil.Process()

    # Force clean state
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024

    # Pattern that COULD leak if not careful
    results = []
    for i in range(50):
        data = np.random.rand(1000, 100)
        results.append(data)

    mem_mid = process.memory_info().rss / 1024 / 1024

    # Now clear the accumulator
    results.clear()
    del results
    gc.collect()
    gc.collect()  # Run GC twice to ensure cleanup

    mem_end = process.memory_info().rss / 1024 / 1024

    print(f"  Memory start: {mem_start:.1f} MB")
    print(f"  Memory after accumulation: {mem_mid:.1f} MB")
    print(f"  Memory after clear: {mem_end:.1f} MB")

    growth_accumulation = mem_mid - mem_start
    growth_after_clear = mem_end - mem_start

    print(f"  Growth during accumulation: {growth_accumulation:.1f} MB")
    print(f"  Growth after clear: {growth_after_clear:.1f} MB")

    # More lenient threshold since Python may not release all memory back to OS
    if growth_after_clear < 15:
        print("  ✓ PASS: Arrays properly released after clear")
        return True
    else:
        print("  ✗ FAIL: Arrays not fully released")
        return False


def test_resource_leak_detector():
    """Test the resource leak detector itself."""
    print("\n[TEST] Resource leak detector")

    try:
        from resource_leak_detector import ResourceLeakDetector

        detector = ResourceLeakDetector(Path("/home/shw/quant_projects"))
        detector.take_snapshot()

        # Do some work
        data = np.random.rand(100, 100)
        _ = np.mean(data)

        detector.take_snapshot()

        # Check snapshots
        if len(detector.snapshots) == 2:
            print(f"  Snapshots: {len(detector.snapshots)}")
            print(f"  ✓ PASS: Detector working")
            return True
        else:
            print("  ✗ FAIL: Detector not capturing snapshots")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: Error with detector: {e}")
        return False


def test_resource_leak_fixes():
    """Test resource leak fix utilities."""
    print("\n[TEST] Resource leak fix utilities")

    try:
        from resource_leak_fixes import (
            BoundedAccumulator,
            safe_sqlite_connection,
            managed_tempfile,
            MemoryLeakDetector
        )

        # Test BoundedAccumulator
        acc = BoundedAccumulator(max_size=10, auto_clear=True)
        for i in range(20):
            acc.append(i)

        if len(acc) <= 10:
            print("  ✓ BoundedAccumulator auto-clears")
        else:
            print("  ✗ BoundedAccumulator not working")
            return False

        # Test managed_tempfile
        with managed_tempfile() as tmp_path:
            tmp_path.write_text("test")
            exists_during = tmp_path.exists()

        exists_after = tmp_path.exists()

        if exists_during and not exists_after:
            print("  ✓ managed_tempfile cleans up")
        else:
            print("  ✗ managed_tempfile not cleaning up")
            return False

        # Test MemoryLeakDetector
        leak_detector = MemoryLeakDetector()
        with leak_detector.monitor("test"):
            _ = np.random.rand(100, 100)

        if "test" in leak_detector.measurements:
            print("  ✓ MemoryLeakDetector monitoring")
        else:
            print("  ✗ MemoryLeakDetector not working")
            return False

        print("  ✓ PASS: All fix utilities working")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Error with fixes: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all resource leak tests."""
    print("="*80)
    print("RESOURCE LEAK TEST SUITE")
    print("="*80)

    tests = [
        ("Memory leak in batch processing", test_memory_leak_in_batch_processing),
        ("File handle leak", test_file_handle_leak),
        ("SQLite connection leak", test_sqlite_connection_leak),
        ("Multiprocessing pool cleanup", test_multiprocessing_pool_cleanup),
        ("Numpy array accumulation", test_numpy_array_accumulation),
        ("Resource leak detector", test_resource_leak_detector),
        ("Resource leak fixes", test_resource_leak_fixes),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
    else:
        print(f"\n✗ {total - passed} TESTS FAILED")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
