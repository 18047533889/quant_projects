"""
End-to-end integration test: Resource cleanup and leak detection.

Tests memory leaks, file handle leaks, database connection leaks, and proper cleanup.
"""

import pytest
import numpy as np
import gc
import weakref
from datetime import datetime, timedelta
import tempfile
import os
import psutil
import threading


def get_current_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_open_file_count():
    """Get number of open file descriptors."""
    process = psutil.Process()
    try:
        return len(process.open_files())
    except Exception:
        return 0


def test_memory_leak_factor_batch_creation():
    """
    Test for memory leaks when creating many FactorBatch objects.
    """
    from quant_evaluator import FactorBatch, AxisRef

    gc.collect()
    initial_memory = get_current_memory_mb()

    T, N = 100, 50
    num_iterations = 100

    for i in range(num_iterations):
        factor_values = np.random.randn(T, N, 1)
        time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=(f"leak_test_factor_{i}",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        # Use the batch
        _ = batch.num_factors
        _ = batch.values.shape

        # Explicitly delete
        del batch
        del factor_values

        # Periodic cleanup
        if i % 20 == 0:
            gc.collect()

    # Final cleanup
    gc.collect()
    final_memory = get_current_memory_mb()

    memory_increase = final_memory - initial_memory

    # Allow some memory increase, but not proportional to iterations
    # If there's a leak, memory would grow by ~100MB+
    assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f} MB increase"


def test_memory_leak_repository_operations():
    """
    Test for memory leaks in AssetRepository operations.
    """
    from factor_assets import (
        AssetRepository,
        AssetMetadata,
        LineageRef,
        create_factor_id,
    )

    gc.collect()
    initial_memory = get_current_memory_mb()

    repo = AssetRepository()
    num_iterations = 200

    for i in range(num_iterations):
        factor_id = create_factor_id(
            name=f"repo_leak_test_{i}",
            version="v1",
            params={"index": i},
        )

        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"repo_leak_test_{i}",
            canonical_hash=f"hash_repo_{i}",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Repository leak test {i}",
        )

        lineage = LineageRef(
            factor_id=factor_id,
            parents=(),
        )

        # Register
        asset = repo.register(metadata, lineage)

        # Retrieve
        retrieved = repo.get(factor_id)

        # Delete references
        del asset
        del retrieved
        del metadata
        del lineage

        if i % 50 == 0:
            gc.collect()

    gc.collect()
    final_memory = get_current_memory_mb()

    memory_increase = final_memory - initial_memory

    # Repository will hold references, but growth should be bounded
    assert memory_increase < 100, f"Potential memory leak: {memory_increase:.1f} MB increase"


def test_numpy_array_cleanup():
    """
    Test that NumPy arrays are properly released.
    """
    gc.collect()
    initial_memory = get_current_memory_mb()

    # Create large arrays
    arrays = []
    for i in range(50):
        large_array = np.random.randn(1000, 500, 10)
        arrays.append(large_array)

    mid_memory = get_current_memory_mb()
    memory_allocated = mid_memory - initial_memory

    # Clear arrays
    arrays.clear()
    gc.collect()

    final_memory = get_current_memory_mb()
    memory_retained = final_memory - initial_memory

    # Most memory should be released
    release_ratio = 1.0 - (memory_retained / max(memory_allocated, 1))

    assert release_ratio > 0.8, f"Arrays not properly released: {release_ratio:.1%} released"


def test_weakref_cleanup():
    """
    Test that objects are properly garbage collected using weak references.
    """
    from quant_evaluator import FactorBatch, AxisRef

    T, N = 50, 20
    factor_values = np.random.randn(T, N, 1)
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("weakref_test",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    # Create weak reference
    weak_batch = weakref.ref(batch)

    # Verify object exists
    assert weak_batch() is not None

    # Delete strong reference
    del batch
    del factor_values
    gc.collect()

    # Weak reference should be dead
    assert weak_batch() is None, "Object not garbage collected"


def test_preprocessing_policy_cleanup():
    """
    Test cleanup of preprocessing policies.
    """
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    gc.collect()
    initial_memory = get_current_memory_mb()

    policies = []
    num_policies = 100

    for i in range(num_policies):
        policy = PreprocessingPolicy(
            policy_id=f"cleanup_policy_{i}",
            transforms=[
                TransformSpec(
                    name="winsorize",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0",
                    parameters={"lower": 0.01, "upper": 0.99}
                ),
                TransformSpec(
                    name="standardize",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0",
                    parameters={"method": "zscore"}
                ),
            ],
        )
        policies.append(policy)

    mid_memory = get_current_memory_mb()

    # Clear policies
    policies.clear()
    gc.collect()

    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    # Policies are small objects, should not leak
    assert memory_increase < 10, f"Potential policy leak: {memory_increase:.1f} MB"


def test_label_bundle_cleanup():
    """
    Test cleanup of LabelBundle objects.
    """
    from quant_evaluator import LabelBundle

    gc.collect()
    initial_memory = get_current_memory_mb()

    T, N = 200, 100
    num_bundles = 50

    for i in range(num_bundles):
        label_values = np.random.randn(T, N)
        dates = [datetime(2024, 1, 1) + timedelta(days=j) for j in range(T)]
        decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
        label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
        label_end = tuple((d + timedelta(days=3)).strftime("%Y-%m-%d") for d in dates)

        labels = LabelBundle(
            target_id=f"cleanup_label_{i}",
            values=label_values,
            horizon=2,
            decision_time=decision_times,
            label_start_time=label_start,
            label_end_time=label_end,
        )

        # Use labels
        _ = labels.values.shape

        # Delete
        del labels
        del label_values

        if i % 10 == 0:
            gc.collect()

    gc.collect()
    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    assert memory_increase < 50, f"Potential label bundle leak: {memory_increase:.1f} MB"


def test_file_handle_cleanup():
    """
    Test that temporary files are properly closed and cleaned up.
    """
    initial_fd_count = get_open_file_count()

    temp_files = []

    # Create temporary files
    for i in range(20):
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_file.write(f"Test data {i}\n" * 1000)
        temp_file.close()
        temp_files.append(temp_file.name)

    # Open and close files
    for path in temp_files:
        with open(path, 'r') as f:
            _ = f.read()

    # Clean up
    for path in temp_files:
        if os.path.exists(path):
            os.unlink(path)

    final_fd_count = get_open_file_count()

    # File descriptors should return to baseline
    fd_increase = final_fd_count - initial_fd_count

    assert fd_increase <= 2, f"File descriptors leaked: {fd_increase} increase"


def test_circular_reference_cleanup():
    """
    Test cleanup of circular references.
    """

    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    gc.collect()
    initial_memory = get_current_memory_mb()

    # Create circular references
    for i in range(1000):
        node1 = Node(i)
        node2 = Node(i + 1)
        node1.next = node2
        node2.next = node1

        # Let them go out of scope (circular reference)

    gc.collect()
    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    # Circular references should be collected by Python's GC
    assert memory_increase < 20, f"Circular references leaked: {memory_increase:.1f} MB"


def test_exception_cleanup():
    """
    Test that resources are cleaned up even when exceptions occur.
    """
    from quant_evaluator import FactorBatch, AxisRef

    gc.collect()
    initial_memory = get_current_memory_mb()

    num_iterations = 50
    exceptions_caught = 0

    for i in range(num_iterations):
        try:
            T, N = 50, 20
            factor_values = np.random.randn(T, N, 1)
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=(f"exception_test_{i}",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Simulate exception every other iteration
            if i % 2 == 0:
                raise RuntimeError("Simulated exception")

            del batch
            del factor_values

        except RuntimeError:
            exceptions_caught += 1
            # Resources should still be cleaned up
            gc.collect()

    gc.collect()
    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    assert exceptions_caught == num_iterations // 2
    assert memory_increase < 30, f"Resources leaked on exception: {memory_increase:.1f} MB"


def test_concurrent_cleanup():
    """
    Test cleanup under concurrent access.
    """
    from quant_evaluator import FactorBatch, AxisRef
    import threading

    gc.collect()
    initial_memory = get_current_memory_mb()

    errors = []
    lock = threading.Lock()

    def create_and_cleanup(thread_id):
        try:
            for i in range(20):
                T, N = 30, 15
                factor_values = np.random.randn(T, N, 1)
                time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
                asset_axis = AxisRef(name="asset", dtype="int64", size=N)

                batch = FactorBatch(
                    factor_ids=(f"concurrent_cleanup_{thread_id}_{i}",),
                    time_axis=time_axis,
                    asset_axis=asset_axis,
                    values=factor_values,
                )

                _ = batch.num_factors

                del batch
                del factor_values

            gc.collect()

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "error": str(e)})

    threads = []
    num_threads = 10

    for i in range(num_threads):
        t = threading.Thread(target=create_and_cleanup, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    gc.collect()
    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    assert len(errors) == 0, f"Concurrent cleanup errors: {errors}"
    assert memory_increase < 50, f"Concurrent cleanup leak: {memory_increase:.1f} MB"


def test_large_object_cleanup():
    """
    Test cleanup of large objects.
    """
    from quant_evaluator import FactorBatch, AxisRef

    gc.collect()
    initial_memory = get_current_memory_mb()

    # Create very large batch
    T, N, F = 1000, 500, 10
    factor_values = np.random.randn(T, N, F)
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=tuple(f"large_factor_{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    mid_memory = get_current_memory_mb()
    memory_allocated = mid_memory - initial_memory

    # Should allocate significant memory
    assert memory_allocated > 30, "Large object not allocated"

    # Delete and cleanup
    del batch
    del factor_values
    gc.collect()

    final_memory = get_current_memory_mb()
    memory_retained = final_memory - initial_memory

    # Most memory should be released
    release_ratio = 1.0 - (memory_retained / max(memory_allocated, 1))

    assert release_ratio > 0.8, f"Large object not released: {release_ratio:.1%} released"


def test_repeated_allocation_deallocation():
    """
    Test repeated allocation and deallocation cycles.
    """
    from quant_evaluator import FactorBatch, AxisRef

    gc.collect()
    initial_memory = get_current_memory_mb()

    T, N = 100, 50
    num_cycles = 100

    for cycle in range(num_cycles):
        # Allocate
        batches = []
        for i in range(10):
            factor_values = np.random.randn(T, N, 1)
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=(f"cycle_{cycle}_factor_{i}",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )
            batches.append(batch)

        # Deallocate
        batches.clear()
        gc.collect()

    final_memory = get_current_memory_mb()
    memory_increase = final_memory - initial_memory

    # Memory should stabilize, not grow with each cycle
    assert memory_increase < 50, f"Memory growing with cycles: {memory_increase:.1f} MB"


def test_reference_counting():
    """
    Test proper reference counting for shared objects.
    """
    from quant_evaluator import FactorBatch, AxisRef
    import sys

    T, N = 50, 20
    factor_values = np.random.randn(T, N, 1)

    # Get initial reference count
    initial_refcount = sys.getrefcount(factor_values)

    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    batch = FactorBatch(
        factor_ids=("refcount_test",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=factor_values,
    )

    # Reference count should increase (batch holds reference)
    after_batch_refcount = sys.getrefcount(factor_values)
    assert after_batch_refcount > initial_refcount

    # Delete batch
    del batch
    gc.collect()

    # Reference count should decrease
    final_refcount = sys.getrefcount(factor_values)
    assert final_refcount <= initial_refcount + 1  # +1 for getrefcount's temporary reference


def test_memory_pressure_handling():
    """
    Test system behavior under memory pressure.
    """
    from quant_evaluator import FactorBatch, AxisRef

    gc.collect()
    initial_memory = get_current_memory_mb()

    batches = []

    try:
        # Create batches until memory pressure
        for i in range(100):
            T, N, F = 500, 200, 5
            factor_values = np.random.randn(T, N, F)
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=tuple(f"pressure_factor_{i}_{j}" for j in range(F)),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )
            batches.append(batch)

            current_memory = get_current_memory_mb()
            memory_used = current_memory - initial_memory

            # Stop if using too much memory
            if memory_used > 500:  # 500 MB limit
                break

    except MemoryError:
        # This is expected under pressure
        pass

    # Clean up
    batches.clear()
    gc.collect()

    final_memory = get_current_memory_mb()
    memory_retained = final_memory - initial_memory

    # Should release most memory
    assert memory_retained < 100, f"Memory retained under pressure: {memory_retained:.1f} MB"


def test_cleanup_verification():
    """
    Comprehensive cleanup verification.
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle
    from factor_assets import AssetRepository, AssetMetadata, LineageRef, create_factor_id
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    gc.collect()
    initial_memory = get_current_memory_mb()
    initial_fd_count = get_open_file_count()

    # Create many objects
    for i in range(50):
        # QE objects
        T, N = 50, 20
        factor_values = np.random.randn(T, N, 1)
        time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=(f"verify_factor_{i}",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        label_values = np.random.randn(T, N)
        dates = [datetime(2024, 1, 1) + timedelta(days=j) for j in range(T)]
        decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
        label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
        label_end = tuple((d + timedelta(days=3)).strftime("%Y-%m-%d") for d in dates)

        labels = LabelBundle(
            target_id=f"verify_label_{i}",
            values=label_values,
            horizon=2,
            decision_time=decision_times,
            label_start_time=label_start,
            label_end_time=label_end,
        )

        # FA objects
        repo = AssetRepository()
        factor_id = create_factor_id(
            name=f"verify_asset_{i}",
            version="v1",
            params={"index": i},
        )

        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"verify_asset_{i}",
            canonical_hash=f"hash_verify_{i}",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description="Verification test",
        )

        lineage = LineageRef(factor_id=factor_id, parents=())
        asset = repo.register(metadata, lineage)

        # FP objects
        policy = PreprocessingPolicy(
            policy_id=f"verify_policy_{i}",
            transforms=[
                TransformSpec(
                    name="standardize",
                    kind=TransformKind.CROSS_SECTIONAL,
                    mode=TransformMode.STATELESS,
                    version="1.0",
                    parameters={}
                )
            ],
        )

        # Delete all references
        del batch, labels, asset, policy, factor_values, label_values, metadata, lineage

        if i % 10 == 0:
            gc.collect()

    # Final cleanup
    gc.collect()

    final_memory = get_current_memory_mb()
    final_fd_count = get_open_file_count()

    memory_increase = final_memory - initial_memory
    fd_increase = final_fd_count - initial_fd_count

    # Verify cleanup
    assert memory_increase < 100, f"Memory not cleaned up: {memory_increase:.1f} MB retained"
    assert fd_increase <= 2, f"File descriptors not cleaned up: {fd_increase} retained"
