"""
End-to-end integration test: Concurrency safety across packages.

Tests thread safety, resource contention, deadlock prevention, and race conditions.
"""

import pytest
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import queue


def test_concurrent_factor_registration():
    """
    Test concurrent factor registration in AssetRepository.

    Multiple threads registering factors simultaneously.
    """
    from factor_assets import (
        AssetRepository,
        AssetMetadata,
        LineageRef,
        LifecycleState,
        create_factor_id,
        DuplicateIdentityError,
    )

    repo = AssetRepository()
    results = []
    errors = []
    lock = threading.Lock()

    def register_factor(index):
        try:
            factor_id = create_factor_id(
                name=f"concurrent_factor_{index}",
                version="v1",
                params={"index": index},
            )

            metadata = AssetMetadata(
                factor_id=factor_id,
                canonical_repr=f"concurrent_factor_{index}",
                canonical_hash=f"hash_concurrent_{index}",
                frequency="daily",
                domains=("equity",),
                timing="daily",
                description=f"Concurrent test factor {index}",
            )

            lineage = LineageRef(
                factor_id=factor_id,
                parents=(),
            )

            asset = repo.register(metadata, lineage)

            with lock:
                results.append({"index": index, "factor_id": factor_id, "success": True})

        except Exception as e:
            with lock:
                errors.append({"index": index, "error": str(e)})

    # Launch concurrent registrations
    threads = []
    num_threads = 20

    for i in range(num_threads):
        t = threading.Thread(target=register_factor, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all succeeded
    assert len(errors) == 0, f"Registration errors: {errors}"
    assert len(results) == num_threads

    # Verify all factors are retrievable
    for result in results:
        asset = repo.get(result["factor_id"])
        assert asset is not None
        assert asset.metadata.factor_id == result["factor_id"]


def test_concurrent_evaluation_requests():
    """
    Test concurrent evaluation requests in QE.

    Multiple threads creating and validating FactorBatch objects.
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle

    T, N = 50, 20
    results = []
    errors = []
    lock = threading.Lock()

    def create_and_evaluate(thread_id):
        try:
            # Each thread creates its own batch
            factor_values = np.random.randn(T, N, 1) * 0.1 + thread_id * 0.01
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=(f"factor_thread_{thread_id}",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Validate batch
            assert batch.num_factors == 1
            assert batch.values.shape == (T, N, 1)

            # Create labels
            label_values = np.random.randn(T, N) * 0.02
            dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
            decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
            label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
            label_end = tuple((d + timedelta(days=3)).strftime("%Y-%m-%d") for d in dates)

            labels = LabelBundle(
                target_id=f"fwd_ret_thread_{thread_id}",
                values=label_values,
                horizon=2,
                decision_time=decision_times,
                label_start_time=label_start,
                label_end_time=label_end,
            )

            with lock:
                results.append({
                    "thread_id": thread_id,
                    "batch_shape": batch.values.shape,
                    "label_shape": labels.values.shape,
                })

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "error": str(e)})

    # Launch concurrent evaluations
    threads = []
    num_threads = 15

    for i in range(num_threads):
        t = threading.Thread(target=create_and_evaluate, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all succeeded
    assert len(errors) == 0, f"Evaluation errors: {errors}"
    assert len(results) == num_threads


def test_concurrent_preprocessing_operations():
    """
    Test concurrent preprocessing policy applications.
    """
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    results = []
    errors = []
    lock = threading.Lock()

    def create_policy(thread_id):
        try:
            policy = PreprocessingPolicy(
                policy_id=f"policy_thread_{thread_id}",
                transforms=[
                    TransformSpec(
                        name="winsorize",
                        kind=TransformKind.CROSS_SECTIONAL,
                        mode=TransformMode.STATELESS,
                        version="1.0",
                        parameters={"lower": 0.01 * thread_id, "upper": 0.99}
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

            # Validate policy
            assert len(policy.transforms) == 2
            assert policy.policy_id == f"policy_thread_{thread_id}"

            with lock:
                results.append({"thread_id": thread_id, "policy_id": policy.policy_id})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "error": str(e)})

    # Launch concurrent policy creations
    threads = []
    num_threads = 10

    for i in range(num_threads):
        t = threading.Thread(target=create_policy, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all succeeded
    assert len(errors) == 0, f"Preprocessing errors: {errors}"
    assert len(results) == num_threads


def test_concurrent_read_write_contention():
    """
    Test read/write contention in AssetRepository.

    Multiple readers and writers accessing the same repository.
    """
    from factor_assets import (
        AssetRepository,
        AssetMetadata,
        LineageRef,
        create_factor_id,
    )

    repo = AssetRepository()

    # Pre-populate with some factors
    pre_populated = []
    for i in range(5):
        factor_id = create_factor_id(
            name=f"prepop_factor_{i}",
            version="v1",
            params={"index": i},
        )

        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"prepop_factor_{i}",
            canonical_hash=f"hash_prepop_{i}",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Pre-populated factor {i}",
        )

        lineage = LineageRef(
            factor_id=factor_id,
            parents=(),
        )

        asset = repo.register(metadata, lineage)
        pre_populated.append(factor_id)

    read_results = []
    write_results = []
    errors = []
    lock = threading.Lock()

    def reader_thread(thread_id):
        try:
            # Read existing factors
            for factor_id in pre_populated:
                asset = repo.get(factor_id)
                assert asset is not None
                time.sleep(0.001)  # Simulate processing

            with lock:
                read_results.append({"thread_id": thread_id, "reads": len(pre_populated)})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "type": "reader", "error": str(e)})

    def writer_thread(thread_id):
        try:
            # Write new factors
            factor_id = create_factor_id(
                name=f"new_factor_{thread_id}",
                version="v1",
                params={"thread_id": thread_id},
            )

            metadata = AssetMetadata(
                factor_id=factor_id,
                canonical_repr=f"new_factor_{thread_id}",
                canonical_hash=f"hash_new_{thread_id}",
                frequency="daily",
                domains=("equity",),
                timing="daily",
                description=f"New factor from thread {thread_id}",
            )

            lineage = LineageRef(
                factor_id=factor_id,
                parents=(),
            )

            asset = repo.register(metadata, lineage)
            time.sleep(0.001)  # Simulate processing

            with lock:
                write_results.append({"thread_id": thread_id, "factor_id": factor_id})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "type": "writer", "error": str(e)})

    # Launch mixed readers and writers
    threads = []
    num_readers = 10
    num_writers = 5

    for i in range(num_readers):
        t = threading.Thread(target=reader_thread, args=(i,))
        threads.append(t)

    for i in range(num_writers):
        t = threading.Thread(target=writer_thread, args=(i,))
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Read/write contention errors: {errors}"
    assert len(read_results) == num_readers
    assert len(write_results) == num_writers


def test_deadlock_prevention():
    """
    Test deadlock prevention in cross-package operations.

    Simulate scenarios that could cause deadlocks and verify they don't occur.
    """
    from factor_assets import AssetRepository, AssetMetadata, LineageRef, create_factor_id
    from quant_evaluator import FactorBatch, AxisRef

    repo = AssetRepository()
    results = []
    errors = []
    lock = threading.Lock()

    def operation_a(thread_id):
        """Register factor then create batch."""
        try:
            # Step 1: Register factor
            factor_id = create_factor_id(
                name=f"deadlock_test_a_{thread_id}",
                version="v1",
                params={"thread": thread_id},
            )

            metadata = AssetMetadata(
                factor_id=factor_id,
                canonical_repr=f"deadlock_test_a_{thread_id}",
                canonical_hash=f"hash_a_{thread_id}",
                frequency="daily",
                domains=("equity",),
                timing="daily",
                description="Deadlock test A",
            )

            lineage = LineageRef(factor_id=factor_id, parents=())
            asset = repo.register(metadata, lineage)

            time.sleep(0.01)  # Simulate processing

            # Step 2: Create batch
            T, N = 10, 5
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)
            batch = FactorBatch(
                factor_ids=(factor_id,),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=np.random.randn(T, N, 1),
            )

            with lock:
                results.append({"thread_id": thread_id, "operation": "A", "success": True})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "operation": "A", "error": str(e)})

    def operation_b(thread_id):
        """Create batch then register factor."""
        try:
            # Step 1: Create batch
            factor_id = create_factor_id(
                name=f"deadlock_test_b_{thread_id}",
                version="v1",
                params={"thread": thread_id},
            )

            T, N = 10, 5
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)
            batch = FactorBatch(
                factor_ids=(factor_id,),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=np.random.randn(T, N, 1),
            )

            time.sleep(0.01)  # Simulate processing

            # Step 2: Register factor
            metadata = AssetMetadata(
                factor_id=factor_id,
                canonical_repr=f"deadlock_test_b_{thread_id}",
                canonical_hash=f"hash_b_{thread_id}",
                frequency="daily",
                domains=("equity",),
                timing="daily",
                description="Deadlock test B",
            )

            lineage = LineageRef(factor_id=factor_id, parents=())
            asset = repo.register(metadata, lineage)

            with lock:
                results.append({"thread_id": thread_id, "operation": "B", "success": True})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "operation": "B", "error": str(e)})

    # Launch interleaved operations
    threads = []
    num_pairs = 10

    for i in range(num_pairs):
        t_a = threading.Thread(target=operation_a, args=(i,))
        t_b = threading.Thread(target=operation_b, args=(i,))
        threads.append(t_a)
        threads.append(t_b)

    # Start all threads
    for t in threads:
        t.start()

    # Wait with timeout to detect deadlocks
    start_time = time.time()
    timeout = 10.0

    for t in threads:
        remaining = timeout - (time.time() - start_time)
        if remaining <= 0:
            pytest.fail("Deadlock detected: threads did not complete within timeout")
        t.join(timeout=remaining)
        if t.is_alive():
            pytest.fail(f"Deadlock detected: thread {t.name} still alive after timeout")

    # Verify no errors
    assert len(errors) == 0, f"Operation errors: {errors}"
    assert len(results) == num_pairs * 2


def test_race_condition_prevention():
    """
    Test race condition prevention in shared state updates.
    """
    from factor_assets import SeenIndex

    seen_index = SeenIndex()
    results = []
    errors = []
    lock = threading.Lock()

    def record_seen(thread_id):
        """Multiple threads recording seen factors."""
        try:
            for i in range(10):
                factor_id = f"factor_{thread_id}_{i}"
                canonical_hash = f"hash_{thread_id}_{i}"

                # Record as seen
                seen_index.record(
                    canonical_hash=canonical_hash,
                    factor_id=factor_id,
                    origin="thread",
                )

                # Verify immediately
                assert seen_index.is_seen(canonical_hash)

            with lock:
                results.append({"thread_id": thread_id, "recorded": 10})

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "error": str(e)})

    # Launch concurrent recordings
    threads = []
    num_threads = 20

    for i in range(num_threads):
        t = threading.Thread(target=record_seen, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Race condition errors: {errors}"
    assert len(results) == num_threads

    # Verify correct total count
    expected_count = num_threads * 10
    assert seen_index.count() == expected_count


def test_concurrent_pipeline_stress():
    """
    Stress test: Complete pipeline with many concurrent users.
    """
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle, MetricValue
    from factor_assets import AssetRepository, AssetMetadata, LineageRef, create_factor_id
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    repo = AssetRepository()
    results = queue.Queue()
    errors = queue.Queue()

    def complete_workflow(user_id):
        """Simulate complete research workflow for one user."""
        try:
            # Step 1: Register factor
            factor_id = create_factor_id(
                name=f"stress_factor_{user_id}",
                version="v1",
                params={"user": user_id},
            )

            metadata = AssetMetadata(
                factor_id=factor_id,
                canonical_repr=f"stress_factor_{user_id}",
                canonical_hash=f"hash_stress_{user_id}",
                frequency="daily",
                domains=("equity",),
                timing="daily",
                description=f"Stress test factor {user_id}",
            )

            lineage = LineageRef(factor_id=factor_id, parents=())
            asset = repo.register(metadata, lineage)

            # Step 2: Create preprocessing policy
            policy = PreprocessingPolicy(
                policy_id=f"policy_stress_{user_id}",
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

            # Step 3: Create and evaluate batch
            T, N = 30, 10
            factor_values = np.random.randn(T, N, 1) * 0.05
            time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=(factor_id,),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Step 4: Create labels and evaluate
            label_values = np.random.randn(T, N) * 0.02
            dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(T)]
            decision_times = tuple(d.strftime("%Y-%m-%d") for d in dates)
            label_start = tuple((d + timedelta(days=1)).strftime("%Y-%m-%d") for d in dates)
            label_end = tuple((d + timedelta(days=3)).strftime("%Y-%m-%d") for d in dates)

            labels = LabelBundle(
                target_id=f"label_stress_{user_id}",
                values=label_values,
                horizon=2,
                decision_time=decision_times,
                label_start_time=label_start,
                label_end_time=label_end,
            )

            # Simulate metric
            simulated_ic = np.random.randn() * 0.02 + 0.03

            metric = MetricValue(
                metric_id="rank_ic",
                value=simulated_ic,
                valid=True,
                observation_count=T * N,
                metric_version="0.1",
            )

            results.put({
                "user_id": user_id,
                "factor_id": factor_id,
                "metric_value": simulated_ic,
                "success": True,
            })

        except Exception as e:
            errors.put({"user_id": user_id, "error": str(e)})

    # Launch many concurrent workflows
    threads = []
    num_users = 50

    start_time = time.time()

    for i in range(num_users):
        t = threading.Thread(target=complete_workflow, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # Collect results
    result_list = []
    while not results.empty():
        result_list.append(results.get())

    error_list = []
    while not errors.empty():
        error_list.append(errors.get())

    # Verify all succeeded
    assert len(error_list) == 0, f"Stress test errors: {error_list}"
    assert len(result_list) == num_users

    print(f"\nStress test completed:")
    print(f"  Users: {num_users}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {num_users / elapsed:.1f} workflows/sec")


def test_resource_cleanup_under_load():
    """
    Test resource cleanup when multiple threads create and destroy objects.
    """
    import gc

    from quant_evaluator import FactorBatch, AxisRef

    created_count = 0
    cleaned_count = 0
    errors = []
    lock = threading.Lock()

    def create_and_cleanup(thread_id):
        """Create batches and let them be garbage collected."""
        try:
            for i in range(10):
                T, N = 20, 10
                factor_values = np.random.randn(T, N, 1)
                time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
                asset_axis = AxisRef(name="asset", dtype="int64", size=N)

                batch = FactorBatch(
                    factor_ids=(f"cleanup_factor_{thread_id}_{i}",),
                    time_axis=time_axis,
                    asset_axis=asset_axis,
                    values=factor_values,
                )

                with lock:
                    nonlocal created_count
                    created_count += 1

                # Use batch briefly
                _ = batch.num_factors

                # Let it go out of scope
                del batch
                del factor_values

            # Force cleanup
            gc.collect()

        except Exception as e:
            with lock:
                errors.append({"thread_id": thread_id, "error": str(e)})

    # Launch concurrent create/cleanup
    threads = []
    num_threads = 10

    for i in range(num_threads):
        t = threading.Thread(target=create_and_cleanup, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Final cleanup
    gc.collect()

    # Verify no errors
    assert len(errors) == 0, f"Cleanup errors: {errors}"
    assert created_count == num_threads * 10
