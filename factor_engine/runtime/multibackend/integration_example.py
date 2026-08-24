# -*- coding: utf-8 -*-
"""Integration example for multibackend memory optimizations.

Demonstrates how the 8 memory optimization modules work together
to provide comprehensive memory management for adaptive_batch_scheduler.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


def example_integrated_memory_management(dag: Any, total_memory_bytes: int = 4 * 1024**3):
    """Example integration of all memory optimization modules.

    Args:
        dag: PhysicalFactorDAG to execute
        total_memory_bytes: Total available memory (default 4 GB)

    Returns:
        Execution results with memory statistics
    """
    from factor_engine.runtime.multibackend import (
        # Shape estimation
        DataShapeEstimator,
        # Memory modeling
        FineGrainedMemoryModel,
        global_memory_model,
        # Batch optimization
        BatchGlobalOptimizer,
        # CSE with representation tracking
        global_cse_manager,
        # Liveness analysis
        LivenessAnalyzer,
        # Streaming execution
        StreamingExecutionPlanner,
        # Budget management
        global_budget_manager,
        # Region isolation
        global_region_manager,
        # Spill strategy
        global_spill_manager,
        # Zero-copy transfers
        global_zerocopy_boundary,
    )

    _logger.info("Initializing integrated memory management...")

    # Step 1: Estimate data shapes for all tasks
    shape_estimator = DataShapeEstimator()
    size_estimates = {}
    for task_id, task in dag.tasks.items():
        shape = shape_estimator.estimate_from_plan(task.node_ref)
        size_estimates[task_id] = shape.total_bytes
        _logger.debug(f"Task {task_id}: estimated {shape.total_bytes // 1024 // 1024} MB")

    # Step 2: Analyze DAG for liveness and predict peak memory
    liveness_analyzer = LivenessAnalyzer()
    liveness_analyzer.analyze_dag(dag, size_estimates=size_estimates)
    peak_prediction = liveness_analyzer.predict_peak_memory(dag, size_estimates=size_estimates)
    _logger.info(
        f"Predicted peak memory: {peak_prediction['peak_bytes'] // 1024 // 1024} MB, "
        f"spill candidates: {len(peak_prediction['spill_candidates'])}"
    )

    # Step 3: Perform batch-global optimizations
    optimizer = BatchGlobalOptimizer(
        min_savings_bytes=10 * 1024 * 1024,  # 10 MB threshold
    )
    opt_result = optimizer.optimize_batch(dag, enable_cse=True)
    _logger.info(
        f"Global optimizations: {opt_result.applied_count} applied, "
        f"savings {opt_result.total_savings_bytes // 1024 // 1024} MB"
    )

    # Step 4: Plan streaming execution
    streaming_planner = StreamingExecutionPlanner()
    streaming_plan = streaming_planner.plan_streaming_execution(dag, enable_fusion=True)
    _logger.info(
        f"Streaming plan: {len(streaming_plan.streaming_regions)} regions, "
        f"{len(streaming_plan.materialization_points)} materialization points"
    )

    # Step 5: Initialize budget manager
    budget_mgr = global_budget_manager(total_budget_bytes=total_memory_bytes)
    _logger.info(f"Budget manager initialized with {total_memory_bytes // 1024 // 1024} MB")

    # Step 6: Create isolated regions for concurrent execution
    region_mgr = global_region_manager(total_memory_bytes=total_memory_bytes)
    for i, region in enumerate(streaming_plan.streaming_regions):
        pool = region_mgr.create_region(f"region_{i}")
        _logger.debug(f"Created region {i} with {pool._limit_bytes // 1024 // 1024} MB limit")

    # Step 7: Initialize spill manager
    spill_mgr = global_spill_manager()
    _logger.info("Spill manager initialized")

    # Step 8: Setup zero-copy boundary
    zerocopy = global_zerocopy_boundary()
    _logger.info(f"Zero-copy boundary initialized (Arrow available: {zerocopy._has_arrow})")

    # Step 9: Create liveness tracker for runtime
    liveness_tracker = liveness_analyzer.create_tracker()

    # Simulation: Execute tasks with integrated memory management
    results = {}
    for task_id in streaming_plan.execution_order[:10]:  # Simulate first 10 tasks
        # Request memory token
        token = budget_mgr.request_token(
            task_id,
            size_estimates.get(task_id, 100 * 1024 * 1024),
            timeout_s=30.0,
        )

        if token is None:
            # Budget exhausted, trigger spilling
            current_live = liveness_tracker.current_live_bytes()
            if spill_mgr.should_spill(current_live):
                candidates = liveness_tracker.get_spill_candidates(min_size_bytes=50 * 1024 * 1024)
                target_bytes = int(current_live * 0.3)  # Free 30%
                to_spill = spill_mgr.select_spill_candidates(candidates, target_bytes)

                for spill_task_id, spill_size in to_spill:
                    _logger.info(f"Spilling {spill_task_id}: {spill_size // 1024 // 1024} MB")
                    # Simulate spill (would get actual data in real execution)
                    liveness_tracker.mark_spilled(spill_task_id)

                # Retry token request
                token = budget_mgr.request_token(task_id, size_estimates.get(task_id, 100 * 1024 * 1024))

        if token:
            # Mark produced
            liveness_tracker.mark_produced(task_id, token.bytes_allocated)

            # Simulate task execution
            result = f"result_{task_id}"
            results[task_id] = result

            # Mark consumed by dependents
            # (In real execution, would track actual dependencies)

            # Return token
            budget_mgr.return_token(token)

    # Collect final statistics
    stats = {
        "shape_estimation": {"tasks_estimated": len(size_estimates)},
        "liveness": liveness_tracker.stats(),
        "optimization": {
            "applied_count": opt_result.applied_count,
            "savings_bytes": opt_result.total_savings_bytes,
        },
        "streaming": streaming_plan.metadata,
        "budget": budget_mgr.stats(),
        "regions": region_mgr.global_stats(),
        "spill": spill_mgr.stats(),
        "zerocopy": zerocopy.stats(),
    }

    _logger.info("Integrated memory management completed successfully")
    return {"results": results, "stats": stats}


def example_memory_model_usage():
    """Example usage of fine-grained memory model."""
    from factor_engine.runtime.multibackend import global_memory_model

    memory_model = global_memory_model()

    # Estimate operator memory for different backends
    for backend in ["pandas", "polars", "arrow", "duckdb"]:
        estimate = memory_model.estimate_operator_memory(
            operator="ts_corr",
            input_bytes=100 * 1024 * 1024,  # 100 MB input
            representation=backend,
            rows=500_000,
            cols=10,
        )
        print(f"{backend:8s}: peak={estimate['peak_bytes'] // 1024 // 1024}MB, "
              f"streaming={estimate['supports_streaming']}")


def example_cse_usage():
    """Example usage of native CSE with representation tracking."""
    from factor_engine.runtime.multibackend import global_cse_manager, NativeCSEKey

    cse_mgr = global_cse_manager()

    # Define computation
    def expensive_computation():
        import time
        time.sleep(0.1)  # Simulate work
        return {"data": [1, 2, 3]}

    # Request with specific representation
    key = NativeCSEKey(
        structural_hash="hash_abc123",
        representation="polars",
    )

    # First call: compute
    result1 = cse_mgr.get_or_compute(key, expensive_computation, size_estimate_bytes=1024)

    # Second call: cache hit
    result2 = cse_mgr.get_or_compute(key, expensive_computation, size_estimate_bytes=1024)

    # Request with different representation: convert
    key_pandas = key.with_representation("pandas")
    result3 = cse_mgr.get_or_compute(
        key_pandas, expensive_computation, accept_conversion=True
    )

    stats = cse_mgr.stats()
    print(f"CSE stats: {stats['hit_count']} hits, "
          f"conversion avoided: {stats['conversion_avoided_rate']:.1%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Memory Model Example:")
    example_memory_model_usage()
    print("\nCSE Example:")
    example_cse_usage()
