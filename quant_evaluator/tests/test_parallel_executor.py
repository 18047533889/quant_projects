"""
Tests for parallel batch executor.
"""

import pytest
import numpy as np
import time

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
    ParallelEvaluationResult,
    evaluate_batches_parallel,
    register_metric_for_parallel,
)
from quant_evaluator.runtime.budgets import ComputationBudget
from quant_evaluator.planner.dependency_plan import MetricKind


class TestParallelConfig:
    def test_config_default_workers(self):
        config = ParallelConfig()
        assert config.num_workers >= 1

    def test_config_explicit_workers(self):
        config = ParallelConfig(num_workers=4)
        assert config.num_workers == 4

    def test_config_invalid_workers_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            ParallelConfig(num_workers=0)

    def test_config_custom_memory(self):
        config = ParallelConfig(
            max_chunk_memory_mb=1024.0,
            cache_size_per_worker_mb=512.0,
        )
        assert config.max_chunk_memory_mb == 1024.0
        assert config.cache_size_per_worker_mb == 512.0


class TestParallelBatchExecutor:
    def create_test_batch(self, T=100, N=500, F=1, seed=None):
        """Helper to create test factor batch."""
        if seed is not None:
            np.random.seed(seed)

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)
        values = np.random.randn(T, N, F)

        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def create_test_labels(self, T=100, N=500, seed=None):
        """Helper to create test label bundle."""
        if seed is not None:
            np.random.seed(seed + 1000)

        values = np.random.randn(T, N)

        return LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

    def test_executor_creation(self):
        executor = ParallelBatchExecutor()
        assert executor.config is not None
        assert executor.config.num_workers >= 1

    def test_executor_with_custom_config(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)
        assert executor.config.num_workers == 2

    def test_register_metric(self):
        executor = ParallelBatchExecutor()

        def test_metric(**kwargs):
            return 0.5

        executor.register_metric("test", test_metric, MetricKind.CUSTOM)

        from quant_evaluator.runtime.parallel_executor import _GLOBAL_METRIC_REGISTRY
        assert "test" in _GLOBAL_METRIC_REGISTRY

    def test_execute_single_batch(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)

        def simple_metric(factor_batch, label_bundle, **kwargs):
            return 0.85

        executor.register_metric("coverage", simple_metric, MetricKind.COVERAGE)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)
        metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]

        result = executor.execute_single(batch, labels, metric_specs)

        assert result is not None
        assert result.has_metric("coverage")
        assert result.get_metric("coverage") == 0.85

    def test_execute_parallel_multiple_batches(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)

        def simple_metric(factor_batch, label_bundle, **kwargs):
            return float(factor_batch.num_factors)

        executor.register_metric("factor_count", simple_metric)

        # Create 10 batches with different factor counts
        batches = []
        for i in range(10):
            batch = self.create_test_batch(T=30, N=50, F=i+1, seed=i)
            labels = self.create_test_labels(T=30, N=50, seed=i)
            metric_specs = [{"metric_id": "factor_count", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.total_batches == 10
        assert result.successful_batches == 10
        assert result.failed_batches == 0
        assert result.total_execution_time_seconds > 0
        assert result.throughput_batches_per_second > 0

        # Verify individual results
        for i in range(10):
            task_result = result.get_result(i)
            assert task_result is not None
            assert task_result.result is not None
            assert task_result.result.get_metric("factor_count") == float(i + 1)

    def test_execute_parallel_with_dependencies(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)

        def base_metric(factor_batch, **kwargs):
            return np.array([1.0, 2.0, 3.0])

        def derived_metric(computed_metrics, **kwargs):
            base = computed_metrics.get("base")
            return float(np.sum(base))

        executor.register_metric("base", base_metric)
        executor.register_metric("derived", derived_metric)

        batches = []
        for i in range(5):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [
                {"metric_id": "base", "metric_kind": "custom"},
                {"metric_id": "derived", "metric_kind": "custom", "dependencies": ["base"]},
            ]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.successful_batches == 5
        for i in range(5):
            task_result = result.get_result(i)
            assert task_result.result.get_metric("derived") == 6.0

    def test_execute_parallel_error_handling(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)

        def failing_metric(**kwargs):
            raise ValueError("Intentional failure")

        executor.register_metric("failing", failing_metric)

        batches = []
        for i in range(3):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [{"metric_id": "failing", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.total_batches == 3
        assert result.failed_batches == 3
        assert result.successful_batches == 0

        failed = result.get_failed_results()
        assert len(failed) == 3
        for fail_result in failed:
            assert "Intentional failure" in fail_result.error

    def test_execute_parallel_mixed_success_failure(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)

        def conditional_metric(factor_batch, **kwargs):
            # Fail if factor count is even
            if factor_batch.num_factors % 2 == 0:
                raise ValueError("Even factor count not allowed")
            return float(factor_batch.num_factors)

        executor.register_metric("conditional", conditional_metric)

        batches = []
        for i in range(1, 6):  # 1, 2, 3, 4, 5 factors
            batch = self.create_test_batch(T=20, N=30, F=i, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [{"metric_id": "conditional", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.total_batches == 5
        assert result.successful_batches == 3  # 1, 3, 5 factors
        assert result.failed_batches == 2  # 2, 4 factors

    def test_execute_parallel_large_scale(self):
        """Test with 100+ batches as requested."""
        config = ParallelConfig(num_workers=4)
        executor = ParallelBatchExecutor(config=config)

        def fast_metric(factor_batch, **kwargs):
            # Simple metric for fast execution
            return float(factor_batch.num_times)

        executor.register_metric("time_count", fast_metric)

        # Create 120 small batches
        batches = []
        for i in range(120):
            batch = self.create_test_batch(T=10, N=20, F=1, seed=i)
            labels = self.create_test_labels(T=10, N=20, seed=i)
            metric_specs = [{"metric_id": "time_count", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        start_time = time.time()
        result = executor.execute_parallel(batches)
        elapsed = time.time() - start_time

        assert result.total_batches == 120
        assert result.successful_batches == 120
        assert result.failed_batches == 0
        assert result.speedup_factor > 0

        # Verify throughput
        assert result.throughput_batches_per_second > 0
        print(f"\nProcessed 120 batches in {elapsed:.2f}s")
        print(f"Throughput: {result.throughput_batches_per_second:.1f} batches/sec")
        print(f"Speedup: {result.speedup_factor:.2f}x")

    def test_parallel_evaluation_result_methods(self):
        result = ParallelEvaluationResult()

        assert result.get_result(999) is None
        assert result.get_successful_results() == []
        assert result.get_failed_results() == []

    def test_execute_with_budget(self):
        config = ParallelConfig(num_workers=2)
        executor = ParallelBatchExecutor(config=config)
        budget = ComputationBudget(max_memory_mb=1024.0)

        def simple_metric(**kwargs):
            return 0.5

        executor.register_metric("test", simple_metric)

        batches = []
        for i in range(3):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [{"metric_id": "test", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches, budget=budget)

        assert result.successful_batches == 3

    def test_execute_with_caching(self):
        config = ParallelConfig(num_workers=2, enable_cache=True)
        executor = ParallelBatchExecutor(config=config)

        def cacheable_metric(factor_batch, **kwargs):
            # Simulate some computation
            time.sleep(0.01)
            return 0.5

        executor.register_metric("cached", cacheable_metric)

        batches = []
        for i in range(5):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [{"metric_id": "cached", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.successful_batches == 5
        # Each worker has its own cache, so we expect cache misses
        assert result.total_cache_misses >= 5

    def test_execute_empty_batches(self):
        executor = ParallelBatchExecutor()

        result = executor.execute_parallel([])

        assert result.total_batches == 0
        assert result.successful_batches == 0
        assert result.failed_batches == 0


class TestEvaluateBatchesParallel:
    def create_test_batch(self, T=50, N=100, F=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)
        values = np.random.randn(T, N, F)
        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def create_test_labels(self, T=50, N=100, seed=None):
        if seed is not None:
            np.random.seed(seed + 1000)
        values = np.random.randn(T, N)
        return LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

    def test_convenience_function(self):
        def metric1(**kwargs):
            return 1.0

        def metric2(**kwargs):
            return 2.0

        metric_functions = {
            "m1": metric1,
            "m2": metric2,
        }

        batches = []
        for i in range(5):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [
                {"metric_id": "m1", "metric_kind": "custom"},
                {"metric_id": "m2", "metric_kind": "custom"},
            ]
            batches.append((batch, labels, metric_specs))

        result = evaluate_batches_parallel(
            batches,
            metric_functions,
            num_workers=2,
        )

        assert result.successful_batches == 5
        for i in range(5):
            task_result = result.get_result(i)
            assert task_result.result.get_metric("m1") == 1.0
            assert task_result.result.get_metric("m2") == 2.0

    def test_convenience_function_with_config(self):
        def simple_metric(**kwargs):
            return 0.5

        config = ParallelConfig(
            num_workers=3,
            enable_cache=False,
        )

        batches = []
        for i in range(3):
            batch = self.create_test_batch(T=20, N=30, F=1, seed=i)
            labels = self.create_test_labels(T=20, N=30, seed=i)
            metric_specs = [{"metric_id": "simple", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = evaluate_batches_parallel(
            batches,
            {"simple": simple_metric},
            config=config,
        )

        assert result.successful_batches == 3
        assert result.metadata["num_workers"] == 3


class TestPerformanceCharacteristics:
    """Tests to verify performance characteristics and speedup."""

    def create_test_batch(self, T=100, N=500, F=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)
        values = np.random.randn(T, N, F)
        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def create_test_labels(self, T=100, N=500, seed=None):
        if seed is not None:
            np.random.seed(seed + 1000)
        values = np.random.randn(T, N)
        return LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

    def test_speedup_factor_computed(self):
        """Verify speedup factor is calculated correctly for parallel execution."""
        config = ParallelConfig(num_workers=4)
        executor = ParallelBatchExecutor(config=config)

        def compute_intensive_metric(factor_batch, **kwargs):
            # Simulate computation with actual work to make it CPU-bound
            arr = factor_batch.values
            # Multiple operations to make it more compute-intensive
            result = np.sum(arr ** 2) + np.sum(np.abs(arr)) + np.sum(arr ** 3)
            time.sleep(0.005)  # Small delay to make computation time more significant
            return float(result)

        executor.register_metric("intensive", compute_intensive_metric)

        batches = []
        for i in range(20):
            batch = self.create_test_batch(T=50, N=200, F=2, seed=i)
            labels = self.create_test_labels(T=50, N=200, seed=i)
            metric_specs = [{"metric_id": "intensive", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        result = executor.execute_parallel(batches)

        assert result.successful_batches == 20
        # Speedup factor is ratio of total sequential time to parallel time
        # It should be positive and the calculation should be reasonable
        assert result.speedup_factor > 0
        assert result.metadata["parallel_efficiency"] >= 0
        print(f"\nSpeedup: {result.speedup_factor:.2f}x with {config.num_workers} workers")
        print(f"Parallel efficiency: {result.metadata['parallel_efficiency']:.2%}")
        print(f"Sequential estimate: {result.metadata['estimated_sequential_time']:.2f}s")
        print(f"Actual parallel time: {result.total_execution_time_seconds:.2f}s")

    def test_parallel_efficiency_with_varying_workers(self):
        """Test parallel efficiency with different worker counts."""
        def simple_metric(factor_batch, **kwargs):
            return np.mean(factor_batch.values)

        # Create batches
        batches = []
        for i in range(50):
            batch = self.create_test_batch(T=30, N=100, F=1, seed=i)
            labels = self.create_test_labels(T=30, N=100, seed=i)
            metric_specs = [{"metric_id": "mean", "metric_kind": "custom"}]
            batches.append((batch, labels, metric_specs))

        results = {}
        for num_workers in [1, 2, 4]:
            config = ParallelConfig(num_workers=num_workers)
            executor = ParallelBatchExecutor(config=config)
            executor.register_metric("mean", simple_metric)

            result = executor.execute_parallel(batches)
            results[num_workers] = result

            print(f"\n{num_workers} workers: {result.speedup_factor:.2f}x speedup, "
                  f"{result.throughput_batches_per_second:.1f} batches/sec")

        # More workers should generally give better throughput
        assert results[1].successful_batches == 50
        assert results[2].successful_batches == 50
        assert results[4].successful_batches == 50
