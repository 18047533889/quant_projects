"""
Parallel batch evaluation using multiprocessing.

Evaluates multiple FactorBatch instances in parallel across CPU cores,
targeting 4-8x throughput improvement on multi-core systems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import multiprocessing as mp
from multiprocessing import Pool
import os
import time
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import Evaluator, EvaluationResult
from quant_evaluator.runtime.budgets import ComputationBudget
from quant_evaluator.planner.dependency_plan import MetricKind


@dataclass
class ParallelConfig:
    """
    Configuration for parallel batch evaluation.

    Controls worker processes, memory limits, and execution behavior.
    """
    num_workers: Optional[int] = None
    max_chunk_memory_mb: float = 512.0
    cache_size_per_worker_mb: float = 256.0
    enable_cache: bool = True
    use_chunking: bool = True
    timeout_seconds: Optional[float] = None
    shared_cache_size_mb: float = 0.0  # Disabled by default (not picklable)

    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")


@dataclass
class BatchEvaluationTask:
    """
    Single batch evaluation task for worker process.

    Contains all inputs needed to evaluate one FactorBatch.
    """
    task_id: int
    factor_batch: FactorBatch
    label_bundle: LabelBundle
    metric_specs: List[Dict]
    config: ParallelConfig
    budget: Optional[ComputationBudget] = None


@dataclass
class BatchEvaluationTaskResult:
    """
    Result from evaluating a single batch task.

    Wraps EvaluationResult with task tracking metadata.
    """
    task_id: int
    result: Optional[EvaluationResult] = None
    error: Optional[str] = None
    worker_id: Optional[int] = None
    execution_time_seconds: float = 0.0


@dataclass
class ParallelEvaluationResult:
    """
    Aggregated results from parallel batch evaluation.

    Contains individual batch results plus parallel execution statistics.
    """
    batch_results: List[BatchEvaluationTaskResult] = field(default_factory=list)
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_execution_time_seconds: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    throughput_batches_per_second: float = 0.0
    speedup_factor: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, task_id: int) -> Optional[BatchEvaluationTaskResult]:
        """Retrieve result for specific task."""
        for result in self.batch_results:
            if result.task_id == task_id:
                return result
        return None

    def get_successful_results(self) -> List[BatchEvaluationTaskResult]:
        """Get all successful task results."""
        return [r for r in self.batch_results if r.result is not None]

    def get_failed_results(self) -> List[BatchEvaluationTaskResult]:
        """Get all failed task results."""
        return [r for r in self.batch_results if r.error is not None]


# Global registry for metric functions (needed for pickling)
_GLOBAL_METRIC_REGISTRY: Dict[str, Callable] = {}
_GLOBAL_METRIC_KINDS: Dict[str, MetricKind] = {}


def register_metric_for_parallel(
    metric_id: str,
    metric_fn: Callable,
    metric_kind: MetricKind = MetricKind.CUSTOM,
):
    """
    Register metric function for parallel execution.

    Metrics must be registered globally before parallel execution
    because they need to be pickled for worker processes.

    Args:
        metric_id: Unique metric identifier
        metric_fn: Metric computation function
        metric_kind: Type of metric
    """
    _GLOBAL_METRIC_REGISTRY[metric_id] = metric_fn
    _GLOBAL_METRIC_KINDS[metric_id] = metric_kind


def _worker_init():
    """Initialize worker process."""
    # Set process-specific random seed for reproducibility
    worker_id = os.getpid()
    np.random.seed((worker_id + int(time.time())) % (2**32))


def _evaluate_batch_task(task: BatchEvaluationTask) -> BatchEvaluationTaskResult:
    """
    Worker function to evaluate a single batch task.

    Args:
        task: Batch evaluation task

    Returns:
        Task result with evaluation outcome
    """
    start_time = time.time()
    worker_id = os.getpid()

    try:
        # Create worker-local evaluator
        evaluator = Evaluator(
            enable_cache=task.config.enable_cache,
            cache_size_mb=task.config.cache_size_per_worker_mb,
            budget=task.budget,
            max_chunk_memory_mb=task.config.max_chunk_memory_mb,
        )

        # Register metrics from global registry
        for metric_id, metric_fn in _GLOBAL_METRIC_REGISTRY.items():
            metric_kind = _GLOBAL_METRIC_KINDS.get(metric_id, MetricKind.CUSTOM)
            evaluator.register_metric(metric_id, metric_fn, metric_kind)

        # Evaluate batch
        result = evaluator.evaluate(
            factor_batch=task.factor_batch,
            label_bundle=task.label_bundle,
            metric_specs=task.metric_specs,
            use_chunking=task.config.use_chunking,
        )

        elapsed = time.time() - start_time

        return BatchEvaluationTaskResult(
            task_id=task.task_id,
            result=result,
            error=None,
            worker_id=worker_id,
            execution_time_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return BatchEvaluationTaskResult(
            task_id=task.task_id,
            result=None,
            error=str(e),
            worker_id=worker_id,
            execution_time_seconds=elapsed,
        )


class ParallelBatchExecutor:
    """
    Parallel executor for multiple FactorBatch evaluations.

    Uses multiprocessing.Pool to distribute batch evaluations across
    CPU cores, targeting 4-8x throughput improvement on multi-core systems.

    Example:
        executor = ParallelBatchExecutor(num_workers=8)

        # Register metrics
        executor.register_metric("ic_mean", compute_ic_mean)

        # Create tasks
        tasks = []
        for i, batch in enumerate(factor_batches):
            tasks.append((batch, label_bundle, metric_specs))

        # Execute in parallel
        result = executor.execute_parallel(tasks)

        print(f"Processed {result.successful_batches} batches")
        print(f"Speedup: {result.speedup_factor:.1f}x")
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel executor.

        Args:
            config: Parallel execution configuration
        """
        self.config = config or ParallelConfig()

        # Clear global registry on init
        _GLOBAL_METRIC_REGISTRY.clear()
        _GLOBAL_METRIC_KINDS.clear()

    def register_metric(
        self,
        metric_id: str,
        metric_fn: Callable,
        metric_kind: MetricKind = MetricKind.CUSTOM,
    ):
        """
        Register metric for parallel execution.

        Args:
            metric_id: Unique metric identifier
            metric_fn: Metric computation function
            metric_kind: Type of metric
        """
        register_metric_for_parallel(metric_id, metric_fn, metric_kind)

    def execute_parallel(
        self,
        batches: List[Tuple[FactorBatch, LabelBundle, List[Dict]]],
        budget: Optional[ComputationBudget] = None,
    ) -> ParallelEvaluationResult:
        """
        Execute multiple batch evaluations in parallel.

        Args:
            batches: List of (factor_batch, label_bundle, metric_specs) tuples
            budget: Optional computation budget per batch

        Returns:
            Aggregated parallel evaluation results
        """
        if not batches:
            return ParallelEvaluationResult(
                total_batches=0,
                successful_batches=0,
                failed_batches=0,
            )

        start_time = time.time()

        # Create tasks
        tasks = []
        for i, (factor_batch, label_bundle, metric_specs) in enumerate(batches):
            task = BatchEvaluationTask(
                task_id=i,
                factor_batch=factor_batch,
                label_bundle=label_bundle,
                metric_specs=metric_specs,
                config=self.config,
                budget=budget,
            )
            tasks.append(task)

        # Execute in parallel
        with Pool(
            processes=self.config.num_workers,
            initializer=_worker_init,
        ) as pool:
            if self.config.timeout_seconds:
                task_results = pool.map_async(
                    _evaluate_batch_task,
                    tasks,
                ).get(timeout=self.config.timeout_seconds)
            else:
                task_results = pool.map(_evaluate_batch_task, tasks)

        total_time = time.time() - start_time

        # Aggregate results
        successful = sum(1 for r in task_results if r.result is not None)
        failed = sum(1 for r in task_results if r.error is not None)

        total_cache_hits = sum(
            r.result.cache_hits for r in task_results if r.result is not None
        )
        total_cache_misses = sum(
            r.result.cache_misses for r in task_results if r.result is not None
        )

        throughput = len(batches) / total_time if total_time > 0 else 0.0

        # Estimate speedup (comparing to sequential execution)
        # Use actual worker execution times to estimate sequential time
        actual_execution_times = [r.execution_time_seconds for r in task_results]
        estimated_sequential_total = sum(actual_execution_times)
        speedup = estimated_sequential_total / total_time if total_time > 0 else 1.0

        result = ParallelEvaluationResult(
            batch_results=task_results,
            total_batches=len(batches),
            successful_batches=successful,
            failed_batches=failed,
            total_execution_time_seconds=total_time,
            total_cache_hits=total_cache_hits,
            total_cache_misses=total_cache_misses,
            throughput_batches_per_second=throughput,
            speedup_factor=speedup,
            metadata={
                "num_workers": self.config.num_workers,
                "estimated_sequential_time": estimated_sequential_total,
                "parallel_efficiency": speedup / self.config.num_workers if self.config.num_workers > 0 else 0.0,
            },
        )

        return result

    def execute_single(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        metric_specs: List[Dict],
        budget: Optional[ComputationBudget] = None,
    ) -> EvaluationResult:
        """
        Execute single batch evaluation (convenience method).

        Args:
            factor_batch: Input factor batch
            label_bundle: Input label bundle
            metric_specs: Metric specifications
            budget: Optional computation budget

        Returns:
            Evaluation result
        """
        batches = [(factor_batch, label_bundle, metric_specs)]
        parallel_result = self.execute_parallel(batches, budget=budget)

        if parallel_result.successful_batches == 0:
            first_result = parallel_result.batch_results[0]
            raise RuntimeError(f"Evaluation failed: {first_result.error}")

        return parallel_result.batch_results[0].result

    def _estimate_sequential_time(self, sample_task: BatchEvaluationTask) -> float:
        """
        Estimate sequential execution time using a sample task.

        DEPRECATED: Now using actual execution times from workers.

        Args:
            sample_task: Sample task to benchmark

        Returns:
            Estimated execution time in seconds
        """
        # This method is deprecated but kept for compatibility
        try:
            start = time.time()
            _evaluate_batch_task(sample_task)
            return time.time() - start
        except Exception:
            # Fallback: assume 1 second per task
            return 1.0


def evaluate_batches_parallel(
    batches: List[Tuple[FactorBatch, LabelBundle, List[Dict]]],
    metric_functions: Dict[str, Callable],
    num_workers: Optional[int] = None,
    config: Optional[ParallelConfig] = None,
    budget: Optional[ComputationBudget] = None,
) -> ParallelEvaluationResult:
    """
    Convenience function for parallel batch evaluation.

    Args:
        batches: List of (factor_batch, label_bundle, metric_specs) tuples
        metric_functions: Dictionary mapping metric_id -> computation function
        num_workers: Number of worker processes (None = auto)
        config: Optional parallel configuration
        budget: Optional computation budget per batch

    Returns:
        Parallel evaluation results
    """
    if config is None:
        config = ParallelConfig(num_workers=num_workers)
    elif num_workers is not None:
        config.num_workers = num_workers

    executor = ParallelBatchExecutor(config=config)

    # Register all metrics
    for metric_id, metric_fn in metric_functions.items():
        executor.register_metric(metric_id, metric_fn)

    return executor.execute_parallel(batches, budget=budget)
