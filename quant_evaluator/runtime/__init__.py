"""
Runtime module for quant_evaluator.

Provides main Evaluator class, intermediate result caching, budget tracking,
parallel batch execution, and streaming evaluation for large datasets.
"""

from quant_evaluator.runtime.evaluator import (
    Evaluator,
    EvaluationResult,
)
from quant_evaluator.runtime.intermediates import (
    IntermediateCache,
    CacheKey,
    CacheEntry,
)
from quant_evaluator.runtime.budgets import (
    ComputationBudget,
    BudgetTracker,
    ResourceUsage,
)
from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
    ParallelEvaluationResult,
    BatchEvaluationTask,
    BatchEvaluationTaskResult,
    evaluate_batches_parallel,
    register_metric_for_parallel,
)
from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    StreamingEvaluationResult,
    StreamingMetricState,
    streaming_ic_updater,
    streaming_coverage_updater,
    streaming_summary_updater,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "IntermediateCache",
    "CacheKey",
    "CacheEntry",
    "ComputationBudget",
    "BudgetTracker",
    "ResourceUsage",
    "ParallelBatchExecutor",
    "ParallelConfig",
    "ParallelEvaluationResult",
    "BatchEvaluationTask",
    "BatchEvaluationTaskResult",
    "evaluate_batches_parallel",
    "register_metric_for_parallel",
    "StreamingEvaluator",
    "StreamingEvaluationResult",
    "StreamingMetricState",
    "streaming_ic_updater",
    "streaming_coverage_updater",
    "streaming_summary_updater",
]
