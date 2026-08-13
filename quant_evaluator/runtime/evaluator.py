"""
Main Evaluator class for batch metric evaluation.

Orchestrates metric computation with planning, caching, and budget tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import time

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError, InsufficientObservations

from quant_evaluator.planner.batch_plan import BatchPlan, ChunkDescriptor, create_batch_plan
from quant_evaluator.planner.dependency_plan import (
    MetricDependencyGraph,
    MetricNode,
    MetricKind,
    resolve_metric_dependencies,
)
from quant_evaluator.runtime.intermediates import (
    IntermediateCache,
    CacheKey,
    compute_input_hash,
)
from quant_evaluator.runtime.budgets import (
    ComputationBudget,
    BudgetTracker,
    ResourceUsage,
)


@dataclass
class EvaluationResult:
    """
    Result of metric evaluation.

    Contains computed metrics, diagnostics, and execution metadata.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    chunks_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    resource_usage: Optional[ResourceUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric_id: str, default=None) -> Any:
        """Retrieve a metric by ID."""
        return self.metrics.get(metric_id, default)

    def has_metric(self, metric_id: str) -> bool:
        """Check if metric exists in results."""
        return metric_id in self.metrics

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        result = {
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "execution_time_seconds": self.execution_time_seconds,
            "chunks_processed": self.chunks_processed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
        if self.resource_usage:
            result["resource_usage"] = self.resource_usage.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class Evaluator:
    """
    Main evaluator for batch metric computation.

    Coordinates planning, execution, caching, and budget tracking for
    efficient large-scale factor evaluation.
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size_mb: float = 1024.0,
        budget: Optional[ComputationBudget] = None,
        max_chunk_memory_mb: float = 512.0,
    ):
        """
        Initialize evaluator.

        Args:
            enable_cache: Whether to enable intermediate caching
            cache_size_mb: Maximum cache size in MB
            budget: Computation budget (None = no limits)
            max_chunk_memory_mb: Maximum memory per chunk
        """
        self.cache = IntermediateCache(max_size_mb=cache_size_mb, enable=enable_cache)
        self.budget = budget or ComputationBudget()
        self.budget_tracker = BudgetTracker(self.budget)
        self.max_chunk_memory_mb = max_chunk_memory_mb

        # Registry of metric functions
        self._metric_functions: Dict[str, Callable] = {}

        # Execution stats
        self._cache_hits = 0
        self._cache_misses = 0

    def register_metric(
        self,
        metric_id: str,
        metric_fn: Callable,
        metric_kind: MetricKind = MetricKind.CUSTOM,
    ):
        """
        Register a metric computation function.

        Args:
            metric_id: Unique metric identifier
            metric_fn: Function that computes the metric
            metric_kind: Type of metric
        """
        self._metric_functions[metric_id] = metric_fn

    def evaluate(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        metric_specs: List[Dict],
        use_chunking: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate metrics on factor batch.

        Args:
            factor_batch: Input factor batch
            label_bundle: Input label bundle
            metric_specs: List of metric specifications
            use_chunking: Whether to use batch chunking

        Returns:
            EvaluationResult with computed metrics

        Raises:
            InvalidContractError: If inputs are invalid
            RuntimeError: If budget exceeded
        """
        start_time = time.time()
        self.budget_tracker.reset()
        self._cache_hits = 0
        self._cache_misses = 0

        # Validate inputs
        self._validate_inputs(factor_batch, label_bundle)

        # Build dependency graph
        dep_graph = resolve_metric_dependencies(metric_specs)

        # Create batch plan
        batch_plan = None
        if use_chunking:
            batch_plan = create_batch_plan(
                factor_batch,
                max_chunk_memory_mb=self.max_chunk_memory_mb,
            )

        # Execute metrics in dependency order
        result = EvaluationResult()

        if batch_plan and batch_plan.num_chunks > 1:
            # Chunked execution
            result = self._evaluate_chunked(
                factor_batch,
                label_bundle,
                dep_graph,
                batch_plan,
            )
        else:
            # Full batch execution
            result = self._evaluate_full(
                factor_batch,
                label_bundle,
                dep_graph,
            )

        # Finalize result
        elapsed = time.time() - start_time
        result.execution_time_seconds = elapsed
        result.cache_hits = self._cache_hits
        result.cache_misses = self._cache_misses
        result.resource_usage = self.budget_tracker.get_current_usage()

        result.metadata["batch_shape"] = (
            factor_batch.num_times,
            factor_batch.num_assets,
            factor_batch.num_factors,
        )
        result.metadata["num_metrics"] = len(dep_graph.nodes)

        return result

    def _evaluate_full(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        dep_graph: MetricDependencyGraph,
    ) -> EvaluationResult:
        """Evaluate all metrics on full batch."""
        result = EvaluationResult()
        result.chunks_processed = 1

        # Get execution order
        execution_order = dep_graph.topological_sort()

        # Execute each metric
        for metric_id in execution_order:
            self.budget_tracker.check_budget(raise_on_exceed=True)

            node = dep_graph.get_node(metric_id)
            if node is None:
                continue

            # Check cache
            cache_key = CacheKey(
                metric_id=metric_id,
                input_hash=compute_input_hash(factor_batch.factor_ids),
            )

            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                result.metrics[metric_id] = cached_value
                self._cache_hits += 1
                continue

            self._cache_misses += 1

            # Compute metric
            metric_value = self._compute_metric(
                metric_id,
                node,
                factor_batch,
                label_bundle,
                result.metrics,
            )

            # Store in result and cache
            result.metrics[metric_id] = metric_value
            self.cache.put(cache_key, metric_value)

            self.budget_tracker.record_operation()

        return result

    def _evaluate_chunked(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        dep_graph: MetricDependencyGraph,
        batch_plan: BatchPlan,
    ) -> EvaluationResult:
        """Evaluate metrics using batch chunking."""
        result = EvaluationResult()
        result.chunks_processed = batch_plan.num_chunks

        # Get execution order
        execution_order = dep_graph.topological_sort()

        # Process each chunk
        chunk_results = []
        for chunk in batch_plan.get_ordered_chunks():
            self.budget_tracker.check_budget(raise_on_exceed=True)

            chunk_result = self._evaluate_chunk(
                factor_batch,
                label_bundle,
                dep_graph,
                chunk,
                execution_order,
            )
            chunk_results.append(chunk_result)

        # Aggregate chunk results
        result.metrics = self._aggregate_chunk_results(
            chunk_results,
            execution_order,
            dep_graph,
        )

        return result

    def _evaluate_chunk(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        dep_graph: MetricDependencyGraph,
        chunk: ChunkDescriptor,
        execution_order: List[str],
    ) -> Dict[str, Any]:
        """Evaluate metrics on a single chunk."""
        chunk_metrics = {}

        # Extract chunk data
        chunk_batch = self._extract_chunk_batch(factor_batch, chunk)
        chunk_labels = self._extract_chunk_labels(label_bundle, chunk)

        # Execute metrics on chunk
        for metric_id in execution_order:
            node = dep_graph.get_node(metric_id)
            if node is None:
                continue

            # Check cache
            cache_key = CacheKey(
                metric_id=metric_id,
                chunk_id=chunk.chunk_id,
                input_hash=compute_input_hash(
                    chunk_batch.factor_ids,
                    time_slice=chunk.time_slice,
                    asset_slice=chunk.asset_slice,
                ),
            )

            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                chunk_metrics[metric_id] = cached_value
                self._cache_hits += 1
                continue

            self._cache_misses += 1

            # Compute metric on chunk
            metric_value = self._compute_metric(
                metric_id,
                node,
                chunk_batch,
                chunk_labels,
                chunk_metrics,
            )

            chunk_metrics[metric_id] = metric_value
            self.cache.put(cache_key, metric_value)

            self.budget_tracker.record_operation()

        return chunk_metrics

    def _compute_metric(
        self,
        metric_id: str,
        node: MetricNode,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        computed_metrics: Dict[str, Any],
    ) -> Any:
        """
        Compute a single metric.

        Args:
            metric_id: Metric identifier
            node: Metric node with metadata
            factor_batch: Input factor batch
            label_bundle: Input label bundle
            computed_metrics: Already computed metrics (dependencies)

        Returns:
            Computed metric value
        """
        if metric_id not in self._metric_functions:
            raise InvalidContractError(f"Metric function not registered: {metric_id}")

        metric_fn = self._metric_functions[metric_id]

        # Prepare arguments
        args = {
            "factor_batch": factor_batch,
            "label_bundle": label_bundle,
            "computed_metrics": computed_metrics,
            "metadata": node.metadata,
        }

        # Call metric function
        try:
            return metric_fn(**args)
        except Exception as e:
            raise RuntimeError(f"Error computing metric {metric_id}: {e}") from e

    def _extract_chunk_batch(
        self,
        factor_batch: FactorBatch,
        chunk: ChunkDescriptor,
    ) -> FactorBatch:
        """Extract factor batch for a chunk."""
        t_start, t_end = chunk.time_slice
        a_start, a_end = chunk.asset_slice
        f_start, f_end = chunk.factor_slice

        chunk_values = factor_batch.values[t_start:t_end, a_start:a_end, f_start:f_end]
        chunk_validity = None
        if factor_batch.validity is not None:
            chunk_validity = factor_batch.validity[t_start:t_end, a_start:a_end, f_start:f_end]

        chunk_factor_ids = factor_batch.factor_ids[f_start:f_end]

        # Create new axes for chunk
        from quant_evaluator.contracts.factor_batch import AxisRef

        chunk_time_axis = AxisRef(
            name=factor_batch.time_axis.name,
            dtype=factor_batch.time_axis.dtype,
            size=t_end - t_start,
        )
        chunk_asset_axis = AxisRef(
            name=factor_batch.asset_axis.name,
            dtype=factor_batch.asset_axis.dtype,
            size=a_end - a_start,
        )

        return FactorBatch(
            factor_ids=chunk_factor_ids,
            time_axis=chunk_time_axis,
            asset_axis=chunk_asset_axis,
            values=chunk_values,
            validity=chunk_validity,
            layout=factor_batch.layout,
            dtype=factor_batch.dtype,
        )

    def _extract_chunk_labels(
        self,
        label_bundle: LabelBundle,
        chunk: ChunkDescriptor,
    ) -> LabelBundle:
        """Extract label bundle for a chunk."""
        t_start, t_end = chunk.time_slice
        a_start, a_end = chunk.asset_slice

        chunk_values = label_bundle.values[t_start:t_end, a_start:a_end] if label_bundle.values.ndim == 2 else label_bundle.values[t_start:t_end]
        chunk_validity = None
        if label_bundle.validity is not None:
            chunk_validity = label_bundle.validity[t_start:t_end, a_start:a_end] if label_bundle.validity.ndim == 2 else label_bundle.validity[t_start:t_end]

        return LabelBundle(
            target_id=label_bundle.target_id,
            values=chunk_values,
            horizon=label_bundle.horizon,
            execution_delay=label_bundle.execution_delay,
            decision_time=label_bundle.decision_time[t_start:t_end],
            execution_time=label_bundle.execution_time[t_start:t_end] if label_bundle.execution_time else tuple(),
            label_start_time=label_bundle.label_start_time[t_start:t_end],
            label_end_time=label_bundle.label_end_time[t_start:t_end],
            validity=chunk_validity,
        )

    def _aggregate_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        execution_order: List[str],
        dep_graph: MetricDependencyGraph,
    ) -> Dict[str, Any]:
        """Aggregate results from multiple chunks."""
        aggregated = {}

        for metric_id in execution_order:
            node = dep_graph.get_node(metric_id)
            if node is None:
                continue

            # Collect metric values from all chunks
            chunk_values = [cr[metric_id] for cr in chunk_results if metric_id in cr]

            if not chunk_values:
                continue

            # Aggregate based on metric kind
            if node.metric_kind == MetricKind.IC:
                # Concatenate IC series
                aggregated[metric_id] = np.concatenate(chunk_values, axis=0)
            elif node.metric_kind == MetricKind.COVERAGE:
                # Average coverage
                aggregated[metric_id] = np.mean(chunk_values)
            elif node.metric_kind == MetricKind.SUMMARY:
                # Take first value (summaries are typically scalar)
                aggregated[metric_id] = chunk_values[0]
            else:
                # Default: concatenate arrays
                if isinstance(chunk_values[0], np.ndarray):
                    aggregated[metric_id] = np.concatenate(chunk_values, axis=0)
                else:
                    aggregated[metric_id] = chunk_values

        return aggregated

    def _validate_inputs(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
    ):
        """Validate input contracts."""
        if factor_batch.num_times != len(label_bundle.values):
            raise InvalidContractError(
                f"Factor time axis ({factor_batch.num_times}) "
                f"does not match label length ({len(label_bundle.values)})"
            )

    def clear_cache(self):
        """Clear intermediate cache."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def get_budget_report(self) -> str:
        """Get formatted budget usage report."""
        return self.budget_tracker.format_usage_report()
