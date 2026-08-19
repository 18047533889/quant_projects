"""
Main Evaluator class for batch metric evaluation.

Orchestrates metric computation with planning, caching, and budget tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import hashlib
import inspect
import json
import types
from types import MappingProxyType
import numpy as np
import time

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import (
    InvalidContractError,
    QuantEvaluatorError,
    UnsupportedMetricError,
)
from quant_evaluator.api.requests import EvaluationRequest

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
)
from quant_evaluator.runtime.budgets import (
    ComputationBudget,
    BudgetTracker,
    ResourceUsage,
)
from quant_evaluator.registry.metrics import get_metric


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
    provenance: Dict[str, Any] = field(default_factory=dict)

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
        if self.provenance:
            result["provenance"] = self.provenance
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

        # Registry metadata is enforced for catalog metrics. Locally registered
        # callables retain their existing contract and signature behavior.
        for metric_id in dep_graph.nodes:
            self._validate_metric_requirements(metric_id, factor_batch, label_bundle)

        # Chunking is opt-in: unsafe metrics default to full-batch execution.
        batch_plan = None
        if use_chunking and self._chunking_is_safe(dep_graph):
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
        result.provenance = self._provenance(factor_batch, label_bundle)
        result.metadata["provenance"] = result.provenance

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
            cache_identity = self._cache_identity(metric_id, node, factor_batch, label_bundle)
            cache_key = CacheKey(
                metric_id=metric_id,
                input_hash=cache_identity,
            )

            cached_value = self.cache.get(cache_key) if cache_identity else None
            if cached_value is not None:
                result.metrics[metric_id] = cached_value
                self._cache_hits += 1
                continue

            self._cache_misses += 1

            self._validate_metric_requirements(
                metric_id, factor_batch, label_bundle, result.metrics
            )

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
            if cache_identity:
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
        for chunk in sorted(
            batch_plan.chunks,
            key=lambda item: (item.time_slice[0], item.asset_slice[0], item.factor_slice[0]),
        ):
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
            cache_identity = self._cache_identity(
                metric_id, node, chunk_batch, chunk_labels, chunk=chunk
            )
            cache_key = CacheKey(
                metric_id=metric_id,
                chunk_id=chunk.chunk_id,
                input_hash=cache_identity,
            )

            cached_value = self.cache.get(cache_key) if cache_identity else None
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
            if cache_identity:
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
        metric_fn = self._resolve_metric_function(metric_id)

        # Prepare arguments
        args = {
            "factor_batch": factor_batch,
            "label_bundle": label_bundle,
            "computed_metrics": computed_metrics,
            "metadata": node.metadata,
        }
        args.update(
            (dependency, computed_metrics[dependency])
            for dependency in node.dependencies
            if dependency in computed_metrics and dependency not in args
        )

        try:
            signature = inspect.signature(metric_fn)
        except (TypeError, ValueError):
            call_args = args
        else:
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_kwargs:
                call_args = args
            else:
                call_args = {
                    name: value
                    for name, value in args.items()
                    if name in signature.parameters
                    and signature.parameters[name].kind
                    in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                }
                unresolved = [
                    name
                    for name, parameter in signature.parameters.items()
                    if parameter.kind
                    in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    and parameter.default is inspect.Parameter.empty
                    and name not in call_args
                ]
                if unresolved:
                    raise InvalidContractError(
                        f"Metric {metric_id} has unresolved required parameters: "
                        f"{', '.join(unresolved)}"
                    )

        # Call metric function
        try:
            return metric_fn(**call_args)
        except QuantEvaluatorError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error computing metric {metric_id}: {e}") from e

    def _validate_metric_requirements(
        self,
        metric_id: str,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        computed_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate requirements declared by registry-owned metrics."""
        if metric_id in self._metric_functions:
            return
        try:
            spec = get_metric(metric_id)
        except KeyError:
            return

        available = {
            "factor_batch": factor_batch,
            "label_bundle": label_bundle,
            "computed_metrics": computed_metrics,
            "metadata": {},
        }
        missing = [name for name in (spec.requires or []) if name not in available or available[name] is None]
        if missing:
            raise InvalidContractError(
                f"Metric {metric_id} missing required input(s): {', '.join(missing)}"
            )
        if spec.min_periods is not None and factor_batch.num_times < spec.min_periods:
            raise InvalidContractError(
                f"Metric {metric_id} requires at least {spec.min_periods} periods; "
                f"got {factor_batch.num_times}"
            )

    def _resolve_metric_function(self, metric_id: str) -> Callable:
        metric_fn = self._metric_functions.get(metric_id)
        if metric_fn is not None:
            return metric_fn
        try:
            metric_fn = get_metric(metric_id).compute_fn
        except KeyError as exc:
            raise InvalidContractError(
                f"Metric function not registered: {metric_id}"
            ) from exc
        if metric_fn is None:
            raise InvalidContractError(f"Metric function not registered: {metric_id}")
        return metric_fn

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
            values=(factor_batch.time_axis.values[t_start:t_end].copy()
                    if factor_batch.time_axis.values is not None else None),
        )
        chunk_asset_axis = AxisRef(
            name=factor_batch.asset_axis.name,
            dtype=factor_batch.asset_axis.dtype,
            size=a_end - a_start,
            values=(factor_batch.asset_axis.values[a_start:a_end].copy()
                    if factor_batch.asset_axis.values is not None else None),
        )

        return FactorBatch(
            factor_ids=chunk_factor_ids,
            time_axis=chunk_time_axis,
            asset_axis=chunk_asset_axis,
            values=chunk_values,
            validity=chunk_validity,
            layout=factor_batch.layout,
            dtype=factor_batch.dtype,
            context_refs=dict(factor_batch.context_refs),
            value_hash=self._array_hash(chunk_values),
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
            source_ref=label_bundle.source_ref,
            calendar_ref=label_bundle.calendar_ref,
            metadata=dict(label_bundle.metadata),
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

            aggregation = node.metadata.get("aggregation")
            if aggregation == "mean":
                raise InvalidContractError(
                    f"Metric {metric_id} requires an explicit weighted reduction protocol"
                )
            if not self._metric_is_partitionable(node) or aggregation not in {
                "concat", "first", "sum"
            }:
                raise InvalidContractError(
                    f"Metric {metric_id} has no safe chunk aggregation contract"
                )

            if aggregation == "concat":
                aggregated[metric_id] = np.concatenate(chunk_values, axis=0)
            elif aggregation == "sum":
                aggregated[metric_id] = np.sum(chunk_values)
            else:
                aggregated[metric_id] = chunk_values[0]

        return aggregated

    @staticmethod
    def _metric_is_partitionable(node: MetricNode) -> bool:
        return node.metadata.get("partitionability", "NOT_PARTITIONABLE") == "PARTITIONABLE"

    def _chunking_is_safe(self, dep_graph: MetricDependencyGraph) -> bool:
        return all(
            self._metric_is_partitionable(node)
            and node.metadata.get("aggregation") in {"concat", "first", "sum"}
            and not node.requires_full_batch
            for node in dep_graph.nodes.values()
        )

    @staticmethod
    def _array_hash(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _freeze(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "shape": value.shape,
                "sha256": Evaluator._array_hash(value),
            }
        if isinstance(value, dict):
            return {str(k): Evaluator._freeze(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [Evaluator._freeze(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _is_immutably_cacheable(value: Any) -> bool:
        if isinstance(value, (str, int, float, bool, bytes, type(None))):
            return True
        if isinstance(value, tuple):
            return all(Evaluator._is_immutably_cacheable(item) for item in value)
        if isinstance(value, frozenset):
            return all(Evaluator._is_immutably_cacheable(item) for item in value)
        return False

    def _cache_identity(self, metric_id, node, factor_batch, label_bundle, chunk=None):
        try:
            metric_fn = self._resolve_metric_function(metric_id)
            if not isinstance(metric_fn, types.FunctionType):
                # Callable instances can hide mutable state behind descriptors or
                # custom attribute access; their semantic identity is unprovable.
                return None
            code = metric_fn.__code__
            closure_state = []
            for cell in metric_fn.__closure__ or ():
                value = cell.cell_contents
                if self._is_immutably_cacheable(value):
                    closure_state.append(self._freeze(value))
                else:
                    # Mutable state, including mutable values nested inside an
                    # immutable container, may change between evaluations.
                    return None
            function_hash = hashlib.sha256(
                json.dumps(
                    [code.co_code.hex(), self._freeze(code.co_consts),
                     self._freeze(metric_fn.__defaults__),
                     self._freeze(metric_fn.__kwdefaults__), closure_state],
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest()
            identity = {
                "metric": metric_id,
                "kind": node.metric_kind.value,
                "metadata": self._freeze(node.metadata),
                "function": [metric_fn.__module__, metric_fn.__qualname__, function_hash],
                "factor_ids": list(factor_batch.factor_ids),
                "factor_values": self._freeze(factor_batch.values),
                "factor_validity": self._freeze(factor_batch.validity),
                "time_axis": self._freeze(factor_batch.time_axis.values),
                "asset_axis": self._freeze(factor_batch.asset_axis.values),
                "context_refs": self._freeze(factor_batch.context_refs),
                "factor_value_hash": factor_batch.value_hash,
                "labels": {
                    "target_id": label_bundle.target_id,
                    "values": self._freeze(label_bundle.values),
                    "validity": self._freeze(label_bundle.validity),
                    "horizon": label_bundle.horizon,
                    "execution_delay": label_bundle.execution_delay,
                    "decision_time": self._freeze(label_bundle.decision_time),
                    "execution_time": self._freeze(label_bundle.execution_time),
                    "label_start_time": self._freeze(label_bundle.label_start_time),
                    "label_end_time": self._freeze(label_bundle.label_end_time),
                    "source_ref": label_bundle.source_ref,
                    "calendar_ref": label_bundle.calendar_ref,
                    "metadata": self._freeze(label_bundle.metadata),
                },
                "chunk": self._freeze(
                    (chunk.time_slice, chunk.asset_slice, chunk.factor_slice)
                ) if chunk else None,
            }
            payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(payload.encode()).hexdigest()
        except Exception:
            return None

    def _provenance(self, factor_batch, label_bundle):
        return MappingProxyType({
            "factor_ids": tuple(factor_batch.factor_ids),
            "time_coordinates": self._freeze(factor_batch.time_axis.values),
            "asset_coordinates": self._freeze(factor_batch.asset_axis.values),
            "context_refs": MappingProxyType(dict(factor_batch.context_refs)),
            "factor_value_hash": factor_batch.value_hash or self._array_hash(factor_batch.values),
            "label_source_ref": label_bundle.source_ref,
            "label_calendar_ref": label_bundle.calendar_ref,
            "label_target_id": label_bundle.target_id,
            "label_value_hash": self._array_hash(label_bundle.values),
            "label_metadata": MappingProxyType(dict(label_bundle.metadata)),
        })

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


def evaluate(
    factors,
    labels=None,
    *,
    context=None,
    metrics=None,
    where=None,
    evaluator=None,
):
    """Evaluate explicit factor and label contracts through the runtime.

    This is the public batch facade.  It delegates computation to ``Evaluator``
    and adapts only scalar values produced by existing metric kernels into the
    canonical :class:`EvaluationBundle` contract.  Unsupported requests fail
    closed rather than fabricating metric metadata.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from quant_evaluator.api.requests import EvaluationBundle, MetricValue
    from quant_evaluator.diagnosis.factor import diagnose_all_factors
    from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
    from quant_evaluator.metrics.label_panel import normalize_label_panel

    request_metadata = {}
    request_fields = {}
    if isinstance(factors, EvaluationRequest):
        request = factors
        factor_batch = request.batch_or_factor_ids
        label_bundle = request.label_bundle
        metric_ids = request.metric_ids
        context = request.context
        if request.slices is not None:
            raise UnsupportedMetricError("EvaluationRequest.slices are not supported by public evaluate")
        request_metadata = dict(request.metadata)
        request_fields = {"tier": request.tier, "cost_budget": request.cost_budget}
    else:
        factor_batch = factors
        label_bundle = labels
        metric_ids = tuple(metrics or ("coverage",))

    if where is not None:
        raise UnsupportedMetricError("where slicing is not supported by public evaluate")
    if not isinstance(factor_batch, FactorBatch) or not isinstance(label_bundle, LabelBundle):
        raise TypeError("evaluate requires a FactorBatch and LabelBundle")
    if label_bundle.values.ndim == 2 and label_bundle.values.shape[1] != factor_batch.num_assets:
        raise InvalidContractError(
            f"Label asset axis ({label_bundle.values.shape[1]}) does not match "
            f"factor asset axis ({factor_batch.num_assets})"
        )
    if (
        label_bundle.validity is not None
        and label_bundle.validity.ndim == 2
        and label_bundle.validity.shape[1] != factor_batch.num_assets
    ):
        raise InvalidContractError(
            f"Label validity asset axis ({label_bundle.validity.shape[1]}) does not match "
            f"factor asset axis ({factor_batch.num_assets})"
        )

    metric_ids = tuple(metric_ids)
    runtime = evaluator or Evaluator()
    metric_specs = []

    def coverage_metric(factor_batch, label_bundle, **kwargs):
        labels_array, label_validity = normalize_label_panel(
            label_bundle, factor_batch.num_assets
        )
        valid = np.isfinite(factor_batch.values) & np.isfinite(labels_array)[:, :, np.newaxis]
        if factor_batch.validity is not None:
            valid &= factor_batch.validity
        if label_validity is not None:
            valid &= label_validity[:, :, np.newaxis]
        counts = np.sum(valid, axis=(0, 1))
        totals = np.full(factor_batch.num_factors, factor_batch.num_times * factor_batch.num_assets)
        return {
            factor_id: (float(count / total) if total else np.nan, int(count))
            for factor_id, count, total in zip(factor_batch.factor_ids, counts, totals)
        }

    def ic_metric(method, min_periods=1):
        def compute(factor_batch, label_bundle, **kwargs):
            series, counts = compute_daily_ic(factor_batch, label_bundle, method=method)
            mean, _ = compute_mean_ic(series, counts, min_periods=min_periods)
            return {
                factor_id: (float(mean[index]), int(np.sum(counts[:, index][np.isfinite(series[:, index])])))
                for index, factor_id in enumerate(factor_batch.factor_ids)
            }
        return compute

    supported = {
        "coverage": coverage_metric,
        "mean_ic": ic_metric("pearson", min_periods=20),
        "pearson_ic": ic_metric("pearson"),
        "rank_ic": ic_metric("spearman"),
    }
    for metric_id in metric_ids:
        if metric_id not in supported:
            raise UnsupportedMetricError(
                f"Public evaluate does not support metric '{metric_id}'"
            )
        runtime.register_metric(metric_id, supported[metric_id])
        metric_specs.append({"metric_id": metric_id, "metric_kind": "custom"})

    result = runtime.evaluate(factor_batch, label_bundle, metric_specs, use_chunking=False)
    grouped_metrics = {factor_id: {} for factor_id in factor_batch.factor_ids}
    for metric_id in metric_ids:
        values = result.get_metric(metric_id)
        if not isinstance(values, dict) or set(values) != set(factor_batch.factor_ids):
            raise UnsupportedMetricError(
                f"Metric '{metric_id}' did not return per-factor scalar values"
            )
        for factor_id, payload in values.items():
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise UnsupportedMetricError(
                    f"Metric '{metric_id}' returned unsupported per-factor metadata"
                )
            value, observation_count = payload
            if not isinstance(value, (bool, int, float, np.number)):
                raise UnsupportedMetricError(
                    f"Metric '{metric_id}' returned a non-scalar value for '{factor_id}'"
                )
            numeric = float(value)
            grouped_metrics[factor_id][metric_id] = MetricValue(
                metric_id=metric_id,
                value=None if not np.isfinite(numeric) else numeric,
                valid=bool(np.isfinite(numeric)),
                observation_count=int(observation_count),
                warnings=() if np.isfinite(numeric) else ("non-finite result",),
            )

    metric_values = (
        dict(grouped_metrics[factor_batch.factor_ids[0]])
        if factor_batch.num_factors == 1
        else {}
    )
    bundle_metadata = dict(request_metadata)
    bundle_metadata.update(request_fields)
    bundle_metadata.update({"context": context, "where": None, "runtime": result.metadata})
    request_id = str(bundle_metadata.pop("request_id", uuid4()))

    return EvaluationBundle(
        request_id=request_id,
        factor_ids=tuple(factor_batch.factor_ids),
        label_id=label_bundle.target_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metric_values=metric_values,
        diagnostics=diagnose_all_factors(factor_batch),
        grouped_metrics=grouped_metrics,
        metric_versions={metric_id: "0.1" for metric_id in metric_ids},
        metadata=bundle_metadata,
        warnings=tuple(result.metadata.get("warnings", ())),
    )
