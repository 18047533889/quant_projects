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

import numpy as np
import time

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import (
    InvalidContractError,
    QuantEvaluatorError,
    UnsupportedMetricError,
)
from quant_evaluator.contracts._hashutil import canonicalize
from quant_evaluator.api.requests import EvaluationRequest

from quant_evaluator.planner.batch_plan import BatchPlan, ChunkDescriptor, create_batch_plan
from quant_evaluator.planner.dependency_plan import (
    MetricDependencyGraph,
    MetricNode,
    MetricKind,
    resolve_metric_dependencies,
)
from quant_evaluator.runtime.intermediates import (
    CacheKey,
)
from quant_evaluator.runtime.cache_v2_adapter import V2IntermediateCache
from quant_evaluator.runtime.budgets import (
    ComputationBudget,
    BudgetTracker,
    ResourceUsage,
)
from quant_evaluator.registry.metrics import (
    CANONICAL_METRIC_ALIASES,
    get_metric,
)


def _resolve_alias(metric_id: str) -> str:
    """Resolve a canonical dotted metric name (e.g. "ic.pearson.mean") to
    its registry name ("mean_ic"). Unknown names pass through unchanged."""
    return CANONICAL_METRIC_ALIASES.get(metric_id, metric_id)


# Resource-node costs are deliberately attached to the real public builders,
# not to artifact class names in an "available" set.  Costs are expressed in
# the same abstract node units accepted by EvaluationRequest.cost_budget.
_ARTIFACT_BUILDERS = {
    "ICSeriesArtifact": {"cost": 1.0, "tier": "core"},
    "QuantileReturnArtifact": {"cost": 2.0, "tier": "extended"},
    "HACPValueVector": {"cost": 1.0, "tier": "research"},
}
_RUNTIME_INPUTS = frozenset({
    "factor_batch", "label_bundle", "computed_metrics", "metadata",
    "factor_values", "forward_returns",
})


def authoritative_array_hash(value: np.ndarray) -> str:
    """Hash ndarray values, never object-buffer pointer addresses.

    Numeric arrays retain the historical dtype+shape+raw-bytes identity.
    Object arrays use the lossless element codec; structured arrays continue
    to fail closed until an explicit schema codec exists.
    """
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    digest = hashlib.sha256()
    digest.update(str(original.dtype).encode())
    digest.update(str(original.shape).encode())
    if array.dtype.hasobject:
        from quant_evaluator.contracts._ndarray_codec import encode_ndarray

        encoded = encode_ndarray(original)
        digest.update(json.dumps(
            encoded["object_values"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
    elif array.dtype.fields is not None:
        raise TypeError("structured arrays require an explicit schema codec")
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()


def _compile_public_artifact_plan(
    resolved_specs, *, portfolio_returns=None, holding_returns=None,
    exposure_panel=None,
):
    """Return unique, actually buildable derived-artifact nodes.

    Unknown artifact requirements fail here, before backend staging.  This is
    intentionally narrower than claiming every formal artifact type can be
    synthesized from factor and label panels.
    """
    required_by = {}
    for metric_id, spec in resolved_specs.items():
        for requirement in spec.requires or ():
            if requirement in _RUNTIME_INPUTS:
                continue
            if requirement == "probe_pnl":
                if portfolio_returns is None and holding_returns is None:
                    raise InvalidContractError(
                        f"Metric {metric_id} requires missing probe_pnl input"
                    )
                if holding_returns is not None:
                    required_by.setdefault("ProbePortfolioArtifact", []).append(metric_id)
                continue
            if requirement == "exposure_panel":
                if exposure_panel is None:
                    raise InvalidContractError(
                        f"Metric {metric_id} requires missing exposure_panel input"
                    )
                required_by.setdefault("ExposureArtifact", []).append(metric_id)
                continue
            if requirement == "p_values":
                # Public p-values are derived from the same daily rank-IC
                # samples used by QE significance metrics, never injected as
                # an unbound caller array.
                required_by.setdefault("ICSeriesArtifact", []).append(metric_id)
                required_by.setdefault("HACPValueVector", []).append(metric_id)
                continue
            if requirement not in _ARTIFACT_BUILDERS:
                raise InvalidContractError(
                    f"Metric {metric_id} requires unavailable artifact builder "
                    f"{requirement}"
                )
            required_by.setdefault(requirement, []).append(metric_id)
    return tuple({
        "artifact": artifact,
        "required_by": tuple(consumers),
        "cost": (_ARTIFACT_BUILDERS[artifact]["cost"]
                 if artifact in _ARTIFACT_BUILDERS else
                 (2.0 if artifact == "ExposureArtifact" else 3.0)),
        "tier": (_ARTIFACT_BUILDERS[artifact]["tier"]
                 if artifact in _ARTIFACT_BUILDERS else "extended"),
    } for artifact, consumers in required_by.items())




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
            # Plain-dict copy: mappingproxy provenance must not leak into
            # to_dict()/pickling paths (cannot pickle 'mappingproxy').
            result["provenance"] = dict(self.provenance)
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
        cache: Optional[V2IntermediateCache] = None,
    ):
        """
        Initialize evaluator.

        Args:
            enable_cache: Whether to enable intermediate caching
            cache_size_mb: Maximum cache size in MB
            budget: Computation budget (None = no limits)
            max_chunk_memory_mb: Maximum memory per chunk
            cache: Explicit intermediate cache implementing the runtime
                CacheProtocol; defaults to a cache_v2-backed
                ``V2IntermediateCache``
        """
        # `is not None`, not truthiness: an empty caller-supplied cache (e.g.
        # one defining __len__/__bool__) must not be silently discarded.
        self.cache = cache if cache is not None else V2IntermediateCache(
            max_size_mb=cache_size_mb, enable=enable_cache
        )
        self.budget = budget or ComputationBudget()
        self.budget_tracker = BudgetTracker(self.budget)
        self.max_chunk_memory_mb = max_chunk_memory_mb

        # Registry of metric functions
        self._metric_functions: Dict[str, Callable] = {}

        # Execution stats
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_identities = {}
        self._dependency_identities = {}

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

        # Immutable FactorBatch/LabelBundle identities are computed once per
        # execution, never once per requested metric or retained by object id
        # across runs (where Python may reuse an id).
        self._batch_identities = {}
        self._dependency_identities = {}

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
            split_axes = {self._chunk_contract(node)['split_axis'] for node in dep_graph.nodes.values()}
            if len(split_axes) != 1:
                raise InvalidContractError('All dependency nodes must share one proven split axis')
            split_axis = next(iter(split_axes))
            plan_kwargs = {'asset_chunk_size': factor_batch.num_assets}
            if split_axis == 'time':
                plan_kwargs['factor_chunk_size'] = factor_batch.num_factors
            else:
                plan_kwargs['time_chunk_size'] = factor_batch.num_times
                per_factor_mb = factor_batch.num_times * factor_batch.num_assets * 16 / (1024 * 1024)
                plan_kwargs['factor_chunk_size'] = max(1, min(factor_batch.num_factors,
                    int(self.max_chunk_memory_mb / max(per_factor_mb, 1e-30))))
            batch_plan = create_batch_plan(
                factor_batch,
                max_chunk_memory_mb=self.max_chunk_memory_mb,
                **plan_kwargs,
            )
            self._validate_chunk_plan(batch_plan, dep_graph)

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
        result.metadata['runtime_contract'] = 'qe.named-chunks-dependency-cache.v2'
        result.provenance = self._provenance(factor_batch, label_bundle)
        result.metadata["provenance"] = result.provenance
        self._batch_identities.clear()

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
            self._validate_metric_requirements(metric_id, factor_batch, label_bundle, result.metrics)
            self._prepare_metric_call(metric_id, node, factor_batch, label_bundle, result.metrics)
            cache_identity = self._cache_identity(metric_id, node, factor_batch, label_bundle, dep_graph=dep_graph)
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
        self._validate_chunk_plan(batch_plan, dep_graph)
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
            sorted(batch_plan.chunks, key=lambda item: (item.time_slice[0], item.asset_slice[0], item.factor_slice[0])),
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
        self._validate_inputs(chunk_batch, chunk_labels)

        # Execute metrics on chunk
        for metric_id in execution_order:
            node = dep_graph.get_node(metric_id)
            if node is None:
                continue

            # Check cache
            self._validate_metric_requirements(metric_id, chunk_batch, chunk_labels, chunk_metrics)
            self._prepare_metric_call(metric_id, node, chunk_batch, chunk_labels, chunk_metrics)
            cache_identity = self._cache_identity(
                metric_id, node, chunk_batch, chunk_labels, chunk=chunk, dep_graph=dep_graph
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

        # Input owners are needed only while this tile's metrics execute.
        self._batch_identities.pop((id(chunk_batch), id(chunk_labels)), None)
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
        metric_fn, call_args = self._prepare_metric_call(metric_id, node, factor_batch, label_bundle, computed_metrics)
        try:
            return metric_fn(**call_args)
        except QuantEvaluatorError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error computing metric {metric_id}: {e}") from e

    def _prepare_metric_call(self, metric_id, node, factor_batch, label_bundle, computed_metrics):
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

        # **kwargs does not make explicit required parameters optional.
        if 'signature' in locals():
            unresolved = [name for name, p in signature.parameters.items()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
                and p.default is inspect.Parameter.empty and name not in call_args]
            if unresolved:
                raise InvalidContractError(f'Metric {metric_id} has unresolved required parameters: {unresolved}')
        return metric_fn, call_args

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
            spec = get_metric(_resolve_alias(metric_id))
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
            metric_fn = get_metric(_resolve_alias(metric_id)).compute_fn
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

        return label_bundle.slice(slice(t_start, t_end), slice(a_start, a_end))

    def _aggregate_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        execution_order: List[str],
        dep_graph: MetricDependencyGraph,
        chunks: Optional[List[ChunkDescriptor]] = None,
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

            contract = self._chunk_contract(node)
            if contract is None or chunks is None or len(chunk_values) != len(chunks):
                raise InvalidContractError(f"Metric {metric_id} requires named axes and complete chunk coordinates")
            aggregation = contract['merge']
            if aggregation == "mean":
                raise InvalidContractError(
                    f"Metric {metric_id} requires an explicit weighted reduction protocol"
                )
            if not self._metric_is_partitionable(node) or aggregation not in {
                "concat", "sum"
            }:
                raise InvalidContractError(
                    f"Metric {metric_id} has no safe chunk aggregation contract"
                )

            if aggregation == "concat":
                axis = contract['output_axes'].index(contract['split_axis'])
                order = sorted(range(len(chunks)), key=lambda i: getattr(chunks[i], contract['split_axis'] + '_slice')[0])
                for i in order:
                    value = np.asarray(chunk_values[i])
                    if value.ndim != len(contract['output_axes']):
                        raise InvalidContractError('Chunk output rank does not match named axes')
                    for dim, name in enumerate(contract['output_axes']):
                        start, end = getattr(chunks[i], name + '_slice')
                        if value.shape[dim] != end - start:
                            raise InvalidContractError('Chunk output does not match declared coordinates')
                aggregated[metric_id] = np.concatenate([chunk_values[i] for i in order], axis=axis)
            elif aggregation == "sum":
                for value, chunk in zip(chunk_values, chunks):
                    shape = tuple(getattr(chunk, name + '_slice')[1] - getattr(chunk, name + '_slice')[0]
                                  for name in contract['output_axes'])
                    if np.shape(value) != shape:
                        raise InvalidContractError('Reduction output does not match named coordinates')
                aggregated[metric_id] = np.sum(np.stack(chunk_values, axis=0), axis=0)

        return aggregated

    @staticmethod
    def _metric_is_partitionable(node: MetricNode) -> bool:
        return node.metadata.get("partitionability", "NOT_PARTITIONABLE") == "PARTITIONABLE"

    def _chunking_is_safe(self, dep_graph: MetricDependencyGraph) -> bool:
        return all(
            self._metric_is_partitionable(node)
            and self._chunk_contract(node) is not None
            and not node.requires_full_batch
            for node in dep_graph.nodes.values()
        )

    @staticmethod
    def _chunk_contract(node):
        contract = node.metadata.get('chunk_contract')
        if not isinstance(contract, dict) or contract.get('version') != 1:
            return None
        axis = contract.get('split_axis')
        axes = contract.get('output_axes')
        merge = contract.get('merge')
        if (axis not in {'time', 'factor'} or not isinstance(axes, (list, tuple))
                or len(set(axes)) != len(axes) or any(a not in {'time','asset','factor'} for a in axes)
                or contract.get('halo') != 0 or merge not in {'concat','sum'}
                or node.metadata.get('aggregation') != merge
                or (merge == 'concat' and axis not in axes)
                or (merge == 'sum' and axis in axes)):
            raise InvalidContractError('Unsupported chunk contract: require named axes, concat/additive state, zero halo')
        return contract

    def _validate_chunk_plan(self, plan, graph):
        for node in graph.nodes.values():
            contract = self._chunk_contract(node)
            if contract is None:
                raise InvalidContractError('No proven chunk reduction contract')
            split = contract['split_axis']
            previous = 0
            for chunk in sorted(plan.chunks, key=lambda c: getattr(c, split + '_slice')[0]):
                start, end = getattr(chunk, split + '_slice')
                if start != previous or end <= start:
                    raise InvalidContractError('Chunk coordinates overlap or omit observations')
                previous = end
                for axis, size in [('time',plan.total_time),('asset',plan.total_assets),('factor',plan.total_factors)]:
                    if axis != split and getattr(chunk, axis + '_slice') != (0,size):
                        raise InvalidContractError('Planner split an undeclared axis; full cross-section/state required')
            total = plan.total_time if split == 'time' else plan.total_factors
            if previous != total:
                raise InvalidContractError('Incomplete chunk coordinates')

    @staticmethod
    def _array_hash(value: np.ndarray) -> str:
        return authoritative_array_hash(value)

    @staticmethod
    def _freeze(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": Evaluator._array_hash(value),
            }
        if isinstance(value, dict):
            return {str(k): Evaluator._freeze(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [Evaluator._freeze(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # Canonical stable form for anything else (deterministic across
        # processes, unlike builtin hash()).
        return canonicalize(value)

    @staticmethod
    def _is_immutably_cacheable(value: Any) -> bool:
        if isinstance(value, (str, int, float, bool, bytes, type(None))):
            return True
        if isinstance(value, tuple):
            return all(Evaluator._is_immutably_cacheable(item) for item in value)
        if isinstance(value, frozenset):
            return all(Evaluator._is_immutably_cacheable(item) for item in value)
        return False

    def _cache_identity(self, metric_id, node, factor_batch, label_bundle, chunk=None, dep_graph=None):
        try:
            metric_fn = self._resolve_metric_function(metric_id)
            if not isinstance(metric_fn, types.FunctionType):
                # Callable instances can hide mutable state behind descriptors or
                # custom attribute access; their semantic identity is unprovable.
                return None
            code = metric_fn.__code__
            contract = node.metadata.get('cache_contract')
            explicit = (isinstance(contract, dict) and contract.get('version') == 1
                        and isinstance(contract.get('semantic_ref'), str) and bool(contract['semantic_ref'])
                        and isinstance(contract.get('dependency_refs'), dict)
                        and all(isinstance(k,str) and isinstance(v,str) and v
                                for k,v in contract['dependency_refs'].items()))
            if contract is not None and not explicit:
                return None
            if metric_id in self._metric_functions and not explicit:
                # User code can depend on object identity or hidden side
                # effects even without globals; registration is not a purity
                # declaration. Only an explicit owner contract opts it in.
                return None
            # A callable's local bytecode cannot identify module state or
            # imported helpers. Opt out unless its owner supplies a versioned
            # semantic/dependency contract; immutable scalar globals are safe
            # to include by value. This applies to catalog functions too.
            global_state = {}
            import dis
            codes = [code]
            names = set()
            while codes:
                current = codes.pop()
                names.update(current.co_names)
                if not explicit and any(ins.opname in {'IMPORT_NAME','IMPORT_FROM','STORE_GLOBAL','DELETE_GLOBAL'}
                                        for ins in dis.get_instructions(current)):
                    return None
                codes.extend(value for value in current.co_consts if isinstance(value, types.CodeType))
            if not explicit and names & {'globals','locals','eval','exec','getattr','setattr','__import__','open'}:
                return None
            for name in names:
                if name in metric_fn.__globals__:
                    value = metric_fn.__globals__[name]
                    if self._is_immutably_cacheable(value):
                        global_state[name] = self._freeze(value)
                    elif not explicit:
                        return None
            if not self._is_immutably_cacheable(metric_fn.__defaults__):
                return None
            if any(not self._is_immutably_cacheable(v) for v in (metric_fn.__kwdefaults__ or {}).values()):
                return None
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
                     self._freeze(metric_fn.__kwdefaults__), closure_state, global_state],
                    sort_keys=True, separators=(",", ":"),
                    default=canonicalize,
                ).encode("utf-8")
            ).hexdigest()
            dependencies = {}
            for dep_id in sorted(getattr(node, 'dependencies', ())):
                if dep_graph is None:
                    return None
                dependency = self._cache_identity(dep_id, dep_graph.get_node(dep_id),
                    factor_batch, label_bundle, chunk=chunk, dep_graph=dep_graph)
                if dependency is None:
                    return None
                dependencies[dep_id] = dependency
            key = (id(factor_batch), id(label_bundle))
            if key not in self._batch_identities:
                batch_identity = {
                    'factor_ids': list(factor_batch.factor_ids),
                    'factor_values': self._freeze(factor_batch.values),
                    'factor_validity': self._freeze(factor_batch.validity),
                    'time_axis': self._freeze(factor_batch.time_axis.values),
                    'asset_axis': self._freeze(factor_batch.asset_axis.values),
                    'context_refs': self._freeze(factor_batch.context_refs),
                    'factor_value_hash': factor_batch.value_hash,
                    'labels': label_bundle.content_hash,
                }
                # Retain owners until this run ends to prevent id reuse.
                self._batch_identities[key] = (factor_batch, label_bundle, batch_identity)
            try:
                registry_identity = self._freeze({k:v for k,v in vars(get_metric(_resolve_alias(metric_id))).items()
                                                  if k != 'compute_fn'})
            except (KeyError, TypeError, ValueError):
                registry_identity = None
            identity = {
                'runtime_contract': 'qe.named-chunks-dependency-cache.v2',
                "metric": metric_id,
                "kind": node.metric_kind.value,
                "metadata": self._freeze(node.metadata),
                "function": [metric_fn.__module__, metric_fn.__qualname__, function_hash],
                "batch": self._batch_identities[key][2],
                "registry": registry_identity,
                "dependencies": dependencies,
                "chunk": self._freeze(
                    (chunk.time_slice, chunk.asset_slice, chunk.factor_slice)
                ) if chunk else None,
            }
            payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except (TypeError, ValueError, OSError):
            return None

    def _provenance(self, factor_batch, label_bundle):
        # Immutable-by-construction plain types (tuples/dicts of scalars and
        # hash digests) — mappingproxy is not picklable and leaks into
        # multiprocessing result payloads, so it must not be used here.
        cached = self._batch_identities.get((id(factor_batch), id(label_bundle)))
        identity = cached[2] if cached else {}
        return {
            "factor_ids": tuple(factor_batch.factor_ids),
            "time_coordinates": self._freeze(factor_batch.time_axis.values),
            "asset_coordinates": self._freeze(factor_batch.asset_axis.values),
            "context_refs": dict(factor_batch.context_refs),
            # factor_value_hash is the caller's optional external reference;
            # the two fields below are authoritative runtime-computed content.
            "factor_value_hash": factor_batch.value_hash,
            "factor_value_bytes_hash": identity['factor_values']['sha256'] if identity else authoritative_array_hash(factor_batch.values),
            "factor_validity_hash": (
                (identity['factor_validity']['sha256'] if identity else authoritative_array_hash(factor_batch.validity))
                if factor_batch.validity is not None else None
            ),
            "label_source_ref": label_bundle.source_ref,
            "label_calendar_ref": label_bundle.calendar_ref,
            "label_target_id": label_bundle.target_id,
            "label_value_hash": self._array_hash(label_bundle.values),
            "label_metadata": dict(label_bundle.metadata),
        }

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
        label_times = label_bundle.observation_time or label_bundle.decision_time
        if factor_batch.time_axis.values is not None and not np.array_equal(
            factor_batch.time_axis.values, np.asarray(label_times)
        ):
            raise InvalidContractError("Factor and label time coordinates do not match")
        if label_bundle.values.ndim == 2 and label_bundle.values.shape[1] != factor_batch.num_assets:
            raise InvalidContractError("Factor and label asset dimensions do not match")
        if label_bundle.asset_axis is not None:
            if factor_batch.asset_axis.values is None or label_bundle.asset_axis.values is None:
                raise InvalidContractError("Bound asset axes require explicit coordinates")
            if not np.array_equal(factor_batch.asset_axis.values, label_bundle.asset_axis.values):
                raise InvalidContractError("Factor and label asset coordinates do not match")

    def clear_cache(self):
        """Clear intermediate cache."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def get_budget_report(self) -> str:
        """Get formatted budget usage report."""
        return self.budget_tracker.format_usage_report()




def _per_factor_observation_counts(
    metric_id: str,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    values: np.ndarray,
    parameters: Optional[Dict[str, Any]] = None,
    ic_cache=None,
) -> np.ndarray:
    """Best-effort per-factor observation counts for facade MetricValues.

    - coverage: number of jointly valid (factor, label) cells.
    - ic.pearson/rank families: number of finite daily IC observations.
    - anything else: number of finite adapted values per factor — honest
      for scalar adapters.
    """
    from quant_evaluator.metrics.ic import compute_daily_ic
    from quant_evaluator.metrics.quality import compute_valid_pair_counts

    registry_name = _resolve_alias(metric_id)
    if registry_name == "coverage":
        return compute_valid_pair_counts(factor_batch, label_bundle)
    spec = get_metric(registry_name)
    parameters = parameters or {}
    default = inspect.signature(spec.compute_fn).parameters.get("min_assets")
    min_assets = parameters.get("min_assets", default.default if default is not None else 20)
    if registry_name in {"pearson_ic", "rank_ic", "mean_ic"} or "ICSeriesArtifact" in (spec.requires or []):
        method = _ic_method_for_metric(registry_name)
        key = (method, min_assets)
        if ic_cache is not None and key in ic_cache:
            series = ic_cache[key]
        else:
            series, _ = compute_daily_ic(factor_batch, label_bundle, method=method, min_assets=min_assets)
            if ic_cache is not None:
                ic_cache[key] = series
        return np.sum(np.isfinite(series), axis=0)
    if registry_name in {"pearson_ic_series", "rank_ic_series"}:
        method = "pearson" if registry_name == "pearson_ic_series" else "spearman"
        series, _ = compute_daily_ic(
            factor_batch, label_bundle, method=method, min_assets=min_assets
        )
        return np.sum(np.isfinite(series), axis=0)
    if registry_name in {"turnover", "factor_turnover_rate"}:
        finite = np.isfinite(factor_batch.values)
        if factor_batch.validity is not None:
            finite &= factor_batch.validity
        pairs = finite[1:] & finite[:-1]
        if registry_name == "factor_turnover_rate":
            from quant_evaluator.metrics.temporal import compute_factor_turnover_rate
            membership = compute_factor_turnover_rate(
                np.where(finite, factor_batch.values, np.nan),
                quantile=parameters.get("quantile", .9))
            return np.isfinite(membership).sum(axis=0)
        valid = (finite.sum(axis=1) >= 2)
        return ((pairs.sum(axis=1) > 0) & valid[1:] & valid[:-1]).sum(axis=0)
    if registry_name == "daily_quantile_monotonicity_rate":
        from quant_evaluator.metrics.registry_adapters import compute_daily_quantile_monotonicity_series_value
        signature = inspect.signature(compute_daily_quantile_monotonicity_series_value)
        kwargs = {key: value for key, value in parameters.items() if key in signature.parameters}
        series = compute_daily_quantile_monotonicity_series_value(
            factor_batch, label_bundle, **kwargs)
        return np.sum(np.isfinite(series), axis=0)
    return (np.isfinite(values)).astype(np.int64)


def _ic_method_for_metric(registry_name: str) -> str:
    """IC correlation method for a registry metric (QE-P1-27).

    Reads the spec-declared ``ic_method`` ("spearman" for rank-family
    metrics, "pearson" by default) so the wrapper IC series and the
    observation-count basis always match the metric's own semantics.
    Falls back to "pearson" for unknown/unregistered names.
    """
    try:
        return get_metric(registry_name).ic_method
    except (KeyError, AttributeError):
        return "pearson"


def _make_panel_wrapper(cfn: Callable, portfolio_returns=None, calendar_snapshot=None) -> Callable:
    """Wrap a registry compute_fn that consumes raw return/factor panels.

    The generic runtime binder supplies ``factor_batch`` / ``label_bundle``
    (and ``computed_metrics`` / ``metadata``); the panel metrics need the raw
    (T, N) forward-return panel (and, for long/short construction, the raw
    factor panel) instead.  Derive the panel-shaped parameters from the
    contracts and forward only the arguments the compute_fn accepts, so the
    registry implementation stays the single source of truth.
    """
    def _panel_wrapper(
        factor_batch: Optional[FactorBatch] = None,
        label_bundle: Optional[LabelBundle] = None,
        computed_metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        try:
            fn_sig = inspect.signature(cfn)
        except (TypeError, ValueError):
            filtered = dict(kwargs)
        else:
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in fn_sig.parameters.values()
            ):
                filtered = dict(kwargs)
            else:
                filtered = {
                    k: v for k, v in kwargs.items() if k in fn_sig.parameters
                }
        if "returns" in fn_sig.parameters:
            if portfolio_returns is None:
                raise InvalidContractError(
                    "Portfolio risk metrics require a ProbePortfolioArtifact; forward labels are not portfolio returns"
                )
            filtered["returns"] = portfolio_returns.values
        if "calendar_snapshot" in fn_sig.parameters:
            filtered["calendar_snapshot"] = calendar_snapshot
            filtered["time_index"] = portfolio_returns.time_index
            filtered["factor_ids"] = portfolio_returns.factor_ids
        if "forward_returns" in fn_sig.parameters:
            if label_bundle is None:
                raise InvalidContractError(
                    "returns-panel metric requires label_bundle"
                )
            filtered["forward_returns"] = _label_return_panel(label_bundle)
        if "factor_values" in fn_sig.parameters:
            if factor_batch is None:
                raise InvalidContractError(
                    "long_short_returns requires factor_batch"
                )
            values = np.asarray(factor_batch.values, dtype=np.float64)
            if factor_batch.validity is not None:
                values = np.where(
                    np.asarray(factor_batch.validity, dtype=bool),
                    values,
                    np.nan,
                )
            filtered["factor_values"] = values
        if "validity_mask" in fn_sig.parameters:
            if factor_batch is not None and factor_batch.validity is not None:
                filtered["validity_mask"] = np.asarray(
                    factor_batch.validity, dtype=bool
                )
        return cfn(**filtered)
    return _panel_wrapper


def _label_return_panel(label_bundle: LabelBundle) -> np.ndarray:
    """Return the raw (T, N) forward-return panel of a LabelBundle."""
    values = np.asarray(label_bundle.values, dtype=np.float64)
    if label_bundle.validity is not None:
        values = np.where(
            np.asarray(label_bundle.validity, dtype=bool), values, np.nan
        )
    return values


# QE-R2: registry metric ids whose compute_fn consumes the raw (T, N) return
# (and factor) panel rather than a per-factor reduction.  Kept out of the
# generic argument binder; the facade wraps them with ``_make_panel_wrapper``.
_RETURNS_PANEL_METRIC_IDS = frozenset({
    "turnover_cost",
    "tracking_error", "information_ratio", "relative_max_drawdown", "mean_investment_fraction",
    "long_short_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "max_drawdown",
    "calmar_ratio",
    "worst_calendar_month", "worst_calendar_quarter", "worst_calendar_year",
    "worst_rolling_21d", "worst_rolling_63d", "worst_rolling_252d",
})


def evaluate(
    factors,
    labels=None,
    *,
    context=None,
    metrics=None,
    where=None,
    evaluator=None,
    split_ref=None,
    backend=None,
    gpu_policy=None,
    metric_parameters=None,
    portfolio_returns=None,
    holding_returns=None,
    portfolio_spec=None,
    trade_eligibility=None,
    calendar_snapshot=None,
    exposure_panel=None,
    quantile_builder_parameters=None,
    generalization_evidence=None,
):
    """Evaluate explicit factor and label contracts through the runtime.

    This is the public batch facade.  It delegates computation to ``Evaluator``
    and adapts only scalar values produced by existing metric kernels into the
    canonical :class:`EvaluationBundle` contract.  Unsupported requests fail
    closed rather than fabricating metric metadata.

    ``split_ref`` (optional, keyword-only) binds the evaluation to a sealed
    test split; when provided (directly or via an ``EvaluationRequest``), the
    guard raises :class:`SealedSplitOverlapError` if the split window overlaps
    the factor/label information boundary (R21 Q5 sealed-test gate).  The
    default ``None`` performs no check, preserving every existing caller.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from quant_evaluator.api.requests import EvaluationBundle, MetricValue
    from quant_evaluator.contracts.sealed_split import check_sealed_split_overlap
    from quant_evaluator.diagnosis.factor import diagnose_all_factors

    request_metadata = {}
    request_fields = {}
    ic_series_cache = {}
    if isinstance(factors, EvaluationRequest) and factors.metric_instances:
        if where is not None:
            raise UnsupportedMetricError("where slicing is not supported by public evaluate")
        from quant_evaluator.runtime.metric_instances import evaluate_instances
        return evaluate_instances(factors, evaluator=evaluator, backend=backend, gpu_policy=gpu_policy)
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
        metric_parameters = request.metric_parameters
        portfolio_returns = request.portfolio_returns
        holding_returns = request.holding_returns
        portfolio_spec = request.portfolio_spec
        trade_eligibility = request.trade_eligibility
        calendar_snapshot = request.calendar_snapshot
        exposure_panel = request.exposure_panel
        generalization_evidence = request.generalization_evidence
        quantile_builder_parameters = request.quantile_builder_parameters
        split_ref = request.split_ref
    else:
        factor_batch = factors
        label_bundle = labels
        metric_ids = tuple(metrics or ("coverage",))
        if split_ref is None:
            split_ref = None

    if where is not None:
        raise UnsupportedMetricError("where slicing is not supported by public evaluate")
    if not isinstance(factor_batch, FactorBatch) or not isinstance(label_bundle, LabelBundle):
        raise TypeError("evaluate requires a FactorBatch and LabelBundle")
    canonical_metrics = tuple(dict.fromkeys(_resolve_alias(mid) for mid in metric_ids))
    generalization_ids={"train_predictive_dimension","validation_predictive_dimension",
        "validation_retention","train_validation_rankic_delta","train_validation_icir_delta",
        "train_validation_sharpe_delta","train_validation_shape_delta","parameter_generalization"}
    if generalization_ids.intersection(canonical_metrics):
        from quant_evaluator.metrics.generalization_evidence import TrainVsValidationArtifact
        if not isinstance(generalization_evidence,TrainVsValidationArtifact):
            raise InvalidContractError("Generalization metrics require typed TrainVsValidationArtifact")
        if tuple(generalization_evidence.factor_ids)!=tuple(factor_batch.factor_ids):
            raise InvalidContractError("Generalization evidence factor axis must match the evaluation request")
        from collections.abc import Mapping
        version_refs=factor_batch.context_refs.get("factor_versions")
        if not isinstance(version_refs,Mapping) or set(version_refs)!=set(factor_batch.factor_ids):
            raise InvalidContractError("Generalization requires authoritative context_refs.factor_versions by factor ID")
        generalization_versions=tuple(version_refs[fid] for fid in factor_batch.factor_ids)
        if generalization_versions!=tuple(generalization_evidence.factor_versions):
            raise InvalidContractError("Generalization evidence factor versions differ from the evaluation request")
    quantile_builder_parameters = dict(quantile_builder_parameters or {})
    metric_parameters = {key: dict(value) for key, value in (metric_parameters or {}).items()}
    builder_floors = {"n_quantiles": 2, "min_assets": 2, "window_size": 1}
    for name, value in quantile_builder_parameters.items():
        if name not in builder_floors or isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < builder_floors[name]:
            raise InvalidContractError(f"Invalid quantile builder parameter {name}: {value!r}")
    if "adaptive_quantile_count" in canonical_metrics:
        from quant_evaluator.contracts.adaptive_bins_policy import AdaptiveBinsPolicy
        for requested_id in metric_ids:
            if _resolve_alias(requested_id) != "adaptive_quantile_count":
                continue
            parameters = metric_parameters.setdefault(requested_id, {})
            if "policy" in parameters and not isinstance(parameters["policy"], AdaptiveBinsPolicy):
                parameters["policy"] = AdaptiveBinsPolicy.from_dict(parameters["policy"]).to_dict()
    adaptive_quantile_binding = None
    adaptive_resolution_artifact = None
    adaptive_gpu_tiles = {}
    adaptive_profile_ids = {"quantile_returns_daily", "quantile_returns_full"}
    requested_adaptive_profiles = adaptive_profile_ids.intersection(canonical_metrics)
    if "adaptive_quantile_count" in canonical_metrics:
        from quant_evaluator.metrics.shape_evidence import compute_adaptive_quantile_count

        adaptive_count_id = next(
            requested_id for requested_id in metric_ids
            if _resolve_alias(requested_id) == "adaptive_quantile_count"
        )
        adaptive_resolution_artifact = compute_adaptive_quantile_count(
            factor_batch, **dict(metric_parameters.get(adaptive_count_id, {})))
        selected = adaptive_resolution_artifact.values[np.isfinite(adaptive_resolution_artifact.values)]
        if requested_adaptive_profiles and selected.size != factor_batch.num_factors:
            raise InvalidContractError(
                "Adaptive quantile profiles require a feasible Q for every factor"
            )
        selected_q = tuple(
            int(value) if np.isfinite(value) else None
            for value in adaptive_resolution_artifact.values
        )
        requested_nq = quantile_builder_parameters.get("n_quantiles")
        if requested_nq is not None and any(q is not None and requested_nq != q for q in selected_q):
            raise InvalidContractError(
                f"quantile_builder_parameters n_quantiles={requested_nq} conflicts with adaptive Q axes={selected_q}"
            )
        for requested_id in metric_ids:
            if _resolve_alias(requested_id) not in requested_adaptive_profiles:
                continue
            parameters = metric_parameters.setdefault(requested_id, {})
            explicit_nq = parameters.get("n_quantiles")
            if explicit_nq is not None and any(q is not None and explicit_nq != q for q in selected_q):
                raise InvalidContractError(
                    f"{requested_id} n_quantiles={explicit_nq} conflicts with adaptive Q axes={selected_q}"
                )
            parameters.pop("n_quantiles", None)
        adaptive_quantile_binding = {
            "selected_q_by_factor": selected_q,
            "factor_ids": tuple(factor_batch.factor_ids),
            "comparison_policy": "largest_fixed_q_feasible_on_every_date",
            "count_artifact_policy": adaptive_resolution_artifact.provenance["adaptive_bins_policy"],
        }
    try:
        resolved_specs = {mid: get_metric(mid) for mid in canonical_metrics}
    except KeyError as exc:
        raise UnsupportedMetricError(str(exc)) from exc
    artifact_plan = _compile_public_artifact_plan(
        resolved_specs,
        portfolio_returns=portfolio_returns,
        holding_returns=holding_returns,
        exposure_panel=exposure_panel,
    )
    if any(mid.startswith("worst_calendar_") for mid in canonical_metrics):
        from data_access.r30.calendar_snapshot import CalendarSnapshot
        if not isinstance(calendar_snapshot, CalendarSnapshot):
            raise InvalidContractError("Calendar metrics require authoritative CalendarSnapshot")
    # Reject malformed parameter plans before CPU execution or GPU allocation.
    for requested_id, overrides in (metric_parameters or {}).items():
        canonical = _resolve_alias(requested_id)
        if canonical not in resolved_specs:
            raise InvalidContractError(f"Parameters supplied for unrequested metric {requested_id}")
        if canonical in generalization_ids and overrides:
            raise InvalidContractError("Generalization policy and inputs must be bound in the typed comparison artifact")
        signature = inspect.signature(resolved_specs[canonical].compute_fn)
        forbidden = {"factor_batch", "label_bundle", "computed_metrics", "metadata",
                     "factor_values", "forward_returns", "returns", "validity_mask",
                     "calendar_snapshot", "time_index", "factor_ids"}
        invalid = (set(overrides) - set(signature.parameters)) | (set(overrides) & forbidden)
        if invalid:
            raise InvalidContractError(f"Invalid parameters for {requested_id}: {sorted(invalid)}")
        if "exposure_panel" in (resolved_specs[canonical].requires or ()):
            if set(overrides)&{"panel","weights"}:
                raise InvalidContractError("Exposure weights and panel must be bound through ExposurePanel")
            for key,floor in (("min_obs",2),("min_finite",1)):
                value=overrides.get(key,floor)
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<floor:
                    raise InvalidContractError(f"{key} must be an integer >= {floor}")
        if canonical.startswith("worst_rolling_") and "window" in overrides:
            expected_window = int(canonical.rsplit("_", 1)[1][:-1])
            if overrides["window"] != expected_window:
                raise InvalidContractError("Fixed rolling metric ID cannot change its window")
    requested_tier = request_fields.get("tier")
    tier_order = {"core": 0, "extended": 1, "research": 2}
    if requested_tier is not None:
        if requested_tier not in tier_order:
            raise InvalidContractError(f"Unknown evaluation tier {requested_tier!r}")
        outside = [mid for mid, spec in resolved_specs.items()
                   if tier_order[spec.tier.value] > tier_order[requested_tier]]
        if outside:
            raise InvalidContractError(f"Metrics exceed requested tier {requested_tier}: {outside}")
    metric_cost = float(len(canonical_metrics))
    builder_cost = float(sum(node["cost"] for node in artifact_plan))
    planned_cost = metric_cost + builder_cost
    budget = request_fields.get("cost_budget")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget < 0:
            raise InvalidContractError("cost_budget must be finite non-negative metric-node units")
        if planned_cost > budget:
            raise InvalidContractError(
                f"Evaluation plan cost {planned_cost} (metrics={metric_cost}, "
                f"artifact_builders={builder_cost}) exceeds cost_budget {budget}"
            )
    # Both low-level and public evaluation validate coordinates before staging.
    runtime = evaluator or Evaluator()
    runtime._validate_inputs(factor_batch, label_bundle)
    if holding_returns is not None:
        from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
        if not isinstance(holding_returns, HoldingReturnPanel) or not isinstance(portfolio_spec, PortfolioSpec):
            raise InvalidContractError("holding_returns requires HoldingReturnPanel and a frozen PortfolioSpec")
        if portfolio_returns is not None:
            raise InvalidContractError("Supply holding inputs or prebuilt portfolio returns, not both")
        if not np.array_equal(holding_returns.time_axis.values, np.asarray(label_bundle.observation_time or label_bundle.decision_time)):
            raise InvalidContractError("Holding-return time coordinates do not match")
        if factor_batch.asset_axis.values is None or not np.array_equal(holding_returns.asset_axis.values, factor_batch.asset_axis.values):
            raise InvalidContractError("Holding-return asset coordinates do not match")
    if any(spec.required_portfolio_leg is not None for spec in resolved_specs.values()) and portfolio_returns is None:
        raise InvalidContractError("Benchmark/capital metrics require an explicitly tagged execution trajectory leg")
    if portfolio_returns is not None:
        from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
        if not isinstance(portfolio_returns, ProbePortfolioArtifact):
            raise InvalidContractError("portfolio_returns must be a ProbePortfolioArtifact, never a LabelBundle")
        if tuple(portfolio_returns.factor_ids) != tuple(factor_batch.factor_ids):
            raise InvalidContractError("Portfolio factor coordinates do not match")
        if tuple(portfolio_returns.time_index) != tuple(label_bundle.observation_time or label_bundle.decision_time):
            raise InvalidContractError("Portfolio time coordinates do not match")
        for mid, spec in resolved_specs.items():
            expected_leg = spec.required_portfolio_leg
            if expected_leg is not None and portfolio_returns.provenance.get("leg") != expected_leg:
                raise InvalidContractError(f"{mid} requires explicitly bound portfolio leg {expected_leg}")
    if trade_eligibility is not None:
        from quant_evaluator.contracts.portfolio_inputs import TradeEligibilityPanel
        if not isinstance(trade_eligibility, TradeEligibilityPanel) or holding_returns is None:
            raise InvalidContractError("TradeEligibilityPanel requires independent holding inputs")
        if not np.array_equal(trade_eligibility.time_axis.values, holding_returns.time_axis.values) or not np.array_equal(trade_eligibility.asset_axis.values, holding_returns.asset_axis.values):
            raise InvalidContractError("TradeEligibilityPanel coordinates do not match holding returns")
    if exposure_panel is not None:
        from quant_evaluator.metrics.exposure_evidence import ExposurePanel
        if not isinstance(exposure_panel, ExposurePanel):
            raise InvalidContractError("exposure_panel must be an ExposurePanel")
        if not exposure_panel.source_ref or not exposure_panel.provider or not exposure_panel.universe_snapshot_ref:
            raise InvalidContractError(
                "ExposurePanel requires source_ref, provider, and universe_snapshot_ref"
            )
        expected_times = tuple(label_bundle.observation_time or label_bundle.decision_time)
        if not exposure_panel.date_index or tuple(exposure_panel.date_index) != expected_times:
            raise InvalidContractError("ExposurePanel time axis does not match evaluation")
        expected_assets = (() if factor_batch.asset_axis.values is None
                           else tuple(factor_batch.asset_axis.values.tolist()))
        if not exposure_panel.security_ids or tuple(exposure_panel.security_ids) != expected_assets:
            raise InvalidContractError("ExposurePanel security axis does not match evaluation")
        if not exposure_panel.factor_ids or tuple(exposure_panel.factor_ids) != tuple(factor_batch.factor_ids):
            raise InvalidContractError("ExposurePanel factor axis does not match evaluation")
        if exposure_panel.values.shape[:2] != (factor_batch.num_times, factor_batch.num_assets):
            raise InvalidContractError("ExposurePanel T/security dimensions do not match evaluation")
    # R21 Q5 sealed-test gate: fail closed when a requested split overlaps
    # the factor/label information boundary.  split_ref=None skips the check
    # entirely (backward compatible with all existing callers).
    check_sealed_split_overlap(
        split_ref,
        decision_times=tuple(label_bundle.decision_time),
        label_start_times=tuple(label_bundle.label_start_time),
        label_end_times=tuple(label_bundle.label_end_time),
        factor_times=(
            tuple(factor_batch.time_axis.values.tolist())
            if factor_batch.time_axis.values is not None
            else None
        ),
    )
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

    # Apply the same leakage and shape guards before either backend starts.
    backend_name = getattr(backend, "value", backend)
    gpu_result = None
    gpu_risk_results = {}
    if backend_name is not None and str(backend_name).lower() in ("cuda", "cuda_strict"):
        from quant_evaluator.runtime.device_session import DeviceEvaluationSession
        from quant_evaluator.runtime.gpu_executor import GPUExecutor
        from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
        gpu_risk_ids = tuple(mid for mid in canonical_metrics
                             if mid.startswith("worst_calendar_") or mid.startswith("worst_rolling_"))
        gpu_plan_metrics = tuple(
            mid for mid in canonical_metrics
            if not (adaptive_resolution_artifact is not None and
                    (mid == "adaptive_quantile_count" or mid in requested_adaptive_profiles))
            and mid not in gpu_risk_ids
        )
        GPUExecutor.validate_metric_plan(gpu_plan_metrics)

        if label_bundle.values.shape != (factor_batch.num_times, factor_batch.num_assets):
            raise InvalidContractError("CUDA labels must match the factor time/asset panel")
        if context is not None:
            raise UnsupportedMetricError("CUDA evaluation context is not implemented")
        if quantile_builder_parameters:
            raise UnsupportedMetricError("CUDA shape-window builder is not implemented")
        policy = gpu_policy or GPUExecutionPolicy()
        with DeviceEvaluationSession(policy) as session:
            executor = GPUExecutor(session)
            if exposure_panel is not None:
                executor.exposure_panel_values = np.where(
                    exposure_panel.validity, exposure_panel.values, np.nan
                ) if exposure_panel.validity is not None else exposure_panel.values
                executor.exposure_style_names = tuple(exposure_panel.style_names)
                executor.exposure_regression_weights = exposure_panel.regression_weights
            if gpu_risk_ids:
                if portfolio_returns is not None:
                    pnl_values = portfolio_returns.values
                    risk_trajectory_source = "prebuilt_probe_portfolio"
                elif holding_returns is not None:
                    pnl_values = executor.build_probe_pnl_tiled(
                        factor_batch, holding_returns, portfolio_spec, trade_eligibility)
                    risk_trajectory_source = "cuda_holding_return_probe"
                else:
                    raise UnsupportedMetricError(
                        "CUDA calendar/rolling metrics require holding returns or a prebuilt probe")
                for mid in gpu_risk_ids:
                    params = dict(metric_parameters.get(mid, {}))
                    if mid.startswith("worst_rolling_"):
                        window = int(mid.rsplit("_", 1)[1][:-1])
                        values, counts = executor.run_worst_rolling_compound(
                            pnl_values, window, min_periods=params.get("min_periods", 1))
                        gpu_risk_results[mid] = {
                            "values": values, "observation_counts": counts,
                            "provenance": {"window": window, "min_periods": params.get("min_periods", 1)},
                        }
                    else:
                        from datetime import date
                        from quant_evaluator.metrics.calendar_returns import _local_session_dates, _period_key, _period_id
                        frequency = mid.removeprefix("worst_calendar_")
                        trajectory_times = (portfolio_returns.time_index if portfolio_returns is not None
                                            else tuple(holding_returns.time_axis.values.tolist()))
                        local_dates = _local_session_dates(trajectory_times, calendar_snapshot.timezone)
                        expected_dates = tuple(date.fromisoformat(str(day)[:10]) for day in calendar_snapshot.trading_days)
                        if not expected_dates or any(a >= b for a, b in zip(expected_dates, expected_dates[1:])):
                            raise InvalidContractError("calendar snapshot trading_days must be strictly increasing")
                        unknown = [day.isoformat() for day in local_dates if day not in set(expected_dates)]
                        if unknown:
                            raise InvalidContractError(f"time_index contains sessions absent from calendar snapshot: {unknown}")
                        observed_row = {day: index for index, day in enumerate(local_dates)}
                        expected_by_period = {}
                        for day in expected_dates:
                            expected_by_period.setdefault(_period_key(day, frequency), []).append(day)
                        row_groups, structural_include, period_plan = [], [], []
                        partial_policy = params.get("partial_policy", "exclude")
                        for key, sessions in expected_by_period.items():
                            rows = tuple(observed_row[day] for day in sessions if day in observed_row)
                            bracketed = expected_dates[0] < sessions[0] and expected_dates[-1] > sessions[-1]
                            complete = len(rows) == len(sessions)
                            row_groups.append(rows)
                            structural_include.append(bool(rows) and (partial_policy == "include" or (complete and bracketed)))
                            period_plan.append((_period_id(key, frequency), len(sessions), len(rows), bracketed, complete))
                        period_values, finite_counts, values, counts = executor.run_grouped_compounding(
                            pnl_values, row_groups, structural_include)
                        gpu_risk_results[mid] = {
                            "values": values, "observation_counts": counts,
                            "period_values": period_values, "finite_counts": finite_counts,
                            "structural_include": tuple(structural_include), "period_plan": tuple(period_plan),
                            "provenance": {"partial_policy": partial_policy,
                                "calendar_snapshot_id": calendar_snapshot.snapshot_id,
                                "calendar_market": calendar_snapshot.market,
                                "calendar_timezone": calendar_snapshot.timezone,
                                "calendar_source_version": calendar_snapshot.source_version},
                        }
            if portfolio_returns is not None:
                # Stage only each factor tile, not the whole job's trajectory.
                executor.prebuilt_portfolio_pnl = portfolio_returns.values
                executor.portfolio_factor_ids = tuple(portfolio_returns.factor_ids)
            executor.metric_parameters = {_resolve_alias(k): dict(v) for k, v in (metric_parameters or {}).items()}
            if holding_returns is not None:
                if executor.holding_return_id is None:
                    executor.holding_return_id = session.stage_holding_returns(holding_returns)
                    executor.portfolio_spec = portfolio_spec
                    if trade_eligibility is not None:
                        executor.trade_eligibility = session.stage_trade_eligibility(trade_eligibility)
            canonical_ids = tuple(dict.fromkeys(_resolve_alias(mid) for mid in metric_ids))
            if adaptive_resolution_artifact is not None:
                q_groups = {}
                for index, q in enumerate(adaptive_quantile_binding["selected_q_by_factor"]):
                    q_groups.setdefault(q, []).append(index)
                for q, indices in q_groups.items():
                    tile = FactorBatch(
                        factor_ids=tuple(factor_batch.factor_ids[index] for index in indices),
                        time_axis=factor_batch.time_axis, asset_axis=factor_batch.asset_axis,
                        values=factor_batch.values[:, :, indices],
                        validity=None if factor_batch.validity is None else factor_batch.validity[:, :, indices],
                        context_refs=factor_batch.context_refs,
                    )
                    executor.metric_parameters = {
                        mid: dict(metric_parameters.get(mid, {})) | {"n_quantiles": q}
                        for mid in requested_adaptive_profiles
                    }
                    adaptive_gpu_tiles[q] = executor.run_tiled(
                        tile, label_bundle, tuple(requested_adaptive_profiles))
            standard_ids = tuple(mid for mid in canonical_ids if mid in gpu_plan_metrics)
            if standard_ids:
                executor.metric_parameters = {_resolve_alias(k): dict(v) for k, v in (metric_parameters or {}).items()
                                              if _resolve_alias(k) in standard_ids}
                bundle = executor.run_tiled(factor_batch, label_bundle, standard_ids)
            else:
                from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
                bundle = BatchEvaluationBundle(tuple(factor_batch.factor_ids), label_bundle.target_id)
                bundle.metadata = session.metadata()
            if adaptive_gpu_tiles:
                bundle.metadata.update({
                    "adaptive_planning_backend": "cpu",
                    "adaptive_profile_backend": "cuda_strict",
                    "adaptive_q_groups": tuple(sorted(adaptive_gpu_tiles)),
                    "adaptive_factor_groups": len(adaptive_gpu_tiles),
                    "adaptive_cuda_no_fallback": True,
                })
            bundle.metadata["probe_trajectory_factor_tiles"] = executor.probe_tiles_processed
            if gpu_risk_results:
                bundle.metadata.update({"calendar_rolling_planning_backend": "cpu",
                    "calendar_rolling_reduction_backend": "cuda_strict",
                    "calendar_rolling_cuda_no_fallback": True,
                    "calendar_rolling_factor_tiles": executor.risk_tiles_processed,
                    "probe_trajectory_factor_tiles": executor.probe_tiles_processed,
                    "calendar_rolling_trajectory_source": risk_trajectory_source})
            for mid in metric_ids:
                canonical = _resolve_alias(mid)
                if canonical != mid:
                    for field in ("scalar_metrics", "series_metrics", "vector_metrics", "observation_counts"):
                        group = getattr(bundle, field)
                        if canonical in group:
                            group[mid] = group[canonical]
        gpu_result = bundle

    metric_ids = tuple(metric_ids)
    metric_parameters = dict(metric_parameters or {})
    if set(metric_parameters) - set(metric_ids):
        raise InvalidContractError("metric_parameters contains an unrequested metric")
    if gpu_result is None:
        if holding_returns is not None:
            from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl
            from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
            options = {k: v for k, v in portfolio_spec.to_dict().items() if k not in {"purpose", "missing_return_policy"}}
            pnl = np.full((factor_batch.num_times, factor_batch.num_factors), np.nan)
            for f in range(factor_batch.num_factors):
                values = factor_batch.values[:, :, f]
                if factor_batch.validity is not None:
                    values = np.where(factor_batch.validity[:, :, f], values, np.nan)
                result = compute_cohort_pnl(values, holding_returns.values,
                    np.ones(holding_returns.values.shape), require_tradable=False,
                    trade_eligibility=trade_eligibility, **options)
                pnl[:, f] = result["pnl_net"]
            portfolio_returns = ProbePortfolioArtifact(pnl,
                time_index=tuple(label_bundle.observation_time or label_bundle.decision_time),
                factor_ids=tuple(factor_batch.factor_ids),
                provenance={"holding_return_ref": holding_returns.content_hash,
                            "portfolio_spec_hash": portfolio_spec.content_hash,
                            "purpose": "RESEARCH_PROBE", "execution_certified": False,
                            "tradability_assumption": "side_specific" if trade_eligibility is not None else "unrestricted_research",
                            "trade_eligibility_ref": trade_eligibility.content_hash if trade_eligibility is not None else None})
        metric_specs = []
        # Wrapper functions injected into the runtime so that derived registry
        # metrics that consume a formal ICSeriesArtifact (a derived metric's
        # ``requires`` now names the artifact class, e.g. ["ICSeriesArtifact"])
        # can be computed from factor_batch + label_bundle without the caller
        # needing to supply the IC series explicitly.  The wrapper only adapts
        # inputs — the registry compute_fn remains the single source of truth
        # for the metric value.
        facade_ic_wrappers: Dict[str, Callable] = {}
        # QE-R2: returns-panel metrics (long_short_returns / sharpe_ratio /
        # sortino_ratio / win_rate) bind panel-shaped parameters that the generic
        # runtime argument binder cannot supply; the facade wraps their registry
        # compute_fn to derive the return series from the factor/label panels.
        facade_panel_wrappers: Dict[str, Callable] = {}
        facade_exposure_wrappers: Dict[str, Callable] = {}
        facade_pvalue_wrappers: Dict[str, Callable] = {}
        quantile_daily_cache = {}
        quantile_profile_cache = {}
        exposure_loading_cache = {}

        # QE-METRIC-P0-02: the registry is the single truth. No local closures
        # shadow it anymore — every requested metric must resolve through
        # get_metric() (with canonical dotted-alias support); its compute_fn is
        # executed by the runtime and the per-factor result is adapted below.
        # Metrics that declare ICSeriesArtifact in ``requires`` are wrapped with
        # a thin input adapter so the public facade can satisfy them from
        # factor_batch + label_bundle; the runtime still calls through to the
        # registry compute_fn, which remains the single source of truth for the
        # metric value.
        if adaptive_resolution_artifact is not None:
            runtime.register_metric(
                "adaptive_quantile_count",
                lambda **_kwargs: adaptive_resolution_artifact,
            )
        for metric_id in metric_ids:
            try:
                spec = get_metric(_resolve_alias(metric_id))
            except KeyError:
                raise UnsupportedMetricError(
                    f"Public evaluate does not support metric '{metric_id}'"
                ) from None
            parameters = dict(metric_parameters.get(metric_id, {}))
            if _resolve_alias(metric_id) in generalization_ids:
                from quant_evaluator.metrics.generalization_evidence import project_generalization_metric
                def _generalization_wrapper(factor_batch=None,label_bundle=None,_mid=_resolve_alias(metric_id),_version=spec.metric_version,**kwargs):
                    return project_generalization_metric(generalization_evidence,_mid,
                        expected_factor_ids=factor_batch.factor_ids,expected_factor_versions=generalization_versions,
                        producer_version=_version)
                runtime.register_metric(metric_id,_generalization_wrapper)
                from quant_evaluator.contracts._hashutil import stable_content_hex
                metric_specs.append({"metric_id":metric_id,"metric_kind":"custom",
                    "metadata":{"parameters":parameters,"generalization_evidence_hash":
                        stable_content_hex(tag="TrainVsValidationArtifact.v2",fields=generalization_evidence.to_dict())}})
                continue
            if adaptive_resolution_artifact is not None and _resolve_alias(metric_id) in requested_adaptive_profiles:
                # Ragged per-factor Q axes are built once by Q-group below and
                # transported through EvaluationBundle.factor_artifacts.
                continue
            if parameters:
                signature = inspect.signature(spec.compute_fn)
                forbidden = {"factor_batch", "label_bundle", "computed_metrics", "metadata",
                             "factor_values", "forward_returns", "returns", "validity_mask"}
                unknown = set(parameters) - set(signature.parameters)
                if unknown or set(parameters) & forbidden:
                    raise InvalidContractError(f"Invalid parameters for {metric_id}: {sorted(unknown | (set(parameters) & forbidden))}")
            metric_specs.append({"metric_id": metric_id, "metric_kind": "custom",
                                 "metadata": {"parameters": parameters}})
            if "exposure_panel" in (spec.requires or ()):
                from quant_evaluator.contracts._hashutil import stable_content_hex
                metric_specs[-1]["metadata"]["exposure_panel_hash"]=stable_content_hex(
                    tag="SecurityExposurePanel.v2",fields=exposure_panel.to_dict())
            if "QuantileReturnArtifact" in (spec.requires or []):
                def make_quantile_wrapper(cfn, min_periods, windowed):
                    def wrapper(factor_batch=None, label_bundle=None, **kwargs):
                        from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
                        q = quantile_builder_parameters.get("n_quantiles",5)
                        n = quantile_builder_parameters.get("min_assets",10)
                        window = quantile_builder_parameters.get("window_size",20)
                        daily_key = (q, n)
                        if daily_key not in quantile_daily_cache:
                            quantile_daily_cache[daily_key], _ = compute_quantile_returns_fast(
                                factor_batch, label_bundle, n_quantiles=q, min_assets=n)
                        daily = quantile_daily_cache[daily_key]
                        key = (daily_key, min_periods, window if windowed else None)
                        if key not in quantile_profile_cache:
                            if windowed:
                                panels = daily[:len(daily)//window*window].reshape(-1,window,q,factor_batch.num_factors)
                                counts = np.isfinite(panels).sum(axis=1)
                                means = np.nansum(panels,axis=1)/np.maximum(counts,1)
                                quantile_profile_cache[key] = np.where(counts == window, means, np.nan)
                            else:
                                days = np.isfinite(daily).sum(axis=0)
                                means = np.nansum(daily,axis=0)/np.maximum(days,1)
                                quantile_profile_cache[key] = np.where(days >= min_periods, means, np.nan)
                        accepted = inspect.signature(cfn).parameters
                        return cfn(quantile_profile_cache[key], **{k:v for k,v in kwargs.items() if k in accepted})
                    return wrapper
                windowed = _resolve_alias(metric_id) in {"shape_stability", "shape_regime_stability", "shape_bootstrap_confidence", "shape_bootstrap_rank_agreement"}
                runtime.register_metric(metric_id, make_quantile_wrapper(spec.compute_fn, spec.min_periods or 20, windowed))
            # QE-R2: returns-based portfolio metrics (long/short backtest family)
            # consume the raw (T, N) forward-return panel — not a per-factor
            # reduction.  Declare full-batch so the chunker never slices the
            # factor/return panel for these metrics, and inject a thin input
            # adapter so the runtime's generic argument binding satisfies their
            # panel-shaped parameters.  The registry compute_fn remains the single
            # source of truth for the metric value.
            if _resolve_alias(metric_id) in _RETURNS_PANEL_METRIC_IDS:
                metric_specs[-1]["requires_full_batch"] = True
                facade_panel_wrappers[metric_id] = _make_panel_wrapper(spec.compute_fn, portfolio_returns, calendar_snapshot)
            if "ICSeriesArtifact" in (spec.requires or []):
                compute_fn = spec.compute_fn
                # QE-P1-27: the wrapper IC series must follow the metric —
                # rank-family specs declare ic_method="spearman", everything
                # else defaults to "pearson".
                wrapper_ic_method = getattr(spec, "ic_method", "pearson")
                def _make_ic_wrapper(cfn: Callable, ic_method: str) -> Callable:
                    def _ic_wrapper(factor_batch=None, label_bundle=None, **kwargs):
                        from quant_evaluator.metrics.ic import compute_daily_ic

                        try:
                            fn_sig = inspect.signature(cfn)
                        except (TypeError, ValueError):
                            filtered = dict(kwargs)
                        else:
                            if any(
                                p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in fn_sig.parameters.values()
                            ):
                                filtered = dict(kwargs)
                            else:
                                filtered = {
                                    k: v
                                    for k, v in kwargs.items()
                                    if k in fn_sig.parameters
                                }
                        min_assets = kwargs.get("min_assets", 20)
                        key = (ic_method, min_assets)
                        if key not in ic_series_cache:
                            ic_series_cache[key], _ = compute_daily_ic(
                                factor_batch, label_bundle, method=ic_method, min_assets=min_assets)
                        ic_series = ic_series_cache[key]
                        return cfn(ic_series, **filtered)
                    return _ic_wrapper
                facade_ic_wrappers[metric_id] = _make_ic_wrapper(
                    compute_fn, wrapper_ic_method
                )
                runtime.register_metric(metric_id, facade_ic_wrappers[metric_id])
            if "exposure_panel" in (spec.requires or []):
                def _make_exposure_wrapper(cfn: Callable, canonical_id: str) -> Callable:
                    def _exposure_wrapper(factor_batch=None, label_bundle=None, **kwargs):
                        from quant_evaluator.metrics.exposure_evidence import build_factor_loading_series

                        raw_exposures = np.where(
                            exposure_panel.validity, exposure_panel.values, np.nan
                        ) if exposure_panel.validity is not None else exposure_panel.values
                        accepted = inspect.signature(cfn).parameters
                        filtered = {k: v for k, v in kwargs.items() if k in accepted}
                        outputs = []
                        for index, factor_id in enumerate(factor_batch.factor_ids):
                            factor_values = factor_batch.values[:, :, index]
                            if factor_batch.validity is not None:
                                factor_values = np.where(
                                    factor_batch.validity[:, :, index], factor_values, np.nan
                                )
                            if canonical_id in {"neutralized_rank_ic", "residual_rank_ic"}:
                                forward_returns = label_bundle.values
                                if label_bundle.validity is not None:
                                    forward_returns = np.where(
                                        label_bundle.validity, forward_returns, np.nan
                                    )
                                outputs.append(cfn(
                                    factor_values, forward_returns, exposure_panel, **filtered
                                ))
                                continue
                            key = (index, kwargs.get("min_obs", 10))
                            if key not in exposure_loading_cache:
                                exposure_loading_cache[key] = build_factor_loading_series(
                                    exposure_panel, factor_values, factor_id=factor_id,
                                    min_obs=key[1],
                                )
                            value = cfn(exposure_loading_cache[key], **filtered)
                            if isinstance(value, dict):
                                value = value.get("absolute_mean", np.nan)
                            outputs.append(value)
                        return np.asarray(outputs, dtype=np.float64)
                    return _exposure_wrapper
                facade_exposure_wrappers[metric_id] = _make_exposure_wrapper(
                    spec.compute_fn, _resolve_alias(metric_id)
                )
                runtime.register_metric(metric_id, facade_exposure_wrappers[metric_id])
            if "p_values" in (spec.requires or []):
                def _make_pvalue_wrapper(cfn: Callable) -> Callable:
                    def _pvalue_wrapper(factor_batch=None, label_bundle=None, **kwargs):
                        from quant_evaluator.metrics.ic import compute_daily_ic
                        from quant_evaluator.metrics.registry_adapters import compute_hac_pvalue_value

                        key = ("spearman", 20)
                        if key not in ic_series_cache:
                            ic_series_cache[key], _ = compute_daily_ic(
                                factor_batch, label_bundle,
                                method="spearman", min_assets=20,
                            )
                        p_values = compute_hac_pvalue_value(
                            ic_series_cache[key], min_periods=30,
                            max_lag=5, kernel="bartlett",
                        )
                        accepted = inspect.signature(cfn).parameters
                        filtered = {k: v for k, v in kwargs.items() if k in accepted}
                        corrected = cfn(p_values, **filtered)
                        # All registered correction kernels return adjusted
                        # p-values first; rejection masks/counts are policy
                        # diagnostics, not scalar metric values.
                        return np.asarray(corrected[0], dtype=np.float64)
                    return _pvalue_wrapper
                facade_pvalue_wrappers[metric_id] = _make_pvalue_wrapper(spec.compute_fn)
                runtime.register_metric(metric_id, facade_pvalue_wrappers[metric_id])

        # QE-R2: register the returns-panel adapters so the runtime resolves the
        # panel-shaped compute_fn through its registered callables (same route as
        # the ICSeriesArtifact wrappers above).
        for metric_id, wrapper in facade_panel_wrappers.items():
            runtime.register_metric(metric_id, wrapper)

        # Bind effective parameters once through the same registered callable used
        # by the runtime. Parameters also enter node metadata and cache identity.
        for metric_id in metric_ids:
            parameters = dict(metric_parameters.get(metric_id, {}))
            fn = runtime._resolve_metric_function(metric_id)
            fn = getattr(fn, "__qe_parameter_base__", fn)
            runtime.register_metric(metric_id, fn)
            if parameters:
                def bind(fn, parameters):
                    def parameterized(**kwargs):
                        kwargs.update(parameters)
                        signature = inspect.signature(fn)
                        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
                        return fn(**kwargs)
                    parameterized.__qe_parameter_base__ = fn
                    return parameterized
                runtime.register_metric(metric_id, bind(fn, parameters))

        result = runtime.evaluate(factor_batch, label_bundle, metric_specs, use_chunking=False)
    else:
        result = gpu_result

    # Backend-independent provenance is computed from the host contracts after
    # execution (and before envelope construction).  CUDA metadata therefore
    # carries the same authoritative value/validity identities as CPU.
    authoritative_provenance = runtime._provenance(factor_batch, label_bundle)
    result.metadata["provenance"] = authoritative_provenance

    from quant_evaluator.contracts.metric_artifacts import (
        ScalarMetricArtifact, SeriesMetricArtifact, VectorMetricArtifact,
    )
    from quant_evaluator.contracts.axis_refs import FactorAxisRef, TimeAxisRef, QuantileAxisRef
    from quant_evaluator.contracts._hashutil import stable_content_hex

    versions = {mid: get_metric(_resolve_alias(mid)).metric_version for mid in metric_ids}
    config_hash = stable_content_hex(tag="EvaluationConfig.v2", fields={
        "metrics": metric_ids, "versions": versions, "parameters": metric_parameters,
        "label_hash": label_bundle.content_hash,
        "factor_ids": factor_batch.factor_ids,
        "factor_values": factor_batch.values,
        "factor_validity": factor_batch.validity,
        "context": context,
        "quantile_builder_parameters": quantile_builder_parameters,
        "calendar_snapshot": None if calendar_snapshot is None else {
            "id": calendar_snapshot.snapshot_id, "market": calendar_snapshot.market,
            "timezone": calendar_snapshot.timezone, "source": calendar_snapshot.source_version,
            "trading_days": calendar_snapshot.trading_days, "sessions": calendar_snapshot.sessions,
            "early_close": calendar_snapshot.early_close},
        "portfolio_returns": portfolio_returns.to_dict() if portfolio_returns is not None and holding_returns is None else None,
        "holding_returns": holding_returns.content_hash if holding_returns is not None else None,
        "portfolio_spec": portfolio_spec.to_dict() if portfolio_spec is not None else None,
        "trade_eligibility": trade_eligibility.content_hash if trade_eligibility is not None else None,
        "exposure_panel": exposure_panel.to_dict() if exposure_panel is not None else None,
        "generalization_evidence": generalization_evidence.to_dict() if generalization_evidence is not None else None,
        "split_ref": split_ref.to_dict() if split_ref is not None else None,
        **request_fields,
    })
    artifacts = {}
    factor_artifacts = {fid: {} for fid in factor_batch.factor_ids}
    if adaptive_resolution_artifact is not None:
        from dataclasses import replace
        from quant_evaluator.contracts.artifact_types import DailyQuantileReturnArtifact

        q_groups = {}
        for index, q in enumerate(adaptive_quantile_binding["selected_q_by_factor"]):
            q_groups.setdefault(q, []).append(index)
        for requested_id in metric_ids:
            canonical_profile = _resolve_alias(requested_id)
            if canonical_profile not in requested_adaptive_profiles:
                continue
            spec = get_metric(canonical_profile)
            base_parameters = dict(metric_parameters.get(requested_id, {}))
            for q, indices in q_groups.items():
                tile_ids = tuple(factor_batch.factor_ids[index] for index in indices)
                tile = FactorBatch(
                    factor_ids=tile_ids,
                    time_axis=factor_batch.time_axis,
                    asset_axis=factor_batch.asset_axis,
                    values=factor_batch.values[:, :, indices],
                    validity=None if factor_batch.validity is None else factor_batch.validity[:, :, indices],
                    context_refs=factor_batch.context_refs,
                )
                accepted = inspect.signature(spec.compute_fn).parameters
                parameters = {key: value for key, value in base_parameters.items() if key in accepted}
                parameters["n_quantiles"] = q
                if q in adaptive_gpu_tiles:
                    gpu_tile = adaptive_gpu_tiles[q]
                    gpu_values = gpu_tile.vector_metrics[canonical_profile]
                    if canonical_profile == "quantile_returns_daily":
                        gpu_counts = gpu_tile.observation_counts[canonical_profile]
                        raw_tile = DailyQuantileReturnArtifact(
                            values=gpu_values,
                            counts=gpu_counts,
                            valid_mask=np.isfinite(gpu_values),
                            time_axis=tuple(label_bundle.observation_time or label_bundle.decision_time),
                            quantile_axis=tuple(range(q)),
                            factor_axis=tile_ids,
                            metric_id=requested_id,
                            producer_version=versions[requested_id],
                            provenance={"execution_backend": "cuda_strict", "no_fallback": True},
                        )
                    else:
                        raw_tile = gpu_values
                else:
                    raw_tile = spec.compute_fn(tile, label_bundle, **parameters)
                binding = dict(adaptive_quantile_binding) | {
                    "selected_q_by_factor": tuple(q for _ in tile_ids),
                    "factor_ids": tile_ids,
                }
                provenance = {
                    "config_hash": config_hash,
                    "metric_version": versions[requested_id],
                    "parameters": dict(parameters),
                    "adaptive_quantile_binding": binding,
                }
                if q in adaptive_gpu_tiles:
                    provenance.update({"execution_backend": "cuda_strict", "no_fallback": True})
                if isinstance(raw_tile, DailyQuantileReturnArtifact):
                    group_artifact = replace(
                        raw_tile, metric_id=requested_id,
                        producer_version=versions[requested_id],
                        provenance=dict(raw_tile.provenance) | provenance,
                    )
                    for local_index, fid in enumerate(tile_ids):
                        factor_artifacts[fid][requested_id] = replace(
                            group_artifact,
                            values=group_artifact.values[:, :, local_index:local_index + 1],
                            counts=group_artifact.counts[:, :, local_index:local_index + 1],
                            valid_mask=group_artifact.valid_mask[:, :, local_index:local_index + 1],
                            factor_axis=(fid,),
                            provenance=dict(group_artifact.provenance) | {
                                "adaptive_quantile_binding": dict(binding) | {
                                    "selected_q_by_factor": (q,), "factor_ids": (fid,)}},
                        )
                else:
                    tile_values = np.asarray(raw_tile, dtype=np.float64)
                    group_artifact = VectorMetricArtifact(
                        metric_id=requested_id,
                        domain=str(getattr(spec.domain, "value", spec.domain) or "metric"),
                        values=tile_values,
                        factor_axis=FactorAxisRef(tile_ids),
                        quantile_axis=QuantileAxisRef(tuple(f"Q{i+1}" for i in range(q))),
                        producer_version=versions[requested_id],
                        provenance=provenance,
                    )
                    for local_index, fid in enumerate(tile_ids):
                        factor_artifacts[fid][requested_id] = VectorMetricArtifact(
                            metric_id=requested_id,
                            domain=group_artifact.domain,
                            values=tile_values[:, local_index:local_index + 1],
                            factor_axis=FactorAxisRef((fid,)),
                            quantile_axis=group_artifact.quantile_axis,
                            producer_version=group_artifact.producer_version,
                            provenance=dict(group_artifact.provenance) | {
                                "adaptive_quantile_binding": dict(binding) | {
                                    "selected_q_by_factor": (q,), "factor_ids": (fid,)}},
                        )
                if len(q_groups) == 1:
                    artifacts[requested_id] = group_artifact
    grouped_metrics = {fid: {} for fid in factor_batch.factor_ids}
    time_index = tuple(label_bundle.observation_time or label_bundle.decision_time)
    for metric_id in metric_ids:
        spec = get_metric(_resolve_alias(metric_id))
        if adaptive_resolution_artifact is not None and _resolve_alias(metric_id) in requested_adaptive_profiles:
            continue
        raw = (gpu_risk_results[_resolve_alias(metric_id)]["values"]
               if _resolve_alias(metric_id) in gpu_risk_results else
               adaptive_resolution_artifact if
               adaptive_resolution_artifact is not None and _resolve_alias(metric_id) == "adaptive_quantile_count"
               else result.get_metric(metric_id))
        from quant_evaluator.contracts.artifact_types import DailyQuantileReturnArtifact
        if gpu_result is not None and _resolve_alias(metric_id) == "quantile_returns_daily":
            params = metric_parameters.get(metric_id, {})
            raw = DailyQuantileReturnArtifact(values=raw,
                counts=gpu_result.observation_counts[metric_id], valid_mask=np.isfinite(raw),
                time_axis=time_index, quantile_axis=tuple(range(raw.shape[1])),
                factor_axis=tuple(factor_batch.factor_ids),
                tie_status_ref=params.get("tie_status_ref"), tradability_ref=params.get("tradability_ref"),
                risk_exposure_ref=params.get("risk_exposure_ref"), producer_version=versions[metric_id],
                metric_id=metric_id)
        if isinstance(raw, DailyQuantileReturnArtifact):
            from dataclasses import replace
            if tuple(raw.factor_axis) != tuple(factor_batch.factor_ids) or tuple(raw.time_axis) != time_index:
                raise InvalidContractError("Daily quantile artifact axes do not match request")
            adaptive_provenance = (
                {"adaptive_quantile_binding": adaptive_quantile_binding}
                if adaptive_quantile_binding is not None and _resolve_alias(metric_id) in adaptive_profile_ids
                else {}
            )
            artifacts[metric_id] = replace(raw, metric_id=metric_id, producer_version=versions[metric_id],
                provenance=dict(raw.provenance) | adaptive_provenance | {"config_hash": config_hash,
                    "split_ref": split_ref.to_dict() if split_ref is not None else None,
                    "factor_value_bytes_hash": authoritative_provenance["factor_value_bytes_hash"],
                    "factor_validity_hash": authoritative_provenance["factor_validity_hash"]})
            continue
        supplied_artifact = raw if isinstance(raw, ScalarMetricArtifact) else None
        if supplied_artifact is not None:
            if tuple(supplied_artifact.factor_axis.factor_ids)!=tuple(factor_batch.factor_ids):
                raise InvalidContractError("Supplied scalar evidence factor axis differs from request")
            raw = supplied_artifact.values
        if _resolve_alias(metric_id) == "long_short_returns":
            raw = raw[2]
        if _resolve_alias(metric_id) == "max_drawdown" and gpu_result is None:
            raw = raw[0]
        values = np.asarray(raw, dtype=np.float64)
        common = dict(metric_id=metric_id, domain=str(getattr(spec.domain, "value", spec.domain) or "metric"),
                      factor_axis=(supplied_artifact.factor_axis if supplied_artifact is not None else FactorAxisRef(tuple(factor_batch.factor_ids))),
                      producer_version=versions[metric_id],
                      provenance={"config_hash": config_hash, "metric_version": versions[metric_id],
                                  "parameters": dict(metric_parameters.get(metric_id, {})),
                                  "factor_value_bytes_hash": authoritative_provenance["factor_value_bytes_hash"],
                                  "factor_validity_hash": authoritative_provenance["factor_validity_hash"]})
        if "exposure_panel" in (spec.requires or ()):
            from quant_evaluator.contracts._ndarray_codec import decode_value
            common["provenance"].update({
                "estimation_scope":"SAME_DATE_DESCRIPTIVE",
                "method_version":"factor_standardized_wls.v2",
                "loading_definition":"beta_k * weighted_sd(risk_k) / weighted_sd(factor)",
                "purity_definition":"time_mean(1 - same_fit_weighted_r_squared)",
                "exposure_source_ref":exposure_panel.source_ref,
                "exposure_provider":exposure_panel.provider,
                "weight_ref":exposure_panel.weight_ref or "equal_weight",
                "universe_snapshot_ref":exposure_panel.universe_snapshot_ref,
            })
        if supplied_artifact is not None:
            common["provenance"] = dict(supplied_artifact.provenance) | common["provenance"]
        if canonical := _resolve_alias(metric_id):
            if canonical in gpu_risk_results:
                risk = gpu_risk_results[canonical]
                common["provenance"].update(risk["provenance"] | {
                    "execution_backend": "cuda_strict", "planning_backend": "cpu",
                    "no_fallback": True, "trajectory_source": risk_trajectory_source})
            elif (gpu_result is not None and
                  "exposure_panel" in (spec.requires or ())):
                common["provenance"].update({
                    "execution_backend": "cuda_strict",
                    "planning_backend": "cpu",
                    "no_fallback": True,
                    "exposure_source_ref": exposure_panel.source_ref,
                    "exposure_provider": exposure_panel.provider,
                    "universe_snapshot_ref": exposure_panel.universe_snapshot_ref,
                    "exposure_kernel_dispatches": gpu_result.metadata.get(
                        "exposure_kernel_dispatches", 0),
                })
                if canonical.startswith("worst_calendar_"):
                    period_rows = []
                    for p, (period_id, expected, observed, bracketed, complete) in enumerate(risk["period_plan"]):
                        finite_counts = risk["finite_counts"][p]
                        compounded = risk["period_values"][p]
                        included = np.isfinite(compounded) & risk["structural_include"][p]
                        period_rows.append({"period_id": period_id,
                            "expected_session_count": expected, "observed_session_count": observed,
                            "finite_return_counts": tuple(int(x) for x in finite_counts),
                            "calendar_coverage_bracketed": bracketed, "sessions_complete": complete,
                            "complete_by_factor": tuple(bool(complete and bracketed and x == expected) for x in finite_counts),
                            "partial": not bool(complete and bracketed),
                            "included_by_factor": tuple(bool(x) for x in included),
                            "compounded_returns": tuple(float(x) if np.isfinite(x) else None for x in compounded)})
                    common["provenance"]["period_rows"] = tuple(period_rows)
        if adaptive_quantile_binding is not None and _resolve_alias(metric_id) in adaptive_profile_ids:
            common["provenance"]["adaptive_quantile_binding"] = adaptive_quantile_binding
        if "QuantileReturnArtifact" in (spec.requires or []):
            common["provenance"]["quantile_builder"] = {"n_quantiles":5,"min_assets":10,"window_size":20} | quantile_builder_parameters
        if "p_values" in (spec.requires or []):
            common["provenance"]["p_value_builder"] = {
                "source_statistic": "daily_spearman_rank_ic",
                "test": "two_sided_hac_mean_not_zero",
                "sample_unit": "daily_ic",
                "min_assets": 20,
                "min_periods": 30,
                "hac_kernel": "bartlett",
                "hac_max_lag": 5,
                "family_axis": "factor",
            }
        canonical = _resolve_alias(metric_id)
        if canonical == "shape_stability":
            common["provenance"]["reference_policy"] = "leave_one_window_out"
            common["provenance"]["aggregation_scale"] = "inverse_fisher_correlation"
        elif canonical == "shape_regime_stability":
            common["provenance"]["reference_policy"] = "consecutive_disjoint_windows_not_economic_regime_labels"
            common["provenance"]["aggregation_scale"] = "inverse_fisher_correlation"
        elif canonical in {"shape_bootstrap_confidence", "shape_bootstrap_rank_agreement"}:
            common["provenance"]["event"] = "spearman_rank_agreement_with_observed_mean_profile"
            common["provenance"]["resampling"] = "moving_blocks"
            common["provenance"]["probability_of_shape_family"] = False
            defaults = {name:p.default for name,p in inspect.signature(spec.compute_fn).parameters.items() if p.default is not inspect.Parameter.empty}
            common["provenance"]["resampling_parameters"] = defaults | dict(metric_parameters.get(metric_id,{}))
        definition_fields = {"risk_free_rate", "periods_per_year", "annualization",
                             "downside_denominator", "mar", "missing_return_policy"}
        effective_definition = {name: parameter.default
            for name, parameter in inspect.signature(spec.compute_fn).parameters.items()
            if name in definition_fields and parameter.default is not inspect.Parameter.empty}
        effective_definition.update({name: value for name, value in metric_parameters.get(metric_id, {}).items()
                                     if name in definition_fields})
        if effective_definition:
            common["provenance"]["effective_definition"] = effective_definition
        if _resolve_alias(metric_id) in _RETURNS_PANEL_METRIC_IDS:
            common["provenance"].update({
                "portfolio_purpose": "RESEARCH_PROBE", "execution_certified": False,
                "tradability_assumption": "side_specific" if trade_eligibility is not None else "unrestricted_research",
                "trade_eligibility_ref": trade_eligibility.content_hash if trade_eligibility is not None else None,
            })
        if values.ndim == 1 and values.shape == (factor_batch.num_factors,):
            if canonical in gpu_risk_results:
                counts = gpu_risk_results[canonical]["observation_counts"]
            elif supplied_artifact is not None and "observation_counts" in supplied_artifact.provenance:
                counts = supplied_artifact.provenance["observation_counts"]
            elif "p_values" in (spec.requires or []):
                counts = np.sum(np.isfinite(ic_series_cache[("spearman", 20)]), axis=0)
            elif gpu_result is not None:
                if metric_id not in gpu_result.observation_counts:
                    raise UnsupportedMetricError(f"GPU metric {metric_id} lacks observation-count evidence")
                counts = gpu_result.observation_counts[metric_id]
                if "exposure_panel" in (spec.requires or ()):
                    from quant_evaluator.metrics.exposure_evidence import FactorLoadingSeries
                    minimum=metric_parameters.get(metric_id,{}).get("min_obs",10)
                    prefix=f"_exposure_{minimum}_"
                    ev=gpu_result.vector_metrics
                    weights=exposure_panel.regression_weights
                    if weights is None: weights=np.ones(exposure_panel.values.shape[:2])
                    risk_valid=np.isfinite(exposure_panel.values)
                    if exposure_panel.validity is not None: risk_valid &= exposure_panel.validity
                    per_factor=[]
                    for i,fid in enumerate(factor_batch.factor_ids):
                        fv=factor_batch.values[:,:,i]
                        if factor_batch.validity is not None: fv=np.where(factor_batch.validity[:,:,i],fv,np.nan)
                        joint=np.isfinite(fv)&risk_valid.all(axis=2)&(weights>0)
                        diagnostics=tuple({"n":int(ev[prefix+"counts"][t,i]),
                            "rank":int(ev[prefix+"rank"][t,i]),
                            "effective_df":int(ev[prefix+"effective_df"][t,i]),
                            "estimation_scope":"SAME_DATE_DESCRIPTIVE",
                            "method_version":"centered_wls_svd.v1"} for t in range(len(fv)))
                        s=FactorLoadingSeries(ev[prefix+"values"][:,:,i],ev[prefix+"raw_loadings"][:,:,i],
                            ev[prefix+"r_squared"][:,i],ev[prefix+"counts"][:,i],exposure_panel.style_names,
                            fid,exposure_panel.source_ref,exposure_panel.provider,exposure_panel.date_index,
                            "support:"+hashlib.sha256(joint.tobytes()+weights.tobytes()).hexdigest(),diagnostics,
                            "factor-values:"+hashlib.sha256(np.ascontiguousarray(fv).tobytes()).hexdigest(),
                            exposure_panel.weight_ref or "equal_weight")
                        per_factor.append(decode_value(s.to_dict()))
                    common["provenance"]["factor_loading_series"]=per_factor
            elif "exposure_panel" in (spec.requires or ()) and canonical not in {"neutralized_rank_ic","residual_rank_ic"}:
                minimum=metric_parameters.get(metric_id,{}).get("min_obs",10)
                from quant_evaluator.metrics.exposure_evidence import build_factor_loading_series
                for i,fid in enumerate(factor_batch.factor_ids):
                    if (i,minimum) not in exposure_loading_cache:
                        fv=factor_batch.values[:,:,i]
                        if factor_batch.validity is not None: fv=np.where(factor_batch.validity[:,:,i],fv,np.nan)
                        exposure_loading_cache[(i,minimum)]=build_factor_loading_series(
                            exposure_panel,fv,factor_id=fid,min_obs=minimum)
                loading_series=[exposure_loading_cache[(i,minimum)] for i in range(factor_batch.num_factors)]
                common["provenance"]["factor_loading_series"]=[decode_value(s.to_dict()) for s in loading_series]
                counts=[]
                for s in loading_series:
                    if canonical=="purity_ratio": n=np.isfinite(s.r_squared).sum()
                    elif canonical=="exposure_drift": n=(np.isfinite(s.values[1:])&np.isfinite(s.values[:-1])).any(axis=1).sum()
                    elif canonical=="max_absolute_style_exposure":
                        from quant_evaluator.metrics.exposure_evidence import compute_max_absolute_style_exposure
                        n=compute_max_absolute_style_exposure(s,min_finite=metric_parameters.get(metric_id,{}).get("min_finite",5))["counts"]
                    else:
                        style=canonical.removesuffix("_exposure")
                        n=np.isfinite(s.values[:,s.style_names.index(style)]).sum() if style in s.style_names else 0
                    counts.append(int(n))
            elif _resolve_alias(metric_id).startswith("worst_rolling_") and portfolio_returns is not None:
                window = int(_resolve_alias(metric_id).rsplit("_", 1)[1][:-1])
                panel = portfolio_returns.values
                counts = np.zeros(factor_batch.num_factors, dtype=np.int64)
                if len(panel) >= window:
                    windows = np.lib.stride_tricks.sliding_window_view(panel, window, axis=0)
                    counts = np.isfinite(windows).all(axis=-1).sum(axis=0)
            elif _resolve_alias(metric_id) in _RETURNS_PANEL_METRIC_IDS and portfolio_returns is not None:
                counts = np.isfinite(portfolio_returns.values).sum(axis=0)
            elif canonical in {"quantile_spread", "quantile_monotonicity"}:
                from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
                q = (quantile_builder_parameters.get("n_quantiles",5) if canonical=="quantile_monotonicity"
                     else metric_parameters.get(metric_id,{}).get("n_quantiles",5))
                minimum = quantile_builder_parameters.get("min_assets",10) if canonical=="quantile_monotonicity" else 10
                daily_key=(q,minimum)
                if daily_key not in quantile_daily_cache:
                    quantile_daily_cache[daily_key],_=compute_quantile_returns_fast(
                        factor_batch,label_bundle,n_quantiles=q,min_assets=minimum)
                daily=quantile_daily_cache[daily_key]
                if canonical=="quantile_spread":
                    counts=np.isfinite(daily[:,-1,:]-daily[:,0,:]).sum(axis=0)
                else:
                    days=np.isfinite(daily).sum(axis=0)
                    profile=np.where(days >= (spec.min_periods or 20),np.nansum(daily,axis=0)/np.maximum(days,1),np.nan)
                    counts=(np.isfinite(profile[:-1]) & np.isfinite(profile[1:])).sum(axis=0)
            else:
                counts = _per_factor_observation_counts(metric_id, factor_batch, label_bundle, values,
                                                        metric_parameters.get(metric_id), ic_series_cache)
            canonical = _resolve_alias(metric_id)
            sample_unit = "finite_metric_value"
            if supplied_artifact is not None and "sample_unit" in supplied_artifact.provenance:
                sample_unit=supplied_artifact.provenance["sample_unit"]
            elif "exposure_panel" in (spec.requires or ()) and canonical not in {"neutralized_rank_ic","residual_rank_ic"}:
                sample_unit="valid_loading_transition" if canonical=="exposure_drift" else "valid_regression_date"
            elif canonical == "coverage":
                sample_unit = "factor_label_pair"
            elif "ICSeriesArtifact" in (spec.requires or []) or canonical in {"pearson_ic", "rank_ic", "mean_ic"}:
                sample_unit = "daily_ic"
            elif "p_values" in (spec.requires or []):
                sample_unit = "daily_ic"
            elif canonical in {"turnover", "factor_turnover_rate"}:
                sample_unit = "valid_transition"
            elif canonical == "quantile_spread":
                sample_unit = "daily_quantile_spread"
            elif canonical == "quantile_monotonicity":
                sample_unit = "valid_adjacent_profile_pair"
            elif canonical == "daily_quantile_monotonicity_rate":
                sample_unit = "valid_daily_quantile_shape"
            elif canonical.startswith("worst_calendar_"):
                sample_unit = "included_calendar_period"
            elif canonical.startswith("worst_rolling_"):
                sample_unit = "mature_rolling_window"
            elif canonical in _RETURNS_PANEL_METRIC_IDS:
                sample_unit = "return_interval"
            common["provenance"]["sample_unit"] = sample_unit
            common["provenance"]["observation_counts"] = tuple(int(n) for n in counts)
            artifacts[metric_id] = ScalarMetricArtifact(values=values, **common)
            for index, fid in enumerate(factor_batch.factor_ids):
                numeric = float(values[index])
                grouped_metrics[fid][metric_id] = MetricValue(
                    metric_id=metric_id, value=numeric if np.isfinite(numeric) else None,
                    valid=bool(np.isfinite(numeric)), observation_count=int(counts[index]),
                    metric_version=versions[metric_id],
                    sample_unit=sample_unit,
                    warnings=() if np.isfinite(numeric) else ("non-finite result",))
        elif values.ndim == 2 and values.shape[1] == factor_batch.num_factors:
            canonical = _resolve_alias(metric_id)
            if canonical == "quantile_returns_full" or spec.output_type == "vector":
                artifacts[metric_id] = VectorMetricArtifact(
                    values=values, quantile_axis=QuantileAxisRef(tuple(f"Q{i+1}" for i in range(len(values)))), **common)
            elif values.shape[0] == len(time_index):
                artifacts[metric_id] = SeriesMetricArtifact(
                    values=values, time_index=time_index, time_axis=TimeAxisRef(time_index), **common)
            else:
                raise UnsupportedMetricError(f"Metric {metric_id} has no declared axis for {values.shape}")
        else:
            raise UnsupportedMetricError(f"Metric {metric_id} cannot be bound to the factor axis: {values.shape}")
    # A series/vector is not implicitly reduced to a scalar grade or objective.
    metric_values = dict(grouped_metrics[factor_batch.factor_ids[0]]) if factor_batch.num_factors == 1 else {}
    bundle_metadata = dict(request_metadata)
    bundle_metadata.update(request_fields)
    bundle_metadata.update({"context": context, "where": None, "runtime": result.metadata})
    bundle_metadata["provenance"] = authoritative_provenance
    bundle_metadata["execution_plan"] = {
        "metric_nodes": canonical_metrics,
        "artifact_builder_nodes": artifact_plan,
        "metric_cost": metric_cost,
        "artifact_builder_cost": builder_cost,
        "planned_cost": planned_cost,
        "cost_unit": "resource_node",
        "tier": requested_tier,
    }
    if holding_returns is not None:
        bundle_metadata["portfolio_purpose"] = "RESEARCH_PROBE"
        bundle_metadata["tradability_assumption"] = "side_specific" if trade_eligibility is not None else "unrestricted_research"
        bundle_metadata["execution_certified"] = False
    if gpu_result is not None:
        bundle_metadata.update(result.metadata)
    request_id = str(bundle_metadata.pop("request_id", uuid4()))

    return EvaluationBundle(
        request_id=request_id,
        factor_ids=tuple(factor_batch.factor_ids),
        label_id=label_bundle.target_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metric_values=metric_values,
        diagnostics=diagnose_all_factors(factor_batch),
        grouped_metrics=grouped_metrics,
        metric_versions=versions,
        config_hash=config_hash,
        artifacts=artifacts,
        factor_artifacts=factor_artifacts,
        metadata=bundle_metadata,
        warnings=tuple(result.metadata.get("warnings", ())),
        split_ref=split_ref,
    )
