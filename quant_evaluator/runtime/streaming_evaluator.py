"""
Streaming evaluator for datasets too large for memory.

Processes data in chunks using generator-based loading, accumulates statistics
incrementally, and maintains constant memory usage for 100k+ factors.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple, Mapping
import sys
import numpy as np
import time

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError, InsufficientObservations, SnapshotMismatchError
from quant_evaluator.metrics.label_panel import normalize_label_panel

from quant_evaluator.planner.dependency_plan import MetricKind
from quant_evaluator.runtime.budgets import ComputationBudget, BudgetTracker, ResourceUsage


@dataclass(frozen=True)
class _InternalChunkDescriptor:
    """Global coordinates for a batch split; never inferred from array shape."""
    time_slice: Tuple[int, int]
    factor_slice: Tuple[int, int]
    asset_slice: Tuple[int, int]
    total_time: int
    total_factors: int
    parent_identity: Tuple[Any, ...]
    factor_ids: Tuple[Any, ...]
    asset_coords: Tuple[Any, ...]
    timing_vectors: Tuple[Tuple[Any, ...], ...]
    timing_offsets: Tuple[Any, ...]


@dataclass
class StreamingMetricState:
    """
    Incremental state for streaming metric computation.

    Holds accumulator values that are updated as chunks arrive.
    """
    metric_id: str
    metric_kind: MetricKind

    # Accumulators for incremental statistics
    sum_values: float = 0.0
    sum_squared: float = 0.0
    count: int = 0

    # For correlation/IC computation
    sum_x: Optional[np.ndarray] = None
    sum_y: Optional[np.ndarray] = None
    sum_xx: Optional[np.ndarray] = None
    sum_yy: Optional[np.ndarray] = None
    sum_xy: Optional[np.ndarray] = None
    valid_counts: Optional[np.ndarray] = None
    valid_count: int = 0
    total_count: int = 0

    # Track expected full dimensions for proper concatenation
    expected_total_time: Optional[int] = None
    expected_total_factors: Optional[int] = None
    accumulated_time: int = 0
    accumulated_factors: int = 0

    # Custom accumulator storage
    custom_state: Dict[str, Any] = field(default_factory=dict)
    current_chunk_descriptor: Optional[_InternalChunkDescriptor] = None

    def finalize(self) -> Any:
        """
        Finalize the accumulated state into a result value.

        Returns:
            Computed metric value
        """
        if self.metric_kind == MetricKind.IC:
            return self._finalize_ic()
        elif self.metric_kind == MetricKind.COVERAGE:
            return self._finalize_coverage()
        elif self.metric_kind == MetricKind.SUMMARY:
            return self._finalize_summary()
        else:
            return self.custom_state.get("result")

    def _finalize_ic(self) -> np.ndarray:
        """Finalize IC computation from accumulated sums."""
        if self.custom_state.get("ic_parts") is not None:
            result = np.full(
                (self.expected_total_time or 0, self.expected_total_factors or 0),
                np.nan,
                dtype=np.float64,
            )
            for coordinates, stats in self.custom_state["ic_parts"].items():
                (t0, t1), (f0, f1) = coordinates
                sx, sy, sxx, syy, sxy, counts = stats
                n = counts.astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    numerator = n * sxy - sx * sy
                    denom_x = n * sxx - sx ** 2
                    denom_y = n * syy - sy ** 2
                    values = numerator / np.sqrt(denom_x * denom_y)
                result[t0:t1, f0:f1] = np.where(
                    (n < 10) | (denom_x <= 0) | (denom_y <= 0), np.nan, values
                )
            return result
        if self.sum_x is None or self.valid_counts is None:
            return np.array([])

        n = self.valid_counts.astype(np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = n * self.sum_xy - self.sum_x * self.sum_y
            denom_x = n * self.sum_xx - self.sum_x ** 2
            denom_y = n * self.sum_yy - self.sum_y ** 2
            ic_values = numerator / np.sqrt(denom_x * denom_y)

        # Mask zero variance or insufficient observations
        mask = (n < 10) | (denom_x <= 0) | (denom_y <= 0)
        ic_values = np.where(mask, np.nan, ic_values)

        return ic_values

    def _finalize_coverage(self) -> float:
        """Finalize coverage from paired valid observations, not chunk ratios."""
        if self.total_count == 0:
            return self.sum_values / self.count if self.count else 0.0
        return self.valid_count / self.total_count

    def _finalize_summary(self) -> Dict[str, float]:
        """Finalize summary statistics."""
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        mean = self.sum_values / self.count
        variance = (self.sum_squared / self.count) - (mean ** 2)
        std = np.sqrt(max(0.0, variance))

        return {"mean": mean, "std": std, "count": self.count}


@dataclass
class StreamingEvaluationResult:
    """
    Result of streaming metric evaluation.

    Contains finalized metrics and execution statistics.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    chunks_processed: int = 0
    total_observations_processed: int = 0
    peak_memory_mb: float = 0.0
    resource_usage: Optional[ResourceUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __getstate__(self) -> Dict[str, Any]:
        # QE-P0: MappingProxyType is not picklable. Normalize any read-only
        # provenance wrapper to a plain dict on the way out so this result
        # always survives pickle, even if provenance was assigned externally.
        state = self.__dict__.copy()
        if isinstance(state.get("provenance"), MappingProxyType):
            state["provenance"] = dict(state["provenance"])
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def get_metric(self, metric_id: str, default=None) -> Any:
        """Retrieve a metric by ID."""
        return self.metrics.get(metric_id, default)

    def has_metric(self, metric_id: str) -> bool:
        """Check if metric exists in results."""
        return metric_id in self.metrics

    def to_dict(self) -> Dict:
        """Convert result to a serialization-friendly dictionary."""
        result = {
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "execution_time_seconds": self.execution_time_seconds,
            "chunks_processed": self.chunks_processed,
            "total_observations_processed": self.total_observations_processed,
            "peak_memory_mb": self.peak_memory_mb,
        }
        if self.resource_usage:
            result["resource_usage"] = self.resource_usage.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result


class StreamingEvaluator:
    """
    Streaming evaluator for large-scale factor evaluation with constant memory.

    Processes data through generators, accumulates statistics incrementally,
    and supports evaluation of 100k+ factors without loading full dataset into memory.
    """

    def __init__(
        self,
        budget: Optional[ComputationBudget] = None,
        chunk_size_time: int = 100,
        chunk_size_factors: int = 1000,
    ):
        """
        Initialize streaming evaluator.

        Args:
            budget: Computation budget (None = no limits)
            chunk_size_time: Number of time periods per chunk
            chunk_size_factors: Number of factors per chunk
        """
        self.budget = budget or ComputationBudget()
        self.budget_tracker = BudgetTracker(self.budget)
        self.chunk_size_time = chunk_size_time
        self.chunk_size_factors = chunk_size_factors

        # Registry of streaming metric functions
        self._streaming_metrics: Dict[str, Callable] = {}

        # Execution stats
        self._peak_memory_mb = 0.0

    @staticmethod
    def _freeze_identity(value: Any) -> Any:
        """Convert context metadata into a deterministic, comparable value."""
        if isinstance(value, dict):
            return tuple(sorted((str(k), StreamingEvaluator._freeze_identity(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(StreamingEvaluator._freeze_identity(v) for v in value)
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value

    def _stream_identity(self, factor_batch: FactorBatch, label_bundle: LabelBundle) -> Tuple[Any, ...]:
        """Return immutable identity/context metadata for a public stream."""
        return (
            self._freeze_identity(factor_batch.context_refs),
            tuple(factor_batch.factor_ids),
            self._freeze_identity(factor_batch.asset_axis.values),
            factor_batch.asset_axis.name,
            factor_batch.asset_axis.dtype,
            label_bundle.target_id,
            label_bundle.horizon,
            label_bundle.execution_delay,
            label_bundle.source_ref,
            label_bundle.calendar_ref,
            self._freeze_identity(label_bundle.metadata),
        )

    def _stream_provenance(
        self,
        parent_identity: Tuple[Any, ...],
        factor_ids: Tuple[Any, ...],
        asset_coords: Tuple[Any, ...],
        timing_rule: Tuple[Tuple[Any, ...], ...],
        timing_rows: int,
        timing_first: Tuple[Tuple[Any, ...], ...],
        timing_last: Tuple[Tuple[Any, ...], ...],
    ) -> Mapping[str, Any]:
        """Build bounded immutable provenance for the completed stream."""
        return MappingProxyType({
            "parent_identity": parent_identity,
            "factor_ids": factor_ids,
            "asset_coords": asset_coords,
            "timing_rule": timing_rule,
            "timing_rows": timing_rows,
            "timing_first": timing_first,
            "timing_last": timing_last,
        })

    @staticmethod
    def _estimate_provenance_memory(stream_state: Dict[str, Any]) -> float:
        """Estimate retained immutable provenance state in MB."""
        seen: set[int] = set()

        def size(value: Any) -> int:
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)

            if isinstance(value, np.ndarray):
                return sys.getsizeof(value) + value.nbytes
            total = sys.getsizeof(value)
            if isinstance(value, ABCMapping):
                return total + sum(size(key) + size(item) for key, item in value.items())
            if isinstance(value, (list, tuple, set, frozenset)):
                return total + sum(size(item) for item in value)
            return total

        total_bytes = sum(
            size(stream_state.get(key))
            for key in (
                "identity", "factor_ids", "asset_coords", "timing_rule",
                "timing_first", "timing_last",
            )
        )
        return total_bytes / (1024 * 1024)

    _PROVENANCE_SAMPLE_ROWS = 8

    @classmethod
    def _timing_sample(
        cls,
        timing_vectors: Tuple[Tuple[Any, ...], ...],
        *,
        from_end: bool = False,
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Retain only a fixed-size endpoint sample for stream provenance."""
        limit = cls._PROVENANCE_SAMPLE_ROWS
        return tuple(
            tuple(values[-limit:] if from_end else values[:limit])
            for values in timing_vectors
        )

    @staticmethod
    def _timing_vectors(label_bundle: LabelBundle) -> Tuple[Tuple[Any, ...], ...]:
        return (
            tuple(label_bundle.decision_time),
            tuple(label_bundle.execution_time),
            tuple(label_bundle.label_start_time),
            tuple(label_bundle.label_end_time),
        )

    @classmethod
    def _append_timing_sample(
        cls,
        sample: Tuple[Tuple[Any, ...], ...],
        current: Tuple[Tuple[Any, ...], ...],
        *,
        from_end: bool = False,
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Merge a chunk into a fixed-size endpoint sample."""
        limit = cls._PROVENANCE_SAMPLE_ROWS
        return tuple(
            tuple((old + new)[-limit:] if from_end else (old + new)[:limit])
            for old, new in zip(sample, current)
        )

    @staticmethod
    def _timing_offsets(label_bundle: LabelBundle) -> Tuple[Tuple[Any, ...], ...]:
        decision = label_bundle.decision_time
        try:
            return (
                tuple(e - d for e, d in zip(label_bundle.execution_time, decision)),
                tuple(s - d for s, d in zip(label_bundle.label_start_time, decision)),
                tuple(e - d for e, d in zip(label_bundle.label_end_time, decision)),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidContractError("label timing coordinates must support relative comparison") from exc

    @staticmethod
    def _timing_rule(timing_offsets: Tuple[Tuple[Any, ...], ...]) -> Tuple[Tuple[Any, ...], ...]:
        """Compare invariant per-row offsets without requiring equal chunk lengths."""
        return tuple(tuple(sorted(set(offsets), key=repr)) for offsets in timing_offsets)

    @staticmethod
    def _coordinates(axis: Any, fallback: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Use explicit axis coordinates when present; otherwise supplied labels."""
        if axis.values is not None:
            return tuple(axis.values.tolist())
        return tuple(fallback)

    def _validate_stream_boundary(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        stream_state: Dict[str, Any],
    ) -> None:
        """Enforce continuity and identity for externally supplied chunks."""
        self._validate_chunk(factor_batch, label_bundle)
        identity = self._stream_identity(factor_batch, label_bundle)
        timing_offsets = self._timing_offsets(label_bundle)
        timing_rule = self._timing_rule(timing_offsets)
        if stream_state.get("identity") is None:
            stream_state["identity"] = identity
            stream_state["timing_rule"] = timing_rule
            stream_state["timing_rows"] = len(label_bundle.decision_time)
            current_vectors = self._timing_vectors(label_bundle)
            stream_state["timing_first"] = self._timing_sample(current_vectors)
            stream_state["timing_last"] = self._timing_sample(current_vectors, from_end=True)
            stream_state["factor_ids"] = tuple(factor_batch.factor_ids)
            stream_state["asset_coords"] = self._coordinates(factor_batch.asset_axis, tuple(range(factor_batch.num_assets)))
        elif identity != stream_state["identity"]:
            raise SnapshotMismatchError("stream identity/context changed between chunks")
        elif timing_rule != stream_state["timing_rule"]:
            raise SnapshotMismatchError("stream label timing changed between chunks")
        else:
            stream_state["timing_rows"] += len(label_bundle.decision_time)
            current_vectors = self._timing_vectors(label_bundle)
            stream_state["timing_last"] = self._append_timing_sample(
                stream_state["timing_last"], current_vectors, from_end=True
            )

        times = self._coordinates(factor_batch.time_axis, tuple(label_bundle.decision_time))
        if len(times) != factor_batch.num_times:
            raise InvalidContractError("time coordinates do not match factor time axis")
        previous = stream_state.get("last_time")
        try:
            if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
                raise InvalidContractError("stream time coordinates must be strictly increasing")
            if previous is not None and times[0] <= previous:
                raise InvalidContractError(
                    "stream time coordinates are not monotonic: duplicate, replayed, or overlapping"
                )
        except TypeError as exc:
            raise InvalidContractError("stream time coordinates must be orderable") from exc
        if times:
            stream_state["last_time"] = times[-1]

    def register_streaming_metric(
        self,
        metric_id: str,
        update_fn: Callable,
        metric_kind: MetricKind = MetricKind.CUSTOM,
    ):
        """
        Register a streaming metric computation function.

        Args:
            metric_id: Unique metric identifier
            update_fn: Function that updates metric state from chunk
                       Signature: (state, factor_batch, label_bundle) -> state
            metric_kind: Type of metric
        """
        if metric_id in self._streaming_metrics:
            raise InvalidContractError(
                f"Streaming metric already registered: {metric_id}"
            )
        self._streaming_metrics[metric_id] = {
            "update_fn": update_fn,
            "metric_kind": metric_kind,
        }

    def evaluate_stream(
        self,
        data_generator: Iterator[Tuple[FactorBatch, LabelBundle]],
        metric_specs: List[Dict],
        *,
        _internal_splitter: bool = False,
    ) -> StreamingEvaluationResult:
        """
        Evaluate metrics on streaming data.

        Args:
            data_generator: Generator yielding (factor_batch, label_bundle) chunks
            metric_specs: List of metric specifications

        Returns:
            StreamingEvaluationResult with finalized metrics

        Raises:
            InvalidContractError: If metrics not registered
            RuntimeError: If budget exceeded
        """
        requested_metric_ids = [spec["metric_id"] for spec in metric_specs]
        if len(requested_metric_ids) != len(set(requested_metric_ids)):
            duplicate_ids = sorted(
                {metric_id for metric_id in requested_metric_ids
                 if requested_metric_ids.count(metric_id) > 1},
                key=str,
            )
            raise InvalidContractError(
                f"duplicate requested metric_specs: {duplicate_ids}"
            )

        start_time = time.time()
        self.budget_tracker.reset()
        self._peak_memory_mb = 0.0

        # Initialize metric states
        metric_states = self._initialize_metric_states(metric_specs)

        # Process chunks
        chunks_processed = 0
        total_obs = 0
        stream_state: Dict[str, Any] = {}

        for item in data_generator:
            if len(item) == 3:
                factor_batch, label_bundle, chunk_descriptor = item
            else:
                factor_batch, label_bundle = item
                chunk_descriptor = None
            self.budget_tracker.check_budget(raise_on_exceed=True)

            # Validate chunk and public stream continuity before updating state.
            if _internal_splitter:
                self._validate_chunk(factor_batch, label_bundle)
            else:
                self._validate_stream_boundary(factor_batch, label_bundle, stream_state)

            for state in metric_states.values():
                state.current_chunk_descriptor = chunk_descriptor

            if chunk_descriptor is not None:
                descriptor_timing_rule = self._timing_rule(chunk_descriptor.timing_offsets)
                if stream_state.get("identity") is None:
                    stream_state["identity"] = chunk_descriptor.parent_identity
                    stream_state["factor_ids"] = chunk_descriptor.factor_ids
                    stream_state["asset_coords"] = chunk_descriptor.asset_coords
                    stream_state["timing_rule"] = descriptor_timing_rule
                    stream_state["timing_rows"] = len(chunk_descriptor.timing_vectors[0])
                    stream_state["timing_first"] = self._timing_sample(chunk_descriptor.timing_vectors)
                    stream_state["timing_last"] = self._timing_sample(
                        chunk_descriptor.timing_vectors, from_end=True
                    )
                    stream_state["last_internal_time_slice"] = chunk_descriptor.time_slice
                elif chunk_descriptor.parent_identity != stream_state["identity"]:
                    raise SnapshotMismatchError("internal chunk parent identity changed")
                elif descriptor_timing_rule != stream_state["timing_rule"]:
                    raise SnapshotMismatchError("internal chunk label timing changed")
                elif chunk_descriptor.time_slice != stream_state["last_internal_time_slice"]:
                    stream_state["timing_rows"] += len(chunk_descriptor.timing_vectors[0])
                    stream_state["timing_first"] = self._append_timing_sample(
                        stream_state["timing_first"],
                        chunk_descriptor.timing_vectors,
                    )
                    stream_state["timing_last"] = self._append_timing_sample(
                        stream_state["timing_last"],
                        chunk_descriptor.timing_vectors,
                        from_end=True,
                    )
                    stream_state["last_internal_time_slice"] = chunk_descriptor.time_slice

            # Update each metric state with this chunk
            for metric_id, state in metric_states.items():
                if metric_id not in self._streaming_metrics:
                    raise InvalidContractError(f"Streaming metric not registered: {metric_id}")

                update_fn = self._streaming_metrics[metric_id]["update_fn"]
                updated_state = update_fn(state, factor_batch, label_bundle)
                metric_states[metric_id] = updated_state

            chunks_processed += 1
            total_obs += factor_batch.num_times * factor_batch.num_assets

            # Track live input and persistent metric-state memory.  IC accumulators
            # grow across public time-partitioned streams and must not be hidden by
            # reporting only the current input chunk.
            live_memory_mb = (
                self._estimate_chunk_memory(factor_batch, label_bundle)
                + self._estimate_metric_states_memory(metric_states)
                + self._estimate_provenance_memory(stream_state)
            )
            self._peak_memory_mb = max(self._peak_memory_mb, live_memory_mb)

            self.budget_tracker.record_operation()

        # Finalize all metrics
        if chunks_processed == 0:
            raise InsufficientObservations("public streaming evaluation received no chunks")
        result = StreamingEvaluationResult()
        for metric_id, state in metric_states.items():
            result.metrics[metric_id] = state.finalize()

        # Set execution metadata
        elapsed = time.time() - start_time
        result.execution_time_seconds = elapsed
        result.chunks_processed = chunks_processed
        result.total_observations_processed = total_obs
        result.peak_memory_mb = self._peak_memory_mb
        result.resource_usage = self.budget_tracker.get_current_usage()

        result.metadata["chunk_size_time"] = self.chunk_size_time
        result.metadata["chunk_size_factors"] = self.chunk_size_factors
        result.metadata["num_metrics"] = len(metric_states)
        if stream_state.get("identity") is not None:
            result.provenance = self._stream_provenance(
                stream_state["identity"],
                stream_state["factor_ids"],
                stream_state["asset_coords"],
                stream_state["timing_rule"],
                stream_state["timing_rows"],
                stream_state["timing_first"],
                stream_state["timing_last"],
            )

        return result

    def evaluate_large_batch(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        metric_specs: List[Dict],
    ) -> StreamingEvaluationResult:
        """
        Evaluate metrics on large batch by converting to stream.

        Automatically chunks the input batch and processes it as a stream.

        Args:
            factor_batch: Large input factor batch
            label_bundle: Large input label bundle
            metric_specs: List of metric specifications

        Returns:
            StreamingEvaluationResult with finalized metrics
        """
        self._validate_chunk(factor_batch, label_bundle)
        generator = self._batch_to_generator(factor_batch, label_bundle, include_descriptors=True)
        return self.evaluate_stream(generator, metric_specs, _internal_splitter=True)

    def _batch_to_generator(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
        *,
        include_descriptors: bool = False,
    ) -> Iterator[Tuple[FactorBatch, LabelBundle]]:
        """
        Convert large batch into chunk generator.

        Yields chunks of (chunk_size_time, N, chunk_size_factors).
        """
        T = factor_batch.num_times
        N = factor_batch.num_assets
        F = factor_batch.num_factors
        parent_identity = self._stream_identity(factor_batch, label_bundle)
        factor_ids = tuple(factor_batch.factor_ids)
        asset_coords = self._coordinates(factor_batch.asset_axis, tuple(range(N)))
        for t_start in range(0, T, self.chunk_size_time):
            t_end = min(t_start + self.chunk_size_time, T)

            for f_start in range(0, F, self.chunk_size_factors):
                f_end = min(f_start + self.chunk_size_factors, F)

                chunk_batch = self._extract_chunk_batch(
                    factor_batch,
                    t_start, t_end,
                    0, N,
                    f_start, f_end,
                )

                chunk_labels = self._extract_chunk_labels(
                    label_bundle,
                    t_start, t_end,
                    0, N,
                )

                if include_descriptors:
                    chunk_timing_vectors = self._timing_vectors(chunk_labels)
                    chunk_timing_offsets = self._timing_offsets(chunk_labels)
                    yield chunk_batch, chunk_labels, _InternalChunkDescriptor(
                        (t_start, t_end), (f_start, f_end), (0, N), T, F,
                        parent_identity, factor_ids, asset_coords,
                        chunk_timing_vectors, chunk_timing_offsets,
                    )
                else:
                    yield chunk_batch, chunk_labels

    def _extract_chunk_batch(
        self,
        factor_batch: FactorBatch,
        t_start: int, t_end: int,
        a_start: int, a_end: int,
        f_start: int, f_end: int,
    ) -> FactorBatch:
        """Extract factor batch chunk."""
        from quant_evaluator.contracts.factor_batch import AxisRef

        chunk_values = factor_batch.values[t_start:t_end, a_start:a_end, f_start:f_end]
        chunk_validity = None
        if factor_batch.validity is not None:
            chunk_validity = factor_batch.validity[t_start:t_end, a_start:a_end, f_start:f_end]

        chunk_factor_ids = factor_batch.factor_ids[f_start:f_end]

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
            value_hash=factor_batch.value_hash,
        )

    def _extract_chunk_labels(
        self,
        label_bundle: LabelBundle,
        t_start: int, t_end: int,
        a_start: int, a_end: int,
    ) -> LabelBundle:
        """Extract label bundle chunk."""
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

    def _initialize_metric_states(
        self,
        metric_specs: List[Dict],
    ) -> Dict[str, StreamingMetricState]:
        """Initialize metric states for all requested metrics."""
        states = {}

        for spec in metric_specs:
            metric_id = spec["metric_id"]
            metric_kind_str = spec.get("metric_kind", "custom")

            # Map string to MetricKind enum
            if metric_kind_str == "ic":
                metric_kind = MetricKind.IC
            elif metric_kind_str == "coverage":
                metric_kind = MetricKind.COVERAGE
            elif metric_kind_str == "summary":
                metric_kind = MetricKind.SUMMARY
            else:
                metric_kind = MetricKind.CUSTOM

            registered = self._streaming_metrics.get(metric_id)
            if registered is None:
                raise InvalidContractError(f"Streaming metric not registered: {metric_id}")
            registered_kind = registered["metric_kind"]
            if metric_kind != registered_kind:
                raise InvalidContractError(
                    f"Streaming metric kind mismatch for {metric_id}: "
                    f"requested {metric_kind.value}, registered {registered_kind.value}"
                )

            states[metric_id] = StreamingMetricState(
                metric_id=metric_id,
                metric_kind=metric_kind,
            )

        return states

    def _validate_chunk(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
    ):
        """Validate chunk contracts."""
        if factor_batch.num_times != len(label_bundle.values):
            raise InvalidContractError(
                f"Factor time axis ({factor_batch.num_times}) "
                f"does not match label length ({len(label_bundle.values)})"
            )
        if (
            label_bundle.values.ndim == 2
            and label_bundle.values.shape[1] != factor_batch.values.shape[1]
        ):
            raise InvalidContractError(
                f"Factor asset axis ({factor_batch.values.shape[1]}) does not match "
                f"label asset dimension ({label_bundle.values.shape[1]})"
            )

        # Explicit factor coordinates are authoritative and must identify the
        # same observations as the label decision times.  Without this check,
        # equal-length but differently dated inputs are paired positionally.
        if factor_batch.time_axis.values is not None:
            factor_times = self._coordinates(factor_batch.time_axis, ())
            label_times = tuple(label_bundle.decision_time)
            try:
                aligned = len(factor_times) == len(label_times) and all(
                    bool(factor_time == label_time)
                    for factor_time, label_time in zip(factor_times, label_times)
                )
            except (TypeError, ValueError):
                aligned = False
            if not aligned:
                raise InvalidContractError(
                    "factor time coordinates do not match label decision_time"
                )

    def _estimate_chunk_memory(
        self,
        factor_batch: FactorBatch,
        label_bundle: LabelBundle,
    ) -> float:
        """Estimate memory usage of current chunk in MB."""
        factor_bytes = factor_batch.values.nbytes
        if factor_batch.validity is not None:
            factor_bytes += factor_batch.validity.nbytes

        label_bytes = label_bundle.values.nbytes
        if label_bundle.validity is not None:
            label_bytes += label_bundle.validity.nbytes

        total_bytes = factor_bytes + label_bytes
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _estimate_metric_states_memory(
        metric_states: Dict[str, StreamingMetricState],
    ) -> float:
        """Estimate bytes retained by NumPy arrays in metric accumulator state."""
        seen: set[int] = set()

        def array_bytes(value: Any) -> int:
            if isinstance(value, np.ndarray):
                identity = id(value)
                if identity in seen:
                    return 0
                seen.add(identity)
                return value.nbytes
            if isinstance(value, dict):
                return sum(array_bytes(key) + array_bytes(item) for key, item in value.items())
            if isinstance(value, (list, tuple, set)):
                return sum(array_bytes(item) for item in value)
            return 0

        total_bytes = 0
        for state in metric_states.values():
            total_bytes += sum(array_bytes(value) for value in vars(state).values())
        return total_bytes / (1024 * 1024)

    def get_budget_report(self) -> str:
        """Get formatted budget usage report."""
        return self.budget_tracker.format_usage_report()


# Built-in streaming metric updaters

def streaming_ic_updater(
    state: StreamingMetricState,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> StreamingMetricState:
    """
    Update IC state incrementally from chunk.

    Accumulates sufficient statistics for correlation computation.
    When chunking by time, stores separate time slices.
    When chunking by assets, accumulates statistics within time periods.
    """
    T, N, F = factor_batch.values.shape

    labels_broadcast, label_validity = normalize_label_panel(
        label_bundle, factor_batch.values.shape[1]
    )

    # Apply validity
    factors = factor_batch.values.copy()
    if factor_batch.validity is not None:
        factors = np.where(factor_batch.validity, factors, np.nan)

    labels = labels_broadcast.copy()
    if label_validity is not None:
        labels = np.where(label_validity, labels, np.nan)

    # Expand labels for broadcasting: (T, N, 1)
    labels_expanded = labels[:, :, np.newaxis] if labels.ndim == 2 else labels[:, np.newaxis, np.newaxis]

    # Compute pairwise finite mask
    finite_mask = np.isfinite(factors) & np.isfinite(labels_expanded)

    # Compute chunk statistics
    chunk_sum_x = np.where(finite_mask, factors, 0.0).sum(axis=1)  # (T, F)
    chunk_sum_y = np.where(finite_mask, labels_expanded, 0.0).sum(axis=1)
    chunk_sum_xx = np.where(finite_mask, factors ** 2, 0.0).sum(axis=1)
    chunk_sum_yy = np.where(finite_mask, labels_expanded ** 2, 0.0).sum(axis=1)
    chunk_sum_xy = np.where(finite_mask, factors * labels_expanded, 0.0).sum(axis=1)
    chunk_valid_counts = np.sum(finite_mask, axis=1, dtype=np.int32)

    descriptor = state.current_chunk_descriptor
    if descriptor is not None:
        if descriptor.parent_identity != state.custom_state.get("parent_identity", descriptor.parent_identity):
            raise SnapshotMismatchError("internal chunk parent identity changed")
        state.custom_state.setdefault("parent_identity", descriptor.parent_identity)
        state.expected_total_time = descriptor.total_time
        state.expected_total_factors = descriptor.total_factors
        state.custom_state.setdefault("ic_parts", {})[
            (descriptor.time_slice, descriptor.factor_slice)
        ] = (
            chunk_sum_x, chunk_sum_y, chunk_sum_xx, chunk_sum_yy,
            chunk_sum_xy, chunk_valid_counts,
        )
        return state

    # Initialize or concatenate accumulators
    if state.sum_x is None:
        # First chunk - initialize
        state.sum_x = chunk_sum_x
        state.sum_y = chunk_sum_y
        state.sum_xx = chunk_sum_xx
        state.sum_yy = chunk_sum_yy
        state.sum_xy = chunk_sum_xy
        state.valid_counts = chunk_valid_counts
        state.accumulated_time = T
        state.accumulated_factors = F
    else:
        # Determine strategy based on accumulated vs chunk dimensions
        # If we've accumulated fewer time periods than this chunk would complete,
        # we're chunking by time and should concatenate
        # If accumulated_time already covers this chunk's periods, we're chunking
        # by assets/factors and should accumulate

        # Check if this is a new time slice or same time with more assets/factors
        prev_T = state.accumulated_time

        # Simple heuristic: if current chunk has same T as accumulated and we haven't
        # seen different factor counts, we're likely chunking by assets
        # Otherwise, we're chunking by time or factors

        curr_T, curr_F = chunk_sum_x.shape
        state_T, state_F = state.sum_x.shape

        # If state has fewer time periods than expected, we're accumulating time
        if state_T == curr_T and state_F == curr_F:
            # Exact same dimensions - check if we're repeating time (asset chunks)
            # or progressing through time
            # Use accumulated_time as indicator: if it equals state_T, we're on first pass
            # through factors, so this is likely new time periods (concatenate)
            # If accumulated_time > state_T, we've wrapped around (accumulate)
            if state.accumulated_time == state_T:
                # First time seeing this time block size - assume new time periods
                state.sum_x = np.concatenate([state.sum_x, chunk_sum_x], axis=0)
                state.sum_y = np.concatenate([state.sum_y, chunk_sum_y], axis=0)
                state.sum_xx = np.concatenate([state.sum_xx, chunk_sum_xx], axis=0)
                state.sum_yy = np.concatenate([state.sum_yy, chunk_sum_yy], axis=0)
                state.sum_xy = np.concatenate([state.sum_xy, chunk_sum_xy], axis=0)
                state.valid_counts = np.concatenate([state.valid_counts, chunk_valid_counts], axis=0)
                state.accumulated_time += curr_T
            else:
                # We've seen this pattern before - accumulate
                state.sum_x += chunk_sum_x
                state.sum_y += chunk_sum_y
                state.sum_xx += chunk_sum_xx
                state.sum_yy += chunk_sum_yy
                state.sum_xy += chunk_sum_xy
                state.valid_counts += chunk_valid_counts
        elif state_T == curr_T:
            # Same time, different factors - chunking by factors, concatenate along F
            state.sum_x = np.concatenate([state.sum_x, chunk_sum_x], axis=1)
            state.sum_y = np.concatenate([state.sum_y, chunk_sum_y], axis=1)
            state.sum_xx = np.concatenate([state.sum_xx, chunk_sum_xx], axis=1)
            state.sum_yy = np.concatenate([state.sum_yy, chunk_sum_yy], axis=1)
            state.sum_xy = np.concatenate([state.sum_xy, chunk_sum_xy], axis=1)
            state.valid_counts = np.concatenate([state.valid_counts, chunk_valid_counts], axis=1)
            state.accumulated_factors += curr_F
        else:
            # Different time - chunking by time, concatenate along T
            state.sum_x = np.concatenate([state.sum_x, chunk_sum_x], axis=0)
            state.sum_y = np.concatenate([state.sum_y, chunk_sum_y], axis=0)
            state.sum_xx = np.concatenate([state.sum_xx, chunk_sum_xx], axis=0)
            state.sum_yy = np.concatenate([state.sum_yy, chunk_sum_yy], axis=0)
            state.sum_xy = np.concatenate([state.sum_xy, chunk_sum_xy], axis=0)
            state.valid_counts = np.concatenate([state.valid_counts, chunk_valid_counts], axis=0)
            state.accumulated_time += curr_T

    return state


def streaming_coverage_updater(
    state: StreamingMetricState,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> StreamingMetricState:
    """
    Update coverage state incrementally from chunk.

    Accumulates count of valid observations.
    """
    total_count = factor_batch.values.shape[0] * factor_batch.values.shape[1] * factor_batch.values.shape[2]
    factor_mask = np.isfinite(factor_batch.values)
    if factor_batch.validity is not None:
        factor_mask &= factor_batch.validity

    label_values, label_validity = normalize_label_panel(
        label_bundle, factor_batch.values.shape[1]
    )
    label_mask = np.isfinite(label_values)
    if label_validity is not None:
        label_mask &= label_validity
    pair_mask = factor_mask & label_mask[:, :, None]
    total_count = int(np.prod(pair_mask.shape))
    state.valid_count += int(np.sum(pair_mask))
    state.total_count += total_count
    state.count += 1
    return state


def streaming_summary_updater(
    state: StreamingMetricState,
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> StreamingMetricState:
    """
    Update summary statistics state incrementally from chunk.

    Accumulates mean and variance using Welford's online algorithm.
    """
    valid_mask = np.isfinite(factor_batch.values)
    if factor_batch.validity is not None:
        valid_mask = valid_mask & factor_batch.validity

    valid_values = factor_batch.values[valid_mask]

    if len(valid_values) > 0:
        state.sum_values += np.sum(valid_values)
        state.sum_squared += np.sum(valid_values ** 2)
        state.count += len(valid_values)

    return state
