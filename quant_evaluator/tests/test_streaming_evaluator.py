"""
Tests for StreamingEvaluator with synthetic large datasets.

Validates constant memory usage, incremental computation correctness,
and performance on 100k+ factor scenarios.
"""

import json

import pytest
import numpy as np
from typing import Iterator, Tuple

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError, InsufficientObservations, SnapshotMismatchError

from quant_evaluator.runtime.streaming_evaluator import (
    StreamingEvaluator,
    StreamingEvaluationResult,
    StreamingMetricState,
    streaming_ic_updater,
    streaming_coverage_updater,
    streaming_summary_updater,
)
from quant_evaluator.planner.dependency_plan import MetricKind
from quant_evaluator.runtime.budgets import ComputationBudget


class TestStreamingMetricState:
    def test_state_creation(self):
        state = StreamingMetricState(
            metric_id="test_metric",
            metric_kind=MetricKind.COVERAGE,
        )

        assert state.metric_id == "test_metric"
        assert state.count == 0
        assert state.sum_values == 0.0

    def test_finalize_coverage(self):
        state = StreamingMetricState(
            metric_id="coverage",
            metric_kind=MetricKind.COVERAGE,
        )

        state.sum_values = 0.85 * 10
        state.count = 10

        result = state.finalize()
        assert abs(result - 0.85) < 1e-6

    def test_finalize_coverage_empty(self):
        state = StreamingMetricState(
            metric_id="coverage",
            metric_kind=MetricKind.COVERAGE,
        )

        result = state.finalize()
        assert result == 0.0

    def test_finalize_summary(self):
        state = StreamingMetricState(
            metric_id="summary",
            metric_kind=MetricKind.SUMMARY,
        )

        # Simulate accumulated values: [1, 2, 3, 4, 5]
        values = np.array([1, 2, 3, 4, 5])
        state.sum_values = np.sum(values)
        state.sum_squared = np.sum(values ** 2)
        state.count = len(values)

        result = state.finalize()

        assert abs(result["mean"] - 3.0) < 1e-6
        assert abs(result["std"] - np.std(values, ddof=0)) < 1e-6
        assert result["count"] == 5

    def test_finalize_ic(self):
        state = StreamingMetricState(
            metric_id="ic",
            metric_kind=MetricKind.IC,
        )

        # Simulate perfect correlation for single (t, f) pair
        T, F = 1, 1
        n = 100

        x = np.arange(n, dtype=np.float64)
        y = 2 * x + 1

        state.sum_x = np.array([[np.sum(x)]])
        state.sum_y = np.array([[np.sum(y)]])
        state.sum_xx = np.array([[np.sum(x ** 2)]])
        state.sum_yy = np.array([[np.sum(y ** 2)]])
        state.sum_xy = np.array([[np.sum(x * y)]])
        state.valid_counts = np.array([[n]], dtype=np.int32)

        result = state.finalize()

        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6  # Perfect correlation


class TestStreamingEvaluationResult:
    def test_result_creation(self):
        result = StreamingEvaluationResult()

        assert result.metrics == {}
        assert result.chunks_processed == 0
        assert result.total_observations_processed == 0

    def test_get_metric(self):
        result = StreamingEvaluationResult()
        result.metrics["ic_mean"] = 0.05

        assert result.get_metric("ic_mean") == 0.05
        assert result.get_metric("missing") is None

    def test_to_dict(self):
        result = StreamingEvaluationResult()
        result.metrics["test"] = 0.5
        result.chunks_processed = 10
        result.peak_memory_mb = 256.0

        data = result.to_dict()

        assert data["metrics"]["test"] == 0.5
        assert data["chunks_processed"] == 10
        assert data["peak_memory_mb"] == 256.0
    def test_to_dict_serializes_immutable_provenance(self):
        from types import MappingProxyType

        provenance = MappingProxyType({
            "parent_identity": (("snapshot", "s1"),),
            "factor_ids": ("factor_0",),
            "asset_coords": ("asset-1",),
            "timing_rule": ((0,), (1,), (2,)),
            "timing_rows": 2,
            "timing_first": ((10, 11), (11, 12), (12, 13), (14, 15)),
            "timing_last": ((10, 11), (11, 12), (12, 13), (14, 15)),
        })
        result = StreamingEvaluationResult(provenance=provenance)

        data = result.to_dict()

        assert data["provenance"] == dict(provenance)
        assert data["provenance"] is not provenance
        json.dumps(data)


class TestStreamingEvaluator:
    def create_test_batch(self, T=100, N=500, F=1):
        """Helper to create test factor batch."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)
        values = np.random.randn(T, N, F)

        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def create_test_labels(self, T=100, N=500):
        """Helper to create test label bundle."""
        values = np.random.randn(T, N)

        return LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

    def test_evaluator_creation(self):
        evaluator = StreamingEvaluator()

        assert evaluator.budget is not None
        assert evaluator.budget_tracker is not None
        assert evaluator.chunk_size_time == 100
        assert evaluator.chunk_size_factors == 1000

    def test_evaluator_custom_chunks(self):
        evaluator = StreamingEvaluator(
            chunk_size_time=50,
            chunk_size_factors=500,
        )

        assert evaluator.chunk_size_time == 50
        assert evaluator.chunk_size_factors == 500

    def test_register_streaming_metric(self):
        evaluator = StreamingEvaluator()

        def dummy_updater(state, factor_batch, label_bundle):
            return state

        evaluator.register_streaming_metric(
            "test_metric",
            dummy_updater,
            MetricKind.CUSTOM,
        )

        assert "test_metric" in evaluator._streaming_metrics

    def test_batch_to_generator(self):
        evaluator = StreamingEvaluator(
            chunk_size_time=50,
            chunk_size_factors=2,
        )

        batch = self.create_test_batch(T=100, N=200, F=5)
        labels = self.create_test_labels(T=100, N=200)

        chunks = list(evaluator._batch_to_generator(batch, labels))

        # Should create (100/50) * (5/2) = 2 * 3 = 6 chunks
        assert len(chunks) == 6

        # Verify first chunk dimensions
        first_batch, first_labels = chunks[0]
        assert first_batch.num_times == 50
        assert first_batch.num_assets == 200
        assert first_batch.num_factors == 2

    def test_batch_to_generator_exact_fit(self):
        evaluator = StreamingEvaluator(
            chunk_size_time=100,
            chunk_size_factors=5,
        )

        batch = self.create_test_batch(T=100, N=200, F=5)
        labels = self.create_test_labels(T=100, N=200)

        chunks = list(evaluator._batch_to_generator(batch, labels))

        # Should create exactly 1 chunk
        assert len(chunks) == 1

    def test_evaluate_large_batch_coverage(self):
        evaluator = StreamingEvaluator(chunk_size_time=50, chunk_size_factors=2)

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        batch = self.create_test_batch(T=100, N=200, F=3)
        labels = self.create_test_labels(T=100, N=200)

        metric_specs = [
            {"metric_id": "coverage", "metric_kind": "coverage"}
        ]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        assert result.has_metric("coverage")
        coverage = result.get_metric("coverage")
        assert 0.0 <= coverage <= 1.0
        assert result.chunks_processed > 1  # Multiple chunks processed

    def test_large_batch_provenance_matches_unsplit_parent(self):
        from types import MappingProxyType

        evaluator = StreamingEvaluator(chunk_size_time=1, chunk_size_factors=1)
        evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)
        batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=AxisRef("time", "int64", 2, np.array([10, 11])),
            asset_axis=AxisRef("asset", "str", 2, np.array(["a", "b"])),
            values=np.ones((2, 2, 2)),
            context_refs={"snapshot": "s1"},
        )
        labels = LabelBundle(
            target_id="target", values=np.ones((2, 2)), horizon=1,
            decision_time=(10, 11), execution_time=(11, 12),
            label_start_time=(12, 13), label_end_time=(14, 15),
            source_ref="labels", calendar_ref="cal", metadata={"revision": 3},
        )

        result = evaluator.evaluate_large_batch(
            batch, labels, [{"metric_id": "coverage", "metric_kind": "coverage"}]
        )

        assert isinstance(result.provenance, MappingProxyType)
        assert result.provenance["parent_identity"] == evaluator._stream_identity(batch, labels)
        assert result.provenance["factor_ids"] == ("f0", "f1")
        assert result.provenance["asset_coords"] == ("a", "b")
        assert result.provenance["timing_rule"] == ((1,), (2,), (4,))
        assert result.provenance["timing_rows"] == 2
        assert result.provenance["timing_first"] == ((10, 11), (11, 12), (12, 13), (14, 15))
        assert result.provenance["timing_last"] == ((10, 11), (11, 12), (12, 13), (14, 15))
        assert result.to_dict()["provenance"] == dict(result.provenance)

    def test_provenance_endpoint_samples_are_bounded(self):
        evaluator, specs = self._boundary_evaluator()

        def chunks():
            for start in range(0, 40):
                values = np.ones((1, 1, 1))
                batch = FactorBatch(
                    factor_ids=("f",),
                    time_axis=AxisRef("time", "int64", 1, np.array([start])),
                    asset_axis=AxisRef("asset", "int64", 1, np.array(["a"])),
                    values=values,
                    context_refs={"snapshot": "s1"},
                )
                labels = LabelBundle(
                    target_id="target", values=np.ones((1, 1)), horizon=1,
                    decision_time=(start,), execution_time=(start + 1,),
                    label_start_time=(start + 2,), label_end_time=(start + 3,),
                    source_ref="labels", calendar_ref="cal",
                )
                yield batch, labels

        result = evaluator.evaluate_stream(chunks(), specs)
        assert result.provenance["timing_rows"] == 40
        assert all(len(values) <= evaluator._PROVENANCE_SAMPLE_ROWS
                   for values in result.provenance["timing_first"])
        assert all(len(values) <= evaluator._PROVENANCE_SAMPLE_ROWS
                   for values in result.provenance["timing_last"])
        assert result.provenance["timing_first"][0] == (0,)
        assert result.provenance["timing_last"][0] == tuple(range(32, 40))

        evaluator = StreamingEvaluator(chunk_size_time=50, chunk_size_factors=2)

        evaluator.register_streaming_metric(
            "summary",
            streaming_summary_updater,
            MetricKind.SUMMARY,
        )

        # Create batch with known statistics
        T, N, F = 100, 200, 1
        values = np.full((T, N, F), 5.0)  # All values = 5.0
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=("factor_0",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )
        labels = self.create_test_labels(T=100, N=200)

        metric_specs = [
            {"metric_id": "summary", "metric_kind": "summary"}
        ]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        assert result.has_metric("summary")
        summary = result.get_metric("summary")
        assert abs(summary["mean"] - 5.0) < 1e-6
        assert abs(summary["std"] - 0.0) < 1e-6
        assert summary["count"] == T * N * F

    def test_evaluate_stream_with_generator(self):
        evaluator = StreamingEvaluator()

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        # Create a simple generator
        def simple_generator():
            for offset in range(0, 100, 20):
                batch = self.create_test_batch(T=20, N=100, F=2)
                labels = LabelBundle(
                    target_id="forward_return_1d",
                    values=np.random.randn(20, 100),
                    horizon=1,
                    decision_time=tuple(range(offset, offset + 20)),
                    label_start_time=tuple(range(offset, offset + 20)),
                    label_end_time=tuple(range(offset + 1, offset + 21)),
                )
                yield batch, labels

        metric_specs = [
            {"metric_id": "coverage", "metric_kind": "coverage"}
        ]

        result = evaluator.evaluate_stream(simple_generator(), metric_specs)

        assert result.has_metric("coverage")
        assert result.chunks_processed == 5
        assert result.execution_time_seconds > 0

    @pytest.mark.parametrize("label_assets", [99, 101])
    def test_rejects_mismatched_two_dimensional_label_asset_axis(self, label_assets):
        evaluator = StreamingEvaluator()
        updater_calls = []

        def recording_updater(state, factor_batch, label_bundle):
            updater_calls.append((factor_batch, label_bundle))
            return state

        evaluator.register_streaming_metric(
            "coverage", recording_updater, MetricKind.COVERAGE
        )
        batch = self.create_test_batch(T=20, N=100, F=2)
        labels = LabelBundle(
            target_id="forward_return_1d",
            values=np.random.randn(20, label_assets),
            horizon=1,
            decision_time=tuple(range(20)),
            label_start_time=tuple(range(20)),
            label_end_time=tuple(range(1, 21)),
        )

        with pytest.raises(InvalidContractError, match="label asset dimension"):
            evaluator.evaluate_stream(
                iter(((batch, labels),)),
                ({"metric_id": "coverage", "metric_kind": "coverage"},),
            )

        assert updater_calls == []

    def test_stream_allows_one_dimensional_labels(self):
        evaluator = StreamingEvaluator()
        evaluator.register_streaming_metric(
            "coverage", streaming_coverage_updater, MetricKind.COVERAGE
        )
        batch = self.create_test_batch(T=20, N=100, F=2)
        labels = LabelBundle(
            target_id="forward_return_1d",
            values=np.random.randn(20),
            horizon=1,
            decision_time=tuple(range(20)),
            label_start_time=tuple(range(20)),
            label_end_time=tuple(range(1, 21)),
        )

        result = evaluator.evaluate_stream(
            iter(((batch, labels),)),
            ({"metric_id": "coverage", "metric_kind": "coverage"},),
        )

        assert result.get_metric("coverage") == pytest.approx(1.0)

    @pytest.mark.parametrize("label_assets", [99, 101])
    def test_large_batch_rejects_parent_label_asset_mismatch_before_updater(
        self, label_assets
    ):
        evaluator = StreamingEvaluator(chunk_size_time=10, chunk_size_factors=1)
        updater_calls = []

        def recording_updater(state, factor_batch, label_bundle):
            updater_calls.append((factor_batch, label_bundle))
            return state

        evaluator.register_streaming_metric(
            "coverage", recording_updater, MetricKind.COVERAGE
        )
        batch = self.create_test_batch(T=20, N=100, F=2)
        labels = self.create_test_labels(T=20, N=label_assets)

        with pytest.raises(InvalidContractError, match="label asset dimension"):
            evaluator.evaluate_large_batch(
                batch,
                labels,
                ({"metric_id": "coverage", "metric_kind": "coverage"},),
            )

        assert updater_calls == []

    def test_large_batch_allows_one_dimensional_labels(self):
        evaluator = StreamingEvaluator(chunk_size_time=10, chunk_size_factors=1)
        evaluator.register_streaming_metric(
            "coverage", streaming_coverage_updater, MetricKind.COVERAGE
        )
        batch = self.create_test_batch(T=20, N=100, F=2)
        labels = LabelBundle(
            target_id="forward_return_1d",
            values=np.random.randn(20),
            horizon=1,
            decision_time=tuple(range(20)),
            label_start_time=tuple(range(20)),
            label_end_time=tuple(range(1, 21)),
        )

        result = evaluator.evaluate_large_batch(
            batch,
            labels,
            ({"metric_id": "coverage", "metric_kind": "coverage"},),
        )

        assert result.get_metric("coverage") == pytest.approx(1.0)

    def test_streaming_ic_correctness(self):
        """Test that streaming IC matches batch IC computation."""
        evaluator = StreamingEvaluator(chunk_size_time=25, chunk_size_factors=1)

        evaluator.register_streaming_metric(
            "ic",
            streaming_ic_updater,
            MetricKind.IC,
        )

        # Create correlated data
        T, N, F = 100, 200, 1
        np.random.seed(42)

        # Factor and label with known correlation
        factor_values = np.random.randn(T, N, F)
        label_values = factor_values[:, :, 0] * 0.5 + np.random.randn(T, N) * 0.5

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=("factor_0",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        labels = LabelBundle(
            target_id="forward_return_1d",
            values=label_values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        metric_specs = [
            {"metric_id": "ic", "metric_kind": "ic"}
        ]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        assert result.has_metric("ic")
        ic_values = result.get_metric("ic")
        assert ic_values.shape == (T, F)

        # Should have reasonable correlation values (not all NaN)
        valid_ics = ic_values[np.isfinite(ic_values)]
        assert len(valid_ics) > 0
        assert np.all(valid_ics >= -1.0) and np.all(valid_ics <= 1.0)

    def test_large_factor_count(self):
        """Test evaluation with 10k+ factors."""
        evaluator = StreamingEvaluator(
            chunk_size_time=50,
            chunk_size_factors=1000,
        )

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        # Create large factor batch
        T, N, F = 100, 500, 10000
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        # Use sparse random to save memory during test
        values = np.random.randn(T, N, F).astype(np.float32)

        batch = FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
            dtype="float32",
        )

        labels = self.create_test_labels(T=T, N=N)

        metric_specs = [
            {"metric_id": "coverage", "metric_kind": "coverage"}
        ]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        assert result.has_metric("coverage")
        # Should process multiple chunks
        assert result.chunks_processed > 1
        # Should track peak memory
        assert result.peak_memory_mb > 0

    def test_constant_memory_usage(self):
        """Verify that memory usage stays constant across chunks."""
        evaluator = StreamingEvaluator(
            chunk_size_time=20,
            chunk_size_factors=100,
        )

        evaluator.register_streaming_metric(
            "summary",
            streaming_summary_updater,
            MetricKind.SUMMARY,
        )

        # Create moderate-sized batch
        T, N, F = 200, 500, 1000
        batch = self.create_test_batch(T=T, N=N, F=F)
        labels = self.create_test_labels(T=T, N=N)

        metric_specs = [
            {"metric_id": "summary", "metric_kind": "summary"}
        ]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        # Peak memory should be reasonable (not accumulating all chunks)
        # Each chunk is ~20 * 500 * 100 * 8 bytes = 8 MB
        # Peak should be close to single chunk size, not sum of all chunks
        expected_chunk_mb = (20 * N * 100 * 8) / (1024 * 1024)
        assert result.peak_memory_mb < expected_chunk_mb * 3  # Allow some overhead

    def test_streaming_ic_peak_memory_includes_growing_accumulators(self):
        """Peak accounting includes persistent IC state, not only the last chunk."""
        evaluator = StreamingEvaluator()
        evaluator.register_streaming_metric(
            "ic", streaming_ic_updater, MetricKind.IC
        )
        num_chunks, num_assets, num_factors = 20, 10, 100

        def chunks():
            for time_value in range(num_chunks):
                factor_values = np.arange(
                    num_assets * num_factors, dtype=np.float64
                ).reshape(1, num_assets, num_factors)
                batch = FactorBatch(
                    factor_ids=tuple(f"f{i}" for i in range(num_factors)),
                    time_axis=AxisRef(
                        "time", "int64", 1, np.array([time_value])
                    ),
                    asset_axis=AxisRef(
                        "asset", "int64", num_assets, np.arange(num_assets)
                    ),
                    values=factor_values,
                )
                labels = LabelBundle(
                    target_id="target",
                    values=np.arange(num_assets, dtype=np.float64).reshape(1, -1),
                    horizon=1,
                    decision_time=(time_value,),
                    label_start_time=(time_value,),
                    label_end_time=(time_value + 1,),
                )
                yield batch, labels

        result = evaluator.evaluate_stream(
            chunks(), [{"metric_id": "ic", "metric_kind": "ic"}]
        )

        accumulator_mb = (
            num_chunks * num_factors * (5 * np.dtype(np.float64).itemsize
                                        + np.dtype(np.int32).itemsize)
            / (1024 * 1024)
        )
        assert result.metrics["ic"].shape == (num_chunks, num_factors)
        assert result.peak_memory_mb >= accumulator_mb

    def test_invalid_chunk_raises(self):
        evaluator = StreamingEvaluator()

        evaluator.register_streaming_metric(
            "test",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        def bad_generator():
            batch = self.create_test_batch(T=100, N=200, F=1)
            labels = self.create_test_labels(T=50, N=200)  # Wrong time dimension
            yield batch, labels

        metric_specs = [{"metric_id": "test", "metric_kind": "coverage"}]

        with pytest.raises(InvalidContractError, match="does not match"):
            evaluator.evaluate_stream(bad_generator(), metric_specs)

    def test_factor_time_coordinates_must_match_label_decision_time(self):
        evaluator = StreamingEvaluator()
        evaluator.register_streaming_metric(
            "coverage", streaming_coverage_updater, MetricKind.COVERAGE
        )
        batch = FactorBatch(
            factor_ids=("f",),
            time_axis=AxisRef("time", "int64", 2, np.array([100, 101])),
            asset_axis=AxisRef("asset", "int64", 1, np.array(["a"])),
            values=np.ones((2, 1, 1)),
        )
        labels = LabelBundle(
            target_id="target",
            values=np.ones((2, 1)),
            horizon=1,
            decision_time=(0, 1),
            label_start_time=(0, 1),
            label_end_time=(1, 2),
        )

        with pytest.raises(
            InvalidContractError,
            match="factor time coordinates do not match label decision_time",
        ):
            evaluator.evaluate_large_batch(
                batch, labels, [{"metric_id": "coverage", "metric_kind": "coverage"}]
            )

    def test_unregistered_metric_raises(self):
        evaluator = StreamingEvaluator()

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "unknown", "metric_kind": "custom"}]

        with pytest.raises(InvalidContractError, match="not registered"):
            evaluator.evaluate_large_batch(batch, labels, metric_specs)

    def test_budget_tracking(self):
        budget = ComputationBudget(max_operations=3, allow_overflow=False)
        evaluator = StreamingEvaluator(
            budget=budget,
            chunk_size_time=50,
            chunk_size_factors=2,
        )

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        # Create batch that will generate many chunks
        batch = self.create_test_batch(T=100, N=200, F=5)
        labels = self.create_test_labels(T=100, N=200)

        metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]

        # Should exceed budget
        with pytest.raises(RuntimeError, match="budget exceeded"):
            evaluator.evaluate_large_batch(batch, labels, metric_specs)

    def test_metadata_tracking(self):
        evaluator = StreamingEvaluator(
            chunk_size_time=30,
            chunk_size_factors=100,
        )

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        batch = self.create_test_batch(T=100, N=200, F=3)
        labels = self.create_test_labels(T=100, N=200)

        metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]

        result = evaluator.evaluate_large_batch(batch, labels, metric_specs)

        assert "chunk_size_time" in result.metadata
        assert result.metadata["chunk_size_time"] == 30
        assert result.metadata["chunk_size_factors"] == 100
        assert result.metadata["num_metrics"] == 1

    def test_empty_generator(self):
        evaluator = StreamingEvaluator()

        evaluator.register_streaming_metric(
            "coverage",
            streaming_coverage_updater,
            MetricKind.COVERAGE,
        )

        def empty_generator():
            return
            yield  # Never executed

        metric_specs = [{"metric_id": "coverage", "metric_kind": "coverage"}]

        with pytest.raises(InsufficientObservations, match="no chunks"):
            evaluator.evaluate_stream(empty_generator(), metric_specs)

    def _boundary_chunks(self):
        values = np.ones((2, 1, 1))
        def make(start, *, factor="f", context=None, target="target"):
            batch = FactorBatch(
                factor_ids=(factor,),
                time_axis=AxisRef("time", "int64", 2, np.array([start, start + 1])),
                asset_axis=AxisRef("asset", "int64", 1, np.array(["asset-1"])),
                values=values,
                context_refs=context or {"snapshot": "s1"},
            )
            labels = LabelBundle(
                target_id=target, values=np.ones((2, 1)), horizon=1,
                decision_time=(start, start + 1), label_start_time=(start, start + 1),
                label_end_time=(start + 1, start + 2), source_ref="labels", calendar_ref="cal",
            )
            return batch, labels
        return make

    def _boundary_evaluator(self):
        evaluator = StreamingEvaluator()
        evaluator.register_streaming_metric("coverage", streaming_coverage_updater, MetricKind.COVERAGE)
        return evaluator, [{"metric_id": "coverage", "metric_kind": "coverage"}]

    def test_stream_reverse_order_rejected(self):
        make = self._boundary_chunks(); evaluator, specs = self._boundary_evaluator()
        with pytest.raises(InvalidContractError, match="monotonic"):
            evaluator.evaluate_stream(iter([make(2), make(0)]), specs)

    def test_stream_replay_rejected(self):
        make = self._boundary_chunks(); evaluator, specs = self._boundary_evaluator()
        with pytest.raises(InvalidContractError, match="duplicate|replayed"):
            evaluator.evaluate_stream(iter([make(0), make(0)]), specs)

    @pytest.mark.parametrize("change", ["context", "identity"])
    def test_stream_context_or_identity_change_rejected(self, change):
        make = self._boundary_chunks(); evaluator, specs = self._boundary_evaluator()
        first = make(0)
        second = make(2, context={"snapshot": "s2"} if change == "context" else {"snapshot": "s1"},
                      factor="g" if change == "identity" else "f")
        with pytest.raises(SnapshotMismatchError, match="identity/context"):
            evaluator.evaluate_stream(iter([first, second]), specs)

    def test_stream_overlap_rejected(self):
        make = self._boundary_chunks(); evaluator, specs = self._boundary_evaluator()
        with pytest.raises(InvalidContractError, match="overlapping|duplicate|replayed"):
            evaluator.evaluate_stream(iter([make(0), make(1)]), specs)

    def test_stream_accepts_orderable_unhashable_time_coordinates(self):
        class UnhashableInt(int):
            __hash__ = None

        evaluator, specs = self._boundary_evaluator()

        def make_chunk(start):
            times = np.array(
                [UnhashableInt(start), UnhashableInt(start + 1)], dtype=object
            )
            batch = FactorBatch(
                factor_ids=("f",),
                time_axis=AxisRef("time", "object", 2, times),
                asset_axis=AxisRef("asset", "int64", 1, np.array(["asset-1"])),
                values=np.ones((2, 1, 1)),
                context_refs={"snapshot": "s1"},
            )
            labels = LabelBundle(
                target_id="target", values=np.ones((2, 1)), horizon=1,
                decision_time=(start, start + 1),
                label_start_time=(start, start + 1),
                label_end_time=(start + 1, start + 2),
                source_ref="labels", calendar_ref="cal",
            )
            return batch, labels

        result = evaluator.evaluate_stream(
            iter([make_chunk(0), make_chunk(2)]), specs
        )

        assert result.chunks_processed == 2
        assert result.provenance["timing_first"][0] == (0, 1)
        assert result.provenance["timing_last"][0] == (0, 1, 2, 3)

    def test_public_timing_rule_allows_unequal_chunk_lengths_and_accumulates_vectors(self):
        evaluator, specs = self._boundary_evaluator()

        def make_chunk(times):
            values = np.ones((len(times), 1, 1))
            batch = FactorBatch(
                factor_ids=("f",),
                time_axis=AxisRef("time", "int64", len(times), np.asarray(times)),
                asset_axis=AxisRef("asset", "int64", 1, np.array(["a"])),
                values=values,
                context_refs={"snapshot": "s1"},
            )
            labels = LabelBundle(
                target_id="target", values=np.ones((len(times), 1)), horizon=1,
                decision_time=tuple(times), execution_time=tuple(t + 1 for t in times),
                label_start_time=tuple(t + 2 for t in times),
                label_end_time=tuple(t + 3 for t in times),
                source_ref="labels", calendar_ref="cal",
            )
            return batch, labels

        chunks = [make_chunk([0]), make_chunk([1, 2])]
        result = evaluator.evaluate_stream(iter(chunks), specs)
        full_batch = FactorBatch(
            factor_ids=("f",),
            time_axis=AxisRef("time", "int64", 3, np.array([0, 1, 2])),
            asset_axis=AxisRef("asset", "int64", 1, np.array(["a"])),
            values=np.ones((3, 1, 1)),
            context_refs={"snapshot": "s1"},
        )
        full_labels = LabelBundle(
            target_id="target", values=np.ones((3, 1)), horizon=1,
            decision_time=(0, 1, 2), execution_time=(1, 2, 3),
            label_start_time=(2, 3, 4), label_end_time=(3, 4, 5),
            source_ref="labels", calendar_ref="cal",
        )
        large_result = evaluator.evaluate_large_batch(full_batch, full_labels, specs)

        assert result.provenance["timing_rows"] == large_result.provenance["timing_rows"]
        assert result.provenance["timing_rule"] == large_result.provenance["timing_rule"]
        assert result.provenance["timing_last"] == large_result.provenance["timing_last"]
        assert result.provenance["timing_rows"] == 3
        assert result.provenance["timing_first"][0] == (0,)
        assert result.provenance["timing_last"][0] == (0, 1, 2)

    def test_public_timing_rule_drift_rejected_with_unequal_chunks(self):
        evaluator, specs = self._boundary_evaluator()

        def make_chunk(times, end_delta=3):
            values = np.ones((len(times), 1, 1))
            batch = FactorBatch(
                factor_ids=("f",),
                time_axis=AxisRef("time", "int64", len(times), np.asarray(times)),
                asset_axis=AxisRef("asset", "int64", 1, np.array(["a"])),
                values=values,
                context_refs={"snapshot": "s1"},
            )
            labels = LabelBundle(
                target_id="target", values=np.ones((len(times), 1)), horizon=1,
                decision_time=tuple(times), execution_time=tuple(t + 1 for t in times),
                label_start_time=tuple(t + 2 for t in times),
                label_end_time=tuple(t + end_delta for t in times),
                source_ref="labels", calendar_ref="cal",
            )
            return batch, labels

        with pytest.raises(SnapshotMismatchError, match="timing"):
            evaluator.evaluate_stream(
                iter([make_chunk([0]), make_chunk([1, 2], end_delta=4)]), specs
            )

    def test_get_budget_report(self):
        budget = ComputationBudget(max_memory_mb=1024.0)
        evaluator = StreamingEvaluator(budget=budget)

        report = evaluator.get_budget_report()

        assert isinstance(report, str)
        assert "Memory" in report


class TestStreamingUpdaters:
    def create_simple_batch(self, T=10, N=20, F=1, constant_value=None):
        """Helper to create simple test batch."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        if constant_value is not None:
            values = np.full((T, N, F), constant_value, dtype=np.float64)
        else:
            values = np.random.randn(T, N, F)

        return FactorBatch(
            factor_ids=tuple(f"factor_{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

    def create_simple_labels(self, T=10, N=20, constant_value=None):
        """Helper to create simple label bundle."""
        if constant_value is not None:
            values = np.full((T, N), constant_value, dtype=np.float64)
        else:
            values = np.random.randn(T, N)

        return LabelBundle(
            target_id="forward_return_1d",
            values=values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

    def test_streaming_coverage_broadcasts_one_dimensional_label_validity(self):
        batch = self.create_simple_batch(T=3, N=2, F=1)
        labels = LabelBundle(
            target_id="target",
            values=np.array([1.0, 2.0, 3.0]),
            validity=np.array([True, False, True]),
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )
        state = StreamingMetricState("coverage", MetricKind.COVERAGE)

        result = streaming_coverage_updater(state, batch, labels).finalize()

        assert state.valid_count == 4
        assert state.total_count == 6
        assert result == pytest.approx(2 / 3)

    def test_streaming_ic_broadcasts_one_dimensional_label_validity(self):
        batch = self.create_simple_batch(T=3, N=2, F=1)
        labels = LabelBundle(
            target_id="target",
            values=np.array([1.0, 2.0, 3.0]),
            validity=np.array([True, False, True]),
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )
        state = StreamingMetricState("ic", MetricKind.IC)

        updated = streaming_ic_updater(state, batch, labels)

        assert updated.valid_counts[:, 0].tolist() == [2, 0, 2]

        state = StreamingMetricState(
            metric_id="coverage",
            metric_kind=MetricKind.COVERAGE,
        )

        batch = self.create_simple_batch(T=10, N=20, F=2)
        labels = self.create_simple_labels(T=10, N=20)

        updated_state = streaming_coverage_updater(state, batch, labels)

        assert updated_state.count == 1
        assert updated_state.valid_count == updated_state.total_count == 10 * 20 * 2
        assert updated_state.finalize() == 1.0

    def test_streaming_coverage_uses_paired_valid_denominator(self):
        """Each factor-label cell is eligible; NaNs and validity remove paired cells."""
        factors = np.array([
            [[1.0, np.nan], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ])
        factor_validity = np.ones_like(factors, dtype=bool)
        factor_validity[1, 0, 1] = False
        labels = np.array([[1.0, np.nan], [2.0, 3.0]])
        label_validity = np.ones_like(labels, dtype=bool)
        label_validity[1, 1] = False
        batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=AxisRef("time", "int64", 2),
            asset_axis=AxisRef("asset", "int64", 2),
            values=factors,
            validity=factor_validity,
        )
        label_bundle = LabelBundle(
            target_id="target", values=labels, horizon=1,
            decision_time=(0, 1), label_start_time=(0, 1), label_end_time=(1, 2),
            validity=label_validity,
        )
        state = StreamingMetricState("coverage", MetricKind.COVERAGE)

        result = streaming_coverage_updater(state, batch, label_bundle).finalize()

        # Eight paired cells: one factor NaN, one factor-invalid cell, two label-NaN
        # cells, and two label-invalid cells leave exactly two valid pairs.
        assert state.valid_count == 2
        assert state.total_count == 8
        assert result == pytest.approx(1 / 4)

    def test_streaming_summary_updater(self):
        state = StreamingMetricState(
            metric_id="summary",
            metric_kind=MetricKind.SUMMARY,
        )

        batch = self.create_simple_batch(T=10, N=20, F=1, constant_value=3.0)
        labels = self.create_simple_labels(T=10, N=20)

        updated_state = streaming_summary_updater(state, batch, labels)

        assert updated_state.count == 10 * 20 * 1
        assert abs(updated_state.sum_values - (3.0 * 10 * 20)) < 1e-6

    def test_streaming_ic_updater_initialization(self):
        state = StreamingMetricState(
            metric_id="ic",
            metric_kind=MetricKind.IC,
        )

        assert state.sum_x is None
        assert state.sum_y is None

        batch = self.create_simple_batch(T=5, N=10, F=2)
        labels = self.create_simple_labels(T=5, N=10)

        updated_state = streaming_ic_updater(state, batch, labels)

        # Should initialize accumulators
        assert updated_state.sum_x is not None
        assert updated_state.sum_x.shape == (5, 2)
        assert updated_state.valid_counts is not None

    def test_streaming_ic_updater_accumulation(self):
        state = StreamingMetricState(
            metric_id="ic",
            metric_kind=MetricKind.IC,
        )

        T, N, F = 5, 100, 1

        # First chunk
        batch1 = self.create_simple_batch(T=T, N=N, F=F)
        labels1 = self.create_simple_labels(T=T, N=N)
        state = streaming_ic_updater(state, batch1, labels1)

        first_sum_x = state.sum_x.copy()
        first_counts = state.valid_counts.copy()
        first_shape = state.sum_x.shape

        # Second chunk with same dimensions - interpreted as new time periods
        batch2 = self.create_simple_batch(T=T, N=N, F=F)
        labels2 = self.create_simple_labels(T=T, N=N)
        state = streaming_ic_updater(state, batch2, labels2)

        # With same T and F, accumulator should concatenate along time dimension
        assert state.sum_x.shape[0] == first_shape[0] + T  # Time dimension grows
        assert state.sum_x.shape[1] == first_shape[1]  # Factor dimension stays same
        assert state.accumulated_time == T * 2

    def test_streaming_ic_perfect_correlation(self):
        state = StreamingMetricState(
            metric_id="ic",
            metric_kind=MetricKind.IC,
        )

        T, N, F = 1, 100, 1

        # Create perfectly correlated data
        np.random.seed(123)
        factor_vals = np.random.randn(T, N, F)
        label_vals = factor_vals[:, :, 0]  # Perfect correlation

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=("factor_0",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_vals,
        )

        labels = LabelBundle(
            target_id="forward_return_1d",
            values=label_vals,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        state = streaming_ic_updater(state, batch, labels)
        ic_result = state.finalize()

        assert ic_result.shape == (T, F)
        assert abs(ic_result[0, 0] - 1.0) < 1e-6  # Perfect correlation
