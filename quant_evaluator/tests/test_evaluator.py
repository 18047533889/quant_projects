"""
Tests for main Evaluator class.
"""

import pytest
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError

from quant_evaluator.runtime.evaluator import Evaluator, EvaluationResult
from quant_evaluator.runtime.budgets import ComputationBudget
from quant_evaluator.planner.dependency_plan import MetricKind, resolve_metric_dependencies


class TestEvaluationResult:
    def test_result_creation(self):
        result = EvaluationResult()

        assert result.metrics == {}
        assert result.diagnostics == {}
        assert result.execution_time_seconds == 0.0

    def test_get_metric(self):
        result = EvaluationResult()
        result.metrics["ic_mean"] = 0.05

        assert result.get_metric("ic_mean") == 0.05
        assert result.get_metric("missing") is None
        assert result.get_metric("missing", default=0.0) == 0.0

    def test_has_metric(self):
        result = EvaluationResult()
        result.metrics["ic_mean"] = 0.05

        assert result.has_metric("ic_mean")
        assert not result.has_metric("missing")

    def test_to_dict(self):
        result = EvaluationResult()
        result.metrics["ic_mean"] = 0.05
        result.execution_time_seconds = 1.5

        data = result.to_dict()

        assert data["metrics"]["ic_mean"] == 0.05
        assert data["execution_time_seconds"] == 1.5


class TestEvaluator:
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
        evaluator = Evaluator()

        assert evaluator.cache is not None
        assert evaluator.budget is not None
        assert evaluator.budget_tracker is not None

    def test_evaluator_with_budget(self):
        budget = ComputationBudget(max_memory_mb=1024.0)
        evaluator = Evaluator(budget=budget)

        assert evaluator.budget == budget
        assert evaluator.budget_tracker.budget == budget

    def test_register_metric(self):
        evaluator = Evaluator()

        def dummy_metric(**kwargs):
            return 0.5

        evaluator.register_metric("test_metric", dummy_metric, MetricKind.CUSTOM)

        assert "test_metric" in evaluator._metric_functions

    def test_evaluate_simple_metric(self):
        evaluator = Evaluator()

        # Register a simple metric
        def coverage_metric(factor_batch, label_bundle, **kwargs):
            return 0.95

        evaluator.register_metric("coverage", coverage_metric, MetricKind.COVERAGE)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [
            {
                "metric_id": "coverage",
                "metric_kind": "coverage",
            }
        ]

        result = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

        assert result.has_metric("coverage")
        assert result.get_metric("coverage") == 0.95
        assert result.execution_time_seconds > 0

    def test_evaluate_with_dependencies(self):
        evaluator = Evaluator()

        # Register metrics with dependencies
        def base_metric(factor_batch, label_bundle, **kwargs):
            return np.array([1, 2, 3])

        def derived_metric(factor_batch, label_bundle, computed_metrics, **kwargs):
            base_value = computed_metrics.get("base_metric")
            return np.mean(base_value)

        evaluator.register_metric("base_metric", base_metric)
        evaluator.register_metric("derived_metric", derived_metric)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [
            {
                "metric_id": "base_metric",
                "metric_kind": "custom",
            },
            {
                "metric_id": "derived_metric",
                "metric_kind": "custom",
                "dependencies": ["base_metric"],
            },
        ]

        result = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

        assert result.has_metric("base_metric")
        assert result.has_metric("derived_metric")
        assert result.get_metric("derived_metric") == 2.0

    def test_registry_compute_fn_fallback_filters_kwargs_and_forwards_dependencies(self):
        evaluator = Evaluator(enable_cache=False)

        batch = self.create_test_batch(T=25, N=10, F=1)
        labels = self.create_test_labels(T=25, N=10)
        result = evaluator.evaluate(
            batch,
            labels,
            [{"metric_id": "coverage", "metric_kind": "custom"}],
            use_chunking=False,
        )

        # Registry-computed coverage is a scalar fraction (per-factor adapter).
        assert result.get_metric("coverage") == 1.0

    def test_evaluate_with_caching(self):
        evaluator = Evaluator(enable_cache=True)

        call_count = [0]

        def counting_metric(factor_batch, label_bundle, **kwargs):
            call_count[0] += 1
            return 0.5

        evaluator.register_metric("test_metric", counting_metric)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "test_metric", "metric_kind": "custom"}]

        # First evaluation
        result1 = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
        assert call_count[0] == 1
        assert result1.cache_misses == 1
        assert result1.cache_hits == 0

        # Mutable closure state is intentionally uncacheable.
        result2 = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
        assert call_count[0] == 2
        assert result2.cache_hits == 0
        assert result2.cache_misses == 1

    def test_evaluate_cache_disabled(self):
        evaluator = Evaluator(enable_cache=False)

        call_count = [0]

        def counting_metric(factor_batch, label_bundle, **kwargs):
            call_count[0] += 1
            return 0.5

        evaluator.register_metric("test_metric", counting_metric)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "test_metric", "metric_kind": "custom"}]

        # First evaluation
        evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
        assert call_count[0] == 1

        # Second evaluation (should recompute)
        evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
        assert call_count[0] == 2

    def test_evaluate_with_chunking(self):
        evaluator = Evaluator(max_chunk_memory_mb=10.0)  # Very small to force chunking

        def simple_metric(factor_batch, label_bundle, **kwargs):
            return factor_batch.num_times

        evaluator.register_metric("time_count", simple_metric)

        batch = self.create_test_batch(T=500, N=2000, F=5)  # Larger batch
        labels = self.create_test_labels(T=500, N=2000)

        metric_specs = [{"metric_id": "time_count", "metric_kind": "custom"}]

        result = evaluator.evaluate(batch, labels, metric_specs, use_chunking=True)

        assert result.has_metric("time_count")
        # With small memory budget, should create multiple chunks
        assert result.chunks_processed >= 1

    def test_validate_inputs_mismatch_raises(self):
        evaluator = Evaluator()

        batch = self.create_test_batch(T=100, N=500, F=1)
        labels = self.create_test_labels(T=50, N=500)  # Wrong time dimension

        metric_specs = [{"metric_id": "test", "metric_kind": "custom"}]

        with pytest.raises(InvalidContractError, match="does not match"):
            evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

    def test_unregistered_metric_raises(self):
        evaluator = Evaluator()

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "unknown_metric", "metric_kind": "custom"}]

        with pytest.raises(InvalidContractError, match="not registered"):
            evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

    def test_metric_computation_error_raises(self):
        evaluator = Evaluator()

        def failing_metric(**kwargs):
            raise ValueError("Computation failed")

        evaluator.register_metric("failing", failing_metric)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "failing", "metric_kind": "custom"}]

        with pytest.raises(RuntimeError, match="Error computing metric"):
            evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

    def test_clear_cache(self):
        evaluator = Evaluator(enable_cache=True)

        def simple_metric(**kwargs):
            return 0.5

        evaluator.register_metric("test", simple_metric)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "test", "metric_kind": "custom"}]

        # Populate cache
        evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)
        assert evaluator.cache.num_entries > 0

        # Clear cache
        evaluator.clear_cache()
        assert evaluator.cache.num_entries == 0

    def test_get_cache_stats(self):
        evaluator = Evaluator(enable_cache=True)

        stats = evaluator.get_cache_stats()

        assert "num_entries" in stats
        assert "size_mb" in stats
        assert stats["num_entries"] == 0

    def test_get_budget_report(self):
        budget = ComputationBudget(max_memory_mb=1024.0)
        evaluator = Evaluator(budget=budget)

        report = evaluator.get_budget_report()

        assert isinstance(report, str)
        assert "Memory" in report

    def test_result_metadata(self):
        evaluator = Evaluator()

        def simple_metric(**kwargs):
            return 0.5

        evaluator.register_metric("test", simple_metric)

        batch = self.create_test_batch(T=50, N=100, F=2)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [{"metric_id": "test", "metric_kind": "custom"}]

        result = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

        assert "batch_shape" in result.metadata
        assert result.metadata["batch_shape"] == (50, 100, 2)
        assert result.metadata["num_metrics"] == 1

    def test_budget_exceeded_raises(self):
        budget = ComputationBudget(max_operations=2, allow_overflow=False)
        evaluator = Evaluator(budget=budget)

        def metric_with_ops(factor_batch, **kwargs):
            # This will cause budget tracker to record operations
            return 0.5

        evaluator.register_metric("test1", metric_with_ops)
        evaluator.register_metric("test2", metric_with_ops)
        evaluator.register_metric("test3", metric_with_ops)
        evaluator.register_metric("test4", metric_with_ops)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        # Create enough metrics to exceed operation budget during evaluation
        metric_specs = [
            {"metric_id": "test1", "metric_kind": "custom"},
            {"metric_id": "test2", "metric_kind": "custom"},
            {"metric_id": "test3", "metric_kind": "custom"},
            {"metric_id": "test4", "metric_kind": "custom"},
        ]

        # Evaluation will record operations and should exceed budget
        with pytest.raises(RuntimeError, match="budget exceeded"):
            evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

    def test_execution_order_respects_dependencies(self):
        evaluator = Evaluator()

        execution_order = []

        def metric_a(**kwargs):
            execution_order.append("A")
            return 1

        def metric_b(computed_metrics, **kwargs):
            execution_order.append("B")
            return computed_metrics["A"] + 1

        def metric_c(computed_metrics, **kwargs):
            execution_order.append("C")
            return computed_metrics["B"] + 1

        evaluator.register_metric("A", metric_a)
        evaluator.register_metric("B", metric_b)
        evaluator.register_metric("C", metric_c)

        batch = self.create_test_batch(T=50, N=100, F=1)
        labels = self.create_test_labels(T=50, N=100)

        metric_specs = [
            {"metric_id": "C", "metric_kind": "custom", "dependencies": ["B"]},
            {"metric_id": "B", "metric_kind": "custom", "dependencies": ["A"]},
            {"metric_id": "A", "metric_kind": "custom"},
        ]

        result = evaluator.evaluate(batch, labels, metric_specs, use_chunking=False)

        # Despite specs order, execution must be A -> B -> C
        assert execution_order == ["A", "B", "C"]
        assert result.get_metric("C") == 3

    def test_cache_binds_immutable_closure_state(self):
        evaluator = Evaluator(enable_cache=True)
        multiplier = 2
        calls = [0]

        def metric(**kwargs):
            calls[0] += 1
            return multiplier

        evaluator.register_metric("metric", metric)
        batch = self.create_test_batch(T=4, N=2)
        labels = self.create_test_labels(T=4, N=2)
        specs = [{"metric_id": "metric", "metric_kind": "custom"}]

        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 2
        multiplier = 3
        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 3
        assert calls[0] == 2

    def test_cache_bypasses_mutable_closure_and_callable_instance(self):
        batch = self.create_test_batch(T=4, N=2)
        labels = self.create_test_labels(T=4, N=2)
        specs = [{"metric_id": "metric", "metric_kind": "custom"}]

        config = {"value": 2}
        evaluator = Evaluator(enable_cache=True)
        evaluator.register_metric("metric", lambda **kwargs: config["value"])
        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 2
        config["value"] = 3
        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 3

        class CallableMetric:
            def __init__(self):
                self.value = 4

            def __call__(self, **kwargs):
                return self.value

        callable_metric = CallableMetric()
        evaluator = Evaluator(enable_cache=True)
        evaluator.register_metric("metric", callable_metric)
        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 4
        callable_metric.value = 5
        assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 5

    def test_cache_rejects_nested_mutable_closure_state(self):
        batch = self.create_test_batch(T=4, N=2)
        labels = self.create_test_labels(T=4, N=2)
        specs = [{"metric_id": "metric", "metric_kind": "custom"}]

        cases = [
            ([2], lambda value: value.__setitem__(0, 3), lambda value: value[0]),
            ({"value": 2}, lambda value: value.__setitem__("value", 3), lambda value: value["value"]),
            ({2}, lambda value: (value.clear(), value.add(3)), lambda value: next(iter(value))),
            (np.array([2]), lambda value: value.__setitem__(0, 3), lambda value: int(value[0])),
        ]
        for mutable, mutate, extract in cases:
            holder = (mutable,)

            def metric(**kwargs):
                return extract(holder[0])

            evaluator = Evaluator(enable_cache=True)
            evaluator.register_metric("metric", metric)
            assert evaluator.evaluate(batch, labels, specs, use_chunking=False).get_metric("metric") == 2
            mutate(mutable)
            second = evaluator.evaluate(batch, labels, specs, use_chunking=False)
            assert second.get_metric("metric") == 3
            assert second.cache_hits == 0

    def test_chunk_mean_requires_weighted_reduction_protocol(self):
        evaluator = Evaluator()
        graph = resolve_metric_dependencies([{
            "metric_id": "mean_metric",
            "metric_kind": "custom",
            "metadata": {"partitionability": "PARTITIONABLE", "aggregation": "mean"},
        }])

        with pytest.raises(InvalidContractError, match="weighted reduction"):
            evaluator._aggregate_chunk_results(
                [{"mean_metric": 1.0}, {"mean_metric": 9.0}],
                ["mean_metric"],
                graph,
            )

    def test_registry_metric_requires_declared_input(self):
        evaluator = Evaluator(enable_cache=False)
        batch = self.create_test_batch(T=25, N=10)
        labels = self.create_test_labels(T=25, N=10)

        with pytest.raises(InvalidContractError, match="missing required input.*ic_series"):
            evaluator.evaluate(
                batch,
                labels,
                [{"metric_id": "mean_ic", "metric_kind": "custom"}],
                use_chunking=False,
            )

    def test_registry_metric_rejects_insufficient_periods(self, monkeypatch):
        from quant_evaluator.registry.metrics import MetricSpec, MetricStatus, MetricTier
        import quant_evaluator.runtime.evaluator as evaluator_module

        spec = MetricSpec(
            name="period_metric",
            display_name="Period Metric",
            description="test metric",
            status=MetricStatus.STABLE,
            tier=MetricTier.CORE,
            compute_fn=lambda factor_batch, label_bundle: 1.0,
            requires=["factor_batch", "label_bundle"],
            min_periods=20,
        )
        monkeypatch.setattr(evaluator_module, "get_metric", lambda name: spec)
        evaluator = Evaluator(enable_cache=False)
        batch = self.create_test_batch(T=19, N=10)
        labels = self.create_test_labels(T=19, N=10)

        with pytest.raises(InvalidContractError, match="requires at least 20 periods"):
            evaluator.evaluate(
                batch,
                labels,
                [{"metric_id": "period_metric", "metric_kind": "custom"}],
                use_chunking=False,
            )
