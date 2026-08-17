"""Tests for QuantEvaluatorAdapter."""

import sys
from types import SimpleNamespace

import numpy as np

from quant_evaluator import AxisRef, FactorBatch, LabelBundle

import pytest
from factor_optimizer.adapters import (
    EvidenceStore,
    InMemoryEvidenceStore,
    QuantEvaluatorAdapter,
    QEOptionalDependencyMissing,
    create_qe_adapter,
    create_mock_qe_adapter,
)
from factor_optimizer.errors import EvidenceUnavailableError


def test_mock_qe_adapter_protocol():
    """Test that mock QE adapter satisfies protocol."""
    adapter = create_mock_qe_adapter()

    # Check protocol compliance
    assert isinstance(adapter, QuantEvaluatorAdapter)

    # Check has required methods
    assert hasattr(adapter, "evaluate")
    assert hasattr(adapter, "get_evidence")
    assert hasattr(adapter, "list_metrics")


def test_mock_qe_adapter_evaluate():
    """Test mock evaluation."""
    adapter = create_mock_qe_adapter()

    # Mock factor batch and labels
    factor_batch = "mock_batch"
    labels = "mock_labels"

    # Evaluate
    result = adapter.evaluate(factor_batch, labels)

    # Check result structure
    assert isinstance(result, dict)
    assert "evaluation_id" in result
    assert "metrics" in result
    assert "diagnostics" in result
    assert "evidence_ref" in result

    # Check metrics
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    # Check diagnostics
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert "coverage" in diagnostics


def test_mock_qe_adapter_evaluate_with_metrics():
    """Test evaluation with specific metrics."""
    adapter = create_mock_qe_adapter()

    # Request specific metrics
    result = adapter.evaluate(
        factor_batch="mock_batch",
        labels="mock_labels",
        metrics=["rank_ic", "ic_mean"],
    )

    # Should only return requested metrics
    metrics = result["metrics"]
    assert "rank_ic" in metrics
    assert "ic_mean" in metrics


def test_mock_qe_adapter_get_evidence():
    """Test evidence retrieval."""
    adapter = create_mock_qe_adapter()

    # First evaluate to create evidence
    result = adapter.evaluate("mock_batch", "mock_labels")
    eval_id = result["evaluation_id"]

    # Get evidence through the published reference
    evidence = adapter.get_evidence(result["evidence_ref"])

    # Check structure and identity consistency
    assert isinstance(evidence, dict)
    assert result["evidence_ref"] == eval_id
    assert evidence["evaluation_id"] == eval_id
    assert evidence["metrics"] == result["metrics"]
    assert "diagnostics" in evidence
    assert "timeseries" in evidence


def test_mock_qe_adapter_get_evidence_not_found():
    """Test evidence retrieval for non-existent evaluation."""
    adapter = create_mock_qe_adapter()

    # Get evidence for non-existent ID
    evidence = adapter.get_evidence("nonexistent_id")

    # Should return structure with error
    assert isinstance(evidence, dict)
    assert evidence["evaluation_id"] == "nonexistent_id"
    assert "error" in evidence.get("diagnostics", {})


def test_mock_qe_adapter_list_metrics():
    """Test metrics listing."""
    adapter = create_mock_qe_adapter()

    # List all metrics
    metrics = adapter.list_metrics()

    # Check structure
    assert isinstance(metrics, list)
    assert len(metrics) > 0

    # Check first metric structure
    first = metrics[0]
    assert "metric_id" in first
    assert "name" in first
    assert "description" in first
    assert "tier" in first
    assert "higher_is_better" in first


def test_mock_qe_adapter_list_metrics_filtered():
    """Test metrics listing with tier filter."""
    adapter = create_mock_qe_adapter()

    # List core metrics only
    core_metrics = adapter.list_metrics(tier="core")
    assert all(m["tier"] == "core" for m in core_metrics)

    # List advanced metrics only
    advanced_metrics = adapter.list_metrics(tier="advanced")
    assert all(m["tier"] == "advanced" for m in advanced_metrics)

    # Should have fewer metrics when filtered
    all_metrics = adapter.list_metrics()
    assert len(core_metrics) < len(all_metrics)


def test_mock_qe_adapter_deterministic_results():
    """Test that mock results are deterministic for same eval_id."""
    adapter = create_mock_qe_adapter()

    # Run two evaluations
    result1 = adapter.evaluate("batch1", "labels1")
    result2 = adapter.evaluate("batch1", "labels1")

    # Different evaluation IDs should produce different results
    assert result1["evaluation_id"] != result2["evaluation_id"]
    assert result1["metrics"] != result2["metrics"]


def test_mock_qe_adapter_metric_ranges():
    """Test that mock metrics are in reasonable ranges."""
    adapter = create_mock_qe_adapter()

    # Run multiple evaluations to check ranges
    for _ in range(10):
        result = adapter.evaluate("mock_batch", "mock_labels")
        metrics = result["metrics"]

        # Check IC metrics are in reasonable range
        if "rank_ic" in metrics:
            assert -0.5 <= metrics["rank_ic"] <= 0.5

        if "ic_mean" in metrics:
            assert -0.5 <= metrics["ic_mean"] <= 0.5

        # Check turnover is positive
        if "turnover" in metrics:
            assert 0 <= metrics["turnover"] <= 1.0

        # Check coverage is reasonable
        coverage = result["diagnostics"].get("coverage", 0)
        assert 0 <= coverage <= 1.0


def _current_qe_fixture():
    times = np.arange(3)
    assets = np.arange(2)
    return (
        FactorBatch(
            factor_ids=("f1",),
            time_axis=AxisRef("time", "int64", 3, times),
            asset_axis=AxisRef("asset", "int64", 2, assets),
            values=np.ones((3, 2, 1), dtype=float),
        ),
        LabelBundle(
            target_id="y",
            values=np.ones((3, 2), dtype=float),
            horizon=1,
            decision_time=tuple(times),
            label_start_time=tuple(times),
            label_end_time=tuple(times),
        ),
    )


def test_current_qe_public_facade_adapter_uses_explicit_shared_evidence_store():
    """Evidence references resolve across adapters sharing the same store."""
    factor_batch, labels = _current_qe_fixture()
    store = InMemoryEvidenceStore()
    writer = create_qe_adapter(evidence_store=store)
    reader = create_qe_adapter(evidence_store=store)

    result = writer.evaluate(factor_batch, labels, metrics=["coverage"])

    assert isinstance(store, EvidenceStore)
    assert result["evaluation_id"] == result["evidence_ref"]
    assert result["evidence_scope"] == "process_local_shared_store"
    assert isinstance(result["metrics"]["coverage"], (int, float))
    assert "coverage" in result["metrics"]
    assert isinstance(result["diagnostics"]["f1"]["is_constant"], bool)
    assert isinstance(result["diagnostics"]["f1"]["has_nans"], bool)
    assert isinstance(result["diagnostics"]["f1"]["has_infs"], bool)

    evidence = reader.get_evidence(result["evaluation_id"])
    assert evidence["evidence_scope"] == "process_local_shared_store"
    assert evidence["metrics"] == result["metrics"]


def test_current_qe_adapter_requires_store_before_evaluation():
    factor_batch, labels = _current_qe_fixture()
    adapter = create_qe_adapter()

    with pytest.raises(EvidenceUnavailableError, match="required before evaluation"):
        adapter.evaluate(factor_batch, labels, metrics=["coverage"])


def test_current_qe_adapter_missing_evidence_fails_closed():
    store = InMemoryEvidenceStore()
    adapter = create_qe_adapter(evidence_store=store)

    with pytest.raises(EvidenceUnavailableError, match="evidence not found"):
        adapter.get_evidence("missing")


def test_in_memory_evidence_store_isolates_stored_and_retrieved_values():
    store = InMemoryEvidenceStore()
    source = {"metrics": {"coverage": 1.0}}
    store.put("evaluation", source)
    source["metrics"]["coverage"] = 0.0

    first = store.get("evaluation")
    assert first == {"metrics": {"coverage": 1.0}}
    first["metrics"]["coverage"] = -1.0

    assert store.get("evaluation") == {"metrics": {"coverage": 1.0}}


def test_current_qe_metric_catalog_does_not_invent_direction():
    """QE registry metadata lacks higher_is_better and the adapter preserves that fact."""
    adapter = create_qe_adapter()
    metrics = adapter.list_metrics(tier="core")

    assert metrics
    assert all(item["tier"] == "core" for item in metrics)
    assert all("higher_is_better" not in item for item in metrics)


def test_create_qe_adapter_rejects_incompatible_public_facade(monkeypatch):
    """An API-incompatible QE facade must be rejected at construction."""
    fake_qe = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "quant_evaluator", fake_qe)
    with pytest.raises(QEOptionalDependencyMissing, match="unavailable or incompatible"):
        create_qe_adapter()


def test_create_qe_adapter_rejects_partial_historical_contract(monkeypatch):
    """A matching evaluate signature alone must not pass compatibility checks."""

    class PartialEvaluator:
        def evaluate(self, factors, labels, metrics, context):
            raise AssertionError("evaluation must not start during construction")

    fake_qe = SimpleNamespace(
        Evaluator=PartialEvaluator,
        DEFAULT_METRICS=("rank_ic",),
        MetricCatalog=type(
            "MetricCatalog",
            (),
            {"list": lambda self, tier=None: []},
        ),
    )
    monkeypatch.setitem(sys.modules, "quant_evaluator", fake_qe)

    with pytest.raises(QEOptionalDependencyMissing, match="unavailable or incompatible"):
        create_qe_adapter()


def test_create_qe_adapter_rejects_positional_only_historical_contract(monkeypatch):
    """Methods called by keyword must accept the required keyword arguments."""

    class PositionalOnlyEvaluator:
        def evaluate(self, factors, labels, metrics, context, /):
            raise AssertionError("evaluation must not start during construction")

        def get_evidence(self, evaluation_id, /):
            raise AssertionError("evidence retrieval must not start during construction")

    fake_qe = SimpleNamespace(
        Evaluator=PositionalOnlyEvaluator,
        DEFAULT_METRICS=("rank_ic",),
        MetricCatalog=type(
            "MetricCatalog",
            (),
            {"list": lambda self, tier=None, /: []},
        ),
    )
    monkeypatch.setitem(sys.modules, "quant_evaluator", fake_qe)

    with pytest.raises(QEOptionalDependencyMissing, match="unavailable or incompatible"):
        create_qe_adapter()


def test_qe_adapter_protocol_interface():
    """Test that protocol defines expected interface."""
    # Check protocol has required methods
    required_methods = [
        "evaluate",
        "get_evidence",
        "list_metrics",
    ]

    for method in required_methods:
        assert hasattr(QuantEvaluatorAdapter, method)


def test_mock_adapter_multiple_evaluations():
    """Test multiple evaluations with same adapter instance."""
    adapter = create_mock_qe_adapter()

    # Run multiple evaluations
    results = []
    for i in range(5):
        result = adapter.evaluate(f"batch_{i}", f"labels_{i}")
        results.append(result)

    # All should have unique IDs
    eval_ids = [r["evaluation_id"] for r in results]
    assert len(eval_ids) == len(set(eval_ids))

    # All should have metrics
    for result in results:
        assert len(result["metrics"]) > 0

    # Should be able to retrieve all evidence
    for eval_id in eval_ids:
        evidence = adapter.get_evidence(eval_id)
        assert evidence["evaluation_id"] == eval_id


def test_mock_adapter_context_parameter():
    """Test evaluation with context parameter."""
    adapter = create_mock_qe_adapter()

    # Evaluate with context
    context = {
        "universe": "top500",
        "period": "2020-01-01_2023-12-31",
        "frequency": "daily",
    }

    result = adapter.evaluate(
        factor_batch="mock_batch",
        labels="mock_labels",
        context=context,
    )

    # Should complete successfully
    assert "evaluation_id" in result
    assert "metrics" in result


def test_mock_adapter_metrics_have_expected_keys():
    """Test that mock metrics include expected keys."""
    adapter = create_mock_qe_adapter()

    # Get metrics catalog
    catalog = adapter.list_metrics()

    expected_metrics = {"rank_ic", "ic_mean", "ic_std", "turnover", "sharpe"}
    found_metrics = {m["metric_id"] for m in catalog}

    # All expected metrics should be present
    assert expected_metrics.issubset(found_metrics)


def test_mock_adapter_higher_is_better_flag():
    """Test that higher_is_better flag is correct."""
    adapter = create_mock_qe_adapter()

    catalog = adapter.list_metrics()

    # Check specific metrics
    for metric in catalog:
        if metric["metric_id"] in ["rank_ic", "ic_mean", "sharpe"]:
            assert metric["higher_is_better"] is True
        elif metric["metric_id"] in ["ic_std", "turnover"]:
            assert metric["higher_is_better"] is False


def test_mock_adapter_evaluation_diagnostics():
    """Test that evaluation diagnostics are present."""
    adapter = create_mock_qe_adapter()

    result = adapter.evaluate("mock_batch", "mock_labels")

    diagnostics = result["diagnostics"]
    assert "coverage" in diagnostics
    assert "warnings" in diagnostics
    assert isinstance(diagnostics["warnings"], list)
    assert "evaluated_at" in diagnostics


def test_mock_adapter_evidence_structure():
    """Test evidence bundle structure."""
    adapter = create_mock_qe_adapter()

    # Evaluate and get evidence
    result = adapter.evaluate("mock_batch", "mock_labels")
    evidence = adapter.get_evidence(result["evaluation_id"])

    # Check all required keys
    required_keys = ["evaluation_id", "metrics", "diagnostics", "timeseries"]
    for key in required_keys:
        assert key in evidence

    # Metrics should match evaluation metrics
    assert evidence["metrics"] == result["metrics"]
