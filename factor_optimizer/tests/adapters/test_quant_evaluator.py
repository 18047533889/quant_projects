"""Tests for QuantEvaluatorAdapter."""

import pytest
from factor_optimizer.adapters import (
    QuantEvaluatorAdapter,
    QEOptionalDependencyMissing,
    create_qe_adapter,
    create_mock_qe_adapter,
)


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

    # Get evidence
    evidence = adapter.get_evidence(eval_id)

    # Check structure
    assert isinstance(evidence, dict)
    assert evidence["evaluation_id"] == eval_id
    assert "metrics" in evidence
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


def test_create_qe_adapter_missing_dependency():
    """Test that missing QE raises appropriate error."""
    # QE doesn't exist yet, so this should always raise
    with pytest.raises(QEOptionalDependencyMissing):
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
