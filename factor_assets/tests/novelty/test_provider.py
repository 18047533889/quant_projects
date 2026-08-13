"""
Tests for QE evidence provider protocol.
"""

import pytest

from factor_assets.novelty import (
    EvidenceQuery,
    EvidenceResult,
    MockEvidenceProvider,
)


def test_evidence_query_creation():
    """Test EvidenceQuery creation."""
    query = EvidenceQuery(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        metric_names=("sharpe", "ic"),
        universe_ref="US_500",
        period_start="2024-01-01",
        period_end="2024-12-31",
    )

    assert query.factor_id == "F001"
    assert query.evaluation_run_id == "RUN_123"
    assert query.metric_names == ("sharpe", "ic")
    assert query.universe_ref == "US_500"


def test_evidence_query_requires_factor_id():
    """EvidenceQuery must have factor_id."""
    with pytest.raises(ValueError, match="factor_id"):
        EvidenceQuery(factor_id="")


def test_evidence_result_creation():
    """Test EvidenceResult creation."""
    result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        primary_metric_name="sharpe",
        primary_metric_value=1.5,
        secondary_metrics={"ic": 0.05, "turnover": 0.3},
        universe_ref="US_500",
        qe_version="1.0",
    )

    assert result.factor_id == "F001"
    assert result.available
    assert result.primary_metric_value == 1.5
    assert result.secondary_metrics["ic"] == 0.05


def test_evidence_result_requires_fields():
    """EvidenceResult must have required fields."""
    with pytest.raises(ValueError, match="factor_id"):
        EvidenceResult(
            factor_id="",
            evaluation_run_id="RUN_123",
            evidence_id="EVD_001",
            timestamp="2024-01-01T00:00:00Z",
            available=True,
        )

    with pytest.raises(ValueError, match="evaluation_run_id"):
        EvidenceResult(
            factor_id="F001",
            evaluation_run_id="",
            evidence_id="EVD_001",
            timestamp="2024-01-01T00:00:00Z",
            available=True,
        )


def test_evidence_result_has_warnings():
    """Test has_warnings property."""
    result_no_warnings = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )
    assert not result_no_warnings.has_warnings

    result_with_warnings = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        warnings=("low_sample_size", "high_turnover"),
    )
    assert result_with_warnings.has_warnings


def test_evidence_result_is_valid():
    """Test is_valid property."""
    valid_result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )
    assert valid_result.is_valid

    unavailable_result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=False,
    )
    assert not unavailable_result.is_valid

    result_with_warnings = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        warnings=("warning",),
    )
    assert not result_with_warnings.is_valid


def test_mock_evidence_provider_add_and_get():
    """Test MockEvidenceProvider add and retrieve."""
    provider = MockEvidenceProvider()

    result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        primary_metric_name="sharpe",
        primary_metric_value=1.5,
    )

    provider.add_evidence(result)

    query = EvidenceQuery(factor_id="F001")
    retrieved = provider.get_evidence(query)

    assert retrieved is not None
    assert retrieved.factor_id == "F001"
    assert retrieved.primary_metric_value == 1.5


def test_mock_evidence_provider_has_evidence():
    """Test MockEvidenceProvider has_evidence."""
    provider = MockEvidenceProvider()

    assert not provider.has_evidence("F001")

    result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )
    provider.add_evidence(result)

    assert provider.has_evidence("F001")
    assert not provider.has_evidence("F999")


def test_mock_evidence_provider_has_evidence_by_run():
    """Test has_evidence with specific run ID."""
    provider = MockEvidenceProvider()

    result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )
    provider.add_evidence(result)

    assert provider.has_evidence("F001", "RUN_123")
    assert not provider.has_evidence("F001", "RUN_999")


def test_mock_evidence_provider_get_latest():
    """Test getting latest evidence."""
    provider = MockEvidenceProvider()

    # Add older evidence
    result1 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        primary_metric_value=1.0,
    )
    provider.add_evidence(result1)

    # Add newer evidence
    result2 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_124",
        evidence_id="EVD_002",
        timestamp="2024-02-01T00:00:00Z",
        available=True,
        primary_metric_value=2.0,
    )
    provider.add_evidence(result2)

    latest = provider.get_latest_evidence("F001")
    assert latest is not None
    assert latest.evaluation_run_id == "RUN_124"
    assert latest.primary_metric_value == 2.0


def test_mock_evidence_provider_query_by_run_id():
    """Test querying evidence by run ID."""
    provider = MockEvidenceProvider()

    result1 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )
    provider.add_evidence(result1)

    result2 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_124",
        evidence_id="EVD_002",
        timestamp="2024-02-01T00:00:00Z",
        available=True,
    )
    provider.add_evidence(result2)

    query = EvidenceQuery(factor_id="F001", evaluation_run_id="RUN_123")
    retrieved = provider.get_evidence(query)

    assert retrieved is not None
    assert retrieved.evaluation_run_id == "RUN_123"


def test_mock_evidence_provider_query_by_metric():
    """Test querying evidence by metric name."""
    provider = MockEvidenceProvider()

    result1 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
        primary_metric_name="sharpe",
    )
    provider.add_evidence(result1)

    result2 = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_124",
        evidence_id="EVD_002",
        timestamp="2024-02-01T00:00:00Z",
        available=True,
        primary_metric_name="ic",
    )
    provider.add_evidence(result2)

    query = EvidenceQuery(factor_id="F001", metric_names=("sharpe",))
    retrieved = provider.get_evidence(query)

    assert retrieved is not None
    assert retrieved.primary_metric_name == "sharpe"


def test_mock_evidence_provider_count():
    """Test counting stored evidence."""
    provider = MockEvidenceProvider()

    assert provider.count() == 0

    provider.add_evidence(
        EvidenceResult(
            factor_id="F001",
            evaluation_run_id="RUN_123",
            evidence_id="EVD_001",
            timestamp="2024-01-01T00:00:00Z",
            available=True,
        )
    )
    assert provider.count() == 1

    provider.add_evidence(
        EvidenceResult(
            factor_id="F002",
            evaluation_run_id="RUN_124",
            evidence_id="EVD_002",
            timestamp="2024-01-01T00:00:00Z",
            available=True,
        )
    )
    assert provider.count() == 2


def test_mock_evidence_provider_clear():
    """Test clearing all evidence."""
    provider = MockEvidenceProvider()

    provider.add_evidence(
        EvidenceResult(
            factor_id="F001",
            evaluation_run_id="RUN_123",
            evidence_id="EVD_001",
            timestamp="2024-01-01T00:00:00Z",
            available=True,
        )
    )

    assert provider.count() == 1

    provider.clear()

    assert provider.count() == 0
    assert not provider.has_evidence("F001")


def test_mock_evidence_provider_multiple_factors():
    """Test provider with multiple factors."""
    provider = MockEvidenceProvider()

    for i in range(1, 4):
        provider.add_evidence(
            EvidenceResult(
                factor_id=f"F{i:03d}",
                evaluation_run_id=f"RUN_{i}",
                evidence_id=f"EVD_{i}",
                timestamp="2024-01-01T00:00:00Z",
                available=True,
            )
        )

    assert provider.count() == 3
    assert provider.has_evidence("F001")
    assert provider.has_evidence("F002")
    assert provider.has_evidence("F003")


def test_evidence_result_secondary_metrics_default():
    """Test secondary_metrics defaults to empty dict."""
    result = EvidenceResult(
        factor_id="F001",
        evaluation_run_id="RUN_123",
        evidence_id="EVD_001",
        timestamp="2024-01-01T00:00:00Z",
        available=True,
    )

    assert result.secondary_metrics == {}
