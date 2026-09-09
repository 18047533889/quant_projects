"""
Tests for QuantEvaluator adapter.

Tests evidence provider integration with mock QE bundles.
"""

import pytest
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
from types import SimpleNamespace

from factor_assets.adapters import OptionalDependencyMissing


# Mock QE types for testing
@dataclass(frozen=True)
class MockMetricValue:
    """Mock QE MetricValue."""
    metric_id: str
    value: Optional[float]
    valid: bool
    observation_count: int
    metric_version: str = "0.1"
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MockFactorDiagnosis:
    """Mock QE FactorDiagnosis."""
    factor_id: str
    num_valid_observations: int
    num_missing: int
    coverage: float
    is_constant: bool
    has_nans: bool
    has_infs: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MockEvaluationBundle:
    """Mock QE EvaluationBundle."""
    request_id: str
    factor_ids: Tuple[str, ...]
    label_id: str
    timestamp: str
    schema_version: str = "0.1"
    metric_values: Dict[str, MockMetricValue] = field(default_factory=dict)
    diagnostics: Dict[str, MockFactorDiagnosis] = field(default_factory=dict)
    grouped_metrics: Optional[Dict[str, Dict[str, MockMetricValue]]] = None
    series_refs: Optional[Dict[str, str]] = None
    metric_versions: Dict[str, str] = field(default_factory=dict)
    config_hash: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


class TestQEEvidenceProvider:
    """Test QE evidence provider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock QE availability by patching the module
        import sys
        from unittest.mock import MagicMock

        # Create mock QE module
        mock_qe = MagicMock()
        mock_qe.EvaluationBundle = MockEvaluationBundle
        mock_qe.MetricValue = MockMetricValue
        mock_qe.FactorDiagnosis = MockFactorDiagnosis
        # Snapshot the REAL quant_evaluator module so it can be restored in
        # teardown_method — this test runs in-process alongside other domains'
        # suites (e.g. DLIB-XPKG cross-package runs), and leaking the MagicMock
        # into sys.modules would break every subsequent quant_evaluator import
        # (pytest would choke on the mock's ``pytest_plugins`` attribute).
        import quant_evaluator as _real_qe
        self._real_qe_module = _real_qe
        sys.modules['quant_evaluator'] = mock_qe

        # Force reload of adapter to pick up mock
        import importlib
        import factor_assets.adapters.quant_evaluator
        importlib.reload(factor_assets.adapters.quant_evaluator)

        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
        self.provider = QEEvidenceProvider()

    def teardown_method(self):
        """Restore the real quant_evaluator module after the mock test."""
        import sys
        import importlib

        if getattr(self, "_real_qe_module", None) is not None:
            sys.modules["quant_evaluator"] = self._real_qe_module
        # Reload any QE-importing FA adapters so they rebind the real module.
        import factor_assets.adapters.quant_evaluator
        importlib.reload(factor_assets.adapters.quant_evaluator)

    def test_create_evidence_bundle_ref_single_factor(self):
        """Test creating evidence bundle ref for single factor."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        # Create mock bundle
        bundle = MockEvaluationBundle(
            request_id="req-001",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            schema_version="0.1",
            metric_values={
                "pearson_ic": MockMetricValue(
                    metric_id="pearson_ic",
                    value=0.05,
                    valid=True,
                    observation_count=1000,
                ),
                "rank_ic": MockMetricValue(
                    metric_id="rank_ic",
                    value=0.03,
                    valid=True,
                    observation_count=1000,
                ),
            },
            config_hash="config-abc123",
            warnings=(),
            metadata={
                "universe_ref": "universe-001",
                "period_start": "2020-01-01",
                "period_end": "2025-12-31",
            },
        )

        # Create bundle ref
        bundle_ref = provider.create_evidence_bundle_ref(
            bundle=bundle,
            run_id="run-001",
            qe_version="0.1.0",
        )

        # Verify bundle ref
        assert bundle_ref.bundle_id == "req-001"
        assert bundle_ref.evaluation_run_id == "run-001"
        assert bundle_ref.factor_ids == ("F1234567890abcdef",)
        assert bundle_ref.timestamp == "2026-08-14T00:00:00Z"
        assert bundle_ref.qe_version == "0.1.0"
        assert bundle_ref.universe_ref == "universe-001"
        assert bundle_ref.period_start == "2020-01-01"
        assert bundle_ref.period_end == "2025-12-31"
        assert bundle_ref.label_ref == "label-001"
        assert bundle_ref.config_hash == "config-abc123"
        assert bundle_ref.primary_metric == "pearson_ic"
        assert bundle_ref.primary_value == 0.05
        assert len(bundle_ref.warnings) == 0

    def test_v6_typed_generalization_bridge_uses_namespaced_diagnosis_keys(self):
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        factor_id = "F1234567890abcdef"
        metrics = {
            name: MockMetricValue(name, value, True, 12, "2.0.0")
            for name, value in (
                ("validation_retention", .4),
                ("train_validation_icir_delta", 1.5),
            )
        }
        provenance = {
            "metric_instance": "rank_ic:h1", "train_split_ref": "train:s1",
            "validation_split_ref": "validation:s1", "factor_versions": ("v1",),
        }
        artifacts = {
            name: SimpleNamespace(
                values=[metric.value], factor_axis=SimpleNamespace(factor_ids=(factor_id,)),
                provenance=provenance,
            ) for name, metric in metrics.items()
        }
        bundle = MockEvaluationBundle(
            request_id="qe:g1", factor_ids=(factor_id,), label_id="h1",
            timestamp="2026-09-08T00:00:00Z", metric_values=metrics,
            metric_versions={name: "2.0.0" for name in metrics}, artifacts=artifacts,
        )
        evidence = QEEvidenceProvider().generalization_diagnosis_evidence(bundle, factor_id)
        assert evidence["generalization.validation_retention.evidence_status"] == "COMPUTED"
        assert evidence["generalization.validation_retention.value"] == .4
        assert evidence["generalization.train_validation_icir_delta.value"] == 1.5

    def test_create_evidence_bundle_ref_with_warnings(self):
        """Test bundle ref creation with warnings."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-002",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            metric_values={},
            warnings=("low_coverage", "high_missing_rate"),
            metadata={},
        )

        bundle_ref = provider.create_evidence_bundle_ref(
            bundle=bundle,
            run_id="run-002",
        )

        assert bundle_ref.has_warnings
        assert len(bundle_ref.warnings) == 2
        assert "low_coverage" in bundle_ref.warnings

    def test_create_evidence_refs_single_factor(self):
        """Test creating individual evidence refs for single factor."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-003",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            metric_values={
                "pearson_ic": MockMetricValue(
                    metric_id="pearson_ic",
                    value=0.05,
                    valid=True,
                    observation_count=1000,
                    metric_version="1.0",
                ),
                "rank_ic": MockMetricValue(
                    metric_id="rank_ic",
                    value=0.03,
                    valid=True,
                    observation_count=1000,
                    metric_version="1.0",
                ),
            },
            metadata={
                "universe_ref": "universe-001",
                "period_start": "2020-01-01",
                "period_end": "2025-12-31",
            },
        )

        # Create evidence refs
        refs = provider.create_evidence_refs(
            bundle=bundle,
            run_id="run-003",
        )

        # Verify refs
        assert len(refs) == 2

        pearson_ref = refs[0]
        assert pearson_ref.evidence_id == "req-003:F1234567890abcdef:pearson_ic"
        assert pearson_ref.evaluation_run_id == "run-003"
        assert pearson_ref.metric_name == "pearson_ic"
        assert pearson_ref.metric_version == "1.0"
        assert pearson_ref.factor_id == "F1234567890abcdef"
        assert pearson_ref.summary_value == 0.05
        assert pearson_ref.universe_ref == "universe-001"

    def test_create_evidence_refs_multi_factor_grouped(self):
        """Test evidence refs for multi-factor bundle with grouped metrics."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-004",
            factor_ids=("F1111111111111111", "F2222222222222222"),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            metric_values={},  # Empty for grouped case
            grouped_metrics={
                "F1111111111111111": {
                    "pearson_ic": MockMetricValue(
                        metric_id="pearson_ic",
                        value=0.05,
                        valid=True,
                        observation_count=1000,
                        metric_version="1.0",
                    ),
                },
                "F2222222222222222": {
                    "pearson_ic": MockMetricValue(
                        metric_id="pearson_ic",
                        value=0.03,
                        valid=True,
                        observation_count=1000,
                        metric_version="1.0",
                    ),
                },
            },
            metadata={},
        )

        refs = provider.create_evidence_refs(
            bundle=bundle,
            run_id="run-004",
        )

        assert len(refs) == 2
        assert refs[0].factor_id == "F1111111111111111"
        assert refs[0].summary_value == 0.05
        assert refs[1].factor_id == "F2222222222222222"
        assert refs[1].summary_value == 0.03

    def test_create_evidence_ref_invalid_metric(self):
        """Test evidence ref with invalid metric."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-005",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            metric_values={
                "pearson_ic": MockMetricValue(
                    metric_id="pearson_ic",
                    value=None,
                    valid=False,
                    observation_count=100,
                    warnings=("insufficient_observations",),
                ),
            },
            metadata={},
        )

        refs = provider.create_evidence_refs(
            bundle=bundle,
            run_id="run-005",
        )

        assert len(refs) == 1
        assert refs[0].summary_value is None  # Invalid metric has no value
        assert refs[0].summary_context == "warnings=1"

    def test_extract_diagnosis_summary(self):
        """Test extracting diagnosis summary."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-006",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            diagnostics={
                "F1234567890abcdef": MockFactorDiagnosis(
                    factor_id="F1234567890abcdef",
                    num_valid_observations=950,
                    num_missing=50,
                    coverage=0.95,
                    is_constant=False,
                    has_nans=True,
                    has_infs=False,
                    warnings=("high_nan_rate",),
                ),
            },
        )

        summary = provider.extract_diagnosis_summary(
            bundle=bundle,
            factor_id="F1234567890abcdef",
        )

        assert summary is not None
        assert summary["num_valid_observations"] == 950
        assert summary["coverage"] == 0.95
        assert summary["is_constant"] is False
        assert summary["has_nans"] is True
        assert summary["has_infs"] is False
        assert len(summary["warnings"]) == 1
        # Verify we don't leak distribution values
        assert "min_value" not in summary
        assert "max_value" not in summary
        assert "mean_value" not in summary

    def test_extract_diagnosis_summary_missing(self):
        """Test extracting diagnosis for missing factor."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        bundle = MockEvaluationBundle(
            request_id="req-007",
            factor_ids=("F1234567890abcdef",),
            label_id="label-001",
            timestamp="2026-08-14T00:00:00Z",
            diagnostics={},
        )

        summary = provider.extract_diagnosis_summary(
            bundle=bundle,
            factor_id="F1234567890abcdef",
        )

        assert summary is None

    def test_invalid_bundle_type(self):
        """Test error handling for invalid bundle type."""
        from factor_assets.adapters.quant_evaluator import QEEvidenceProvider

        provider = QEEvidenceProvider()

        with pytest.raises(TypeError, match="Expected QEEvaluationBundle"):
            provider.create_evidence_bundle_ref(
                bundle="not-a-bundle",
                run_id="run-001",
            )

        with pytest.raises(TypeError, match="Expected QEEvaluationBundle"):
            provider.create_evidence_refs(
                bundle={"invalid": "dict"},
                run_id="run-001",
            )
