"""
QuantEvaluator adapter for evidence integration.

Converts QE EvaluationBundle to FA EvidenceRef/EvidenceBundleRef.
Provides evidence references without duplicating metric truth.

This is an OPTIONAL adapter — FA core does not depend on QE.
"""

from typing import Protocol, Optional, Dict, Tuple
from datetime import datetime

try:
    from quant_evaluator import EvaluationBundle as QEEvaluationBundle
    from quant_evaluator import MetricValue as QEMetricValue
    from quant_evaluator import FactorDiagnosis as QEFactorDiagnosis
    QE_AVAILABLE = True
except ImportError:
    QE_AVAILABLE = False
    QEEvaluationBundle = None
    QEMetricValue = None
    QEFactorDiagnosis = None

from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.adapters import OptionalDependencyMissing


class EvidenceProvider(Protocol):
    """
    Protocol for obtaining evidence references from evaluation results.

    FA defines this protocol; adapters implement it for specific evaluators.
    """

    def create_evidence_bundle_ref(
        self,
        bundle: object,
        run_id: str,
    ) -> EvidenceBundleRef:
        """
        Create FA evidence bundle reference from evaluation result.

        Args:
            bundle: Evaluation bundle from evaluator
            run_id: Evaluation run identifier

        Returns:
            FA EvidenceBundleRef
        """
        ...

    def create_evidence_refs(
        self,
        bundle: object,
        run_id: str,
    ) -> Tuple[EvidenceRef, ...]:
        """
        Extract individual evidence refs for each metric.

        Args:
            bundle: Evaluation bundle from evaluator
            run_id: Evaluation run identifier

        Returns:
            Tuple of EvidenceRef for each metric
        """
        ...


class QEEvidenceProvider:
    """
    EvidenceProvider implementation for QuantEvaluator.

    Converts QE EvaluationBundle to FA evidence references.
    Does not duplicate metric values — only creates references.
    """

    def __init__(self):
        if not QE_AVAILABLE:
            raise OptionalDependencyMissing("quant_evaluator", "QEEvidenceProvider")

    def create_evidence_bundle_ref(
        self,
        bundle: "QEEvaluationBundle",
        run_id: str,
        qe_version: Optional[str] = None,
    ) -> EvidenceBundleRef:
        """
        Create FA evidence bundle reference from QE EvaluationBundle.

        Args:
            bundle: QE EvaluationBundle
            run_id: Evaluation run identifier
            qe_version: QE version (optional, taken from bundle if not provided)

        Returns:
            FA EvidenceBundleRef

        Raises:
            OptionalDependencyMissing: If QE is not available
            ValueError: If bundle is invalid
        """
        if not isinstance(bundle, QEEvaluationBundle):
            raise TypeError(f"Expected QEEvaluationBundle, got {type(bundle).__name__}")

        # Extract primary metric (first valid metric with value)
        primary_metric = None
        primary_value = None
        for metric_id, metric_val in bundle.metric_values.items():
            if metric_val.valid and metric_val.value is not None:
                primary_metric = metric_id
                primary_value = metric_val.value
                break

        return EvidenceBundleRef(
            bundle_id=bundle.request_id,  # Use request_id as bundle identifier
            evaluation_run_id=run_id,
            factor_ids=bundle.factor_ids,
            timestamp=bundle.timestamp,
            qe_version=qe_version or bundle.schema_version,
            universe_ref=bundle.metadata.get("universe_ref"),
            period_start=bundle.metadata.get("period_start"),
            period_end=bundle.metadata.get("period_end"),
            label_ref=bundle.label_id,
            config_hash=bundle.config_hash,
            primary_metric=primary_metric,
            primary_value=primary_value,
            warnings=bundle.warnings,
        )

    def create_evidence_refs(
        self,
        bundle: "QEEvaluationBundle",
        run_id: str,
    ) -> Tuple[EvidenceRef, ...]:
        """
        Extract individual evidence refs for each metric in the bundle.

        Args:
            bundle: QE EvaluationBundle
            run_id: Evaluation run identifier

        Returns:
            Tuple of EvidenceRef for each metric

        Raises:
            OptionalDependencyMissing: If QE is not available
            ValueError: If bundle is invalid
        """
        if not isinstance(bundle, QEEvaluationBundle):
            raise TypeError(f"Expected QEEvaluationBundle, got {type(bundle).__name__}")

        refs = []

        # Single-factor bundle: one evidence ref per metric
        if len(bundle.factor_ids) == 1:
            factor_id = bundle.factor_ids[0]
            for metric_id, metric_val in bundle.metric_values.items():
                refs.append(
                    self._create_evidence_ref(
                        bundle=bundle,
                        run_id=run_id,
                        factor_id=factor_id,
                        metric_id=metric_id,
                        metric_val=metric_val,
                    )
                )

        # Multi-factor bundle: extract grouped metrics if available
        elif bundle.grouped_metrics:
            for factor_id, metrics in bundle.grouped_metrics.items():
                for metric_id, metric_val in metrics.items():
                    refs.append(
                        self._create_evidence_ref(
                            bundle=bundle,
                            run_id=run_id,
                            factor_id=factor_id,
                            metric_id=metric_id,
                            metric_val=metric_val,
                        )
                    )

        return tuple(refs)

    def _create_evidence_ref(
        self,
        bundle: "QEEvaluationBundle",
        run_id: str,
        factor_id: str,
        metric_id: str,
        metric_val: "QEMetricValue",
    ) -> EvidenceRef:
        """Create a single evidence ref for a factor-metric pair."""
        # Generate evidence ID from bundle + factor + metric
        evidence_id = f"{bundle.request_id}:{factor_id}:{metric_id}"

        # Extract bounded summary (value only if valid, no full distributions)
        summary_value = metric_val.value if metric_val.valid else None
        summary_context = None
        if metric_val.warnings:
            summary_context = f"warnings={len(metric_val.warnings)}"

        return EvidenceRef(
            evidence_id=evidence_id,
            evaluation_run_id=run_id,
            metric_name=metric_id,
            metric_version=metric_val.metric_version,
            timestamp=bundle.timestamp,
            factor_id=factor_id,
            universe_ref=bundle.metadata.get("universe_ref"),
            period_start=bundle.metadata.get("period_start"),
            period_end=bundle.metadata.get("period_end"),
            summary_value=summary_value,
            summary_context=summary_context,
        )

    def extract_diagnosis_summary(
        self,
        bundle: "QEEvaluationBundle",
        factor_id: str,
    ) -> Optional[Dict[str, any]]:
        """
        Extract bounded diagnosis summary for a factor.

        Returns summary statistics only — no raw factor values.

        Args:
            bundle: QE EvaluationBundle
            factor_id: Factor to extract diagnosis for

        Returns:
            Dict with bounded summary or None if not available
        """
        if not isinstance(bundle, QEEvaluationBundle):
            raise TypeError(f"Expected QEEvaluationBundle, got {type(bundle).__name__}")

        diagnosis = bundle.diagnostics.get(factor_id)
        if not diagnosis:
            return None

        return {
            "num_valid_observations": diagnosis.num_valid_observations,
            "coverage": diagnosis.coverage,
            "is_constant": diagnosis.is_constant,
            "has_nans": diagnosis.has_nans,
            "has_infs": diagnosis.has_infs,
            "warnings": diagnosis.warnings,
            # Note: Do NOT include min/max/mean values to avoid leaking distributions
        }


__all__ = [
    "EvidenceProvider",
    "QEEvidenceProvider",
]
