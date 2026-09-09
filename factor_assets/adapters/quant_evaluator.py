"""
QuantEvaluator adapter for evidence integration.

Converts QE EvaluationBundle to FA EvidenceRef/EvidenceBundleRef.
Provides evidence references without duplicating metric truth.

This is an OPTIONAL adapter — FA core does not depend on QE.
"""

from typing import Protocol, Optional, Dict, Tuple, Mapping
from datetime import datetime
from types import MappingProxyType
import hashlib
import json

import numpy as np

try:
    try:  # compatibility seam used by the adapter's isolated mock tests
        from quant_evaluator import EvaluationBundle as QEEvaluationBundle
        from quant_evaluator import MetricValue as QEMetricValue
        from quant_evaluator import FactorDiagnosis as QEFactorDiagnosis
    except ImportError:
        from quant_evaluator.api.requests import (
            EvaluationBundle as QEEvaluationBundle,
            MetricValue as QEMetricValue,
            FactorDiagnosis as QEFactorDiagnosis,
        )
    from quant_evaluator.contracts.factor_batch import FactorBatch as QEFactorBatch
    QE_AVAILABLE = True
except ImportError:
    QE_AVAILABLE = False
    QEEvaluationBundle = None
    QEMetricValue = None
    QEFactorDiagnosis = None
    QEFactorBatch = None

try:
    from quant_evaluator.runtime.evaluator import authoritative_array_hash
except ImportError:
    authoritative_array_hash = None

try:
    from quant_evaluator.metrics.interactions.pairwise import (
        PairwiseCorrelationArtifact as QEPairwiseCorrelationArtifact,
        PairwiseMeasurementStatus as QEPairwiseMeasurementStatus,
    )
except ImportError:
    QEPairwiseCorrelationArtifact = None
    QEPairwiseMeasurementStatus = None

from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.adapters import OptionalDependencyMissing
from factor_assets.clustering.incremental import (
    CertifiedPairwiseEvidence,
    CertifiedWindowEvidence,
    PairwiseEvidenceStatus,
)
from factor_assets.profiling.metric_grading import MetricGradeArtifact, grade_metric_evidence


def validate_policy_runtime_capability(policy, *, use_case: str) -> Mapping[str, Mapping[str, str]]:
    """Compile required FA policy metrics against QE's sealed runtime authority.

    This is a read-only adapter over QE's one registry; it never defines or
    caches a second metric catalog.
    """
    if not QE_AVAILABLE:
        raise OptionalDependencyMissing("quant_evaluator", "policy capability validation")
    from quant_evaluator.registry.metrics import (
        catalog_snapshot, registry_state, resolve_alias,
    )
    if registry_state() != "sealed":
        raise RuntimeError("QE metric registry must be sealed before policy compilation")
    catalog = catalog_snapshot()
    floors = policy.admission_for(use_case)
    compiled: dict[str, Mapping[str, str]] = {}
    failures: list[str] = []
    for dimension_id in floors.required_dimension_ids:
        rule = policy.dimension_rule(dimension_id)
        for policy_metric_id in rule.required_metric_ids:
            try:
                grade_rule = policy.metric_rule(policy_metric_id)
            except KeyError:
                failures.append(f"{dimension_id}:{policy_metric_id}:NO_GRADE_RULE")
                continue
            requested = grade_rule.runtime_metric_id or policy.canonical_metric_id(policy_metric_id)
            runtime_id = resolve_alias(requested)
            spec = catalog.get(runtime_id)
            if spec is None:
                failures.append(f"{dimension_id}:{policy_metric_id}:{runtime_id}:NOT_REGISTERED")
                continue
            if spec.compute_fn is None:
                failures.append(f"{dimension_id}:{policy_metric_id}:{runtime_id}:NO_RUNTIME_IMPLEMENTATION")
                continue
            compiled[policy_metric_id] = {
                "runtime_metric_id": runtime_id,
                "metric_version": spec.metric_version,
                "status": spec.status.value,
            }
    if failures:
        raise ValueError("required policy metric capability unavailable: " + ";".join(failures))
    return MappingProxyType(compiled)


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
        if not QE_AVAILABLE and QEPairwiseCorrelationArtifact is None:
            raise OptionalDependencyMissing("quant_evaluator", "QEEvidenceProvider")

    def create_pairwise_evidence(
        self,
        artifact: object,
        *,
        expected_factor_ids: Tuple[str, str],
        expected_window_ref: str,
        expected_universe_ref: str,
    ) -> CertifiedPairwiseEvidence:
        """Adapt QE's typed result without recomputing or coercing its value."""
        if QEPairwiseCorrelationArtifact is None or not isinstance(
            artifact, QEPairwiseCorrelationArtifact
        ):
            raise TypeError("artifact must be a QE PairwiseCorrelationArtifact")
        if set(expected_factor_ids) != {artifact.factor_id_a, artifact.factor_id_b}:
            raise ValueError("QE pairwise factor IDs do not match the requested pair")
        if artifact.window_ref != expected_window_ref:
            raise ValueError("QE pairwise window_ref is not comparable to the caller window")
        if artifact.universe_ref != expected_universe_ref:
            raise ValueError("QE pairwise universe_ref is not comparable to the caller universe")
        computed = artifact.status is QEPairwiseMeasurementStatus.COMPUTED
        windows = tuple(
            CertifiedWindowEvidence(
                window_ref=window.window_ref,
                signed_similarity=(float(window.correlation)
                                   if window.status is QEPairwiseMeasurementStatus.COMPUTED else None),
                status=(PairwiseEvidenceStatus.CERTIFIED
                        if window.status is QEPairwiseMeasurementStatus.COMPUTED
                        else PairwiseEvidenceStatus.UNMEASURED),
                pair_count=window.pair_count,
                n_days=window.n_days,
                confidence_interval=window.confidence_interval,
                uncertainty_scale=window.uncertainty_scale,
            ) for window in artifact.windows
        )
        return CertifiedPairwiseEvidence(
            factor_id_a=artifact.factor_id_a,
            factor_id_b=artifact.factor_id_b,
            signed_similarity=(float(artifact.correlation) if computed else None),
            status=(PairwiseEvidenceStatus.CERTIFIED if computed
                    else PairwiseEvidenceStatus.UNMEASURED),
            pair_count=artifact.pair_count,
            window_ref=artifact.window_ref,
            universe_ref=artifact.universe_ref,
            sample_ref=artifact.sample_ref,
            windows=windows,
        )

    def generalization_diagnosis_evidence(
        self,
        bundle: "QEEvaluationBundle",
        factor_id: str,
    ) -> Mapping[str, object]:
        """Bridge typed QE generalization metrics to FA diagnosis keys.

        This is a projection only: it neither recomputes values nor assigns a
        grade.  Missing typed artifacts or identity provenance remain UNKNOWN.
        """
        if not isinstance(bundle, QEEvaluationBundle):
            raise TypeError(f"Expected QEEvaluationBundle, got {type(bundle).__name__}")
        if factor_id not in bundle.factor_ids:
            raise ValueError("factor_id is absent from the QE bundle")
        index = tuple(bundle.factor_ids).index(factor_id)
        grouped = bundle.grouped_metrics or {}
        factor_metrics = grouped.get(factor_id, {})
        if len(bundle.factor_ids) == 1:
            factor_metrics = bundle.metric_values
        output: Dict[str, object] = {}
        for metric_id in ("validation_retention", "train_validation_icir_delta"):
            key = f"generalization.{metric_id}"
            metric = factor_metrics.get(metric_id)
            artifact = (bundle.artifacts or {}).get(metric_id)
            provenance = getattr(artifact, "provenance", {}) if artifact is not None else {}
            axis = getattr(getattr(artifact, "factor_axis", None), "factor_ids", ())
            identities_complete = (
                tuple(axis) == tuple(bundle.factor_ids)
                and isinstance(provenance, Mapping)
                and bool(provenance.get("metric_instance"))
                and bool(provenance.get("train_split_ref"))
                and bool(provenance.get("validation_split_ref"))
                and len(tuple(provenance.get("factor_versions", ()))) == len(bundle.factor_ids)
            )
            value = None
            computed = bool(metric is not None and metric.valid
                            and metric.value is not None and identities_complete)
            if computed:
                typed_value = float(np.asarray(artifact.values, dtype=float)[index])
                if not np.isfinite(typed_value) or typed_value != float(metric.value):
                    raise ValueError("QE grouped metric disagrees with typed generalization artifact")
                value = typed_value
            output[f"{key}.evidence_status"] = "COMPUTED" if computed else "UNKNOWN"
            output[f"{key}.value"] = value
            output[f"{key}.evaluation_ref"] = bundle.request_id
            output[f"{key}.metric_version"] = (
                metric.metric_version if metric is not None else bundle.metric_versions.get(metric_id)
            )
        return MappingProxyType(output)

    def grade_typed_metrics(
        self,
        bundle: "QEEvaluationBundle",
        factor_batch: "QEFactorBatch",
        *,
        expected_config_hash: str | None = None,
        expected_evaluation_ref: str | None = None,
        policy=None,
        cost_budget_bps: float | None = None,
    ) -> Tuple[object, ...]:
        """Grade scalar QE metrics while preserving their production identity.

        Values and validity come only from QE's typed ``EvaluationBundle``.
        Input provenance is checked against QE's runtime provenance and the
        actual ``FactorBatch``.  Missing identity stays explicitly UNKNOWN and
        ungraded; contradictory identity fails closed.
        """
        if not isinstance(bundle, QEEvaluationBundle):
            raise TypeError(f"Expected QEEvaluationBundle, got {type(bundle).__name__}")
        if QEFactorBatch is None or not isinstance(factor_batch, QEFactorBatch):
            raise TypeError("factor_batch must be a QE FactorBatch")
        if len(bundle.factor_ids) != 1 or len(factor_batch.factor_ids) != 1:
            raise ValueError("typed metric grading currently requires exactly one factor")
        if tuple(bundle.factor_ids) != tuple(factor_batch.factor_ids):
            raise ValueError("QE bundle factor identity does not match FactorBatch")
        if expected_config_hash is not None and bundle.config_hash != expected_config_hash:
            raise ValueError("QE bundle config_hash does not match expected configuration")
        if expected_evaluation_ref is not None and bundle.request_id != expected_evaluation_ref:
            raise ValueError("QE bundle request_id does not match expected evaluation_ref")
        if cost_budget_bps is not None:
            if (isinstance(cost_budget_bps, bool) or
                    not isinstance(cost_budget_bps, (int, float)) or
                    not np.isfinite(cost_budget_bps) or cost_budget_bps <= 0):
                raise ValueError("cost_budget_bps must be a finite positive number")

        if policy is None:
            from factor_assets.profiling.policies import get_health_policy
            policy = get_health_policy()

        metadata = bundle.metadata if isinstance(bundle.metadata, Mapping) else {}
        runtime = metadata.get("runtime", {})
        provenance = metadata.get("provenance", {})
        if not isinstance(provenance, Mapping) and isinstance(runtime, Mapping):
            provenance = runtime.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        authoritative_complete = all(
            key in provenance for key in (
                "factor_ids", "time_coordinates", "asset_coordinates",
                "context_refs", "factor_value_bytes_hash", "factor_validity_hash",
            )
        )
        if provenance:
            if tuple(provenance.get("factor_ids", ())) != tuple(factor_batch.factor_ids):
                raise ValueError("QE runtime factor identity does not match FactorBatch")
            expected_axes = (
                ("time_coordinates", factor_batch.time_axis.values),
                ("asset_coordinates", factor_batch.asset_axis.values),
            )
            for name, values in expected_axes:
                recorded = provenance.get(name)
                if values is None:
                    authoritative_complete = False
                    continue
                actual_hash = authoritative_array_hash(values)
                if not isinstance(recorded, Mapping) or recorded.get("sha256") != actual_hash:
                    raise ValueError(f"QE runtime {name} does not match FactorBatch")
            if ("factor_value_bytes_hash" in provenance and
                    provenance.get("factor_value_bytes_hash") != authoritative_array_hash(factor_batch.values)):
                raise ValueError("QE runtime factor values do not match FactorBatch")
            actual_validity_hash = (
                authoritative_array_hash(factor_batch.validity)
                if factor_batch.validity is not None else None
            )
            if ("factor_validity_hash" in provenance and
                    provenance.get("factor_validity_hash") != actual_validity_hash):
                raise ValueError("QE runtime factor validity does not match FactorBatch")

        factor_id = factor_batch.factor_ids[0]
        context_refs = dict(factor_batch.context_refs)
        recorded_context = provenance.get("context_refs", {}) if provenance else {}
        if "context_refs" in provenance and (
                not isinstance(recorded_context, Mapping) or dict(recorded_context) != context_refs):
            raise ValueError("QE runtime context refs do not match FactorBatch")
        definitions = context_refs.get("factor_definition_refs", {})
        factor_definition_id = definitions.get(factor_id, "") if isinstance(definitions, Mapping) else ""
        factor_value_ref = context_refs.get("factor_value_ref", "")
        config_hash = bundle.config_hash or ""
        evaluation_ref = bundle.request_id or ""
        axis_payload = {
            "factor_ids": list(factor_batch.factor_ids),
            "time": provenance.get("time_coordinates") if provenance else None,
            "asset": provenance.get("asset_coordinates") if provenance else None,
        }
        factor_axis_ref = ""
        if (authoritative_complete and factor_batch.time_axis.values is not None and
                factor_batch.asset_axis.values is not None and
                isinstance(axis_payload["time"], Mapping) and
                isinstance(axis_payload["asset"], Mapping)):
            factor_axis_ref = "factor-axis:" + hashlib.sha256(
                json.dumps(axis_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        identity_complete = authoritative_complete and all((
            factor_definition_id, factor_value_ref, factor_axis_ref,
            config_hash, evaluation_ref,
        ))

        grades = []
        for metric_id in sorted(bundle.metric_values):
            metric = bundle.metric_values[metric_id]
            computed = metric.valid and metric.value is not None and identity_complete
            bindings = ((metric_id, metric.value),)
            if metric_id == "turnover_cost":
                # QE reports realized portfolio cost in bps.  FA's cost_drag
                # policy is a return fraction, while budget utilization is a
                # distinct ratio requiring an explicit account/policy budget.
                fraction = None if metric.value is None else float(metric.value) / 10_000.0
                bindings = (("cost_drag", fraction),)
                if cost_budget_bps is not None:
                    utilization = (
                        None if metric.value is None
                        else float(metric.value) / float(cost_budget_bps)
                    )
                    bindings += (("cost_budget_utilization", utilization),)
            for policy_metric_id, policy_value in bindings:
                try:
                    policy.metric_rule(policy_metric_id)
                except KeyError:
                    grades.append(MetricGradeArtifact(
                        metric_id=policy_metric_id,
                        metric_version=metric.metric_version,
                        value=policy_value,
                        evidence_status="COMPUTED" if computed else (
                            "NOT_COMPUTED_STAGE" if identity_complete else "UNKNOWN"
                        ),
                        grade=None,
                        desirability=None,
                        grading_policy_id=policy.policy_id,
                        grading_policy_version=policy.policy_version,
                        evaluation_ref=evaluation_ref,
                        factor_definition_id=factor_definition_id,
                        factor_value_ref=factor_value_ref,
                        factor_axis_ref=factor_axis_ref,
                        config_hash=config_hash,
                        created_from_refs=(evaluation_ref, factor_value_ref, factor_axis_ref, config_hash)
                        if identity_complete else (),
                        applicability="NOT_APPLICABLE",
                        applicability_reason="NO_GRADE_RULE_FOR_METRIC_VERSION",
                        reason_codes=("NO_GRADE_RULE_FOR_METRIC_VERSION",),
                    ))
                    continue
                grades.append(grade_metric_evidence(
                    metric_id=policy_metric_id,
                    value=policy_value,
                    evidence_status="COMPUTED" if computed else (
                        "NOT_COMPUTED_STAGE" if identity_complete else "UNKNOWN"
                    ),
                    policy=policy,
                    evaluation_ref=evaluation_ref,
                    metric_version=metric.metric_version,
                    factor_definition_id=factor_definition_id,
                    factor_value_ref=factor_value_ref,
                    factor_axis_ref=factor_axis_ref,
                    config_hash=config_hash,
                    created_from_refs=(evaluation_ref, factor_value_ref, factor_axis_ref, config_hash)
                    if identity_complete else (),
                ))
        return tuple(grades)

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
