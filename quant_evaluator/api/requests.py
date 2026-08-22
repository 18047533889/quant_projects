"""
Evaluation request and result bundles.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.sealed_split import SealedSplitRef


@dataclass(frozen=True)
class EvaluationRequest:
    """
    Typed request for factor evaluation.

    Contains factor batch or IDs, labels, metrics to compute, slices, and context.

    ``split_ref`` (optional, default ``None``) carries the sealed-test split
    this evaluation is bound to.  ``None`` means no sealed-split check
    (backward compatible with every existing caller); when provided, the
    runtime raises :class:`SealedSplitOverlapError` if the split window
    overlaps the factor/label information boundary (R21 Q5 sealed-test gate).
    """
    batch_or_factor_ids: Any
    label_bundle: Any
    metric_ids: Tuple[str, ...] = ("pearson_ic", "rank_ic", "coverage")
    slices: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    tier: str = "core"
    cost_budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    split_ref: Optional[SealedSplitRef] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict (for hashing/audit)."""
        payload = {
            "tier": self.tier,
            "cost_budget": self.cost_budget,
            "metric_ids": tuple(self.metric_ids),
            "context": self.context,
            "slices": self.slices,
            "metadata": dict(self.metadata),
        }
        if self.split_ref is not None:
            payload["split_ref"] = self.split_ref.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationRequest":
        """Rebuild from :meth:`to_dict` output (``split_ref`` restored if present)."""
        if not isinstance(payload, dict):
            raise TypeError("EvaluationRequest.from_dict requires a dict")
        split = payload.get("split_ref")
        return cls(
            batch_or_factor_ids=payload.get("batch_or_factor_ids"),
            label_bundle=payload.get("label_bundle"),
            metric_ids=tuple(payload.get("metric_ids", ())),
            slices=payload.get("slices"),
            context=payload.get("context"),
            tier=payload.get("tier", "core"),
            cost_budget=payload.get("cost_budget"),
            metadata=dict(payload.get("metadata", {})),
            split_ref=SealedSplitRef.from_dict(split) if split is not None else None,
        )


@dataclass(frozen=True)
class MetricValue:
    """A single computed metric with metadata."""
    metric_id: str
    value: Optional[float]
    valid: bool
    observation_count: int
    metric_version: str = "0.1"
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FactorDiagnosis:
    """Diagnostic information for a factor or batch."""
    factor_id: str
    num_valid_observations: int
    num_missing: int
    coverage: float
    is_constant: bool
    has_nans: bool
    has_infs: bool
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationBundle:
    """
    Typed result bundle from evaluation.

    Contains versioned metrics, diagnostics, optional series refs, and all metadata
    needed to validate evidence. Does NOT contain admission decisions.

    ``split_ref`` mirrors the request's sealed-test split reference (default
    ``None``) so the bundle traces the sealed evaluation back to the split it
    was bound to.
    """
    request_id: str
    factor_ids: Tuple[str, ...]
    label_id: str
    timestamp: str
    schema_version: str = "0.1"
    metric_values: Dict[str, MetricValue] = field(default_factory=dict)
    diagnostics: Dict[str, FactorDiagnosis] = field(default_factory=dict)
    grouped_metrics: Optional[Dict[str, Dict[str, MetricValue]]] = None
    series_refs: Optional[Dict[str, str]] = None
    metric_versions: Dict[str, str] = field(default_factory=dict)
    config_hash: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    split_ref: Optional[SealedSplitRef] = None

    def get_metric(self, metric_id: str, factor_id: Optional[str] = None) -> Optional[MetricValue]:
        """Get a metric value by ID, optionally for a specific factor."""
        if factor_id and self.grouped_metrics and factor_id in self.grouped_metrics:
            return self.grouped_metrics[factor_id].get(metric_id)
        return self.metric_values.get(metric_id)

    def get_diagnosis(self, factor_id: str) -> Optional[FactorDiagnosis]:
        """Get diagnostic info for a factor."""
        return self.diagnostics.get(factor_id)
