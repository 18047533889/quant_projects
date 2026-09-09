"""
Evaluation request and result bundles.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.sealed_split import SealedSplitRef
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef


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

    Serialization (DLIB-QE-003, Option A): ``to_dict`` / ``from_dict`` are a
    strict round-trip.  The request references the durable factor-value and
    label-bundle artifacts via :class:`FactorValueRef` / :class:`LabelBundleRef`
    (identity + provenance only, never the large raw arrays).  The raw
    ``batch_or_factor_ids`` / ``label_bundle`` payloads are runtime-only and
    deliberately NOT serialized.
    """
    batch_or_factor_ids: Any
    label_bundle: Any
    metric_ids: Tuple[str, ...] = ("pearson_ic", "rank_ic", "coverage")
    slices: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    tier: str = "core"
    # Planner metric-node units (not wall seconds or money). Canonical aliases
    # share one node; hard admission is enforced before backend allocation.
    cost_budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    split_ref: Optional[SealedSplitRef] = None
    factor_value_ref: Optional[FactorValueRef] = None
    label_bundle_ref: Optional[LabelBundleRef] = None
    metric_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    portfolio_returns: Any = None  # runtime-only ProbePortfolioArtifact, never forward labels
    holding_returns: Any = None  # runtime-only HoldingReturnPanel
    portfolio_spec: Any = None
    trade_eligibility: Any = None  # runtime-only TradeEligibilityPanel
    calendar_snapshot: Any = None  # runtime-only authoritative DA CalendarSnapshot
    exposure_panel: Any = None  # runtime-only axis-bound ExposurePanel
    generalization_evidence: Any = None  # runtime-only TrainVsValidationArtifact
    quantile_builder_parameters: Dict[str, int] = field(default_factory=dict)
    metric_instances: Tuple[Any, ...] = ()
    scenario_inputs: Dict[str, Any] = field(default_factory=dict)  # runtime-only typed bindings

    def __post_init__(self):
        from quant_evaluator.contracts.metric_artifacts import FrozenMapping
        object.__setattr__(self, "metric_ids", tuple(self.metric_ids))
        from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario
        instances = tuple(self.metric_instances)
        if any(not isinstance(item, MetricInstance) for item in instances):
            raise TypeError("metric_instances must contain MetricInstance objects")
        if any(not isinstance(item, EvaluationScenario) for item in self.scenario_inputs.values()):
            raise TypeError("scenario_inputs must contain EvaluationScenario objects")
        object.__setattr__(self, "metric_instances", instances)
        from types import MappingProxyType
        object.__setattr__(self, "scenario_inputs", MappingProxyType(dict(self.scenario_inputs)))
        for name in ("metadata", "metric_parameters", "context", "slices", "quantile_builder_parameters"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, FrozenMapping(value))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict (strict round-trip).

        Serializes the reference fields (``factor_value_ref`` /
        ``label_bundle_ref`` / ``split_ref``) and the scalar request fields.
        The raw ``batch_or_factor_ids`` / ``label_bundle`` payloads are
        runtime-only and deliberately omitted (they are large arrays).
        """
        payload = {
            "tier": self.tier,
            "cost_budget": self.cost_budget,
            "metric_ids": tuple(self.metric_ids),
            "context": self.context,
            "slices": self.slices,
            "metadata": dict(self.metadata),
            "metric_parameters": {k: dict(v) for k, v in self.metric_parameters.items()},
            "portfolio_spec": self.portfolio_spec.to_dict() if self.portfolio_spec is not None else None,
            "quantile_builder_parameters": dict(self.quantile_builder_parameters),
            "metric_instances": [item.to_dict() for item in self.metric_instances],
        }
        if self.split_ref is not None:
            payload["split_ref"] = self.split_ref.to_dict()
        if self.factor_value_ref is not None:
            payload["factor_value_ref"] = self.factor_value_ref.to_dict()
        if self.label_bundle_ref is not None:
            payload["label_bundle_ref"] = self.label_bundle_ref.to_dict()
        from quant_evaluator.contracts._ndarray_codec import encode_value
        return encode_value(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationRequest":
        """Rebuild from :meth:`to_dict` output (strict round-trip).

        Restores ``split_ref`` / ``factor_value_ref`` / ``label_bundle_ref``
        when present.  ``batch_or_factor_ids`` / ``label_bundle`` are not
        carried by ``to_dict`` and default to ``None``.
        """
        if not isinstance(payload, dict):
            raise TypeError("EvaluationRequest.from_dict requires a dict")
        from quant_evaluator.contracts._ndarray_codec import decode_value
        payload = decode_value(payload)
        split = payload.get("split_ref")
        fv_ref = payload.get("factor_value_ref")
        lb_ref = payload.get("label_bundle_ref")
        from quant_evaluator.contracts.portfolio_inputs import PortfolioSpec
        from quant_evaluator.contracts.metric_instance import MetricInstance
        return cls(
            batch_or_factor_ids=payload.get("batch_or_factor_ids"),
            label_bundle=payload.get("label_bundle"),
            metric_ids=tuple(payload.get("metric_ids", ())),
            metric_instances=tuple(MetricInstance.from_dict(item) for item in payload.get("metric_instances", ())),
            slices=payload.get("slices"),
            context=payload.get("context"),
            tier=payload.get("tier", "core"),
            cost_budget=payload.get("cost_budget"),
            metadata=dict(payload.get("metadata", {})),
            metric_parameters={k: dict(v) for k, v in payload.get("metric_parameters", {}).items()},
            quantile_builder_parameters=dict(payload.get("quantile_builder_parameters", {})),
            portfolio_spec=PortfolioSpec(**payload["portfolio_spec"]) if payload.get("portfolio_spec") is not None else None,
            split_ref=SealedSplitRef.from_dict(split) if split is not None else None,
            factor_value_ref=(
                FactorValueRef.from_dict(fv_ref) if fv_ref is not None else None
            ),
            label_bundle_ref=(
                LabelBundleRef.from_dict(lb_ref) if lb_ref is not None else None
            ),
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
    sample_unit: str = "unspecified"


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
    API / query aggregation view of an evaluation result (DLIB-QE-002).

    This is the *query/aggregation* view returned to API and reporting
    consumers.  It aggregates per-metric values, diagnostics, grouped metrics,
    series refs, and metadata for a single evaluation.  It does NOT own domain
    identity, storage, read-model, or API responsibilities — those live in
    the canonical durable :class:`EvaluationArtifact`
    (``quant_evaluator.contracts.evaluation_artifact``) and the runtime /
    adapters respectively.  This bundle is a projection for return + reporting.

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
    artifacts: Dict[str, Any] = field(default_factory=dict)
    factor_artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    instance_results: Dict[str, Any] = field(default_factory=dict)
    instance_specs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Lossless transport projection; durable identity remains EvaluationArtifact's."""
        from dataclasses import fields
        from quant_evaluator.contracts._ndarray_codec import encode_value
        scalar_fields = ("request_id", "factor_ids", "label_id", "timestamp", "schema_version",
                         "series_refs", "metric_versions", "config_hash", "warnings", "metadata")
        payload = {name: encode_value(getattr(self, name)) for name in scalar_fields}
        if set(self.instance_specs) != set(self.instance_results):
            raise ValueError("Instance definitions and results must have identical coordinates")
        payload["instance_specs"] = {key: value.to_dict() for key, value in self.instance_specs.items()}
        payload["instance_results"] = {key: value.to_dict() for key, value in self.instance_results.items()}
        def record(value):
            return {f.name: encode_value(getattr(value, f.name)) for f in fields(value)}
        payload["metric_values"] = {k: record(v) for k, v in self.metric_values.items()}
        payload["diagnostics"] = {k: record(v) for k, v in self.diagnostics.items()}
        payload["grouped_metrics"] = None if self.grouped_metrics is None else {
            fid: {k: record(v) for k, v in group.items()} for fid, group in self.grouped_metrics.items()}
        payload["split_ref"] = self.split_ref.to_dict() if self.split_ref is not None else None
        allowed = {"scalar", "series", "vector", "daily_quantile"}
        payload["artifacts"] = {}
        for mid, artifact in self.artifacts.items():
            if artifact.artifact_kind not in allowed:
                raise ValueError(f"Unsupported bundle artifact kind {artifact.artifact_kind}")
            payload["artifacts"][mid] = {"kind": artifact.artifact_kind, "payload": artifact.to_dict()}
        payload["factor_artifacts"] = {}
        for fid, artifacts in self.factor_artifacts.items():
            if fid not in self.factor_ids:
                raise ValueError(f"Unknown factor artifact coordinate {fid}")
            payload["factor_artifacts"][fid] = {}
            for mid, artifact in artifacts.items():
                if artifact.artifact_kind not in allowed:
                    raise ValueError(f"Unsupported factor artifact kind {artifact.artifact_kind}")
                payload["factor_artifacts"][fid][mid] = {
                    "kind": artifact.artifact_kind, "payload": artifact.to_dict()}
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationBundle":
        from quant_evaluator.contracts._ndarray_codec import decode_value
        from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact, SeriesMetricArtifact, VectorMetricArtifact
        from quant_evaluator.contracts.artifact_types import DailyQuantileReturnArtifact
        kinds = {"scalar": ScalarMetricArtifact, "series": SeriesMetricArtifact,
                 "vector": VectorMetricArtifact, "daily_quantile": DailyQuantileReturnArtifact}
        values = {name: decode_value(payload[name]) for name in (
            "request_id", "factor_ids", "label_id", "timestamp", "schema_version",
            "series_refs", "metric_versions", "config_hash", "warnings", "metadata")}
        values["factor_ids"] = tuple(values["factor_ids"])
        values["warnings"] = tuple(values["warnings"])
        from quant_evaluator.contracts.metric_instance import MetricInstance
        specs = {key: MetricInstance.from_dict(value) for key, value in payload.get("instance_specs", {}).items()}
        children = {key: cls.from_dict(value) for key, value in payload.get("instance_results", {}).items()}
        if set(specs) != set(children):
            raise ValueError("Instance definitions and results disagree")
        for key, spec in specs.items():
            child = children[key]
            if key != spec.instance_id or child.factor_ids != values["factor_ids"]:
                raise ValueError("Instance identity or factor coordinates disagree")
            if child.instance_results or set(child.metric_versions) != {spec.metric_id}:
                raise ValueError("Instance result must contain exactly its canonical metric")
            if child.metric_versions[spec.metric_id] != spec.metric_version:
                raise ValueError("Instance result implementation version disagrees")
        values["instance_specs"] = specs
        values["instance_results"] = children
        def record(kind, data):
            decoded = decode_value(data)
            decoded["warnings"] = tuple(decoded.get("warnings", ()))
            return kind(**decoded)
        values["metric_values"] = {k: record(MetricValue, v) for k,v in payload["metric_values"].items()}
        values["diagnostics"] = {k: record(FactorDiagnosis, v) for k,v in payload["diagnostics"].items()}
        grouped = payload.get("grouped_metrics")
        values["grouped_metrics"] = None if grouped is None else {
            fid: {k: record(MetricValue,v) for k,v in group.items()} for fid,group in grouped.items()}
        split = payload.get("split_ref")
        values["split_ref"] = SealedSplitRef.from_dict(split) if split is not None else None
        values["artifacts"] = {}
        for mid, wrapper in payload.get("artifacts", {}).items():
            if wrapper["kind"] not in kinds:
                raise ValueError(f"Unsupported bundle artifact kind {wrapper['kind']}")
            artifact = kinds[wrapper["kind"]].from_dict(wrapper["payload"])
            if artifact.metric_id != mid:
                raise ValueError("Bundle artifact key and metric identity disagree")
            axis = artifact.factor_axis
            ids = tuple(axis) if isinstance(axis, tuple) else tuple(axis.factor_ids)
            if ids != values["factor_ids"]:
                raise ValueError("Bundle artifact factor coordinates disagree")
            values["artifacts"][mid] = artifact
        values["factor_artifacts"] = {}
        for fid, artifacts in payload.get("factor_artifacts", {}).items():
            if fid not in values["factor_ids"]:
                raise ValueError("Bundle factor artifact coordinate disagrees")
            values["factor_artifacts"][fid] = {}
            for mid, wrapper in artifacts.items():
                if wrapper["kind"] not in kinds:
                    raise ValueError(f"Unsupported bundle artifact kind {wrapper['kind']}")
                artifact = kinds[wrapper["kind"]].from_dict(wrapper["payload"])
                if artifact.metric_id != mid:
                    raise ValueError("Bundle factor artifact key and metric identity disagree")
                axis = artifact.factor_axis
                ids = tuple(axis) if isinstance(axis, tuple) else tuple(axis.factor_ids)
                if ids != (fid,):
                    raise ValueError("Factor artifact must bind exactly its outer factor coordinate")
                values["factor_artifacts"][fid][mid] = artifact
        return cls(**values)

    @property
    def scalar_metrics(self):
        return {k: a.values for k, a in self.artifacts.items() if a.artifact_kind == "scalar"}

    @property
    def series_metrics(self):
        return {k: a.values for k, a in self.artifacts.items() if a.artifact_kind == "series"}

    @property
    def vector_metrics(self):
        return {k: a.values for k, a in self.artifacts.items() if a.artifact_kind == "vector"}

    def get_metric(self, metric_id: str, factor_id: Optional[str] = None) -> Optional[MetricValue]:
        """Get a metric value by ID, optionally for a specific factor."""
        if factor_id and self.grouped_metrics and factor_id in self.grouped_metrics:
            return self.grouped_metrics[factor_id].get(metric_id)
        return self.metric_values.get(metric_id)

    def get_diagnosis(self, factor_id: str) -> Optional[FactorDiagnosis]:
        """Get diagnostic info for a factor."""
        return self.diagnostics.get(factor_id)
