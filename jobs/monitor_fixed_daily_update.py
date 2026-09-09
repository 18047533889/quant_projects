"""Maturity-gated monitoring for an already-published E2E-H segment."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from modeling.monitoring import (
    CampaignRequestPolicy, DRIFT_DIMENSIONS, DriftContract, LifecycleAction,
    MonitorAction, RealisedMetric,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.evaluator import evaluate
from quant_platform.app.contracts import ArtifactRef


@dataclass(frozen=True)
class FixedDailyMonitoringResult:
    status: str
    lifecycle_action: LifecycleAction | None
    factor_value_ref: str
    label_bundle_ref: str
    mature_time_count: int
    mature_cell_count: int
    rank_ic: float | None = None
    coverage: float | None = None
    actions: Mapping[str, MonitorAction] | None = None
    monitor_policy_hash: str = ""
    generation_ids: tuple[str, ...] = ()
    evaluated_window_hash: str = ""
    valid_ic_date_count: int = 0


@dataclass(frozen=True)
class FixedMonitorPolicy:
    policy_id: str
    min_mature_dates: int
    min_assets: int
    reference_ic: float
    reference_coverage: float
    external_drift_evidence: Mapping[str, float]
    drift_contract: DriftContract
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("monitor policy_id is required")
        if (isinstance(self.min_mature_dates, bool)
                or not isinstance(self.min_mature_dates, int)
                or self.min_mature_dates < 2):
            raise ValueError("min_mature_dates must be an integer >= 2")
        if (isinstance(self.min_assets, bool) or not isinstance(self.min_assets, int)
                or self.min_assets < 2):
            raise ValueError("min_assets must be an integer >= 2")
        if not np.isfinite(self.reference_ic) or not np.isfinite(self.reference_coverage):
            raise ValueError("monitor reference IC and coverage must be finite")
        expected = set(DRIFT_DIMENSIONS) - {"ic_decay", "coverage"}
        metrics = {str(k): float(v) for k, v in self.external_drift_evidence.items()}
        if set(metrics) != expected or not all(np.isfinite(tuple(metrics.values()))):
            raise ValueError("monitor policy requires finite non-QE drift dimensions")
        thresholds = {
            name: {
                "monitor": threshold.monitor, "retrain": threshold.retrain,
                "block": threshold.block,
            }
            for name, threshold in sorted(self.drift_contract.thresholds.items())
        }
        payload = {
            "policy_id": self.policy_id, "min_mature_dates": self.min_mature_dates,
            "min_assets": self.min_assets, "reference_ic": self.reference_ic,
            "reference_coverage": self.reference_coverage,
            "external_drift_evidence": metrics, "drift_thresholds": thresholds,
        }
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        object.__setattr__(self, "external_drift_evidence", MappingProxyType(metrics))
        object.__setattr__(self, "content_hash", digest)


def _subset_label(label: LabelBundle, indices: np.ndarray) -> LabelBundle:
    take = indices.tolist()
    timing = {
        name: (
            tuple(getattr(label, name)[i] for i in take)
            if getattr(label, name) else ()
        )
        for name in (
            "decision_time", "execution_time", "signal_available_time",
            "label_start_time", "label_end_time", "observation_time",
        )
    }
    values = label.values[indices]
    validity = None if label.validity is None else label.validity[indices]
    return LabelBundle(
        target_id=label.target_id, values=values, horizon=label.horizon,
        execution_delay=label.execution_delay, validity=validity,
        source_ref=label.source_ref, calendar_ref=label.calendar_ref,
        price_convention=label.price_convention, metadata=dict(label.metadata),
        asset_axis=label.asset_axis, schema_version=label.schema_version, **timing,
    )


def monitor_fixed_daily_update(
    *, artifact_id: str, generation_coordinator: Any, artifact_resolver: Any,
    segment_generation_ids: tuple[str, ...], label_bundle: LabelBundle,
    asof: Any, monitor_policy: FixedMonitorPolicy,
    campaign_policy: CampaignRequestPolicy | None = None,
) -> FixedDailyMonitoringResult:
    """Evaluate only PIT-mature labels and feed real QE metrics to the monitor."""
    active = generation_coordinator.resolve_active(artifact_id)
    if active is None:
        raise ValueError("fixed daily factor value has no active published generation")
    if (not isinstance(segment_generation_ids, tuple) or not segment_generation_ids
            or len(segment_generation_ids) > 64
            or len(set(segment_generation_ids)) != len(segment_generation_ids)):
        raise ValueError("segment_generation_ids must be 1..64 unique durable refs")
    if active["generation_id"] not in segment_generation_ids:
        raise ValueError("bounded segment refs must include the current active generation")
    rows = []
    for generation_id in segment_generation_ids:
        found = generation_coordinator.db.query(
            "SELECT * FROM artifact_generations WHERE generation_id=? AND artifact_id=? "
            "AND status='COMPLETE'", (generation_id, artifact_id),
        )
        if len(found) != 1:
            raise ValueError("segment ref is not a COMPLETE generation of this artifact")
        rows.append(found[0])
    documents = []
    for row in rows:
        artifact = json.loads(row["artifact_json"])
        if (artifact.get("artifact_id") != artifact_id
                or artifact.get("content_hash") != row["content_hash"]):
            raise ValueError("historical artifact reference differs from generation bytes")
        artifact["created_at"] = pd.Timestamp(artifact["created_at"]).to_pydatetime()
        reference = ArtifactRef(**artifact)
        try:
            payload = artifact_resolver.open(reference)
        except Exception as exc:
            raise ValueError("published segment bytes are missing or unverifiable") from exc
        if (not isinstance(payload, bytes) or len(payload) != reference.size_bytes
                or hashlib.sha256(payload).hexdigest() != row["content_hash"]):
            raise ValueError("resolved physical factor bytes differ from durable reference")
        if payload.hex() != row["payload_hex"]:
            raise ValueError("resolved physical factor bytes differ from staged generation")
        documents.append(json.loads(payload))
    identity = None
    assets = None
    by_date: dict[pd.Timestamp, np.ndarray] = {}
    for document in documents:
        current_identity = (
            document.get("factor_definition_hash"), document.get("recipe_hash"),
            tuple(document.get("fitted_state_refs", ())),
        )
        if not all(current_identity[:2]):
            raise ValueError("published segment lacks frozen factor/recipe identity")
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError("bounded segment refs mix factor, recipe, or fitted state")
        current_assets = tuple(document.get("assets", ()))
        current_values = np.asarray(document.get("values"), dtype=np.float64)
        current_dates = tuple(pd.Timestamp(value) for value in document.get("dates", ()))
        if not current_dates or not current_assets or current_values.shape != (len(current_dates), len(current_assets)):
            raise ValueError("published fixed-update payload has invalid axes or values")
        if any(pd.isna(value) for value in current_dates) or len(set(current_dates)) != len(current_dates):
            raise ValueError("published segment dates must be unique and non-NaT")
        if len(set(current_assets)) != len(current_assets):
            raise ValueError("published segment assets must be unique")
        if assets is None:
            assets = current_assets
        elif assets != current_assets:
            raise ValueError("bounded segments have different asset axes")
        for date, row_values in zip(current_dates, current_values):
            prior = by_date.get(date)
            if prior is not None and not np.array_equal(prior, row_values, equal_nan=True):
                raise ValueError("overlapping segment coordinates contain conflicting values")
            by_date[date] = np.array(row_values, copy=True)
    dates = tuple(sorted(by_date))
    values = np.stack([by_date[date] for date in dates])
    assert assets is not None and identity is not None
    evaluated_window_hash = stable_content_hex(
        tag="E2E-H.monitor-window.v1",
        fields={
            "generation_ids": segment_generation_ids,
            "dates": tuple(value.isoformat() for value in dates),
            "assets": assets, "values": values,
            "factor_definition_hash": identity[0], "recipe_hash": identity[1],
            "fitted_state_refs": identity[2],
            "monitor_policy_hash": monitor_policy.content_hash,
        },
    )
    if not isinstance(label_bundle, LabelBundle):
        raise TypeError("label_bundle must be a frozen LabelBundle")
    if not label_bundle.source_ref or not label_bundle.calendar_ref:
        raise ValueError("label bundle requires source_ref and calendar_ref")
    if label_bundle.asset_axis is None or label_bundle.asset_axis.values is None:
        raise ValueError("label bundle requires explicit asset coordinates")
    if tuple(map(str, label_bundle.asset_axis.values.tolist())) != tuple(map(str, assets)):
        raise ValueError("label asset axis differs from published factor segment")
    if tuple(pd.Timestamp(value) for value in label_bundle.observation_time) != dates:
        raise ValueError("label observation axis differs from published factor segment")
    if label_bundle.values.shape != values.shape:
        raise ValueError("label value shape differs from published factor segment")

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ValueError("monitor asof must not be NaT")
    ends = tuple(pd.Timestamp(value) for value in label_bundle.label_end_time)
    if len(ends) != len(dates):
        raise ValueError("label_end_time cardinality differs from factor dates")
    if any(pd.isna(value) for value in ends):
        raise ValueError("label_end_time must not contain NaT")
    try:
        mature = np.asarray([value <= asof_ts for value in ends], dtype=bool)
    except TypeError as exc:
        raise ValueError("asof and label_end_time require one explicit timezone policy") from exc
    indices = np.flatnonzero(mature)
    if not len(indices):
        return FixedDailyMonitoringResult(
            "WAIT_FOR_MATURITY", LifecycleAction.WAIT_FOR_MATURITY,
            active["content_hash"], label_bundle.content_hash, 0, 0,
            monitor_policy_hash=monitor_policy.content_hash,
            generation_ids=segment_generation_ids,
            evaluated_window_hash=evaluated_window_hash,
        )

    if len(indices) < monitor_policy.min_mature_dates or len(assets) < monitor_policy.min_assets:
        return FixedDailyMonitoringResult(
            "INSUFFICIENT_MATURE_SAMPLE", LifecycleAction.WAIT_FOR_MATURITY,
            active["content_hash"], label_bundle.content_hash, len(indices),
            int(np.isfinite(label_bundle.values[indices]).sum()),
            monitor_policy_hash=monitor_policy.content_hash,
            generation_ids=segment_generation_ids,
            evaluated_window_hash=evaluated_window_hash,
        )

    factor_validity = np.isfinite(values[indices, :, None])
    factors = FactorBatch(
        factor_ids=(identity[0],),
        time_axis=AxisRef("time", "datetime64[ns]", len(indices),
                          values=np.asarray([dates[i] for i in indices])),
        asset_axis=AxisRef("asset", "object", len(assets), values=np.asarray(assets)),
        values=values[indices, :, None], validity=factor_validity,
        context_refs={"active_factor_value_ref": active["content_hash"],
                      "evaluated_window_ref": evaluated_window_hash,
                      "label_bundle_ref": label_bundle.content_hash},
        value_hash=evaluated_window_hash,
    )
    mature_labels = _subset_label(label_bundle, indices)
    evaluation = evaluate(
        factors, mature_labels, metrics=("rank_ic", "rank_ic_series", "coverage"),
        metric_parameters={
            "rank_ic": {"min_assets": monitor_policy.min_assets},
            "rank_ic_series": {"min_assets": monitor_policy.min_assets},
        },
    )
    factor_id = factors.factor_ids[0]
    rank_ic = evaluation.get_metric("rank_ic", factor_id)
    coverage = evaluation.get_metric("coverage", factor_id)
    series = evaluation.artifacts.get("rank_ic_series")
    valid_ic_dates = 0 if series is None else int(np.isfinite(series.values[:, 0]).sum())
    if rank_ic is None or coverage is None or not rank_ic.valid or not coverage.valid:
        return FixedDailyMonitoringResult(
            "INSUFFICIENT_MATURE_SAMPLE", LifecycleAction.WAIT_FOR_MATURITY,
            active["content_hash"], label_bundle.content_hash, len(indices),
            int(np.isfinite(mature_labels.values).sum()),
            monitor_policy_hash=monitor_policy.content_hash,
            generation_ids=segment_generation_ids,
            evaluated_window_hash=evaluated_window_hash,
            valid_ic_date_count=valid_ic_dates,
        )
    if valid_ic_dates < monitor_policy.min_mature_dates:
        return FixedDailyMonitoringResult(
            "INSUFFICIENT_MATURE_SAMPLE", LifecycleAction.WAIT_FOR_MATURITY,
            active["content_hash"], label_bundle.content_hash, len(indices),
            int(np.isfinite(mature_labels.values).sum()),
            rank_ic=float(rank_ic.value), coverage=float(coverage.value),
            monitor_policy_hash=monitor_policy.content_hash,
            generation_ids=segment_generation_ids,
            evaluated_window_hash=evaluated_window_hash,
            valid_ic_date_count=valid_ic_dates,
        )
    fixed = dict(monitor_policy.external_drift_evidence)
    current_ic, current_coverage = float(rank_ic.value), float(coverage.value)
    fixed["ic_decay"] = max(0.0, float(monitor_policy.reference_ic) - current_ic)
    fixed["coverage"] = abs(current_coverage - float(monitor_policy.reference_coverage))
    available_at = max(ends[i] for i in indices).isoformat()
    asof_text = asof_ts.isoformat()
    realised = RealisedMetric(
        prediction_id=active["content_hash"], metric_name="ic_decay",
        value=fixed["ic_decay"], horizon_sessions=label_bundle.horizon,
        decision_at=max(dates[i] for i in indices).isoformat(),
        metric_available_at=available_at, observed_at=asof_text,
    )
    actions = monitor_policy.drift_contract.assess(
        fixed, campaign_policy=campaign_policy,
        realised_metrics={"ic_decay": realised}, asof=asof_text,
    )
    return FixedDailyMonitoringResult(
        "MONITORED_PARTIAL_EXTERNAL_DIMENSIONS", None,
        active["content_hash"], label_bundle.content_hash,
        len(indices), int(np.isfinite(mature_labels.values).sum()),
        current_ic, current_coverage, actions, monitor_policy.content_hash,
        segment_generation_ids, evaluated_window_hash, valid_ic_dates,
    )


__all__ = ["FixedDailyMonitoringResult", "FixedMonitorPolicy", "monitor_fixed_daily_update"]
