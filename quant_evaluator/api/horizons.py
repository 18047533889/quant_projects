"""Public predictive-IC evaluation across explicit forward-label horizons."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact
from quant_evaluator.contracts.statistical_evidence import HorizonCurveEvidence


def _frozen_mask(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    frozen = np.array(array, copy=True)
    frozen.flags.writeable = False
    return frozen


@dataclass(frozen=True)
class HorizonEvaluationBundle:
    """Immutable multi-horizon result retaining every daily IC and sample mask."""

    factor_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    sample_policy: str
    as_of: Any
    daily_ic_artifacts: Mapping[int, SeriesMetricArtifact]
    curve_evidence: Mapping[str, HorizonCurveEvidence]
    maturity_masks: Mapping[int, np.ndarray]
    evaluation_masks: Mapping[int, np.ndarray]
    common_mask: np.ndarray
    maturity_counts: Mapping[int, int]
    sample_counts: Mapping[int, int]
    label_content_refs: Mapping[int, str]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors, horizons = tuple(self.factor_ids), tuple(self.horizons)
        if not factors or not horizons:
            raise ValueError("factor_ids and horizons cannot be empty")
        if set(self.daily_ic_artifacts) != set(horizons):
            raise ValueError("daily IC artifacts must cover every horizon")
        first = self.daily_ic_artifacts[horizons[0]]
        t = first.values.shape[0]
        n = np.asarray(self.common_mask).shape[1]
        artifacts = {}
        for horizon in horizons:
            artifact = self.daily_ic_artifacts[horizon]
            if not isinstance(artifact, SeriesMetricArtifact) or artifact.values.shape != (t, len(factors)):
                raise ValueError("every horizon requires a full aligned SeriesMetricArtifact")
            artifacts[horizon] = artifact
        maturity = {h: _frozen_mask(self.maturity_masks[h], (t,), f"maturity_masks[{h}]")
                    for h in horizons}
        evaluation = {h: _frozen_mask(self.evaluation_masks[h], (t, n), f"evaluation_masks[{h}]")
                      for h in horizons}
        object.__setattr__(self, "factor_ids", factors)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "daily_ic_artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "curve_evidence", MappingProxyType(dict(self.curve_evidence)))
        object.__setattr__(self, "maturity_masks", MappingProxyType(maturity))
        object.__setattr__(self, "evaluation_masks", MappingProxyType(evaluation))
        object.__setattr__(self, "common_mask", _frozen_mask(self.common_mask, (t, n), "common_mask"))
        object.__setattr__(self, "maturity_counts", MappingProxyType(dict(self.maturity_counts)))
        object.__setattr__(self, "sample_counts", MappingProxyType(dict(self.sample_counts)))
        object.__setattr__(self, "label_content_refs", MappingProxyType(dict(self.label_content_refs)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _same_axis(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object))


def evaluate_horizons(
    factors: FactorBatch,
    labels: Mapping[int, LabelBundle],
    *,
    as_of: Any,
    sample_policy: str = "common",
    min_assets: int = 10,
    min_periods: int = 20,
    zero_tolerance: float = 1e-12,
    backend: Any = None,
) -> HorizonEvaluationBundle:
    """Evaluate predictive rank-IC curves against explicit label bundles.

    ``common`` intersects label validity and label-time maturity at the T×N
    cell level before every leaf evaluation. ``per_horizon`` preserves each
    label's own eligible sample and marks the curves as non-comparable samples.
    """
    if not isinstance(factors, FactorBatch):
        raise TypeError("factors must be a FactorBatch")
    if sample_policy not in {"common", "per_horizon"}:
        raise ValueError("sample_policy must be 'common' or 'per_horizon'")
    if as_of is None:
        raise ValueError("as_of is required")
    if isinstance(min_assets, bool) or not isinstance(min_assets, int) or min_assets < 2:
        raise ValueError("min_assets must be an integer >= 2")
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError("min_periods must be a positive integer")
    if not isinstance(labels, Mapping) or len(labels) < 2:
        raise ValueError("labels must contain at least two horizons")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in labels):
        raise ValueError("horizons must be unique positive integers")
    horizons = tuple(sorted(labels))
    if factors.time_axis.values is None or factors.asset_axis.values is None:
        raise ValueError("factor time and asset axes require explicit coordinates")
    t, n = factors.num_times, factors.num_assets

    decision_axis = observation_axis = asset_axis = None
    own_masks: dict[int, np.ndarray] = {}
    maturity_masks: dict[int, np.ndarray] = {}
    label_refs: dict[int, str] = {}
    for horizon in horizons:
        label = labels[horizon]
        if not isinstance(label, LabelBundle) or label.horizon != horizon:
            raise ValueError(f"label mapping key {horizon} must match LabelBundle.horizon")
        if label.values.shape != (t, n) or label.asset_axis is None or label.asset_axis.values is None:
            raise ValueError("every label must be a bound T×N panel")
        if not label.observation_time:
            raise ValueError("every label requires an explicit observation_time axis")
        if not _same_axis(label.observation_time, factors.time_axis.values):
            raise ValueError("label observation axis differs from factor time axis")
        if not _same_axis(label.asset_axis.values, factors.asset_axis.values):
            raise ValueError("label asset axis differs from factor asset axis")
        if decision_axis is None:
            decision_axis = label.decision_time
            observation_axis = label.observation_time
            asset_axis = label.asset_axis.values
        elif (not _same_axis(label.decision_time, decision_axis)
              or not _same_axis(label.observation_time, observation_axis)
              or not _same_axis(label.asset_axis.values, asset_axis)):
            raise ValueError("all horizon labels must share decision, observation and asset axes")
        try:
            mature = np.asarray([end <= as_of for end in label.label_end_time], dtype=bool)
        except TypeError as exc:
            raise ValueError("as_of and label_end_time must be explicitly comparable") from exc
        validity = np.isfinite(label.values)
        if label.validity is not None:
            validity &= label.validity
        maturity_masks[horizon] = mature
        own_masks[horizon] = validity & mature[:, None]
        label_refs[horizon] = label.content_hash

    common_mask = np.logical_and.reduce([own_masks[h] for h in horizons])
    selected_masks = {
        h: common_mask if sample_policy == "common" else own_masks[h]
        for h in horizons
    }
    artifacts: dict[int, SeriesMetricArtifact] = {}
    from quant_evaluator.runtime.evaluator import evaluate
    for horizon in horizons:
        selected_label = replace(labels[horizon], validity=selected_masks[horizon])
        result = evaluate(
            factors, selected_label, metrics=("rank_ic_series",), backend=backend,
            metric_parameters={"rank_ic_series": {"min_assets": min_assets}},
        )
        artifacts[horizon] = result.artifacts["rank_ic_series"]

    from quant_evaluator.metrics.statistical_evidence import build_horizon_curve_evidence
    curves: dict[str, HorizonCurveEvidence] = {}
    for factor_index, factor_id in enumerate(factors.factor_ids):
        curves[factor_id] = build_horizon_curve_evidence(
            {h: artifacts[h].values[:, factor_index] for h in horizons},
            label_refs=label_refs, min_periods=min_periods,
            zero_tolerance=zero_tolerance,
        )
    return HorizonEvaluationBundle(
        factor_ids=tuple(factors.factor_ids), horizons=horizons,
        sample_policy=sample_policy, as_of=as_of,
        daily_ic_artifacts=artifacts, curve_evidence=curves,
        maturity_masks=maturity_masks, evaluation_masks=selected_masks,
        common_mask=common_mask,
        maturity_counts={h: int(maturity_masks[h].sum()) for h in horizons},
        sample_counts={h: int(selected_masks[h].sum()) for h in horizons},
        label_content_refs=label_refs,
        provenance={
            "sample_policy": sample_policy,
            "samples_comparable_across_horizons": sample_policy == "common",
            "min_assets": min_assets,
            "min_periods": min_periods,
            "zero_tolerance": curves[factors.factor_ids[0]].zero_tolerance,
            "backend": getattr(backend, "value", backend),
            "factor_value_hash": factors.value_hash,
        },
    )


__all__ = ["HorizonEvaluationBundle", "evaluate_horizons"]
