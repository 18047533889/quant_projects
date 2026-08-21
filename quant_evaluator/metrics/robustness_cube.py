"""SpecRobustnessCube: metric dispersion across slice specs.

Given a list of *specs* — (slice family x metric) combinations — the cube
runs each metric on every slice, content-addresses every spec, and
summarizes per-metric dispersion across specs. A factor whose metric
value collapses in one slice (e.g. an IC that flips sign or vanishes in
a sub-period or sub-universe) is flagged as fragile.

The cube is intentionally simple: it composes
:class:`quant_evaluator.metrics.slicing.SliceEngine` and never
re-implements metric math.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import MetricArtifact, ScalarMetricArtifact
from quant_evaluator.metrics.slicing import SliceEngine, SliceView

__all__ = ["MetricSpec", "RobustnessSummary", "SpecRobustnessCube"]


def _hash_content(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricSpec:
    """One (slices x metric_fn) combination.

    Attributes:
        metric_id: Registry/dotted name of the metric (used as the cube's
            metric axis).
        metric_fn: ``callable(batch, labels) -> MetricArtifact``.
        slices: The slice views this spec evaluates the metric over.
        spec_id: Content-addressed sha256 of (metric_id, slice spec_ids).
    """

    metric_id: str
    metric_fn: Callable[[FactorBatch, LabelBundle], MetricArtifact]
    slices: Tuple[SliceView, ...]
    spec_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.metric_id).strip():
            raise InvalidContractError("MetricSpec.metric_id must be non-empty")
        if not callable(self.metric_fn):
            raise InvalidContractError("MetricSpec.metric_fn must be callable")
        if not self.slices:
            raise InvalidContractError(
                f"MetricSpec {self.metric_id!r} has no slices"
            )
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(
            self,
            "spec_id",
            _hash_content(
                {
                    "metric_id": self.metric_id,
                    "slice_spec_ids": [v.spec_id for v in self.slices],
                    "slice_keys": [v.slice_key for v in self.slices],
                }
            ),
        )


@dataclass(frozen=True)
class RobustnessSummary:
    """Dispersion of one metric across all slices of one spec.

    All arrays are per-factor (shape ``(F,)``). A factor is flagged
    fragile when its value collapses somewhere: either the sign flips
    between slices, the range exceeds ``max_range`` times the mean
    magnitude, or any slice yields a non-finite value while others are
    finite.
    """

    metric_id: str
    spec_id: str
    slice_keys: Tuple[str, ...]
    min_values: np.ndarray
    max_values: np.ndarray
    mean_values: np.ndarray
    sign_flip: np.ndarray
    fragile: Tuple[int, ...]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "spec_id": self.spec_id,
            "slice_keys": list(self.slice_keys),
            "min_values": self.min_values.tolist(),
            "max_values": self.max_values.tolist(),
            "mean_values": self.mean_values.tolist(),
            "sign_flip": self.sign_flip.tolist(),
            "fragile_factor_indices": list(self.fragile),
            "reason": self.reason,
        }


class SpecRobustnessCube:
    """Evaluate (slice x metric) specs and summarize metric dispersion.

    Usage::

        cube = SpecRobustnessCube()
        result = cube.evaluate(batch, labels, specs=[spec_a, spec_b])
        result.artifacts["spec_a"][("time:whole[0:6]", "mean_ic")]
        result.summaries["mean_ic"].fragile  # (0,) -> factor 0 is fragile
    """

    def __init__(
        self,
        engine: Optional[SliceEngine] = None,
        max_range_ratio: float = 2.0,
        min_abs_mean: float = 1e-12,
    ) -> None:
        if max_range_ratio <= 0:
            raise InvalidContractError("max_range_ratio must be positive")
        self.engine = engine if engine is not None else SliceEngine()
        self.max_range_ratio = float(max_range_ratio)
        self.min_abs_mean = float(min_abs_mean)

    def evaluate(
        self,
        batch: FactorBatch,
        labels: LabelBundle,
        specs: Sequence[MetricSpec],
    ) -> "CubeResult":
        """Run every spec; return artifacts keyed by (slice, metric)."""
        if not specs:
            raise InvalidContractError("SpecRobustnessCube requires at least one spec")
        artifacts: Dict[str, Dict[Tuple[str, str], MetricArtifact]] = {}
        metric_ids = []
        for spec in specs:
            if spec.metric_id in artifacts:
                raise InvalidContractError(
                    f"Duplicate metric_id {spec.metric_id!r} across specs; "
                    f"metric ids must be unique within one cube evaluation"
                )
            metric_ids.append(spec.metric_id)
            per_slice = self.engine.compute(spec.metric_fn, batch, labels, spec.slices)
            cell: Dict[Tuple[str, str], MetricArtifact] = {}
            for slice_key, artifact in per_slice.items():
                cell[(slice_key, spec.metric_id)] = artifact
            artifacts[spec.metric_id] = cell

        summaries = {
            metric_id: self._summarize(metric_id, spec, artifacts[metric_id])
            for metric_id, spec in zip(metric_ids, specs)
        }
        return CubeResult(
            artifacts=artifacts,
            summaries=summaries,
            spec_ids={spec.metric_id: spec.spec_id for spec in specs},
        )

    def _summarize(
        self,
        metric_id: str,
        spec: MetricSpec,
        cell: Mapping[Tuple[str, str], MetricArtifact],
    ) -> RobustnessSummary:
        slice_keys = tuple(k for (k, m) in cell if m == metric_id)
        # Stack per-slice per-factor scalar values into (S, F).
        frames: List[np.ndarray] = []
        for key in slice_keys:
            artifact = cell[(key, metric_id)]
            if isinstance(artifact, ScalarMetricArtifact):
                frames.append(np.asarray(artifact.values, dtype=float))
            else:
                raise InvalidContractError(
                    f"SpecRobustnessCube dispersion summaries require "
                    f"ScalarMetricArtifact payloads, got "
                    f"{type(artifact).__name__} for slice {key!r}"
                )
        matrix = np.vstack(frames)  # (S, F)
        with np.errstate(invalid="ignore"):
            vmin = np.nanmin(matrix, axis=0)
            vmax = np.nanmax(matrix, axis=0)
            vmean = np.nanmean(matrix, axis=0)

        sign_flip = (vmin < 0) & (vmax > 0)
        any_finite = np.isfinite(vmean)
        non_finite_slice = np.any(~np.isfinite(matrix), axis=0) & any_finite
        mean_mag = np.abs(vmean)
        wide_range = (
            any_finite
            & (mean_mag > self.min_abs_mean)
            & ((vmax - vmin) > self.max_range_ratio * mean_mag)
        )
        fragile_mask = sign_flip | non_finite_slice | wide_range
        reasons = []
        if np.any(sign_flip):
            reasons.append("sign_flip")
        if np.any(non_finite_slice):
            reasons.append("non_finite_slice")
        if np.any(wide_range):
            reasons.append("range_gt_%.1fx_mean" % self.max_range_ratio)
        if not reasons:
            reasons.append("none")
        return RobustnessSummary(
            metric_id=metric_id,
            spec_id=spec.spec_id,
            slice_keys=slice_keys,
            min_values=vmin,
            max_values=vmax,
            mean_values=vmean,
            sign_flip=sign_flip,
            fragile=tuple(int(i) for i in np.nonzero(fragile_mask)[0]),
            reason="+".join(reasons),
        )


@dataclass(frozen=True)
class CubeResult:
    """Cube cell artifacts plus per-metric dispersion summaries."""

    artifacts: Dict[str, Dict[Tuple[str, str], MetricArtifact]]
    summaries: Dict[str, RobustnessSummary]
    spec_ids: Dict[str, str]

    def values(self, metric_id: str) -> np.ndarray:
        """All slice values for one metric as a (S, F) matrix."""
        cell = self.artifacts[metric_id]
        keys = sorted(k for (k, m) in cell if m == metric_id)
        return np.vstack(
            [np.asarray(cell[(k, metric_id)].values, dtype=float) for k in keys]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_ids": dict(self.spec_ids),
            "summaries": {
                mid: s.to_dict() for mid, s in self.summaries.items()
            },
        }
