"""
Treatment eligibility engine for the auto-treatment optimizer.

The engine takes a :class:`FactorProfileArtifact` and an
:class:`ExistingTreatmentSignature` and produces a
:class:`TreatmentSearchSpace` — the set of *legal* transforms (with parameter
ranges) that the optimizer may search over.

The engine is rule-based (phase 1). It encodes domain knowledge about which
treatments are meaningful for which factor families, and prunes treatments
that would be meaningless or duplicative.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from factor_preprocess.contracts.factor_profile import FactorProfileArtifact
from factor_preprocess.contracts.treatment_lineage import (
    ExistingTreatmentSignature,
    TransformSemanticID,
    TransformStage,
    TransformStep,
    TransformLineage,
)

# Known semantic families.
PRICE_VOLUME = "PRICE_VOLUME"
HIGH_TURNOVER = "HIGH_TURNOVER"
FUNDAMENTAL = "FUNDAMENTAL"
SPARSE_UPDATE = "SPARSE_UPDATE"
EVENT = "EVENT"
BINARY = "BINARY"
DISCRETE = "DISCRETE"

# RAW / NO-OP is always a candidate in every search space.
RAW_SEMANTIC_ID = "RAW:noop"

# Temporal smoothing transforms (with default parameter ranges).
SMOOTHING_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "trailing_sma": {"window": (3.0, 60.0)},
    "ewma": {"halflife": (3.0, 60.0)},
    "kama": {"er_window": (5.0, 30.0), "fast_span": (2.0, 10.0), "slow_span": (20.0, 60.0)},
    "one_sided_iir_lowpass": {"alpha": (0.05, 0.5)},
    "kalman_local_level": {"process_noise": (0.001, 0.1), "measurement_noise": (0.1, 1.0)},
    "trailing_median": {"window": (3.0, 30.0)},
}

# Short-halflife event-decay smoothing.
EVENT_DECAY_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "event_decay": {"halflife": (1.0, 5.0)},
}

# Freshness-aware fill for sparse/fundamental factors.
FRESHNESS_FILL_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "freshness_aware_fill": {"max_lag": (1.0, 20.0)},
}

# Outlier / winsor.
WINSOR_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cs_winsor": {"lower": (0.005, 0.05), "upper": (0.95, 0.995)},
}

# Representation / scaling.
RANK_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cs_rank": {"pct": (1.0, 1.0)},
}
ZSCORE_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cs_zscore": {"ddof": (1.0, 1.0)},
}

# Neutralization.
NEUTRALIZATION_TRANSFORMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ols_neutralize": {},
    "industry_neutral": {},
    "size_neutral": {},
    "dual_neutral": {},
}


@dataclass(frozen=True)
class TreatmentSearchSpace:
    """The set of legal transforms (with parameter ranges) for a factor.

    ``allowed_transform_ids`` is a dict mapping transform name -> parameter
    ranges. ``RAW:noop`` is always present.
    """

    factor_id: str
    allowed_transform_ids: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=dict
    )
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "allowed_transform_ids", dict(self.allowed_transform_ids))
        object.__setattr__(self, "notes", tuple(self.notes))

    def allows(self, transform_name: str) -> bool:
        """True if the given transform name is in the search space."""
        return transform_name in self.allowed_transform_ids

    def transform_names(self) -> List[str]:
        """Return the allowed transform names (RAW always included)."""
        return list(self.allowed_transform_ids.keys())


class TreatmentEligibilityEngine:
    """Rule-based phase-1 eligibility engine."""

    def __init__(self):
        self._smoothing_budget = 1.0  # full budget by default

    def build_search_space(
        self,
        profile: FactorProfileArtifact,
        existing: Optional[ExistingTreatmentSignature] = None,
    ) -> TreatmentSearchSpace:
        """Build the legal search space for a factor profile."""
        existing = existing or ExistingTreatmentSignature()
        family = profile.semantic_family.upper()

        allowed: Dict[str, Dict[str, Tuple[float, float]]] = {}
        notes: List[str] = []

        # RAW / NO-OP is ALWAYS a candidate.
        allowed[RAW_SEMANTIC_ID] = {}

        # ---- Temporal smoothing budget ----
        smoothing_budget = self._smoothing_budget
        if existing.temporal_smoothing:
            # Already smoothed: reduce the smoothing budget.
            smoothing_budget *= 0.5
            notes.append("existing temporal smoothing -> smoothing budget halved")

        # ---- Family-specific rules ----
        if family in (PRICE_VOLUME, HIGH_TURNOVER):
            # Price/volume and high-turnover factors CAN be smoothed.
            allowed.update(self._smoothing_transforms(smoothing_budget))
            allowed.update(WINSOR_TRANSFORMS)
            allowed.update(RANK_TRANSFORMS)
            allowed.update(ZSCORE_TRANSFORMS)
            allowed.update(self._neutralization_transforms(existing))
            notes.append("price_volume/high_turnover: temporal smoothing allowed")

        elif family in (FUNDAMENTAL, SPARSE_UPDATE):
            # Fundamental / sparse-update factors: forbid short temporal
            # smoothing; allow freshness-aware fill, winsor, rank/zscore,
            # industry/size/dual neutralization.
            allowed.update(FRESHNESS_FILL_TRANSFORMS)
            allowed.update(WINSOR_TRANSFORMS)
            allowed.update(RANK_TRANSFORMS)
            allowed.update(ZSCORE_TRANSFORMS)
            allowed.update(self._neutralization_transforms(existing))
            notes.append("fundamental/sparse_update: short temporal smoothing forbidden")

        elif family == EVENT:
            # Event factors: allow event-decay (short-halflife smoothing),
            # forbid meaningless long-window smoothing.
            allowed.update(EVENT_DECAY_TRANSFORMS)
            allowed.update(WINSOR_TRANSFORMS)
            allowed.update(RANK_TRANSFORMS)
            allowed.update(ZSCORE_TRANSFORMS)
            allowed.update(self._neutralization_transforms(existing))
            notes.append("event: event-decay allowed, long-window smoothing forbidden")

        elif family in (BINARY, DISCRETE):
            # Binary/discrete factors: forbid default winsor/zscore unless an
            # explicit policy is present. We do not auto-admit them here.
            allowed.update(RANK_TRANSFORMS)
            allowed.update(self._neutralization_transforms(existing))
            notes.append("binary/discrete: winsor/zscore require explicit policy")

        else:
            # Unknown family: conservative default — RAW + rank only.
            allowed.update(RANK_TRANSFORMS)
            notes.append(f"unknown family {family!r}: conservative default")

        # ---- Low-turnover profile: reduce smoothing budget ----
        turnover = profile.time_behavior.get("raw_turnover")
        if turnover is not None and turnover < 0.1:
            # Already low turnover: reduce smoothing budget further.
            for name in list(allowed.keys()):
                if name in SMOOTHING_TRANSFORMS or name in EVENT_DECAY_TRANSFORMS:
                    allowed[name] = self._halve_ranges(allowed[name])
            notes.append("low raw_turnover -> smoothing budget trimmed")

        # ---- Already industry-neutral: prune industry neutralization ----
        if existing.industry_neutral:
            allowed.pop("industry_neutral", None)
            allowed.pop("dual_neutral", None)
            notes.append("already industry-neutral -> industry/dual neutralization pruned")

        return TreatmentSearchSpace(
            factor_id=profile.factor_id,
            allowed_transform_ids=allowed,
            notes=tuple(notes),
        )

    # -- helpers -----------------------------------------------------------

    def _smoothing_transforms(
        self, budget: float
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Return smoothing transforms scaled by the budget."""
        if budget >= 1.0:
            return {k: dict(v) for k, v in SMOOTHING_TRANSFORMS.items()}
        return {
            name: self._halve_ranges(params)
            for name, params in SMOOTHING_TRANSFORMS.items()
        }

    def _halve_ranges(
        self, params: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Halve the upper bound of each parameter range (trim budget)."""
        out: Dict[str, Tuple[float, float]] = {}
        for key, (lo, hi) in params.items():
            out[key] = (lo, max(lo, hi * 0.5))
        return out

    def _neutralization_transforms(
        self, existing: ExistingTreatmentSignature
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Return neutralization transforms, pruning already-applied ones."""
        out: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if not existing.industry_neutral:
            out["industry_neutral"] = {}
            out["dual_neutral"] = {}
        if not existing.size_neutral:
            out["size_neutral"] = {}
        out["ols_neutralize"] = {}
        return out


__all__ = [
    "TreatmentEligibilityEngine",
    "TreatmentSearchSpace",
    "PRICE_VOLUME",
    "HIGH_TURNOVER",
    "FUNDAMENTAL",
    "SPARSE_UPDATE",
    "EVENT",
    "BINARY",
    "DISCRETE",
    "RAW_SEMANTIC_ID",
]
