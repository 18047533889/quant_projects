"""
Treatment eligibility engine for the auto-treatment optimizer.

The engine takes a :class:`FactorProfileArtifact` and an
:class:`ExistingTreatmentSignature` and produces a
:class:`TreatmentSearchSpace` — the set of *legal* transforms (with parameter
ranges) that the optimizer may search over.

Single-transform-authority (DLIB-FP-002)
----------------------------------------
This engine does NOT maintain a second ``transform_id -> parameters`` catalog.
The executable catalog authority is :class:`TransformRegistry`
(``registry/transforms.py``). The engine instead uses
:class:`EligibilityRuleRegistry` (``eligibility/rules.py``) to decide which
*semantic treatment families* a factor type may search, then maps each family
to the concrete transform ids in the canonical TransformRegistry. Every
proposed transform is validated against the canonical registry, so no dangling
or duplicate transform id can ever be returned.

Every transform the engine can propose must resolve to a real executable
transform id in the canonical registry (industry_neutral / size_neutral /
dual_neutral / event_decay / freshness_aware_fill all resolve to real
executable ids — DLIB-FP-014).

RAW is always a candidate so "treated got worse" is discoverable (DLIB-FP-024).
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
from factor_preprocess.eligibility.rules import (
    get_default_eligibility_rules,
    TreatmentFamily,
    FactorFamily,
    RAW_SEMANTIC_ID,
)
from factor_preprocess.registry.transforms import (
    get_default_registry,
    TransformCategory,
)
from factor_preprocess.contracts._deep_freeze import deep_freeze

# Backwards-compatible family name constants.
PRICE_VOLUME = FactorFamily.PRICE_VOLUME.value
HIGH_TURNOVER = FactorFamily.HIGH_TURNOVER.value
FUNDAMENTAL = FactorFamily.FUNDAMENTAL.value
SPARSE_UPDATE = FactorFamily.SPARSE_UPDATE.value
EVENT = FactorFamily.EVENT.value
BINARY = FactorFamily.BINARY.value
DISCRETE = FactorFamily.DISCRETE.value


# Map a semantic treatment family -> concrete transform ids + parameter ranges.
# Parameter ranges are the *search bounds* the optimizer sweeps; they are kept
# here (as search policy), but the existence/admission/semantics of each
# transform come from the canonical TransformRegistry. Every name here must
# resolve in the registry (verified at search-space construction).
_FAMILY_TO_TRANSFORMS: Dict[TreatmentFamily, Dict[str, Dict[str, Tuple[float, float]]]] = {
    TreatmentFamily.SMOOTHING: {
        "trailing_sma": {"window": (3.0, 60.0)},
        "ewma": {"halflife": (3.0, 60.0)},
        "kama": {"er_window": (5.0, 30.0), "fast_span": (2.0, 10.0), "slow_span": (20.0, 60.0)},
        "one_sided_iir_lowpass": {"alpha": (0.05, 0.5)},
        "kalman_local_level": {"process_noise": (0.001, 0.1), "measurement_noise": (0.1, 1.0)},
        "trailing_median": {"window": (3.0, 30.0)},
        "robust_ewma": {"halflife": (3.0, 60.0)},
    },
    TreatmentFamily.EVENT_DECAY: {
        "event_decay": {"halflife": (1.0, 5.0)},
    },
    TreatmentFamily.FRESHNESS_FILL: {
        "freshness_aware_fill": {"max_lag": (1.0, 20.0)},
    },
    TreatmentFamily.WINSOR: {
        "cs_winsor": {"lower": (0.005, 0.05), "upper": (0.95, 0.995)},
    },
    TreatmentFamily.RANK: {
        "cs_rank": {"pct": (1.0, 1.0)},
    },
    TreatmentFamily.ZSCORE: {
        "cs_zscore": {"ddof": (1.0, 1.0)},
    },
    TreatmentFamily.NEUTRALIZATION: {
        "ols_neutralize": {},
        "industry_neutral": {},
        "size_neutral": {},
        "dual_neutral": {},
    },
}


@dataclass(frozen=True)
class TreatmentSearchSpace:
    """The set of legal transforms (with parameter ranges) for a factor.

    ``allowed_transform_ids`` is a dict mapping transform name -> parameter
    ranges. ``RAW:noop`` is always present. Deeply frozen so it is hash-safe
    (DLIB-FP-026).
    """

    factor_id: str
    allowed_transform_ids: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=dict
    )
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(
            self,
            "allowed_transform_ids",
            deep_freeze(self.allowed_transform_ids),
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    def allows(self, transform_name: str) -> bool:
        """True if the given transform name is in the search space."""
        return transform_name in self.allowed_transform_ids

    def transform_names(self) -> List[str]:
        """Return the allowed transform names (RAW always included)."""
        return list(self.allowed_transform_ids.keys())


class TreatmentEligibilityEngine:
    """Rule-based phase-1 eligibility engine (single-transform-authority)."""

    def __init__(self, rules=None, registry=None):
        self._rules = rules or get_default_eligibility_rules()
        self._registry = registry or get_default_registry()
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

        # RAW / NO-OP is ALWAYS a candidate (DLIB-FP-024).
        allowed[RAW_SEMANTIC_ID] = {}

        # ---- Temporal smoothing budget ----
        smoothing_budget = self._smoothing_budget
        if existing.temporal_smoothing:
            smoothing_budget *= 0.5
            notes.append("existing temporal smoothing -> smoothing budget halved")

        # ---- Family-specific rules via the rule registry (DLIB-FP-023) ----
        allowed_families = self._rules.allowed_families(family)
        policy_version = self._rules.policy_version_for(family)
        if policy_version:
            notes.append(f"family-rule policy_version={policy_version}")

        for tf in allowed_families:
            if tf == TreatmentFamily.SMOOTHING:
                allowed.update(self._smoothing_transforms(smoothing_budget))
                notes.append(f"{family.lower()}: temporal smoothing allowed")
            elif tf == TreatmentFamily.EVENT_DECAY:
                allowed.update(_FAMILY_TO_TRANSFORMS[TreatmentFamily.EVENT_DECAY])
                notes.append("event: event-decay allowed, long-window smoothing forbidden")
            elif tf == TreatmentFamily.FRESHNESS_FILL:
                allowed.update(_FAMILY_TO_TRANSFORMS[TreatmentFamily.FRESHNESS_FILL])
                notes.append("fundamental/sparse_update: freshness fill allowed")
            elif tf == TreatmentFamily.WINSOR:
                allowed.update(_FAMILY_TO_TRANSFORMS[TreatmentFamily.WINSOR])
            elif tf == TreatmentFamily.RANK:
                allowed.update(_FAMILY_TO_TRANSFORMS[TreatmentFamily.RANK])
            elif tf == TreatmentFamily.ZSCORE:
                allowed.update(_FAMILY_TO_TRANSFORMS[TreatmentFamily.ZSCORE])
            elif tf == TreatmentFamily.NEUTRALIZATION:
                allowed.update(self._neutralization_transforms(existing))
            else:
                notes.append(f"unhandled family {tf.value!r}")

        # ---- Single-transform-authority validation ----
        # Every proposed transform MUST resolve to a canonical registry entry.
        # (DLIB-FP-002 / DLIB-FP-014.)
        for name in list(allowed.keys()):
            if name == RAW_SEMANTIC_ID:
                continue
            meta = self._registry.get(name)
            if meta is None:
                allowed.pop(name, None)
                notes.append(f"dropped unresolvable transform {name!r} (not in registry)")

        # ---- Low-turnover profile: reduce smoothing budget ----
        turnover = profile.time_behavior.get("raw_turnover")
        if turnover is not None and turnover < 0.1:
            for name in list(allowed.keys()):
                if name in _FAMILY_TO_TRANSFORMS[TreatmentFamily.SMOOTHING] or \
                   name in _FAMILY_TO_TRANSFORMS[TreatmentFamily.EVENT_DECAY]:
                    allowed[name] = self._halve_ranges(allowed[name])
            notes.append("low raw_turnover -> smoothing budget trimmed")

        # ---- Already industry-neutral: prune duplicate neutralization ----
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
        base = _FAMILY_TO_TRANSFORMS[TreatmentFamily.SMOOTHING]
        if budget >= 1.0:
            return {k: dict(v) for k, v in base.items()}
        return {
            name: self._halve_ranges(params) for name, params in base.items()
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
