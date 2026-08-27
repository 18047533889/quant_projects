"""
EligibilityRuleRegistry — the *single* authority for which semantic treatment
families a factor type may search (DLIB-FP-014).

The registry answers: *given a factor type, which semantic treatment families
are legal to search?* It does NOT maintain a transform_id -> parameters
catalog (that is :class:`TransformRegistry`'s job). The rule registry is the
semantic layer; the transform registry is the executable layer. Together they
let the eligibility engine propose only transforms that (a) the factor type is
allowed to search AND (b) actually resolve to a registered, executable
transform in the canonical TransformRegistry.

Every proposal the engine makes is validated against the canonical registry so
no dangling transform id can ever be returned (single-transform-authority).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum


class TreatmentFamily(str, Enum):
    """Semantic treatment families a factor type may search."""

    SMOOTHING = "smoothing"
    EVENT_DECAY = "event_decay"
    FRESHNESS_FILL = "freshness_fill"
    WINSOR = "winsor"
    RANK = "rank"
    ZSCORE = "zscore"
    NEUTRALIZATION = "neutralization"


class FactorFamily(str, Enum):
    """Known factor semantic families."""

    PRICE_VOLUME = "PRICE_VOLUME"
    HIGH_TURNOVER = "HIGH_TURNOVER"
    FUNDAMENTAL = "FUNDAMENTAL"
    SPARSE_UPDATE = "SPARSE_UPDATE"
    EVENT = "EVENT"
    BINARY = "BINARY"
    DISCRETE = "DISCRETE"


# RAW / NO-OP is always a candidate in every search space.
RAW_SEMANTIC_ID = "RAW:noop"


@dataclass(frozen=True)
class FamilyRule:
    """A single family -> legal treatment families rule (versioned)."""

    factor_family: str
    allowed_families: Tuple[TreatmentFamily, ...]
    policy_version: str = "1.0.0"
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "factor_family", str(self.factor_family))
        object.__setattr__(self, "allowed_families", tuple(self.allowed_families))
        object.__setattr__(self, "notes", tuple(self.notes))

    def allows(self, family: TreatmentFamily) -> bool:
        return family in self.allowed_families


class EligibilityRuleRegistry:
    """Versioned catalog of factor-family -> allowed treatment families."""

    def __init__(self):
        self._rules: Dict[str, FamilyRule] = {}
        self._defaults: Dict[str, Tuple[TreatmentFamily, ...]] = {}

    def register(self, rule: FamilyRule) -> None:
        key = rule.factor_family.upper()
        self._rules[key] = rule

    def set_default(self, family: str, allowed: Tuple[TreatmentFamily, ...]) -> None:
        self._defaults[family.upper()] = tuple(allowed)

    def rule_for(self, family: str) -> Optional[FamilyRule]:
        return self._rules.get(family.upper())

    def allowed_families(self, family: str) -> Tuple[TreatmentFamily, ...]:
        rule = self._rules.get(family.upper())
        if rule is not None:
            return rule.allowed_families
        return self._defaults.get(family.upper(), (TreatmentFamily.RANK,))

    def policy_version_for(self, family: str) -> Optional[str]:
        rule = self._rules.get(family.upper())
        return rule.policy_version if rule is not None else None


def create_default_eligibility_rules() -> EligibilityRuleRegistry:
    """Create the canonical family-rule registry (DLIB-FP-023)."""
    reg = EligibilityRuleRegistry()

    # PRICE_VOLUME + HIGH_TURNOVER: RAW/EWMA/KAMA/trailing median/one-sided
    # IIR/robust smoother, focus turnover + IC decay + cost-adjusted.
    reg.register(FamilyRule(
        factor_family=FactorFamily.PRICE_VOLUME.value,
        allowed_families=(
            TreatmentFamily.SMOOTHING,
            TreatmentFamily.WINSOR,
            TreatmentFamily.RANK,
            TreatmentFamily.ZSCORE,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("price_volume: temporal smoothing allowed; focus turnover/IC-decay",),
    ))
    reg.register(FamilyRule(
        factor_family=FactorFamily.HIGH_TURNOVER.value,
        allowed_families=(
            TreatmentFamily.SMOOTHING,
            TreatmentFamily.WINSOR,
            TreatmentFamily.RANK,
            TreatmentFamily.ZSCORE,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("high_turnover: temporal smoothing allowed; focus turnover/IC-decay",),
    ))

    # FUNDAMENTAL / SPARSE_UPDATE: winsor/rank/zscore/neutralization/freshness,
    # NOT aggressive smoothing.
    reg.register(FamilyRule(
        factor_family=FactorFamily.FUNDAMENTAL.value,
        allowed_families=(
            TreatmentFamily.FRESHNESS_FILL,
            TreatmentFamily.WINSOR,
            TreatmentFamily.RANK,
            TreatmentFamily.ZSCORE,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("fundamental: freshness-fill allowed; short temporal smoothing forbidden",),
    ))
    reg.register(FamilyRule(
        factor_family=FactorFamily.SPARSE_UPDATE.value,
        allowed_families=(
            TreatmentFamily.FRESHNESS_FILL,
            TreatmentFamily.WINSOR,
            TreatmentFamily.RANK,
            TreatmentFamily.ZSCORE,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("sparse_update: freshness-fill allowed; short temporal smoothing forbidden",),
    ))

    # EVENT: event_decay / freshness / sparse-update; not long smoothing.
    reg.register(FamilyRule(
        factor_family=FactorFamily.EVENT.value,
        allowed_families=(
            TreatmentFamily.EVENT_DECAY,
            TreatmentFamily.FRESHNESS_FILL,
            TreatmentFamily.WINSOR,
            TreatmentFamily.RANK,
            TreatmentFamily.ZSCORE,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("event: event-decay + freshness allowed; long-window smoothing forbidden",),
    ))

    # BINARY / DISCRETE: NOT zscore / aggressive-winsor / Kalman.
    reg.register(FamilyRule(
        factor_family=FactorFamily.BINARY.value,
        allowed_families=(
            TreatmentFamily.RANK,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("binary: zscore/aggressive-winsor require explicit policy",),
    ))
    reg.register(FamilyRule(
        factor_family=FactorFamily.DISCRETE.value,
        allowed_families=(
            TreatmentFamily.RANK,
            TreatmentFamily.NEUTRALIZATION,
        ),
        policy_version="1.0.0",
        notes=("discrete: zscore/aggressive-winsor require explicit policy",),
    ))

    # Unknown family: conservative default — RAW + rank only.
    reg.set_default("UNKNOWN", (TreatmentFamily.RANK,))
    return reg


_default_eligibility_rules: Optional[EligibilityRuleRegistry] = None


def get_default_eligibility_rules() -> EligibilityRuleRegistry:
    """Get or create the default family-rule registry."""
    global _default_eligibility_rules
    if _default_eligibility_rules is None:
        _default_eligibility_rules = create_default_eligibility_rules()
    return _default_eligibility_rules


__all__ = [
    "TreatmentFamily",
    "FactorFamily",
    "FamilyRule",
    "EligibilityRuleRegistry",
    "create_default_eligibility_rules",
    "get_default_eligibility_rules",
    "RAW_SEMANTIC_ID",
]
