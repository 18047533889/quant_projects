"""
Treatment lineage contracts for the auto-treatment optimizer.

A *treatment lineage* is the ordered list of transforms that have been (or
will be) applied to a factor. Two transforms with the same function name but
different *semantic stages* are DIFFERENT treatments: e.g. a cross-sectional
rank applied before neutralization (``rank@pre_neutralization``) is not the
same treatment as a rank applied after neutralization
(``rank@post_neutralization``). Deduplication must therefore key on the
semantic stage, not the bare function name.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TransformStage(str, Enum):
    """Semantic stage of a transform within a treatment pipeline.

    Two transforms at different stages are semantically distinct even if they
    share a function name.
    """

    PRE_NEUTRALIZATION = "pre_neutralization"
    POST_NEUTRALIZATION = "post_neutralization"
    TEMPORAL = "temporal"
    MISSINGNESS = "missingness"
    OUTLIER = "outlier"
    REPRESENTATION = "representation"
    NEUTRALIZATION = "neutralization"
    SCALING = "scaling"


class TransformSemanticID:
    """
    Canonical semantic identifier of a transform treatment.

    Examples
    --------
    - ``"WINSOR:q01_q99"``
    - ``"CS_RANK:pct"``
    - ``"INDUSTRY_NEUTRAL:SW_L1"``

    The semantic ID is the deduplication key. Two transforms with the same
    semantic ID are considered the same treatment regardless of the underlying
    function name.
    """

    __slots__ = ("value",)

    def __init__(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("TransformSemanticID must be a non-empty string")
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"TransformSemanticID({self.value!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TransformSemanticID):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class TransformStep:
    """A single transform step in a treatment lineage.

    ``semantic_id`` is the deduplication key. ``stage`` is the semantic stage
    at which the transform is applied. ``name`` is the underlying FE DSL
    transform name (informational only; not the dedup key).
    """

    semantic_id: TransformSemanticID
    stage: TransformStage
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.semantic_id, TransformSemanticID):
            object.__setattr__(self, "semantic_id", TransformSemanticID(self.semantic_id))
        if not isinstance(self.stage, TransformStage):
            object.__setattr__(self, "stage", TransformStage(self.stage))
        if not self.name:
            raise ValueError("TransformStep.name cannot be empty")
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class TransformLineage:
    """Ordered list of transform steps applied to a factor."""

    steps: Tuple[TransformStep, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))

    def __iter__(self):
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def semantic_ids(self) -> List[str]:
        """Return the ordered semantic IDs of the lineage."""
        return [step.semantic_id.value for step in self.steps]

    def has_semantic(self, semantic_id: str) -> bool:
        """True if any step carries the given semantic ID."""
        return any(step.semantic_id.value == semantic_id for step in self.steps)

    def has_stage(self, stage: TransformStage) -> bool:
        """True if any step is applied at the given semantic stage."""
        return any(step.stage == stage for step in self.steps)

    def dedupe(self) -> "TransformLineage":
        """Return a lineage with duplicate (semantic_id, stage) pairs removed.

        Deduplication keys on the semantic stage, NOT the bare function name:
        ``rank@pre_neutralization`` and ``rank@post_neutralization`` are
        DIFFERENT treatments and are both kept. First occurrence wins.
        """
        seen: List[Tuple[str, str]] = []
        kept: List[TransformStep] = []
        for step in self.steps:
            key = (step.semantic_id.value, step.stage.value)
            if key not in seen:
                seen.append(key)
                kept.append(step)
        return TransformLineage(tuple(kept))


@dataclass(frozen=True)
class ExistingTreatmentSignature:
    """Summary of treatments already applied to a factor.

    Used by the eligibility engine to avoid duplicating treatments (e.g. do
    not re-neutralize an already-industry-neutral factor).
    """

    winsor: bool = False
    winsor_params: Dict[str, Any] = field(default_factory=dict)
    industry_neutral: bool = False
    industry_schema: Optional[str] = None
    size_neutral: bool = False
    cs_rank: bool = False
    temporal_smoothing: bool = False
    temporal_smoothing_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "winsor_params", dict(self.winsor_params))
        object.__setattr__(
            self, "temporal_smoothing_params", dict(self.temporal_smoothing_params)
        )


# ---------------------------------------------------------------------------
# FE DSL transform name -> semantic ID mapping.
# ---------------------------------------------------------------------------
# Small, explicit mapping table. Unknown names map to None so the optimizer
# NEVER guesses a semantic ID for a transform it does not recognize.
_FE_DSL_SEMANTIC_MAP: Dict[str, str] = {
    # Outlier / winsor
    "cs_winsor": "WINSOR:cs",
    "winsor": "WINSOR:cs",
    # Rank / representation
    "cs_rank": "CS_RANK:pct",
    "rank": "CS_RANK:pct",
    # Z-score / scaling
    "cs_zscore": "ZSCORE:cs",
    "zscore": "ZSCORE:cs",
    "cs_scale": "SCALE:cs",
    "cs_demean": "DEMEAN:cs",
    # Temporal smoothing
    "ewma": "SMOOTH:ewma",
    "rolling_mean": "SMOOTH:trailing_sma",
    "trailing_sma": "SMOOTH:trailing_sma",
    "kama": "SMOOTH:kama",
    "one_sided_iir_lowpass": "SMOOTH:one_sided_iir_lowpass",
    "kalman_local_level": "SMOOTH:kalman_local_level",
    "trailing_median": "SMOOTH:trailing_median",
    # Missingness / freshness
    "forward_fill": "FILL:forward",
    "freshness_aware_fill": "FILL:freshness_aware",
    # Neutralization
    "ols_neutralize": "NEUTRAL:ols",
    "industry_neutral": "INDUSTRY_NEUTRAL:SW_L1",
    "size_neutral": "SIZE_NEUTRAL:log_mktcap",
    "dual_neutral": "DUAL_NEUTRAL:industry_size",
    # Event decay
    "event_decay": "EVENT_DECAY:short_halflife",
}


def map_fe_dsl_to_semantic(name: str) -> Optional[str]:
    """Map a known FE DSL transform name to a semantic ID.

    Returns ``None`` for unknown names so the optimizer never guesses a
    semantic identity for a transform it does not recognize.
    """
    return _FE_DSL_SEMANTIC_MAP.get(name)


__all__ = [
    "TransformStage",
    "TransformSemanticID",
    "TransformStep",
    "TransformLineage",
    "ExistingTreatmentSignature",
    "map_fe_dsl_to_semantic",
]
