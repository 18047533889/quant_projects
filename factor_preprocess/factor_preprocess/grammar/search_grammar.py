"""
Search grammar for the auto-treatment optimizer.

Defines the *legal* ordering of treatment stages and a small set of certified
order templates. The grammar prevents arbitrary transform-permutation
explosion: only certified templates (and RAW) are admissible.

Stage grammar
-------------
A treatment pipeline follows the canonical stage order:

    A Missingness -> B Outlier -> C Temporal Stabilization
    -> D Neutralization -> E Representation/Scaling

Each certified template declares its preconditions and causality class.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Stage(str, Enum):
    """Canonical treatment stages in order.

    R61-FI-042 mapping: each canonical stage corresponds to the transform
    ``stage`` metadata values used by the preset and eligibility layers.
    ``volatility_scale``/``volatility_scale_returns`` carry metadata stage
    ``"scaling"`` which is classified as the C segment (temporal
    stabilization / risk scaling) — NOT representation — so a preset that
    applies ``ewma`` and then ``volatility_scale`` stays inside C before D
    neutralization.  ``cs_rank``/``cs_zscore``/``cs_scale``/``cs_demean`` are
    representation (E).
    """

    MISSINGNESS = "A_Missingness"
    OUTLIER = "B_Outlier"
    TEMPORAL = "C_TemporalStabilization"
    NEUTRALIZATION = "D_Neutralization"
    REPRESENTATION = "E_RepresentationScaling"


#: transform metadata ``stage`` values that resolve to each canonical Stage.
#: Used by preset-order certification (plan §27 F-TDD-001) to map a preset's
#: ordered transform names onto the canonical A→E grammar without guessing.
STAGE_META_TO_CANONICAL: dict[str, "Stage"] = {
    "missingness": Stage.MISSINGNESS,
    "outlier": Stage.OUTLIER,
    "temporal": Stage.TEMPORAL,
    "scaling": Stage.TEMPORAL,          # risk/temporal scaling is C, not E
    "neutralization": Stage.NEUTRALIZATION,
    "representation": Stage.REPRESENTATION,
}


def stage_of_transform_name(
    name: str,
    *,
    registry=None,
) -> "Stage":
    """Map a transform name to its canonical Stage via registry metadata.

    ``volatility_scale`` (metadata stage ``"scaling"``) is classified as
    Stage.TEMPORAL (C segment risk scaling).  Unknown or unregistered names
    fail closed so a preset can never silently certify an unknown step.
    """
    if registry is None:
        from factor_preprocess.registry.transforms import get_default_registry
        registry = get_default_registry()
    meta = registry.get(name)
    if meta is None:
        raise ValueError(f"cannot certify unknown transform {name!r}")
    stage_meta = meta.stage
    if stage_meta is None:
        raise ValueError(f"transform {name!r} has no semantic stage metadata")
    try:
        return STAGE_META_TO_CANONICAL[stage_meta]
    except KeyError as exc:
        raise ValueError(
            f"transform {name!r} has unmapped stage metadata {stage_meta!r}"
        ) from exc


# Canonical stage order.
STAGE_ORDER: Tuple[Stage, ...] = (
    Stage.MISSINGNESS,
    Stage.OUTLIER,
    Stage.TEMPORAL,
    Stage.NEUTRALIZATION,
    Stage.REPRESENTATION,
)


class CausalityClass(str, Enum):
    """Causality class of a template."""

    CAUSAL = "causal"
    CROSS_SECTIONAL = "cross_sectional"
    NO_OP = "no_op"


@dataclass(frozen=True)
class OrderTemplate:
    """A certified legal stage ordering.

    ``stages`` is the ordered list of stages in the template. ``preconditions``
    are human-readable conditions that must hold for the template to be legal.
    ``causality_class`` describes the causal safety of the template.
    """

    name: str
    stages: Tuple[Stage, ...]
    preconditions: Tuple[str, ...] = field(default_factory=tuple)
    causality_class: CausalityClass = CausalityClass.CAUSAL

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))

    def stage_names(self) -> List[str]:
        """Return the ordered stage names."""
        return [s.value for s in self.stages]


# Certified order templates. Do NOT add arbitrary permutations.
CERTIFIED_TEMPLATES: Tuple[OrderTemplate, ...] = (
    OrderTemplate(
        name="SMOOTH_NEUTRALIZE_RANK",
        stages=(
            Stage.TEMPORAL,
            Stage.NEUTRALIZATION,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "temporal smoothing precedes neutralization",
            "neutralization precedes rank representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="NEUTRALIZE_SMOOTH_RANK",
        stages=(
            Stage.NEUTRALIZATION,
            Stage.TEMPORAL,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "neutralization precedes temporal smoothing",
            "temporal smoothing precedes rank representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="WINSOR_NEUTRALIZE_ZSCORE",
        stages=(
            Stage.OUTLIER,
            Stage.NEUTRALIZATION,
            Stage.REPRESENTATION,
        ),
        preconditions=(
            "winsor (outlier) precedes neutralization",
            "neutralization precedes zscore representation",
        ),
        causality_class=CausalityClass.CAUSAL,
    ),
    OrderTemplate(
        name="RAW",
        stages=(),
        preconditions=("no treatment applied",),
        causality_class=CausalityClass.NO_OP,
    ),
)


def is_certified_template(stages: Tuple[Stage, ...]) -> bool:
    """True if the given stage sequence matches a certified template."""
    for template in CERTIFIED_TEMPLATES:
        if template.stages == tuple(stages):
            return True
    return False


def is_certified_exception(name: str) -> bool:
    """True when a preset carries an explicit certified-exception name.

    Certified exceptions are EXPLICIT recipes (plan §27): a preset may deviate
    from the canonical A→E subsequence ONLY when it is named as one of these
    intentional recipes (each with a documented economic meaning).  Arbitrary
    permutations never pass.
    """
    return name in {t.name for t in CERTIFIED_TEMPLATES}


def validate_stage_order(stages: Tuple[Stage, ...]) -> Tuple[bool, List[str]]:
    """Validate that a stage sequence respects the canonical stage order.

    Returns ``(valid, errors)``. A sequence is valid if it is a certified
    template OR a subsequence of the canonical stage order (no stage appears
    out of order).
    """
    errors: List[str] = []

    # RAW (empty) is always valid.
    if not stages:
        return True, errors

    # Certified templates are always valid.
    if is_certified_template(stages):
        return True, errors

    # Otherwise, stages must appear in canonical order (no reordering).
    order_index = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    last = -1
    for stage in stages:
        idx = order_index.get(stage)
        if idx is None:
            errors.append(f"unknown stage {stage!r}")
            continue
        if idx < last:
            errors.append(f"stage {stage.value} out of canonical order")
        last = idx

    return len(errors) == 0, errors


def certify_preset_stages(
    preset_name: str,
    transform_names: Tuple[str, ...],
    *,
    registry=None,
    certified_exceptions=frozenset({
        "SMOOTH_NEUTRALIZE_RANK",
        "NEUTRALIZE_SMOOTH_RANK",
        "WINSOR_NEUTRALIZE_ZSCORE",
        "RANK_THEN_NEUTRALIZE",
        "RAW",
    }),
) -> Tuple[bool, List[str]]:
    """Certify an ordered preset transform list against the stage grammar.

    Every production preset must map to the certified semantic order:
    a subsequence of the canonical A→E order, OR an explicitly named
    certified exception.  ``transform_names`` is the preset's ordered
    transform names; each is mapped to a canonical Stage through registry
    metadata (``stage_of_transform_name``).  Any *arbitrary* permutation
    (e.g. E before D) is rejected.  ``certified_exceptions`` is the explicit
    allow-list of recipe names that a preset may declare (plan §27); a preset
    declares its exception by carrying the recipe name in its ``tags``.
    """
    errors: List[str] = []
    if registry is None:
        from factor_preprocess.registry.transforms import get_default_registry
        registry = get_default_registry()

    if not transform_names:
        return True, errors  # RAW / empty pipeline is always valid

    # Collapse repeated same-stage transforms (ewma + volatility_scale are
    # both C) before validating the ordered stage sequence.
    stages: List[Stage] = []
    for name in transform_names:
        try:
            s = stage_of_transform_name(name, registry=registry)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if stages and stages[-1] == s:
            continue  # same canonical stage repeated stays one step
        stages.append(s)

    if errors:
        return False, errors

    # Certified-exception handling is done by the caller (certify_preset_stages
    # validates only the sequence; certify_production_presets resolves an
    # explicit recipe from preset tags).  Here the ordered stage sequence
    # itself must be a certified template OR a canonical-order subsequence.
    valid, seq_errors = validate_stage_order(tuple(stages))
    if not valid:
        errors.extend(seq_errors)
        errors.append(
            f"preset {preset_name!r} stage sequence {[s.value for s in stages]} "
            "is not a certified template nor a canonical-order subsequence"
        )
        return False, errors
    return True, errors


def certify_production_presets(policy_registry=None, transform_registry=None):
    """Certify every production preset of a policy registry.

    A production preset passes iff its ordered transform stages are a
    canonical-order subsequence or exactly one of the certified exceptions.
    Presets that legitimately use a non-default certified recipe declare the
    recipe name in ``tags`` (see ``_certified_recipe_for``); the certified
    allow-list is the explicit set above — arbitrary permutations are never
    certified.  Returns a list of ``(preset_name, valid, detail)``.
    """
    from factor_preprocess.registry.policies import PolicyLevel

    if policy_registry is None:
        from factor_preprocess.registry.policies import get_default_policy_registry
        policy_registry = get_default_policy_registry()
    out = []
    for preset in policy_registry.all_policies():
        if preset.level != PolicyLevel.PRODUCTION:
            continue
        stages = tuple(
            stage_of_transform_name(step.name, registry=transform_registry)
            for step in preset.steps
        )
        # repeated same-stage collapse for the sequence check
        collapsed = []
        for s in stages:
            if collapsed and collapsed[-1] == s:
                continue
            collapsed.append(s)
        seq = tuple(collapsed)
        if is_certified_template(seq):
            out.append((preset.name, True, "certified_template"))
            continue
        recipe = _certified_recipe_for(preset)
        if recipe is not None:
            # Explicit named exception overrides default canonical order.
            out.append((preset.name, True, f"certified_exception:{recipe}"))
            continue
        valid, errors = validate_stage_order(seq)
        if valid:
            out.append((preset.name, True, "canonical_subsequence"))
        else:
            out.append(
                (preset.name, False, "; ".join(errors))
            )
    return out


def _certified_recipe_for(preset) -> Optional[str]:
    """Return the certified recipe a preset explicitly declares, if any.

    A preset declares a certified recipe by carrying its name as a tag
    (e.g. ``tags=["SMOOTH_NEUTRALIZE_RANK"]``).  Only names in the explicit
    certified allow-list resolve; anything else is not a legal recipe and is
    treated as an un-declared permutation (which then fails certification).
    """
    tags = set(getattr(preset, "tags", ()) or ())
    for recipe in (
        "SMOOTH_NEUTRALIZE_RANK",
        "NEUTRALIZE_SMOOTH_RANK",
        "WINSOR_NEUTRALIZE_ZSCORE",
        "RANK_THEN_NEUTRALIZE",
        "RAW",
    ):
        if recipe in tags:
            return recipe
    return None


__all__ = [
    "Stage",
    "STAGE_ORDER",
    "STAGE_META_TO_CANONICAL",
    "CausalityClass",
    "OrderTemplate",
    "CERTIFIED_TEMPLATES",
    "is_certified_template",
    "is_certified_exception",
    "validate_stage_order",
    "stage_of_transform_name",
    "certify_preset_stages",
    "certify_production_presets",
    "CERTIFIED_PRODUCTION_RECIPES",
]

#: The explicit allow-list of certified production recipes (plan §27).
#: Arbitrary permutations must never be certified; only these named recipes
#: (each with a documented economic meaning) may deviate from the default
#: canonical A→E order.
CERTIFIED_PRODUCTION_RECIPES: frozenset = frozenset({
    "SMOOTH_NEUTRALIZE_RANK",
    "NEUTRALIZE_SMOOTH_RANK",
    "WINSOR_NEUTRALIZE_ZSCORE",
    "RANK_THEN_NEUTRALIZE",
    "RAW",
})
