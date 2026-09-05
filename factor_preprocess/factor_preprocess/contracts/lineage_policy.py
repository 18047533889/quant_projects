"""
Lineage deduplication & canonicalization policy (R61-FI-043, plan §26 F4).

The duplicate guard makes an ordered transform lineage safe to execute:

- **Redundant operations** — applying the same univariate transform a second
  time — are **canonicalized** when the fold is lossless and **rejected**
  otherwise:

  - ``rank(rank(x))``  -> canonicalized to a single ``CS_RANK:pct`` step
    (rank of rank is exactly rank — lossless fold).
  - ``zscore(zscore(x))`` -> canonicalized to a single z-score step
    (cross-sectional z-score is a fixed point over finite values — the second
    application is a numerical no-op, so the fold is lossless).
  - ``winsor(winsor(x))`` with equal bounds -> canonicalized to a single
    winsor step (clipping is idempotent).
  - ``identical EWMA twice`` -> **rejected**. A cascade of two first-order
    EWMA low-pass filters is a second-order filter that NO single EWMA (of any
    halflife) equals, so no lossless fold exists.
  - ``size-neutral(size-neutral(x))`` and any repeated neutralization ->
    **rejected** (re-residualization is not provably a no-op under NaN / PIT
    exposure sets / finite-sample regressions, and is never expressible as one
    canonical step).

- **Legitimate superset treatments** — e.g. ``industry neutral`` followed by
  ``size neutral`` — are **collapsed into a single canonical superset step**
  (``DUAL_NEUTRAL:industry_size``) instead of being applied twice in
  sequence (DLIB-FP-DUP-003). The optimizer must never emit two sequential
  univariate neutralizations when the canonical dual form exists.

- **Idempotent table versioning** — the policy table
  (:data:`IDEMPOTENT_SEMANTIC_CLASSES`) carries a version
  (:data:`LINEAGE_POLICY_VERSION`); :func:`canonicalize_lineage` refuses to
  canonicalize under an unknown table version (fail closed).

The guard is deliberately conservative:

- All comparisons are over the *semantic id* (never the function name), so
  ``cs_rank`` and ``rank`` both map to ``CS_RANK:pct``.
- Redundant folds/rejections fire only on *adjacent* steps with no intervening
  transform. ``rank@pre_neutralization`` then ``rank@post_neutralization``
  stay distinct (existing lineage semantics).
- When a fold cannot be proven lossless the guard rejects instead of guessing
  (future-leakage / silent-signal-destruction safety: 宁严勿松).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from factor_preprocess.contracts.treatment_lineage import (
    TransformSemanticID,
    TransformStep,
    TransformLineage,
)
from factor_preprocess.errors import ContractError


# Versioned policy table (R61-FI-043 DLIB-FP-DUP-004).
#
# Bump on any semantic change to the classification tables below.
# ``canonicalize_lineage`` keys canonicalization on this exact version, so a
# lineage declared under an old table is never silently re-canonicalized under
# a new one — a version mismatch fails closed.
LINEAGE_POLICY_VERSION = "2026-09-05.1"

# Semantic ids that are redundant when applied a second time back-to-back on
# the same axis. These participate in the versioned policy table.
IDEMPOTENT_SEMANTIC_CLASSES: Tuple[str, ...] = (
    "CS_RANK:pct",
    "CROSS_SECTIONAL_ZSCORE:cs",
    "ZSCORE:cs",
    "WINSOR:cs",
)

# Losslessly-collapsible repeated cross-sectional transforms: the second
# application is an exact or numerical no-op, so the pair folds to ONE step.
_CROSS_IDEMPOTENT_FOLDABLE: Tuple[str, ...] = (
    "CS_RANK:pct",
    "CROSS_SECTIONAL_ZSCORE:cs",
    "ZSCORE:cs",
    "WINSOR:cs",
)

# Temporal smoothing semantic families: applying two members of the SAME
# family in sequence (including an identical EWMA twice) is a cascade that no
# single step of that family reproduces -> always rejected, never folded.
TEMPORAL_FAMILY_OF: Dict[str, str] = {
    "SMOOTH:ewma": "SMOOTH",
    "SMOOTH:trailing_sma": "SMOOTH",
    "SMOOTH:trailing_median": "SMOOTH",
    "SMOOTH:kama": "SMOOTH",
    "SMOOTH:one_sided_iir_lowpass": "SMOOTH",
    "SMOOTH:kalman_local_level": "SMOOTH",
    "SMOOTH:robust_ewma": "SMOOTH",
    "EVENT_DECAY:short_halflife": "SMOOTH",
}

# Neutralization topology. Each single-step canonical neutralization declares
# the exposure axes it neutralizes. The ONLY canonical superset of the
# univariate industry/size neutralizations is the explicit dual form.
NEUTRALIZATION_TOPOLOGY: Dict[str, Tuple[str, ...]] = {
    "DUAL_NEUTRAL:industry_size": ("industry", "size"),
    "INDUSTRY_NEUTRAL:SW_L1": ("industry",),
    "SIZE_NEUTRAL:log_mktcap": ("size",),
    "NEUTRAL:ols": (),
}

# Semantic families that re-standardize the same cross-sectional axis. A rank
# then a z-score (or z-score then rank) over the same axis is a redundant
# double standardization and is rejected.
_CS_AXIS_FAMILY: Dict[str, str] = {
    "CS_RANK:pct": "rank",
    "CROSS_SECTIONAL_ZSCORE:cs": "zscore",
    "ZSCORE:cs": "zscore",
}

CANONICAL_DUAL_SID = "DUAL_NEUTRAL:industry_size"


class RedundancyClass(str, Enum):
    """Outcome classification of the guard (audit / tests)."""

    OK = "ok"
    COLLAPSED = "collapsed"
    REJECTED = "rejected"


class LineagePolicyError(ContractError):
    """A lineage violates the dedup / canonicalization policy."""


class RedundantTransformError(LineagePolicyError):
    """A redundant operation has no lossless fold and is rejected."""


class UnsupportedPolicyVersionError(LineagePolicyError):
    """The lineage was declared under an unsupported policy table version."""


@dataclass(frozen=True)
class LineagePolicyDecision:
    """Result of canonicalizing / validating a lineage under the policy."""

    lineage: TransformLineage
    redundancy_class: RedundancyClass
    rejections: Tuple[Tuple[str, str, str], ...] = ()
    policy_version: str = LINEAGE_POLICY_VERSION

    @property
    def is_valid(self) -> bool:
        """True when the lineage passed the guard (collapsed counts as valid)."""
        return self.redundancy_class is not RedundancyClass.REJECTED

    @property
    def was_collapsed(self) -> bool:
        """True when a redundant / superset chain was folded to fewer steps."""
        return self.redundancy_class is RedundancyClass.COLLAPSED

    @property
    def canonical_steps(self):
        """Ordered canonical steps after folds/rejections (audit convenience)."""
        return self.lineage.steps


def _sid(step: TransformStep) -> str:
    value = step.semantic_id
    return value.value if hasattr(value, "value") else str(value)


def _stage_eq(a: TransformStep, b: TransformStep) -> bool:
    return a.stage == b.stage


def _is_ewma(step: TransformStep) -> bool:
    return _sid(step).startswith("SMOOTH:ewma")


def _is_neutralization(step: TransformStep) -> bool:
    return _sid(step) in NEUTRALIZATION_TOPOLOGY or _sid(step).startswith("NEUTRAL:")


def _neutralization_axes(sid: str) -> Tuple[str, ...]:
    return NEUTRALIZATION_TOPOLOGY.get(sid, ())


def _new_step(step: TransformStep, sid: Optional[str] = None) -> TransformStep:
    """Build a canonical TransformStep from an existing one."""
    return TransformStep(
        semantic_id=TransformSemanticID(sid if sid is not None else _sid(step)),
        stage=step.stage,
        name=step.name,
        parameters=dict(step.parameters),
    )


def _emit_rejection(
    rejections: List[Tuple[str, str, str]],
    step: TransformStep,
    nxt: TransformStep,
    reason: str,
) -> None:
    rejections.append((_sid(step), _sid(nxt), reason))


def canonicalize_lineage(
    lineage,
    *,
    policy_version: str = LINEAGE_POLICY_VERSION,
) -> LineagePolicyDecision:
    """Validate / canonicalize a lineage under the duplicate-guard policy.

    See the module docstring for the exact fold vs. reject semantics.
    ``policy_version`` must equal :data:`LINEAGE_POLICY_VERSION`; anything else
    raises :class:`UnsupportedPolicyVersionError` (fail closed).
    """
    if policy_version != LINEAGE_POLICY_VERSION:
        raise UnsupportedPolicyVersionError(
            f"unsupported lineage policy version {policy_version!r}; "
            f"supported: {LINEAGE_POLICY_VERSION!r}"
        )
    if not isinstance(lineage, TransformLineage):
        lineage = TransformLineage(tuple(lineage))

    steps: List[TransformStep] = list(lineage.steps)
    kept: List[TransformStep] = []
    rejections: List[Tuple[str, str, str]] = []

    i = 0
    while i < len(steps):
        step = steps[i]
        nxt = steps[i + 1] if i + 1 < len(steps) else None

        if nxt is None:
            kept.append(step)
            i += 1
            continue

        a_sid = _sid(step)
        b_sid = _sid(nxt)

        # ---- 1. lossless repeated cross-sectional folds ---------------------
        if a_sid == b_sid and a_sid in _CROSS_IDEMPOTENT_FOLDABLE and _stage_eq(step, nxt):
            if a_sid.startswith("CS_RANK:pct"):
                # rank(rank(x)) == rank(x): keep the FIRST occurrence.
                kept.append(step)
            elif a_sid.startswith("WINSOR:cs"):
                # winsor(winsor(x)) == winsor(x) (clipping is idempotent).
                kept.append(step)
            else:
                # zscore(zscore(x)) is a fixed point; keep the FIRST z-score
                # (the second application is a no-op).
                kept.append(step)
            _emit_rejection(
                rejections, step, nxt,
                f"redundant {a_sid} collapsed to a single step (lossless fold)",
            )
            i += 2
            continue

        # ---- 3. legitimate superset collapse --------------------------------
        # industry neutral -> size neutral | dual neutral  (either order) folds
        # into ONE canonical DUAL_NEUTRAL step, never two sequential passes.
        axes_a = set(_neutralization_axes(a_sid))
        axes_b = set(_neutralization_axes(b_sid))
        if axes_a and axes_b and axes_a.isdisjoint(axes_b):
            union_axes = axes_a | axes_b
            if union_axes == {"industry", "size"}:
                canonical = TransformStep(
                    semantic_id=TransformSemanticID(CANONICAL_DUAL_SID),
                    stage=nxt.stage,
                    name="dual_neutral",
                    parameters=dict(nxt.parameters),
                )
                kept.append(canonical)
                _emit_rejection(
                    rejections, step, nxt,
                    f"{a_sid} then {b_sid} collapsed into one canonical "
                    f"{CANONICAL_DUAL_SID} step",
                )
                i += 2
                continue

        # ---- 2. rejected redundant operations --------------------------------
        # 2a. identical EWMA twice / same temporal-family cascade.
        fam_a = TEMPORAL_FAMILY_OF.get(a_sid)
        fam_b = TEMPORAL_FAMILY_OF.get(b_sid)
        if fam_a is not None and fam_a == fam_b:
            raise RedundantTransformError(
                "redundant temporal smoothing: "
                f"{a_sid} then {b_sid} is a cascade that cannot be losslessly "
                "folded into a single step of the same family "
                "(DLIB-FP-DUP-001)"
            )

        # 2b. double standardization on the same axis (rank then zscore or
        #     zscore then rank).
        if _stage_eq(step, nxt):
            cs_a = _CS_AXIS_FAMILY.get(a_sid)
            cs_b = _CS_AXIS_FAMILY.get(b_sid)
            if cs_a is not None and cs_b is not None and cs_a != cs_b:
                raise RedundantTransformError(
                    "redundant double standardization on the same axis: "
                    f"{a_sid} then {b_sid} "
                    "(DLIB-FP-DUP-001)"
                )

        # 2c. repeated neutralization (covers size-neutral(size-neutral(x))).
        if _is_neutralization(step) and _is_neutralization(nxt):
            raise RedundantTransformError(
                "repeated neutralization in sequence is forbidden: "
                f"{a_sid} then {b_sid}; declare a single canonical "
                f"neutralization step (e.g. {CANONICAL_DUAL_SID}) "
                "(DLIB-FP-DUP-003)"
            )

        kept.append(step)
        i += 1

    out_lineage = TransformLineage(tuple(kept))
    redundancy_class = (
        RedundancyClass.COLLAPSED if rejections else RedundancyClass.OK
    )
    return LineagePolicyDecision(
        lineage=out_lineage,
        redundancy_class=redundancy_class,
        rejections=tuple(rejections),
        policy_version=policy_version,
    )


def is_losslessly_collapsible(step_a, step_b) -> Tuple[bool, Optional[Dict[str, object]]]:
    """Pure predicate: can ``step_a`` then ``step_b`` fold to ONE step?

    Returns ``(True, folded_params)`` only for lossless folds. Search spaces
    can use this to avoid proposing redundant chains in the first place.
    """
    a_sid = _sid(step_a)
    b_sid = _sid(step_b)

    if a_sid == b_sid and a_sid in _CROSS_IDEMPOTENT_FOLDABLE:
        return True, dict(step_a.parameters)

    # Superset collapse to dual neutralization.
    axes_a = set(_neutralization_axes(a_sid))
    axes_b = set(_neutralization_axes(b_sid))
    if axes_a and axes_b and axes_a.isdisjoint(axes_b):
        if axes_a | axes_b == {"industry", "size"}:
            return True, {"canonical_sid": CANONICAL_DUAL_SID}

    return False, None


__all__ = [
    "LINEAGE_POLICY_VERSION",
    "IDEMPOTENT_SEMANTIC_CLASSES",
    "NEUTRALIZATION_TOPOLOGY",
    "CANONICAL_DUAL_SID",
    "RedundancyClass",
    "LineagePolicyError",
    "RedundantTransformError",
    "UnsupportedPolicyVersionError",
    "LineagePolicyDecision",
    "canonicalize_lineage",
    "is_losslessly_collapsible",
]
