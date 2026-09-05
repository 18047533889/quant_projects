"""
TreatmentRecipe — canonical prescription for applying a treatment to a factor.

A :class:`TreatmentRecipe` is a deep-immutable, content-addressed artifact that
describes, in *order*, every transform step that converts a raw factor value
into a treated (model-ready) value. Because transform ordering changes the
numerical outcome (``winsor->neutralize->smooth->rank`` is a DIFFERENT treatment
than ``smooth->winsor->neutralize->rank``), the ordered step sequence is a
first-class part of the recipe's identity.

Immutability
------------
The recipe is a ``frozen=True`` dataclass, but ``frozen`` alone does not make
nested containers immutable. ``ordered_steps``, every step's ``parameters``,
``existing_treatment_signature`` and ``neutralization_spec`` are all *deeply*
frozen via :func:`deep_freeze` so the object is safe to hash and use as an
identity key (DLIB-FP-003, DLIB-FP-026).

Content identity
----------------
``content_hash`` is derived-only. A caller-supplied value is validated against
the recomputed digest and rejected on mismatch (fail closed). Two recipes with
the same ordered step sequence (semantic id + params + stage) but different
ordering, or the same recipe on two different factor values, must hash to
different identities.

SPEC vs MATERIALIZATION identity (P0-FP #103)
---------------------------------------------
``content_hash`` / ``treatment_identity`` / ``spec_identity`` describe the
treatment *specification* ONLY — the transform chain, ordered parameters and
neutralization spec identity. The materialization context (data snapshot /
universe / split / asof PIT anchor / price basis / unit) is NEVER part of the
recipe content: an identical recipe materialized against two different data
contexts shares the same spec identity but MUST produce different
:class:`~factor_preprocess.contracts.treatment_spec.
TreatmentMaterializationIdentity` values (use ``recipe.materialize(...)``).
The two identity classes are deliberately different types so a spec identity
can never be confused with a materialization identity.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from collections.abc import Mapping
from enum import Enum
import hashlib

from factor_preprocess.errors import InvalidContractError, TreatmentMaterializationError
from factor_preprocess.contracts._deep_freeze import deep_freeze


class FitBoundary(str, Enum):
    """Where the fitted statistics may be drawn from."""

    EXPANDING = "expanding"
    ROLLING = "rolling"
    FULL_SAMPLE_RESEARCH = "full_sample_research"  # never auto-admits to production


class RecipeSchemaVersion(str, Enum):
    """Schema version of the recipe artifact."""

    V1 = "1.0.0"


def _stable_repr(value: Any) -> str:
    """Deterministic canonical string form for content hashing (fail-closed)."""
    # Frozen params are dict-like (FrozenDict / Mapping) — normalize them.
    if isinstance(value, Mapping):
        value = dict(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{_stable_repr(k)}:{_stable_repr(v)}"
            for k, v in sorted(value.items(), key=lambda kv: _stable_repr(kv[0]))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(item) for item in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "<" + ",".join(sorted(_stable_repr(item) for item in value)) + ">"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    # Enums compare equal to their .value by identity; hash via value.
    if isinstance(value, Enum):
        return f"Enum:{type(value).__name__}:{_stable_repr(value.value)}"
    # NeutralizationSpec participates in the recipe content hash (its semantic
    # surface is part of the recipe identity). Normalize via its own
    # content-stable form so the hash is cross-process deterministic.
    if type(value).__name__ == "NeutralizationSpec" and type(value).__module__.endswith(
        "neutralization.spec"
    ):
        from factor_preprocess.contracts.treatment_spec import neutralization_spec_identity
        return f"NeutralizationSpec{{identity:{neutralization_spec_identity(value)}}}"
    raise TypeError(
        "_stable_repr does not support type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


@dataclass(frozen=True)
class RecipeStep:
    """A single ordered transform step within a :class:`TreatmentRecipe`."""

    step_id: str
    semantic_transform_id: str
    implementation_ref: str
    stage: str
    requires_fit: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.step_id:
            raise InvalidContractError("RecipeStep.step_id cannot be empty")
        if not self.semantic_transform_id:
            raise InvalidContractError("RecipeStep.semantic_transform_id cannot be empty")
        if not self.implementation_ref:
            raise InvalidContractError("RecipeStep.implementation_ref cannot be empty")
        if not self.stage:
            raise InvalidContractError("RecipeStep.stage cannot be empty")
        object.__setattr__(self, "parameters", deep_freeze(self.parameters))


@dataclass(frozen=True)
class TreatmentRecipe:
    """Deep-immutable, content-addressed prescription for a treatment."""

    recipe_id: str
    source_factor_definition_ref: str
    source_factor_value_ref: str
    ordered_steps: tuple

    # Optional context that still participates in content identity.
    existing_treatment_signature: Optional[Dict[str, Any]] = None
    neutralization_spec: Optional[Dict[str, Any]] = None
    fit_boundary: FitBoundary = FitBoundary.EXPANDING
    registry_snapshot_identity: Optional[str] = None
    policy_identity: Optional[str] = None
    causality_certificate_ref: Optional[str] = None
    schema_version: RecipeSchemaVersion = RecipeSchemaVersion.V1
    content_hash: str = ""

    def __post_init__(self):
        if not self.recipe_id:
            raise InvalidContractError("TreatmentRecipe.recipe_id cannot be empty")
        if not self.source_factor_definition_ref:
            raise InvalidContractError(
                "TreatmentRecipe.source_factor_definition_ref cannot be empty"
            )
        if not self.source_factor_value_ref:
            raise InvalidContractError(
                "TreatmentRecipe.source_factor_value_ref cannot be empty"
            )
        if not isinstance(self.ordered_steps, (list, tuple)) or not self.ordered_steps:
            raise InvalidContractError("TreatmentRecipe.ordered_steps cannot be empty")
        for step in self.ordered_steps:
            if not isinstance(step, RecipeStep):
                raise InvalidContractError(
                    "TreatmentRecipe.ordered_steps entries must be RecipeStep"
                )
        # step_id uniqueness (DLIB-FP-003).
        ids = [step.step_id for step in self.ordered_steps]
        if len(ids) != len(set(ids)):
            raise InvalidContractError("TreatmentRecipe step_id must be unique")

        # Deep-freeze everything so the object is hash-safe.
        object.__setattr__(
            self, "ordered_steps", tuple(self.ordered_steps)
        )

        # R61-FI-043 duplicate-guard (plan §26 F4): a recipe whose ordered step
        # lineage is redundant (rank(rank(x)), EWMA twice, repeated
        # neutralization, ...) and cannot be losslessly canonicalized FAILS
        # CLOSED at construction. The caller must canonicalize the lineage
        # first (e.g. via ``canonicalize_lineage``); a silently redundant
        # recipe must never exist as a content-addressed artifact.
        self._validate_lineage_guard()

        object.__setattr__(
            self, "existing_treatment_signature",
            deep_freeze(self.existing_treatment_signature),
        )
        object.__setattr__(
            self, "neutralization_spec", deep_freeze(self.neutralization_spec)
        )
        if not isinstance(self.fit_boundary, FitBoundary):
            object.__setattr__(self, "fit_boundary", FitBoundary(self.fit_boundary))
        if not isinstance(self.schema_version, RecipeSchemaVersion):
            object.__setattr__(
                self, "schema_version", RecipeSchemaVersion(self.schema_version)
            )

        actual_hash = self._derive_content_hash()
        if self.content_hash and self.content_hash != actual_hash:
            raise InvalidContractError(
                "TreatmentRecipe.content_hash does not match recipe content "
                "(fail closed on caller-supplied mismatch)"
            )
        object.__setattr__(self, "content_hash", actual_hash)
        # Materialization context (data snapshot / universe / split / asof /
        # price basis) is NEVER part of the recipe content hash: the recipe is
        # the SPEC, and the spec identity must be data-independent (P0-FP #103).

    # -- identity ----------------------------------------------------------
    def _derive_content_hash(self) -> str:
        """Content-derived identity over the ordered step sequence and context.

        The ordered ``semantic_transform_id`` + ``stage`` + ``parameters`` of
        each step feed the digest in sequence, so a reordering of the steps
        changes the identity (DLIB-FP-017).
        """
        step_components = []
        for step in self.ordered_steps:
            step_components.append(
                {
                    "step_id": step.step_id,
                    "semantic": step.semantic_transform_id,
                    "stage": step.stage,
                    "requires_fit": step.requires_fit,
                    "params": step.parameters,
                    "impl": step.implementation_ref,
                }
            )
        components = {
            "recipe_id": self.recipe_id,
            "source_factor_definition_ref": self.source_factor_definition_ref,
            "source_factor_value_ref": self.source_factor_value_ref,
            "ordered_steps": step_components,
            "existing_treatment_signature": self.existing_treatment_signature,
            "neutralization_spec": self.neutralization_spec,
            "fit_boundary": self.fit_boundary.value,
            "registry_snapshot_identity": self.registry_snapshot_identity,
            "policy_identity": self.policy_identity,
            "causality_certificate_ref": self.causality_certificate_ref,
            "schema_version": self.schema_version.value,
        }
        return hashlib.sha256(_stable_repr(components).encode("utf-8")).hexdigest()

    @property
    def treatment_identity(self) -> str:
        """Content-derived identity over the ordered steps + context.

        Alias of ``content_hash`` provided so the ordered step sequence is
        unambiguously part of the treatment identity (DLIB-FP-017).
        """
        return self.content_hash

    @property
    def spec_identity(self) -> "TreatmentSpecIdentity":
        """SPEC identity of this recipe (definition-only, data-independent).

        See :class:`factor_preprocess.contracts.treatment_spec.
        TreatmentSpecIdentity`. Derived from the treatment definition ONLY
        (ordered transform names + params + neutralization spec identity); the
        materialization context (data snapshot / universe / split / asof /
        price basis) is deliberately excluded (P0-FP #103). The identity is
        derived and cannot be caller-supplied.
        """
        from factor_preprocess.contracts.treatment_spec import (
            TreatmentSpecIdentity,
            neutralization_spec_identity,
        )
        steps = []
        for step in self.ordered_steps:
            steps.append(
                (step.step_id, step.semantic_transform_id, step.stage, step.parameters)
            )
        return TreatmentSpecIdentity(
            spec_id=self.recipe_id,
            recipe_ref=self.content_hash,
            semantic_transform_ids=tuple(steps),
            neutralization_spec_identity=neutralization_spec_identity(
                self.neutralization_spec
            ),
            fit_boundary=self.fit_boundary,
        )

    @property
    def is_full_sample_research(self) -> bool:
        """True if this recipe fits on the full sample (research-only).

        A full-sample-research recipe MUST NOT be materialized against an
        evaluation-valid split (validation / test / production). The runtime
        rejection is enforced by :func:`ensure_materialization_split_valid`
        (P0-FP #103).
        """
        return self.fit_boundary == FitBoundary.FULL_SAMPLE_RESEARCH

    def materialize(
        self,
        *,
        materialization_ref: str,
        data_snapshot_ref: str,
        universe_ref: str,
        split_ref: str,
        split,
        neutralization_spec=None,
        asof_timestamp=None,
        price_basis=None,
        industry_schema=None,
        size_definition=None,
        pit_identity=None,
        unit=None,
        require_split_valid: bool = True,
    ) -> "TreatmentMaterializationIdentity":
        """Materialize this recipe against a concrete context.

        Runs the fail-closed split check (a FULL_SAMPLE_RESEARCH recipe is
        rejected on evaluation-valid splits) and derives the
        :class:`TreatmentMaterializationIdentity` from the spec identity PLUS
        the materialization context, including the A股 neutralization binding.
        """
        from factor_preprocess.contracts.treatment_spec import (
            materialize_identity,
        )
        return materialize_identity(
            self,
            materialization_ref=materialization_ref,
            data_snapshot_ref=data_snapshot_ref,
            universe_ref=universe_ref,
            split_ref=split_ref,
            split=split,
            neutralization_spec=neutralization_spec,
            asof_timestamp=asof_timestamp,
            price_basis=price_basis,
            industry_schema=industry_schema,
            size_definition=size_definition,
            pit_identity=pit_identity,
            unit=unit,
            require_split_valid=require_split_valid,
        )

    def step_semantic_ids(self) -> tuple:
        """Ordered semantic transform ids of the recipe steps."""
        return tuple(step.semantic_transform_id for step in self.ordered_steps)

    def _validate_lineage_guard(self) -> None:
        """R61-FI-043 fail-closed duplicate-guard over the ordered steps.

        Redundant / non-canonicalizable chains (e.g. two sequential
        neutralizations, EWMA twice, ``rank(rank(x))`` as separate steps)
        must never materialize as a content-addressed recipe. Losslessly
        collapsible pairs (``rank(rank(x))``, industry+size superset) are
        also rejected here — the caller must canonicalize BEFORE constructing
        the recipe, so the recipe artifact always holds the canonical form.
        """
        from factor_preprocess.contracts.lineage_policy import (
            canonicalize_lineage,
            RedundancyClass,
        )
        from factor_preprocess.contracts.treatment_lineage import (
            TransformLineage,
            TransformStep as LStep,
            TransformSemanticID,
        )
        steps = []
        for step in self.ordered_steps:
            steps.append(
                LStep(
                    semantic_id=TransformSemanticID(step.semantic_transform_id),
                    stage=step.stage,
                    name=step.implementation_ref,
                    parameters=dict(step.parameters),
                )
            )
        decision = canonicalize_lineage(TransformLineage(tuple(steps)))
        if decision.redundancy_class is RedundancyClass.REJECTED:
            raise InvalidContractError(
                f"TreatmentRecipe {self.recipe_id!r} violates the lineage "
                f"duplicate-guard: redundant steps are not losslessly "
                f"collapsible; canonicalize the lineage before constructing "
                f"the recipe"
            )
        if decision.was_collapsed:
            # A foldable chain (rank(rank(x)), industry->dual, ...) must be
            # canonicalized by the caller, not stored in pre-folded form.
            reasons = "; ".join(r[2] for r in decision.rejections)
            raise InvalidContractError(
                f"TreatmentRecipe {self.recipe_id!r} carries a redundant "
                f"chain that must be canonicalized first: {reasons}"
            )


__all__ = ["TreatmentRecipe", "RecipeStep", "FitBoundary", "RecipeSchemaVersion"]
