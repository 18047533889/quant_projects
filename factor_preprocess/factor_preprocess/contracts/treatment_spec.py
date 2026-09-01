"""
Treatment identity split — SPEC vs MATERIALIZATION (P0-FP #103).

The audit splits the treatment identity into two *typed* classes so a spec
identity can never be confused with a materialization identity:

- :class:`TreatmentSpecIdentity` — derived from the treatment definition ONLY
  (recipe semantic hash: transform names, ordered params, neutralization spec
  identity). Identical spec -> identical spec identity regardless of data.
- :class:`TreatmentMaterializationIdentity` — derived from the spec identity
  PLUS the materialization context (``data_snapshot_ref`` / ``universe_ref`` /
  ``split_ref`` / ``asof_timestamp`` / ``price_basis``, and the A股
  neutralization binding ``industry_schema`` / ``size_definition`` /
  ``pit_identity`` / ``unit``). It records WHAT was actually materialized, so
  two materializations of the SAME spec against DIFFERENT contexts (or with
  different neutralization bindings) produce different materialization
  identities.

Both identities are deep-immutable, content-addressed and hash-safe
(DLIB-FP-026). A caller-supplied ``identity`` is validated against the
recomputed digest and rejected on mismatch (fail closed).

``price_basis``
---------------
The A股 (A-share) price-basis convention is carried explicitly: treated values
are defined on a specific price basis. Supported values: ``"adj"`` (后复权
adjusted) and ``"unadj"`` (未复权 raw). ``"adj"`` is the default for treated
model-input values (vwap-to-vwap 后复权 basis used by the platform).

``unit``
--------
The unit/basis of the materialized values is explicit and carried: ``"decimal"``
(1.0 == 100% == 1.0) vs ``"basis_point"`` (1.0 == 0.01%). Factor values that
land on a ``basis_point`` scale MUST be converted before cross-sectional
treatment; the unit is part of the materialization identity so two
materializations differing only in unit are distinct.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from collections.abc import Mapping
from enum import Enum
import hashlib

from factor_preprocess.errors import InvalidContractError, TreatmentMaterializationError
from factor_preprocess.contracts._deep_freeze import deep_freeze
from factor_preprocess.contracts.treatment_recipe import (
    TreatmentRecipe,
    FitBoundary,
)
from factor_preprocess.neutralization.spec import (
    NeutralizationSpec,
    PitIdentity,
    NeutralizationMethod,
    ExposureSet,
)
from factor_preprocess.contracts.treatment_lineage import TransformStage


class PriceBasis(str, Enum):
    """Price basis the treated values are defined on (A股 convention)."""

    ADJ = "adj"        # 后复权 adjusted price basis (vwap-to-vwap 后复权)
    UNADJ = "unadj"    # 未复权 raw price basis


class ValueUnit(str, Enum):
    """Unit/basis of the materialized values (A股 convention)."""

    DECIMAL = "decimal"            # 1.0 == 100% (default for treated values)
    BASIS_POINT = "basis_point"    # 1.0 == 0.01% (return-scale raw values)


class MaterializationSplit(str, Enum):
    """Which split a materialization is valid for (PIT contract).

    ``FULL_SAMPLE_RESEARCH`` is the only split that may consume
    full-sample-research treatments; evaluation-valid splits (VALIDATION /
    TEST / PRODUCTION) must reject them at runtime.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"
    FULL_SAMPLE_RESEARCH = "full_sample_research"

    @property
    def is_evaluation_valid(self) -> bool:
        """True for splits that feed evaluation/production decisions.

        These splits MUST fail closed against full-sample-research treatments.
        """
        return self in (
            MaterializationSplit.VALIDATION,
            MaterializationSplit.TEST,
            MaterializationSplit.PRODUCTION,
        )


def _stable_repr(value: Any) -> str:
    """Deterministic canonical string form for content hashing (fail-closed)."""
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
    if isinstance(value, Enum):
        return f"Enum:{type(value).__name__}:{_stable_repr(value.value)}"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, NeutralizationSpec):
        # Content-stable identity of a neutralization spec: the full semantic
        # recipe surface (method, exposure set, schema, size, standardization,
        # weights, min_obs, condition policy, PIT identity).
        return (
            "NeutralizationSpec{"
            f"method:{_stable_repr(value.method)}:"
            f"exposure_set:{_stable_repr(value.exposure_set)}:"
            f"industry_schema:{_stable_repr(value.industry_schema)}:"
            f"size_definition:{_stable_repr(value.size_definition)}:"
            f"standardization:{_stable_repr(value.standardization)}:"
            f"weights:{_stable_repr(value.weights)}:"
            f"min_obs:{_stable_repr(value.min_obs)}:"
            f"condition_number_policy:{_stable_repr(value.condition_number_policy)}:"
            f"pit_identity:{_stable_repr(value.pit_identity)}"
            "}"
        )
    raise TypeError(
        "_stable_repr does not support type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _content_hash(value: Any) -> str:
    """SHA-256 hex digest derived from the actual content of ``value``."""
    return hashlib.sha256(_stable_repr(value).encode("utf-8")).hexdigest()


def _enum_or(value, enum_cls, field_name):
    """Normalize a raw string to ``enum_cls`` (fail closed on unknown)."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        return enum_cls(value)
    raise InvalidContractError(
        f"{field_name} must be a {enum_cls.__name__} or its value string, got "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def neutralization_spec_identity(spec: Optional[NeutralizationSpec]) -> Optional[str]:
    """Content identity of a :class:`NeutralizationSpec`.

    The identity covers the full *semantic recipe* surface: method,
    exposure_set, industry_schema, size_definition, standardization, weights,
    min_obs, condition_number_policy and pit_identity. Two specs differing in
    any of these fields produce different identities, so a neutralization
    binding change (e.g. SW_L1 -> SW_L2, or size -> beta) changes the spec
    identity — and therefore the materialization identity (P0-FP #103).
    """
    if spec is None:
        return None
    if not isinstance(spec, NeutralizationSpec):
        raise InvalidContractError(
            "neutralization spec must be a NeutralizationSpec"
        )
    return _content_hash(spec)


# A股 (A-share) neutralization binding convention. The neutralization binding
# must NAME the A股 conventions (sw_l1 industry schema + size proxy) so the
# materialization identity carries them explicitly (P0-FP #103 contract).
ASHARE_INDUSTRY_SCHEMA = "SW_L1"
ASHARE_SIZE_DEFINITION = "log_mktcap"


@dataclass(frozen=True)
class TreatmentSpecIdentity:
    """Content identity of a treatment *definition* (SPEC).

    Derived from the treatment definition ONLY — the recipe semantic hash
    (ordered transform names + stage + ordered params) and the neutralization
    spec identity. Identical spec -> identical spec identity regardless of the
    data it is later materialized against.

    This class deliberately carries NO materialization context (no data
    snapshot / universe / split / asof anchor). The materialization context
    lives in :class:`TreatmentMaterializationIdentity` and never shares a type
    with this class.
    """

    spec_id: str
    recipe_ref: str
    semantic_transform_ids: tuple = ()
    neutralization_spec_identity: Optional[str] = None
    fit_boundary: FitBoundary = FitBoundary.EXPANDING
    identity: str = ""

    def __post_init__(self):
        if not self.spec_id:
            raise InvalidContractError("TreatmentSpecIdentity.spec_id cannot be empty")
        if not self.recipe_ref:
            raise InvalidContractError("TreatmentSpecIdentity.recipe_ref cannot be empty")
        if not isinstance(self.fit_boundary, FitBoundary):
            object.__setattr__(self, "fit_boundary", FitBoundary(self.fit_boundary))
        object.__setattr__(
            self, "semantic_transform_ids", tuple(self.semantic_transform_ids)
        )

        actual = self._derive_identity()
        if self.identity and self.identity != actual:
            raise InvalidContractError(
                "TreatmentSpecIdentity.identity does not match derived spec "
                "content (fail closed on caller-supplied mismatch)"
            )
        object.__setattr__(self, "identity", actual)

    def _derive_identity(self) -> str:
        """Content identity over the SPEC surface ONLY (no data context)."""
        components = {
            "kind": "treatment_spec",
            "spec_id": self.spec_id,
            "recipe_ref": self.recipe_ref,
            "semantic_transform_ids": self.semantic_transform_ids,
            "neutralization_spec_identity": self.neutralization_spec_identity,
            "fit_boundary": self.fit_boundary.value,
        }
        return _content_hash(components)

    @classmethod
    def from_recipe(
        cls,
        recipe: TreatmentRecipe,
        neutralization_spec: Optional[NeutralizationSpec] = None,
    ) -> "TreatmentSpecIdentity":
        """Derive a spec identity from a :class:`TreatmentRecipe`.

        The spec identity is derived from the recipe semantic surface ONLY —
        ordered step (semantic id, stage, ordered params) + neutralization spec
        identity + fit boundary. Data context is intentionally excluded.
        """
        steps = []
        for step in recipe.ordered_steps:
            steps.append(
                (
                    step.step_id,
                    step.semantic_transform_id,
                    step.stage,
                    step.parameters,
                )
            )
        return cls(
            spec_id=recipe.recipe_id,
            recipe_ref=recipe.content_hash,
            semantic_transform_ids=tuple(steps),
            neutralization_spec_identity=neutralization_spec_identity(
                neutralization_spec
            ),
            fit_boundary=recipe.fit_boundary,
        )


@dataclass(frozen=True)
class TreatmentMaterializationIdentity:
    """Content identity of a treatment *materialization*.

    Derived from the spec identity PLUS the materialization context:

    - ``data_snapshot_ref`` — the concrete input data snapshot it was
      materialized against;
    - ``universe_ref`` — the universe used;
    - ``split_ref`` / ``split`` — which split was materialized;
    - ``asof_timestamp`` — the PIT anchor (point-in-time);
    - ``price_basis`` — adj (后复权) vs unadj (未复权);
    - the A股 neutralization binding (``industry_schema`` / ``size_definition``
      / ``pit_identity`` / ``unit``).

    Two materializations differing ONLY in the neutralization binding or unit
    MUST produce different materialization identities (P0-FP #103). This class
    is deliberately a DIFFERENT type from :class:`TreatmentSpecIdentity` so a
    materialization identity can never be mistaken for a spec identity.
    """

    spec_identity: TreatmentSpecIdentity
    materialization_ref: str
    data_snapshot_ref: str
    universe_ref: str
    split_ref: str
    split: MaterializationSplit = MaterializationSplit.FULL_SAMPLE_RESEARCH
    asof_timestamp: Optional[datetime] = None
    price_basis: PriceBasis = PriceBasis.ADJ

    # A股 neutralization binding — part of the materialization identity.
    industry_schema: Optional[str] = None
    size_definition: Optional[str] = None
    pit_identity: PitIdentity = PitIdentity.PIT
    unit: ValueUnit = ValueUnit.DECIMAL
    neutralization_spec_identity: Optional[str] = None

    identity: str = ""

    def __post_init__(self):
        if not isinstance(self.spec_identity, TreatmentSpecIdentity):
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.spec_identity must be a "
                "TreatmentSpecIdentity"
            )
        if not self.materialization_ref:
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.materialization_ref cannot be empty"
            )
        if not self.data_snapshot_ref:
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.data_snapshot_ref cannot be empty"
            )
        if not self.universe_ref:
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.universe_ref cannot be empty"
            )
        if not self.split_ref:
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.split_ref cannot be empty"
            )
        object.__setattr__(self, "split", _enum_or(self.split, MaterializationSplit, "split"))
        object.__setattr__(self, "price_basis", _enum_or(self.price_basis, PriceBasis, "price_basis"))
        object.__setattr__(self, "pit_identity", _enum_or(self.pit_identity, PitIdentity, "pit_identity"))
        object.__setattr__(self, "unit", _enum_or(self.unit, ValueUnit, "unit"))

        # Fail closed: a caller-supplied identity must match the derived one.
        actual = self._derive_identity()
        if self.identity and self.identity != actual:
            raise InvalidContractError(
                "TreatmentMaterializationIdentity.identity does not match "
                "materialization content (fail closed on caller-supplied mismatch)"
            )
        object.__setattr__(self, "identity", actual)

    def _derive_identity(self) -> str:
        """Content identity over spec identity + full materialization context.

        The A股 neutralization binding (industry_schema / size_definition /
        pit_identity / unit) is wired into the derivation, so two
        materializations differing only in the neutralization binding produce
        different identities (P0-FP #103 item 3).
        """
        components = {
            "kind": "treatment_materialization",
            "spec_identity": self.spec_identity.identity,
            "materialization_ref": self.materialization_ref,
            "data_snapshot_ref": self.data_snapshot_ref,
            "universe_ref": self.universe_ref,
            "split_ref": self.split_ref,
            "split": self.split.value,
            "asof_timestamp": self.asof_timestamp,
            "price_basis": self.price_basis.value,
            # A股 neutralization binding surface.
            "industry_schema": self.industry_schema,
            "size_definition": self.size_definition,
            "pit_identity": self.pit_identity.value,
            "unit": self.unit.value,
            "neutralization_spec_identity": self.neutralization_spec_identity,
        }
        return _content_hash(components)


# Re-export the shared stage enum for convenience when reasoning about
# semantic stages of a materialized treatment.
TransformStage = TransformStage


def neutralization_binding(
    spec: Optional[NeutralizationSpec],
    *,
    industry_schema: Optional[str] = None,
    size_definition: Optional[str] = None,
) -> Dict[str, Any]:
    """A股 (A-share) neutralization binding surface.

    Narrows a :class:`NeutralizationSpec` to the A股 conventions the material
    was actually computed against: the industry schema (e.g. ``SW_L1``), the
    size proxy (e.g. ``log_mktcap``), the PIT identity of the exposures, and
    the resolved unit. The returned surface is wired into the materialization
    identity, so two materializations differing only in the neutralization
    binding produce different identities (P0-FP #103 item 3).

    When the spec already pins a schema/definition those values are carried
    through; explicit overrides win (they describe the *materialized*
    binding). When nothing is pinned, the A股 defaults (``SW_L1`` /
    ``log_mktcap``) are used so the contract names the A股 conventions even for
    bare ``ols_neutralize`` kernels.
    """
    spec_industry = spec.industry_schema if spec is not None else None
    spec_size = spec.size_definition if spec is not None else None
    resolved_industry = industry_schema or spec_industry or ASHARE_INDUSTRY_SCHEMA
    resolved_size = size_definition or spec_size or ASHARE_SIZE_DEFINITION
    return {
        "industry_schema": resolved_industry,
        "size_definition": resolved_size,
        "pit_identity": spec.pit_identity if spec is not None else PitIdentity.PIT,
    }


def ensure_materialization_split_valid(
    recipe: TreatmentRecipe,
    split: MaterializationSplit,
    *,
    neutralization_spec: Optional[NeutralizationSpec] = None,
) -> None:
    """Runtime fail-closed check applied where treatments are materialized.

    Raises :class:`TreatmentMaterializationError` when:

    - a ``FitBoundary.FULL_SAMPLE_RESEARCH`` recipe is materialized against an
      evaluation-valid split (VALIDATION / TEST / PRODUCTION). Full-sample
      leakage must NEVER be silently allowed;
    - a research-only neutralization method (pca / kernel / quantile / lad) is
      materialized against an evaluation-valid split;
    - the split enum string is not recognized (fail closed on unknown values).

    Research splits (TRAIN / FULL_SAMPLE_RESEARCH) are always allowed.
    """
    if not isinstance(split, MaterializationSplit):
        raise InvalidContractError(
            f"split must be a MaterializationSplit or its value string, got "
            f"{type(split).__module__}.{type(split).__qualname__}"
        )

    if split.is_evaluation_valid:
        if recipe.fit_boundary == FitBoundary.FULL_SAMPLE_RESEARCH:
            raise TreatmentMaterializationError(
                f"refusing to materialize full-sample-research recipe "
                f"{recipe.recipe_id!r} against evaluation-valid split "
                f"{split.value!r} (full-sample leakage must never be allowed)"
            )
        if neutralization_spec is not None and not neutralization_spec.is_production_admissible:
            raise TreatmentMaterializationError(
                f"refusing to materialize research-only neutralization method "
                f"{neutralization_spec.method.value!r} against evaluation-valid "
                f"split {split.value!r}"
            )


def materialize_identity(
    recipe: TreatmentRecipe,
    *,
    materialization_ref: str,
    data_snapshot_ref: str,
    universe_ref: str,
    split_ref: str,
    split: MaterializationSplit,
    neutralization_spec: Optional[NeutralizationSpec] = None,
    asof_timestamp: Optional[datetime] = None,
    price_basis: PriceBasis = PriceBasis.ADJ,
    industry_schema: Optional[str] = None,
    size_definition: Optional[str] = None,
    pit_identity: Optional[PitIdentity] = None,
    unit: ValueUnit = ValueUnit.DECIMAL,
    require_split_valid: bool = True,
) -> TreatmentMaterializationIdentity:
    """Materialize a recipe into a :class:`TreatmentMaterializationIdentity`.

    This is the ONE place a treatment gets materialized against a context. It
    runs the fail-closed split check (unless explicitly disabled) and derives
    the materialization identity from the spec identity PLUS the full
    materialization context, including the A股 neutralization binding.

    ``neutralization_spec`` may be a :class:`NeutralizationSpec` instance or a
    plain dict of its fields (dict form is accepted so callers without the
    spec class can still bind the A股 conventions).
    """
    # ---- normalize optional neutralization spec --------------------------
    if isinstance(neutralization_spec, dict):
        neutralization_spec = NeutralizationSpec(**neutralization_spec)

    # ---- runtime fail-closed split check ----------------------------------
    if require_split_valid:
        ensure_materialization_split_valid(
            recipe, split, neutralization_spec=neutralization_spec
        )

    # ---- spec identity (definition-only) ----------------------------------
    spec_identity = recipe.spec_identity
    if neutralization_spec is not None:
        spec_identity = TreatmentSpecIdentity(
            spec_id=spec_identity.spec_id,
            recipe_ref=spec_identity.recipe_ref,
            semantic_transform_ids=spec_identity.semantic_transform_ids,
            neutralization_spec_identity=neutralization_spec_identity(
                neutralization_spec
            ),
            fit_boundary=spec_identity.fit_boundary,
        )

    # ---- A股 neutralization binding surface --------------------------------
    binding = neutralization_binding(
        neutralization_spec,
        industry_schema=industry_schema,
        size_definition=size_definition,
    )
    if pit_identity is not None:
        binding["pit_identity"] = pit_identity

    return TreatmentMaterializationIdentity(
        spec_identity=spec_identity,
        materialization_ref=materialization_ref,
        data_snapshot_ref=data_snapshot_ref,
        universe_ref=universe_ref,
        split_ref=split_ref,
        split=split,
        asof_timestamp=asof_timestamp,
        price_basis=price_basis,
        industry_schema=binding["industry_schema"],
        size_definition=binding["size_definition"],
        pit_identity=binding["pit_identity"],
        unit=unit,
        neutralization_spec_identity=neutralization_spec_identity(
            neutralization_spec
        ),
    )


def spec_to_materialization_identity(
    recipe: TreatmentRecipe,
    neutralization_spec: Optional[NeutralizationSpec],
    *,
    materialization_ref: str,
    data_snapshot_ref: str,
    universe_ref: str,
    split_ref: str,
    split: MaterializationSplit,
    asof_timestamp: Optional[datetime] = None,
    price_basis: PriceBasis = PriceBasis.ADJ,
    industry_schema: Optional[str] = None,
    size_definition: Optional[str] = None,
    unit: ValueUnit = ValueUnit.DECIMAL,
) -> TreatmentMaterializationIdentity:
    """Deprecated alias kept for callers that used the previous naming.

    Prefer :func:`materialize_identity` (this module). The alias name is kept
    so a materialization identity is never confused with a spec identity in
    existing call sites — it returns the typed materialization identity.
    """
    return materialize_identity(
        recipe,
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
        unit=unit,
    )


__all__ = [
    "PriceBasis",
    "ValueUnit",
    "MaterializationSplit",
    "TreatmentSpecIdentity",
    "TreatmentMaterializationIdentity",
    "neutralization_spec_identity",
    "neutralization_binding",
    "ensure_materialization_split_valid",
    "materialize_identity",
    "spec_to_materialization_identity",
    "ASHARE_INDUSTRY_SCHEMA",
    "ASHARE_SIZE_DEFINITION",
]
