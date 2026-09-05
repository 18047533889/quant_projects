"""Versioned repair-family registry + diagnosis-specific repair policy (R61-FI-034).

Plan §20 E3/E4/E5.  A repair *family* is a named, versioned class of treatment
actions (e.g. ``CAUSAL_SMOOTHING``, ``SIZE_NEUTRALIZATION``) that:

- declares the diagnoses it is eligible for;
- names an owner execution domain (``FE`` = FactorEngine stateless/DSL surface,
  ``FP`` = FactorPreprocess fitted/stateful surface) — this module only records
  the string, never imports FE/FP;
- declares a parameter schema and a parameter prior (per-plan §24 the parameter
  domain is bound to the family);
- caps how many candidates the family may propose per diagnosis;
- declares a causality class, a complexity cost and the required QE evidence
  profile (referenced by *string* profile id — ``QE_EVIDENCE_PROFILE`` policy
  ids such as ``CHEAP_SCREEN_CN_1D``; this module never imports QE).

The versioned :class:`DiagnosisRepairPolicy` maps a diagnosis (plus a
taxonomy/evidence context) to the repair families allowed for it.  The
:func:`diagnosis_search_budget` generator turns a diagnosis list into the
search budget consumed by the multi-fidelity Stage 2 scheduler (FI-036).

No grading thresholds live in FO: every numeric anchor (if any appears) is
versioned on a policy object; this module carries vocabularies + rule tables
only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from factor_optimizer.policy.repair import DiagnosisKind, canonical_diagnosis_name


class ExecutionDomain(Enum):
    """Owner execution domain of a repair family (plan §20 E3).

    ``FE`` families act on the stateless FactorEngine DSL surface
    (operator/parameter/window changes that recompile to a new definition).
    ``FP`` families act on the FactorPreprocess fitted/stateful treatment
    surface (smoothing/neutralization with fitted state).

    The optimizer records these tokens and routes the actual execution through
    the owning package's adapter; this module never imports FE or FP.
    """

    FE = "FE"
    FP = "FP"


class CausalityClass(Enum):
    """Causality classification of a repair family (plan §20 E3 / matrix notes).

    ``CAUSAL`` families only depend on past information (fit/apply causal,
    no centered look-ahead).  ``STATELESS`` families are pure per-row
    transforms.  ``ABANDON``/``REPRESENTATION`` are structural (no new value
    computation).  ``UNKNOWN`` is the explicit fail-closed default — a family
    must not silently claim causality.
    """

    CAUSAL = "causal"
    STATELESS = "stateless"
    STRUCTURAL = "structural"
    ABANDON = "abandon"
    UNKNOWN = "unknown"


class RepairFamily(Enum):
    """The 20 canonical repair families (plan §20 E3)."""

    NO_OP_RAW = "no_op_raw"
    SIGN_ORIENTATION = "sign_orientation"
    WINDOW_REFINEMENT = "window_refinement"
    DECAY_REFINEMENT = "decay_refinement"
    CAUSAL_SMOOTHING = "causal_smoothing"
    ROBUST_OUTLIER = "robust_outlier"
    MISSINGNESS_FRESHNESS = "missingness_freshness"
    U_SHAPE_REPAIR = "u_shape_repair"
    INVERTED_U_REPAIR = "inverted_u_repair"
    TAIL_SATURATION = "tail_saturation"
    TAIL_HINGE = "tail_hinge"
    INDUSTRY_NEUTRALIZATION = "industry_neutralization"
    SIZE_NEUTRALIZATION = "size_neutralization"
    STYLE_NEUTRALIZATION = "style_neutralization"
    REPRESENTATION_RANK = "representation_rank"
    REPRESENTATION_ZSCORE = "representation_zscore"
    ROBUST_SCALE = "robust_scale"
    OPERATOR_SWAP = "operator_swap"
    LOW_DOF_INTERACTION = "low_dof_interaction"
    ABANDON = "abandon"

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Upper-snake canonical family names."""
        return tuple(m.name for m in cls)


#: Canonical family names (plan §20 E3), upper-snake, order-independent.
PLAN_E3_FAMILY_NAMES: tuple[str, ...] = tuple(RepairFamily.names())


@dataclass(frozen=True)
class ParameterSchema:
    """Declared parameter schema of a repair family (plan §20 E3).

    ``params`` maps each parameter name to a compact schema string:
    - ``"choice:<v1>|<v2>|..."`` — categorical candidates;
    - ``"float:<low>:<high>"`` — continuous domain;
    - ``"int:<low>:<high>"`` — integer domain;
    - ``"flag"`` — boolean.

    The schema is declarative metadata consumed by the conditional-search
    layer (FI-035) to build per-family conditional parameter domains.  No
    sampling happens here.
    """

    params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise TypeError("ParameterSchema.params must be a Mapping")
        normalized = dict(self.params)
        for name, spec in normalized.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("parameter names must be non-empty strings")
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f"parameter {name!r} spec must be a non-empty string")
            if spec.startswith("choice:"):
                candidates = spec.split(":", 1)[1]
                if "|" not in candidates:
                    raise ValueError(
                        f"choice spec for {name!r} must list candidates: "
                        "'choice:a|b|c'"
                    )
            elif spec.startswith("float:"):
                parts = spec.split(":")
                if len(parts) != 3:
                    raise ValueError(
                        f"float spec for {name!r} must be 'float:<low>:<high>'"
                    )
                low, high = float(parts[1]), float(parts[2])
                if low >= high:
                    raise ValueError(
                        f"float spec for {name!r} requires low < high"
                    )
            elif spec.startswith("int:"):
                parts = spec.split(":")
                if len(parts) != 3:
                    raise ValueError(
                        f"int spec for {name!r} must be 'int:<low>:<high>'"
                    )
                low, high = int(parts[1]), int(parts[2])
                if low >= high:
                    raise ValueError(f"int spec for {name!r} requires low < high")
            elif spec != "flag":
                raise ValueError(
                    f"unknown parameter spec for {name!r}: {spec!r} (must be "
                    "choice:/float:/int:/flag)"
                )
        object.__setattr__(self, "params", normalized)

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.params)

    def to_dict(self) -> Dict[str, str]:
        return dict(self.params)


@dataclass(frozen=True)
class ParameterPrior:
    """Declared parameter prior of a repair family (plan §20 E3 / §24).

    ``prior`` maps each parameter name to a JSON-safe prior value.  The value
    is the family's default anchor — the conditional search's injectable prior
    provider may override it per factor using natural horizon / existing DSL
    parameter values / operator metadata / diagnosis evidence.  ``prior`` is
    optional per parameter (a parameter with no prior samples from the schema
    domain).
    """

    prior: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prior, Mapping):
            raise TypeError("ParameterPrior.prior must be a Mapping")
        object.__setattr__(self, "prior", dict(self.prior))

    def value_of(self, name: str) -> Optional[Any]:
        return self.prior.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.prior)


@dataclass(frozen=True)
class RepairFamilyDeclaration:
    """One versioned repair-family declaration (plan §20 E3 fields).

    Attributes:
        family: The :class:`RepairFamily` member.
        eligible_diagnoses: Diagnosis kinds this family may repair (a subset
            of the canonical plan-E2 set — upper-snake names).
        owner: Execution-domain token (``FE``/``FP``).
        parameter_schema: Declared per-parameter schema.
        parameter_prior: Declared per-parameter prior (may be empty).
        maximum_candidates: Maximum number of candidates this family may
            propose per diagnosis (int >= 1).
        causality_class: :class:`CausalityClass` token.
        complexity_cost: Relative complexity cost (int >= 0; ``0`` = free,
            e.g. representation/sign changes; higher = more expensive).
        required_evidence_profile: QE EvidenceProfile *id string* (policy
            ``QE_EVIDENCE_PROFILE``).  Referenced by string only — FO never
            imports QE.
        policy_id: Stable policy-family identifier this declaration belongs
            to (``FO_REPAIR_FAMILY``).
        policy_version: Semantic-version string (``"1.0.0"``).
    """

    family: RepairFamily
    eligible_diagnoses: FrozenSet[str]
    owner: ExecutionDomain
    parameter_schema: ParameterSchema = field(default_factory=ParameterSchema)
    parameter_prior: ParameterPrior = field(default_factory=ParameterPrior)
    maximum_candidates: int = 2
    causality_class: CausalityClass = CausalityClass.UNKNOWN
    complexity_cost: int = 1
    required_evidence_profile: str = "CHEAP_SCREEN_CN_1D"
    policy_id: str = "FO_REPAIR_FAMILY"
    policy_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.family, RepairFamily):
            raise TypeError(
                "RepairFamilyDeclaration.family must be a RepairFamily member"
            )
        if not isinstance(self.owner, ExecutionDomain):
            raise ValueError("owner must be an ExecutionDomain token (FE/FP)")
        if not isinstance(self.causality_class, CausalityClass):
            raise ValueError("causality_class must be a CausalityClass token")
        if isinstance(self.maximum_candidates, bool) or not isinstance(
            self.maximum_candidates, int
        ) or self.maximum_candidates < 1:
            raise ValueError(
                "maximum_candidates must be a positive int (>= 1)"
            )
        if isinstance(self.complexity_cost, bool) or not isinstance(
            self.complexity_cost, int
        ) or self.complexity_cost < 0:
            raise ValueError("complexity_cost must be a non-negative int")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        diagnoses = frozenset(self.eligible_diagnoses)
        for name in diagnoses:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "eligible_diagnoses must be non-empty upper-snake strings"
                )
            # Validate against the canonical E2 vocabulary.  Legacy-only kinds
            # (e.g. LOW_SIGNAL alias) are resolved by canonical name.
            try:
                DiagnosisKind[name]
            except KeyError:
                raise ValueError(
                    f"unknown eligible diagnosis {name!r}; expected one of the "
                    f"canonical E2 diagnosis kind names"
                ) from None
        object.__setattr__(self, "eligible_diagnoses", diagnoses)
        if isinstance(self.parameter_schema, dict):
            object.__setattr__(
                self, "parameter_schema", ParameterSchema(self.parameter_schema)
            )
        elif not isinstance(self.parameter_schema, ParameterSchema):
            raise TypeError("parameter_schema must be a ParameterSchema")
        if isinstance(self.parameter_prior, dict):
            object.__setattr__(
                self, "parameter_prior", ParameterPrior(self.parameter_prior)
            )
        elif not isinstance(self.parameter_prior, ParameterPrior):
            raise TypeError("parameter_prior must be a ParameterPrior")

    @property
    def family_name(self) -> str:
        """Upper-snake canonical family name (``CAUSAL_SMOOTHING``)."""
        return self.family.name

    def eligible_for(self, diagnosis: DiagnosisKind) -> bool:
        """True when this family may repair *diagnosis*."""
        return diagnosis.name in self.eligible_diagnoses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family.name,
            "eligible_diagnoses": sorted(self.eligible_diagnoses),
            "owner": self.owner.value,
            "parameter_schema": self.parameter_schema.to_dict(),
            "parameter_prior": self.parameter_prior.to_dict(),
            "maximum_candidates": self.maximum_candidates,
            "causality_class": self.causality_class.value,
            "complexity_cost": self.complexity_cost,
            "required_evidence_profile": self.required_evidence_profile,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


def _schema(*specs: Tuple[str, str]) -> ParameterSchema:
    return ParameterSchema(dict(specs))


def _prior(**values: Any) -> ParameterPrior:
    return ParameterPrior(values)


#: The 20 canonical repair-family declarations (plan §20 E3), versioned under
#: ``FO_REPAIR_FAMILY`` policy id / ``1.0.0``.
_REPAIR_FAMILY_DECLARATIONS: Tuple[RepairFamilyDeclaration, ...] = (
    RepairFamilyDeclaration(
        family=RepairFamily.NO_OP_RAW,
        eligible_diagnoses=frozenset(
            {"LOW_PREDICTIVE", "U_SHAPE", "HIGH_TURNOVER", "HIGH_COST_DRAG",
             "SIZE_EXPOSURE", "INDUSTRY_EXPOSURE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(("keep_raw", "flag")),
        parameter_prior=_prior(keep_raw=True),
        maximum_candidates=1,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=0,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.SIGN_ORIENTATION,
        eligible_diagnoses=frozenset(
            {"LOW_PREDICTIVE", "LOW_SIGNAL", "LOW_STATISTICAL_CONFIDENCE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(("direction", "choice:flip|keep")),
        parameter_prior=_prior(direction="flip"),
        maximum_candidates=2,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=0,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.WINDOW_REFINEMENT,
        eligible_diagnoses=frozenset(
            {"HIGH_TURNOVER", "UNSTABLE_IC", "RECENT_DEGRADATION",
             "HIGH_COMPLEXITY", "LOW_PREDICTIVE", "SPARSE_FACTOR",
             "HIGH_VARIANCE", "REGIME_DEPENDENT"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("window_mode", "choice:shorter|longer|same"),
            ("scale", "float:0.5:2.0"),
        ),
        parameter_prior=_prior(window_mode="shorter", scale=0.7),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=1,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.DECAY_REFINEMENT,
        eligible_diagnoses=frozenset(
            {"HIGH_TURNOVER", "UNSTABLE_IC", "HIGH_COST_DRAG", "HIGH_VARIANCE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("decay", "float:0.0:0.95"),
            ("half_life_relative", "flag"),
        ),
        parameter_prior=_prior(decay=0.5, half_life_relative=True),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=1,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.CAUSAL_SMOOTHING,
        eligible_diagnoses=frozenset(
            {"HIGH_TURNOVER", "HIGH_COST_DRAG", "HIGH_VARIANCE", "UNSTABLE_IC",
             "RECENT_DEGRADATION"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("method", "choice:EWMA|KAMA|IIR|Kalman"),
            ("natural_time_scale_relative", "float:0.1:2.0"),
        ),
        parameter_prior=_prior(method="EWMA", natural_time_scale_relative=0.5),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.ROBUST_OUTLIER,
        eligible_diagnoses=frozenset(
            {"TOP_TAIL_COLLAPSE", "NUMERICAL_INSTABILITY", "HIGH_DRAWDOWN",
             "NEGATIVE_TAIL_RISK", "HIGH_TIE_RATIO", "LONG_UNDERWATER"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("lower_quantile", "float:0.005:0.05"),
            ("upper_quantile", "float:0.95:0.995"),
        ),
        parameter_prior=_prior(lower_quantile=0.01, upper_quantile=0.99),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=1,
        required_evidence_profile="SHAPE_DIAGNOSTIC_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.MISSINGNESS_FRESHNESS,
        eligible_diagnoses=frozenset(
            {"POOR_COVERAGE", "STALE_DATA", "SPARSE_FACTOR",
             "DATA_QUALITY_FAILURE", "LOW_CAPACITY"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("mode", "choice:fill|drop|flag"),
            ("freshness_window", "int:1:20"),
        ),
        parameter_prior=_prior(mode="flag", freshness_window=5),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=1,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.U_SHAPE_REPAIR,
        eligible_diagnoses=frozenset({"U_SHAPE"}),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("center", "float:0.0:1.0"),
            ("power", "float:0.5:3.0"),
            ("asymmetry", "flag"),
        ),
        parameter_prior=_prior(center=0.5, power=2.0, asymmetry=False),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.INVERTED_U_REPAIR,
        eligible_diagnoses=frozenset({"INVERTED_U"}),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("center", "float:0.0:1.0"),
            ("power", "float:0.5:3.0"),
            ("asymmetry", "flag"),
        ),
        parameter_prior=_prior(center=0.5, power=2.0, asymmetry=False),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.TAIL_SATURATION,
        eligible_diagnoses=frozenset(
            {"TOP_TAIL_COLLAPSE", "BOTTOM_TAIL_COLLAPSE", "TAIL_ONLY",
             "NEGATIVE_TAIL_RISK", "HIGH_DRAWDOWN", "LONG_UNDERWATER"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("saturation_quantile", "float:0.90:0.99"),
            ("saturate", "choice:top|bottom|both"),
        ),
        parameter_prior=_prior(saturation_quantile=0.97, saturate="top"),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=1,
        required_evidence_profile="SHAPE_DIAGNOSTIC_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.TAIL_HINGE,
        eligible_diagnoses=frozenset(
            {"TOP_TAIL_COLLAPSE", "BOTTOM_TAIL_COLLAPSE", "TAIL_ONLY",
             "LOW_PREDICTIVE", "U_SHAPE", "INVERTED_U"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("hinge", "choice:top|bottom"),
            ("hinge_value", "float:-1.0:1.0"),
        ),
        parameter_prior=_prior(hinge="top", hinge_value=0.0),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.INDUSTRY_NEUTRALIZATION,
        eligible_diagnoses=frozenset(
            {"INDUSTRY_EXPOSURE", "MULTI_STYLE_EXPOSURE", "BETA_EXPOSURE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("exposure_set", "choice:industry|industry_size"),
            ("method", "choice:ols|weighted_ols"),
        ),
        parameter_prior=_prior(exposure_set="industry", method="ols"),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=3,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.SIZE_NEUTRALIZATION,
        eligible_diagnoses=frozenset(
            {"SIZE_EXPOSURE", "MULTI_STYLE_EXPOSURE", "TOP_TAIL_COLLAPSE",
             "BOTTOM_TAIL_COLLAPSE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("exposure_set", "choice:size|size_industry"),
            ("method", "choice:ols|weighted_ols"),
        ),
        parameter_prior=_prior(exposure_set="size", method="ols"),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=3,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.STYLE_NEUTRALIZATION,
        eligible_diagnoses=frozenset(
            {"BETA_EXPOSURE", "LIQUIDITY_EXPOSURE", "VOLATILITY_EXPOSURE",
             "MULTI_STYLE_EXPOSURE", "SIZE_EXPOSURE"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("exposure_set", "choice:beta|liquidity|volatility|style_multi"),
            ("method", "choice:ols|weighted_ols"),
        ),
        parameter_prior=_prior(exposure_set="style_multi", method="ols"),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=3,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.REPRESENTATION_RANK,
        eligible_diagnoses=frozenset(
            {"TOP_TAIL_COLLAPSE", "BOTTOM_TAIL_COLLAPSE", "TAIL_ONLY",
             "HIGH_TURNOVER", "LOW_CAPACITY", "HIGH_TIE_RATIO"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("rank_axis", "choice:cross_sectional|ts"),
            ("tie_method", "choice:average|min"),
        ),
        parameter_prior=_prior(rank_axis="cross_sectional", tie_method="average"),
        maximum_candidates=2,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=1,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.REPRESENTATION_ZSCORE,
        eligible_diagnoses=frozenset(
            {"LOW_CAPACITY", "HIGH_COST_DRAG", "SIZE_EXPOSURE"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("zscore_axis", "choice:cross_sectional|ts"),
            ("cap", "float:2.0:5.0"),
        ),
        parameter_prior=_prior(zscore_axis="cross_sectional", cap=3.0),
        maximum_candidates=2,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=1,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.ROBUST_SCALE,
        eligible_diagnoses=frozenset(
            {"NUMERICAL_INSTABILITY", "TOP_TAIL_COLLAPSE", "HIGH_DRAWDOWN",
             "LOW_CAPACITY", "HIGH_COST_DRAG"}
        ),
        owner=ExecutionDomain.FP,
        parameter_schema=_schema(
            ("scale", "choice:mad|iqr|std"),
            ("center", "choice:median|mean"),
        ),
        parameter_prior=_prior(scale="mad", center="median"),
        maximum_candidates=3,
        causality_class=CausalityClass.STATELESS,
        complexity_cost=1,
        required_evidence_profile="SHAPE_DIAGNOSTIC_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.OPERATOR_SWAP,
        eligible_diagnoses=frozenset(
            {"NUMERICAL_INSTABILITY", "HIGH_COMPLEXITY", "RECENT_DEGRADATION",
             "UNSTABLE_IC", "LOW_PREDICTIVE", "NONSTATIONARY_SHAPE",
             "REGIME_DEPENDENT"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("target_family", "choice:delay|delta|ratio|rolling"),
            ("aggressiveness", "choice:conservative|moderate|aggressive"),
        ),
        parameter_prior=_prior(target_family="delay", aggressiveness="conservative"),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.LOW_DOF_INTERACTION,
        eligible_diagnoses=frozenset(
            {"OVERFIT_GENERALIZATION", "OVERFITTING", "HIGH_COMPLEXITY",
             "RECENT_DEGRADATION"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(
            ("interaction_degree", "int:1:2"),
            ("doe_reduction", "float:0.5:1.0"),
        ),
        parameter_prior=_prior(interaction_degree=1, doe_reduction=0.7),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=2,
        required_evidence_profile="MODEL_FEATURE_DIAGNOSTIC_CN_1D",
    ),
    RepairFamilyDeclaration(
        family=RepairFamily.ABANDON,
        eligible_diagnoses=frozenset(
            {"INTEGRITY_FAILURE", "PIT_VIOLATION", "LABEL_TIMING_VIOLATION",
             "DATA_QUALITY_FAILURE", "SEMANTIC_DUPLICATE",
             "VALUE_NEAR_DUPLICATE", "LOW_NOVELTY", "TIMING_VIOLATION",
             "DOMAIN_MISMATCH"}
        ),
        owner=ExecutionDomain.FE,
        parameter_schema=_schema(),
        parameter_prior=_prior(),
        maximum_candidates=1,
        causality_class=CausalityClass.ABANDON,
        complexity_cost=0,
        required_evidence_profile="CHEAP_SCREEN_CN_1D",
    ),
)


class RepairFamilyRegistry:
    """Versioned registry of repair-family declarations (plan §20 E3).

    Immutable after construction: lookups resolve a family by upper-snake
    name and expose the per-diagnosis eligible families.  The registry is
    versioned on every declaration (``policy_id`` + ``policy_version``) so a
    consumer can audit exactly which declaration version produced a repair
    proposal.
    """

    def __init__(
        self,
        declarations: Sequence[RepairFamilyDeclaration],
        policy_id: str = "FO_REPAIR_FAMILY",
        policy_version: str = "1.0.0",
    ) -> None:
        if not isinstance(declarations, (list, tuple)) or not declarations:
            raise ValueError("registry requires a non-empty declaration list")
        if not all(isinstance(d, RepairFamilyDeclaration) for d in declarations):
            raise TypeError(
                "registry declarations must be RepairFamilyDeclaration instances"
            )
        families = [d.family for d in declarations]
        if len(families) != len(set(families)):
            raise ValueError("registry declarations must be unique per family")
        self._declarations: Dict[RepairFamily, RepairFamilyDeclaration] = {
            d.family: d for d in declarations
        }
        self._policy_id = policy_id
        self._policy_version = policy_version
        for declaration in declarations:
            if declaration.policy_id != policy_id:
                raise ValueError(
                    f"declaration {declaration.family.name!r} policy_id "
                    f"{declaration.policy_id!r} does not match registry "
                    f"{policy_id!r}"
                )

    @classmethod
    def default(cls) -> "RepairFamilyRegistry":
        """The canonical 20-family registry (plan §20 E3)."""
        return cls(_REPAIR_FAMILY_DECLARATIONS)

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def family_names(self) -> tuple[str, ...]:
        """Upper-snake names of every declared family, sorted."""
        return tuple(sorted(d.family.name for d in self._declarations.values()))

    @property
    def declarations(self) -> Tuple[RepairFamilyDeclaration, ...]:
        return tuple(self._declarations.values())

    def get(self, family: RepairFamily | str) -> RepairFamilyDeclaration:
        """Resolve a family declaration (by member or upper-snake name)."""
        if isinstance(family, str):
            try:
                family = RepairFamily[family]
            except KeyError:
                raise ValueError(
                    f"unknown repair family {family!r}; expected one of "
                    f"{PLAN_E3_FAMILY_NAMES}"
                ) from None
        try:
            return self._declarations[family]
        except KeyError:
            raise ValueError(
                f"repair family {family.name!r} is not registered in this "
                "RepairFamilyRegistry"
            ) from None

    def eligible_families_for(
        self, diagnosis: DiagnosisKind | str
    ) -> Tuple[RepairFamilyDeclaration, ...]:
        """Declarations eligible for a diagnosis, in registry order."""
        kind = _as_diagnosis(diagnosis)
        return tuple(
            d for d in self._declarations.values() if d.eligible_for(kind)
        )

    def __len__(self) -> int:
        return len(self._declarations)

    def __contains__(self, family: RepairFamily | str) -> bool:
        try:
            self.get(family)
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self._policy_id,
            "policy_version": self._policy_version,
            "declarations": [d.to_dict() for d in self.declarations],
        }


def _as_diagnosis(diagnosis: DiagnosisKind | str) -> DiagnosisKind:
    if isinstance(diagnosis, DiagnosisKind):
        return diagnosis
    if isinstance(diagnosis, str):
        try:
            return DiagnosisKind[diagnosis]
        except KeyError:
            raise ValueError(
                f"unknown diagnosis kind name {diagnosis!r}; expected one of "
                "the canonical E2 diagnosis kind names"
            ) from None
    raise TypeError("diagnosis must be a DiagnosisKind or an upper-snake name")


# ---------------------------------------------------------------------------
# Diagnosis-specific repair policy (plan §20 E4)
# ---------------------------------------------------------------------------


class DomainToken(Enum):
    """Taxonomy/evidence tokens a repair rule may condition on.

    The plan §20 E4 examples are *evidence-conditioned* rules (HIGH_TURNOVER +
    PRICE_VOLUME domain → smoothing; TOP_TAIL_COLLAPSE + outlier evidence →
    robust outlier).  The policy receives an optional evidence context
    (``Mapping[str, object]``) and classifies it into these domain/evidence
    tokens; an absent/unclear token never matches a condition.
    """

    PRICE_VOLUME = "price_volume"
    OUTLIER_EVIDENCE = "outlier_evidence"
    SIZE_CONCENTRATION = "size_concentration"
    STABLE_SHAPE_CONFIDENCE = "stable_shape_confidence"
    ALPHA_AFTER_RESIDUAL = "alpha_after_residual"
    NONE = "none"


def classify_evidence_context(
    evidence: Optional[Mapping[str, object]],
) -> frozenset[DomainToken]:
    """Deterministic, conservative tokenization of a repair evidence context.

    A key is only asserted when it is present AND the caller-provided value
    matches an explicit predicate — absence/unknown never matches (fail
    closed).  This mirrors the FA evidence-status iron law (missing evidence
    never fires a rule) and lets the repair policy (plan §20 E4) gate the
    diagnosis→family rules on taxonomy/evidence tokens.
    """
    evidence = evidence or {}
    tokens: set[DomainToken] = set()

    def _truthy(name: str) -> bool:
        value = evidence.get(name)
        return bool(value)  # explicit True / non-empty / non-zero

    def _number(name: str, op, threshold: float) -> bool:
        value = evidence.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return bool(op(abs(float(value)), threshold))

    # PRICE_VOLUME: data-domain taxonomy token present.
    if _truthy("data_domains_price_volume"):
        tokens.add(DomainToken.PRICE_VOLUME)
    # OUTLIER_EVIDENCE: explicit outlier-ratio / outlier-flag evidence.
    if _truthy("evidence.outlier") or _number(
        "outlier_ratio", lambda a, t: a > t, 0.0
    ):
        tokens.add(DomainToken.OUTLIER_EVIDENCE)
    # SIZE_CONCENTRATION: explicit size-exposure evidence.
    if _truthy("exposure.size") or _number(
        "size_exposure", lambda a, t: a > t, 0.01
    ):
        tokens.add(DomainToken.SIZE_CONCENTRATION)
    # STABLE_SHAPE_CONFIDENCE: stable shape evidence.
    if _truthy("shape.stable") or _number(
        "shape_confidence", lambda a, t: a >= t, 0.5
    ):
        tokens.add(DomainToken.STABLE_SHAPE_CONFIDENCE)
    # ALPHA_AFTER_RESIDUAL: residual-alpha evidence present and positive.
    if _truthy("evidence.residual_alpha") or _number(
        "residual_rank_ic", lambda a, t: a > t, 0.0
    ):
        tokens.add(DomainToken.ALPHA_AFTER_RESIDUAL)
    if not tokens:
        tokens.add(DomainToken.NONE)
    return frozenset(tokens)


@dataclass(frozen=True)
class RepairRule:
    """One diagnosis→repair-family rule (plan §20 E4).

    Attributes:
        rule_id: Stable rule identifier.
        diagnosis: Diagnosis kind the rule applies to.
        allowed_families: Ordered upper-snake repair-family names permitted
            for this diagnosis+context.
        requires_tokens: Evidence-context tokens that must ALL be present for
            the rule to fire (empty = context-free).
        rationale: Human-readable plan §20 E4 rationale.
    """

    rule_id: str
    diagnosis: str
    allowed_families: Tuple[str, ...]
    requires_tokens: Tuple[DomainToken, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string")
        _as_diagnosis(self.diagnosis)  # validate canonical name
        if not isinstance(self.allowed_families, (tuple, list)) or not self.allowed_families:
            raise ValueError("allowed_families must be a non-empty list of family names")
        families = tuple(self.allowed_families)
        for name in families:
            if name not in PLAN_E3_FAMILY_NAMES:
                raise ValueError(
                    f"unknown repair family {name!r} in rule {self.rule_id!r}"
                )
        object.__setattr__(self, "allowed_families", families)
        object.__setattr__(self, "requires_tokens", tuple(self.requires_tokens))


@dataclass(frozen=True)
class DiagnosisRepairPolicy:
    """Versioned diagnosis→repair-family rule table (plan §20 E4).

    The rules are evaluated in registry order.  For a diagnosis + optional
    evidence context, the first rule whose ``requires_tokens`` are all present
    yields its ordered allowed families.  No grading thresholds live here —
    only the rule table and its version.

    Attributes:
        policy_id: Stable policy-family identifier (``FO_DIAGNOSIS_REPAIR``).
        policy_version: SemVer string (``"1.0.0"``).
        rules: Ordered tuple of :class:`RepairRule`.
    """

    policy_id: str = "FO_DIAGNOSIS_REPAIR"
    policy_version: str = "1.0.0"
    rules: Tuple[RepairRule, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        object.__setattr__(self, "rules", tuple(self.rules))

    def families_for(
        self,
        diagnosis: DiagnosisKind | str,
        evidence: Optional[Mapping[str, object]] = None,
        domain_token: Optional[DomainToken] = None,
    ) -> Tuple[str, ...]:
        """Ordered repair-family names allowed for a diagnosis.

        Evidence-conditioned rules (plan §20 E4) fire only when every required
        token is present; otherwise the context-free fallback rule for the
        diagnosis is used.  An unknown diagnosis (no rule at all) yields an
        empty tuple — nothing is silently invented.
        """
        kind = _as_diagnosis(diagnosis)
        tokens = (
            classify_evidence_context(evidence)
            if domain_token is None
            else frozenset({domain_token})
        )
        for rule in self.rules:
            if rule.diagnosis != kind.name:
                continue
            if not rule.requires_tokens:
                return rule.allowed_families
            if all(t in tokens for t in rule.requires_tokens):
                return rule.allowed_families
        return ()


#: Canonical diagnosis-repair policy (plan §20 E4 examples, context-free
#: fallbacks first, then the evidence-conditioned rules).
DEFAULT_DIAGNOSIS_REPAIR_RULES: Tuple[RepairRule, ...] = (
    # OVERFIT_GENERALIZATION -> reduce degrees of freedom / simpler definition
    RepairRule(
        rule_id="FO_E4_OVERFIT_DOF",
        diagnosis="OVERFIT_GENERALIZATION",
        allowed_families=("LOW_DOF_INTERACTION", "WINDOW_REFINEMENT"),
        rationale="reduce degrees of freedom / narrower recipe / simpler definition",
    ),
    RepairRule(
        rule_id="FO_E4_OVERFITTING_DOF",
        diagnosis="OVERFITTING",
        allowed_families=("LOW_DOF_INTERACTION", "WINDOW_REFINEMENT"),
        rationale="legacy OVERFITTING alias routes to the same DOF reduction",
    ),
    # HIGH_TURNOVER + PRICE_VOLUME -> CAUSAL_SMOOTHING / DECAY_REFINEMENT
    RepairRule(
        rule_id="FO_E4_TURNOVER_PV",
        diagnosis="HIGH_TURNOVER",
        allowed_families=("CAUSAL_SMOOTHING", "DECAY_REFINEMENT"),
        requires_tokens=(DomainToken.PRICE_VOLUME,),
        rationale="HIGH_TURNOVER + PRICE_VOLUME -> CAUSAL_SMOOTHING / DECAY_REFINEMENT",
    ),
    RepairRule(
        rule_id="FO_E4_TURNOVER_FALLBACK",
        diagnosis="HIGH_TURNOVER",
        allowed_families=("DECAY_REFINEMENT", "WINDOW_REFINEMENT"),
        rationale="context-free HIGH_TURNOVER fallback",
    ),
    # U_SHAPE + stable shape confidence -> U_SHAPE_REPAIR
    RepairRule(
        rule_id="FO_E4_U_SHAPE_STABLE",
        diagnosis="U_SHAPE",
        allowed_families=("U_SHAPE_REPAIR",),
        requires_tokens=(DomainToken.STABLE_SHAPE_CONFIDENCE,),
        rationale="U_SHAPE + stable shape confidence -> U_SHAPE_REPAIR",
    ),
    RepairRule(
        rule_id="FO_E4_U_SHAPE_FALLBACK",
        diagnosis="U_SHAPE",
        allowed_families=("U_SHAPE_REPAIR", "TAIL_HINGE"),
        rationale="context-free U_SHAPE fallback",
    ),
    # INVERTED_U -> INVERTED_U_REPAIR
    RepairRule(
        rule_id="FO_E4_INVU",
        diagnosis="INVERTED_U",
        allowed_families=("INVERTED_U_REPAIR", "TAIL_HINGE"),
        rationale="INVERTED_U shape repair family",
    ),
    # TOP_TAIL_COLLAPSE + outlier evidence -> ROBUST_OUTLIER / TAIL_SATURATION
    RepairRule(
        rule_id="FO_E4_TOP_CLIFF_OUTLIER",
        diagnosis="TOP_TAIL_COLLAPSE",
        allowed_families=("ROBUST_OUTLIER", "TAIL_SATURATION"),
        requires_tokens=(DomainToken.OUTLIER_EVIDENCE,),
        rationale="TOP_TAIL_COLLAPSE + outlier evidence -> ROBUST_OUTLIER / TAIL_SATURATION",
    ),
    # TOP_TAIL_COLLAPSE + size exposure concentration -> SIZE_NEUTRALIZATION
    RepairRule(
        rule_id="FO_E4_TOP_CLIFF_SIZE",
        diagnosis="TOP_TAIL_COLLAPSE",
        allowed_families=("SIZE_NEUTRALIZATION",),
        requires_tokens=(DomainToken.SIZE_CONCENTRATION,),
        rationale="TOP_TAIL_COLLAPSE + size exposure concentration -> SIZE_NEUTRALIZATION",
    ),
    RepairRule(
        rule_id="FO_E4_TOP_CLIFF_FALLBACK",
        diagnosis="TOP_TAIL_COLLAPSE",
        allowed_families=("ROBUST_OUTLIER", "TAIL_SATURATION", "TAIL_HINGE"),
        rationale="context-free TOP_TAIL_COLLAPSE fallback",
    ),
    # BOTTOM_TAIL_COLLAPSE -> tail repairs
    RepairRule(
        rule_id="FO_E4_BOT_CLIFF",
        diagnosis="BOTTOM_TAIL_COLLAPSE",
        allowed_families=("TAIL_SATURATION", "ROBUST_OUTLIER"),
        rationale="bottom tail collapse repair families",
    ),
    # SIZE_EXPOSURE + alpha remains after size residual test -> SIZE_NEUTRALIZATION
    RepairRule(
        rule_id="FO_E4_SIZE_RESIDUAL",
        diagnosis="SIZE_EXPOSURE",
        allowed_families=("SIZE_NEUTRALIZATION",),
        requires_tokens=(DomainToken.ALPHA_AFTER_RESIDUAL,),
        rationale="SIZE_EXPOSURE + alpha remains after size residual test -> SIZE_NEUTRALIZATION",
    ),
    RepairRule(
        rule_id="FO_E4_SIZE_FALLBACK",
        diagnosis="SIZE_EXPOSURE",
        allowed_families=("SIZE_NEUTRALIZATION", "REPRESENTATION_ZSCORE"),
        rationale="context-free SIZE_EXPOSURE fallback",
    ),
    # Exposure diagnoses
    RepairRule(
        rule_id="FO_E4_INDUSTRY",
        diagnosis="INDUSTRY_EXPOSURE",
        allowed_families=("INDUSTRY_NEUTRALIZATION",),
        rationale="industry exposure -> industry neutralization",
    ),
    RepairRule(
        rule_id="FO_E4_BETA",
        diagnosis="BETA_EXPOSURE",
        allowed_families=("STYLE_NEUTRALIZATION", "INDUSTRY_NEUTRALIZATION"),
        rationale="beta exposure -> style neutralization",
    ),
    RepairRule(
        rule_id="FO_E4_LIQUIDITY",
        diagnosis="LIQUIDITY_EXPOSURE",
        allowed_families=("STYLE_NEUTRALIZATION", "REPRESENTATION_RANK"),
        rationale="liquidity exposure -> style neutralization",
    ),
    RepairRule(
        rule_id="FO_E4_VOLATILITY",
        diagnosis="VOLATILITY_EXPOSURE",
        allowed_families=("STYLE_NEUTRALIZATION",),
        rationale="volatility exposure -> style neutralization",
    ),
    RepairRule(
        rule_id="FO_E4_MULTI_STYLE",
        diagnosis="MULTI_STYLE_EXPOSURE",
        allowed_families=("STYLE_NEUTRALIZATION", "SIZE_NEUTRALIZATION"),
        rationale="multi-style exposure -> style/size neutralization",
    ),
    # Predictive / stability diagnoses
    RepairRule(
        rule_id="FO_E4_LOW_PREDICTIVE_SIGN",
        diagnosis="LOW_PREDICTIVE",
        allowed_families=("SIGN_ORIENTATION", "OPERATOR_SWAP", "WINDOW_REFINEMENT"),
        rationale="weak predictive signal -> sign check / operator swap / window",
    ),
    RepairRule(
        rule_id="FO_E4_LOW_PREDICTIVE_NOOP",
        diagnosis="LOW_SIGNAL",
        allowed_families=("SIGN_ORIENTATION", "NO_OP_RAW"),
        rationale="legacy LOW_SIGNAL alias",
    ),
    RepairRule(
        rule_id="FO_E4_UNSTABLE_IC",
        diagnosis="UNSTABLE_IC",
        allowed_families=("CAUSAL_SMOOTHING", "WINDOW_REFINEMENT"),
        rationale="unstable IC -> smoothing / window refinement",
    ),
    RepairRule(
        rule_id="FO_E4_RECENT_DEGRADATION",
        diagnosis="RECENT_DEGRADATION",
        allowed_families=("CAUSAL_SMOOTHING", "WINDOW_REFINEMENT", "OPERATOR_SWAP"),
        rationale="recent degradation -> smoothing / window / operator",
    ),
    RepairRule(
        rule_id="FO_E4_LOW_STAT_CONF",
        diagnosis="LOW_STATISTICAL_CONFIDENCE",
        allowed_families=("WINDOW_REFINEMENT", "SIGN_ORIENTATION"),
        rationale="low statistical confidence -> broaden effective sample",
    ),
    RepairRule(
        rule_id="FO_E4_HIGH_VARIANCE",
        diagnosis="HIGH_VARIANCE",
        allowed_families=("CAUSAL_SMOOTHING", "DECAY_REFINEMENT", "WINDOW_REFINEMENT"),
        rationale="legacy HIGH_VARIANCE alias",
    ),
    # Data-quality diagnoses
    RepairRule(
        rule_id="FO_E4_POOR_COVERAGE",
        diagnosis="POOR_COVERAGE",
        allowed_families=("MISSINGNESS_FRESHNESS", "WINDOW_REFINEMENT"),
        rationale="poor coverage -> missingness/freshness handling",
    ),
    RepairRule(
        rule_id="FO_E4_STALE",
        diagnosis="STALE_DATA",
        allowed_families=("MISSINGNESS_FRESHNESS",),
        rationale="stale data -> freshness handling",
    ),
    RepairRule(
        rule_id="FO_E4_SPARSE",
        diagnosis="SPARSE_FACTOR",
        allowed_families=("MISSINGNESS_FRESHNESS", "WINDOW_REFINEMENT"),
        rationale="sparse factor -> missingness / widen window",
    ),
    RepairRule(
        rule_id="FO_E4_DQ",
        diagnosis="DATA_QUALITY_FAILURE",
        allowed_families=("MISSINGNESS_FRESHNESS", "ABANDON"),
        rationale="data-quality failure -> freshness or abandon",
    ),
    RepairRule(
        rule_id="FO_E4_HIGH_TIE",
        diagnosis="HIGH_TIE_RATIO",
        allowed_families=("REPRESENTATION_RANK", "ROBUST_OUTLIER"),
        rationale="high tie ratio -> representation change",
    ),
    RepairRule(
        rule_id="FO_E4_NUM_INSTAB",
        diagnosis="NUMERICAL_INSTABILITY",
        allowed_families=("ROBUST_SCALE", "ROBUST_OUTLIER", "OPERATOR_SWAP"),
        rationale="numerical instability -> robust scale/outlier/operator",
    ),
    RepairRule(
        rule_id="FO_E4_LOW_CAPACITY",
        diagnosis="LOW_CAPACITY",
        allowed_families=("REPRESENTATION_ZSCORE", "REPRESENTATION_RANK"),
        rationale="low capacity -> representation change",
    ),
    # Portfolio-economics diagnoses
    RepairRule(
        rule_id="FO_E4_COST_DRAG",
        diagnosis="HIGH_COST_DRAG",
        allowed_families=("CAUSAL_SMOOTHING", "DECAY_REFINEMENT"),
        rationale="cost drag -> smoothing / decay",
    ),
    RepairRule(
        rule_id="FO_E4_DRAWDOWN",
        diagnosis="HIGH_DRAWDOWN",
        allowed_families=("ROBUST_OUTLIER", "TAIL_SATURATION", "ROBUST_SCALE"),
        rationale="high drawdown -> robust outlier / tail saturation / scale",
    ),
    RepairRule(
        rule_id="FO_E4_UNDERWATER",
        diagnosis="LONG_UNDERWATER",
        allowed_families=("ROBUST_OUTLIER", "TAIL_SATURATION"),
        rationale="long underwater -> robust/tail repair",
    ),
    RepairRule(
        rule_id="FO_E4_NEG_TAIL",
        diagnosis="NEGATIVE_TAIL_RISK",
        allowed_families=("TAIL_SATURATION", "ROBUST_OUTLIER"),
        rationale="negative tail risk -> tail saturation / robust outlier",
    ),
    RepairRule(
        rule_id="FO_E4_REGIME",
        diagnosis="REGIME_DEPENDENT",
        allowed_families=("WINDOW_REFINEMENT", "OPERATOR_SWAP"),
        rationale="regime-dependent -> window/operator adaptation",
    ),
    # Shape diagnoses
    RepairRule(
        rule_id="FO_E4_TAIL_ONLY",
        diagnosis="TAIL_ONLY",
        allowed_families=("TAIL_HINGE", "TAIL_SATURATION"),
        rationale="tail-only signal -> hinge/saturation",
    ),
    RepairRule(
        rule_id="FO_E4_NONSTATIONARY_SHAPE",
        diagnosis="NONSTATIONARY_SHAPE",
        allowed_families=("OPERATOR_SWAP", "WINDOW_REFINEMENT"),
        rationale="nonstationary shape -> operator/window",
    ),
    # Novelty / redundancy / complexity
    RepairRule(
        rule_id="FO_E4_DUP",
        diagnosis="SEMANTIC_DUPLICATE",
        allowed_families=("ABANDON",),
        rationale="semantic duplicate -> abandon (alias suffices)",
    ),
    RepairRule(
        rule_id="FO_E4_NEAR_DUP",
        diagnosis="VALUE_NEAR_DUPLICATE",
        allowed_families=("ABANDON",),
        rationale="near-duplicate -> abandon",
    ),
    RepairRule(
        rule_id="FO_E4_LOW_NOVELTY",
        diagnosis="LOW_NOVELTY",
        allowed_families=("OPERATOR_SWAP",),
        rationale="low novelty -> operator swap to differentiate",
    ),
    RepairRule(
        rule_id="FO_E4_HIGH_COMPLEXITY",
        diagnosis="HIGH_COMPLEXITY",
        allowed_families=("WINDOW_REFINEMENT", "LOW_DOF_INTERACTION", "OPERATOR_SWAP"),
        rationale="high complexity -> window/DOF reduction",
    ),
    # Integrity / timing (hard rejects — no repair, ABANDON only)
    RepairRule(
        rule_id="FO_E4_INTEGRITY",
        diagnosis="INTEGRITY_FAILURE",
        allowed_families=("ABANDON",),
        rationale="integrity failure is a hard reject",
    ),
    RepairRule(
        rule_id="FO_E4_PIT",
        diagnosis="PIT_VIOLATION",
        allowed_families=("ABANDON",),
        rationale="PIT violation is a hard reject",
    ),
    RepairRule(
        rule_id="FO_E4_LABEL_TIMING",
        diagnosis="LABEL_TIMING_VIOLATION",
        allowed_families=("ABANDON",),
        rationale="label timing violation is a hard reject",
    ),
    RepairRule(
        rule_id="FO_E4_TIMING_LEGACY",
        diagnosis="TIMING_VIOLATION",
        allowed_families=("ABANDON",),
        rationale="legacy timing violation alias -> abandon",
    ),
    RepairRule(
        rule_id="FO_E4_DOMAIN_MISMATCH",
        diagnosis="DOMAIN_MISMATCH",
        allowed_families=("ABANDON",),
        rationale="domain mismatch -> abandon",
    ),
)


def default_diagnosis_repair_policy() -> DiagnosisRepairPolicy:
    """The canonical diagnosis-repair policy (plan §20 E4)."""
    return DiagnosisRepairPolicy(rules=DEFAULT_DIAGNOSIS_REPAIR_RULES)


# ---------------------------------------------------------------------------
# Search budget generated from diagnosis (plan §20 E5 / §31 Stage 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairBudgetConfig:
    """Configurable search-budget bounds (plan §20 E5).

    Attributes:
        max_primary_diagnoses: Maximum primary diagnoses the search may act
            on (default 3).
        max_repair_families_per_diagnosis: Maximum repair families per
            diagnosis (default 2).
        max_candidates_per_family: ``(min, max)`` inclusive bound on
            candidates per family (default 2-4).
        soft_total_candidates: ``(min, max)`` soft total candidate budget
            (default 6-12).
    """

    max_primary_diagnoses: int = 3
    max_repair_families_per_diagnosis: int = 2
    min_candidates_per_family: int = 2
    max_candidates_per_family: int = 4
    min_total_candidates: int = 6
    max_total_candidates: int = 12

    def __post_init__(self) -> None:
        for name in (
            "max_primary_diagnoses",
            "max_repair_families_per_diagnosis",
            "min_candidates_per_family",
            "max_candidates_per_family",
            "min_total_candidates",
            "max_total_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.min_candidates_per_family > self.max_candidates_per_family:
            raise ValueError(
                "min_candidates_per_family must be <= max_candidates_per_family"
            )
        if self.min_total_candidates > self.max_total_candidates:
            raise ValueError(
                "min_total_candidates must be <= max_total_candidates"
            )
        # Soft total budget is at least the family-floor coverage (E5: soft
        # 6-12 while each family allows 2-4; with max_primary_diagnoses=3 and
        # 2 families each that is 12 candidates — consistent).
        if self.max_total_candidates > (
            self.max_primary_diagnoses
            * self.max_repair_families_per_diagnosis
            * self.max_candidates_per_family
        ):
            # Not an error: the soft total is a cap independent of the family
            # expansion.  Keep the declared value (bounded below by min).
            pass

    @property
    def candidates_per_family_range(self) -> Tuple[int, int]:
        return (self.min_candidates_per_family, self.max_candidates_per_family)


@dataclass(frozen=True)
class RepairCandidateSlot:
    """One planned candidate slot in a diagnosis-directed search budget.

    Attributes:
        diagnosis: Upper-snake diagnosis kind name.
        family: Upper-snake repair-family name.
        max_candidates: Number of candidates this family may propose for the
            diagnosis (within the per-family config bound).
    """

    diagnosis: str
    family: str
    max_candidates: int = 2

    def __post_init__(self) -> None:
        _as_diagnosis(self.diagnosis)
        if self.family not in PLAN_E3_FAMILY_NAMES:
            raise ValueError(
                f"unknown repair family {self.family!r} in candidate slot"
            )
        if (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive int")


@dataclass(frozen=True)
class DiagnosisSearchBudget:
    """A concrete, bounded candidate plan for one factor (plan §20 E5).

    Attributes:
        primary_diagnoses: Ordered upper-snake diagnosis names the search
            will act on (≤ ``max_primary_diagnoses``).
        slots: Candidate slots — one per (diagnosis, family) pair.
        max_total_candidates: Hard ceiling on candidates the search may
            propose for this factor (the soft total budget's upper bound).
        min_total_candidates: Soft floor the generator aims to reach.
        policy_id / policy_version: Versioned policy that produced this
            budget.
    """

    primary_diagnoses: Tuple[str, ...]
    slots: Tuple[RepairCandidateSlot, ...]
    max_total_candidates: int = 12
    min_total_candidates: int = 6
    policy_id: str = "FO_DIAGNOSIS_REPAIR"
    policy_version: str = "1.0.0"

    @property
    def total_candidates_ceiling(self) -> int:
        """Sum of every slot's candidate allowance."""
        return sum(slot.max_candidates for slot in self.slots)

    def within_soft_budget(self) -> bool:
        """True when the plan respects the soft 6-12 total candidate budget.

        An empty plan (no repairable diagnosis, or none supplied) carries zero
        candidates — it is trivially *below* the soft floor, which is not a
        violation (nothing was invented); only an over-ceiling plan violates
        the budget.
        """
        return self.total_candidates_ceiling <= self.max_total_candidates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_diagnoses": list(self.primary_diagnoses),
            "slots": [
                {
                    "diagnosis": s.diagnosis,
                    "family": s.family,
                    "max_candidates": s.max_candidates,
                }
                for s in self.slots
            ],
            "max_total_candidates": self.max_total_candidates,
            "min_total_candidates": self.min_total_candidates,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


def diagnosis_search_budget(
    diagnoses: Sequence[DiagnosisKind | str],
    *,
    policy: Optional[DiagnosisRepairPolicy] = None,
    registry: Optional[RepairFamilyRegistry] = None,
    evidence: Optional[Mapping[str, object]] = None,
    config: Optional[RepairBudgetConfig] = None,
) -> DiagnosisSearchBudget:
    """Generate a concrete candidate plan from diagnoses (plan §20 E5).

    The generator:
    1. takes at most ``max_primary_diagnoses`` diagnoses (preserving input
       order) that actually have an allowed family;
    2. for each primary diagnosis takes at most
       ``max_repair_families_per_diagnosis`` families from the policy rule;
    3. per family caps candidates by the family's declared maximum and the
       config's per-family bound;
    4. enforces the soft total candidate budget 6-12 by trimming per-family
       allowances from the last family backward when the ceiling would exceed
       ``max_total_candidates``.

    A diagnosis with no allowed family is skipped (never invented).  The
    returned plan is bounded and versioned.
    """
    policy = policy or default_diagnosis_repair_policy()
    registry = registry or RepairFamilyRegistry.default()
    config = config or RepairBudgetConfig()

    primary: List[str] = []
    slots: List[RepairCandidateSlot] = []
    for raw in diagnoses:
        if len(primary) >= config.max_primary_diagnoses:
            break
        kind = _as_diagnosis(raw)
        if kind.name in {p for p in primary}:
            continue
        allowed = policy.families_for(kind, evidence=evidence)
        if not allowed:
            continue
        # A diagnosis whose every allowed family is ABANDON proposes no
        # candidates — skip it rather than spending primary-diagnosis budget.
        repairable = [
            family_name
            for family_name in allowed[: config.max_repair_families_per_diagnosis]
            if family_name != "ABANDON"
        ]
        if not repairable:
            continue
        primary.append(kind.name)
        for family_name in repairable:
            declaration = registry.get(family_name)
            per_family = min(
                declaration.maximum_candidates,
                config.max_candidates_per_family,
            )
            per_family = max(per_family, config.min_candidates_per_family)
            slots.append(
                RepairCandidateSlot(
                    diagnosis=kind.name,
                    family=family_name,
                    max_candidates=per_family,
                )
            )
        if len(primary) >= config.max_primary_diagnoses:
            break

    # Enforce the soft total candidate budget: trim the ceiling down to
    # max_total_candidates by reducing the later slots first.
    ceiling = sum(slot.max_candidates for slot in slots)
    if ceiling > config.max_total_candidates:
        trimmed: List[RepairCandidateSlot] = []
        over = ceiling - config.max_total_candidates
        for slot in reversed(slots):
            if over <= 0:
                trimmed.append(slot)
                continue
            reduction = min(slot.max_candidates - config.min_candidates_per_family, over)
            if reduction <= 0:
                trimmed.append(slot)
                continue
            trimmed.append(
                RepairCandidateSlot(
                    diagnosis=slot.diagnosis,
                    family=slot.family,
                    max_candidates=slot.max_candidates - reduction,
                )
            )
            over -= reduction
        # Trim in the original order (slots trimmed from the end, so reverse).
        trimmed.reverse()
        # Drop diagnosis that lost every candidate (ceiling floor still holds
        # via the min_candidates_per_family floor on retained slots).
        slots = [s for s in trimmed if s.max_candidates >= 1]
    # If the ceiling is still above the hard max (possible when per-family
    # minimums alone exceed it), the plan fails closed rather than silently
    # exceeding the budget.
    ceiling = sum(slot.max_candidates for slot in slots)
    if ceiling > config.max_total_candidates:
        raise ValueError(
            "diagnosis_search_budget cannot satisfy the soft total budget: "
            f"per-family minimums already sum to {ceiling} > "
            f"{config.max_total_candidates}; raise max_total_candidates or "
            "lower per-family minimums"
        )
    return DiagnosisSearchBudget(
        primary_diagnoses=tuple(primary),
        slots=tuple(slots),
        max_total_candidates=config.max_total_candidates,
        min_total_candidates=config.min_total_candidates,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
    )
