"""
Neutralization specification contract for the auto-treatment optimizer.

A :class:`NeutralizationSpec` is the *semantic recipe* for neutralizing a
factor. The generic ``ols_neutralize`` kernel is the computational engine, but
an explicit spec is what makes a neutralization reproducible and auditable.

Admission policy
----------------
- ``OLS``, ``RIDGE``, ``HUBER`` are the production-admissible methods.
- Advanced methods (``pca``, ``kernel``, ``quantile``, ``lad``) are
  RESEARCH/STAGING only and CANNOT auto-admit to production.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class NeutralizationMethod(str, Enum):
    """Neutralization regression method."""

    OLS = "ols"
    RIDGE = "ridge"
    HUBER = "huber"
    # Research/staging only — cannot auto-admit to production.
    PCA = "pca"
    KERNEL = "kernel"
    QUANTILE = "quantile"
    LAD = "lad"


class ExposureSet(str, Enum):
    """The set of exposures to neutralize against."""

    NONE = "none"
    INDUSTRY = "industry"
    SIZE = "size"
    INDUSTRY_SIZE = "industry_size"
    INDUSTRY_SIZE_BETA = "industry_size_beta"
    CUSTOM_STYLE_SET = "custom_style_set"


class Standardization(str, Enum):
    """How exposures are standardized before fitting."""

    NONE = "none"
    ZSCORE = "zscore"
    RANK = "rank"


class ConditionNumberPolicy(str, Enum):
    """How ill-conditioned exposure matrices are handled."""

    FAIL = "fail"
    WARN = "warn"
    REGULARIZE = "regularize"


class PitIdentity(str, Enum):
    """PIT (point-in-time) identity of the exposures used."""

    PIT = "pit"
    AS_OF = "as_of"
    RESTATED = "restated"


# Methods that are production-admissible.
_PRODUCTION_METHODS = {
    NeutralizationMethod.OLS,
    NeutralizationMethod.RIDGE,
    NeutralizationMethod.HUBER,
}

# Advanced methods that are research/staging only.
_RESEARCH_METHODS = {
    NeutralizationMethod.PCA,
    NeutralizationMethod.KERNEL,
    NeutralizationMethod.QUANTILE,
    NeutralizationMethod.LAD,
}


@dataclass(frozen=True)
class NeutralizationSpec:
    """Semantic recipe for a neutralization treatment.

    Parameters
    ----------
    method : NeutralizationMethod
        Regression method. OLS/RIDGE/HUBER are production-admissible; the
        advanced methods are research/staging only.
    exposure_set : ExposureSet
        Which exposures to neutralize against.
    industry_schema : str, optional
        Industry classification schema (e.g. ``SW_L1``, ``GICS_L2``).
    size_definition : str, optional
        Size proxy definition (e.g. ``log_mktcap``).
    standardization : Standardization
        How exposures are standardized before fitting.
    weights : dict, optional
        Observation weights (e.g. ``{"scheme": "sqrt_mktcap"}``).
    min_obs : int
        Minimum valid observations per date to fit.
    condition_number_policy : ConditionNumberPolicy
        How to handle ill-conditioned exposure matrices.
    pit_identity : PitIdentity
        PIT identity of the exposures used.
    """

    method: NeutralizationMethod = NeutralizationMethod.OLS
    exposure_set: ExposureSet = ExposureSet.INDUSTRY
    industry_schema: Optional[str] = None
    size_definition: Optional[str] = None
    standardization: Standardization = Standardization.ZSCORE
    weights: Dict[str, Any] = field(default_factory=dict)
    min_obs: int = 10
    condition_number_policy: ConditionNumberPolicy = ConditionNumberPolicy.WARN
    pit_identity: PitIdentity = PitIdentity.PIT

    def __post_init__(self):
        if not isinstance(self.method, NeutralizationMethod):
            object.__setattr__(self, "method", NeutralizationMethod(self.method))
        if not isinstance(self.exposure_set, ExposureSet):
            object.__setattr__(self, "exposure_set", ExposureSet(self.exposure_set))
        if not isinstance(self.standardization, Standardization):
            object.__setattr__(
                self, "standardization", Standardization(self.standardization)
            )
        if not isinstance(self.condition_number_policy, ConditionNumberPolicy):
            object.__setattr__(
                self,
                "condition_number_policy",
                ConditionNumberPolicy(self.condition_number_policy),
            )
        if not isinstance(self.pit_identity, PitIdentity):
            object.__setattr__(self, "pit_identity", PitIdentity(self.pit_identity))
        if self.min_obs < 1:
            raise ValueError("min_obs must be >= 1")
        object.__setattr__(self, "weights", dict(self.weights))

        # Industry/size exposure sets require the corresponding schema.
        if self.exposure_set in (
            ExposureSet.INDUSTRY,
            ExposureSet.INDUSTRY_SIZE,
            ExposureSet.INDUSTRY_SIZE_BETA,
        ) and not self.industry_schema:
            raise ValueError(
                f"exposure_set {self.exposure_set.value} requires industry_schema"
            )
        if self.exposure_set in (
            ExposureSet.SIZE,
            ExposureSet.INDUSTRY_SIZE,
            ExposureSet.INDUSTRY_SIZE_BETA,
        ) and not self.size_definition:
            raise ValueError(
                f"exposure_set {self.exposure_set.value} requires size_definition"
            )

    @property
    def is_production_admissible(self) -> bool:
        """True if the method can auto-admit to production."""
        return self.method in _PRODUCTION_METHODS

    @property
    def is_research_only(self) -> bool:
        """True if the method is research/staging only."""
        return self.method in _RESEARCH_METHODS

    def kernel_name(self) -> str:
        """Return the underlying computational kernel name.

        The generic ``ols_neutralize`` is the kernel for OLS; RIDGE/HUBER map
        to their respective kernels. Advanced methods map to their research
        kernels.
        """
        return {
            NeutralizationMethod.OLS: "ols_neutralize",
            NeutralizationMethod.RIDGE: "ridge_neutralize",
            NeutralizationMethod.HUBER: "huber_neutralize",
            NeutralizationMethod.PCA: "pca_neutralize",
            NeutralizationMethod.KERNEL: "kernel_neutralize",
            NeutralizationMethod.QUANTILE: "quantile_neutralize",
            NeutralizationMethod.LAD: "lad_neutralize",
        }[self.method]


__all__ = [
    "NeutralizationMethod",
    "ExposureSet",
    "Standardization",
    "ConditionNumberPolicy",
    "PitIdentity",
    "NeutralizationSpec",
]
