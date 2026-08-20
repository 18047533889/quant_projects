# -*- coding: utf-8 -*-
"""Base learner interface + learner registry (Model Layer Major Redesign
taskbook §36–§40, §48).

A predictive learner is a **pooled-panel, multi-year, walk-forward, artifact-
backed** model.  The core contract:

    frozen = learner.fit(X_train, y_train, ...)     # once, on train only
    pred   = learner.predict(frozen, X)             # repeated, frozen only

Production scoring NEVER calls ``fit`` (§22).  Learners are registered in
:data:`LEARNER_REGISTRY` (the Model Registry, §48) with their sample-adequacy
and parameter-search contracts so the hard gates can verify them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from modeling.contracts import (
    ModelExecutionClass,
    ParameterSearchPolicy,
    ParamRole,
    SampleAdequacyContract,
)

__all__ = [
    "LearnerSpec",
    "FrozenModel",
    "BaseLearner",
    "ModelConvergenceError",
    "LEARNER_REGISTRY",
    "register_learner",
    "get_learner",
    "learner_param_policy",
    "default_sample_contracts",
]


class ModelConvergenceError(ValueError):
    """Raised when an iterative learner exhausts ``max_iter`` without converging.

    A non-converged iterate is NEVER returned as a frozen model — the fit fails
    closed so callers cannot silently consume a degraded artifact.  Subclasses
    :class:`ValueError` so callers that already treat fit failures as value
    errors keep working.
    """


@dataclass(frozen=True)
class LearnerSpec:
    """Model specification: which learner, with which (certified) hyperparams."""

    learner_name: str
    family: str
    hyperparams: dict[str, Any] = field(default_factory=dict)
    sample_contract: SampleAdequacyContract | None = None
    parameter_policies: dict[str, ParameterSearchPolicy] = field(default_factory=dict)
    random_seed: int | None = None


@dataclass(frozen=True)
class FrozenModel:
    """The frozen fitted model + fit telemetry.  Immutable — never re-fit."""

    learner_name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, learner: "BaseLearner", X: np.ndarray) -> np.ndarray:
        return learner.predict(self, X)


class _ContractFamilyDescriptor:
    """Dual-context accessor for :attr:`BaseLearner.contract_family`.

    ``modeling.sample_policy.resolve_sample_contract`` reads the attribute with
    ``getattr(learner, "contract_family", None)`` and treats the value as a
    *string key* in an ``or`` chain (``sample_contract_family or
    contract_family or family``).  A plain method or ``@property`` would surface
    as a truthy bound-method / descriptor object at class level and hijack the
    chain.  This descriptor returns:

    * the resolved key string on an **instance** (``learner.contract_family``);
    * ``""`` on **class** access — falsy, so ``resolve_sample_contract`` falls
      through to the bare ``family`` key for learners that declare no explicit
      ``sample_contract_family`` (preserving the legacy fallback).
    """

    def __get__(self, obj: Any, objtype: Any = None) -> str:
        if obj is None:  # class-level access via ``getattr(cls, ...)``
            return ""
        return obj.sample_contract_family or obj.family


class BaseLearner(ABC):
    """Pooled-panel predictive learner interface.

    ``sample_contract_family`` names the §5 default contract this learner is
    bound to (``"linear"`` for pcr/pls/elastic_net, ``"regime"``, ``"moe"``).
    The trainer resolves ``default_sample_contracts().get(learner.sample_contract_family)``
    — the explicit declaration fixes the bug where ``family`` keys
    (pcr/pls/elastic_net) silently missed the linear contract.
    """

    name: str = ""
    family: str = ""
    sample_contract_family: str = ""
    execution_class: ModelExecutionClass = ModelExecutionClass.PREDICTIVE_SUPERVISED

    def __init__(self, spec: LearnerSpec) -> None:
        if not self.name:
            self.name = spec.learner_name
        if not self.family:
            self.family = spec.family
        self.spec = spec

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        aux: dict[str, np.ndarray | None] | None = None,
    ) -> FrozenModel:
        """Fit on training observations only.  Raises ``ValueError`` (fail
        closed) when the sample does not satisfy the learner's adequacy
        contract, never silently returning a degraded model.

        ``aux`` carries per-row auxiliary arrays the learner may need (e.g.
        ``{"regime_state": <1-D array>}`` for regime/MoE gating).  The trainer
        extracts these from the panel's context columns; a learner that does
        not need them ignores the argument."""

    @abstractmethod
    def predict(self, frozen: FrozenModel, X: np.ndarray) -> np.ndarray:
        """Score from a frozen model.  MUST NOT call ``fit``."""

    def required_feature_count(self) -> int:
        return 1

    def validate_params(self) -> None:
        """Reject out-of-domain hyperparameters at call boundary (fail closed)."""

    contract_family = _ContractFamilyDescriptor()

    def effective_parameter_count(
        self,
        hyperparams: dict[str, Any] | None = None,
        n_features: int = 0,
        *,
        n_rows: int | None = None,
        **extra: Any,
    ) -> int:
        """True free-parameter count of the fitted model, used for the §5
        ``obs/parameter`` gate.

        ``len(hyperparams)`` is NOT a valid proxy (an ElasticNet with 50
        features and ``{alpha, l1_ratio}`` fits ~51 coefficients).  Each learner
        overrides this with its real complexity.

        The positional ``(hyperparams, n_features)`` call remains accepted for
        the trainer's pre-fit adequacy gate; ``n_rows`` and ``extra`` are
        keyword-only refinements that let callers pass the observed panel size
        and the frozen model's fitted structure (e.g. ``coef`` for ElasticNet).

        Default (conservative) estimate: a saturated linear model
        ``n_features + 1``, capped at ~1 parameter per 10 training rows when
        ``n_rows`` is supplied.
        """
        nf = max(1, int(n_features) or 1)
        if n_rows:
            return min(nf + 1, max(1, int(n_rows) // 10))
        return nf + 1

    # -- sample-adequacy -----------------------------------------------------
    def check_adequacy(self, telemetry: dict[str, Any], free_parameters: int) -> tuple[bool, list[str]]:
        """Gate a training run against the learner's SampleAdequacyContract.

        Every contract field with a measured telemetry value is enforced:
        raw/effective obs, unique dates/stocks, obs-per-parameter,
        cross-section peers, missing fraction and date coverage.  A field that
        CAN be measured but is not supplied (e.g. missing_fraction) is reported
        as a failure rather than silently skipped, so the gate never degrades.
        """
        contract = self.spec.sample_contract
        if contract is None:
            return True, []
        from modeling.contracts import sample_adequacy_met

        raw = int(telemetry.get("raw_obs", 0))
        effective = int(telemetry.get("finite_obs", telemetry.get("effective_obs", 0)))
        missing_fraction = None
        if raw:
            missing_fraction = 1.0 - effective / raw
        failures: list[str] = []
        # A contract field the telemetry CAN supply must be supplied; a learner
        # that cannot measure a REQUIRED field must fail closed (never skip).
        unmeasured: list[str] = []
        if contract.max_missing_fraction is not None and missing_fraction is None:
            unmeasured.append("missing_fraction")
        if contract.min_date_coverage is not None and telemetry.get("date_coverage") is None:
            unmeasured.append("date_coverage")
        if contract.min_cross_section_peers is not None and telemetry.get("median_stocks_per_date") is None:
            unmeasured.append("cross_section_peers")
        if unmeasured:
            failures.append(
                "cannot measure required adequacy fields: " + ", ".join(unmeasured)
            )

        met, field_failures = sample_adequacy_met(
            contract=contract,
            raw_obs=raw,
            effective_obs=effective,
            unique_dates=int(telemetry.get("n_unique_dates", 0)),
            unique_stocks=int(telemetry.get("n_unique_stocks", 0)),
            free_parameter_count=free_parameters,
            regime_obs=telemetry.get("regime_obs"),
            expert_obs=telemetry.get("expert_obs"),
            state_transitions=telemetry.get("state_transitions"),
            cross_section_peers=telemetry.get("median_stocks_per_date"),
            missing_fraction=missing_fraction,
            date_coverage=telemetry.get("date_coverage"),
        )
        return (met and not failures), field_failures + failures


# --------------------------------------------------------------------------- #
# Learner registry (§48) — the Model Registry for predictive supervised models.
# --------------------------------------------------------------------------- #
LEARNER_REGISTRY: dict[str, type[BaseLearner]] = {}


def register_learner(cls: type[BaseLearner]) -> type[BaseLearner]:
    name = cls.name or cls.__name__.lower()
    if name in LEARNER_REGISTRY:
        raise ValueError(f"learner {name!r} already registered")
    LEARNER_REGISTRY[name] = cls
    return cls


def get_learner(name: str) -> type[BaseLearner]:
    if name not in LEARNER_REGISTRY:
        raise KeyError(f"unknown learner {name!r}; registered: {sorted(LEARNER_REGISTRY)}")
    return LEARNER_REGISTRY[name]


def learner_param_policy(
    policies: dict[str, ParameterSearchPolicy], param: str
) -> ParameterSearchPolicy | None:
    """Search policy for one parameter of a learner (or None if ungoverned)."""
    return policies.get(param)


#: §6.2 / §6.1 enterprise default sample-adequacy floors.
def default_sample_contracts() -> dict[str, SampleAdequacyContract]:
    linear = SampleAdequacyContract(
        min_raw_obs=50_000,
        min_effective_obs=10_000,
        min_unique_dates=252,
        min_unique_stocks=30,
        min_obs_per_parameter=200.0,
        max_missing_fraction=0.30,
        min_date_coverage=0.5,
    )
    regime = SampleAdequacyContract(
        min_raw_obs=100_000,
        min_effective_obs=50_000,
        min_unique_dates=504,
        min_unique_stocks=30,
        min_obs_per_parameter=300.0,
        min_regime_obs=5_000,
        min_state_transitions=12,
        max_missing_fraction=0.30,
        min_date_coverage=0.5,
    )
    moe = SampleAdequacyContract(
        min_raw_obs=200_000,
        min_effective_obs=100_000,
        min_unique_dates=504,
        min_unique_stocks=30,
        min_obs_per_parameter=300.0,
        min_expert_obs=1_000,
        max_missing_fraction=0.30,
        min_date_coverage=0.5,
    )
    # Canonical family names map onto the three contract tiers.  The trainer
    # resolves via ``learner.sample_contract_family`` (explicit declaration),
    # so PCR/PLS/ENet all land on the linear contract — never None.
    return {
        "linear": linear,
        "pcr": linear,
        "pls": linear,
        "elastic_net": linear,
        "regime": regime,
        "moe": moe,
    }
