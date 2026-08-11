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
    "LEARNER_REGISTRY",
    "register_learner",
    "get_learner",
    "learner_param_policy",
    "default_sample_contracts",
]


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


class BaseLearner(ABC):
    """Pooled-panel predictive learner interface."""

    name: str = ""
    family: str = ""
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

    # -- sample-adequacy -----------------------------------------------------
    def check_adequacy(self, telemetry: dict[str, Any], free_parameters: int) -> tuple[bool, list[str]]:
        """Gate a training run against the learner's SampleAdequacyContract."""
        contract = self.spec.sample_contract
        if contract is None:
            return True, []
        from modeling.contracts import sample_adequacy_met

        return sample_adequacy_met(
            contract=contract,
            raw_obs=int(telemetry.get("raw_obs", 0)),
            effective_obs=int(telemetry.get("finite_obs", 0)),
            unique_dates=int(telemetry.get("n_unique_dates", 0)),
            unique_stocks=int(telemetry.get("n_unique_stocks", 0)),
            free_parameter_count=free_parameters,
            expert_obs=None,
            regime_obs=None,
        )


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
    return {"linear": linear, "regime": regime, "moe": moe}
