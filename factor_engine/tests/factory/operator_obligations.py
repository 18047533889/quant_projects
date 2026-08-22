# -*- coding: utf-8 -*-
"""R35 §60 / §186: operator test obligation matrix (automatic).

Each operator automatically gets a TestObligation with the dimensions its
RiskProfile requires:
- elementwise   : semantic, edge, parameter, backend, batch
- rolling TS    : + future_perturb, chunk, incremental
- source-bound  : + source_pit, revision
- model         : + timing, label_maturity, oracle, convergence
- stateful model: + checkpoint

Every model-like canonical additionally requires the model obligations (§61):
MODEL_ORACLE / FIT_CUTOFF / FUTURE_PERTURBATION / LABEL_MATURITY /
SCALER_HISTORY_ONLY / HYPERPARAM_HISTORY_ONLY / CONDITION_NUMBER /
NON_CONVERGENCE / MISSING_PATTERN / PARAM_DOMAIN / DETERMINISM.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

__all__ = [
    "TestObligation",
    "RiskProfile",
    "obligations_for",
    "model_obligations",
    "OBLIGATION_DIMENSIONS",
]

OBLIGATION_DIMENSIONS: tuple[str, ...] = (
    "execution",
    "semantic_golden",
    "parameter_domain",
    "edge",
    "temporal",
    "source_pit",
    "backend_parity",
    "batch_parity",
    "chunk",
    "incremental",
    "concurrency",
    "performance",
)


@dataclass(frozen=True)
class TestObligation:
    canonical: str
    risk_profile: str          # elementwise / rolling_ts / source_bound / model / stateful_model
    required: frozenset[str] = field(default_factory=frozenset)
    model_required: frozenset[str] = field(default_factory=frozenset)

    @property
    def dimensions(self) -> list[str]:
        return sorted(self.required)


@dataclass(frozen=True)
class RiskProfile:
    elementwise: bool = False
    rolling_ts: bool = False
    source_bound: bool = False
    model: bool = False
    stateful: bool = False


def _base_obligations() -> frozenset[str]:
    return frozenset(
        {"execution", "semantic_golden", "parameter_domain", "edge", "backend_parity", "batch_parity"}
    )


def obligations_for(
    canonical: str,
    *,
    risk: RiskProfile | None = None,
    is_model: bool = False,
) -> TestObligation:
    """Build the obligation for a canonical from its risk profile."""
    profile = risk or RiskProfile()
    req = set(_base_obligations())
    if profile.rolling_ts:
        req |= {"temporal", "chunk", "incremental", "concurrency"}
    if profile.source_bound:
        req |= {"source_pit"}
    if profile.stateful:
        req |= {"checkpoint"}
    if profile.model or is_model:
        req |= {"temporal", "chunk", "incremental"}
    # every obligation includes performance (measured, not correctness gating)
    req.add("performance")
    return TestObligation(
        canonical=canonical,
        risk_profile="model" if (profile.model or is_model) else
                     "stateful_model" if profile.stateful else
                     "rolling_ts" if profile.rolling_ts else
                     "source_bound" if profile.source_bound else "elementwise",
        required=frozenset(req),
        model_required=frozenset(model_obligations()) if (profile.model or is_model) else frozenset(),
    )


def model_obligations() -> tuple[str, ...]:
    """§61: the required extra test dimensions for every model-like canonical."""
    return (
        "MODEL_ORACLE",
        "FIT_CUTOFF",
        "FUTURE_PERTURBATION",
        "LABEL_MATURITY",
        "SCALER_HISTORY_ONLY",
        "HYPERPARAM_HISTORY_ONLY",
        "CONDITION_NUMBER",
        "NON_CONVERGENCE",
        "MISSING_PATTERN",
        "PARAM_DOMAIN",
        "DETERMINISM",
    )
