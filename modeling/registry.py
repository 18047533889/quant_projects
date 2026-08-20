# -*- coding: utf-8 -*-
"""Model registry (Model Layer Major Redesign taskbook §48).

A thread-safe registry of predictive supervised models: each entry binds a
model name to its execution class, learner class, walk-forward training spec
and per-parameter search policies.  ``register_default_learners`` registers the
five built-in learners with the §3.3 presets and the §69 parameter policies.
"""
from __future__ import annotations

import threading
from typing import Any

from modeling.contracts import ModelExecutionClass, ParameterSearchPolicy
from modeling.walk_forward import WalkForwardSpec

__all__ = ["ModelRegistry", "MODEL_REGISTRY", "register_default_learners"]


def _preset_to_spec(preset: Any) -> WalkForwardSpec:
    if preset.train_mode == "rolling":
        train_expanding, decay = False, None
    elif preset.train_mode == "expanding":
        train_expanding, decay = True, None
    elif preset.train_mode == "decay_weighted_expanding":
        train_expanding, decay = True, preset.decay_half_life_bars
    else:
        raise ValueError(f"unknown train_mode {preset.train_mode!r}")
    purge_policy = "label_interval" if preset.purge_by_label_interval else "purge_bars"
    return WalkForwardSpec(
        train_lookback_bars=preset.train_lookback_bars,
        train_expanding=train_expanding,
        validation_bars=preset.validation_bars,
        test_bars=preset.test_bars,
        step_bars=preset.step_bars,
        retrain_every_bars=preset.retrain_every_bars,
        purge_policy=purge_policy,
        embargo_bars=preset.embargo_bars,
        min_train_dates=preset.min_train_dates,
        min_train_stocks=preset.min_train_stocks,
        min_train_obs=preset.min_train_obs,
        decay_half_life_bars=decay,
    )


class ModelRegistry:
    """Thread-safe model registry (§48)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}

    def register(
        self,
        model_name: str,
        execution_class: ModelExecutionClass,
        learner_cls: type,
        training_spec: WalkForwardSpec | None = None,
        parameter_policies: dict[str, ParameterSearchPolicy] | None = None,
    ) -> None:
        with self._lock:
            if model_name in self._records:
                raise ValueError(f"model {model_name!r} already registered")
            self._records[model_name] = {
                "model_name": model_name,
                "execution_class": execution_class,
                "learner_cls": learner_cls,
                "training_spec": training_spec,
                "parameter_policies": dict(parameter_policies or {}),
            }

    def get(self, model_name: str) -> dict[str, Any]:
        with self._lock:
            if model_name not in self._records:
                raise KeyError(f"model {model_name!r} not registered")
            return self._records[model_name]

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._records)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._records

    def registered_predictive_learners(self) -> list[str]:
        with self._lock:
            return sorted(
                name
                for name, rec in self._records.items()
                if rec["execution_class"] == ModelExecutionClass.PREDICTIVE_SUPERVISED
            )


MODEL_REGISTRY = ModelRegistry()


def register_default_learners() -> None:
    """Register the five built-in predictive learners (idempotent)."""
    from modeling.learners import (
        ElasticNetLearner,
        MixtureOfExpertsLearner,
        PCRLearner,
        PLSLearner,
        RegimeLearner,
    )
    from modeling.presets import (
        PREDICTIVE_LINEAR_DEFAULT,
        PREDICTIVE_MOE_DEFAULT,
        PREDICTIVE_REGIME_DEFAULT,
        recommendation_map,
    )

    rec_map = recommendation_map()

    def _policies(family: str) -> dict[str, ParameterSearchPolicy]:
        policies: dict[str, ParameterSearchPolicy] = {}
        for (fam, param), rec in rec_map.items():
            if fam == family:
                policies[param] = ParameterSearchPolicy(
                    role=rec["role"], searchable=bool(rec["searchable_by_miner"])
                )
        return policies

    entries = [
        (PCRLearner, PREDICTIVE_LINEAR_DEFAULT),
        (PLSLearner, PREDICTIVE_LINEAR_DEFAULT),
        (ElasticNetLearner, PREDICTIVE_LINEAR_DEFAULT),
        (RegimeLearner, PREDICTIVE_REGIME_DEFAULT),
        (MixtureOfExpertsLearner, PREDICTIVE_MOE_DEFAULT),
    ]
    for learner_cls, preset in entries:
        if MODEL_REGISTRY.has(learner_cls.name):
            continue
        MODEL_REGISTRY.register(
            learner_cls.name,
            ModelExecutionClass.PREDICTIVE_SUPERVISED,
            learner_cls,
            training_spec=_preset_to_spec(preset),
            parameter_policies=_policies(learner_cls.family),
        )
