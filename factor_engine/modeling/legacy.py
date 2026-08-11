# -*- coding: utf-8 -*-
"""Legacy local predictive classification (§2.1 / §67).

The five short-window per-stock rolling operators are NOT deleted — DSL and
historical factor identity must be preserved.  They are reclassified as
``LOCAL_ROLLING_ESTIMATOR``, ``default_searchable=False`` and
``research_only=True`` via this additive registry.  The registry is a single
authority consumed by:

* :func:`classification_of` — per-canonical legacy metadata;
* ``scripts/generate_model_layer_redesign_evidence.py`` — the classification
  ledger; and
* the hard-gate ``MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH``
  (these operators are legacy-local, NOT artifact-backed predictive learners).

The registry intentionally does not edit the concurrent-shared
``cleaned_operators/cross_section/panel_model.py`` registrations.
"""
from __future__ import annotations

from modeling.contracts import LegacyLocalPredictive, ModelExecutionClass

__all__ = [
    "LEGACY_LOCAL_PREDICTIVE_CANONICALS",
    "classification_of",
    "classify_execution_class",
    "ALL_PREDICTIVE_LEARNER_CANONICALS",
]

#: The §2.1 legacy local rolling predictive operators.
LEGACY_LOCAL_PREDICTIVE_CANONICALS: dict[str, LegacyLocalPredictive] = {
    "panel_rolling_pcr_forecast": LegacyLocalPredictive(
        canonical="panel_rolling_pcr_forecast",
        min_effective_obs=None,   # floor = max(60, 10 * free_params)
        min_obs_per_parameter=10.0,
    ),
    "panel_rolling_pls_forecast": LegacyLocalPredictive(
        canonical="panel_rolling_pls_forecast",
        min_effective_obs=None,
        min_obs_per_parameter=10.0,
    ),
    "panel_rolling_elastic_net_forecast": LegacyLocalPredictive(
        canonical="panel_rolling_elastic_net_forecast",
        min_effective_obs=None,
        min_obs_per_parameter=10.0,
    ),
    "panel_regime_conditioned_forecast": LegacyLocalPredictive(
        canonical="panel_regime_conditioned_forecast",
        min_effective_obs=None,
        min_obs_per_parameter=10.0,
        min_regime_obs=30,
    ),
    "panel_mixture_of_experts_score": LegacyLocalPredictive(
        canonical="panel_mixture_of_experts_score",
        min_effective_obs=None,
        min_obs_per_parameter=10.0,
        min_expert_obs=30,
    ),
}


def classification_of(canonical: str) -> LegacyLocalPredictive | None:
    """Legacy metadata for a canonical, or ``None`` if it is not a legacy local
    predictive variant."""
    return LEGACY_LOCAL_PREDICTIVE_CANONICALS.get(canonical)


def classify_execution_class(canonical: str) -> ModelExecutionClass:
    """Execution-class classification used by the classification ledger.

    Priority:
    1. legacy local predictive (this registry) -> LOCAL_ROLLING_ESTIMATOR
    2. explicit mapping in :data:`MODEL_OPERATOR_EXECUTION_CLASS`
    3. heuristic: forecast/regime/moe/pca-family -> LOCAL_ROLLING_ESTIMATOR;
       kalman/cusum/markov/stateful -> RECURSIVE_STATE_ESTIMATOR;
       knn/peer/cs_/group cross-sectional -> SAME_TIME_CROSS_SECTIONAL;
       dmd/ssa/rqa/lyapunov/te/hsic/entropy/spectral -> RESEARCH_STRUCTURAL;
       default -> LOCAL_ROLLING_ESTIMATOR.
    """
    if canonical in LEGACY_LOCAL_PREDICTIVE_CANONICALS:
        return ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR
    explicit = MODEL_OPERATOR_EXECUTION_CLASS.get(canonical)
    if explicit is not None:
        return explicit
    name = canonical.lower()
    if any(tok in name for tok in ("kalman", "cusum", "markov", "state_filter", "latch")):
        return ModelExecutionClass.RECURSIVE_STATE_ESTIMATOR
    if any(
        tok in name
        for tok in ("dmd", "ssa", "rqa", "recurrence", "lyapunov", "transfer_entropy",
                    "hsic", "granger", "entropy", "spectral", "multifractal", "wavelet")
    ):
        return ModelExecutionClass.RESEARCH_STRUCTURAL
    if name.startswith("cs_") or "peer" in name or "knn" in name or "cross_section" in name:
        return ModelExecutionClass.SAME_TIME_CROSS_SECTIONAL
    return ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR


#: Explicit execution-class overrides (beyond the legacy registry + heuristics).
MODEL_OPERATOR_EXECUTION_CLASS: dict[str, ModelExecutionClass] = {}


#: Artifact-backed predictive learners registered in the modeling registry.
#: Populated at runtime by :func:`modeling.registry.registered_predictive_learners`.
def _default_learner_set() -> tuple[str, ...]:
    return (
        "predictive_pcr",
        "predictive_pls",
        "predictive_elastic_net",
        "predictive_regime",
        "predictive_mixture_of_experts",
    )


ALL_PREDICTIVE_LEARNER_CANONICALS: frozenset[str] = frozenset(_default_learner_set())
