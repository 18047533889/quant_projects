"""N11/N14 bridge: accepted replayable OOF set trial becomes manifest evidence."""
from quant_platform.app.modeling_manifest_bridge import ModelFeatureCompatibilityEvidence
from modeling.feature_set_trials import (FeatureSetTrialEvidence, load_frozen_model_ref,
                                          validate_feature_set_trial_evidence)


def compatibility_from_set_trial(trial, *, model_artifact_ref: str):
    if not isinstance(trial, FeatureSetTrialEvidence):
        raise TypeError("trial must be FeatureSetTrialEvidence produced by governed retraining")
    validate_feature_set_trial_evidence(trial)
    if not trial.accepted:
        raise ValueError("set-level OOF trial was not accepted")
    if not trial.model_refs or not trial.baseline_model_refs or not trial.oof_row_ids:
        raise ValueError("real OOF models and rows are required")
    for ref in (*trial.baseline_model_refs, *trial.model_refs):
        load_frozen_model_ref(ref)
    load_frozen_model_ref(model_artifact_ref)
    if model_artifact_ref not in trial.model_refs:
        raise ValueError("model_artifact_ref must be one of this trial's persisted candidate models")
    return ModelFeatureCompatibilityEvidence(
        trial.evidence_ref, model_artifact_ref, trial.action.feature_set_version_ref, "RETRAINED"
    )


__all__ = ["compatibility_from_set_trial"]
