"""Governed, paired OOF trials for whole feature-set actions (V5 N11)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from modeling.learners.base import BaseLearner, FrozenModel
from modeling.trainer_governance import FeatureExperimentSpec


@dataclass(frozen=True)
class FeatureSetAction:
    action_id: str
    operation: str
    resulting_features: tuple[str, ...]
    feature_set_version_ref: str
    cluster_ref: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "resulting_features", tuple(self.resulting_features))
        if self.operation not in {"ADD", "REPLACE", "DROP_CLUSTER"}:
            raise ValueError("operation must be ADD, REPLACE, or DROP_CLUSTER")
        if not self.action_id or not self.feature_set_version_ref or not self.resulting_features:
            raise ValueError("action identity, feature-set ref, and resulting features are required")
        if len(set(self.resulting_features)) != len(self.resulting_features):
            raise ValueError("resulting features must be unique")
        if self.operation == "DROP_CLUSTER" and not self.cluster_ref:
            raise ValueError("DROP_CLUSTER requires cluster_ref")


@dataclass(frozen=True)
class FeatureSetTrialEvidence:
    action: FeatureSetAction
    baseline_score: float
    candidate_score: float
    paired_delta: float
    oof_row_ids: tuple[str, ...]
    fold_refs: tuple[str, ...]
    experiment_intent_hash: str
    min_delta: float
    baseline_model_refs: tuple[str, ...]
    model_refs: tuple[str, ...]
    evidence_ref: str
    accepted: bool


def validate_feature_set_trial_evidence(trial: FeatureSetTrialEvidence) -> None:
    if not isinstance(trial, FeatureSetTrialEvidence):
        raise TypeError("trial must be FeatureSetTrialEvidence")
    payload = {"action": trial.action.__dict__, "spec": trial.experiment_intent_hash,
               "rows": trial.oof_row_ids, "baseline_score": trial.baseline_score,
               "candidate_score": trial.candidate_score, "delta": trial.paired_delta,
               "min_delta": trial.min_delta, "accepted": trial.accepted,
               "baseline_models": trial.baseline_model_refs, "models": trial.model_refs}
    expected = "feature-set-trial:" + sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    if expected != trial.evidence_ref:
        raise ValueError("feature-set trial evidence hash mismatch")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype), "shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"frozen model contains non-serializable value {type(value).__name__}")


def load_frozen_model_ref(model_ref: str) -> FrozenModel:
    """Resolve and integrity-check a content-addressed OOF fold model."""
    prefix = "frozen-model:"
    if not isinstance(model_ref, str) or not model_ref.startswith(prefix):
        raise ValueError("invalid frozen model ref")
    digest, path = model_ref[len(prefix):].split(":", 1)
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("frozen model artifact hash mismatch")
    payload = json.loads(raw)
    def restore(v):
        if isinstance(v, dict) and "__ndarray__" in v:
            arr = np.asarray(v["__ndarray__"], dtype=v["dtype"])
            if tuple(arr.shape) != tuple(v["shape"]):
                raise ValueError("frozen model array shape mismatch")
            return arr
        if isinstance(v, dict): return {k: restore(x) for k, x in v.items()}
        if isinstance(v, list): return [restore(x) for x in v]
        return v
    return FrozenModel(payload["learner_name"], payload["family"],
                       restore(payload["params"]), restore(payload["metadata"]))


def _score(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 2 or np.std(pred[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(pred[mask], y[mask])[0, 1])


def run_feature_set_trials(
    *, X_by_feature: Mapping[str, Sequence[float]], y: Sequence[float],
    row_ids: Sequence[str], baseline_features: tuple[str, ...], actions: Sequence[FeatureSetAction],
    folds: Sequence[tuple[Sequence[int], Sequence[int]]], experiment_spec: FeatureExperimentSpec,
    learner_factory: Callable[[int], BaseLearner], budget_tracker, min_delta: float = 0.0,
    artifact_store_dir: str | Path | None = None,
) -> tuple[FeatureSetTrialEvidence, ...]:
    """Retrain baseline and every action on identical folds and OOF rows."""
    if len(folds) != len(experiment_spec.fold_refs):
        raise ValueError("folds must match frozen fold_refs")
    if not np.isfinite(min_delta):
        raise ValueError("min_delta must be finite")
    if artifact_store_dir is None:
        raise ValueError("artifact_store_dir is required for replayable OOF model evidence")
    artifact_dir = Path(artifact_store_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=float)
    row_ids = tuple(map(str, row_ids))
    if len(y) != len(row_ids):
        raise ValueError("y and row_ids must align")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique")
    if any(len(np.asarray(v)) != len(y) for v in X_by_feature.values()):
        raise ValueError("all features must align with y")
    required_methods = ("reserve_evaluation", "start_evaluation", "commit_evaluation", "release_evaluation")
    if any(not callable(getattr(budget_tracker, m, None)) for m in required_methods):
        raise TypeError("durable budget tracker is required")

    baseline_set = set(baseline_features)
    if not baseline_features or len(baseline_set) != len(baseline_features):
        raise ValueError("baseline_features must be non-empty and unique")
    if any(f not in X_by_feature for f in baseline_features):
        raise ValueError("baseline feature is unavailable")
    seen_valid: set[int] = set()
    checked_folds = []
    for train_raw, valid_raw in folds:
        train_input, valid_input = np.asarray(train_raw), np.asarray(valid_raw)
        if (not np.issubdtype(train_input.dtype, np.integer)
                or not np.issubdtype(valid_input.dtype, np.integer)):
            raise ValueError("fold indices must have integer dtype")
        train, valid = train_input.astype(int, copy=False), valid_input.astype(int, copy=False)
        if train.ndim != 1 or valid.ndim != 1 or not len(train) or not len(valid):
            raise ValueError("each fold requires non-empty one-dimensional train/valid indices")
        if len(set(train.tolist())) != len(train) or len(set(valid.tolist())) != len(valid):
            raise ValueError("fold indices must be unique")
        if np.any(train < 0) or np.any(valid < 0) or np.any(train >= len(y)) or np.any(valid >= len(y)):
            raise ValueError("fold index out of bounds")
        if set(train) & set(valid):
            raise ValueError("train and validation rows must be disjoint")
        if int(train.max()) >= int(valid.min()):
            raise ValueError("folds must be forward ordered")
        if seen_valid & set(valid):
            raise ValueError("validation rows must not repeat across folds")
        seen_valid.update(map(int, valid)); checked_folds.append((train, valid))
    for action in actions:
        candidate = set(action.resulting_features)
        if any(f not in X_by_feature for f in action.resulting_features):
            raise ValueError("action feature is unavailable")
        valid_change = ((action.operation == "ADD" and candidate > baseline_set) or
                        (action.operation == "REPLACE" and bool(baseline_set-candidate) and bool(candidate-baseline_set)) or
                        (action.operation == "DROP_CLUSTER" and candidate < baseline_set))
        if not valid_change:
            raise ValueError(f"{action.operation} does not match the declared resulting feature set")

    def fit_oof(features, label):
        X = np.column_stack([np.asarray(X_by_feature[f], dtype=float) for f in features])
        pred = np.full(len(y), np.nan)
        refs = []
        for fold_ref, (train, valid) in zip(experiment_spec.fold_refs, checked_folds):
            learner = learner_factory(len(features))
            frozen = learner.fit(X[train], y[train])
            if not isinstance(frozen, FrozenModel):
                raise TypeError("learner must produce a real FrozenModel")
            pred[valid] = learner.predict(frozen, X[valid])
            payload = {"schema": "oof-frozen-model.v1", "label": label, "fold": fold_ref,
                       "features": list(features), "learner_name": frozen.learner_name,
                       "family": frozen.family, "params": _jsonable(frozen.params),
                       "metadata": _jsonable(frozen.metadata)}
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            digest = sha256(raw).hexdigest(); path = artifact_dir / f"{digest}.json"
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("content-addressed model artifact collision")
            if not path.exists():
                fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", dir=artifact_dir)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
            ref = f"frozen-model:{digest}:{path}"
            loaded = load_frozen_model_ref(ref)
            if loaded.learner_name != frozen.learner_name or loaded.family != frozen.family:
                raise ValueError("persisted frozen model failed replay validation")
            refs.append(ref)
        return pred, tuple(refs)

    baseline_attempt = experiment_spec.attempt_id("baseline", 0)
    cost = experiment_spec.estimated_cost_per_evaluation
    if not budget_tracker.reserve_evaluation(cost, attempt_id=baseline_attempt):
        raise ValueError("feature-set experiment budget exhausted before baseline")
    budget_tracker.start_evaluation(attempt_id=baseline_attempt)
    try:
        baseline_pred, baseline_refs = fit_oof(baseline_features, "baseline")
        budget_tracker.commit_evaluation(cost, cost, attempt_id=baseline_attempt)
    except Exception:
        budget_tracker.commit_evaluation(cost, cost, attempt_id=baseline_attempt)
        raise
    results = []
    for i, action in enumerate(actions):
        attempt_id = experiment_spec.attempt_id(action.operation.lower(), i)
        if not budget_tracker.reserve_evaluation(cost, attempt_id=attempt_id):
            raise ValueError("feature-set experiment budget exhausted")
        started = False
        try:
            budget_tracker.start_evaluation(attempt_id=attempt_id); started = True
            candidate_pred, model_refs = fit_oof(action.resulting_features, action.action_id)
            common = np.isfinite(baseline_pred) & np.isfinite(candidate_pred) & np.isfinite(y)
            if common.sum() < 2:
                raise ValueError("fewer than two common finite OOF rows")
            base_score, candidate_score = _score(baseline_pred[common], y[common]), _score(candidate_pred[common], y[common])
            delta = candidate_score - base_score
            budget_tracker.commit_evaluation(cost, cost, attempt_id=attempt_id)
        except Exception:
            if started: budget_tracker.commit_evaluation(cost, cost, attempt_id=attempt_id)
            else: budget_tracker.release_evaluation(cost, attempt_id=attempt_id)
            raise
        accepted = bool(np.isfinite(delta) and delta >= min_delta)
        payload = {"action": action.__dict__, "spec": experiment_spec.evaluation_intent_hash(),
                   "rows": tuple(np.asarray(row_ids)[common]), "baseline_score": base_score,
                   "candidate_score": candidate_score, "delta": delta, "min_delta": min_delta,
                   "accepted": accepted, "baseline_models": baseline_refs, "models": model_refs}
        evidence_ref = "feature-set-trial:" + sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        results.append(FeatureSetTrialEvidence(action, base_score, candidate_score, delta,
                                               tuple(np.asarray(row_ids)[common]), experiment_spec.fold_refs,
                                               experiment_spec.evaluation_intent_hash(), min_delta, baseline_refs,
                                               model_refs, evidence_ref, accepted))
    return tuple(results)


__all__ = ["FeatureSetAction", "FeatureSetTrialEvidence", "run_feature_set_trials", "load_frozen_model_ref",
           "validate_feature_set_trial_evidence"]
