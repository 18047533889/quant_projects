# -*- coding: utf-8 -*-
"""Focused persistence and immutability hardening tests for model artifacts."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from modeling.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    FrozenPreprocessing,
    ModelArtifact,
    ModelArtifactManifest,
)
from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec


class _Learner(BaseLearner):
    name = "artifact_hardening_test"
    family = "linear"

    def fit(self, X, y, *, weights=None, aux=None):  # pragma: no cover - not used
        raise AssertionError("fit must not run")

    def predict(self, frozen, X):
        return X


def _artifact() -> ModelArtifact:
    nested = {"grid": [1, {"alpha": np.float64(0.5)}]}
    manifest = ModelArtifactManifest(
        model_name="model",
        model_version="v1",
        artifact_id="artifact-1",
        train_start="2020-01-01",
        train_end="2020-12-31",
        validation_start="2021-01-01",
        validation_end="2021-06-30",
        selection_train_end="2020-12-31",
        final_fit_start="2020-01-01",
        final_fit_end="2021-06-30",
        refit_used_validation=True,
        training_cutoff="2021-07-01",
        available_at="2021-07-02",
        hyperparameters=nested,
    )
    learner = _Learner(LearnerSpec("artifact_hardening_test", "linear"))
    source = np.array([1.0, 2.0])
    return ModelArtifact(
        manifest,
        learner,
        FrozenModel(
            "artifact_hardening_test",
            "linear",
            params={"coef": source, "nested": [{"x": 1}]},
            metadata={"code_hash": "abc"},
        ),
        FrozenPreprocessing([{"kind": "standardize", "mean": source, "scale": [1.0, 1.0]}]),
        fit_info={"scores": [{"ic": np.float64(0.2)}]},
    )


def test_artifact_state_is_deeply_frozen_and_detached():
    artifact = _artifact()

    with pytest.raises(TypeError):
        artifact.manifest.hyperparameters["new"] = 1
    with pytest.raises(TypeError):
        artifact.manifest.hyperparameters["grid"][1]["alpha"] = 2
    with pytest.raises(TypeError):
        artifact.frozen.params["nested"][0]["x"] = 2
    with pytest.raises(ValueError):
        artifact.frozen.params["coef"][0] = 9
    with pytest.raises(TypeError):
        artifact.preprocessing.steps[0]["mean"] = [9.0, 9.0]
    with pytest.raises(TypeError):
        artifact.fit_info["scores"][0]["ic"] = 9


def test_roundtrip_preserves_new_lineage_fields_and_emits_plain_data(tmp_path):
    artifact = _artifact()
    payload = artifact.to_dict()

    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["manifest"]["selection_train_end"] == "2020-12-31"
    assert payload["manifest"]["final_fit_end"] == "2021-06-30"
    assert payload["manifest"]["refit_used_validation"] is True
    assert isinstance(payload["frozen"]["params"]["coef"], list)

    path = tmp_path / "nested" / "artifact.json"
    artifact.save(path)
    loaded = ModelArtifact.load(path, _Learner(LearnerSpec("artifact_hardening_test", "linear")))
    assert loaded.manifest.final_fit_end == artifact.manifest.final_fit_end
    assert loaded.manifest.lineage_hash() == artifact.manifest.lineage_hash()
    assert loaded.cache_key() == artifact.cache_key()


def test_schema_and_lineage_gates_fail_closed():
    artifact = _artifact()
    payload = artifact.to_dict()

    missing_schema = dict(payload)
    missing_schema.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        ModelArtifact.from_dict(missing_schema, artifact.learner)

    future_schema = dict(payload, schema_version=ARTIFACT_SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version"):
        ModelArtifact.from_dict(future_schema, artifact.learner)

    tampered = artifact.to_dict()
    tampered["manifest"]["available_at"] = "2021-07-03"
    with pytest.raises(ValueError, match="lineage hash mismatch"):
        ModelArtifact.from_dict(tampered, artifact.learner)

    no_hash = artifact.to_dict()
    no_hash["manifest"].pop("lineage_hash")
    with pytest.raises(ValueError, match="lineage hash mismatch"):
        ModelArtifact.from_dict(no_hash, artifact.learner)


def test_save_is_canonical_atomic_and_never_overwrites(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    artifact.save(path)
    first = path.read_bytes()

    decoded = json.loads(first)
    assert first == json.dumps(
        decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        artifact.save(path)
    assert path.read_bytes() == first
    assert list(tmp_path.glob(".artifact.json.*")) == []


def test_canonical_serialization_rejects_non_finite_and_unsupported_values(tmp_path):
    artifact = _artifact()
    artifact.learner.spec = LearnerSpec(
        "artifact_hardening_test", "linear", hyperparams={"bad": float("nan")}
    )
    with pytest.raises(ValueError):
        artifact.save(tmp_path / "nan.json")

    artifact.learner.spec = LearnerSpec(
        "artifact_hardening_test", "linear", hyperparams={"bad": object()}
    )
    with pytest.raises(TypeError, match="not JSON serializable"):
        artifact.save(tmp_path / "object.json")
