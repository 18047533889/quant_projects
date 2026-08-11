# -*- coding: utf-8 -*-
"""Model artifact — first-class citizen (Model Layer Major Redesign taskbook
§21 / §22 / §31 / §52).

A :class:`ModelArtifact` bundles the frozen model, the frozen preprocessing
(train-only fit: scaler / PCA / imputer / feature order / feature dtype), and a
:class:`ModelArtifactManifest` recording every lineage fact §21 requires.  The
cache key (§52) includes asof / vintage / cutoff so a future-trained artifact
can never be reused for history.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec

__all__ = [
    "ModelArtifactManifest",
    "ModelArtifact",
    "PREPROCESSING_IDENTITY_KINDS",
]


# --------------------------------------------------------------------------- #
# §21 manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelArtifactManifest:
    """Lineage manifest — every fact a frozen prediction needs to be auditable."""

    model_name: str
    model_version: str
    artifact_id: str

    train_start: str
    train_end: str
    validation_start: str | None = None
    validation_end: str | None = None

    #: End of the data window used for HYPERPARAMETER SELECTION (the original
    #: train window).  Distinct from ``final_fit_end`` so a ``train_plus_validation``
    #: refit can never present itself as available before its real data.
    selection_train_end: str | None = None
    #: Data window used for the FINAL model fit.  ``final_fit_end`` is
    #: ``train_end`` when ``retrain_policy="train_only"`` and ``validation_end``
    #: when ``"train_plus_validation"``.  ``training_cutoff == final_fit_end`` and
    #: ``available_at >= final_fit_end`` — an artifact is never loadable before
    #: the last observation it actually saw (§50 / availability lookahead guard).
    final_fit_start: str | None = None
    final_fit_end: str | None = None
    #: True when the FINAL fit was refit on train+validation (i.e. the artifact
    #: has seen every validation observation up to ``final_fit_end``).
    refit_used_validation: bool = False

    decision_clock_id: str = "AFTER_CLOSE_TO_NEXT_VWAP"
    label_contract_id: str = "vwap_to_vwap"
    feature_schema_hash: str = ""
    data_source_hash: str = ""
    universe_hash: str = ""

    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing_state_hash: str = ""

    fit_code_commit: str = ""
    fit_code_component_hash: str = ""

    random_seed: int | None = None
    solver_version: str | None = None

    #: When this artifact became legal (available_at); as-of resolution uses it.
    available_at: str = ""
    #: Training cutoff date — artifacts are only loadable at/after this.
    training_cutoff: str = ""

    def __post_init__(self) -> None:
        if self.final_fit_end is None:
            object.__setattr__(self, "final_fit_end", self.train_end)
        if self.final_fit_start is None:
            object.__setattr__(self, "final_fit_start", self.train_start)
        # Fail-closed invariants: an artifact can never claim to be available
        # before the last observation used by its FINAL fit.
        if self.final_fit_start and self.final_fit_end and self.final_fit_start > self.final_fit_end:
            raise ValueError(
                f"final_fit_start {self.final_fit_start} > final_fit_end "
                f"{self.final_fit_end} — invalid final-fit window"
            )
        if self.training_cutoff and self.training_cutoff < self.final_fit_end:
            raise ValueError(
                f"training_cutoff {self.training_cutoff} < final_fit_end "
                f"{self.final_fit_end} — Artifact Availability Lookahead (forbidden)"
            )
        if self.available_at and self.available_at < self.final_fit_end:
            raise ValueError(
                f"available_at {self.available_at} < final_fit_end {self.final_fit_end} — "
                "an artifact cannot be loadable before its final fit data"
            )
        if self.selection_train_end is not None and self.selection_train_end > self.final_fit_end:
            raise ValueError(
                f"selection_train_end {self.selection_train_end} > final_fit_end "
                f"{self.final_fit_end} — selection cannot extend past the final fit"
            )

    def lineage_hash(self) -> str:
        payload = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "selection_train_end": self.selection_train_end,
            "final_fit_start": self.final_fit_start,
            "final_fit_end": self.final_fit_end,
            "refit_used_validation": self.refit_used_validation,
            "available_at": self.available_at,
            "training_cutoff": self.training_cutoff,
            "decision_clock_id": self.decision_clock_id,
            "label_contract_id": self.label_contract_id,
            "feature_schema_hash": self.feature_schema_hash,
            "data_source_hash": self.data_source_hash,
            "universe_hash": self.universe_hash,
            "hyperparameters": self.hyperparameters,
            "preprocessing_state_hash": self.preprocessing_state_hash,
            "fit_code_commit": self.fit_code_commit,
            "fit_code_component_hash": self.fit_code_component_hash,
            "random_seed": self.random_seed,
            "solver_version": self.solver_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


# --------------------------------------------------------------------------- #
# Frozen preprocessing — a compact, immutable, train-only-fit transform chain.
# --------------------------------------------------------------------------- #
PREPROCESSING_IDENTITY_KINDS = ("imputer", "winsor", "standardize", "pca", "feature_select")


class FrozenPreprocessing:
    """Immutable preprocessing fit on TRAIN ONLY (§13.2/§13.3/§13.4/§31).

    Each step is serialised as plain numpy state.  ``transform`` only applies
    the frozen state — it never recomputes statistics from the incoming data.
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        for s in steps:
            if s.get("kind") not in PREPROCESSING_IDENTITY_KINDS:
                raise ValueError(f"unknown preprocessing kind {s.get('kind')!r}")
        self._steps = steps

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self._steps

    def state_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self._steps, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest()

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X
        for step in self._steps:
            kind = step["kind"]
            if kind == "imputer":
                out = _apply_imputer(out, step)
            elif kind == "winsor":
                out = _apply_winsor(out, step)
            elif kind == "standardize":
                out = _apply_standardize(out, step)
            elif kind == "pca":
                out = _apply_pca(out, step)
            elif kind == "feature_select":
                out = out[:, step["feature_indices"]]
        return out


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _apply_imputer(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    """Mean imputation using train-only column means (§13.3)."""
    means = step["means"]
    out = X.copy()
    mask = ~np.isfinite(out)
    fill = np.broadcast_to(np.asarray(means, dtype=np.float64), out.shape)
    out = np.where(mask, fill, out)
    return out


def _apply_winsor(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    lows = np.asarray(step["lows"], dtype=np.float64)
    highs = np.asarray(step["highs"], dtype=np.float64)
    return np.clip(X, lows, highs)


def _apply_standardize(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(step["mean"], dtype=np.float64)
    scale = np.asarray(step["scale"], dtype=np.float64)
    return (X - mean) / scale


def _apply_pca(X: np.ndarray, step: dict[str, Any]) -> np.ndarray:
    """Project with the frozen train-only PCA (component matrix in rows)."""
    components = np.asarray(step["components"], dtype=np.float64)  # (k, d)
    center = np.asarray(step["center"], dtype=np.float64)
    return (X - center) @ components.T


# --------------------------------------------------------------------------- #
# ModelArtifact
# --------------------------------------------------------------------------- #
class ModelArtifact:
    """A frozen, as-of-legal model artifact (§21/§22/§50/§52)."""

    def __init__(
        self,
        manifest: ModelArtifactManifest,
        learner: BaseLearner,
        frozen: FrozenModel,
        preprocessing: FrozenPreprocessing,
        fit_info: dict[str, Any] | None = None,
    ) -> None:
        self.manifest = manifest
        self.learner = learner
        self.frozen = frozen
        self.preprocessing = preprocessing
        self.fit_info = dict(fit_info or {})
        self._created = False  # prediction guard (§22)

    # -- identity ------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.manifest.model_name

    @property
    def version(self) -> str:
        return self.manifest.model_version

    @property
    def artifact_id(self) -> str:
        return self.manifest.artifact_id

    def cache_key(self) -> str:
        """§52 cache key — includes asof / vintage / cutoff."""
        m = self.manifest
        return "|".join(
            [
                m.model_name,
                m.model_version,
                m.training_cutoff,
                m.validation_end or "",
                m.feature_schema_hash,
                m.label_contract_id,
                self.preprocessing.state_hash(),
                m.data_source_hash,
                m.universe_hash,
                self.frozen.metadata.get("code_hash", ""),
            ]
        )

    def is_legal_asof(self, asof: Any, training_cutoff: Any = None) -> bool:
        """§50 — an artifact is only legal when it existed at/after its
        training cutoff and its ``available_at`` is not after ``asof``.

        ``asof`` / ``training_cutoff`` may be date strings, ``Timestamp`` or
        comparable scalars; mixed str/Timestamp comparisons are normalised so
        the check is robust to caller convention."""
        avail = self.manifest.available_at
        cutoff = self.manifest.training_cutoff
        if training_cutoff is not None and cutoff:
            if _cmp_less(training_cutoff, cutoff):
                return False
        if avail and asof is not None:
            if _cmp_less(asof, avail):
                return False
        return True


    # -- scoring (§22: prediction must never fit) -----------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Frozen scoring path.  Guaranteed NOT to call learner.fit (the
        predictor layer additionally asserts this via instrumentation)."""
        Xt = self.preprocessing.transform(np.asarray(X, dtype=np.float64))
        return self.learner.predict(self.frozen, Xt)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": {
                **self.manifest.__dict__,
                "lineage_hash": self.manifest.lineage_hash(),
            },
            "learner_name": self.learner.name,
            "learner_spec": {
                "learner_name": self.learner.spec.learner_name,
                "family": self.learner.spec.family,
                "hyperparams": self.learner.spec.hyperparams,
                "random_seed": self.learner.spec.random_seed,
            },
            "frozen": {
                "learner_name": self.frozen.learner_name,
                "family": self.frozen.family,
                "params": self.frozen.params,
                "metadata": self.frozen.metadata,
            },
            "preprocessing_steps": self.preprocessing.steps,
            "fit_info": self.fit_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], learner: BaseLearner) -> "ModelArtifact":
        from modeling.contracts import SampleAdequacyContract

        m = data["manifest"]
        manifest = ModelArtifactManifest(**{k: v for k, v in m.items() if k != "lineage_hash"})
        spec = LearnerSpec(
            learner_name=data["learner_spec"]["learner_name"],
            family=data["learner_spec"]["family"],
            hyperparams=data["learner_spec"]["hyperparams"],
            random_seed=data["learner_spec"]["random_seed"],
        )
        learner.spec = spec
        frozen = FrozenModel(
            learner_name=data["frozen"]["learner_name"],
            family=data["frozen"]["family"],
            params=data["frozen"]["params"],
            metadata=data["frozen"]["metadata"],
        )
        return cls(
            manifest=manifest,
            learner=learner,
            frozen=frozen,
            preprocessing=FrozenPreprocessing(data["preprocessing_steps"]),
            fit_info=data.get("fit_info", {}),
        )

    def save(self, path: Any) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=_json_default)

    @classmethod
    def load(cls, path: Any, learner: BaseLearner | None = None) -> "ModelArtifact":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if learner is None:
            from modeling.learners import get_learner

            learner = get_learner(data["learner_name"])(
                LearnerSpec(data["learner_spec"]["learner_name"], data["learner_spec"]["family"])
            )
        return cls.from_dict(data, learner)


def _cmp_less(a: Any, b: Any) -> bool:
    """``a < b`` with str/Timestamp normalisation."""
    try:
        return bool(a < b)
    except TypeError:
        try:
            return bool(pd.Timestamp(a) < pd.Timestamp(b))
        except Exception:
            raise TypeError(f"cannot compare asof values {a!r} < {b!r}")
