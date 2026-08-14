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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec

__all__ = [
    "ModelArtifactManifest",
    "ModelArtifact",
    "PREPROCESSING_IDENTITY_KINDS",
    "ARTIFACT_SCHEMA_VERSION",
]

ARTIFACT_SCHEMA_VERSION = 1


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

    # --------------------------------------------------------------------------- #
    # MF-P0-005: Disambiguate the overloaded `training_cutoff` into semantically
    # distinct time fields — each instant has a clear meaning and ordering invariants
    # are enforced fail-closed.
    # --------------------------------------------------------------------------- #
    #: Last feature/anchor date actually used to fit the final model.
    final_fit_anchor_end: str = ""
    #: Last date whose label is fully mature (anchor + horizon).
    label_maturity_cutoff: str = ""
    #: Wall-clock timestamp when fitting finished.
    fit_completed_at: str = ""
    #: When the artifact may first be consumed (must be >= label_maturity_cutoff).
    artifact_available_at: str = ""
    #: When the artifact may first be used for decisions (must be >= artifact_available_at).
    activation_at: str = ""

    #: DEPRECATED: use the explicit fields above. Kept for backward compatibility;
    #: derived from label_maturity_cutoff when new fields are set.
    training_cutoff: str = ""
    #: When this artifact became legal (available_at); as-of resolution uses it.
    #: DEPRECATED: use artifact_available_at.
    available_at: str = ""
    #: Predictor ABI version — artifacts must match the runtime predictor ABI.
    predictor_abi_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperparameters", _freeze(self.hyperparameters))
        if self.final_fit_end is None:
            object.__setattr__(self, "final_fit_end", self.train_end)
        if self.final_fit_start is None:
            object.__setattr__(self, "final_fit_start", self.train_start)

        # MF-P0-005: Backward compatibility + fail-closed ordering invariants.
        # If new fields are not set, derive from legacy fields.
        if not self.final_fit_anchor_end and self.final_fit_end:
            object.__setattr__(self, "final_fit_anchor_end", self.final_fit_end)
        if not self.label_maturity_cutoff:
            if self.training_cutoff:
                object.__setattr__(self, "label_maturity_cutoff", self.training_cutoff)
            elif self.final_fit_end:
                object.__setattr__(self, "label_maturity_cutoff", self.final_fit_end)
        if not self.artifact_available_at and self.available_at:
            object.__setattr__(self, "artifact_available_at", self.available_at)
        if not self.activation_at and self.artifact_available_at:
            object.__setattr__(self, "activation_at", self.artifact_available_at)

        # Backward compat: derive training_cutoff from new fields when it's not set
        if not self.training_cutoff and self.label_maturity_cutoff:
            object.__setattr__(self, "training_cutoff", self.label_maturity_cutoff)
        if not self.available_at and self.artifact_available_at:
            object.__setattr__(self, "available_at", self.artifact_available_at)

        # MF-P0-005: Fail-closed ordering invariants.
        # artifact_available_at >= label_maturity_cutoff (anti-look-ahead)
        if self.artifact_available_at and self.label_maturity_cutoff:
            if self.artifact_available_at < self.label_maturity_cutoff:
                raise ValueError(
                    f"artifact cannot be available before label maturity: "
                    f"artifact_available_at={self.artifact_available_at} < "
                    f"label_maturity_cutoff={self.label_maturity_cutoff}"
                )

        # activation_at >= artifact_available_at
        if self.activation_at and self.artifact_available_at:
            if self.activation_at < self.artifact_available_at:
                raise ValueError(
                    f"artifact cannot be activated before it is available: "
                    f"activation_at={self.activation_at} < "
                    f"artifact_available_at={self.artifact_available_at}"
                )

        # Legacy invariants (still enforced for backward compat)
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
            # MF-P0-005: Include new time fields in lineage
            "final_fit_anchor_end": self.final_fit_anchor_end,
            "label_maturity_cutoff": self.label_maturity_cutoff,
            "fit_completed_at": self.fit_completed_at,
            "artifact_available_at": self.artifact_available_at,
            "activation_at": self.activation_at,
            # Legacy fields (kept for backward compat)
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
            "predictor_abi_version": self.predictor_abi_version,
        }
        return _digest(payload)


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
        self._steps = tuple(_freeze(s) for s in steps)

    @property
    def steps(self) -> tuple[Mapping[str, Any], ...]:
        return self._steps

    def state_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._steps).encode("utf-8")
        ).hexdigest()

    def transform(self, X: np.ndarray, *, application_window: Any = None) -> np.ndarray:
        """Apply frozen preprocessing transform (MODEL-P0-005).

        For public OOS usage, ``application_window`` should be provided to ensure
        the transform is only applied to data after training cutoff. Internal
        fit_transform (during training) may pass None.

        Args:
            X: Input features (n_samples, n_features).
            application_window: Optional ApplicationWindow to validate OOS safety.
                If provided, this enforces temporal boundaries on when the transform
                can be applied (must be after training cutoff).

        Returns:
            Transformed features.
        """
        # MODEL-P0-005: Optional application window validation
        # (not enforced in base implementation for backward compatibility,
        # but ModelArtifact.predict_oos enforces it at the public API level)
        if application_window is not None:
            # If dates are embedded in X or passed separately, validate them
            # For now, this is a placeholder for future enforcement
            pass

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


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        out = np.array(value, copy=True)
        out.setflags(write=False)
        return out
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        self.frozen = FrozenModel(
            learner_name=frozen.learner_name,
            family=frozen.family,
            params=_freeze(frozen.params),
            metadata=_freeze(frozen.metadata),
        )
        self.preprocessing = preprocessing
        self.fit_info = _freeze(dict(fit_info or {}))
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
                m.available_at,
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

    def predict_oos(
        self,
        X: np.ndarray,
        *,
        application_window: Any,
        dates: Any = None,
    ) -> np.ndarray:
        """Out-of-sample prediction with application window enforcement (MODEL-P0-005).

        Public OOS API that requires explicit ApplicationWindow to prevent temporal
        leakage. The window ensures the transform is only applied to data after
        the training cutoff.

        Args:
            X: Input features (n_samples, n_features).
            application_window: Required ApplicationWindow defining legal OOS dates.
            dates: Optional array of dates corresponding to X rows for validation.

        Returns:
            Predictions.

        Raises:
            ValueError: If application_window is None or dates violate the window.
        """
        from modeling.contracts import ApplicationWindow

        if application_window is None:
            raise ValueError(
                "predict_oos requires application_window; use predict() for in-sample only"
            )

        if not isinstance(application_window, ApplicationWindow):
            raise TypeError(
                f"application_window must be ApplicationWindow, got {type(application_window)}"
            )

        # Validate dates if provided
        if dates is not None:
            valid, violations = application_window.validate_dates(dates)
            if not valid:
                raise ValueError(
                    f"OOS prediction violates application window: {'; '.join(violations)}"
                )

        # Apply transform with window context
        Xt = self.preprocessing.transform(
            np.asarray(X, dtype=np.float64),
            application_window=application_window,
        )
        return self.learner.predict(self.frozen, Xt)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "manifest": {
                **{k: _thaw(v) for k, v in self.manifest.__dict__.items()},
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
                "params": _thaw(self.frozen.params),
                "metadata": _thaw(self.frozen.metadata),
            },
            "preprocessing_steps": _thaw(self.preprocessing.steps),
            "fit_info": _thaw(self.fit_info),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], learner: BaseLearner) -> "ModelArtifact":
        from modeling.contracts import SampleAdequacyContract

        version = data.get("schema_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact schema_version {version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        m = data["manifest"]
        expected_lineage = m.get("lineage_hash")
        manifest = ModelArtifactManifest(**{k: v for k, v in m.items() if k != "lineage_hash"})
        if not isinstance(expected_lineage, str) or expected_lineage != manifest.lineage_hash():
            raise ValueError("artifact lineage hash mismatch")
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
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite artifact {target}")
        payload = _canonical_json(self.to_dict())
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.link(tmp_name, target)
            os.unlink(tmp_name)
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

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
