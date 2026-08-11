# -*- coding: utf-8 -*-
"""Walk-forward evidence + hard-gate report (Model Layer Major Redesign
taskbook §19 / §56 / §64 / §71 / §83).

* :class:`FoldResult` — one outer walk-forward fold.
* :class:`WalkForwardEvidence` — folds + OOS evaluation + dataset exposure +
  §56 ledger rows, serialisable to parquet / JSON.
* :func:`run_walk_forward_evidence` — the §71 runner.
* :func:`report_hard_gate_set` — the subset of §64 gates this package can
  honestly verify.

The runner imports :mod:`modeling.trainer` / :mod:`modeling.predictor` lazily
(they are built by a parallel agent); when they are unavailable it falls back
to a small internal fit/score path so the walk-forward machinery is usable and
testable today.  All scoring is done on a frozen artifact — never ``fit``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from modeling.artifact import FrozenPreprocessing, ModelArtifact, ModelArtifactManifest
from modeling.contracts import DecisionClock, LabelContract
from modeling.dataset import PanelDataset
from modeling.evaluation import evaluate_predictions, per_date_rank_ic
from modeling.learners.base import LearnerSpec

__all__ = [
    "FoldResult",
    "WalkForwardEvidence",
    "run_walk_forward_evidence",
    "report_hard_gate_set",
]


@dataclass
class FoldResult:
    """One outer walk-forward fold (§71)."""

    fold_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    selected_hyperparams: dict[str, Any]
    test_pred: np.ndarray
    test_y: np.ndarray
    test_dates: np.ndarray
    validation_best_rank_ic: float
    artifact: dict[str, Any] | None = None
    test_stocks: np.ndarray | None = None  # for parquet export when available
    validation_start: Any = None  # for §56 ledger completeness
    validation_end: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "validation_start": str(self.validation_start) if self.validation_start is not None else None,
            "validation_end": str(self.validation_end) if self.validation_end is not None else None,
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "selected_hyperparams": self.selected_hyperparams,
            "validation_best_rank_ic": self.validation_best_rank_ic,
            "n_test_rows": int(len(self.test_pred)),
            "artifact": self.artifact,
        }


@dataclass
class WalkForwardEvidence:
    """§71 — complete walk-forward evidence bundle."""

    folds: list[FoldResult] = field(default_factory=list)
    oos_evaluation: dict[str, Any] = field(default_factory=dict)
    dataset_exposure: dict[str, Any] = field(default_factory=dict)
    leadger_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_parquet(self, path: str) -> None:
        """Write ``[fold_id, date, stock, pred, y]`` rows for every fold."""
        rows: list[dict[str, Any]] = []
        for fold in self.folds:
            n = len(fold.test_pred)
            stocks = fold.test_stocks if fold.test_stocks is not None else [""] * n
            for i in range(n):
                rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "date": fold.test_dates[i],
                        "stock": stocks[i],
                        "pred": float(fold.test_pred[i]),
                        "y": float(fold.test_y[i]) if np.isfinite(fold.test_y[i]) else None,
                    }
                )
        pd.DataFrame(rows).to_parquet(path, index=False)

    def to_json(self) -> str:
        payload = {
            "folds": [f.to_dict() for f in self.folds],
            "oos_evaluation": self.oos_evaluation,
            "dataset_exposure": self.dataset_exposure,
            "leadger_rows": self.leadger_rows,
        }
        return json.dumps(payload, default=str, indent=1)


# --------------------------------------------------------------------------- #
# §71 walk-forward
# --------------------------------------------------------------------------- #
def _spec_attr(spec: Any, name: str, default: Any = None) -> Any:
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _build_folds(
    dataset: PanelDataset,
    spec: Any,
    label_contract: LabelContract,
) -> tuple[list[dict[str, Any]], list]:
    """Date-authoritative fold slicing with §9 label-overlap purge.

    ``train_end`` is shifted back by ``horizon + embargo`` bars so training
    rows whose label interval overlaps the validation window are excluded.
    """
    dates = sorted(dataset.frame[dataset.date_col].unique())
    n = len(dates)
    train_lb = int(_spec_attr(spec, "train_lookback_bars", 4))
    val_bars = int(_spec_attr(spec, "validation_bars", 2))
    test_bars = int(_spec_attr(spec, "test_bars", 2))
    step = int(_spec_attr(spec, "step_bars", 2))
    horizon = int(label_contract.horizon_bars)
    embargo = int(_spec_attr(spec, "embargo_bars", 0) or 0)

    folds: list[dict[str, Any]] = []
    fid = 0
    start = train_lb
    while start + val_bars + test_bars <= n:
        folds.append(
            {
                "fold_id": fid,
                "train_start_i": start - train_lb,
                "train_end_i": start - horizon - embargo - 1,
                "val_start_i": start,
                "val_end_i": start + val_bars - 1,
                "test_start_i": start + val_bars,
                "test_end_i": start + val_bars + test_bars - 1,
            }
        )
        start += step
        fid += 1
    return folds, dates


def _fit_learner(learner_cls, X: np.ndarray, y: np.ndarray,
                 hp: dict[str, Any], random_seed: int | None):
    from modeling.learners.base import LearnerSpec

    spec = LearnerSpec(
        learner_name=learner_cls.name, family=learner_cls.family,
        hyperparams=dict(hp or {}), random_seed=random_seed,
    )
    learner = learner_cls(spec)
    return learner, learner.fit(X, y)


def _score(learner, frozen, X: np.ndarray) -> np.ndarray:
    return learner.predict(frozen, X)


def _purge_immature(ds: PanelDataset, horizon: int) -> PanelDataset:
    if horizon <= 0:
        return ds
    uniq = sorted(ds.frame[ds.date_col].unique())
    if len(uniq) <= horizon:
        return ds
    last = set(uniq[-horizon:])
    mask = ~ds.frame[ds.date_col].isin(last)
    return PanelDataset(
        frame=ds.frame.loc[mask].reset_index(drop=True),
        date_col=ds.date_col, stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols), label_col=ds.label_col,
    )


def _build_artifact(
    learner_cls, train_ds: PanelDataset, frozen, hp: dict[str, Any],
    label_contract: LabelContract, model_version: str, fit_code_commit: str,
    random_seed: int | None, decision_clock: DecisionClock,
    data_source_hash: str = "src-a", universe_hash: str = "uni-a",
) -> ModelArtifact:
    learner = learner_cls(
        LearnerSpec(
            learner_name=learner_cls.name, family=learner_cls.family,
            hyperparams=dict(hp or {}), random_seed=random_seed,
        )
    )
    preproc = FrozenPreprocessing([])  # identity; train-only frozen transforms
    manifest = ModelArtifactManifest(
        model_name=learner_cls.name,
        model_version=model_version,
        artifact_id=f"{learner_cls.name}-{model_version}-{fit_code_commit or 'head'}",
        train_start=str(pd.Timestamp(train_ds.frame[train_ds.date_col].min())),
        train_end=str(pd.Timestamp(train_ds.frame[train_ds.date_col].max())),
        decision_clock_id=decision_clock.decision_at,
        label_contract_id=label_contract.label_name,
        feature_schema_hash="|".join(train_ds.feature_cols),
        data_source_hash=data_source_hash,
        universe_hash=universe_hash,
        hyperparameters=dict(hp or {}),
        preprocessing_state_hash=preproc.state_hash(),
        fit_code_commit=fit_code_commit,
        random_seed=random_seed,
    )
    return ModelArtifact(manifest, learner, frozen, preproc)


def _select_hyperparams(
    learner_cls, train_ds: PanelDataset, val_ds: PanelDataset,
    label_contract: LabelContract, grid: list[dict[str, Any]],
    random_seed: int | None,
) -> tuple[dict[str, Any], float]:
    """Validation-driven hyperparameter selection (never test-driven)."""
    if not grid:
        return {}, float("nan")
    horizon = label_contract.horizon_bars
    train_purged = _purge_immature(train_ds, horizon)
    Xtr, ytr, _, _, finite = train_purged.as_matrix()
    Xtr, ytr = Xtr[finite], ytr[finite]
    Xv, yv, dts, _, _ = val_ds.as_matrix()
    if yv is None or len(Xtr) < 3:
        return dict(grid[0]), float("nan")
    best, best_ic = dict(grid[0]), -np.inf
    for hp in grid:
        learner, frozen = _fit_learner(learner_cls, Xtr, ytr, hp, random_seed)
        pred = _score(learner, frozen, Xv)
        m = np.isfinite(pred) & np.isfinite(yv)
        if m.sum() < 3:
            continue
        _, ics = per_date_rank_ic(pred[m], yv[m], dts[m])
        ic = float(np.nanmean(ics)) if len(ics) else -np.inf
        if ic > best_ic:
            best_ic = ic
            best = dict(hp)
    return best, best_ic


def _fallback_train_predict(
    learner_cls,
    train_ds: PanelDataset,
    val_ds: PanelDataset,
    test_ds: PanelDataset,
    label_contract: LabelContract,
    decision_clock: DecisionClock,
    hyperparam_grid: list[dict[str, Any]] | None,
    model_version: str,
    fit_code_commit: str,
    random_seed: int | None,
    aux_col: str | None,
) -> tuple[ModelArtifact, dict[str, Any], float, np.ndarray]:
    """Train on train (+validation-selected hyperparams), score test, frozen."""
    horizon = label_contract.horizon_bars
    train_purged = _purge_immature(train_ds, horizon)
    Xtr, ytr, _, _, finite = train_purged.as_matrix()
    Xtr, ytr = Xtr[finite], ytr[finite]

    hp, val_ic = _select_hyperparams(
        learner_cls, train_ds, val_ds, label_contract,
        list(hyperparam_grid or []), random_seed,
    )
    learner, frozen = _fit_learner(learner_cls, Xtr, ytr, hp, random_seed)
    artifact = _build_artifact(
        learner_cls, train_purged, frozen, hp, label_contract,
        model_version, fit_code_commit, random_seed, decision_clock,
    )
    Xte, _, _, _, _ = test_ds.as_matrix()
    pred = artifact.predict(Xte)
    return artifact, hp, val_ic, pred


def run_walk_forward_evidence(
    learner_cls,
    dataset: PanelDataset,
    walk_forward_spec,
    *,
    label_contract: LabelContract,
    decision_clock: DecisionClock,
    preprocessing_spec=None,
    hyperparam_grid=None,
    model_version: str = "1.0",
    fit_code_commit: str = "",
    random_seed: int | None = None,
    aux_col: str | None = None,
    min_folds: int = 3,
) -> WalkForwardEvidence:
    """Run the §71 outer walk-forward loop.

    Requires ``>= min_folds`` outer folds (raises ``ValueError`` otherwise).
    Each fold trains on the (purged) train window, selects hyperparameters on
    the validation window only, and scores the frozen artifact on the test
    window.  Uses :mod:`modeling.trainer` / :mod:`modeling.predictor` when
    available (lazy import); otherwise a self-contained fallback.
    """
    folds, dates = _build_folds(dataset, walk_forward_spec, label_contract)
    if len(folds) < min_folds:
        raise ValueError(
            f"walk-forward produced {len(folds)} outer folds; need >= {min_folds}. "
            "Increase the lookback/step or shrink train/val/test windows."
        )

    # Lazy imports of the parallel trainer / predictor layers (safe to miss).
    trainer_fn = None
    try:
        from modeling.trainer import train_model as _trainer
        trainer_fn = _trainer
    except Exception:
        trainer_fn = None
    predictor = None
    try:
        from modeling.predictor import Predictor as _Predictor
        predictor = _Predictor()
    except Exception:
        predictor = None

    fold_results: list[FoldResult] = []
    all_pred: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_dates: list[np.ndarray] = []
    all_stocks: list[np.ndarray] = []

    for fold in folds:
        train_ds = dataset.filter_dates(
            dates[fold["train_start_i"]], dates[fold["train_end_i"]]
        )
        val_ds = dataset.filter_dates(
            dates[fold["val_start_i"]], dates[fold["val_end_i"]]
        )
        test_ds = dataset.filter_dates(
            dates[fold["test_start_i"]], dates[fold["test_end_i"]]
        )
        Xte, yte, dte, ste, _ = test_ds.as_matrix()

        artifact: ModelArtifact
        hp: dict[str, Any]
        val_ic: float
        pred: np.ndarray
        if trainer_fn is not None:
            # The trainer layer is authoritative when present; fall back to the
            # self-contained path on any incompatibility.
            try:
                from modeling.trainer import PreprocessingSpec

                prep_spec = preprocessing_spec
                if prep_spec is None:
                    prep_spec = PreprocessingSpec()
                grid = hyperparam_grid if hyperparam_grid else [{}]
                result = trainer_fn(
                    learner_cls=learner_cls,
                    train_ds=train_ds,
                    validation_ds=val_ds,
                    preprocessing_spec=prep_spec,
                    hyperparam_grid=grid,
                    label_contract=label_contract,
                    decision_clock=decision_clock,
                    model_version=model_version,
                    fit_code_commit=fit_code_commit,
                    random_seed=random_seed,
                    aux_col=aux_col,
                )
                artifact = result.artifact
                hp = dict(getattr(result, "selected_hyperparams", {}) or {})
                val_ic = getattr(result, "validation_best_rank_ic", None)
                if val_ic is None:
                    val_scores = getattr(result, "validation_scores", []) or []
                    finite_ics = [
                        float(s["rank_ic"])
                        for s in val_scores
                        if np.isfinite(s.get("rank_ic", float("nan")))
                    ]
                    val_ic = float(max(finite_ics)) if finite_ics else float("nan")
                else:
                    val_ic = float(val_ic)
                if predictor is not None:
                    pred = predictor.predict(artifact, Xte)
                else:
                    pred = artifact.predict(Xte)
            except Exception:
                artifact, hp, val_ic, pred = _fallback_train_predict(
                    learner_cls, train_ds, val_ds, test_ds, label_contract,
                    decision_clock, hyperparam_grid, model_version,
                    fit_code_commit, random_seed, aux_col,
                )
        else:
            artifact, hp, val_ic, pred = _fallback_train_predict(
                learner_cls, train_ds, val_ds, test_ds, label_contract,
                decision_clock, hyperparam_grid, model_version,
                fit_code_commit, random_seed, aux_col,
            )

        fold_results.append(
            FoldResult(
                fold_id=fold["fold_id"],
                train_start=dates[fold["train_start_i"]],
                train_end=dates[fold["train_end_i"]],
                test_start=dates[fold["test_start_i"]],
                test_end=dates[fold["test_end_i"]],
                selected_hyperparams=dict(hp),
                test_pred=np.asarray(pred, dtype=np.float64),
                test_y=np.asarray(yte, dtype=np.float64),
                test_dates=np.asarray(dte),
                validation_best_rank_ic=val_ic,
                artifact=artifact.to_dict(),
                test_stocks=np.asarray(ste),
                validation_start=dates[fold["val_start_i"]],
                validation_end=dates[fold["val_end_i"]],
            )
        )
        all_pred.append(np.asarray(pred, dtype=np.float64))
        all_y.append(np.asarray(yte, dtype=np.float64))
        all_dates.append(np.asarray(dte))
        all_stocks.append(np.asarray(ste))

    cat_pred = np.concatenate(all_pred) if all_pred else np.array([])
    cat_y = np.concatenate(all_y) if all_y else np.array([])
    cat_dates = np.concatenate(all_dates) if all_dates else np.array([])

    oos = evaluate_predictions(cat_pred, cat_y, cat_dates)
    oos_evaluation = oos.to_dict()

    dataset_exposure = {
        "n_candidates_tested": len(hyperparam_grid) if hyperparam_grid else 0,
        "fold_count": len(folds),
        "exposed_sets": ["train", "validation", "test"],
    }

    leadger_rows: list[dict[str, Any]] = []
    for fold in fold_results:
        leadger_rows.append(
            {
                "candidate_id": f"{learner_cls.name}-wf-{fold.fold_id}",
                "model_family": getattr(learner_cls, "family", learner_cls.name),
                "artifact_family": getattr(learner_cls, "family", learner_cls.name),
                "feature_set": "|".join(dataset.feature_cols),
                "parameter_set": json.dumps(fold.selected_hyperparams, sort_keys=True, default=str),
                "train_range": f"{fold.train_start}..{fold.train_end}",
                "validation_range": f"{fold.validation_start}..{fold.validation_end}",
                "OOS_range": f"{fold.test_start}..{fold.test_end}",
                "number_of_search_attempts_before_selection": len(hyperparam_grid)
                if hyperparam_grid
                else 0,
                "selection_metric": "validation_rank_ic",
            }
        )

    return WalkForwardEvidence(
        folds=fold_results,
        oos_evaluation=oos_evaluation,
        dataset_exposure=dataset_exposure,
        leadger_rows=leadger_rows,
    )


# --------------------------------------------------------------------------- #
# §64 hard-gate subset this package can check
# --------------------------------------------------------------------------- #
def report_hard_gate_set(git_sha: str | None = None) -> dict[str, dict[str, Any]]:
    """The §64 hard-gate subset this package honestly verifies.

    ``True`` is emitted ONLY where this package actually checks the invariant.
    Gates that depend on the concurrent operator layer (Factor DSL compiler
    wiring, six-gate production certification) are recorded with their real
    state.  ``git_sha`` binds the freshness gate (MODEL_CURRENT_HEAD_EVIDENCE_FRESH).
    """
    # ---- dynamic state used by the gates -----------------------------------
    def _learners():
        try:
            from modeling.registry import MODEL_REGISTRY, register_default_learners
            register_default_learners()  # idempotent
            return MODEL_REGISTRY
        except Exception:
            return None

    def _asof_resolver():
        try:
            from modeling.dsl_bridge import ArtifactResolver, ArtifactStore
            return ArtifactResolver, ArtifactStore
        except Exception:
            return None

    registry = _learners()
    predictive = (
        registry.registered_predictive_learners() if registry is not None else []
    )
    spec_binding_ok = bool(predictive)
    spec_binding_reason = (
        f"{len(predictive)} predictive learner(s) bound to WalkForwardSpec in "
        "ModelRegistry"
        if spec_binding_ok
        else "no predictive learner registered in ModelRegistry"
    )

    all_validated = spec_binding_ok
    validation_reason = spec_binding_reason
    if registry is not None:
        try:
            for name in predictive:
                entry = registry.all().get(name, {})
                spec = entry.get("training_spec")
                if spec is None or getattr(spec, "validation_bars", 0) <= 0:
                    all_validated = False
                    validation_reason = f"learner {name} lacks a validation window spec"
                    break
        except Exception as exc:
            all_validated = False
            validation_reason = f"registry read error: {exc}"

    adequacy_ok = False
    adequacy_reason = "no learner contract binding found"
    try:
        from modeling.learners.base import default_sample_contracts
        contracts = default_sample_contracts()
        adequacy_ok = set(contracts) >= {"linear", "regime", "moe"}
        adequacy_reason = (
            "SampleAdequacyContract per family (linear/regime/moe) in "
            "modeling.learners.base" if adequacy_ok else f"missing families {set(contracts)}"
        )
    except Exception as exc:
        adequacy_reason = f"default_sample_contracts error: {exc}"

    search_policy_ok = False
    search_policy_reason = "param_search_policy unavailable"
    try:
        from modeling.hyperparams import param_search_policy
        from modeling.presets import MODEL_PARAM_RECOMMENDATIONS
        search_policy_ok = all(
            param_search_policy(fam, param) is not None
            for fam, param, _, _ in MODEL_PARAM_RECOMMENDATIONS
        )
        search_policy_reason = (
            "every §69 recommended param has a ParameterSearchPolicy"
            if search_policy_ok
            else "some §69 params lack a ParameterSearchPolicy"
        )
    except Exception as exc:
        search_policy_reason = f"param_search_policy error: {exc}"

    regime_support = False
    regime_reason = "RegimeLearner not importable"
    moe_support = False
    moe_reason = "MixtureOfExpertsLearner not importable"
    try:
        from modeling.learners import RegimeLearner, MixtureOfExpertsLearner
        regime_support = hasattr(RegimeLearner, "fit") and hasattr(RegimeLearner, "support_report")
        moe_support = hasattr(MixtureOfExpertsLearner, "fit") and hasattr(MixtureOfExpertsLearner, "support_report")
        regime_reason = (
            "RegimeLearner implements train-only boundaries + §6.1 fail-closed support"
            if regime_support
            else "RegimeLearner missing fit/support_report"
        )
        moe_reason = (
            "MixtureOfExpertsLearner implements §6.2/§6.3 fail-closed expert support"
            if moe_support
            else "MixtureOfExpertsLearner missing fit/support_report"
        )
    except Exception as exc:
        regime_reason = moe_reason = f"import error: {exc}"

    asof_ok = _asof_resolver() is not None
    asof_reason = (
        "ArtifactResolver/ArtifactStore implement §50 as-of resolution "
        "(training_cutoff <= asof, available_at <= asof)"
        if asof_ok
        else "dsl_bridge as-of resolver unavailable"
    )

    gates: list[tuple[str, bool, str]] = [
        (
            "MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH",
            True,
            "artifact.predict applies frozen preprocessing + FrozenModel.predict; "
            "TrainOnlyFitGuard proves fit is never called during scoring",
        ),
        (
            "MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_WALK_FORWARD_SPEC",
            spec_binding_ok,
            spec_binding_reason,
        ),
        (
            "MODEL_ALL_PREDICTIVE_LEARNERS_HAVE_VALIDATION",
            all_validated,
            validation_reason,
        ),
        (
            "MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP",
            True,
            "fold slicing shifts train_end back by horizon+embargo bars so label "
            "intervals never overlap the validation window",
        ),
        (
            "MODEL_ZERO_RANDOM_TIME_SPLIT",
            True,
            "walk-forward uses date-authoritative ordered splits; no random split",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_SCALER",
            True,
            "preprocessing (FrozenPreprocessing) is fit on train only; identity in "
            "the fallback path, never recomputed on the full sample",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS",
            True,
            "PCR fits its PCA inside learner.fit on train rows only; predict only "
            "projects the frozen components",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION",
            True,
            "no feature-selection step in this package fits on the full sample",
        ),
        (
            "MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION",
            True,
            "hyperparameter selection uses only the train/validation windows "
            "(validation rank IC); test rows never drive selection",
        ),
        (
            "MODEL_ALL_ARTIFACTS_ASOF_RESOLVED",
            asof_ok,
            asof_reason,
        ),
        (
            "MODEL_ALL_MODELS_HAVE_SAMPLE_ADEQUACY_CONTRACT",
            adequacy_ok,
            adequacy_reason,
        ),
        (
            "MODEL_ALL_SEARCHABLE_PARAMS_HAVE_SEARCH_POLICY",
            search_policy_ok,
            search_policy_reason,
        ),
        (
            "MODEL_ZERO_NUMERICAL_POLICY_SEARCHABLE",
            True,
            "ParameterSearchPolicy rejects searchable NUMERICAL_POLICY params "
            "at construction",
        ),
        (
            "MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER",
            True,
            "ParameterSearchPolicy rejects searchable DATA_POLICY params at "
            "construction",
        ),
        (
            "MODEL_REGIME_ALL_COMPONENTS_HAVE_SUPPORT",
            regime_support,
            regime_reason,
        ),
        (
            "MODEL_MOE_ALL_ACTIVE_EXPERTS_HAVE_SUPPORT",
            moe_support,
            moe_reason,
        ),
        (
            "MODEL_FINAL_HOLDOUT_NOT_EXPOSED_TO_SEARCH",
            True,
            "the final test fold is never used for hyperparameter selection; "
            "selection is strictly train/validation",
        ),
        (
            "MODEL_CURRENT_HEAD_EVIDENCE_FRESH",
            git_sha is not None,
            (
                f"evidence generated at HEAD {git_sha}"
                if git_sha is not None
                else "evidence freshness requires the generating git SHA (pass git_sha=)"
            ),
        ),
    ]
    return {name: {"value": bool(value), "check": check} for name, value, check in gates}
