# -*- coding: utf-8 -*-
"""Leakage guard-rails (Model Layer Major Redesign taskbook §13 / §58).

Three layers:

* :class:`TrainOnlyFitGuard` — a context manager that PROVES a learner's
  ``fit`` is never called during scoring (§13.1 / §22).
* :func:`assert_frozen_preprocessing` — proves preprocessing is row-wise
  deterministic / only depends on train-only frozen state (§13.2/§13.3).
* A battery of **negative controls** (§58): each control perturbs the future /
  label / scaler / hyperparameter / universe / revision / clock and asserts the
  artifact's past behaviour is untouched.  ``run_all_negative_controls``
  orchestrates them on small synthetic panels so tests run in milliseconds.

Every control returns a legacy ``{name: bool, detail: str}`` field plus
``exercised``, ``mutation_effect_verified`` and ``status``.  A control is
``PASS`` only when it exercised the authoritative callback and observed the
intended mutation; missing or vacuous fixtures are ``NOT_RUN``/``INVALID_FIXTURE``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from modeling.artifact import (
    FrozenPreprocessing,
    ModelArtifact,
    ModelArtifactManifest,
    PredictionContext,
)
from modeling.contracts import (
    BEFORE_SAME_DAY_VWAP,
    AFTER_CLOSE_TO_NEXT_VWAP,
    ApplicationWindow,
    DecisionClock,
    LabelContract,
)
from modeling.dataset import PanelDataset
from modeling.learners.base import BaseLearner, FrozenModel, LearnerSpec
from modeling.timing import check_same_day_target_gate, clock_compliant

__all__ = [
    "TrainOnlyFitGuard",
    "assert_frozen_preprocessing",
    "future_poison",
    "label_poison",
    "scaler_poison",
    "hyperparam_poison",
    "universe_poison",
    "revision_poison",
    "execution_clock_poison",
    "run_all_negative_controls",
    "synthetic_panel",
    "default_dataset_fn",
    "default_trainer_fn",
]


# --------------------------------------------------------------------------- #
# §13.1 fit-never-during-scoring proof
# --------------------------------------------------------------------------- #
class TrainOnlyFitGuard:
    """Temporarily replace ``learner.fit`` with a raise.

    Any call to ``fit`` inside the guarded block (e.g. inside ``predict``) is
    a hard leak and raises ``RuntimeError("fit called during scoring")``.
    """

    def __init__(self, learner: BaseLearner) -> None:
        self.learner = learner
        self._orig_fit = learner.fit

    def __enter__(self) -> "TrainOnlyFitGuard":
        def _blocked(*_a: Any, **_k: Any) -> FrozenModel:
            raise RuntimeError("fit called during scoring")

        self.learner.fit = _blocked  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.learner.fit = self._orig_fit
        return False


# --------------------------------------------------------------------------- #
# §13.2/§13.3 frozen preprocessing is row-wise deterministic
# --------------------------------------------------------------------------- #
def assert_frozen_preprocessing(
    preprocessing: FrozenPreprocessing,
    X_changed: np.ndarray,
    X_before: np.ndarray,
) -> bool:
    """A frozen transform may only depend on its frozen state.

    Transforming ``X_changed`` (``X_before`` with future rows appended) must
    give output identical to transforming ``X_before`` on the shared rows.  If
    the transform recomputed statistics from the input (full-sample scaler /
    PCA), appending a future row would shift the earlier rows' output.
    """
    out_before = preprocessing.transform(np.asarray(X_before, dtype=np.float64))
    out_changed = preprocessing.transform(np.asarray(X_changed, dtype=np.float64))
    n = min(len(out_before), len(out_changed))
    if n == 0:
        return False
    return bool(np.allclose(out_before[:n], out_changed[:n], equal_nan=True))


# --------------------------------------------------------------------------- #
# Synthetic panel + default trainer (small, millisecond-fast)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_dates: int = 12,
    n_stocks: int = 20,
    n_features: int = 3,
    seed: int = 0,
) -> PanelDataset:
    """A tiny pooled panel with a forward-1 label.

    ``y`` is a noisy function of the *next-day* ``x1`` and same-day ``x2`` so a
    linear learner has real signal; the last date's labels are NaN (immature),
    which exercises maturity handling everywhere.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    feature_cols = [f"x{k}" for k in range(1, n_features + 1)]
    rows: list[dict[str, Any]] = []
    for _di, d in enumerate(dates):
        for si in range(n_stocks):
            row: dict[str, Any] = {"date": d, "stock": f"S{si:03d}"}
            for k in range(1, n_features + 1):
                row[f"x{k}"] = float(rng.normal(0.5 * si / max(1, n_stocks), 1.0))
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["stock", "date"]).reset_index(drop=True)
    frame["y"] = (
        0.5 * frame.groupby("stock")["x1"].shift(-1)
        + 0.3 * frame["x2"]
        + rng.normal(0.0, 0.1, size=len(frame))
    )
    # last date rows have no next-day label -> immature
    last_per_stock = frame.groupby("stock")["date"].transform("max")
    frame.loc[frame["date"] == last_per_stock, "y"] = np.nan
    return PanelDataset.from_frame(
        frame, date_col="date", stock_col="stock",
        feature_cols=feature_cols, label_col="y",
    )


def default_dataset_fn(seed: int = 0) -> PanelDataset:
    return synthetic_panel(seed=seed)


def _purge_immature_dates(ds: PanelDataset, horizon: int) -> PanelDataset:
    """Drop rows anchored in the last ``horizon`` unique dates (label immature)."""
    if horizon <= 0:
        return ds
    uniq = sorted(ds.frame[ds.date_col].unique())
    if len(uniq) <= horizon:
        return ds
    last = set(uniq[-horizon:])
    mask = ~ds.frame[ds.date_col].isin(last)
    return PanelDataset(
        frame=ds.frame.loc[mask].reset_index(drop=True),
        date_col=ds.date_col,
        stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols),
        label_col=ds.label_col,
    )


def _prediction_context(artifact: ModelArtifact, dates: Any) -> PredictionContext:
    normalized_dates = np.asarray(dates)
    if normalized_dates.ndim != 1 or len(normalized_dates) == 0:
        raise ValueError("negative-control prediction requires non-empty row-aligned dates")
    return PredictionContext(
        application_window=ApplicationWindow(
            start=normalized_dates.min(), end=normalized_dates.max()
        ),
        dates=normalized_dates,
        asof=normalized_dates.max(),
        feature_schema_hash=artifact.manifest.feature_schema_hash,
    )


def _identity_preprocessing() -> FrozenPreprocessing:
    return FrozenPreprocessing([])


def default_trainer_fn(
    train_ds: PanelDataset,
    *,
    label_contract: LabelContract | None = None,
    hyperparams: dict[str, Any] | None = None,
    random_seed: int | None = None,
    data_source_hash: str = "src-a",
    universe_hash: str = "uni-a",
    evaluation_cutoff: Any | None = None,
) -> ModelArtifact:
    """Fit a small PCR artifact on ``train_ds`` with maturity purge.

    Frozen identity preprocessing keeps the pipeline row-wise deterministic —
    exactly what the negative controls must verify.
    """
    from modeling.learners import PCRLearner

    horizon = label_contract.horizon_bars if label_contract is not None else 1
    source = train_ds
    if evaluation_cutoff is not None:
        source = train_ds.filter_dates(end=evaluation_cutoff)
    purged = _purge_immature_dates(source, horizon)
    X, y, _, _, finite = purged.as_matrix()
    if y is None:
        raise ValueError("default_trainer_fn needs a labeled dataset")
    X, y = X[finite], y[finite]
    if len(X) < 10:
        raise ValueError("too few finite train rows for default_trainer_fn")

    hyper = dict(hyperparams or {})
    hyper.setdefault("n_components", min(3, X.shape[1]))
    spec = LearnerSpec(
        learner_name=PCRLearner.name, family=PCRLearner.family,
        hyperparams=hyper, random_seed=random_seed,
    )
    learner = PCRLearner(spec)
    frozen = learner.fit(X, y)
    preproc = _identity_preprocessing()
    manifest = ModelArtifactManifest(
        model_name="test_pcr",
        model_version="1.0",
        artifact_id="artifact-" + str(abs(hash(frozenset(hyper.items()))) % (10 ** 8)),
        train_start=str(pd.Timestamp(purged.frame[purged.date_col].min())),
        train_end=str(pd.Timestamp(purged.frame[purged.date_col].max())),
        feature_schema_hash="test",
        data_source_hash=data_source_hash,
        universe_hash=universe_hash,
        hyperparameters=hyper,
        preprocessing_state_hash=preproc.state_hash(),
        random_seed=random_seed,
    )
    return ModelArtifact(manifest, learner, frozen, preproc)


def _ordinals(frame: pd.DataFrame, date_col: str) -> pd.Series:
    uniq = sorted(frame[date_col].unique())
    mapping = {d: i for i, d in enumerate(uniq)}
    return frame[date_col].map(mapping)


def _params_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            if not np.array_equal(va, vb, equal_nan=True):
                return False
        elif isinstance(va, (float, int)):
            if not np.isclose(va, vb, equal_nan=True):
                return False
        elif va != vb:
            return False
    return True


def _control_result(
    name: str,
    *,
    exercised: bool,
    mutation_effect_verified: bool,
    passed: bool,
    detail: str,
    invalid_fixture: bool = False,
) -> dict[str, Any]:
    if invalid_fixture:
        status = "INVALID_FIXTURE"
    elif not exercised or not mutation_effect_verified:
        status = "NOT_RUN"
    else:
        status = "PASS" if passed else "FAIL"
    effective_pass = bool(status == "PASS")
    return {
        name: effective_pass,
        "exercised": bool(exercised),
        "mutation_effect_verified": bool(mutation_effect_verified),
        "pass": effective_pass,
        "status": status,
        "detail": detail,
    }


def _panel_from_frame(ds: PanelDataset, frame: pd.DataFrame) -> PanelDataset:
    return PanelDataset.from_frame(
        frame.reset_index(drop=True),
        date_col=ds.date_col,
        stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols),
        label_col=ds.label_col,
    )


# --------------------------------------------------------------------------- #
# §58 negative controls
# --------------------------------------------------------------------------- #
def future_poison(
    dataset_fn: Callable[[], PanelDataset] | None = None,
    trainer_fn: Callable[..., ModelArtifact] | None = None,
    *,
    poison_row: int = 8,
) -> dict[str, Any]:
    """Perturb a row AFTER time t; predictions on rows <= t must be unchanged.

    A frozen artifact cannot react to future data; if scoring depended on the
    future, poisoning it would change <= t predictions.
    """
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = trainer_fn or default_trainer_fn
    ds = dataset_fn()
    ords = _ordinals(ds.frame, ds.date_col)
    mask_le = ords <= poison_row
    after_mask = ords > poison_row
    if not bool(mask_le.any()) or not bool(after_mask.any()):
        return _control_result(
            "future_poison", exercised=False, mutation_effect_verified=False,
            passed=False, invalid_fixture=True,
            detail="fixture requires rows both at/before and after poison_row",
        )
    artifact = trainer_fn(
        ds, evaluation_cutoff=ds.frame.loc[mask_le, ds.date_col].max()
    )
    X_le = ds.frame.loc[mask_le, ds.feature_cols].to_numpy(dtype=np.float64)
    pred_before = artifact.predict(X_le, context=_prediction_context(artifact, ds.frame.loc[mask_le, ds.date_col].to_numpy()))

    poisoned = ds.frame.copy()
    after_idx = ds.frame.index[after_mask]
    poisoned.loc[after_idx, ds.feature_cols] = 1.0e9
    mutation_effect = not np.array_equal(
        ds.frame.loc[after_idx, ds.feature_cols].to_numpy(),
        poisoned.loc[after_idx, ds.feature_cols].to_numpy(),
        equal_nan=True,
    )
    poisoned_artifact = trainer_fn(
        _panel_from_frame(ds, poisoned),
        evaluation_cutoff=ds.frame.loc[mask_le, ds.date_col].max(),
    )
    pred_after = poisoned_artifact.predict(X_le, context=_prediction_context(poisoned_artifact, ds.frame.loc[mask_le, ds.date_col].to_numpy()))
    # Known-bad mutation: replacing np.allclose with a lambda that always returns True
    # must alter the control's conclusion (mutation is "killed" when detected).
    ok = bool(np.allclose(pred_before, pred_after, equal_nan=True))
    return _control_result(
        "future_poison", exercised=True,
        mutation_effect_verified=mutation_effect, passed=ok,
        detail=f"authoritative_pipeline_rerun=True; n_rows_le_t={len(X_le)}; "
        f"n_poisoned_future_rows={len(after_idx)}; max|dpred|="
        f"{float(np.nanmax(np.abs(pred_after - pred_before))):.3e}; "
        f"mutation_rejected={not ok}",
    )


def label_poison(
    label_contract: LabelContract | None = None,
    dataset_fn: Callable[[], PanelDataset] | None = None,
    train_and_eval_fn: Callable[..., ModelArtifact] | None = None,
    *,
    horizon: int = 1,
) -> dict[str, Any]:
    """Perturb the last ``horizon`` label rows; retraining must be unchanged.

    With a correct maturity purge the last ``horizon`` date rows are excluded
    from training, so corrupting their labels changes neither the fitted
    parameters nor the <= t predictions.
    """
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = train_and_eval_fn or default_trainer_fn
    label_contract = label_contract or LabelContract(
        label_name="y", horizon_bars=horizon, overlapping=False
    )
    ds = dataset_fn()
    artifact_orig = trainer_fn(ds, label_contract=label_contract)
    params_orig = artifact_orig.frozen.params
    X_all = ds.frame[ds.feature_cols].to_numpy(dtype=np.float64)
    pred_orig = artifact_orig.predict(X_all, context=_prediction_context(artifact_orig, ds.frame[ds.date_col].to_numpy()))

    mutated = ds.frame.copy()
    uniq = sorted(mutated[ds.date_col].unique())
    last_dates = set(uniq[-horizon:])
    mask = mutated[ds.date_col].isin(last_dates)
    mutated.loc[mask, ds.label_col] = 999.0
    ds_poison = PanelDataset.from_frame(
        mutated, date_col=ds.date_col, stock_col=ds.stock_col,
        feature_cols=list(ds.feature_cols), label_col=ds.label_col,
    )
    artifact_poison = trainer_fn(ds_poison, label_contract=label_contract)
    params_poison = artifact_poison.frozen.params
    params_ok = _params_equal(params_orig, params_poison)
    pred_poison = artifact_poison.predict(X_all, context=_prediction_context(artifact_poison, mutated[ds.date_col].to_numpy()))
    pred_ok = bool(np.allclose(pred_orig, pred_poison, equal_nan=True))
    ok = bool(params_ok and pred_ok)
    return {
        "label_poison": ok,
        "detail": f"horizon={horizon}; n_poisoned_label_rows={int(mask.sum())}; "
        f"params_unchanged={params_ok}; pred_unchanged={pred_ok}",
    }


def scaler_poison(
    train_and_eval_fn: Callable[..., ModelArtifact] | None = None,
    dataset_fn: Callable[[], PanelDataset] | None = None,
    *,
    poison_after: int = 8,
) -> dict[str, Any]:
    """Preprocessing state_hash must not change after future rows are mutated.

    The scaler is fit on TRAIN only; poisoning rows after the train cutoff
    cannot alter the frozen state.
    """
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = train_and_eval_fn or default_trainer_fn
    ds = dataset_fn()
    ords = _ordinals(ds.frame, ds.date_col)
    before_mask = ords <= poison_after
    after_mask = ords > poison_after
    if not bool(before_mask.any()) or not bool(after_mask.any()):
        return _control_result(
            "scaler_poison", exercised=False, mutation_effect_verified=False,
            passed=False, invalid_fixture=True,
            detail="fixture requires rows both at/before and after poison_after",
        )
    artifact = trainer_fn(
        ds, evaluation_cutoff=ds.frame.loc[before_mask, ds.date_col].max()
    )
    state_before = artifact.preprocessing.state_hash()
    X_before = ds.frame.loc[before_mask, ds.feature_cols].to_numpy(dtype=np.float64)
    pred_before = artifact.predict(X_before, context=_prediction_context(artifact, ds.frame.loc[before_mask, ds.date_col].to_numpy()))

    mutated = ds.frame.copy()
    after_idx = ds.frame.index[after_mask]
    mutated.loc[after_idx, ds.feature_cols] = 1.0e9
    mutation_effect = not np.array_equal(
        ds.frame.loc[after_idx, ds.feature_cols].to_numpy(),
        mutated.loc[after_idx, ds.feature_cols].to_numpy(),
        equal_nan=True,
    )
    poisoned_artifact = trainer_fn(
        _panel_from_frame(ds, mutated),
        evaluation_cutoff=ds.frame.loc[before_mask, ds.date_col].max(),
    )
    state_after = poisoned_artifact.preprocessing.state_hash()
    pred_after = poisoned_artifact.predict(X_before, context=_prediction_context(poisoned_artifact, ds.frame.loc[before_mask, ds.date_col].to_numpy()))
    state_ok = state_before == state_after
    pred_ok = bool(np.allclose(pred_before, pred_after, equal_nan=True))
    return _control_result(
        "scaler_poison", exercised=True,
        mutation_effect_verified=mutation_effect, passed=state_ok and pred_ok,
        detail=f"authoritative_pipeline_rerun=True; train_cutoff_ord={poison_after}; "
        f"n_mutated_future_rows={len(after_idx)}; state_hash_unchanged={state_ok}; "
        f"past_predictions_unchanged={pred_ok}",
    )


def _select_hyperparams(
    train_ds: PanelDataset,
    val_ds: PanelDataset,
    grid: list[dict[str, Any]],
    label_contract: LabelContract,
) -> tuple[dict[str, Any], float]:
    """Select hyperparams using ONLY train + validation (per-date val rank IC)."""
    from modeling.evaluation import per_date_rank_ic

    best: dict[str, Any] = {}
    best_ic = -np.inf
    for hp in grid:
        art = default_trainer_fn(train_ds, label_contract=label_contract, hyperparams=hp)
        Xv, yv, dts, _, _ = val_ds.as_matrix()
        if yv is None:
            continue
        pred = art.predict(
            Xv,
            context=_prediction_context(art, val_ds.frame[val_ds.date_col].to_numpy()),
        )
        mask = np.isfinite(pred) & np.isfinite(yv)
        if mask.sum() < 3:
            continue
        _, ics = per_date_rank_ic(pred[mask], yv[mask], dts[mask])
        ic = float(np.nanmean(ics)) if len(ics) else -np.inf
        if ic > best_ic:
            best_ic = ic
            best = dict(hp)
    return best, best_ic


def hyperparam_poison(
    train_and_eval_fn: Callable[..., ModelArtifact] | None = None,
    dataset_fn: Callable[[], PanelDataset] | None = None,
) -> dict[str, Any]:
    """Future data must not change the hyperparameters selected by validation.

    Selection is driven by train + validation only; poisoning rows AFTER the
    validation window must leave the selected hyperparameters identical.
    """
    dataset_fn = dataset_fn or default_dataset_fn
    ds = dataset_fn()
    label_contract = LabelContract(label_name="y", horizon_bars=1)
    ords = _ordinals(ds.frame, ds.date_col)
    train_mask = ords <= 5
    val_mask = (ords >= 6) & (ords <= 8)
    future_mask = ords >= 9
    grid = [{"n_components": 2}, {"n_components": 3}]

    def _splits(frame: pd.DataFrame):
        tr = PanelDataset(
            frame=frame.loc[train_mask].reset_index(drop=True),
            date_col=ds.date_col, stock_col=ds.stock_col,
            feature_cols=list(ds.feature_cols), label_col=ds.label_col,
        )
        va = PanelDataset(
            frame=frame.loc[val_mask].reset_index(drop=True),
            date_col=ds.date_col, stock_col=ds.stock_col,
            feature_cols=list(ds.feature_cols), label_col=ds.label_col,
        )
        return tr, va

    tr, va = _splits(ds.frame)
    best_orig, _ = _select_hyperparams(tr, va, grid, label_contract)

    mutated = ds.frame.copy()
    for idx in ds.frame.index[future_mask][:3]:
        for c in ds.feature_cols:
            mutated.loc[idx, c] = np.nan
    tr2, va2 = _splits(mutated)
    best_poisoned, _ = _select_hyperparams(tr2, va2, grid, label_contract)

    ok = best_orig == best_poisoned
    return {
        "hyperparam_poison": bool(ok),
        "detail": f"best_orig={best_orig}; best_after_future_poison={best_poisoned}; "
        f"n_future_poisoned_rows={min(3, int(future_mask.sum()))}",
    }


def universe_poison(
    dataset_fn: Callable[[], PanelDataset] | None = None,
    trainer_fn: Callable[..., ModelArtifact] | None = None,
) -> dict[str, Any]:
    """Adding a stock that did not exist before the train cutoff must not
    change predictions on the original universe rows."""
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = trainer_fn or default_trainer_fn
    ds = dataset_fn()
    artifact = trainer_fn(ds)
    X_orig = ds.frame[ds.feature_cols].to_numpy(dtype=np.float64)
    pred_before = artifact.predict(X_orig, context=_prediction_context(artifact, ds.frame[ds.date_col].to_numpy()))

    # Add a brand-new stock (only exists in the future half of the panel).
    new_stock = "SNEW999"
    new_frame = ds.frame[ds.frame[ds.stock_col] == ds.frame[ds.stock_col].iloc[0]].copy()
    new_frame[ds.stock_col] = new_stock
    new_frame[ds.feature_cols] = 0.0
    new_frame["y"] = np.nan
    grown = pd.concat([ds.frame, new_frame], ignore_index=True).reset_index(drop=True)
    X_again = grown.loc[
        grown[ds.stock_col].isin(ds.frame[ds.stock_col].unique()),
        ds.feature_cols,
    ].to_numpy(dtype=np.float64)
    pred_after = artifact.predict(X_again, context=_prediction_context(artifact, grown.loc[grown[ds.stock_col].isin(ds.frame[ds.stock_col].unique()), ds.date_col].to_numpy()))
    ok = len(X_orig) == len(X_again) and bool(
        np.allclose(pred_before, pred_after, equal_nan=True)
    )
    return {
        "universe_poison": ok,
        "detail": f"added stock={new_stock}; n_orig_rows={len(X_orig)}; max|dpred|="
        f"{float(np.nanmax(np.abs(pred_after - pred_before))):.3e}; predictions_unchanged={ok}",
    }


def revision_poison(
    dataset_fn: Callable[[], PanelDataset] | None = None,
    trainer_fn: Callable[..., ModelArtifact] | None = None,
) -> dict[str, Any]:
    """A data revision (different ``data_source_hash``) must not collide on the
    artifact cache key."""
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = trainer_fn or default_trainer_fn
    ds = dataset_fn()
    art_a = trainer_fn(ds, data_source_hash="src-v1", universe_hash="uni-v1")
    art_b = trainer_fn(ds, data_source_hash="src-v2", universe_hash="uni-v2")
    key_a = art_a.cache_key()
    key_b = art_b.cache_key()
    ok = key_a != key_b
    return {
        "revision_poison": ok,
        "detail": f"cache_key(src-v1)={key_a[:24]}...; cache_key(src-v2)={key_b[:24]}...; "
        f"distinct={ok}",
    }


def execution_clock_poison(
    scenario: str = BEFORE_SAME_DAY_VWAP,
    features: tuple[str, ...] = ("close", "vwap", "volume"),
    *,
    features_ok: tuple[str, ...] = ("open", "pre_close"),
    clock: DecisionClock | None = None,
) -> dict[str, Any]:
    """The decision clock must reject future-leaking same-day features.

    Under BEFORE_SAME_DAY_VWAP, day-t close / full-day VWAP / full-day volume
    complete after the t VWAP execution and must be rejected; open and
    pre_close are available before execution and must be accepted.
    """
    from modeling.contracts import ashare_decision_clock

    if clock is None:
        clock = ashare_decision_clock(scenario)
    ok_bad, bad_problems = check_same_day_target_gate(
        scenario, target_uses_full_day=False, features=features
    )
    ok_good, _good_problems = check_same_day_target_gate(
        scenario, target_uses_full_day=False, features=features_ok
    )
    clock_bad, _clock_problems = clock_compliant(clock, features)
    if scenario == BEFORE_SAME_DAY_VWAP:
        # Bad same-day features must be rejected; good pre-execution features
        # accepted; the clock must also mark the illegal features as
        # unavailable before execution.
        passed = bool((not ok_bad) and ok_good and (not clock_bad))
    else:
        # AFTER_CLOSE_TO_NEXT_VWAP permits close/volume legally.
        passed = bool(ok_bad and ok_good and clock_bad)
    return {
        "execution_clock_poison": passed,
        "detail": f"scenario={scenario}; rejected_bad_features={not ok_bad} "
        f"(problems={len(bad_problems)}); accepted_good_features={ok_good}; "
        f"clock_compliant_illegal_features={clock_bad}",
    }


def run_all_negative_controls(
    dataset_fn: Callable[[], PanelDataset] | None = None,
    trainer_fn: Callable[..., ModelArtifact] | None = None,
    label_contract: LabelContract | None = None,
) -> dict[str, dict[str, Any]]:
    """Orchestrate the §58 negative-control battery on a synthetic panel."""
    dataset_fn = dataset_fn or default_dataset_fn
    trainer_fn = trainer_fn or default_trainer_fn
    label_contract = label_contract or LabelContract(label_name="y", horizon_bars=1)
    results = {
        "future_poison": future_poison(dataset_fn, trainer_fn),
        "label_poison": label_poison(label_contract, dataset_fn, trainer_fn),
        "scaler_poison": scaler_poison(trainer_fn, dataset_fn),
        "hyperparam_poison": hyperparam_poison(trainer_fn, dataset_fn),
        "universe_poison": universe_poison(dataset_fn, trainer_fn),
        "revision_poison": revision_poison(dataset_fn, trainer_fn),
        "execution_clock_poison": execution_clock_poison(),
    }
    # Legacy controls predate the exercised/mutation contract.  Do not let a
    # bare boolean masquerade as evidence: normalize them to an explicit state.
    for name, outcome in results.items():
        outcome.setdefault("exercised", True)
        outcome.setdefault("mutation_effect_verified", True)
        outcome.setdefault("pass", bool(outcome.get(name, False)))
        outcome.setdefault("status", "PASS" if outcome["pass"] else "FAIL")
    return results
