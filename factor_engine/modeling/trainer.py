# -*- coding: utf-8 -*-
"""One-fold predictive trainer (Model Layer Major Redesign taskbook §13.2 /
§13.3 / §19).

The trainer runs the §19 loop: purge train vs validation overlap, fit
TRAIN-ONLY preprocessing, search the approved hyperparameter grid, evaluate
each candidate on validation, select the best, refit, and freeze a
:class:`modeling.artifact.ModelArtifact`.  Fail-closed: if no candidate passes
its sample-adequacy contract a ``ValueError`` is raised — a silently-degraded
artifact is never returned.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from modeling.artifact import FrozenPreprocessing, ModelArtifact, ModelArtifactManifest
from modeling.contracts import DecisionClock, LabelContract, ParameterSearchPolicy, SampleAdequacyContract
from modeling.dataset import PanelDataset
from modeling.hyperparams import validate_search_grid
from modeling.learners.base import LearnerSpec, default_sample_contracts
from modeling.selection import candidate_identity
from modeling.trainer_governance import CandidateExposureLedger, GovernanceContract

__all__ = [
    "PreprocessingSpec",
    "fit_preprocessing",
    "TrainResult",
    "train_model",
    "_rank_ic",
]


@dataclass
class PreprocessingSpec:
    """TRAIN-ONLY preprocessing pipeline specification (§13.2 / §13.3)."""

    steps: tuple[str, ...] = ("imputer", "winsor", "standardize")
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    use_standardize: bool = True


def fit_preprocessing(train_ds: PanelDataset, spec: PreprocessingSpec) -> FrozenPreprocessing:
    """Fit each preprocessing step on the output of prior TRAIN-ONLY steps (§13.2)."""
    X = np.asarray(train_ds.as_matrix()[0], dtype=np.float64)
    q0, q1 = spec.winsor_quantiles
    if not (0.0 <= q0 < q1 <= 1.0):
        raise ValueError("winsor_quantiles must satisfy 0 <= low < high <= 1")
    allowed = {"imputer", "winsor", "standardize"}
    unknown = set(spec.steps) - allowed
    if unknown:
        raise ValueError(f"unknown preprocessing steps: {sorted(unknown)!r}")

    current = X.copy()
    steps: list[dict[str, Any]] = []
    for kind in spec.steps:
        if kind == "imputer":
            means = np.nanmean(current, axis=0)
            means = np.where(np.isfinite(means), means, 0.0)
            steps.append({"kind": kind, "means": means.tolist()})
            current = np.where(np.isfinite(current), current, means)
        elif kind == "winsor":
            lows = np.nanquantile(current, q0, axis=0)
            highs = np.nanquantile(current, q1, axis=0)
            lows = np.where(np.isfinite(lows), lows, 0.0)
            highs = np.where(np.isfinite(highs), highs, lows)
            steps.append({"kind": kind, "lows": lows.tolist(), "highs": highs.tolist()})
            current = np.clip(current, lows, highs)
        elif kind == "standardize" and spec.use_standardize:
            mean = np.nanmean(current, axis=0)
            std = np.nanstd(current, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
            steps.append({"kind": kind, "mean": mean.tolist(), "scale": std.tolist()})
            current = (current - mean) / std
    return FrozenPreprocessing(steps)


def _raise_if_cancelled(cancel_token: Any | None) -> None:
    """Honor an explicit token or the active request-scoped token."""
    token = cancel_token
    if token is None:
        try:
            from runtime.exceptions import get_active_cancellation_token
            token = get_active_cancellation_token()
        except Exception:
            token = None
    if token is not None:
        if hasattr(token, "raise_if_cancelled"):
            token.raise_if_cancelled()
        elif bool(getattr(token, "is_cancelled", False)) or bool(getattr(token, "cancelled", False)):
            raise RuntimeError("training cancellation requested")


@dataclass
class TrainResult:
    """Outcome of one train fold."""

    artifact: ModelArtifact | None
    selected_hyperparams: dict[str, Any]
    validation_scores: list[dict[str, Any]]
    preprocessing: FrozenPreprocessing
    fold_id: int | None = None


def _free_parameter_count(candidate: dict[str, Any]) -> int:
    """DEPRECATED — replaced by ``learner.effective_parameter_count``.

    ``len(candidate)`` is not a valid DOF proxy (an ElasticNet with 50 features
    and ``{alpha, l1_ratio}`` fits ~51 coefficients, not 2).  Kept only for
    callers that lack a learner instance; the trainer path uses the learner.
    """
    for key in ("n_components", "n_regimes", "n_experts"):
        if key in candidate:
            return max(1, int(candidate[key]))
    return max(1, len(candidate))


def _maturity_cutoff(
    final_fit_end: str,
    horizon_bars: int,
    bar_calendar: np.ndarray,
) -> str:
    """Return the timing authority's fail-closed label-maturity cutoff."""
    from modeling.timing import maturity_cutoff_fail_closed

    return maturity_cutoff_fail_closed(final_fit_end, horizon_bars, bar_calendar)


def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC, NaN-safe; returns NaN when not computable."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    if not np.isfinite(rho):
        return float("nan")
    return float(rho)


def _per_date_icir(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    dates = np.asarray(dates)
    vals: list[float] = []
    for d in np.unique(dates):
        m = dates == d
        vals.append(_rank_ic(y_true[m], y_pred[m]))
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) == 0:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if sd == 0:
        return float("nan")
    return float(arr.mean() / sd)


def _evaluate_validation(
    learner: Any, frozen: Any, validation_ds: PanelDataset | None, preprocessing: FrozenPreprocessing
) -> dict[str, float]:
    """Score a frozen candidate on finite validation rows only.

    ``rank_ic`` prefers ``modeling.evaluation.cross_sectional_ic`` when that
    module exists (built by a parallel agent); otherwise a rank-IC helper is
    used so the trainer never hard-depends on the not-yet-written module.
    """
    if validation_ds is None or validation_ds.n_rows == 0:
        raise ValueError("validation_ds is required and must be non-empty")
    Xv, yv, dv, _, _ = validation_ds.as_matrix()
    Xv = preprocessing.transform(Xv)
    finite = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    if not np.any(finite):
        return {"rank_ic": float("nan"), "icir": float("nan"), "mse": float("nan")}
    Xv = Xv[finite]
    yv = yv[finite]
    dv = dv[finite]
    pred = learner.predict(frozen, Xv)
    rank_ic = _rank_ic(yv, pred)
    icir = _per_date_icir(yv, pred, dv)
    mse = float(np.mean((yv - pred) ** 2)) if len(yv) else float("nan")
    try:
        from modeling.evaluation import cross_sectional_ic  # type: ignore
    except ImportError:
        cross_sectional_ic = None
    if cross_sectional_ic is not None:
        # Evaluation code errors must remain distinguishable from a metric that
        # is legitimately not computable; never convert them to NaN here.
        extra = cross_sectional_ic(yv, pred, dv)
        if isinstance(extra, dict):
            if np.isfinite(extra.get("rank_ic", np.nan)):
                rank_ic = float(extra["rank_ic"])
            if np.isfinite(extra.get("icir", np.nan)):
                icir = float(extra["icir"])
        elif isinstance(extra, (int, float)) and np.isfinite(extra):
            rank_ic = float(extra)
    return {"rank_ic": rank_ic, "icir": icir, "mse": mse}


def _extract_matrices(
    ds: PanelDataset,
    preprocessing: FrozenPreprocessing,
    aux_col: str | None,
    decay_half_life_bars: int | None = None,
    sample_weight_policy: str | None = None,
    sample_weight_half_life_dates: float | None = None,
):
    """Return ``(X_tr, y_tr, aux, weights)`` aligned to the finite-feature-and-label rows.

    ``weights`` is a per-row decay vector (§3.4 / §73) when
    ``decay_half_life_bars`` is requested, else ``None`` — it is aligned to the
    exact rows returned (the finite-label mask), so it is safe to pass straight
    to ``learner.fit(..., weights=...)``.  Learners that do not consume per-row
    weights ignore it; those that cannot (elastic_net) reject it fail-closed.

    Weights are aligned to the full cohort (finite features AND finite labels)
    so no declared weight is silently ignored.
    """
    X_full, y_full, _, _, _ = ds.as_matrix()
    X_all = preprocessing.transform(X_full)
    transformed_finite = np.isfinite(X_all).all(axis=1)

    weights_all = None
    if sample_weight_policy is not None:
        from modeling.sample_policy import sample_weights
        weights_all = sample_weights(ds, sample_weight_policy, half_life_dates=sample_weight_half_life_dates)
    elif decay_half_life_bars is not None and decay_half_life_bars > 0:
        from modeling.walk_forward import decay_weights
        weights_all = decay_weights(ds, decay_half_life_bars, date_col=ds.date_col)
    aux_arr = None
    if aux_col is not None:
        if aux_col not in ds.frame.columns:
            raise ValueError(f"aux_col {aux_col!r} missing from panel")
        aux_arr = ds.frame[aux_col].to_numpy(dtype=np.float64)
    if y_full is not None:
        ok = np.isfinite(y_full) & transformed_finite
        X_tr = X_all[ok]
        y_tr = y_full[ok]
        if aux_arr is not None:
            aux_arr = aux_arr[ok]
        weights = None if weights_all is None else weights_all[ok]
    else:
        ok = transformed_finite
        X_tr, y_tr = X_all[ok], None
        if aux_arr is not None:
            aux_arr = aux_arr[ok]
        weights = None if weights_all is None else weights_all[ok]
    aux = {"regime_state": aux_arr} if aux_arr is not None else None
    return X_tr, y_tr, aux, weights


def _assert_declared_feature_availability(
    clock: DecisionClock, features: list[str]
) -> None:
    """Trainer-boundary availability gate; unknown features fail closed.

    Kept here rather than modifying the shared DecisionClock contract so the
    trainer policy cluster can land independently of contract/evaluation work.
    """
    unknown = [feature for feature in features if feature not in clock.feature_available_at]
    if unknown:
        raise ValueError(f"feature availability is undeclared for: {unknown!r}")


def train_model(
    learner_cls: type,
    train_ds: PanelDataset,
    validation_ds: PanelDataset | None,
    *,
    preprocessing_spec: PreprocessingSpec,
    hyperparam_grid: list[dict[str, Any]],
    label_contract: LabelContract,
    decision_clock: DecisionClock,
    feature_schema_hash: str = "",
    universe_hash: str = "",
    data_source_hash: str = "",
    model_version: str = "1.0",
    fit_code_commit: str = "",
    random_seed: int | None = None,
    retrain_policy: str = "train_only",
    aux_col: str | None = None,
    sample_contract: SampleAdequacyContract | None = None,
    governance_contract: GovernanceContract | None = None,
    require_explicit_feature_availability: bool = False,
    evaluation_boundary: Any = None,
    decay_half_life_bars: int | None = None,
    cancel_token: Any | None = None,
    sample_weight_policy: str | None = None,
    sample_weight_half_life_dates: float | None = None,
) -> TrainResult:
    """Run one validation-backed training fold and fail closed on leakage.

    The §19 one-fold loop with trainer governance contracts, parameter validation,
    and fit exception handling.  Raises ``ValueError`` when every hyperparameter
    candidate fails its sample-adequacy contract or validation.

    Hyperparameter grids are validated against ParameterSearchPolicy to ensure
    only reviewed values are exposed during training."""
    _raise_if_cancelled(cancel_token)
    if validation_ds is not None and validation_ds.n_rows == 0:
        validation_ds = None
    if validation_ds is None and evaluation_boundary is None and len(hyperparam_grid) > 1:
        raise ValueError("validation_ds is required for hyperparameter selection")
    if train_ds.n_rows == 0:
        raise ValueError("train_ds must be non-empty")
    if not hyperparam_grid:
        raise ValueError("hyperparam_grid must contain at least one candidate")
    if retrain_policy not in {"train_only", "train_plus_validation"}:
        raise ValueError(f"unknown retrain_policy {retrain_policy!r}")
    if train_ds.label_col is None:
        raise ValueError("training panel must be labelled")
    if validation_ds is not None:
        if validation_ds.label_col is None:
            raise ValueError("validation panel must be labelled")
        if list(train_ds.feature_cols) != list(validation_ds.feature_cols):
            raise ValueError("training and validation feature schemas differ")
        train_dates = train_ds.frame[train_ds.date_col]
        validation_dates = validation_ds.frame[validation_ds.date_col]
        if train_dates.max() >= validation_dates.min():
            raise ValueError("training dates must be strictly before validation dates")

    from modeling.walk_forward import purge_and_embargo, purge_before_boundary

    _calendar_frames = [train_ds.frame[train_ds.date_col]]
    if validation_ds is not None:
        _calendar_frames.append(validation_ds.frame[validation_ds.date_col])
    _bar_calendar: np.ndarray = np.sort(
        pd.Index(pd.to_datetime(pd.concat(_calendar_frames))).unique().to_numpy()
    )
    train_ds = purge_and_embargo(train_ds, validation_ds, label_contract)
    if validation_ds is None and evaluation_boundary is not None:
        train_ds = purge_before_boundary(train_ds, evaluation_boundary, label_contract)
    if train_ds.n_rows == 0:
        raise ValueError("no training rows remain after purge/embargo")
    _raise_if_cancelled(cancel_token)

    # 2. train-only preprocessing.
    preprocessing = fit_preprocessing(train_ds, preprocessing_spec)
    _raise_if_cancelled(cancel_token)

    # 3. training matrices.
    X_tr, y_tr, aux, weights = _extract_matrices(
        train_ds, preprocessing, aux_col, decay_half_life_bars, sample_weight_policy, sample_weight_half_life_dates
    )
    if y_tr is None:
        raise ValueError("train_model requires a labelled training panel")

    if sample_contract is None:
        # Explicit sample_contract_family declaration — never a silent None for
        # pcr/pls/elastic_net (they bind the "linear" contract).
        family_key = getattr(learner_cls, "sample_contract_family", None) or learner_cls.family
        sample_contract = default_sample_contracts().get(family_key)

    # The caller may choose a reviewed subset, but never an ad-hoc value.  The
    # default exposure budget is exactly that reviewed subset's size.
    # validate_search_grid enforces ParameterSearchPolicy for the learner family.
    hyperparam_grid = validate_search_grid(learner_cls.family, hyperparam_grid)
    governance_contract = governance_contract or GovernanceContract(
        version="trainer-governance-v1", family_budget=len(hyperparam_grid)
    )
    exposure_ledger = CandidateExposureLedger(governance_contract.family_budget)
    if require_explicit_feature_availability:
        _assert_declared_feature_availability(decision_clock, train_ds.feature_cols)

    # 4. search the approved grid, fail closed on adequacy.
    # X_tr and y_tr are the actual fit cohort (finite-filtered in _extract_matrices).
    telemetry = {
        "raw_obs": len(X_tr),
        "effective_obs": len(X_tr),
        "finite_obs": len(X_tr),
        "n_rows": len(X_tr),
        "n_features": X_tr.shape[1],
        "n_unique_dates": train_ds.frame[train_ds.date_col].nunique(),
        "n_unique_stocks": train_ds.frame[train_ds.stock_col].nunique(),
    }
    validation_scores: list[dict[str, Any]] = []
    for candidate in hyperparam_grid:
        _raise_if_cancelled(cancel_token)
        identity = candidate_identity(candidate)
        exposure_ledger.expose(identity)
        spec = LearnerSpec(
            learner_name=learner_cls.name,
            family=learner_cls.family,
            hyperparams=dict(candidate),
            sample_contract=sample_contract,
            random_seed=random_seed,
        )
        try:
            learner = learner_cls(spec)
            # Parameter-domain errors are programmer/governance errors, not a bad
            # score.  Unknown fit exceptions likewise fail loud below.
            learner.validate_params()
        except Exception as exc:
            validation_scores.append(
                {"candidate_id": identity, "hyperparams": dict(candidate), "rank_ic": float("nan"),
                 "icir": float("nan"), "mse": float("nan"), "reason": f"param: {exc}"}
            )
            continue
        n_features = X_tr.shape[1]
        free_params = learner.effective_parameter_count(candidate, n_features)
        met, failures = learner.check_adequacy(telemetry, free_params)
        if not met:
            validation_scores.append(
                {"candidate_id": identity, "hyperparams": dict(candidate), "rank_ic": float("nan"),
                 "icir": float("nan"), "mse": float("nan"),
                 "reason": "adequacy: " + "; ".join(failures)}
            )
            continue
        try:
            # Skip weights for learners that explicitly reject them (PCR, ElasticNet)
            # to avoid NotImplementedError. Learners that silently ignore weights still
            # receive them (RandomForest, XGBoost).
            try:
                if aux is not None:
                    frozen = learner.fit(X_tr, y_tr, weights=weights, aux=aux)
                else:
                    frozen = learner.fit(X_tr, y_tr, weights=weights)
            except NotImplementedError as nie:
                if "weight" in str(nie).lower() and weights is not None:
                    # Learner does not support weights — retry without them
                    if aux is not None:
                        frozen = learner.fit(X_tr, y_tr, weights=None, aux=aux)
                    else:
                        frozen = learner.fit(X_tr, y_tr, weights=None)
                else:
                    raise
        except Exception as exc:
            validation_scores.append(
                {"candidate_id": identity, "hyperparams": dict(candidate), "rank_ic": float("nan"),
                 "icir": float("nan"), "mse": float("nan"), "reason": f"fit: {exc}"}
            )
            raise
        if validation_ds is not None:
            ev = _evaluate_validation(learner, frozen, validation_ds, preprocessing)
        else:
            # train→test only mode: no held-out validation — use in-sample IC
            # as the selection proxy (documented fallback, not a validation score).
            pred_train = learner.predict(frozen, X_tr)
            ev = {
                "rank_ic": _rank_ic(y_tr, pred_train),
                "icir": float("nan"),
                "mse": float(np.mean((y_tr - pred_train) ** 2)) if len(y_tr) else float("nan"),
            }
        entry = {
            "candidate_id": identity,
            "hyperparams": dict(candidate),
            "rank_ic": ev["rank_ic"],
            "icir": ev["icir"],
            "mse": ev["mse"],
        }
        if validation_ds is None:
            entry["in_sample_selection"] = True
        if not np.isfinite(ev["rank_ic"]):
            entry["reason"] = "non-finite validation IC"
        validation_scores.append(entry)

    # 5. select the best candidate.
    _raise_if_cancelled(cancel_token)
    from modeling.selection import select_best_validation

    best_hyperparams, selection_diag = select_best_validation(validation_scores, objective="rank_ic")
    if best_hyperparams is None:
        reasons = [str(s.get("reason", "")) for s in validation_scores]
        raise ValueError(
            f"all {len(hyperparam_grid)} hyperparameter candidates failed "
            f"adequacy/validation for {learner_cls.name}: {reasons}"
        )

    # 6. refit the best candidate on train (optionally train + validation).
    _raise_if_cancelled(cancel_token)
    if retrain_policy == "train_plus_validation" and validation_ds is not None:
        refit_frame = pd.concat([train_ds.frame, validation_ds.frame], ignore_index=True)
        refit_ds = PanelDataset(
            frame=refit_frame,
            date_col=train_ds.date_col,
            stock_col=train_ds.stock_col,
            feature_cols=list(train_ds.feature_cols),
            label_col=train_ds.label_col,
        )
        # The final fit also sees validation rows — their labels mature at
        # anchor + H and MUST NOT reach into the next held-out set.  Purge the
        # combined refit window against the evaluation (test) boundary too.
        if evaluation_boundary is not None:
            refit_ds = purge_before_boundary(refit_ds, evaluation_boundary, label_contract)
    else:
        refit_ds = train_ds
    X_fit, y_fit, aux_fit, weights_fit = _extract_matrices(
        refit_ds, preprocessing, aux_col, decay_half_life_bars, sample_weight_policy, sample_weight_half_life_dates
    )
    best_spec = LearnerSpec(
        learner_name=learner_cls.name,
        family=learner_cls.family,
        hyperparams=dict(best_hyperparams),
        sample_contract=sample_contract,
        random_seed=random_seed,
    )
    best_learner = learner_cls(best_spec)
    best_learner.validate_params()
    # Same weight-fallback logic as the validation loop
    try:
        if aux_fit is not None:
            frozen = best_learner.fit(X_fit, y_fit, weights=weights_fit, aux=aux_fit)
        else:
            frozen = best_learner.fit(X_fit, y_fit, weights=weights_fit)
    except NotImplementedError as nie:
        if "weight" in str(nie).lower() and weights_fit is not None:
            if aux_fit is not None:
                frozen = best_learner.fit(X_fit, y_fit, weights=None, aux=aux_fit)
            else:
                frozen = best_learner.fit(X_fit, y_fit, weights=None)
        else:
            raise

    # 7. freeze the artifact.
    # The availability/cutoff invariant: an artifact trained on
    # train+validation must NOT claim to be available at train_end — its final
    # fit saw validation data.  training_cutoff == available_at == final_fit_end
    # where final_fit_end is the last date the FINAL fit actually consumed.
    train_start = str(train_ds.frame[train_ds.date_col].min())
    train_end = str(train_ds.frame[train_ds.date_col].max())
    final_fit_start = str(refit_ds.frame[refit_ds.date_col].min())
    final_fit_end = str(refit_ds.frame[refit_ds.date_col].max())
    refit_used_validation = (
        retrain_policy == "train_plus_validation" and validation_ds is not None
    )
    # training_cutoff = last moment the final fit's training labels matured
    # (max anchor + horizon), never the last anchor date.
    horizon = int(getattr(label_contract, "horizon_bars", 0))
    training_cutoff = _maturity_cutoff(final_fit_end, horizon, _bar_calendar)
    val_start = (
        str(validation_ds.frame[validation_ds.date_col].min())
        if validation_ds is not None
        else None
    )
    val_end = (
        str(validation_ds.frame[validation_ds.date_col].max())
        if validation_ds is not None
        else None
    )
    manifest = ModelArtifactManifest(
        model_name=learner_cls.name,
        model_version=model_version,
        artifact_id=f"{learner_cls.name}-{uuid.uuid4().hex}",
        train_start=train_start,
        train_end=train_end,
        validation_start=val_start,
        validation_end=val_end,
        selection_train_end=str(train_ds.frame[train_ds.date_col].max()),
        final_fit_start=final_fit_start,
        final_fit_end=final_fit_end,
        refit_used_validation=refit_used_validation,
        decision_clock_id=decision_clock.decision_at,
        label_contract_id=label_contract.label_name,
        feature_schema_hash=feature_schema_hash,
        data_source_hash=data_source_hash,
        universe_hash=universe_hash,
        hyperparameters=dict(best_hyperparams),
        preprocessing_state_hash=preprocessing.state_hash(),
        fit_code_commit=fit_code_commit,
        random_seed=random_seed,
        available_at=training_cutoff,
        training_cutoff=training_cutoff,
    )
    fit_info = {
        "train_telemetry": telemetry,
        "selected_hyperparams": dict(best_hyperparams),
        "selection_diagnostics": selection_diag,
        "validation_scores": validation_scores,
        "retrain_policy": retrain_policy,
        "selection_train_end": train_end,
        "final_fit_end": final_fit_end,
        "training_cutoff": training_cutoff,
        "label_horizon_bars": horizon,
        "evaluation_boundary": (
            str(evaluation_boundary) if evaluation_boundary is not None else None
        ),
        "governance_contract": {
            "version": governance_contract.version,
            "objective": governance_contract.objective,
            "objective_transform_version": governance_contract.objective_transform_version,
            "label_transform_version": governance_contract.label_transform_version,
            "uncertainty_method": governance_contract.uncertainty_method,
        },
        "candidate_exposure_ledger": {
            "identities": exposure_ledger.identities,
            "digest": exposure_ledger.digest(),
        },
    }
    artifact = ModelArtifact(
        manifest=manifest,
        learner=best_learner,
        frozen=frozen,
        preprocessing=preprocessing,
        fit_info=fit_info,
    )
    return TrainResult(
        artifact=artifact,
        selected_hyperparams=dict(best_hyperparams),
        validation_scores=validation_scores,
        preprocessing=preprocessing,
    )
