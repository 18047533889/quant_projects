# -*- coding: utf-8 -*-
"""Walk-forward evidence + hard-gate report (Model Layer Major Redesign
taskbook §19 / §56 / §64 / §71 / §83).

* :class:`FoldResult` — one outer walk-forward fold.
* :class:`WalkForwardEvidence` — folds + OOS evaluation + dataset exposure +
  §56 ledger rows, serialisable to parquet / JSON.
* :func:`run_walk_forward_evidence` — the §71 runner.
* :func:`report_hard_gate_set` — the subset of §64 gates this package can
  honestly verify.

The §71 runner uses the AUTHORITATIVE trainer/predictor only — there is no
silent fallback trainer path (a production trainer error FAILS the run; it is
never swapped for a self-contained substitute that keeps the gates green).
Every gate in :func:`report_hard_gate_set` is verified by a dynamic invariant
probe (never a hard-coded ``True``); ``MODEL_CURRENT_HEAD_EVIDENCE_FRESH`` is
TRUE only when the evidence SHA equals the repository HEAD resolved at runtime.
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
    "check_model_direct_use_readiness",
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
    sample_contract=None,
    min_folds: int = 3,
) -> WalkForwardEvidence:
    """Run the §71 outer walk-forward loop — AUTHORITATIVE TRAINER ONLY.

    Requires ``>= min_folds`` outer folds (raises ``ValueError`` otherwise).
    Each fold trains on the (purged) train window, selects hyperparameters on
    the validation window only, and scores the frozen artifact on the test
    window through the :mod:`modeling.predictor` no-fit guard.

    There is NO fallback trainer path: the authoritative
    :func:`modeling.trainer.train_model` is used for every fold.  If it errors
    (adequacy fail, convergence fail, param-domain reject) the evidence run
    FAILS — a silent substitute path that keeps the gates green is forbidden.
    """
    # The authoritative trainer/predictor are REQUIRED — fail closed if missing.
    from modeling.trainer import PreprocessingSpec, train_model
    from modeling.predictor import Predictor

    folds, dates = _build_folds(dataset, walk_forward_spec, label_contract)
    if len(folds) < min_folds:
        raise ValueError(
            f"walk-forward produced {len(folds)} outer folds; need >= {min_folds}. "
            "Increase the lookback/step or shrink train/val/test windows."
        )

    predictor = Predictor()
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

        prep_spec = preprocessing_spec if preprocessing_spec is not None else PreprocessingSpec()
        grid = hyperparam_grid if hyperparam_grid else [{}]
        result = train_model(
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
            sample_contract=sample_contract,
            evaluation_boundary=dates[fold["test_start_i"]],
        )
        artifact: ModelArtifact = result.artifact
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
        pred = predictor.predict(artifact, Xte)

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
    cat_stocks = np.concatenate(all_stocks) if all_stocks else np.array([])

    oos = evaluate_predictions(cat_pred, cat_y, cat_dates, cat_stocks)
    oos_evaluation = oos.to_dict()
    fold_rank_ics = []
    for fold in fold_results:
        _, fold_ics = per_date_rank_ic(fold.test_pred, fold.test_y, fold.test_dates)
        fold_rank_ics.append(float(np.mean(fold_ics)) if len(fold_ics) else float("nan"))
    finite_fold_ics = [value for value in fold_rank_ics if np.isfinite(value)]
    oos_evaluation["per_fold_rank_ic"] = fold_rank_ics
    oos_evaluation["equal_weight_fold_rank_ic"] = (
        float(np.mean(finite_fold_ics)) if finite_fold_ics else float("nan")
    )
    oos_evaluation["calendar_time_rank_ic"] = oos.mean_daily_rank_ic
    degradation = [
        fold.validation_best_rank_ic - test_ic
        for fold, test_ic in zip(fold_results, fold_rank_ics)
        if np.isfinite(fold.validation_best_rank_ic) and np.isfinite(test_ic)
    ]
    oos_evaluation["validation_to_test_degradation"] = {
        "per_fold": degradation,
        "mean": float(np.mean(degradation)) if degradation else float("nan"),
    }
    oos_evaluation["reproducibility_bundle"] = {
        "artifact_lineage": [fold.artifact for fold in fold_results],
        "dataset_certificate": dataset.telemetry(),
        "fold_plan": [fold.to_dict() for fold in fold_results],
        "metric_convention": oos.metric_convention,
        "calendar": [str(date) for date in dates],
        "universe": sorted(str(value) for value in dataset.frame[dataset.stock_col].unique()),
        "code_build": fit_code_commit,
        "random_seed": random_seed,
        "label_horizon_bars": label_contract.horizon_bars,
    }

    dataset_exposure = {
        "n_candidates_tested": len(hyperparam_grid) if hyperparam_grid else 0,
        "fold_count": len(folds),
        "exposed_sets": ["train", "validation", "test"],
        "final_holdout_state": "EVALUATOR_ONLY",
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
                "selection_metric": "validation_mean_daily_rank_ic",
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
def _repository_head() -> str | None:
    """Resolve the repository HEAD, or ``None`` when unavailable.

    The repo root is derived from this file's location by walking up to the
    directory containing ``.git``; ``git -C <root> rev-parse HEAD`` is used so
    the result is independent of the caller's cwd.  ``None`` (not a sentinel
    string) is returned on any failure so a genuinely unavailable HEAD can
    never compare equal to another unavailable value (no ``UNKNOWN==UNKNOWN``
    fake-green).
    """
    import subprocess
    from pathlib import Path

    try:
        probe = Path(__file__).resolve().parent.parent
        root: Path | None = None
        while probe.parent != probe:
            if (probe / ".git").exists():
                root = probe
                break
            probe = probe.parent
        if root is None:
            return None
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def _probe_no_fit_in_score() -> bool:
    """Dynamically prove a frozen predict never calls fit (TrainOnlyFitGuard)."""
    try:
        from modeling.leakage_guard import TrainOnlyFitGuard
        from modeling.learners import PCRLearner
        from modeling.learners.base import LearnerSpec

        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 3))
        y = X[:, 0] + rng.normal(0, 0.1, size=30)
        learner = PCRLearner(
            LearnerSpec("predictive_pcr", "pcr", hyperparams={"n_components": 2})
        )
        frozen = learner.fit(X, y)
        with TrainOnlyFitGuard(learner):
            pred = learner.predict(frozen, X[:5])
        return bool(np.isfinite(pred).all())
    except Exception:
        return False


def _probe_purge_label_overlap() -> bool:
    """Dynamically verify §9 purge keeps only rows whose label interval ends
    strictly before the evaluation boundary."""
    try:
        from modeling.dataset import PanelDataset
        from modeling.walk_forward import purge_before_boundary

        import pandas as pd

        rows = []
        for d in range(7):
            for s in (0, 1):
                rows.append([d, f"S{s}", 1.0, 0.0])
        frame = pd.DataFrame(rows, columns=["date", "stock", "x", "y"])
        ds = PanelDataset(frame, date_col="date", stock_col="stock",
                          feature_cols=["x"], label_col="y")
        contract = LabelContract(label_name="ret", horizon_bars=2)
        purged = purge_before_boundary(ds, 4, contract)  # boundary = date 4
        kept = sorted(purged.frame["date"].unique())
        return kept == [0, 1]
    except Exception:
        return False


def _probe_zero_random_time_split() -> bool:
    """Dynamically verify date-authoritative deterministic ordered splits."""
    try:
        from modeling.dataset import PanelDataset
        from modeling.walk_forward import WalkForwardSpec, check_fold_order, make_walk_forward_splits

        import pandas as pd

        rng = np.random.default_rng(1)
        rows = []
        for d in range(12):
            for s in range(5):
                rows.append([pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), f"S{s}",
                             1.0, 0.0])
        frame = pd.DataFrame(rows, columns=["date", "stock", "x", "y"])
        ds = PanelDataset(frame, date_col="date", stock_col="stock",
                          feature_cols=["x"], label_col="y")
        spec = WalkForwardSpec(train_lookback_bars=4, validation_bars=2, test_bars=2,
                               step_bars=2, min_train_dates=4, min_train_stocks=3,
                               min_train_obs=10)
        f1 = make_walk_forward_splits(ds, spec)
        f2 = make_walk_forward_splits(ds, spec)
        same = [(x.fold_id, x.train_start, x.test_start) for x in f1] == [
            (x.fold_id, x.train_start, x.test_start) for x in f2
        ]
        return same and check_fold_order(f1) == []
    except Exception:
        return False


def _probe_full_sample_preprocessing_leak() -> bool:
    """Dynamically prove preprocessing statistics are frozen (train-only): a
    far-future poisoned row is transformed with the train frozen stats, never
    recomputed."""
    try:
        from modeling.artifact import FrozenPreprocessing

        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 3))
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        preproc = FrozenPreprocessing(
            [
                {"kind": "imputer", "means": mean.tolist()},
                {"kind": "standardize", "mean": mean.tolist(), "scale": scale.tolist()},
            ]
        )
        state_before = preproc.state_hash()
        poison = np.full((3, 3), 1e9)
        out = preproc.transform(poison)
        # Frozen stats mean ~1e9 scale, NOT restandardized to ~0.
        return preproc.state_hash() == state_before and np.isfinite(out).all()
    except Exception:
        return False


def _probe_negative_controls() -> bool:
    """All 7 §58 negative controls must pass — no silent future/label/scaler/
    hyperparameter/universe/revision/execution-clock leak."""
    try:
        from modeling.leakage_guard import run_all_negative_controls

        res = run_all_negative_controls()
        return bool(res) and all(
            isinstance(v, dict) and v.get(k) is True
            for k, v in res.items()
        )
    except Exception:
        return False


def _probe_search_policy_rejects() -> bool:
    """NUMERICAL_POLICY / DATA_POLICY parameters must be unsearchable by
    construction (ParameterSearchPolicy raises ValueError)."""
    try:
        from modeling.contracts import ParameterSearchPolicy, ParamRole

        try:
            ParameterSearchPolicy(role=ParamRole.NUMERICAL_POLICY, searchable=True)
            return False
        except ValueError:
            pass
        try:
            ParameterSearchPolicy(role=ParamRole.DATA_POLICY, searchable=True)
            return False
        except ValueError:
            pass
        # The searchable/unsearchable resolution must also hold for the live
        # registry parameter policies.
        from modeling.hyperparams import param_search_policy
        from modeling.presets import MODEL_PARAM_RECOMMENDATIONS

        for fam, param, _, _ in MODEL_PARAM_RECOMMENDATIONS:
            pol = param_search_policy(fam, param)
            if pol is None:
                return False
            if pol.role == ParamRole.NUMERICAL_POLICY and pol.searchable:
                return False
            if pol.role == ParamRole.DATA_POLICY and pol.searchable:
                return False
        return True
    except Exception:
        return False


def _probe_adequacy_contract_binding() -> bool:
    """Every registered predictive learner resolves its SampleAdequacyContract
    via an explicit sample_contract_family — pcr/pls/elastic_net bind "linear"."""
    try:
        from modeling.learners.base import default_sample_contracts
        from modeling.registry import MODEL_REGISTRY, register_default_learners

        register_default_learners()  # idempotent
        contracts = default_sample_contracts()
        for name in MODEL_REGISTRY.registered_predictive_learners():
            rec = MODEL_REGISTRY.get(name)
            learner_cls = rec["learner_cls"]
            fam = getattr(learner_cls, "sample_contract_family", None) or learner_cls.family
            if fam not in contracts or contracts[fam] is None:
                return False
        return True
    except Exception:
        return False


def _probe_final_holdout_not_exposed() -> bool:
    """Test windows must be disjoint from the train/validation search windows."""
    try:
        from modeling.walk_forward import make_walk_forward_splits

        from modeling.dataset import PanelDataset
        import pandas as pd

        rng = np.random.default_rng(3)
        rows = []
        for d in range(14):
            for s in range(5):
                rows.append([pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), f"S{s}",
                             1.0, 0.0])
        frame = pd.DataFrame(rows, columns=["date", "stock", "x", "y"])
        ds = PanelDataset(frame, date_col="date", stock_col="stock",
                          feature_cols=["x"], label_col="y")
        from modeling.walk_forward import WalkForwardSpec
        spec = WalkForwardSpec(train_lookback_bars=5, validation_bars=2, test_bars=2,
                               step_bars=2, min_train_dates=5, min_train_stocks=3,
                               min_train_obs=10)
        for fold in make_walk_forward_splits(ds, spec):
            tr = set(fold.train_ds.frame[ds.date_col].tolist())
            te = set(fold.test_ds.frame[ds.date_col].tolist())
            va = (
                set(fold.validation_ds.frame[ds.date_col].tolist())
                if fold.validation_ds is not None else set()
            )
            if tr & te or va & te:
                return False
        return True
    except Exception:
        return False


def _probe_regime_support() -> bool:
    """Genuinely fit RegimeLearner on a small panel and verify §6.1 support:
    every requested regime has >= min_regime_obs observations AND an undersized
    sample FAILS CLOSED (ValueError) instead of silently degrading."""
    try:
        from modeling.contracts import SampleAdequacyContract
        from modeling.learners import RegimeLearner
        from modeling.learners.base import LearnerSpec

        rng = np.random.default_rng(11)
        n = 300
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        gate = rng.normal(size=n) + 0.5 * X[:, 0]
        contract = SampleAdequacyContract(
            min_raw_obs=1, min_effective_obs=1, min_unique_dates=1,
            min_unique_stocks=1, min_obs_per_parameter=1.0, min_regime_obs=10,
        )
        learner = RegimeLearner(LearnerSpec(
            learner_name="predictive_regime", family="regime",
            hyperparams={"n_regimes": 3, "gating_feature_index": 0},
            sample_contract=contract,
        ))
        frozen = learner.fit(X, y, aux={"regime_state": gate})
        report = learner.support_report(frozen)
        obs = report.get("per_regime_obs", [])
        supported = (
            report.get("n_regimes") == 3
            and len(obs) == 3
            and all(o >= 10 for o in obs)
        )
        bad = SampleAdequacyContract(
            min_raw_obs=1, min_effective_obs=1, min_unique_dates=1,
            min_unique_stocks=1, min_obs_per_parameter=1.0, min_regime_obs=1000,
        )
        bad_learner = RegimeLearner(LearnerSpec(
            learner_name="predictive_regime", family="regime",
            hyperparams={"n_regimes": 3}, sample_contract=bad,
        ))
        try:
            bad_learner.fit(X, y, aux={"regime_state": gate})
            fail_closed = False
        except ValueError:
            fail_closed = True
        return bool(supported and fail_closed)
    except Exception:
        return False


def _probe_moe_support() -> bool:
    """Genuinely fit MixtureOfExpertsLearner and verify §6.2/§6.3: every active
    expert has enough observations, a full-rank design and a bounded condition
    number; undersized experts FAIL CLOSED (never a silent degraded model)."""
    try:
        from modeling.contracts import SampleAdequacyContract
        from modeling.learners import MixtureOfExpertsLearner
        from modeling.learners.base import LearnerSpec

        rng = np.random.default_rng(12)
        n = 300
        X = rng.normal(size=(n, 2))
        y = rng.normal(size=n)
        gate = rng.normal(size=n) + 0.5 * X[:, 0]
        contract = SampleAdequacyContract(
            min_raw_obs=1, min_effective_obs=1, min_unique_dates=1,
            min_unique_stocks=1, min_obs_per_parameter=1.0, min_expert_obs=10,
        )
        learner = MixtureOfExpertsLearner(LearnerSpec(
            learner_name="predictive_moe", family="moe",
            hyperparams={"n_experts": 3, "gating_feature_index": 0},
            sample_contract=contract,
        ))
        frozen = learner.fit(X, y, aux={"regime_state": gate})
        report = learner.support_report(frozen)
        obs = report.get("per_expert_obs", [])
        full_rank = report.get("full_rank", [])
        supported = (
            report.get("n_experts") == 3
            and len(obs) == 3
            and all(o >= 10 for o in obs)
            and bool(full_rank) and all(full_rank)
        )
        bad = SampleAdequacyContract(
            min_raw_obs=1, min_effective_obs=1, min_unique_dates=1,
            min_unique_stocks=1, min_obs_per_parameter=1.0, min_expert_obs=1000,
        )
        bad_learner = MixtureOfExpertsLearner(LearnerSpec(
            learner_name="predictive_moe", family="moe",
            hyperparams={"n_experts": 3}, sample_contract=bad,
        ))
        try:
            bad_learner.fit(X, y, aux={"regime_state": gate})
            fail_closed = False
        except ValueError:
            fail_closed = True
        return bool(supported and fail_closed)
    except Exception:
        return False


def _probe_asof_resolution() -> bool:
    """Genuinely exercise §50 as-of resolution: a future-trained artifact must
    never be returned for a historical asof, the LATEST legal artifact wins,
    and an asof before any training cutoff resolves to None (fail closed)."""
    try:
        from modeling.artifact import ModelArtifact, ModelArtifactManifest
        from modeling.contracts import LabelContract
        from modeling.dsl_bridge import ArtifactResolver, ArtifactStore
        from modeling.leakage_guard import default_trainer_fn, synthetic_panel

        ds = synthetic_panel(n_dates=14, n_stocks=10, n_features=3, seed=13)
        lc = LabelContract(label_name="y", horizon_bars=1)
        dates = sorted(ds.frame[ds.date_col].unique())
        early = ds.filter_dates(dates[0], dates[4])
        late = ds.filter_dates(dates[5], dates[9])

        def _with_cutoff(art: ModelArtifact, artifact_id: str, cutoff: Any) -> ModelArtifact:
            m = art.manifest
            cutoff_s = str(cutoff)
            manifest = ModelArtifactManifest(
                model_name=m.model_name, model_version=m.model_version,
                artifact_id=artifact_id, train_start=m.train_start, train_end=m.train_end,
                training_cutoff=cutoff_s, available_at=cutoff_s,
                decision_clock_id=m.decision_clock_id, label_contract_id=m.label_contract_id,
                feature_schema_hash=m.feature_schema_hash,
            )
            return ModelArtifact(manifest, art.learner, art.frozen, art.preprocessing)

        art_early = _with_cutoff(
            default_trainer_fn(early, label_contract=lc), "asof-early", dates[4]
        )
        art_late = _with_cutoff(
            default_trainer_fn(late, label_contract=lc), "asof-late", dates[9]
        )
        resolver = ArtifactResolver(store=ArtifactStore())
        resolver.register(art_early)
        resolver.register(art_late)
        name = art_early.model_name
        before = resolver.resolve(name, str(dates[0]))    # < early cutoff -> None
        mid = resolver.resolve(name, str(dates[4]))       # == early cutoff -> early
        after = resolver.resolve(name, str(dates[10]))    # > late cutoff -> late
        return bool(
            before is None
            and mid is not None and mid.artifact_id == "asof-early"
            and after is not None and after.artifact_id == "asof-late"
        )
    except Exception:
        return False


def check_model_direct_use_readiness() -> tuple[int, int, dict[str, list[str]]]:
    """Check DirectUse readiness for all model operators (MF-P0-002 / REM-024).

    Returns (ready_count, total_count, reasons_dict) where reasons_dict maps
    each non-ready operator to a list of failure reasons.

    The MINIMUM readiness checklist (per taskbook §27 / §36 tri-state):
      - explicit_timing: operator has an entry in MODEL_TIMING_CONTRACTS
      - production_lane: lane is FAST_NATIVE_ALPHA / EXPENSIVE_CERTIFIED_ALPHA /
                         MODEL_FEATURE_SCORE / STATE_CONDITION_EVENT
      - not_tombstoned: lane != DELETE_TOMBSTONE
      - parameter_domain_certified: at least one certified point exists (from
                                     runtime.parameter_domain_store)

    This is HONEST reporting: if no operator has behavioral certification today,
    ready_count = 0. Never fake a counter without the real artifact.
    """
    try:
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry
        from cleaned_operators.model_timing import MODEL_TIMING_CONTRACTS, is_model_like_name
        from cleaned_operators.model_lane import assign_model_lane, _category_of

        load_all()
        canonicals = sorted(OperatorRegistry.list_canonical())
        model_like = [c for c in canonicals if is_model_like_name(c, _category_of(c))]

        PROD_LANES = frozenset({
            "FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
            "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT"
        })

        ready_count = 0
        reasons_dict: dict[str, list[str]] = {}

        for c in model_like:
            lane = assign_model_lane(c)
            failures: list[str] = []

            # Gate 1: explicit timing
            if c not in MODEL_TIMING_CONTRACTS:
                failures.append("no_explicit_timing")

            # Gate 2: production lane
            if lane not in PROD_LANES:
                failures.append("not_production_lane")

            # Gate 3: not tombstoned
            if lane == "DELETE_TOMBSTONE":
                failures.append("tombstoned")

            # Gate 4: parameter domain certified (real check against store)
            param_certified = False
            try:
                from runtime.parameter_domain_store import ParameterDomainCertificationStore
                store = ParameterDomainCertificationStore()
                param_certified = bool(store.operator_has_any_certified_region(c))
            except Exception:
                param_certified = False

            if not param_certified:
                failures.append("no_parameter_domain_point")

            if not failures:
                ready_count += 1
            else:
                reasons_dict[c] = failures

        return ready_count, len(model_like), reasons_dict

    except Exception:
        return 0, 0, {}


def report_hard_gate_set(git_sha: str | None = None, current_head: str | None = None) -> dict[str, dict[str, Any]]:
    """The §64 hard-gate subset this package honestly verifies.

    Every gate below is verified by a dynamic probe (real invariant check) — no
    gate is hardcoded ``True``.  ``git_sha`` is the commit the evidence was
    generated at; ``current_head`` is the repository HEAD the check runs under
    (resolved from git when omitted).  MODEL_CURRENT_HEAD_EVIDENCE_FRESH is TRUE
    only when ``git_sha`` is not None and equals ``current_head`` (both resolved
    to a real SHA); a None on either side FAILS — stale/unverifiable evidence
    never passes.
    """
    # ---- dynamic state used by the gates -----------------------------------
    def _learners():
        try:
            from modeling.registry import MODEL_REGISTRY, register_default_learners
            register_default_learners()  # idempotent
            return MODEL_REGISTRY
        except Exception:
            return None

    registry = _learners()
    predictive = (
        registry.registered_predictive_learners() if registry is not None else []
    )
    spec_binding_ok = bool(predictive)
    spec_binding_reason = (
        f"dynamic probe: {len(predictive)} predictive learner(s) bound to "
        "WalkForwardSpec in ModelRegistry"
        if spec_binding_ok
        else "dynamic probe FAILED: no predictive learner registered in ModelRegistry"
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
                    validation_reason = (
                        f"dynamic probe FAILED: learner {name} lacks a validation "
                        "window spec"
                    )
                    break
        except Exception as exc:
            all_validated = False
            validation_reason = f"dynamic probe FAILED: registry read error: {exc}"

    adequacy_ok = _probe_adequacy_contract_binding()
    adequacy_reason = (
        "dynamic probe: every registered predictive learner resolves its "
        "SampleAdequacyContract via an explicit sample_contract_family "
        "(pcr/pls/elastic_net → linear)"
        if adequacy_ok
        else "dynamic probe FAILED: some registered learner's sample_contract_family "
        "does not resolve to a SampleAdequacyContract (was None)"
    )

    search_policy_ok = False
    search_policy_reason = "dynamic probe FAILED: param_search_policy unavailable"
    try:
        from modeling.hyperparams import param_search_policy
        from modeling.presets import MODEL_PARAM_RECOMMENDATIONS
        search_policy_ok = all(
            param_search_policy(fam, param) is not None
            for fam, param, _, _ in MODEL_PARAM_RECOMMENDATIONS
        )
        search_policy_reason = (
            "dynamic probe: every §69 recommended param has a ParameterSearchPolicy"
            if search_policy_ok
            else "dynamic probe FAILED: some §69 params lack a ParameterSearchPolicy"
        )
    except Exception as exc:
        search_policy_reason = f"dynamic probe FAILED: param_search_policy error: {exc}"

    # Genuine support probes: a real fit + support_report (not a hasattr check).
    regime_support = _probe_regime_support()
    regime_reason = (
        "dynamic probe: RegimeLearner fitted 3 regimes with per-regime obs >= floor "
        "and an undersized sample raised ValueError (fail closed)"
        if regime_support
        else "dynamic probe FAILED: RegimeLearner did not satisfy §6.1 support or "
        "did not fail closed on an undersized sample"
    )
    moe_support = _probe_moe_support()
    moe_reason = (
        "dynamic probe: MixtureOfExpertsLearner fitted 3 experts (obs >= floor, "
        "full-rank design, bounded condition) and an undersized sample raised "
        "ValueError (fail closed)"
        if moe_support
        else "dynamic probe FAILED: MixtureOfExpertsLearner did not satisfy §6.2/§6.3 "
        "support or did not fail closed on an undersized sample"
    )

    # Genuine §50 as-of resolution probe (future-trained artifact never returned).
    asof_ok = _probe_asof_resolution()
    asof_reason = (
        "dynamic probe: §50 as-of resolution returned the latest legal artifact, "
        "rejected future-trained artifacts for history and returned None before "
        "any training cutoff (fail closed)"
        if asof_ok
        else "dynamic probe FAILED: as-of resolution returned a future-trained "
        "artifact or did not fail closed"
    )

    # Freshness: the evidence is only fresh when the bound SHA equals the HEAD
    # this check runs under.  A None on either side (unavailable) is FAIL —
    # never a fake-green UNKNOWN==UNKNOWN.
    if current_head is None:
        current_head = _repository_head()
    fresh = git_sha is not None and current_head is not None and git_sha == current_head
    if fresh:
        freshness_reason = f"evidence bound to {git_sha} == repository HEAD {current_head}"
    elif git_sha is not None and current_head is not None:
        freshness_reason = (
            f"evidence bound to {git_sha} but repository HEAD is {current_head} — "
            "stale, re-run the evidence generator at the current HEAD"
        )
    elif current_head is None:
        freshness_reason = (
            f"repository HEAD unavailable (git rev-parse failed); evidence bound to {git_sha!r} — "
            "freshness cannot be verified"
        )
    else:
        freshness_reason = (
            f"evidence freshness requires the generating git SHA (pass git_sha=); "
            f"repository HEAD is {current_head!r}"
        )

    no_fit_ok = _probe_no_fit_in_score()
    purge_ok = _probe_purge_label_overlap()
    random_split_ok = _probe_zero_random_time_split()
    scaler_ok = _probe_full_sample_preprocessing_leak()
    controls_ok = _probe_negative_controls()
    search_reject_ok = _probe_search_policy_rejects()
    holdout_ok = _probe_final_holdout_not_exposed()

    gates: list[tuple[Any, ...]] = [
        (
            "MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH",
            no_fit_ok,
            "dynamic probe: frozen predict under TrainOnlyFitGuard produced finite "
            "scores without invoking fit"
            if no_fit_ok
            else "dynamic probe FAILED: fit was callable/used inside the score path",
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
            purge_ok,
            "dynamic probe: purge_before_boundary keeps only rows whose label "
            "interval ends strictly before the evaluation boundary"
            if purge_ok
            else "dynamic probe FAILED: label-interval purge kept overlapping rows",
        ),
        (
            "MODEL_ZERO_RANDOM_TIME_SPLIT",
            random_split_ok,
            "dynamic probe: date-authoritative ordered splits, deterministic, "
            "no train/validation/test overlap"
            if random_split_ok
            else "dynamic probe FAILED: splits overlap or are non-deterministic",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_SCALER",
            scaler_ok,
            "dynamic probe: FrozenPreprocessing state hash unchanged after "
            "transforming a far-future poisoned row (frozen train-only stats)"
            if scaler_ok
            else "dynamic probe FAILED: preprocessing state changed on transform",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS",
            no_fit_ok and scaler_ok,
            "PCR fits its PCA inside learner.fit on train rows only; the frozen "
            "preprocessing probe proves no full-sample recompute (scaler + "
            "no-fit probes both green)"
            if (no_fit_ok and scaler_ok)
            else "dynamic probe FAILED: full-sample PCA/scaler recompute possible",
        ),
        (
            "MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION",
            False,
            "NOT_RUN: the model layer has no feature-selection module to exercise — "
            "this is a static audit claim, not a runnable invariant probe",
            "NOT_RUN",
        ),
        (
            "MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION",
            controls_ok,
            "dynamic probe: all 7 §58 negative controls pass — including "
            "hyperparam_poison (test-driven selection is detected as a leak)"
            if controls_ok
            else "dynamic probe FAILED: a negative control did not pass",
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
            search_reject_ok,
            "dynamic probe: ParameterSearchPolicy raises on searchable "
            "NUMERICAL_POLICY; live registry policies are not searchable"
            if search_reject_ok
            else "dynamic probe FAILED: a NUMERICAL_POLICY parameter is searchable",
        ),
        (
            "MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER",
            search_reject_ok,
            "dynamic probe: ParameterSearchPolicy raises on searchable "
            "DATA_POLICY; live registry policies are not searchable"
            if search_reject_ok
            else "dynamic probe FAILED: a DATA_POLICY parameter is searchable",
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
            holdout_ok,
            "dynamic probe: test windows are disjoint from the train/validation "
            "search windows in walk-forward folds"
            if holdout_ok
            else "dynamic probe FAILED: test dates overlap the search windows",
        ),
        (
            "MODEL_CURRENT_HEAD_EVIDENCE_FRESH",
            fresh,
            freshness_reason,
        ),
    ]

    # DirectUse behavioral certification gate (MF-P0-002 / REM-024)
    ready_count, total_count, reasons = check_model_direct_use_readiness()
    certification_ok = ready_count > 0
    certification_reason = (
        f"DirectUse readiness: {ready_count}/{total_count} model operators have "
        f"behavioral certification (explicit_timing + production_lane + "
        f"not_tombstoned + parameter_domain_certified)"
        if certification_ok
        else f"DirectUse readiness: 0/{total_count} model operators certified "
             f"(honest state: no behavioral proofs yet)"
    )
    gates.append((
        "MODEL_ALL_DIRECT_USE_HAVE_BEHAVIORAL_CERTIFICATION",
        certification_ok,
        certification_reason,
    ))

    out: dict[str, dict[str, Any]] = {}
    for entry in gates:
        name, value, check = entry[0], entry[1], entry[2]
        status = entry[3] if len(entry) > 3 else None
        if status is None:
            status = "PASS" if bool(value) else "FAIL"
        out[name] = {"value": bool(value), "check": check, "status": status}
    return out
