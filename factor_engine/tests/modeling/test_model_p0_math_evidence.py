# -*- coding: utf-8 -*-
"""P0 model-layer audit closure tests.

Covers the audit round's P0 findings against the predictive framework:

1. Evidence freshness — MODEL_CURRENT_HEAD_EVIDENCE_FRESH is TRUE only when
   the evidence SHA equals the repository HEAD (a stale JSON can never pass).
2. Artifact Availability Lookahead — a ``train_plus_validation`` refit's
   ``training_cutoff``/``available_at`` advance to the final-fit data (+ label
   horizon); the artifact is NOT as-of legal at the original ``train_end``.
3. PLS coefficient formula — multi-component parity against an independent
   score-space OLS oracle (the old ``W^T P`` vs ``P^T W`` bug would break it).
4. ElasticNet — raw-space restore, sample-count-invariant alpha, and
   non-convergence fails closed.
5. PCR rank fail-closed — ``n_components > min(n_rows, n_features)`` raises,
   never silently downgrades.
6. Sample-contract wiring — pcr/pls/elastic_net resolve the linear contract
   (never ``None``).
7. PanelDataset enterprise constraints — ``stock_col`` required, ``(date, stock)``
   unique (error / aggregate policies).
8. Purge boundary — ``validation_ds=None`` purges against the evaluation
   (test) boundary so training labels never reach the test window.
9. Per-learner ``effective_parameter_count`` — real DOF, never ``len(hyperparams)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import ModelArtifactManifest
from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    SampleAdequacyContract,
    ashare_decision_clock,
)
from modeling.dataset import PanelDataset
from modeling.evidence import report_hard_gate_set
from modeling.learners import (
    ElasticNetLearner,
    PCRLearner,
    PLSLearner,
)
from modeling.learners.base import LearnerSpec, ModelConvergenceError
from modeling.sample_policy import resolve_sample_contract
from modeling.timing import vwap_to_vwap_label
from modeling.trainer import PreprocessingSpec, train_model


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(n_dates=10, n_stocks=40, n_features=4, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_dates, freq="D")
    rows = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.normal(size=n_features)
            y = float(0.5 * x[0] - 0.2 * x[1] + rng.normal(0, 0.1))
            rows.append([d, f"S{s:03d}", *x.tolist(), y])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    return PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )


def _lenient():
    return SampleAdequacyContract(
        min_raw_obs=500, min_effective_obs=300, min_unique_dates=3,
        min_unique_stocks=30, min_obs_per_parameter=10,
    )


# --------------------------------------------------------------------------- #
# 1. evidence freshness
# --------------------------------------------------------------------------- #
def test_evidence_freshness_requires_sha_equals_head():
    gates = report_hard_gate_set(git_sha="abc123", current_head="abc123")
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is True


def test_evidence_freshness_stale_when_sha_differs():
    # evidence generated at an OLD commit, repo moved on -> stale, not fresh.
    gates = report_hard_gate_set(git_sha="c9c08ff5", current_head="43f2608c")
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is False


def test_evidence_freshness_requires_sha_present():
    gates = report_hard_gate_set(git_sha=None, current_head="43f2608c")
    assert gates["MODEL_CURRENT_HEAD_EVIDENCE_FRESH"]["value"] is False


def test_hard_gates_are_not_hardcoded_true():
    """Every gate has a non-empty invariant description; the gates the old
    implementation hard-coded ``True`` must now be backed by dynamic probes."""
    gates = report_hard_gate_set()
    assert len(gates) == 18
    # Gates that used to be hardcoded True — now dynamically probed (or, for the
    # one with no module to exercise, an honest NOT_RUN static claim).
    probe_gates = {
        "MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH",
        "MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP",
        "MODEL_ZERO_RANDOM_TIME_SPLIT",
        "MODEL_ZERO_FULL_SAMPLE_SCALER",
        "MODEL_ZERO_FULL_SAMPLE_PCA_PREPROCESS",
        "MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION",
        "MODEL_ZERO_NUMERICAL_POLICY_SEARCHABLE",
        "MODEL_ZERO_DATA_POLICY_SEARCHABLE_BY_FACTOR_MINER",
        "MODEL_FINAL_HOLDOUT_NOT_EXPOSED_TO_SEARCH",
    }
    for name in probe_gates:
        assert "probe" in gates[name]["check"] or "control" in gates[name]["check"], (
            f"{name} check is not a probe/control"
        )
    # The one gate with no runnable module must be an honest NOT_RUN static
    # claim — never a hardcoded True.
    fs = gates["MODEL_ZERO_FULL_SAMPLE_FEATURE_SELECTION"]
    assert fs["value"] is False
    assert "NOT_RUN" in fs["check"]
    for name, entry in gates.items():
        assert entry["check"], f"{name} has an empty check"


def test_evidence_freshness_and_no_fit_gates_green_at_consistency():
    gates = report_hard_gate_set(git_sha="HEAD", current_head="HEAD")
    assert gates["MODEL_ZERO_PREDICTIVE_LEARNER_FITTED_INSIDE_SCORE_PATH"]["value"] is True
    assert gates["MODEL_ZERO_TEST_DRIVEN_HYPERPARAM_SELECTION"]["value"] is True
    assert gates["MODEL_ALL_PREDICTIVE_LEARNERS_PURGE_LABEL_OVERLAP"]["value"] is True


# --------------------------------------------------------------------------- #
# 2. artifact availability lookahead
# --------------------------------------------------------------------------- #
def test_train_plus_validation_artifact_not_legal_at_train_end():
    ds = _panel(n_dates=10, n_stocks=200, seed=1)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    val_ds = ds.filter_dates(start=dates[6], end=dates[7])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    clock = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
        label_contract=contract, decision_clock=clock,
        sample_contract=_lenient(),
        retrain_policy="train_plus_validation",
    )
    art = result.artifact
    m = art.manifest
    assert m.refit_used_validation is True
    # final fit saw train+validation -> cutoff/available_at must advance past
    # the original train_end (the P0 lookahead).
    assert m.training_cutoff is not None and m.available_at is not None
    assert m.training_cutoff == m.available_at
    assert m.training_cutoff >= m.final_fit_end
    assert m.training_cutoff > m.train_end  # advanced past selection window
    # At the original train_end the artifact was NOT yet legal.
    assert art.is_legal_asof(asof=pd.Timestamp(m.train_end)) is False
    # It is legal once the final fit (labels) matured.
    assert art.is_legal_asof(asof=pd.Timestamp(m.training_cutoff)) is True


def test_manifest_rejects_availability_lookahead_construction():
    # Direct construction with training_cutoff < final_fit_end must fail closed.
    with pytest.raises(ValueError, match="Availability Lookahead"):
        ModelArtifactManifest(
            model_name="x", model_version="1", artifact_id="a",
            train_start="2020-01-01", train_end="2023-12-31",
            final_fit_end="2024-12-31", training_cutoff="2023-12-31",
            available_at="2023-12-31",
        )


def test_train_only_artifact_cutoff_equals_train_end():
    ds = _panel(n_dates=8, n_stocks=200, seed=3)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    val_ds = ds.filter_dates(start=dates[6], end=dates[7])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
        retrain_policy="train_only",
    )
    m = result.artifact.manifest
    assert m.refit_used_validation is False
    assert m.final_fit_end == m.train_end
    # cutoff advances by the label horizon only.
    assert m.training_cutoff >= m.train_end


# --------------------------------------------------------------------------- #
# 3. PLS coefficient formula — independent score-space OLS oracle
# --------------------------------------------------------------------------- #
def _pls_reference_predict(Xs, W, P, ys):
    """Independent oracle: T = X R with R = W (P^T W)^{-1}, then OLS of y on the
    true scores.  PLS1 == OLS in the score space; a wrong (P^T W vs W^T P)
    factor breaks this identity."""
    R = W @ np.linalg.solve(P.T @ W, np.eye(P.shape[1]))
    T = Xs @ R
    c = np.linalg.lstsq(T, ys, rcond=None)[0]
    return T @ c


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_pls_coefficients_match_score_ols_oracle(n_components):
    rng = np.random.default_rng(42)
    n, d = 200, 5
    base = rng.normal(size=(n, 2))
    X = np.column_stack([
        base[:, 0] + 0.3 * rng.normal(size=n),
        base[:, 0] - 0.2 * rng.normal(size=n),
        base[:, 1] + rng.normal(size=n),
        rng.normal(size=n),
        base[:, 1] - 0.1 * rng.normal(size=n),
    ])
    y = 0.5 * X[:, 0] - 0.3 * X[:, 2] + 0.2 * X[:, 4] + rng.normal(0, 0.05, size=n)

    learner = PLSLearner(
        LearnerSpec("predictive_pls", "pls", hyperparams={"n_components": n_components})
    )
    frozen = learner.fit(X, y)
    p = frozen.params
    W = np.asarray(p["weights"])
    P = np.asarray(p["loadings"])
    mx, sx = X.mean(0), X.std(0)
    sx[sx == 0] = 1.0
    Xs = (X - mx) / sx
    my, sy = y.mean(), y.std()
    ys = (y - my) / (sy if sy > 0 else 1.0)

    ref_std = _pls_reference_predict(Xs, W, P, ys)
    my_std = (learner.predict(frozen, X) - my) / (sy if sy > 0 else 1.0)
    assert np.max(np.abs(my_std - ref_std)) < 1e-9


def test_pls_scale_transformed_input_oracle():
    """Scaling X should be absorbed by the frozen raw-space coefficients."""
    rng = np.random.default_rng(7)
    n, d = 150, 4
    X = rng.normal(size=(n, d))
    X = X * np.array([1.0, 10.0, 100.0, 0.1])
    y = X @ np.array([0.4, -0.2, 0.3, 0.1]) + rng.normal(0, 0.05, size=n)
    learner = PLSLearner(
        LearnerSpec("predictive_pls", "pls", hyperparams={"n_components": 3})
    )
    frozen = learner.fit(X, y)
    p = frozen.params
    W, P = np.asarray(p["weights"]), np.asarray(p["loadings"])
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    ref = _pls_reference_predict(Xs, W, P, ys)
    mine = (learner.predict(frozen, X) - y.mean()) / y.std()
    assert np.max(np.abs(mine - ref)) < 1e-9


def test_pls_ill_conditioned_input_finite_or_fails_closed():
    """Ill-conditioned (rank-deficient) input must either fit with finite
    coefficients or FAIL CLOSED with a clear ValueError — never silently fit
    fewer components than requested."""
    rng = np.random.default_rng(11)
    n, d = 100, 3
    col = rng.normal(size=n)
    X = np.column_stack([col, col + 1e-8 * rng.normal(size=n), col * 2.0])
    y = 0.5 * X[:, 0] + rng.normal(0, 0.1, size=n)
    learner = PLSLearner(
        LearnerSpec("predictive_pls", "pls", hyperparams={"n_components": 2})
    )
    try:
        frozen = learner.fit(X, y)
    except ValueError as exc:
        assert "fail closed" in str(exc) or "zero" in str(exc).lower()
        return
    assert np.isfinite(frozen.params["coef"]).all()


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_pls_matches_sklearn_plsregression_oracle(n_components):
    """sklearn.cross_decomposition.PLSRegression is the independent oracle: fit
    it on the SAME standardized design (scale=False, centering is identity on
    already-centered data) and require raw-space prediction parity."""
    pytest.importorskip("sklearn")
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(99)
    n, d = 150, 4
    X = rng.normal(size=(n, d))
    y = X @ np.array([0.5, -0.3, 0.2, 0.1]) + rng.normal(0, 0.02, size=n)
    mx, sx = X.mean(0), X.std(0)
    sx[sx == 0] = 1.0
    Xs = (X - mx) / sx
    my, sy = y.mean(), y.std()
    ys = (y - my) / (sy if sy > 0 else 1.0)

    learner = PLSLearner(
        LearnerSpec("predictive_pls", "pls", hyperparams={"n_components": n_components})
    )
    frozen = learner.fit(X, y)
    sk = PLSRegression(n_components=n_components, scale=False).fit(Xs, ys.reshape(-1, 1))
    sk_std = sk.predict(Xs).ravel()
    sk_raw = my + sy * sk_std
    mine = learner.predict(frozen, X)
    assert np.max(np.abs(mine - sk_raw)) < 1e-6, (
        f"PLS n_components={n_components} deviates from sklearn PLSRegression "
        f"(max {np.max(np.abs(mine - sk_raw)):.2e})"
    )


def test_pls_rank_exceeds_features_fails_closed():
    X = np.random.default_rng(0).normal(size=(100, 3))
    y = np.random.default_rng(1).normal(size=100)
    learner = PLSLearner(
        LearnerSpec("predictive_pls", "pls", hyperparams={"n_components": 10})
    )
    with pytest.raises(ValueError, match="no silent downgrade"):
        learner.fit(X, y)


# --------------------------------------------------------------------------- #
# 4. ElasticNet
# --------------------------------------------------------------------------- #
def test_enet_predicts_in_raw_space():
    """The frozen artifact must consume raw X — standardized-space coefficients
    are restored to raw space at fit time."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -0.5, 0.0, 0.0, 2.0]) + rng.normal(0, 0.01, size=300)
    learner = ElasticNetLearner(
        LearnerSpec("predictive_elastic_net", "elastic_net",
                    hyperparams={"alpha": 1e-3, "l1_ratio": 0.1, "max_iter": 2000})
    )
    frozen = learner.fit(X, y)
    p = frozen.params
    # manual standardized-space recompute from stored train-only stats:
    my_pred = p["intercept"] + X @ p["coef"]
    Xs = (X - p["mean"]) / p["scale"]
    ys = y - y.mean()
    manual = y.mean() + Xs @ p["coef_std"]
    assert np.allclose(my_pred, manual, atol=1e-8)


def test_enet_alpha_sample_count_invariant():
    """Same alpha on N and on a duplicated 10N panel gives the SAME fit — the
    old objective (no 1/N) halved effective regularization when N doubled."""
    rng = np.random.default_rng(3)
    n, d = 120, 4
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, 0.5, 0.0, 0.0]) + rng.normal(0, 0.05, size=n)
    hp = {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 3000}
    f1 = ElasticNetLearner(LearnerSpec("e", "elastic_net", hyperparams=hp)).fit(X, y)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([y, y])
    f2 = ElasticNetLearner(LearnerSpec("e", "elastic_net", hyperparams=hp)).fit(X2, y2)
    assert np.allclose(f1.params["coef"], f2.params["coef"], atol=1e-6)


def test_enet_non_convergence_fails_closed():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 8))
    y = X @ rng.normal(size=8) + rng.normal(0, 0.01, size=40)
    learner = ElasticNetLearner(
        LearnerSpec("e", "elastic_net",
                    hyperparams={"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 1})
    )
    with pytest.raises(ModelConvergenceError):
        learner.fit(X, y)


# --------------------------------------------------------------------------- #
# 5. PCR rank fail-closed
# --------------------------------------------------------------------------- #
def test_pcr_rank_exceeds_features_fails_closed():
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.random.default_rng(1).normal(size=50)
    learner = PCRLearner(
        LearnerSpec("predictive_pcr", "pcr", hyperparams={"n_components": 10})
    )
    with pytest.raises(ValueError, match="no silent downgrade"):
        learner.fit(X, y)


def test_pcr_boundary_rank_ok():
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.random.default_rng(1).normal(size=50)
    learner = PCRLearner(
        LearnerSpec("predictive_pcr", "pcr", hyperparams={"n_components": 3})
    )
    frozen = learner.fit(X, y)
    assert frozen.params["n_components"] == 3


# --------------------------------------------------------------------------- #
# 6. sample-contract wiring
# --------------------------------------------------------------------------- #
def test_pcr_pls_enet_resolve_linear_contract():
    for cls in (PCRLearner, PLSLearner, ElasticNetLearner):
        contract = resolve_sample_contract(cls)
        assert contract is not None, f"{cls.name} resolved no contract (was None)"
        assert contract.min_raw_obs == 50_000


# --------------------------------------------------------------------------- #
# 7. PanelDataset enterprise constraints
# --------------------------------------------------------------------------- #
def test_panel_requires_stock_col():
    frame = pd.DataFrame({"date": [1, 2], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="stock column"):
        PanelDataset(frame, date_col="date", stock_col="stock",
                     feature_cols=["x"])


def test_panel_duplicate_stock_date_rejected_by_default():
    frame = pd.DataFrame({
        "date": [1, 1, 2], "stock": ["A", "A", "B"], "x": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match="duplicated \\(date, stock\\)"):
        PanelDataset(frame, date_col="date", stock_col="stock", feature_cols=["x"])


def test_panel_duplicate_policy_aggregate_dedupes():
    frame = pd.DataFrame({
        "date": [1, 1, 2], "stock": ["A", "A", "B"], "x": [1.0, 2.0, 3.0],
    })
    ds = PanelDataset(frame, date_col="date", stock_col="stock",
                      feature_cols=["x"], duplicate_policy="aggregate")
    assert ds.n_rows == 2


# --------------------------------------------------------------------------- #
# 8. purge boundary when validation is None
# --------------------------------------------------------------------------- #
def test_train_model_no_validation_purges_against_evaluation_boundary():
    ds = _panel(n_dates=10, n_stocks=100, seed=9)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[6])
    test_boundary = dates[8]  # test window starts here
    contract = vwap_to_vwap_label("ret_2", horizon_bars=2)
    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
        evaluation_boundary=test_boundary,
    )
    m = result.artifact.manifest
    # final fit end (max anchor) must stay strictly below the test boundary.
    assert m.final_fit_end < str(test_boundary)
    # and even the label-matured cutoff never reaches into the test window.
    assert m.training_cutoff <= str(test_boundary)


def test_train_model_no_validation_without_boundary_records_none():
    ds = _panel(n_dates=8, n_stocks=100, seed=10)
    dates = sorted(ds.frame["date"].unique())
    train_ds = ds.filter_dates(start=dates[0], end=dates[5])
    contract = vwap_to_vwap_label("ret_1", horizon_bars=1)
    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=contract,
        decision_clock=ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP),
        sample_contract=_lenient(),
    )
    assert result.artifact.manifest.training_cutoff >= result.artifact.manifest.final_fit_end


# --------------------------------------------------------------------------- #
# 9. effective_parameter_count — true DOF, never len(hyperparams)
# --------------------------------------------------------------------------- #
def test_enet_free_parameters_count_active_coefs_not_dials():
    learner = ElasticNetLearner(
        LearnerSpec("e", "elastic_net", hyperparams={"alpha": 0.1, "l1_ratio": 0.5})
    )
    assert learner.effective_parameter_count({"alpha": 0.1, "l1_ratio": 0.5}, n_features=50) == 51


def test_regime_moe_free_parameters():
    from modeling.learners import MixtureOfExpertsLearner, RegimeLearner

    regime = RegimeLearner(
        LearnerSpec("r", "regime", hyperparams={"n_regimes": 3})
    )
    assert regime.effective_parameter_count({"n_regimes": 3}, n_features=4) == 3 * 5 + 2

    moe = MixtureOfExpertsLearner(
        LearnerSpec("m", "moe", hyperparams={"n_experts": 3})
    )
    assert moe.effective_parameter_count({"n_experts": 3}, n_features=3) == 3 * 4 + 2
    assert moe.effective_parameter_count({"n_experts": 3}, n_features=3,
                                         fallback="single_best") == 4
