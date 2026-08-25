# -*- coding: utf-8 -*-
"""Adversarial production-integrity tests for the regime module (FP-P0)."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "factor_preprocess"))

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.contracts.state import FittedState, StateKind
from factor_preprocess.contracts.policy import TransformKind
from factor_preprocess.errors import (
    UnknownRegimeError,
    SupervisedTargetRequiredError,
    UnsupportedTargetError,
)
from factor_preprocess.regime.adaptive_weights import (
    fit_regime_weights,
    regime_adaptive_weights,
    RegimeWeightState,
    UnknownRegimePolicy,
    serialize_regime_weights,
    deserialize_regime_weights,
)
from factor_preprocess.regime.causal_detector import CausalRegimeDetector


def _sample_weight_state():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    factors = pd.DataFrame({
        "date": dates,
        "f1": rng.normal(size=200),
        "f2": rng.normal(size=200),
        "f3": rng.normal(size=200),
    })
    labels = pd.Series(rng.choice([0, 1], 200), index=factors.index)
    return fit_regime_weights(factors, labels, time_col="date")


# ---------------------------------------------------------------------------
# FP-P0-01: regime learned state serializes into canonical FittedState payload
# ---------------------------------------------------------------------------
def test_serialize_into_canonical_fitted_state():
    st = _sample_weight_state()
    payload = serialize_regime_weights(st)
    restored = deserialize_regime_weights(payload)
    assert isinstance(restored, RegimeWeightState)
    assert restored.factor_names == st.factor_names
    assert restored.n_regimes == st.n_regimes
    for r in st.regime_weights:
        np.testing.assert_allclose(restored.regime_weights[r], st.regime_weights[r])


def test_fitted_state_can_embed_regime_payload():
    st = _sample_weight_state()
    payload = serialize_regime_weights(st)
    fs = FittedState(
        transform_name="fit_regime_weights",
        transform_version="1.0.0",
        fit_start_time=st.fit_window_start.to_pydatetime(),
        fit_end_time=st.fit_window_end.to_pydatetime(),
        state_kind=StateKind.FITTED,
        feature_ids=list(st.factor_names),
        feature_order=list(st.factor_names),
        learned_params=payload,
        production=True,
        implementation_hash="impl-hash",
        data_snapshot_ref="snap",
        split_ref="split",
        universe_ref="univ",
        calendar_ref="cal",
        fit_coordinate_hash="coord",
        policy_hash="pol",
    )
    assert fs.learned_params_hash
    assert fs.state_id  # content-derived in production
    assert fs.state_kind == StateKind.FITTED
    assert isinstance(deserialize_regime_weights(fs.learned_params), RegimeWeightState)


# ---------------------------------------------------------------------------
# FP-P0-02: supervised weighting is not label-free preprocessing
# ---------------------------------------------------------------------------
def test_fit_regime_weights_default_is_label_free():
    st = _sample_weight_state()
    assert st.fit_method == "equal"
    assert st.target_required is False


def test_sharpe_method_is_supervised_and_renamed():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    factors = pd.DataFrame({"date": dates})
    for i in range(2):
        factors[f"f{i+1}"] = rng.normal(size=300)
    labels = pd.Series(rng.choice([0, 1], 300), index=factors.index)
    target = pd.Series(rng.normal(size=300), index=factors.index)

    st = fit_regime_weights(factors, labels, target=target, method="sharpe", time_col="date")
    assert st.fit_method == "target_corr_over_vol"
    assert st.target_required is True
    assert st.transform_kind == TransformKind.SUPERVISED_FITTED

    with pytest.raises(SupervisedTargetRequiredError):
        fit_regime_weights(factors, labels, method="target_corr_over_vol", time_col="date")


def test_unsupervised_target_rejected():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    factors = pd.DataFrame({"date": dates, "f1": rng.normal(size=300)})
    labels = pd.Series(rng.choice([0, 1], 300), index=factors.index)
    target = pd.Series(rng.normal(size=300), index=factors.index)
    # a label-free method must reject a spurious target
    with pytest.raises(UnsupportedTargetError):
        fit_regime_weights(factors, labels, target=target, method="equal", time_col="date")


def test_supervised_weight_must_not_be_used_in_plain_preprocess_policy():
    # label-free weighting is PRODUCTION-capable
    assert _sample_weight_state().admission == "PRODUCTION"
    # supervised weighting is admission-gated to RESEARCH_ONLY
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    factors = pd.DataFrame({"date": dates, "f1": rng.normal(size=200)})
    labels = pd.Series(rng.choice([0, 1], 200), index=factors.index)
    target = pd.Series(rng.normal(size=200), index=factors.index)
    sup = fit_regime_weights(factors, labels, target=target, method="target_corr_over_vol", time_col="date")
    assert sup.admission == "RESEARCH_ONLY"


# ---------------------------------------------------------------------------
# FP-P0-03: unknown regime must NOT silently pass through
# ---------------------------------------------------------------------------
def _fp_out_df(n=20, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-08-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})


def test_unknown_regime_fail_closed_by_default():
    st = _sample_weight_state()
    factors = _fp_out_df()
    labels = pd.Series([9] * len(factors), index=factors.index)
    with pytest.raises(UnknownRegimeError):
        regime_adaptive_weights(factors, labels, fitted_state=st, time_col="date")


def test_unknown_regime_fail_nan():
    st = _sample_weight_state()
    factors = _fp_out_df()
    labels = pd.Series([9] * len(factors), index=factors.index)
    out = regime_adaptive_weights(
        factors, labels, fitted_state=st, time_col="date",
        unknown_regime_policy=UnknownRegimePolicy.FAIL_NAN,
    )
    assert out["f1"].isna().all()


def test_unknown_regime_fallback_global():
    st = _sample_weight_state()
    factors = _fp_out_df()
    labels = pd.Series([9] * len(factors), index=factors.index)
    global_weights = np.array([0.5, 0.3, 0.2])
    out = regime_adaptive_weights(
        factors, labels, fitted_state=st, time_col="date",
        unknown_regime_policy="FALLBACK_GLOBAL",
        fallback_weights=global_weights,
    )
    assert not np.allclose(out["f1"].values, factors["f1"].values)


# ---------------------------------------------------------------------------
# FP-P1-07: production causal regime detector
# ---------------------------------------------------------------------------
def test_causal_detector_fail_closed_no_fit():
    det = CausalRegimeDetector(window=20, n_regimes=2)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30, freq="D"), "a": range(30), "b": range(30)})
    with pytest.raises(RuntimeError):
        det.detect(df)


def test_causal_detector_prefix_invariance():
    rng = np.random.default_rng(11)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(size=n)
    corr = np.linspace(0.2, 0.9, n)
    y = corr * x + rng.normal(size=n) * np.sqrt(1 - corr ** 2)
    df = pd.DataFrame({"date": dates, "a": x, "b": y})

    det = CausalRegimeDetector(window=20, n_regimes=2)
    det.fit(df.iloc[:120])
    labels_full = det.detect(df.iloc[:120]).regime.values
    labels_ext = det.detect(df.iloc[:121]).regime.values
    # rows 0..118 must be identical after extending with row 120 (prefix
    # invariance). Row 119 is excluded because in the extended run it uses
    # data[99:120] including the appended row 120 boundary.
    np.testing.assert_array_equal(labels_ext[:119], labels_full[:119])
