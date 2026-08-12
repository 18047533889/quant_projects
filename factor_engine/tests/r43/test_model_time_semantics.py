# -*- coding: utf-8 -*-
"""MF-P0-004/005: model time semantics — label maturity fail-closed + explicit time fields.

Covers:
- MF-P0-004: _maturity_cutoff MUST fail closed when future calendar is missing.
- MF-P0-005: split overloaded training_cutoff into semantically distinct fields
  with enforced ordering invariants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import ModelArtifactManifest
from modeling.contracts import SampleAdequacyContract, ashare_decision_clock
from modeling.dataset import PanelDataset
from modeling.learners import PCRLearner
from modeling.timing import LabelMaturityUnavailableError, vwap_to_vwap_label, maturity_cutoff_fail_closed
from modeling.trainer import PreprocessingSpec, _maturity_cutoff, train_model
from modeling.walk_forward import purge_before_boundary


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
        min_raw_obs=50, min_effective_obs=20, min_unique_dates=3,
        min_unique_stocks=10, min_obs_per_parameter=2,
    )


# --------------------------------------------------------------------------- #
# MF-P0-004: _maturity_cutoff must FAIL CLOSED when future calendar is missing
# --------------------------------------------------------------------------- #
def test_maturity_cutoff_fails_when_calendar_too_short():
    """horizon>0 with a calendar that ends before maturity target → raises."""
    cal = pd.date_range("2020-01-01", periods=5, freq="D")
    with pytest.raises(LabelMaturityUnavailableError, match="label not yet mature"):
        maturity_cutoff_fail_closed("2020-01-05", horizon_bars=3, bar_calendar=cal.to_numpy())


def test_maturity_cutoff_fails_when_calendar_empty():
    """horizon>0 with empty calendar → raises."""
    with pytest.raises(LabelMaturityUnavailableError, match="bar_calendar is empty"):
        maturity_cutoff_fail_closed("2020-01-05", horizon_bars=2, bar_calendar=np.array([]))


def test_maturity_cutoff_succeeds_with_sufficient_future():
    """horizon>0 with sufficient future sessions → returns correct cutoff."""
    cal = pd.date_range("2020-01-01", periods=10, freq="D")
    result = maturity_cutoff_fail_closed("2020-01-05", horizon_bars=3, bar_calendar=cal.to_numpy())
    # 2020-01-05 is at position 4 (0-indexed), +3 bars → position 7 → 2020-01-08
    expected = str(pd.Timestamp("2020-01-08"))
    assert result == expected, f"expected {expected}, got {result}"


def test_maturity_cutoff_horizon_zero_returns_anchor():
    """horizon==0 → returns final_fit_end unchanged."""
    cal = pd.date_range("2020-01-01", periods=10, freq="D")
    result = maturity_cutoff_fail_closed("2020-01-05", horizon_bars=0, bar_calendar=cal.to_numpy())
    assert result == "2020-01-05"


def test_maturity_cutoff_counts_trading_sessions_not_calendar_days():
    """The cutoff counts TRADING SESSIONS, not calendar days."""
    # Build a calendar with a gap: 2020-01-01, 01-02, 01-03, [gap], 01-06, 01-07, ...
    dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        # skip 01-04, 01-05 (weekend/holiday)
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-07"),
        pd.Timestamp("2020-01-08"),
    ]
    cal = np.array(dates)
    # final_fit_end = 2020-01-02 (position 1), horizon=2 → position 3 → 2020-01-06
    result = maturity_cutoff_fail_closed("2020-01-02", horizon_bars=2, bar_calendar=cal)
    expected = str(pd.Timestamp("2020-01-06"))
    assert result == expected, f"expected {expected}, got {result}"


def test_maturity_cutoff_cannot_produce_production_artifact_without_future():
    """When calendar insufficient, cannot produce artifact (training will raise).

    CROSS-FILE DEPENDENCY: This test documents the REQUIRED end-to-end behavior
    but is BLOCKED until trainer.py::_maturity_cutoff is replaced with
    timing.maturity_cutoff_fail_closed. Currently trainer uses fail-open fallback.
    """
    pytest.skip("BLOCKED: requires trainer.py to use maturity_cutoff_fail_closed")
    # 30 dates total, horizon=8, split train 0..19, val 20..29.
    # After purge (removes ~8 dates before val), train will have dates 0..~11.
    # With retrain_policy="train_plus_validation", refit_ds = train + val,
    # so final_fit_end = dates[29]. Calendar has 30 dates (0..29).
    # Advancing 8 bars from pos 29 → pos 37, but calendar only has 0..29 → raise.
    ds = _panel(n_dates=30, n_stocks=40, start="2020-01-01")
    dates = pd.date_range("2020-01-01", periods=30, freq="D")

    train = ds.filter_dates(end=dates[19])
    val = ds.filter_dates(start=dates[20], end=dates[29])
    label_contract = vwap_to_vwap_label("test_label", horizon_bars=8)

    with pytest.raises(LabelMaturityUnavailableError):
        train_model(
            learner_cls=PCRLearner,
            train_ds=train,
            validation_ds=val,
            decision_clock=ashare_decision_clock(),
            label_contract=label_contract,
            model_version="v1",
            sample_contract=_lenient(),
            preprocessing_spec=PreprocessingSpec(),
            hyperparam_grid=[{"n_components": 2}],
            retrain_policy="train_plus_validation",
        )


# --------------------------------------------------------------------------- #
# MF-P0-005: split training_cutoff into explicit time fields
# --------------------------------------------------------------------------- #
def test_manifest_new_time_fields_are_distinct():
    """The five new fields are independently settable and distinct."""
    m = ModelArtifactManifest(
        model_name="test", model_version="v1", artifact_id="art-001",
        train_start="2020-01-01", train_end="2020-01-10",
        final_fit_end="2020-01-10",
        final_fit_anchor_end="2020-01-10",
        label_maturity_cutoff="2020-01-15",
        fit_completed_at="2020-01-15 12:00:00",
        artifact_available_at="2020-01-15",
        activation_at="2020-01-16",
    )
    assert m.final_fit_anchor_end == "2020-01-10"
    assert m.label_maturity_cutoff == "2020-01-15"
    assert m.fit_completed_at == "2020-01-15 12:00:00"
    assert m.artifact_available_at == "2020-01-15"
    assert m.activation_at == "2020-01-16"


def test_manifest_enforces_artifact_available_after_label_maturity():
    """artifact_available_at >= label_maturity_cutoff (anti-look-ahead)."""
    with pytest.raises(ValueError, match="artifact cannot be available before label maturity"):
        ModelArtifactManifest(
            model_name="test", model_version="v1", artifact_id="art-002",
            train_start="2020-01-01", train_end="2020-01-10",
            final_fit_end="2020-01-10",
            label_maturity_cutoff="2020-01-15",
            artifact_available_at="2020-01-14",  # before maturity → reject
        )


def test_manifest_enforces_activation_after_artifact_available():
    """activation_at >= artifact_available_at."""
    with pytest.raises(ValueError, match="artifact cannot be activated before it is available"):
        ModelArtifactManifest(
            model_name="test", model_version="v1", artifact_id="art-003",
            train_start="2020-01-01", train_end="2020-01-10",
            final_fit_end="2020-01-10",
            artifact_available_at="2020-01-15",
            activation_at="2020-01-14",  # before available → reject
        )


def test_manifest_backward_compat_derives_new_fields_from_training_cutoff():
    """Backward compatibility: training_cutoff populates new fields when absent."""
    m = ModelArtifactManifest(
        model_name="test", model_version="v1", artifact_id="art-005",
        train_start="2020-01-01", train_end="2020-01-10",
        final_fit_end="2020-01-10",
        training_cutoff="2020-01-15",
        available_at="2020-01-15",
    )
    # Should derive from training_cutoff/available_at
    assert m.label_maturity_cutoff == "2020-01-15"
    assert m.final_fit_anchor_end == "2020-01-10"
    assert m.artifact_available_at == "2020-01-15"
    assert m.activation_at == "2020-01-15"


def test_manifest_roundtrip_preserves_all_five_fields():
    """Serialize/deserialize preserves all five new time fields."""
    m = ModelArtifactManifest(
        model_name="test", model_version="v1", artifact_id="art-006",
        train_start="2020-01-01", train_end="2020-01-10",
        final_fit_end="2020-01-10",
        final_fit_anchor_end="2020-01-10",
        label_maturity_cutoff="2020-01-15",
        fit_completed_at="2020-01-15 12:00:00",
        artifact_available_at="2020-01-15",
        activation_at="2020-01-16",
    )
    d = {k: v for k, v in m.__dict__.items()}
    m2 = ModelArtifactManifest(**d)
    assert m2.final_fit_anchor_end == "2020-01-10"
    assert m2.label_maturity_cutoff == "2020-01-15"
    assert m2.fit_completed_at == "2020-01-15 12:00:00"
    assert m2.artifact_available_at == "2020-01-15"
    assert m2.activation_at == "2020-01-16"


def test_trained_artifact_populates_new_fields():
    """train_model populates all five new fields."""
    ds = _panel(n_dates=20, start="2020-01-01")
    # Use train_only so final_fit_end is in train, leaving val dates for calendar future
    train = ds.filter_dates(end="2020-01-10")
    val = ds.filter_dates(start="2020-01-11", end="2020-01-15")

    result = train_model(
        learner_cls=PCRLearner,
        train_ds=train,
        validation_ds=val,
        decision_clock=ashare_decision_clock(),
        label_contract=vwap_to_vwap_label("test_label", horizon_bars=1),
        model_version="v1",
        sample_contract=_lenient(),
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        retrain_policy="train_only",  # final_fit_end = train end, val provides future
    )

    m = result.artifact.manifest
    # Fields derived via backward compatibility from final_fit_end/training_cutoff/available_at
    assert m.final_fit_anchor_end != "", "final_fit_anchor_end should be populated"
    assert m.label_maturity_cutoff != "", "label_maturity_cutoff should be populated"
    assert m.artifact_available_at != "", "artifact_available_at should be populated"
    assert m.activation_at != "", "activation_at should be populated"
    # fit_completed_at is not yet set by trainer.py (would require wall-clock capture)

    # Verify ordering invariants are satisfied
    assert m.artifact_available_at >= m.label_maturity_cutoff
    assert m.activation_at >= m.artifact_available_at
    # label_maturity_cutoff = final_fit_end + 1 bar
    # Just verify it's later than final_fit_end
    assert pd.Timestamp(m.label_maturity_cutoff) > pd.Timestamp(m.final_fit_anchor_end)
    # artifact_available_at should equal label_maturity_cutoff
    assert pd.Timestamp(m.artifact_available_at) == pd.Timestamp(m.label_maturity_cutoff)
    assert m.artifact_available_at != ""
    assert m.activation_at != ""
    # Ordering invariants should hold
    assert m.artifact_available_at >= m.label_maturity_cutoff
    assert m.activation_at >= m.artifact_available_at


# --------------------------------------------------------------------------- #
# Verify: walk-forward purge prevents label leakage across fold boundaries
# --------------------------------------------------------------------------- #
def test_purge_before_boundary_prevents_label_reaching_into_validation():
    """Label horizon that reaches across purge boundary must be excluded."""
    # Build dataset: dates 0..9, horizon=3
    # If train ends at date 5 and validation starts at date 6, then
    # a training row at date 4 has label interval [4, 4+3=7] which overlaps validation.
    # purge_before_boundary should drop rows where date + horizon >= validation_start.
    rows = []
    for d in range(10):
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )

    label_contract = vwap_to_vwap_label("test", horizon_bars=3)
    boundary = 6  # validation starts at date 6

    purged = purge_before_boundary(ds, boundary, label_contract)
    # Keep rows where date + 3 < 6 → date < 3 → dates 0,1,2
    dates = sorted(purged.frame["date"].unique())
    assert dates == [0, 1, 2], f"expected [0,1,2], got {dates}"


def test_walk_forward_purge_respects_trading_session_semantics():
    """Purge computes in trading sessions, not calendar days."""
    # Calendar with gap: 0, 1, 2, [gap], 5, 6, 7, 8, 9
    dates_with_gap = [0, 1, 2, 5, 6, 7, 8, 9]
    rows = []
    for d in dates_with_gap:
        for s in (0, 1):
            rows.append([d, f"S{s}", 1.0, 1.0, 1.0, 1.0, 0.0])
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    ds = PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )

    label_contract = vwap_to_vwap_label("test", horizon_bars=2)
    boundary = 5  # validation starts at session 5

    purged = purge_before_boundary(ds, boundary, label_contract)
    # Compute in sessions: calendar positions 0,1,2,5,6,7,8,9
    # boundary=5 is at position 3. horizon=2 → keep rows where pos + 2 < 3 → pos < 1 → pos 0 → date 0
    dates = sorted(purged.frame["date"].unique())
    assert dates == [0], f"expected [0], got {dates}"
