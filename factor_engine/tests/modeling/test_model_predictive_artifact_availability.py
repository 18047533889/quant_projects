# -*- coding: utf-8 -*-
"""P0 fix — Artifact Availability Lookahead.

Previously the ``train_plus_validation`` flow refit the final model on
train+validation but froze ``training_cutoff`` / ``available_at`` at the
original ``train_end`` — a backtest in the validation period would believe the
artifact was available while the model had actually seen the whole validation
window.  Fix: the manifest records ``selection_train_end`` / ``validation_end``
/ ``final_fit_end`` / ``final_fit_start`` / ``refit_used_validation`` and the
artifact is only legal at/after the last moment the final fit actually used
data (final-fit anchors + label horizon), never at ``train_end``.

Assertions here follow the *implemented* contract: ``training_cutoff`` is the
maturity-adjusted cutoff (``final_fit_end + horizon`` on the bar calendar),
which is strictly >= ``final_fit_end`` — the key guarantee is that the artifact
is NOT legal before the end of the data its final fit consumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.artifact import ModelArtifactManifest
from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    LabelContract,
    SampleAdequacyContract,
    ashare_decision_clock,
)
from modeling.dataset import PanelDataset
from modeling.learners import PCRLearner
from modeling.trainer import PreprocessingSpec, train_model


def _lenient() -> SampleAdequacyContract:
    return SampleAdequacyContract(
        min_raw_obs=500,
        min_effective_obs=300,
        min_unique_dates=3,
        min_unique_stocks=30,
        min_obs_per_parameter=10,
    )


def _multi_year_panel(n_stocks: int = 100, seed: int = 0) -> PanelDataset:
    """A pooled panel covering train 2019-2023 / validation 2024 / test 2025."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", "2025-12-31")
    rows: list[list] = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.normal(size=4)
            rows.append(
                [d, f"S{s:03d}", *x.tolist(), float(0.5 * x[0] - 0.2 * x[1] + rng.normal(0, 0.1))]
            )
    frame = pd.DataFrame(rows, columns=["date", "stock", "f0", "f1", "f2", "f3", "label"])
    return PanelDataset(
        frame=frame, date_col="date", stock_col="stock",
        feature_cols=["f0", "f1", "f2", "f3"], label_col="label",
    )


def _contract() -> LabelContract:
    return LabelContract(label_name="ret_1", horizon_bars=1)


def _clock():
    return ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)


# --------------------------------------------------------------------------- #
# #1 — train_plus_validation final artifact must not be legal at train_end
# --------------------------------------------------------------------------- #
def test_train_plus_validation_cutoff_advances_past_train_end():
    ds = _multi_year_panel()
    train_ds = ds.filter_dates(start="2019-01-01", end="2023-12-31")
    val_ds = ds.filter_dates(start="2024-01-01", end="2024-12-31")

    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}, {"n_components": 3}],
        label_contract=_contract(),
        decision_clock=_clock(),
        sample_contract=_lenient(),
        retrain_policy="train_plus_validation",
    )
    art = result.artifact
    m = art.manifest

    # The final fit consumed the whole validation window.
    assert m.refit_used_validation is True
    assert m.final_fit_start is not None
    assert m.final_fit_end == m.validation_end
    assert m.final_fit_end != m.train_end

    # Selection happened on the train window only.
    assert m.selection_train_end == m.train_end

    # Cutoff / availability advanced past the selection window (the P0 bug:
    # they used to freeze at train_end).
    assert m.training_cutoff is not None and m.available_at is not None
    assert m.training_cutoff >= m.final_fit_end
    assert m.available_at >= m.final_fit_end
    assert m.training_cutoff > m.train_end  # never the old train_end value

    # Not legal at the old train_end; legal once the final fit matured.
    assert art.is_legal_asof(asof=pd.Timestamp(m.train_end)) is False
    assert art.is_legal_asof(asof=pd.Timestamp(m.training_cutoff)) is True


def test_train_only_keeps_cutoff_at_train_window():
    """Validation exists but the model is NOT refit on it (train_only): the
    artifact saw only train data, so its final-fit window is the train window."""
    ds = _multi_year_panel()
    train_ds = ds.filter_dates(start="2019-01-01", end="2023-12-31")
    val_ds = ds.filter_dates(start="2024-01-01", end="2024-12-31")

    result = train_model(
        PCRLearner, train_ds, val_ds,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=_contract(),
        decision_clock=_clock(),
        sample_contract=_lenient(),
        retrain_policy="train_only",
    )
    m = result.artifact.manifest
    assert m.refit_used_validation is False
    assert m.final_fit_end == m.train_end
    assert m.training_cutoff >= m.final_fit_end
    assert m.available_at >= m.final_fit_end


def test_no_validation_train_to_test_cutoff_stays_train_window():
    """validation_ds is None (train→test only): final fit is the train window."""
    ds = _multi_year_panel()
    train_ds = ds.filter_dates(start="2019-01-01", end="2023-12-31")

    result = train_model(
        PCRLearner, train_ds, None,
        preprocessing_spec=PreprocessingSpec(),
        hyperparam_grid=[{"n_components": 2}],
        label_contract=_contract(),
        decision_clock=_clock(),
        sample_contract=_lenient(),
    )
    m = result.artifact.manifest
    assert m.refit_used_validation is False
    assert m.final_fit_end == m.train_end
    assert m.training_cutoff >= m.final_fit_end
    assert m.available_at >= m.final_fit_end


# --------------------------------------------------------------------------- #
# #1 — manifest fails closed on lookahead construction
# --------------------------------------------------------------------------- #
def test_manifest_rejects_training_cutoff_before_final_fit_end():
    with pytest.raises(ValueError, match="Availability Lookahead"):
        ModelArtifactManifest(
            model_name="x", model_version="1", artifact_id="a",
            train_start="2020-01-01", train_end="2023-12-31",
            final_fit_end="2024-12-31", training_cutoff="2023-12-31",
            available_at="2023-12-31",
        )


def test_manifest_rejects_available_at_before_final_fit_end():
    with pytest.raises(ValueError):
        ModelArtifactManifest(
            model_name="x", model_version="1", artifact_id="a",
            train_start="2020-01-01", train_end="2023-12-31",
            final_fit_end="2024-12-31", training_cutoff="2024-12-31",
            available_at="2024-06-01",
        )


def test_manifest_roundtrips_new_fields():
    m = ModelArtifactManifest(
        model_name="x", model_version="1", artifact_id="a",
        train_start="2020-01-01", train_end="2023-12-31",
        validation_start="2024-01-01", validation_end="2024-12-31",
        selection_train_end="2023-12-31",
        final_fit_start="2020-01-01", final_fit_end="2024-12-31",
        refit_used_validation=True,
        training_cutoff="2024-12-31", available_at="2024-12-31",
    )
    assert m.refit_used_validation is True
    assert m.selection_train_end == "2023-12-31"
    assert m.final_fit_start == "2020-01-01"
    assert m.final_fit_end == "2024-12-31"
    # lineage hash includes the new lineage facts (different window → different hash)
    m2 = ModelArtifactManifest(
        model_name="x", model_version="1", artifact_id="a",
        train_start="2020-01-01", train_end="2023-12-31",
        validation_start="2024-01-01", validation_end="2024-12-31",
        selection_train_end="2023-12-31",
        final_fit_start="2020-01-01", final_fit_end="2024-06-30",
        refit_used_validation=True,
        training_cutoff="2024-06-30", available_at="2024-06-30",
    )
    assert m.lineage_hash() != m2.lineage_hash()
