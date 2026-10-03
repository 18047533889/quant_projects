"""Bounded contract checks for automatic SMA proposals and FE execution."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest


def _prior_window_mean(values, window):
    """Independent causal oracle: previous complete finite observations only."""
    result = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        prior = values[max(0, i - window):i]
        if len(prior) == window and np.isfinite(prior).all():
            result[i] = sum(float(value) for value in prior) / window
    return result


def test_fe_sma_adapter_matches_independent_sparse_asset_oracle():
    from factor_optimizer.adapters.repair_execution import compile_value_repair

    dates = pd.date_range("2026-01-01", periods=5)
    frame = pd.DataFrame({
        "asset_id": ["A", "B"] * 5,
        "date": np.repeat(dates, 2),
        "value": [1., 10., 2., np.nan, 3., 30., 4., 40., 5., 50.],
    })
    plan = compile_value_repair(
        "CAUSAL_SMOOTHING",
        {"method": "SMA", "natural_time_scale_relative": 1.0},
        natural_time_scale=3,
        training_context_ref="bounded-sma-test",
    )
    actual = plan.execute(frame, allow_research=True).to_numpy()

    expected = np.full(len(frame), np.nan)
    for asset in frame["asset_id"].unique():
        rows = np.flatnonzero(frame["asset_id"].to_numpy() == asset)
        expected[rows] = _prior_window_mean(frame["value"].to_numpy()[rows], 3)
    np.testing.assert_equal(actual, expected)
    assert plan.transform == "trailing_sma"
    assert dict(plan.parameters) == {"window": 3, "min_periods": 3}


@pytest.mark.parametrize("window,min_periods", [(0, 0), (3, 4), (True, 1)])
def test_fe_lagged_sma_rejects_invalid_window_contract(window, min_periods):
    from factor_engine.backend.long_smoothing import lagged_mean

    frame = pd.DataFrame({"asset_id": ["A"], "date": [1], "value": [1.]})
    with pytest.raises(ValueError):
        lagged_mean(frame, window=window, min_periods=min_periods)


def test_automatic_search_evaluates_sma_on_train_and_freezes_winner():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch

    rng = np.random.default_rng(831)
    n_time, n_assets, window = 240, 40, 3
    values = rng.normal(size=(n_time, n_assets))
    labels_values = np.full_like(values, np.nan)
    labels_values[window:] = np.stack([
        values[t - window:t].mean(axis=0) for t in range(window, n_time)
    ])
    times = np.arange(n_time)
    time_axis = AxisRef("time", "int", n_time, times)
    asset_axis = AxisRef("asset", "str", n_assets,
                         np.asarray([f"asset-{i}" for i in range(n_assets)]))
    batch = FactorBatch(("signal",), time_axis, asset_axis, values[:, :, None])
    labels = LabelBundle(
        "sma-oracle", labels_values, 1,
        decision_time=tuple(times),
        label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2),
        asset_axis=asset_axis,
    )
    config = BatchOptimizationConfig(
        families=("CAUSAL_SMOOTHING",), natural_time_scale=3,
        selection_objective="rank_ic", bootstrap_draws=99,
    )

    baseline = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    result = baseline.factors["signal"]
    sma_candidates = [candidate for candidate in result.candidates
                      if candidate.get("transform") == "trailing_sma"]
    assert sma_candidates
    assert any(candidate["status"] == "train_evaluated" for candidate in sma_candidates)
    assert result.validation_candidate_identity == result.plan_identity
    assert result.selected_family == "CAUSAL_SMOOTHING"
    assert result.plan.transform == "trailing_sma"

    altered_values = values.copy()
    altered_values[144:] = rng.normal(loc=100., size=altered_values[144:].shape)
    altered_labels = labels_values.copy()
    altered_labels[144:] = rng.normal(loc=-100., size=altered_labels[144:].shape)
    altered = optimize_factor_batch(
        replace(batch, values=altered_values[:, :, None]),
        replace(labels, values=altered_labels),
        config=config, allow_research=True,
    ).factors["signal"]
    assert altered.validation_candidate_identity == result.validation_candidate_identity
    assert altered.train_gain == result.train_gain
