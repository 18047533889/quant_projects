# -*- coding: utf-8 -*-
"""Focused R41 evaluation/OOS/IC correctness tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.evaluation import (
    EvaluationContractError,
    block_aware_ic,
    cross_sectional_ic,
    evaluate_predictions,
)


def _panel(n_dates=3, n_assets=10):
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=n_dates), n_assets)
    securities = np.tile([f"sid-{i:02d}" for i in range(n_assets)], n_dates)
    pred = np.tile(np.arange(n_assets, dtype=float), n_dates)
    y = pred.copy()
    return pred, y, dates, securities


def test_turnover_uses_security_identity_and_is_row_order_invariant():
    pred, y, dates, securities = _panel()
    first = evaluate_predictions(pred, y, dates, securities)
    rng = np.random.default_rng(42)
    order = np.concatenate([
        np.flatnonzero(dates == date)[rng.permutation(10)] for date in pd.unique(dates)
    ])
    shuffled = evaluate_predictions(pred[order], y[order], dates[order], securities[order])
    assert first.turnover == pytest.approx(0.0)
    assert shuffled.turnover == pytest.approx(first.turnover)
    assert shuffled.long_short_spread == pytest.approx(first.long_short_spread)


def test_tied_scores_use_security_id_as_canonical_tie_break():
    pred, y, dates, securities = _panel(n_dates=2)
    pred[:] = 1.0
    first = evaluate_predictions(pred, y, dates, securities)
    reverse = np.concatenate([np.flatnonzero(dates == date)[::-1] for date in pd.unique(dates)])
    second = evaluate_predictions(pred[reverse], y[reverse], dates[reverse], securities[reverse])
    assert first.turnover == pytest.approx(0.0)
    assert second.turnover == pytest.approx(first.turnover)
    assert second.long_short_spread == pytest.approx(first.long_short_spread)


def test_rank_ic_is_mean_daily_not_pooled_and_icir_uses_ddof_one():
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=2), [3, 30])
    pred = np.concatenate([np.arange(3), np.arange(30)]).astype(float)
    y = np.concatenate([np.arange(3), np.arange(30)[::-1]]).astype(float)
    report = evaluate_predictions(pred, y, dates)
    assert report.mean_daily_rank_ic == pytest.approx(0.0)
    assert report.rank_ic == pytest.approx(report.mean_daily_rank_ic)
    assert report.pooled_rank_correlation != pytest.approx(report.rank_ic)
    bundle = cross_sectional_ic(y, pred, dates)
    assert bundle["rank_ic"] == pytest.approx(0.0)
    assert bundle["icir"] == pytest.approx(0.0)


def test_block_aware_ic_uses_calendar_positions_not_valid_date_compaction():
    sessions = list(pd.bdate_range("2024-01-02", periods=4))
    dates = np.repeat(sessions, 3)
    pred = np.tile(np.arange(3), 4).astype(float)
    y = pred.copy()
    y[3:6] = 1.0  # session 1 has undefined IC
    block_dates, block_ics = block_aware_ic(
        pred, y, dates, overlap_horizon=2, calendar_sessions=sessions
    )
    assert block_dates == [sessions[0], sessions[2]]
    assert block_ics.tolist() == pytest.approx([1.0, 1.0])


def test_weights_and_eligible_universe_are_strict_and_explicit():
    pred, y, dates, securities = _panel(n_dates=2)
    eligible = np.ones(len(pred), dtype=bool)
    eligible[-5:] = False
    pred[-5:] = np.nan
    report = evaluate_predictions(pred, y, dates, securities, eligible=eligible)
    assert report.coverage == pytest.approx(1.0)
    assert report.coverage_layers["evaluated"] == pytest.approx(1.0)
    with pytest.raises(EvaluationContractError, match="weights"):
        evaluate_predictions(pred, y, dates, securities, weights=np.full(len(pred), np.nan))
    with pytest.raises(EvaluationContractError, match="eligible"):
        evaluate_predictions(pred, y, dates, securities, eligible=np.ones(len(pred)))


def test_grouped_metrics_require_row_alignment_and_pit_asof_evidence():
    pred, y, dates, securities = _panel(n_dates=2)
    labels = np.tile(["small"] * 5 + ["large"] * 5, 2)
    with pytest.raises(EvaluationContractError, match="requires"):
        evaluate_predictions(pred, y, dates, securities, extra={"cap_labels": labels})
    with pytest.raises(EvaluationContractError, match="row-aligned"):
        evaluate_predictions(
            pred, y, dates, securities,
            extra={"cap_labels": labels[:-1], "cap_label_asof_dates": dates[:-1]},
        )
    future_asof = dates.to_numpy(copy=True)
    future_asof[0] = dates[-1]
    with pytest.raises(EvaluationContractError, match="future-known"):
        evaluate_predictions(
            pred, y, dates, securities,
            extra={"cap_labels": labels, "cap_label_asof_dates": future_asof},
        )


def test_invalid_calendar_dates_fail_closed_and_undefined_is_not_zero():
    pred = np.arange(10, dtype=float)
    y = pred.copy()
    with pytest.raises(EvaluationContractError, match="calendar"):
        evaluate_predictions(pred, y, np.array(["bad"] * 10), np.arange(10))
    dates = np.repeat(pd.Timestamp("2024-01-02"), 10)
    report = evaluate_predictions(pred, y, dates)
    assert np.isnan(report.turnover)
    assert report.metric_values["turnover"].status == "unavailable"
    assert np.isnan(report.icir)
    assert report.metric_values["daily_rank_ic_ir"].status == "undefined"
