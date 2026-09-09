# -*- coding: utf-8 -*-
"""Behavioral tests for the nine daily event-response operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.event_response  # noqa: F401
from factor_engine.cleaned_operators.registry import OperatorRegistry

CANONICALS = (
    "event_historical_response_mean",
    "event_historical_response_sign_balance",
    "event_hawkes_branching_ratio_proxy",
    "event_response_peak_lag",
    "event_response_decay_rate",
    "event_response_dispersion",
    "event_response_reversal_strength",
    "event_response_effective_events",
    "event_response_overlap_ratio",
)


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    response = np.zeros(40, dtype=float)
    event = np.zeros(40, dtype=float)
    event[[5, 12, 20, 28]] = 1.0
    response[6:9] = [1.0, 2.0, 3.0]
    response[13:16] = [2.0, 4.0, 6.0]
    response[21:24] = [-1.0, -2.0, -3.0]
    response[29:32] = [3.0, 6.0, 9.0]
    index = pd.date_range("2024-01-01", periods=40)
    return (
        pd.DataFrame({"A": response}, index=index),
        pd.DataFrame({"A": event}, index=index),
    )


def _history_params() -> dict[str, int]:
    return {"history_window": 30, "horizon": 3, "min_events": 2}


def _calculate(
    name: str, response: pd.DataFrame, event: pd.DataFrame
) -> pd.DataFrame:
    """Calculate one operator with a valid, non-vacuous family-specific call."""
    if name == "event_hawkes_branching_ratio_proxy":
        return _op(name).calculate(event, window=30, max_lag=3, min_events=2)
    if name in {"event_response_effective_events", "event_response_overlap_ratio"}:
        return _op(name).calculate(
            event, history_window=30, horizon=3, refractory=5
        )
    return _op(name).calculate(response, event, **_history_params())


def test_registered_metadata_and_declared_parameters() -> None:
    expected_inputs = {
        "event_historical_response_mean": ("response", "event"),
        "event_historical_response_sign_balance": ("response", "event"),
        "event_hawkes_branching_ratio_proxy": ("event",),
        "event_response_peak_lag": ("response", "event"),
        "event_response_decay_rate": ("response", "event"),
        "event_response_dispersion": ("response", "event"),
        "event_response_reversal_strength": ("response", "event"),
        "event_response_effective_events": ("event",),
        "event_response_overlap_ratio": ("event",),
    }
    for name in CANONICALS:
        metadata = _op(name).metadata
        assert metadata.name == name
        assert tuple(metadata.param_names[: len(expected_inputs[name])]) == expected_inputs[name]
        assert metadata.return_type == "series"


@pytest.mark.parametrize("name", CANONICALS)
def test_all_operators_are_deterministic_on_nonvacuous_inputs(name: str) -> None:
    response, event = _panel()
    first = _calculate(name, response, event)
    second = _calculate(name, response, event)
    assert np.isfinite(first.iloc[-1, 0])
    pd.testing.assert_frame_equal(first, second, check_exact=True)


@pytest.mark.parametrize("name", CANONICALS)
def test_all_operators_preserve_two_column_axes(name: str) -> None:
    response, event = _panel()
    response = pd.concat([response, response * 2.0], axis=1)
    event = pd.concat([event, event], axis=1)
    response.columns = event.columns = ["A", "B"]

    result = _calculate(name, response, event)
    assert result.index.equals(response.index)
    assert result.columns.equals(response.columns)
    assert np.isfinite(result.iloc[-1]).all()


def test_historical_mean_and_sign_balance_independent_oracle() -> None:
    response, event = _panel()
    params = _history_params()
    mean_sum = _op("event_historical_response_mean").calculate(
        response, event, mode="sum", **params
    )
    mean_mean = _op("event_historical_response_mean").calculate(
        response, event, mode="mean", **params
    )
    sign = _op("event_historical_response_sign_balance").calculate(
        response, event, **params
    )

    # At t=39 the cohort is [9, 36]: event totals are 12, -6, and 18.
    totals = np.asarray([12.0, -6.0, 18.0])
    assert mean_sum.iloc[-1, 0] == pytest.approx(float(totals.mean()))
    assert mean_mean.iloc[-1, 0] == pytest.approx(float((totals / 3.0).mean()))
    assert sign.iloc[-1, 0] == pytest.approx(float(np.sign(totals).mean()))


def test_curve_statistics_independent_oracle() -> None:
    response, event = _panel()
    params = _history_params()
    peak = _op("event_response_peak_lag").calculate(response, event, **params)
    decay = _op("event_response_decay_rate").calculate(response, event, **params)
    dispersion = _op("event_response_dispersion").calculate(response, event, **params)
    reversal = _op("event_response_reversal_strength").calculate(response, event, **params)

    curves = np.asarray([[2.0, 4.0, 6.0], [-1.0, -2.0, -3.0], [3.0, 6.0, 9.0]])
    mean_curve = curves.mean(axis=0)
    xs = np.arange(1.0, 4.0)
    logged = np.log(np.abs(mean_curve) + 1e-12)
    expected_decay = np.sum((xs - xs.mean()) * (logged - logged.mean())) / np.sum(
        (xs - xs.mean()) ** 2
    )
    totals = curves.sum(axis=1)

    assert peak.iloc[-1, 0] == 1.0
    assert decay.iloc[-1, 0] == pytest.approx(expected_decay)
    assert dispersion.iloc[-1, 0] == pytest.approx(
        np.std(totals, ddof=1) / np.mean(np.abs(totals))
    )
    assert reversal.iloc[-1, 0] == 0.0


def test_episode_diagnostics_and_hawkes_are_nonvacuous() -> None:
    _, event = _panel()
    effective = _op("event_response_effective_events").calculate(
        event, history_window=30, horizon=3, refractory=5
    )
    overlap = _op("event_response_overlap_ratio").calculate(
        event, history_window=30, horizon=3, refractory=5
    )
    hawkes = _op("event_hawkes_branching_ratio_proxy").calculate(
        event, window=30, max_lag=3, min_events=2
    )
    assert effective.iloc[-1, 0] == 3.0
    assert overlap.iloc[-1, 0] == 0.0
    assert np.isfinite(hawkes.iloc[-1, 0])


def test_clustered_episode_diagnostics_independent_counts() -> None:
    event = np.zeros(20)
    event[[4, 5, 6, 12]] = 1.0
    frame = pd.DataFrame({"A": event})
    effective = _op("event_response_effective_events").calculate(
        frame, history_window=15, horizon=2, refractory=3
    )
    overlap = _op("event_response_overlap_ratio").calculate(
        frame, history_window=15, horizon=2, refractory=3
    )
    assert effective.iloc[-1, 0] == 2.0
    assert overlap.iloc[-1, 0] == pytest.approx(1.0 - 2.0 / 4.0)


def test_missing_response_path_is_excluded_not_shortened() -> None:
    response, event = _panel()
    response.iloc[14, 0] = np.nan  # invalidates the entire event-12 path
    params = _history_params()
    mean = _op("event_historical_response_mean").calculate(
        response, event, mode="sum", **params
    )
    dispersion = _op("event_response_dispersion").calculate(
        response, event, **params
    )
    remaining = np.asarray([-6.0, 18.0])
    assert mean.iloc[-1, 0] == pytest.approx(float(remaining.mean()))
    assert dispersion.iloc[-1, 0] == pytest.approx(
        np.std(remaining, ddof=1) / np.mean(np.abs(remaining))
    )


def test_missing_event_coverage_has_distinct_response_and_diagnostic_policy() -> None:
    response, event = _panel()
    event.iloc[27, 0] = np.nan  # predecessor coverage for event 28 is unknown
    params = _history_params()
    mean = _op("event_historical_response_mean").calculate(
        response, event, mode="sum", refractory=3, **params
    )
    effective = _op("event_response_effective_events").calculate(
        event, history_window=30, horizon=3, refractory=3
    )
    overlap = _op("event_response_overlap_ratio").calculate(
        event, history_window=30, horizon=3, refractory=3
    )

    # Response estimation fail-closes by excluding only the uncertain anchor;
    # diagnostics describe the whole raw cohort and stay NaN when its event
    # coverage/denominator is unknown. These are intentionally different sample
    # policies, not a claim that both operators use an identical denominator.
    assert mean.iloc[-1, 0] == pytest.approx((12.0 - 6.0) / 2.0)
    assert np.isnan(effective.iloc[-1, 0])
    assert np.isnan(overlap.iloc[-1, 0])


@pytest.mark.parametrize(
    ("name", "arity"),
    [
        ("event_historical_response_mean", 2),
        ("event_historical_response_sign_balance", 2),
        ("event_hawkes_branching_ratio_proxy", 1),
        ("event_response_peak_lag", 2),
        ("event_response_decay_rate", 2),
        ("event_response_dispersion", 2),
        ("event_response_reversal_strength", 2),
        ("event_response_effective_events", 1),
        ("event_response_overlap_ratio", 1),
    ],
)
def test_empty_inputs_return_empty_frame(name: str, arity: int) -> None:
    empty = pd.DataFrame(index=pd.RangeIndex(0), columns=["A"], dtype=float)
    result = _op(name).calculate(*([empty] * arity))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == empty.shape
    assert result.empty


@pytest.mark.parametrize(
    "name",
    [
        "event_historical_response_mean",
        "event_historical_response_sign_balance",
        "event_response_peak_lag",
        "event_response_decay_rate",
        "event_response_dispersion",
        "event_response_reversal_strength",
    ],
)
def test_two_input_operators_preserve_single_column_axes(name: str) -> None:
    response, event = _panel()
    result = _op(name).calculate(response, event, **_history_params())
    assert result.index.equals(response.index)
    assert result.columns.equals(response.columns)
    assert np.isfinite(result.iloc[-1, 0])


def test_nonfinite_event_rows_are_missing_not_events() -> None:
    event = pd.DataFrame({"A": [0.0] * 12})
    event.iloc[[2, 8], 0] = [np.inf, -np.inf]
    event.iloc[5, 0] = 1.0
    effective = _op("event_response_effective_events").calculate(
        event, history_window=8, horizon=1, refractory=0
    )
    overlap = _op("event_response_overlap_ratio").calculate(
        event, history_window=8, horizon=1, refractory=0
    )
    # Inf is non-finite event coverage, so the current diagnostic cohort is
    # unknown rather than treating either value as an observed event or zero.
    assert np.isnan(effective.iloc[-1, 0])
    assert np.isnan(overlap.iloc[-1, 0])


@pytest.mark.parametrize("name", CANONICALS)
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_all_operator_nonfinite_input_roles(name, missing):
    response, event = _panel()
    # Preserve each original operator's NaN/Inf handling obligation, with valid
    # arity and an observed control column rather than only return-type checks.
    response["UNKNOWN"] = missing
    event["UNKNOWN"] = missing
    if name == "event_hawkes_branching_ratio_proxy":
        result = _op(name).calculate(event, window=30, max_lag=3, min_events=2)
    elif name in ("event_response_effective_events", "event_response_overlap_ratio"):
        result = _op(name).calculate(event, history_window=30, horizon=3)
    else:
        result = _op(name).calculate(response, event, **_history_params())
    assert result.index.equals(response.index)
    assert result.columns.equals(response.columns)
    assert np.isfinite(result["A"].iloc[-1])
    assert result["UNKNOWN"].isna().all()
    assert not np.isinf(result.to_numpy()).any()
