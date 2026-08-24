from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from factor_engine.stateful_runtime import execute_stateful_segment


@pytest.mark.parametrize("canonical", ["ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr"])
def test_ewm_checkpoint_matches_full_pandas_history(canonical: str) -> None:
    timestamps = pd.date_range("2024-01-01", periods=7, tz="UTC")
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 4.0, 7.0])
    y = np.array([2.0, 1.0, 3.0, 8.0, 7.0, np.nan, 9.0])
    inputs = {"x": x} if canonical in {"ts_ewm_std", "ts_ewm_var"} else {"x": x, "y": y}
    common = {"instrument": "A", "input_identity": {"dataset": "unit"}, "params": {"span": 3}}
    first = execute_stateful_segment(
        canonical, {key: values[:3] for key, values in inputs.items()},
        timestamps=timestamps[:3], starts_at_dataset_origin=True, **common,
    )
    second = execute_stateful_segment(
        canonical, {key: values[3:] for key, values in inputs.items()},
        timestamps=timestamps[3:], checkpoint=first.checkpoint, **common,
    )
    sx, sy = pd.Series(x), pd.Series(y)
    expected = {
        "ts_ewm_std": sx.ewm(span=3, adjust=False).std(),
        "ts_ewm_var": sx.ewm(span=3, adjust=False).var(),
        "ts_ewm_cov": sx.ewm(span=3, adjust=False).cov(sy),
        "ts_ewm_corr": sx.ewm(span=3, adjust=False).corr(sy),
    }[canonical].to_numpy()
    np.testing.assert_allclose(
        np.concatenate([first.values, second.values]), expected, equal_nan=True
    )


def test_checkpoint_binds_parameters_inputs_and_adjustment() -> None:
    x = np.arange(10, dtype=float)
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    first = execute_stateful_segment(
        "ts_ema", {"x": x[:5]}, timestamps=timestamps[:5], instrument="A",
        input_identity={"dataset": "x", "adjustment": "split"}, params={"span": 3},
        starts_at_dataset_origin=True,
    )
    cases = (
        ({"dataset": "different", "adjustment": "split"}, {"span": 3}, {"x": x[5:]}),
        ({"dataset": "x", "adjustment": "total_return"}, {"span": 3}, {"x": x[5:]}),
        ({"dataset": "x", "adjustment": "split"}, {"span": 4}, {"x": x[5:]}),
        ({"dataset": "x", "adjustment": "split"}, {"span": 3}, {"close": x[5:]}),
    )
    for identity, params, inputs in cases:
        with pytest.raises(Exception, match="fingerprint"):
            execute_stateful_segment(
                "ts_ema", inputs, timestamps=timestamps[5:], instrument="A",
                input_identity=identity, params=params, checkpoint=first.checkpoint,
                starts_at_dataset_origin=False,
            )


def test_checkpoint_json_is_strict_standard_json() -> None:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    result = execute_stateful_segment(
        "ts_ema", {"x": [np.nan, np.nan]}, timestamps=timestamps, instrument="A",
        input_identity={"dataset": "x"}, params={"span": 3}, starts_at_dataset_origin=True,
    )
    payload = result.checkpoint.to_json()
    assert "NaN" not in payload and "Infinity" not in payload
    # R5-09: the EMA state is the (weighted_avg, old_wt, valid_count) tuple; an
    # all-NaN segment never seeds the average, so weighted_avg is null.
    assert json.loads(payload)["state"]["ema"]["weighted_avg"] is None


def test_rejects_duplicate_and_out_of_order_timestamps() -> None:
    for timestamps in (
        ["2024-01-01", "2024-01-01"],
        ["2024-01-02", "2024-01-01"],
    ):
        with pytest.raises(ValueError, match="unique and monotonic"):
            execute_stateful_segment(
                "ts_ema", {"x": [1.0, 2.0]}, timestamps=timestamps, instrument="A",
                input_identity={"dataset": "x"}, params={"span": 3}, starts_at_dataset_origin=True,
            )
