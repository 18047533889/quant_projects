from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stateful_runtime import execute_stateful_segment


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
    assert json.loads(payload)["state"]["last_ema"] is None


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
