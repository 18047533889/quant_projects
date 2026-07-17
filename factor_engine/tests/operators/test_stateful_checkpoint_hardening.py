# -*- coding: utf-8 -*-
from __future__ import annotations

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


def test_ema_full_runtime_matches_segmented_with_missing_rows() -> None:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    x = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0])
    timestamps = pd.date_range("2024-01-01", periods=len(x), freq="D", tz="UTC")
    identity = {"dataset": "missing", "adjustment": "split"}
    first = execute_stateful_segment(
        "ts_ema", {"x": x[:4]}, timestamps=timestamps[:4], instrument="A",
        input_identity=identity, params={"span": 3}, starts_at_dataset_origin=True,
    )
    second = execute_stateful_segment(
        "ts_ema", {"x": x[4:]}, timestamps=timestamps[4:], instrument="A",
        input_identity=identity, params={"span": 3}, checkpoint=first.checkpoint,
        starts_at_dataset_origin=False,
    )
    segmented = np.concatenate([first.values, second.values])
    panel = pd.DataFrame({"A": x}, index=timestamps)
    full = OperatorRegistry.get("ts_ema", backend="pandas_numpy").calculate(panel, 3)
    np.testing.assert_allclose(
        segmented, full["A"].to_numpy(), equal_nan=True, rtol=1e-12, atol=1e-12
    )
