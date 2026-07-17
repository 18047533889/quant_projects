# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stateful_runtime import execute_stateful_segment


def _assert_segmented(canonical, inputs, params, split=40):
    size = len(next(iter(inputs.values())))
    timestamps = pd.date_range("2024-01-01", periods=size, freq="D", tz="UTC")
    identity = {"dataset": "synthetic", "frequency": "1d", "adjustment": "test"}
    full = execute_stateful_segment(
        canonical, inputs, timestamps=timestamps, instrument="A",
        input_identity=identity, params=params, starts_at_dataset_origin=True,
    )
    first_inputs = {key: value[:split] for key, value in inputs.items()}
    second_inputs = {key: value[split:] for key, value in inputs.items()}
    first = execute_stateful_segment(
        canonical, first_inputs, timestamps=timestamps[:split], instrument="A",
        input_identity=identity, params=params, starts_at_dataset_origin=True,
    )
    second = execute_stateful_segment(
        canonical, second_inputs, timestamps=timestamps[split:], instrument="A",
        input_identity=identity, params=params, checkpoint=first.checkpoint,
        starts_at_dataset_origin=False,
    )
    combined = np.concatenate([first.values, second.values])
    np.testing.assert_allclose(combined, full.values, equal_nan=True, rtol=1e-12, atol=1e-12)
    assert second.checkpoint.as_of == timestamps[-1].isoformat()


def test_ema_checkpoint_is_numerically_incremental():
    x = np.linspace(10.0, 30.0, 80) + np.sin(np.arange(80))
    _assert_segmented("ts_ema", {"x": x}, {"span": 10})


def test_wilder_and_macd_checkpoint_families():
    close = 100 + np.cumsum(np.sin(np.arange(90) / 4.0) + 0.2)
    high = close + 1.0 + 0.1 * np.cos(np.arange(90))
    low = close - 1.0 - 0.1 * np.sin(np.arange(90))
    _assert_segmented("RSI_WILDER", {"x": close}, {"window": 14})
    _assert_segmented("ATR_WILDER", {"high": high, "low": low, "close": close}, {"window": 14})
    _assert_segmented("ADX", {"high": high, "low": low, "close": close}, {"window": 14}, split=50)
    for canonical in ("MACD_line", "MACD_signal", "MACD_hist"):
        _assert_segmented(canonical, {"x": close}, {"fast": 12, "slow": 26, "signal": 9})


def test_checkpoint_identity_and_segment_order_are_enforced():
    x = np.arange(10, dtype=float)
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    first = execute_stateful_segment(
        "ts_ema", {"x": x[:5]}, timestamps=timestamps[:5], instrument="A",
        input_identity={"dataset": "x"}, params={"span": 3},
        starts_at_dataset_origin=True,
    )
    with pytest.raises(Exception, match="fingerprint"):
        execute_stateful_segment(
            "ts_ema", {"x": x[5:]}, timestamps=timestamps[5:], instrument="A",
            input_identity={"dataset": "different"}, params={"span": 3},
            checkpoint=first.checkpoint, starts_at_dataset_origin=False,
        )
    with pytest.raises(ValueError, match="strictly after"):
        execute_stateful_segment(
            "ts_ema", {"x": x[4:6]}, timestamps=timestamps[4:6], instrument="A",
            input_identity={"dataset": "x"}, params={"span": 3},
            checkpoint=first.checkpoint, starts_at_dataset_origin=False,
        )
