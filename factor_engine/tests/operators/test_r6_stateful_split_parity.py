# -*- coding: utf-8 -*-
"""R6 system-level tests C & D — stateful full/incremental split parity and
checkpoint-revision rejection.

C.  For every segmented canonical and every split point ``s in 1..N-1``:
    ``full == prefix + restore + suffix`` (bit-exact where defined).
D.  A checkpoint whose input identity has changed (source data before the
    checkpoint revised) must be REJECTED — stale checkpoints are never reused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stateful_contract import StatefulCheckpointRegistry, StatefulContractError
from stateful_runtime import execute_stateful_segment

SEGMENTED = (
    "ts_ema",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_ewm_cov",
    "ts_ewm_corr",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
)


def _inputs_for(canonical: str, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    high = 10.0 + np.cumsum(rng.normal(0, 0.01, n))
    low = high - np.abs(rng.normal(0, 0.005, n)) - 0.001
    close = (high + low) / 2.0 + rng.normal(0, 0.001, n)
    if canonical in {"ts_ema", "RSI_WILDER"}:
        return {"x": high}
    if canonical in {"MACD_line", "MACD_signal", "MACD_hist"}:
        return {"x": close}
    if canonical in {"ts_ewm_std", "ts_ewm_var"}:
        return {"x": close}
    if canonical in {"ts_ewm_cov", "ts_ewm_corr"}:
        return {"x": close, "y": high}
    if canonical in {"ATR_WILDER", "ADX"}:
        return {"high": high, "low": low, "close": close}
    raise AssertionError(canonical)


def _params_for(canonical: str) -> dict[str, int]:
    if canonical == "ts_ema":
        return {"span": 20}
    if canonical.startswith("ts_ewm"):
        return {"span": 20}
    if canonical == "RSI_WILDER":
        return {"window": 14}
    if canonical in {"ATR_WILDER", "ADX"}:
        return {"window": 14}
    return {"fast": 12, "slow": 26, "signal": 9}


@pytest.mark.parametrize("canonical", SEGMENTED)
def test_r6_c_split_parity_all_splits(canonical: str):
    N = 48
    rng = np.random.default_rng(hash(canonical) % (2**32))
    idx = pd.date_range("2024-01-01", periods=N, freq="D", tz="UTC")
    ts = list(idx)
    inputs = _inputs_for(canonical, rng, N)
    ident = {"source_hash": "r6-parity", "instrument": "A"}
    params = _params_for(canonical)

    def run(chunk_slice, tslice, cp=None, origin=False):
        return execute_stateful_segment(
            canonical, inputs=chunk_slice, timestamps=tslice,
            instrument="A", input_identity=ident, params=params,
            checkpoint=cp, starts_at_dataset_origin=origin,
        )

    full = run(inputs, ts, origin=True)
    for s in range(1, N):
        pre_in = {k: v[:s] for k, v in inputs.items()}
        post_in = {k: v[s:] for k, v in inputs.items()}
        pre = run(pre_in, ts[:s], origin=True)
        post = run(post_in, ts[s:], cp=pre.checkpoint)
        cat = np.concatenate([pre.values, post.values])
        assert np.allclose(cat, full.values, equal_nan=True), (
            f"{canonical}: split {s} breaks full == prefix + restore + suffix"
        )


def test_r6_c_ema_split_parity_with_gaps():
    """Split parity must also hold with NaN gaps (state is carried, not reset)."""
    N = 40
    rng = np.random.default_rng(11)
    x = 10.0 + np.cumsum(rng.normal(0, 0.01, N))
    x[7:10] = np.nan  # interior gap
    x[22] = np.nan
    idx = pd.date_range("2024-01-01", periods=N, freq="D", tz="UTC")
    ts = list(idx)
    ident = {"source_hash": "r6-gap", "instrument": "A"}

    def run(chunk, tslice, cp=None, origin=False):
        return execute_stateful_segment(
            "ts_ema", inputs={"x": chunk}, timestamps=tslice,
            instrument="A", input_identity=ident, params={"span": 20},
            checkpoint=cp, starts_at_dataset_origin=origin,
        )

    full = run(x, ts, origin=True)
    for s in (6, 12, 20, 30):
        pre = run(x[:s], ts[:s], origin=True)
        post = run(x[s:], ts[s:], cp=pre.checkpoint)
        cat = np.concatenate([pre.values, post.values])
        assert np.allclose(cat, full.values, equal_nan=True), f"gap split {s}"


def test_r6_d_checkpoint_rejected_on_input_identity_change():
    N = 40
    rng = np.random.default_rng(3)
    x = 10.0 + np.cumsum(rng.normal(0, 0.01, N))
    idx = pd.date_range("2024-01-01", periods=N, freq="D", tz="UTC")
    ts = list(idx)

    original_ident = {"source_hash": "dataset@v1", "instrument": "A"}
    revised_ident = {"source_hash": "dataset@v2", "instrument": "A"}

    def run(ident, chunk, tslice, cp=None, origin=False):
        return execute_stateful_segment(
            "ts_ema", inputs={"x": chunk}, timestamps=tslice,
            instrument="A", input_identity=ident, params={"span": 20},
            checkpoint=cp, starts_at_dataset_origin=origin,
        )

    pre = run(original_ident, x[:25], ts[:25], origin=True)
    # Source data before the checkpoint was revised -> identity changes -> the
    # old checkpoint must be rejected, never silently reused.
    with pytest.raises(StatefulContractError):
        run(revised_ident, x[25:], ts[25:], cp=pre.checkpoint)

    # Same identity (no revision) still accepts the checkpoint.
    post = run(original_ident, x[25:], ts[25:], cp=pre.checkpoint)
    assert post.values[-1] == pytest.approx(
        execute_stateful_segment(
            "ts_ema", inputs={"x": x}, timestamps=ts,
            instrument="A", input_identity=original_ident, params={"span": 20},
            starts_at_dataset_origin=True,
        ).values[-1]
    )


def test_r6_d_checkpoint_rejected_on_instrument_mismatch():
    N = 30
    rng = np.random.default_rng(5)
    x = 10.0 + np.cumsum(rng.normal(0, 0.01, N))
    idx = pd.date_range("2024-01-01", periods=N, freq="D", tz="UTC")
    ts = list(idx)
    ident = {"source_hash": "v1", "instrument": "A"}
    pre = execute_stateful_segment(
        "ts_ema", inputs={"x": x[:20]}, timestamps=ts[:20],
        instrument="A", input_identity=ident, params={"span": 20},
        starts_at_dataset_origin=True,
    )
    with pytest.raises(StatefulContractError):
        execute_stateful_segment(
            "ts_ema", inputs={"x": x[20:]}, timestamps=ts[20:],
            instrument="B", input_identity=dict(ident, instrument="B"),
            params={"span": 20}, checkpoint=pre.checkpoint,
        )
