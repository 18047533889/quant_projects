from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.math_certificate import (
    check_chunk_boundary_invariance_all_streamable,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.recursive_kernel import adx_segment
from factor_engine.runtime.execution_contract import _minimum_warmup_rows
from factor_engine.stateful_contract import (
    StatefulCheckpointRegistry,
    StatefulContractError,
)


def test_h22_adx_actual_winner_declares_window_default_and_domain() -> None:
    load_all(include_research=True)
    winner = OperatorRegistry.get("ADX", "pandas_numpy", mode="production")
    spec = winner.metadata.param_specs["window"]
    assert spec.default == 14
    assert spec.min == 1
    with pytest.raises(Exception, match="window"):
        winner.calculate(
            pd.DataFrame([[2.0]]), pd.DataFrame([[1.0]]),
            pd.DataFrame([[1.5]]), window=0,
        )


@pytest.mark.parametrize("window,expected", [(1, 2), (5, 10), (14, 28), (20, 40)])
def test_h22_adx_minimum_history_is_bound_to_window(window: int, expected: int) -> None:
    state_spec = StatefulCheckpointRegistry.get("ADX")
    assert state_spec.minimum_history_for({"window": window}) == expected
    rows, unknown = _minimum_warmup_rows("ADX", {"window": window})
    assert rows >= expected
    assert not unknown


def test_h22_adx_gate_records_the_parameter_it_executes() -> None:
    ok, detail = check_chunk_boundary_invariance_all_streamable(
        n_bars=80, n_random_chunkings=1
    )
    row = detail["per_canonical"]["ADX"]
    assert ok, detail
    assert row["test_executed"]
    assert row["parameters"]["window"] == 14
    assert row["minimum_history"] == 28


@pytest.mark.parametrize("window", [0, -1, 1.5, float("nan"), True, "14"])
def test_h22_adx_minimum_history_rejects_invalid_window(window) -> None:
    with pytest.raises(StatefulContractError, match="positive integer"):
        StatefulCheckpointRegistry.get("ADX").minimum_history_for({"window": window})


def test_h22_adx_winner_and_segment_share_parameterized_maturity() -> None:
    load_all(include_research=True)
    rng = np.random.default_rng(7)
    close = 100 + rng.normal(size=70).cumsum()
    high, low = close + 0.5, close - 0.5
    winner = OperatorRegistry.get("ADX", "pandas_numpy", mode="production")
    for window in (5, 14, 20):
        frames = [pd.DataFrame(v, columns=["x"]) for v in (high, low, close)]
        expected = winner.calculate(*frames, window=window).iloc[:, 0].to_numpy()
        actual, _state = adx_segment(high, low, close, {}, window)
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        assert np.flatnonzero(np.isfinite(actual))[0] == 2 * window - 1


def test_h22_adx_old_semantic_checkpoint_is_rejected() -> None:
    state_spec = StatefulCheckpointRegistry.get("ADX")
    state = {}
    for name in state_spec.checkpoint_fields:
        if name in {"tr", "plus_dm", "minus_dm", "dx"}:
            state[name] = {"weighted_avg": 1.0, "old_wt": 1.0, "valid_count": 14}
        elif name == "last_timestamp":
            state[name] = "2024-01-01T00:00:00+00:00"
        else:
            state[name] = 1.0
    checkpoint = StatefulCheckpointRegistry.create_checkpoint(
        "ADX", instrument="X", as_of=state["last_timestamp"],
        state=state, input_identity={"source": "review"},
    )
    with pytest.raises(StatefulContractError, match="semantic version mismatch"):
        StatefulCheckpointRegistry.validate(replace(checkpoint, semantic_version="3.0"))
