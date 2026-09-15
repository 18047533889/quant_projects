from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import execution_contract


TARGETS = ("MACD_line", "MACD_signal", "MACD_hist", "RSI_WILDER")


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    state = np.nan
    count = 0
    alpha = 1.0 / window
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        state = value if count == 0 else alpha * value + (1.0 - alpha) * state
        count += 1
        if count >= window:
            out[i] = state
    return out


def _rsi_oracle(values: np.ndarray, window: int) -> np.ndarray:
    delta = np.r_[np.nan, np.diff(values)]
    gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
    loss = np.where(np.isnan(delta), np.nan, np.maximum(-delta, 0.0))
    avg_gain = _wilder(gain, window)
    avg_loss = _wilder(loss, window)
    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    out[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return out


def test_macd_family_matches_independent_recursive_oracle() -> None:
    values = np.array([10.0, 11.5, 10.25, 13.0, 12.0, 14.5, 13.25, 15.0])
    panel = pd.DataFrame({"A": values})
    fast, slow, signal = 3, 5, 2
    line = _ema(values, fast) - _ema(values, slow)
    signal_line = _ema(line, signal)
    expected = {
        "MACD_line": line,
        "MACD_signal": signal_line,
        "MACD_hist": line - signal_line,
    }
    for canonical, oracle in expected.items():
        actual = OperatorRegistry.get(canonical).calculate(
            panel, fast=fast, slow=slow, signal=signal
        )["A"].to_numpy()
        np.testing.assert_allclose(actual, oracle, rtol=1e-13, atol=1e-13)


def test_rsi_wilder_matches_independent_recursive_oracle() -> None:
    values = np.array([10.0, 12.0, 11.0, 14.0, 13.0, 13.5, 12.5, 15.0])
    panel = pd.DataFrame({"A": values})
    actual = OperatorRegistry.get("RSI_WILDER").calculate(panel, window=3)
    np.testing.assert_allclose(
        actual["A"].to_numpy(), _rsi_oracle(values, 3), equal_nan=True,
        rtol=1e-13, atol=1e-13,
    )


def test_macd_relation_and_strict_integer_domains() -> None:
    panel = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    for canonical in ("MACD_line", "MACD_signal", "MACD_hist"):
        op = OperatorRegistry.get(canonical)
        with pytest.raises(ValueError, match="fast must be smaller than slow"):
            op.calculate(panel, fast=5, slow=5, signal=2)
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel, fast=True, slow=5, signal=2)
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get("RSI_WILDER").calculate(panel, window=True)


def test_recursive_history_is_not_a_finite_warmup_claim() -> None:
    for canonical in TARGETS:
        lookback = OperatorRegistry._catalog[canonical]["contract"]["lookback"]
        assert lookback["kind"] == "recursive_state"
        assert lookback["finite_warmup_is_approximation"] is True
        state = execution_contract(canonical)
        assert state.state_model == "recursive"
        assert state.chunking == "checkpoint"


def test_declared_contract_is_complete_on_runtime_backends() -> None:
    for canonical in TARGETS:
        for backend in ("pandas_numpy", "polars"):
            metadata = OperatorRegistry.get(canonical, backend=backend).metadata
            assert metadata.panel_params == ("x",)
            assert set(metadata.scalar_params) == (
                {"window"} if canonical == "RSI_WILDER" else {"fast", "slow", "signal"}
            )
            assert set(metadata.param_specs) == set(metadata.scalar_params)
