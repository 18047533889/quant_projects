"""RSI missing-observation semantics on the actual long-frame emitter."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _ewm(values, window):
    """Independent adjust=False/ignore_na=False recurrence, including gaps."""
    alpha = 1.0 / window
    decay = 1.0 - alpha
    state, weight, count = np.nan, 1.0, 0
    out = []
    for value in values:
        observed = np.isfinite(value)
        count += int(observed)
        if np.isfinite(state):
            weight *= decay
            if observed:
                state = (weight * state + alpha * value) / (weight + alpha)
                weight = 1.0
        elif observed:
            state = value
        out.append(state if count >= window else np.nan)
    return np.asarray(out)


def _oracle(values, window):
    delta = np.r_[np.nan, np.diff(values)]
    gains = _ewm(np.maximum(delta, 0), window)
    losses = _ewm(np.maximum(-delta, 0), window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gains / losses)
    out[(losses == 0) & (gains > 0)] = 100
    out[(gains == 0) & (losses > 0)] = 0
    out[(gains == 0) & (losses == 0)] = 50
    return out


@pytest.mark.parametrize("backend", ["pandas", "polars_long"])
@pytest.mark.parametrize("window", [2, 3, 6])
def test_rsi_gaps_warmup_instrument_isolation_and_future_prefix(backend, window):
    dates = pd.date_range("2026-01-01", periods=24)
    t = np.arange(24, dtype=float)
    panel = pd.DataFrame({
        "mixed": 100 + t + 3 * np.sin(t),
        "flat": np.full(24, 13.0),
        "falling": 70 - t,
        "empty": np.full(24, np.nan),
    }, index=dates)
    panel.index.name = "timestamp"
    panel.columns.name = "instrument"
    panel.loc[dates[[0, 1, 11, 12, 22, 23]], "mixed"] = np.nan
    panel.loc[dates[[9, 10, 11]], "flat"] = np.nan
    panel.loc[dates[[0, 9, 19]], "falling"] = np.nan

    def execute(frame):
        source = InMemorySeriesSource(data={"close": frame.stack(future_stack=True)})
        result = FactorEngine(
            backend=build_backend(backend), data_source=source, run_mode="research",
        ).run(Factor(name="rsi_gap", expr=F("RSI_WILDER")(col("close"), window)))
        if backend == "polars_long":
            assert result.get("used_polars_long_path") is True
        return result["result"].unstack("instrument").reindex(
            index=frame.index, columns=frame.columns,
        )

    actual = execute(panel)
    expected = panel.apply(lambda values: _oracle(values.to_numpy(), window))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    changed = panel.copy()
    changed.iloc[18:, :3] = changed.iloc[18:, :3] * -2.3 + 17
    mutated = execute(changed)
    np.testing.assert_allclose(
        actual.iloc[:18], mutated.iloc[:18], rtol=1e-12, atol=1e-12, equal_nan=True,
    )
