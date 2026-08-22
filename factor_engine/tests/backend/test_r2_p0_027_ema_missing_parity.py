"""R2-P0-027 EMA/Wilder missing-value parity regressions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.technical.polars_indicators_v2 import _ema, _wilder

load_all()


def _recursive_oracle(values: list[float | None], alpha: float, minimum: int) -> np.ndarray:
    """Independent adjust=False EWM oracle with pandas ignore_na=False gaps."""
    out: list[float] = []
    state = math.nan
    observed = 0
    missing = 0
    for value in values:
        if value is None or not np.isfinite(value):
            missing += 1
            out.append(state if observed >= minimum else math.nan)
            continue
        observed += 1
        if not np.isfinite(state):
            state = float(value)
        else:
            effective_alpha = alpha / (alpha + (1.0 - alpha) ** (missing + 1))
            state = effective_alpha * float(value) + (1.0 - effective_alpha) * state
        missing = 0
        out.append(state if observed >= minimum else math.nan)
    return np.asarray(out)


def _polars_eval(values: list[float | None], expression: pl.Expr) -> np.ndarray:
    return pl.DataFrame({"x": values}).select(expression.alias("x"))["x"].to_numpy()


def test_ema_and_wilder_match_independent_gap_oracle() -> None:
    values: list[float | None] = [None, 1.0, 2.0, 3.0, None, 5.0, None, None, 8.0]
    expected_ema = _recursive_oracle(values, alpha=0.5, minimum=3)
    expected_wilder = _recursive_oracle(values, alpha=1.0 / 3.0, minimum=3)
    np.testing.assert_allclose(_polars_eval(values, _ema(pl.col("x"), 3)), expected_ema, equal_nan=True)
    np.testing.assert_allclose(_polars_eval(values, _wilder(pl.col("x"), 3)), expected_wilder, equal_nan=True)


def test_ema_and_wilder_match_selected_pandas_authority_with_nan_gaps() -> None:
    values = np.asarray([1.0, 2.0, 3.0, np.nan, 5.0, np.nan, np.nan, 8.0])
    pandas_ema = pd.Series(values).ewm(span=3, adjust=False, min_periods=3).mean().to_numpy()
    pandas_wilder = pd.Series(values).ewm(alpha=1.0 / 3.0, adjust=False, min_periods=3).mean().to_numpy()
    np.testing.assert_allclose(_polars_eval(values.tolist(), _ema(pl.col("x"), 3)), pandas_ema, equal_nan=True)
    np.testing.assert_allclose(_polars_eval(values.tolist(), _wilder(pl.col("x"), 3)), pandas_wilder, equal_nan=True)


def test_nested_dema_future_invariance_matches_pandas() -> None:
    pandas_op = OperatorRegistry.get("DEMA", backend="pandas_numpy")
    polars_op = OperatorRegistry.get("DEMA", backend="polars")
    assert pandas_op is not None
    assert polars_op is not None
    values = np.asarray([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    kwargs = {"window": 3}
    first = polars_op.calculate(pl.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    extended = polars_op.calculate(pl.DataFrame({"x": np.r_[values, 1000.0]}), **kwargs)["x"].to_numpy()
    expected = pandas_op.calculate(pd.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    np.testing.assert_allclose(first, expected, equal_nan=True)
    np.testing.assert_array_equal(first, extended[: values.size])
