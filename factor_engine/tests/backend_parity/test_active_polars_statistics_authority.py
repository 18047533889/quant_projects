"""Selected-bootstrap Polars rolling-statistics behavioral regressions.

These tests verify the implementation selected for the ``polars`` registry backend and
its numerical contract. They do not certify native-expression execution, absence of a
pandas delegate, or production eligibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _bootstrap_registry() -> None:
    load_all()


def _frame(values: list[float]) -> pl.DataFrame:
    return pl.from_pandas(pd.DataFrame({"A": values}))


def _selected(name: str, *, module: str, class_name: str):
    operator = OperatorRegistry.get(name, backend="polars")
    assert operator is not None
    assert type(operator).__module__ == module
    assert type(operator).__name__ == class_name
    return operator


def _assert_float_contract(
    actual: pd.Series, expected: pd.Series | np.ndarray, *, atol: float, rtol: float
) -> None:
    actual_values = actual.to_numpy(dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
    np.testing.assert_array_equal(np.isposinf(actual_values), np.isposinf(expected_values))
    np.testing.assert_array_equal(np.isneginf(actual_values), np.isneginf(expected_values))
    finite = np.isfinite(expected_values)
    np.testing.assert_allclose(
        actual_values[finite], expected_values[finite], atol=atol, rtol=rtol
    )


def test_selected_moment_is_window_local_and_requires_full_window() -> None:
    values = [1.0, 2.0, 10.0, 4.0, np.nan, 7.0]
    operator = _selected(
        "ts_moment",
        module="factor_engine.cleaned_operators.research_polars",
        class_name="TSMomentNativePolars",
    )
    out = operator.calculate(_frame(values), d=4, k=2).to_pandas()["A"]
    expected = pd.Series(values).rolling(4, min_periods=4).apply(
        lambda window: float(((window - window.mean()) ** 2).mean()), raw=True
    )
    _assert_float_contract(out, expected, atol=1e-12, rtol=1e-12)


def test_selected_kurt_and_skew_match_pandas_authority() -> None:
    values = [1.0, 2.0, 10.0, 4.0, np.nan, 7.0, np.inf, 8.0]
    frame = _frame(values)
    kurt_operator = _selected(
        "ts_kurt",
        module="factor_engine.cleaned_operators.common.time_series",
        class_name="TSKurtosisPolars",
    )
    skew_operator = _selected(
        "ts_skew",
        module="factor_engine.cleaned_operators.common.time_series",
        class_name="TSSkewnessPolars",
    )
    out_kurt = kurt_operator.calculate(frame, d=4).to_pandas()["A"]
    out_skew = skew_operator.calculate(frame, d=4).to_pandas()["A"]
    expected_kurt = pd.Series(values).rolling(4, min_periods=1).kurt()
    expected_skew = pd.Series(values).rolling(4, min_periods=1).skew()
    _assert_float_contract(out_kurt, expected_kurt, atol=1e-12, rtol=1e-12)
    _assert_float_contract(out_skew, expected_skew, atol=1e-12, rtol=1e-12)


def test_selected_conditional_covariance_matches_pairwise_oracle() -> None:
    x = [1.0, 2.0, 10.0, 4.0, np.nan, 7.0, np.inf, 8.0]
    y = [2.0, 1.0, 8.0, 3.0, 5.0, np.nan, 4.0, 9.0]
    condition = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0]
    operator = _selected(
        "ts_cov_if",
        module="factor_engine.cleaned_operators.weighted_moment_ext",
        class_name="_PolarsOp",
    )
    out = operator.calculate(
        _frame(x), _frame(y), _frame(condition), window=4, min_periods=2
    ).to_pandas()["A"]
    expected = []
    for row in range(len(x)):
        start = max(0, row - 3)
        xa = np.asarray(x[start : row + 1], dtype=float)
        ya = np.asarray(y[start : row + 1], dtype=float)
        ca = np.asarray(condition[start : row + 1], dtype=float)
        mask = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(ca) & (ca == 1.0)
        expected.append(
            float(np.cov(xa[mask], ya[mask], ddof=1)[0, 1])
            if mask.sum() >= 2
            else np.nan
        )
    _assert_float_contract(out, np.asarray(expected), atol=1e-12, rtol=1e-12)
