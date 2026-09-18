"""Independent extreme-magnitude oracles for core group operators."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from tests.backend_parity.test_three_backend_parity import _result_series, _run
from tests.helpers import InMemorySeriesSource


CASES = {
    "repeated": np.array([1e308, 1e308]),
    "symmetric": np.array([1e308, -1e308]),
    "tiny_repeated": np.array([1e-308, 1e-308]),
    "tiny_symmetric": np.array([1e-308, -1e-308]),
}
EXPECTED = {
    ("repeated", "group_neutralize"): np.array([0.0, 0.0]),
    ("repeated", "group_mean"): np.array([1e308, 1e308]),
    ("repeated", "group_zscore"): np.array([0.0, 0.0]),
    ("repeated", "group_normalize"): np.array([0.5, 0.5]),
    ("symmetric", "group_neutralize"): np.array([1e308, -1e308]),
    ("symmetric", "group_mean"): np.array([0.0, 0.0]),
    ("symmetric", "group_zscore"): np.array([
        1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
    ]),
    ("symmetric", "group_normalize"): np.array([1.0, 0.0]),
    ("tiny_repeated", "group_neutralize"): np.array([0.0, 0.0]),
    ("tiny_repeated", "group_mean"): np.array([1e-308, 1e-308]),
    ("tiny_repeated", "group_zscore"): np.array([0.0, 0.0]),
    ("tiny_repeated", "group_normalize"): np.array([0.5, 0.5]),
    ("tiny_symmetric", "group_neutralize"): np.array([1e-308, -1e-308]),
    ("tiny_symmetric", "group_mean"): np.array([0.0, 0.0]),
    ("tiny_symmetric", "group_zscore"): np.array([
        1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
    ]),
    ("tiny_symmetric", "group_normalize"): np.array([1.0, 0.0]),
}


def _wide_inputs(values):
    columns = ["A", "B"]
    x_pd = pd.DataFrame([values], columns=columns)
    g_pd = pd.DataFrame([[1.0, 1.0]], columns=columns)
    x_pl = pl.DataFrame({name: [value] for name, value in zip(columns, values)})
    g_pl = pl.DataFrame({name: [1.0] for name in columns})
    return x_pd, g_pd, x_pl, g_pl


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    "canonical",
    ["group_neutralize", "group_mean", "group_zscore", "group_normalize"],
)
@pytest.mark.parametrize("path", ["pandas_registry", "polars_registry", "polars_long"])
def test_group_extreme_magnitude_matches_independent_oracle(case_name, canonical, path):
    load_all()
    values = CASES[case_name]
    expected = EXPECTED[(case_name, canonical)]

    if path in {"pandas_registry", "polars_registry"}:
        x_pd, g_pd, x_pl, g_pl = _wide_inputs(values)
        if path == "pandas_registry":
            operator = OperatorRegistry.get(canonical, backend="pandas_numpy")
            actual = operator.calculate(x_pd, g_pd).to_numpy()[0]
        else:
            operator = OperatorRegistry.get(canonical, backend="polars")
            actual = operator.calculate(x_pl, g_pl).select(["A", "B"]).to_numpy()[0]
    else:
        timestamp = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product(
            [[timestamp], ["A", "B"]], names=["timestamp", "instrument"]
        )
        source = InMemorySeriesSource(data={
            "x": pd.Series(values, index=index),
            "group": pd.Series([1.0, 1.0], index=index),
        })
        expr = make_cleaned_call_factory(canonical)(col("x"), col("group"))
        run = _run(source, expr, "polars_long")
        assert run.get("used_polars_long_path") is True
        assert not run.get("polars_long_fallback_reason")
        actual = _result_series(run).to_numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0, equal_nan=True)
