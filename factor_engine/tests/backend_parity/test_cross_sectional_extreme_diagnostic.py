"""Independent finite-extreme oracles for remaining cross-sectional families."""
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
    "repeated": [1e308, 1e308],
    "symmetric": [1e308, -1e308],
    "tiny_repeated": [1e-308, 1e-308],
    "tiny_symmetric": [1e-308, -1e-308],
}
EXPECTED = {
    ("repeated", "group_std"): [0.0, 0.0],
    ("symmetric", "group_std"): [np.sqrt(2.0) * 1e308] * 2,
    ("repeated", "scale"): [0.5, 0.5],
    ("symmetric", "scale"): [0.5, -0.5],
    ("repeated", "rank"): [0.5, 0.5],
    ("symmetric", "rank"): [1.0, 0.0],
    ("repeated", "group_rank_weighted_value"): [5e307, 5e307],
    ("symmetric", "group_rank_weighted_value"): [(2.0 / 3.0) * 1e308, -(1.0 / 3.0) * 1e308],
    ("tiny_repeated", "group_std"): [0.0, 0.0],
    ("tiny_symmetric", "group_std"): [np.sqrt(2.0) * 1e-308] * 2,
    ("tiny_repeated", "scale"): [0.5, 0.5],
    ("tiny_symmetric", "scale"): [0.5, -0.5],
    ("tiny_repeated", "rank"): [0.5, 0.5],
    ("tiny_symmetric", "rank"): [1.0, 0.0],
    ("tiny_repeated", "group_rank_weighted_value"): [0.5e-308, 0.5e-308],
    ("tiny_symmetric", "group_rank_weighted_value"): [(2.0 / 3.0) * 1e-308, -(1.0 / 3.0) * 1e-308],
}

@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("canonical", ["group_std", "scale", "rank", "group_rank_weighted_value"])
@pytest.mark.parametrize("path", ["pandas_registry", "polars_registry", "polars_long"])
def test_cross_sectional_extreme_oracle(case_name, canonical, path):
    load_all()
    values = CASES[case_name]
    expected = EXPECTED[(case_name, canonical)]
    grouped = canonical.startswith("group_")
    if path != "polars_long":
        columns = ["A", "B"]
        if path == "pandas_registry":
            x = pd.DataFrame([values], columns=columns)
            group = pd.DataFrame([[1.0, 1.0]], columns=columns)
            op = OperatorRegistry.get(canonical, backend="pandas_numpy")
        else:
            x = pl.DataFrame({"A": [values[0]], "B": [values[1]]})
            group = pl.DataFrame({"A": [1.0], "B": [1.0]})
            op = OperatorRegistry.get(canonical, backend="polars")
        actual = op.calculate(x, group).to_numpy()[0] if grouped else op.calculate(x).to_numpy()[0]
    else:
        timestamp = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product([[timestamp], ["A", "B"]], names=["timestamp", "instrument"])
        source = InMemorySeriesSource(data={
            "x": pd.Series(values, index=index),
            "group": pd.Series([1.0, 1.0], index=index),
        })
        factory = make_cleaned_call_factory(canonical)
        expr = factory(col("x"), col("group")) if grouped else factory(col("x"))
        run = _run(source, expr, "polars_long")
        assert run.get("used_polars_long_path") is True
        actual = _result_series(run).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("path", ["pandas_registry", "polars_registry", "polars_long"])
def test_group_std_unrepresentable_is_not_clipped(path):
    load_all()
    values = [1.7e308, -1.7e308]
    if path != "polars_long":
        columns = ["A", "B"]
        if path == "pandas_registry":
            x = pd.DataFrame([values], columns=columns)
            group = pd.DataFrame([[1.0, 1.0]], columns=columns)
            op = OperatorRegistry.get("group_std", backend="pandas_numpy")
        else:
            x = pl.DataFrame({"A": [values[0]], "B": [values[1]]})
            group = pl.DataFrame({"A": [1.0], "B": [1.0]})
            op = OperatorRegistry.get("group_std", backend="polars")
        actual = op.calculate(x, group).to_numpy()[0]
    else:
        timestamp = pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_product(
            [[timestamp], ["A", "B"]], names=["timestamp", "instrument"]
        )
        source = InMemorySeriesSource(data={
            "x": pd.Series(values, index=index),
            "group": pd.Series([1.0, 1.0], index=index),
        })
        factory = make_cleaned_call_factory("group_std")
        run = _run(source, factory(col("x"), col("group")), "polars_long")
        assert run.get("used_polars_long_path") is True
        actual = _result_series(run).to_numpy()
    assert np.all(~np.isfinite(actual))
