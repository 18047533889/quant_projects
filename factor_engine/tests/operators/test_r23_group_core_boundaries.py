import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.common.group_polars import (
    GroupCountPolars,
    GroupMaxPolars,
    GroupMeanPolars,
    GroupMinPolars,
    GroupNormalizePolars,
    GroupRankPolars,
    GroupStdPolars,
    GroupSumPolars,
    GroupZscorePolars,
)


def _values(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(["A", "B", "C", "D", "E", "F"]).to_numpy()[0]


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        (GroupMeanPolars, [2.0, 2.0, np.nan, np.nan, 5.0, np.nan]),
        (GroupSumPolars, [4.0, 4.0, np.nan, np.nan, 5.0, np.nan]),
        (GroupMinPolars, [1.0, 1.0, np.nan, np.nan, 5.0, np.nan]),
        (GroupMaxPolars, [3.0, 3.0, np.nan, np.nan, 5.0, np.nan]),
        (GroupCountPolars, [2.0, 2.0, np.nan, np.nan, 1.0, np.nan]),
        (GroupNormalizePolars, [0.0, 1.0, np.nan, np.nan, 0.5, np.nan]),
        (GroupRankPolars, [0.5, 1.0, np.nan, np.nan, 1.0, np.nan]),
        (GroupStdPolars, [np.sqrt(2.0), np.sqrt(2.0), np.nan, np.nan, 0.0, np.nan]),
        (GroupZscorePolars, [-1 / np.sqrt(2.0), 1 / np.sqrt(2.0), np.nan, np.nan, 0.0, np.nan]),
    ],
)
def test_group_core_accepts_string_labels_and_excludes_nonfinite(operator, expected):
    x = pl.DataFrame({
        "A": [1.0], "B": [3.0], "C": [np.inf],
        "D": [np.nan], "E": [5.0], "F": [7.0],
    })
    group = pl.DataFrame({
        "A": ["tech"], "B": ["tech"], "C": ["tech"],
        "D": ["tech"], "E": ["bank"], "F": [""],
    })

    actual = _values(operator()._calculate_series(x, group))

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_group_core_fallback_policy_is_live_and_nonfinite_safe():
    x = pl.DataFrame({"A": [1.0], "B": [3.0], "C": [np.inf]})
    missing = pl.DataFrame({"A": [None], "B": [None], "C": [None]})

    assert np.isnan(_values3(GroupMeanPolars()._calculate_series(x, missing))).all()
    np.testing.assert_allclose(
        _values3(GroupMeanPolars()._calculate_series(x, missing, fallback_policy="global")),
        [2.0, 2.0, np.nan],
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="group labels are missing"):
        GroupMeanPolars()._calculate_series(x, missing, fallback_policy="error")


def _values3(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(["A", "B", "C"]).to_numpy()[0]


def test_group_core_rejects_column_identity_mismatch():
    x = pl.DataFrame({"A": [1.0], "B": [2.0]})
    wrong_group = pl.DataFrame({"B": ["g"], "A": ["g"]})

    with pytest.raises(ValueError, match="exactly match"):
        GroupMeanPolars()._calculate_series(x, wrong_group)


def test_group_core_rejects_unknown_fallback_even_with_present_groups():
    x = pl.DataFrame({"A": [1.0], "B": [2.0]})
    group = pl.DataFrame({"A": ["g"], "B": ["g"]})

    with pytest.raises(ValueError, match="invalid fallback_policy"):
        GroupMeanPolars()._calculate_series(x, group, fallback_policy="typo")


@pytest.mark.parametrize("operator", [
    GroupMeanPolars, GroupSumPolars, GroupMinPolars, GroupMaxPolars,
    GroupCountPolars, GroupNormalizePolars, GroupRankPolars, GroupStdPolars,
    GroupZscorePolars,
])
def test_group_core_integer_input_keeps_missing_output_as_float_nan(operator):
    x = pl.DataFrame({"A": [1], "B": [3], "C": [7]})
    group = pl.DataFrame({"A": ["g"], "B": ["g"], "C": [""]})

    actual = operator().calculate(x, group).select(["A", "B", "C"])

    assert all(dtype in (pl.Float32, pl.Float64) for dtype in actual.dtypes)
    assert np.isnan(actual["C"][0])


def test_group_core_rejects_mismatched_date_identity_before_pairing_rows():
    x = pl.DataFrame({
        "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "A": [1.0, 2.0], "B": [3.0, 4.0],
    })
    group = pl.DataFrame({
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")],
        "A": ["x", "y"], "B": ["x", "y"],
    })

    with pytest.raises(ValueError, match="different PanelIdentity|metadata axes must exactly match"):
        GroupMeanPolars().calculate(x, group)


@pytest.mark.parametrize("name", [
    "group_mean", "group_sum", "group_min", "group_max", "group_count",
    "group_std", "group_zscore", "group_rank", "group_normalize",
])
def test_registry_entry_executes_string_groups_with_pandas_parity(name):
    load_all()
    columns = ["A", "B", "C", "D", "E"]
    x_pd = pd.DataFrame([[1.0, 3.0, np.nan, 5.0, 5.0]], columns=columns)
    g_pd = pd.DataFrame([["tech", "tech", "tech", "bank", "bank"]], columns=columns)
    x_pl = pl.DataFrame({column: x_pd[column].to_numpy() for column in columns})
    g_pl = pl.DataFrame({column: g_pd[column].to_numpy() for column in columns})

    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    expected_module = (
        "common.polars_daily_native" if name == "group_normalize"
        else "common.group_polars"
    )
    assert type(polars_op).__module__.endswith(expected_module)
    expected = pandas_op.calculate(x_pd, g_pd).to_numpy()
    actual = polars_op.calculate(x_pl, g_pl).select(columns).to_numpy()

    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("name", [
    "group_mean", "group_sum", "group_min", "group_max", "group_count",
    "group_std", "group_zscore", "group_rank", "group_normalize",
])
def test_registry_fallback_modes_and_nonfinite_match_across_backends(name):
    load_all()
    columns = ["A", "B", "C", "D"]
    x_pd = pd.DataFrame([[1.0, 3.0, np.inf, np.nan]], columns=columns)
    g_pd = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], columns=columns)
    x_pl = pl.DataFrame({column: x_pd[column].to_numpy() for column in columns})
    g_pl = pl.DataFrame({column: g_pd[column].to_numpy() for column in columns})
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")

    for policy in ("nan", "global", "keep_original"):
        expected = pandas_op.calculate(x_pd, g_pd, fallback_policy=policy).to_numpy()
        actual = polars_op.calculate(x_pl, g_pl, fallback_policy=policy).select(columns).to_numpy()
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    for operator, x, group in ((pandas_op, x_pd, g_pd), (polars_op, x_pl, g_pl)):
        with pytest.raises(ValueError, match="group labels are missing"):
            operator.calculate(x, group, fallback_policy="error")


def test_registry_group_ops_are_row_local_with_dynamic_membership():
    load_all()
    names = [
        "group_mean", "group_sum", "group_min", "group_max", "group_count",
        "group_std", "group_zscore", "group_rank", "group_normalize",
    ]
    columns = ["A", "B", "C", "D"]
    x_prefix = pd.DataFrame([[1.0, 2.0, 10.0, 20.0]], columns=columns)
    g_prefix = pd.DataFrame([["x", "x", "y", "y"]], columns=columns)
    x_full = pd.concat([x_prefix, pd.DataFrame([[100.0, 4.0, 7.0, 8.0]], columns=columns)], ignore_index=True)
    g_full = pd.concat([g_prefix, pd.DataFrame([["y", "x", "x", "y"]], columns=columns)], ignore_index=True)

    for name in names:
        for backend in ("pandas_numpy", "polars"):
            operator = OperatorRegistry.get(name, backend=backend)
            if backend == "polars":
                prefix = operator.calculate(pl.from_pandas(x_prefix), pl.from_pandas(g_prefix)).to_numpy()[0]
                full = operator.calculate(pl.from_pandas(x_full), pl.from_pandas(g_full)).to_numpy()[0]
            else:
                prefix = operator.calculate(x_prefix, g_prefix).to_numpy()[0]
                full = operator.calculate(x_full, g_full).to_numpy()[0]
            np.testing.assert_allclose(prefix, full, equal_nan=True)
