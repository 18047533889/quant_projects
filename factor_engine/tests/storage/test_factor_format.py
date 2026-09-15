"""因子长表/宽表 pivot API 测试。"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from factor_engine.storage.factor_format import (
    long_table_to_series,
    pivot_long_to_wide,
    pivot_multi_factor_long_to_wide,
    series_to_long_table,
    unpivot_wide_to_long,
)
from factor_engine.storage.result_store import PandasResultStore
from tests.storage.test_materializer import _setup_lake

pd = pytest.importorskip("pandas")


def _series() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)


def test_series_long_roundtrip():
    original = _series()
    long_df = series_to_long_table(original)
    restored = long_table_to_series(long_df)
    pd.testing.assert_series_equal(restored, original, check_names=False)


@pytest.mark.parametrize("asset_dtype", [object, "string"])
def test_long_table_to_series_preserves_asset_axis_dtype(asset_dtype):
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "asset": pd.Series(["A", "B", "A", "B", "A", "B"], dtype=asset_dtype),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    restored = long_table_to_series(frame)

    asset_level = restored.index.get_level_values("asset")
    assert asset_level.dtype == frame["asset"].dtype
    # Exercise the regular-timestamp branch that rebuilds the MultiIndex to
    # retain frequency; it must not change the asset dtype either.
    assert restored.index.levels[0].freq is not None
    assert restored.index.levels[1].dtype == frame["asset"].dtype


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_long_table_to_series_preserves_timestamp_unit(unit):
    frame = pd.DataFrame(
        {
            "datetime": pd.Series(
                np.array(["2024-01-02", "2024-01-03", "2024-01-04"], dtype=f"datetime64[{unit}]")
            ),
            "asset": ["A", "A", "A"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    restored = long_table_to_series(frame)

    assert restored.index.levels[0].dtype == np.dtype(f"datetime64[{unit}]")
    assert list(restored.index.get_level_values(0)) == list(pd.to_datetime(frame["datetime"]))


def test_long_table_to_series_does_not_reduce_timestamp_precision():
    frame = pd.DataFrame(
        {
            "datetime": pd.Series(
                np.array(["2024-01-02T00:00:00.000000001"], dtype="datetime64[ns]")
            ),
            "asset": ["A"],
            "value": [1.0],
        }
    )

    restored = long_table_to_series(frame)

    assert restored.index.levels[0].dtype == np.dtype("datetime64[ns]")
    assert restored.index[0][0].nanosecond == 1


def test_pivot_long_to_wide_and_back():
    long_df = series_to_long_table(_series())
    wide = pivot_long_to_wide(long_df)
    assert list(wide.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert list(wide.columns.astype(str)) == ["A", "B"]
    assert wide.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(1.0)

    back = unpivot_wide_to_long(wide)
    restored = long_table_to_series(back)
    pd.testing.assert_series_equal(restored, _series(), check_names=False)


def test_pivot_multi_factor_long_to_wide():
    long_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "asset": ["A", "B", "A", "B"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    wide = pivot_multi_factor_long_to_wide(long_df, ["f1", "f2"])
    assert wide.columns.names == ["factor_id", "asset"]
    assert wide[("f1", "A")].loc[pd.Timestamp("2024-01-02")] == pytest.approx(1.0)
    assert wide[("f2", "B")].loc[pd.Timestamp("2024-01-03")] == pytest.approx(40.0)


def test_result_store_load_factor_wide(tmp_path):
    _setup_lake(
        tmp_path,
        "f1",
        ["2024-01-15", "2024-01-16"],
        ["A", "B"],
        [1.0, 2.0, 3.0, 4.0],
    )
    from factor_engine.storage.catalog import FactorCatalog

    store = PandasResultStore(tmp_path, FactorCatalog(tmp_path / "_catalog.sqlite"))
    wide = store.load_factor_wide("f1")
    assert wide.shape == (2, 2)
    assert wide.loc[pd.Timestamp("2024-01-15"), "A"] == pytest.approx(1.0)

    series = store.load_factor_series("f1")
    assert isinstance(series.index, pd.MultiIndex)
    assert len(series) == 4


def test_result_store_to_wide_multi_factor(tmp_path):
    _setup_lake(tmp_path, "f1", ["2024-01-15"], ["A"], [1.0], ast_hash="h1")
    _setup_lake(tmp_path, "f2", ["2024-01-15"], ["A"], [2.0], ast_hash="h2")
    from factor_engine.storage.catalog import FactorCatalog

    store = PandasResultStore(tmp_path, FactorCatalog(tmp_path / "_catalog.sqlite"))
    wide = store.to_wide(["f1", "f2"])
    assert ("f1", "A") in wide.columns
    assert ("f2", "A") in wide.columns

    single = store.to_wide(["f1"])
    assert single.shape == (1, 1)
