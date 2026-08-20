"""Focused DuckDB/pandas parity probes for cross-sectional statistics."""
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


_MODULE = Path(__file__).parents[1] / "backend" / "sql_pushdown" / "advanced_sql_operators.py"
_SPEC = importlib.util.spec_from_file_location("advanced_sql_operators_standalone", _MODULE)
_ADVANCED = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_ADVANCED)


def test_cs_zscore_ignores_nonfinite_values_like_pandas():
    frame = pd.DataFrame(
        {
            "ts": [1, 1, 1, 1],
            "inst": ["a", "b", "c", "d"],
            "_v": [1.0, 2.0, np.nan, np.inf],
        }
    )
    con = duckdb.connect()
    try:
        con.register("input_data", frame)
        sql = _ADVANCED.cs_zscore_sql(
            "SELECT * FROM input_data",
            dialect=_ADVANCED.SqlDialect.DUCKDB,
        )
        actual = con.execute(sql).df().sort_values("inst").reset_index(drop=True)
    finally:
        con.close()

    pivot = frame.pivot(index="ts", columns="inst", values="_v").replace(
        [np.inf, -np.inf], np.nan
    )
    expected = pivot.sub(pivot.mean(axis=1), axis=0).div(
        pivot.std(axis=1).replace(0, 1), axis=0
    )
    expected = expected.stack(dropna=False).rename("_v").reset_index()
    expected = expected.sort_values("inst").reset_index(drop=True)

    pd.testing.assert_series_equal(actual["_v"], expected["_v"], check_names=False)
    assert actual.loc[actual["inst"].isin(["c", "d"]), "_v"].isna().all()
