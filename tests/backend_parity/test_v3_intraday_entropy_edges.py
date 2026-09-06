import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("duckdb")

from factor_engine.cleaned_operators.intraday.polars_intraday_full import _entropy
from factor_engine.cleaned_operators.microstructure.intraday_agg import IntraEntropy
from tests.backend_parity.intraday_minute_parity import pl_entropy, sql_entropy


def _wide(values_a, values_b=None):
    values_b = values_a if values_b is None else values_b
    ts = pd.date_range("2024-01-02 09:31", periods=len(values_a), freq="min")
    return pd.DataFrame({"A": values_a, "B": values_b}, index=ts)


def _long(wide):
    frame = wide.rename_axis("ts").reset_index().melt("ts", var_name="inst", value_name="volume")
    frame["date"] = frame["ts"].dt.date
    return pl.from_pandas(frame)


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.0, 0.0, 0.0], np.nan),
        ([4.0, 0.0, 0.0], 0.0),
        ([1.0, 1.0, 0.0], 1.0),
        ([1.0, -1.0, 1.0], np.nan),
        ([np.inf, 1.0, 1.0], 1.0),
    ],
)
def test_intraday_entropy_backends_preserve_weight_contract(values, expected):
    wide = _wide(values)
    ref = IntraEntropy()._calculate_series(wide).iloc[0, 0]
    native_input = pl.from_pandas(wide.rename_axis("date"), include_index=True)
    native = _entropy(native_input, True)["A"][0]
    helper = pl_entropy(_long(wide))["A"].iloc[0]
    if np.isnan(expected):
        assert np.isnan(ref) and np.isnan(native) and np.isnan(helper)
    else:
        assert ref == pytest.approx(expected)
        assert native == pytest.approx(expected)
        assert helper == pytest.approx(expected)


def test_intraday_entropy_sql_keeps_all_zero_groups():
    import duckdb

    long = _long(_wide([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])).to_pandas()
    con = duckdb.connect()
    try:
        con.register("minutes", long)
        got = sql_entropy(con, "minutes")
    finally:
        con.close()
    assert np.isnan(got.loc[pd.Timestamp("2024-01-02"), "A"])
    assert got.loc[pd.Timestamp("2024-01-02"), "B"] == pytest.approx(
        1.0
    )
