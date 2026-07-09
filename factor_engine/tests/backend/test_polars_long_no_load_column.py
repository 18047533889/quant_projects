# -*- coding: utf-8
"""polars_long 不得在最后偷偷 ``load_column``。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine


class _ScanOnlySource:
    """仅提供 ``scan_polars_long``；``load_column`` 一调用即失败。"""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self._data = data
        self.load_column_calls = 0

    def load_column(self, name: str):
        self.load_column_calls += 1
        raise AssertionError(f"polars_long must not call load_column({name!r})")

    def scan_polars_long(self, columns: list[str]):
        import polars as pl

        from storage.factor_format import series_to_long_table

        merged = None
        tcol = icol = None
        for name in sorted(columns):
            series = self._data[name]
            if tcol is None:
                tcol = str(series.index.names[0])
                icol = str(series.index.names[1])
            part = series_to_long_table(
                series,
                timestamp_col=tcol,
                asset_col=icol,
                value_col=name,
            )
            merged = part if merged is None else merged.merge(part, on=[tcol, icol], how="outer")
            merged[name] = merged[name].astype("float64")
        renamed = merged.rename(columns={tcol: "ts", icol: "inst"})
        return pl.from_pandas(renamed).lazy()


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0], index=idx)
    return _ScanOnlySource(data={"close": close})


def test_polars_long_does_not_load_column(source):
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 2)
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=expr))
    assert out.get("used_polars_long_path") is True
    assert out.get("used_polars_long_native") is True
    assert source.load_column_calls == 0
    assert len(out["result"]) == 4


def test_polars_long_prefetch_skipped(source):
    from planner.sql_io import should_skip_column_prefetch
    from api.factor import Factor
    from api.cleaned_ops import make_cleaned_call_factory
    from api.columns import col

    factor = Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    plan, _ = eng.compile(factor)
    assert should_skip_column_prefetch([plan], input_dq_check=False, backend=eng.backend)
