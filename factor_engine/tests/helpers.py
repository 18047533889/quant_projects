from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from storage.datasource import DataSource

FE_ROOT = Path(__file__).resolve().parents[1]
QUANT_ROOT = FE_ROOT.parent


@dataclass
class InMemorySeriesSource(DataSource):
    data: dict[str, pd.Series]

    def load_column(self, name: str) -> Any:
        return self.data[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        return {n: self.data[n] for n in names if n in self.data}

    def scan_polars_long(self, columns: list[str]):
        """返回 ``ts / inst / <columns>`` 的 Polars LazyFrame（无 panel 往返）。"""
        import polars as pl

        from storage.factor_format import series_to_long_table

        merged: pd.DataFrame | None = None
        tcol = icol = None
        for name in sorted(columns):
            series = self.data[name]
            if tcol is None:
                tcol = str(series.index.names[0])
                icol = str(series.index.names[1])
            part = series_to_long_table(
                series,
                timestamp_col=tcol,
                asset_col=icol,
                value_col=name,
            )
            if merged is None:
                merged = part
            else:
                merged = merged.merge(part, on=[tcol, icol], how="outer")
            merged[name] = merged[name].astype("float64")
        if merged is None:
            raise ValueError("scan_polars_long: no columns")
        renamed = merged.rename(columns={tcol: "ts", icol: "inst"})
        return pl.from_pandas(renamed).lazy()

    def scan_index_long(self):
        """返回 ``ts / inst`` 唯一轴 LazyFrame。"""
        import polars as pl

        lf = self.scan_polars_long(sorted(self.data.keys()))
        return lf.select(["ts", "inst"]).unique()


class NoLoadColumnSource:
    """仅 ``scan_polars_long``；任何列读取/prefetch 一调用即失败。"""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self._data = data

    def load_column(self, name: str):
        raise AssertionError(f"load_column must not be called: {name!r}")

    def load_columns(self, names: list[str]):
        raise AssertionError(f"load_columns must not be called: {names!r}")

    def prefetch_columns(self, names: list[str]):
        raise AssertionError(f"prefetch_columns must not be called: {names!r}")

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

    def scan_index_long(self):
        return self.scan_polars_long(sorted(self._data.keys())).select(["ts", "inst"]).unique()
