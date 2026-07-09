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
