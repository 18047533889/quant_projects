from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .catalog import FactorCatalog
from .exceptions import FactorNotFoundError
from .factor_format import (
    long_table_to_series,
    pivot_long_to_wide,
    pivot_multi_factor_long_to_wide,
    series_to_long_table,
)


class ResultStore(ABC):
    @abstractmethod
    def write(self, factor_name: str, result) -> None:
        raise NotImplementedError


class PandasResultStore:
    def __init__(self, lake_root: str | Path, catalog: FactorCatalog | None = None) -> None:
        self.lake_root = Path(lake_root)
        self.catalog = catalog or FactorCatalog(self.lake_root / "_catalog.sqlite")

    def load_factor(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if self.catalog.get_factor_info(factor_id) is None:
            raise FactorNotFoundError(f"Factor not found: {factor_id}")

        files = self._partition_files(factor_id, start=start, end=end)
        if not files:
            raise FactorNotFoundError(f"Factor data not found: {factor_id}")

        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        if start is not None:
            frame = frame[frame["datetime"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["datetime"] <= pd.Timestamp(end)]
        return frame.sort_values(["datetime", "asset"]).reset_index(drop=True)

    def load_factor_series(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """读取单因子并返回 MultiIndex Series。"""
        return long_table_to_series(self.load_factor(factor_id, start=start, end=end))

    def load_factor_wide(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读取单因子宽表 panel（index=datetime, columns=asset）。"""
        return pivot_long_to_wide(self.load_factor(factor_id, start=start, end=end))

    def load_factor_panel(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """``load_factor_wide`` 别名。"""
        return self.load_factor_wide(factor_id, start=start, end=end)

    def load_factors(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for factor_id in factor_ids:
            frame = PandasResultStore.load_factor(self, factor_id, start=start, end=end)
            frame = frame[["datetime", "asset", "value"]].rename(columns={"value": factor_id})
            if merged is None:
                merged = frame
            else:
                merged = merged.merge(frame, on=["datetime", "asset"], how="outer")
        if merged is None:
            return pd.DataFrame(columns=["datetime", "asset"])
        return merged.sort_values(["datetime", "asset"]).reset_index(drop=True)

    def load_factors_wide(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读取多因子宽表 panel（列 MultiIndex: factor_id × asset）。"""
        long_df = self.load_factors(factor_ids, start=start, end=end)
        return pivot_multi_factor_long_to_wide(long_df, factor_ids)

    def to_wide(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """单因子返回 datetime×asset；多因子返回 (factor_id, asset) 列宽表。"""
        if len(factor_ids) == 1:
            return self.load_factor_wide(factor_ids[0], start=start, end=end)
        return self.load_factors_wide(factor_ids, start=start, end=end)

    def to_pandas(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return self.load_factors(factor_ids, start=start, end=end)

    def _partition_files(
        self,
        factor_id: str,
        *,
        start: str | None,
        end: str | None,
    ) -> list[Path]:
        factor_dir = self.lake_root / "factors" / factor_id
        if not factor_dir.exists():
            return []

        start_year = pd.Timestamp(start).year if start is not None else None
        end_year = pd.Timestamp(end).year if end is not None else None
        files: list[Path] = []
        for partition in sorted(factor_dir.glob("year=*/data.parquet")):
            try:
                year = int(partition.parent.name.split("=", 1)[1])
            except (IndexError, ValueError):
                year = None
            if start_year is not None and year is not None and year < start_year:
                continue
            if end_year is not None and year is not None and year > end_year:
                continue
            files.append(partition)
        return files


class PolarsResultStore(PandasResultStore):
    def load_factor_wide(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ):
        import polars as pl

        frame = super().load_factor_wide(factor_id, start=start, end=end)
        return pl.from_pandas(frame.reset_index()).lazy()

    def load_factors_wide(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ):
        import polars as pl

        frame = super().load_factors_wide(factor_ids, start=start, end=end)
        return pl.from_pandas(frame.reset_index()).lazy()

    def load_factor(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ):
        import polars as pl

        frame = super().load_factor(factor_id, start=start, end=end)
        return pl.from_pandas(frame).lazy()

    def load_factors(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ):
        import polars as pl

        frame = PandasResultStore.load_factors(self, factor_ids, start=start, end=end)
        return pl.from_pandas(frame).lazy()

    def to_pandas(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return PandasResultStore.load_factors(self, factor_ids, start=start, end=end)


def build_result_store(lake_root: str | Path, catalog: FactorCatalog | None = None):
    try:
        import polars  # noqa: F401
    except ModuleNotFoundError:
        return PandasResultStore(lake_root, catalog)
    return PolarsResultStore(lake_root, catalog)
