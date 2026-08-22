from abc import ABC, abstractmethod
import os
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
    """因子结果写入抽象接口。
    
    参数:
        无
    """
    @abstractmethod
    def write(self, factor_name: str, result) -> None:
        """write。
        
        参数:
            factor_name: 因子名称
            result: 因子计算结果 Series
        
        返回:
            无
        """
        raise NotImplementedError


class PandasResultStore:
    """基于本地 Parquet 因子湖的 pandas 读取器。
    
    参数:
        lake_root: 因子湖根目录
        catalog: FactorCatalog 实例（可选）
    """
    def __init__(self, lake_root: str | Path, catalog: FactorCatalog | None = None) -> None:
        """初始化实例。
        
        参数:
            lake_root: 因子湖根目录
            catalog: FactorCatalog 实例（可选）
        
        返回:
            无
        """
        self.lake_root = Path(lake_root)
        self.catalog = catalog or FactorCatalog(self.lake_root / "_catalog.sqlite")

    def load_factor(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """load_factor。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        if self.catalog.get_factor_info(factor_id) is None:
            raise FactorNotFoundError(f"Factor not found: {factor_id}")

        files, _fmt = self._partition_files(factor_id, start=start, end=end)
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
        """读取单因子并返回 MultiIndex Series。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.Series
        """
        return long_table_to_series(self.load_factor(factor_id, start=start, end=end))

    def load_factor_wide(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读取单因子宽表 panel（index=datetime, columns=asset）。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        files, fmt = self._partition_files(factor_id, start=start, end=end)
        if not files:
            raise FactorNotFoundError(f"Factor data not found: {factor_id}")
        if fmt == "wide":
            return self._load_wide_panels(files, start=start, end=end)
        return pivot_long_to_wide(self.load_factor(factor_id, start=start, end=end))

    @staticmethod
    def _load_wide_panels(
        files: list[Path],
        *,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        """_load_wide_panels。
        
        参数:
            files: 见函数签名
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        panels: list[pd.DataFrame] = []
        for path in files:
            panel = pd.read_parquet(path)
            if "datetime" in panel.columns:
                panel = panel.set_index("datetime")
            panels.append(panel)
        combined = pd.concat(panels, axis=0)
        combined.index = pd.to_datetime(combined.index)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        if start is not None:
            combined = combined[combined.index >= pd.Timestamp(start)]
        if end is not None:
            combined = combined[combined.index <= pd.Timestamp(end)]
        return combined

    def load_factor_panel(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """``load_factor_wide`` 别名。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        return self.load_factor_wide(factor_id, start=start, end=end)

    def load_factors(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """load_factors。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
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
        """读取多因子宽表 panel（列 MultiIndex: factor_id × asset）。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        long_df = self.load_factors(factor_ids, start=start, end=end)
        return pivot_multi_factor_long_to_wide(long_df, factor_ids)

    def to_wide(
        self,
        factor_ids: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """单因子返回 datetime×asset；多因子返回 (factor_id, asset) 列宽表。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
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
        """to_pandas。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        return self.load_factors(factor_ids, start=start, end=end)

    def load_matrix(
        self,
        factor_ids: list[str],
        *,
        universe: str,
        frequency: str = "1d",
        matrix_root: str | Path | None = None,
        start: str | None = None,
        end: str | None = None,
        instrument_filter: list[str] | None = None,
    ) -> pd.DataFrame:
        """读取 factor_matrix 宽表（``FactorMatrixMaterializer`` 布局）。

        R39 PERF-065（pushdown）：``start``/``end`` 作为 ``time_range`` 下推给
        ``load_matrix``（hive year/month 分区裁剪 + ``columns=`` pushdown）；
        ``instrument_filter`` 作 row 过滤。调用方仍可依赖同样的返回契约。

        参数:
            factor_ids: 因子 ID 列表
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            matrix_root: factor_matrix 根目录（可选）
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
            instrument_filter: 标的过滤集合（可选）

        返回:
            pd.DataFrame
        """
        from storage.materialize.factor_matrix_materializer import FactorMatrixMaterializer

        root = matrix_root
        if root is None:
            root = os.environ.get("FACTOR_MATRIX_ROOT")
        if root is None:
            root = self.lake_root.parent / "factor_matrix"
        time_range = (start, end) if (start is not None or end is not None) else None
        frame = FactorMatrixMaterializer.load_matrix(
            root,
            universe=universe,
            frequency=frequency,
            factor_ids=factor_ids,
            time_range=time_range,
            instrument_filter=instrument_filter,
        )
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            if start is not None:
                frame = frame[frame["datetime"] >= pd.Timestamp(start)]
            if end is not None:
                frame = frame[frame["datetime"] <= pd.Timestamp(end)]
        return frame.sort_values(["datetime", "asset"]).reset_index(drop=True)

    def _partition_files(
        self,
        factor_id: str,
        *,
        start: str | None,
        end: str | None,
    ) -> tuple[list[Path], str]:
        """_partition_files。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            tuple[list[Path], str]
        """
        factor_dir = self.lake_root / "factors" / factor_id
        if not factor_dir.exists():
            return [], "long"

        wide_files = self._filter_partition_paths(
            sorted(factor_dir.glob("**/panel.parquet")),
            start=start,
            end=end,
        )
        if wide_files:
            return wide_files, "wide"

        long_files = self._filter_partition_paths(
            sorted(factor_dir.glob("**/data.parquet")),
            start=start,
            end=end,
        )
        return long_files, "long"

    @staticmethod
    def _filter_partition_paths(
        paths: list[Path],
        *,
        start: str | None,
        end: str | None,
    ) -> list[Path]:
        """_filter_partition_paths。
        
        参数:
            paths: 见函数签名
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            list[Path]
        """
        start_year = pd.Timestamp(start).year if start is not None else None
        end_year = pd.Timestamp(end).year if end is not None else None
        files: list[Path] = []
        for partition in paths:
            year = None
            for segment in partition.parts:
                if segment.startswith("year="):
                    try:
                        year = int(segment.split("=", 1)[1])
                    except ValueError:
                        year = None
                    break
            if start_year is not None and year is not None and year < start_year:
                continue
            if end_year is not None and year is not None and year > end_year:
                continue
            files.append(partition)
        return files


class PolarsResultStore(PandasResultStore):
    """返回 Polars LazyFrame 的因子湖读取器。
    
    参数:
        无
    """
    def load_factor_wide(
        self,
        factor_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ):
        """load_factor_wide。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            无
        """
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
        """load_factors_wide。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            无
        """
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
        """load_factor。
        
        参数:
            factor_id: 因子唯一标识
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            无
        """
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
        """load_factors。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            无
        """
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
        """to_pandas。
        
        参数:
            factor_ids: 因子 ID 列表
            start: 起始时间（含）（可选）
            end: 结束时间（含）（可选）
        
        返回:
            pd.DataFrame
        """
        return PandasResultStore.load_factors(self, factor_ids, start=start, end=end)


def build_result_store(lake_root: str | Path, catalog: FactorCatalog | None = None):
    """按环境依赖构造 Pandas 或 Polars 结果存储。
    
    参数:
        lake_root: 因子湖根目录
        catalog: FactorCatalog 实例（可选）
    
    返回:
        无
    """
    try:
        import polars  # noqa: F401
    except ModuleNotFoundError:
        return PandasResultStore(lake_root, catalog)
    return PolarsResultStore(lake_root, catalog)
