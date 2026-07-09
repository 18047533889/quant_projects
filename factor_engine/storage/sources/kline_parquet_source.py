from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from logging_utils import get_logger

from .datasource import DataSource


logger = get_logger("storage.kline_parquet_source")


@dataclass
class KlineParquetSource(DataSource):
    """K 线专用 parquet 数据源。

    PR1 起底层改走 `data_access.DuckDBEngine`：一次 SQL 读所有文件，
    性能与并发稳定性都比老的 `pd.read_parquet` 逐文件循环好。
    公开 API（`load_column(name) → MultiIndex Series`）保持不变，
    因子代码无需改动。
    """

    root: str
    instrument_column: str = "ticker"
    timestamp_column: str = "window_start"
    fields: dict[str, str] = field(default_factory=dict)
    file_pattern: str = "*.parquet"
    max_files: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    timestamp_unit: str = "ns"
    timestamp_utc: bool = True
    normalize_timestamp: bool = True
    sort_index: bool = True
    bar_freq: str = "1d"
    _column_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _file_cache: list[Path] | None = field(default=None, init=False, repr=False)

    def load_column(self, name: str):
        if name in self._column_cache:
            return self._column_cache[name]

        source_name = self.fields.get(name, name)
        selected_files = self._selected_files()
        if not selected_files:
            raise FileNotFoundError(f"No parquet files found under {self.root}")

        read_columns = list(dict.fromkeys(
            [self.instrument_column, self.timestamp_column, source_name]
        ))
        try:
            df = self._duckdb_read_batch(selected_files, read_columns)
        except Exception as exc:
            logger.warning(
                "K 线批量读失败（%s），降级到逐文件读取", exc,
            )
            df = self._duckdb_read_per_file(selected_files, read_columns)

        series = self._frame_to_series(df, source_name, name)
        self._column_cache[name] = series
        return series

    def prefetch_columns(self, names: list[str]) -> None:
        """批量预取列，单次 DuckDB 读多列以减少 IO。"""
        missing = [n for n in names if n not in self._column_cache]
        if not missing:
            return

        selected_files = self._selected_files()
        if not selected_files:
            raise FileNotFoundError(f"No parquet files found under {self.root}")

        source_names = [self.fields.get(n, n) for n in missing]
        read_columns = list(dict.fromkeys(
            [self.instrument_column, self.timestamp_column, *source_names]
        ))
        try:
            df = self._duckdb_read_batch(selected_files, read_columns)
        except Exception as exc:
            logger.warning("K 线 prefetch 批量读失败（%s），逐列降级", exc)
            for col in missing:
                self.load_column(col)
            return

        for logical, physical in zip(missing, source_names):
            if logical in self._column_cache:
                continue
            self._column_cache[logical] = self._frame_to_series(df, physical, logical)

    def prefetch_panels(self, names: list[str]) -> None:
        self.prefetch_columns(names)

    def _frame_to_series(self, df, source_name: str, logical_name: str):
        df = df.rename(
            columns={
                self.instrument_column: "instrument",
                self.timestamp_column: "timestamp",
                source_name: logical_name,
            }
        )
        df["timestamp"] = self._convert_timestamp(df["timestamp"])
        if self.start_date:
            df = df[df["timestamp"] >= self.start_date]
        if self.end_date:
            df = df[df["timestamp"] <= self.end_date]
        series = df.set_index(["timestamp", "instrument"])[logical_name]

        if self.sort_index:
            series = series.sort_index()
        return series

    # ---- 内部：DuckDB 读 ----

    def _duckdb_read_batch(self, files: list[Path], read_columns: list[str]):
        try:
            from data_access.engine import get_shared_engine
        except ModuleNotFoundError:
            return self._pandas_read_files(files, read_columns)

        engine = get_shared_engine()
        col_clause = ", ".join(f'"{c}"' for c in read_columns)
        sql = f"SELECT {col_clause} FROM read_parquet(?, union_by_name=true)"
        table = engine.execute_arrow(sql, [[str(p) for p in files]])
        return table.to_pandas(self_destruct=True, split_blocks=True)

    def _duckdb_read_per_file(self, files: list[Path], read_columns: list[str]):
        import pandas as pd
        try:
            from data_access.engine import get_shared_engine
        except ModuleNotFoundError:
            return self._pandas_read_files(files, read_columns, skip_errors=True)

        engine = get_shared_engine()
        col_clause = ", ".join(f'"{c}"' for c in read_columns)
        sql = f"SELECT {col_clause} FROM read_parquet(?, union_by_name=true)"

        frames = []
        errors: list[str] = []
        for path in files:
            try:
                table = engine.execute_arrow(sql, [str(path)])
                frames.append(
                    table.to_pandas(self_destruct=True, split_blocks=True)
                )
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                logger.warning("读取 K 线 parquet 失败: %s - %s", path, exc)
        if not frames:
            preview = "\n".join(errors[:5])
            raise RuntimeError(
                f"所有 K 线 parquet 都读失败: root={self.root}\n前几个错误:\n{preview}"
            )
        return pd.concat(frames, ignore_index=True)

    def _pandas_read_files(
        self,
        files: list[Path],
        read_columns: list[str],
        *,
        skip_errors: bool = False,
    ):
        import pandas as pd

        frames = []
        errors: list[str] = []
        for path in files:
            try:
                frames.append(pd.read_parquet(path, columns=read_columns))
            except Exception as exc:
                if not skip_errors:
                    raise
                errors.append(f"{path}: {exc}")
                logger.warning("读取 K 线 parquet 失败: %s - %s", path, exc)

        if not frames:
            preview = "\n".join(errors[:5])
            raise RuntimeError(
                f"所有 K 线 parquet 都读失败: root={self.root}\n前几个错误:\n{preview}"
            )
        return pd.concat(frames, ignore_index=True)

    def _selected_files(self) -> list[Path]:
        if self._file_cache is None:
            root = Path(self.root)
            self._file_cache = sorted(root.rglob(self.file_pattern))

        files = self._file_cache
        start = date.fromisoformat(self.start_date) if self.start_date else None
        end = date.fromisoformat(self.end_date) if self.end_date else None
        if start or end:
            files = [path for path in files if self._matches_date_range(path, start, end)]

        if self.max_files is not None:
            files = files[: self.max_files]

        return files

    def _matches_date_range(self, path: Path, start: date | None, end: date | None) -> bool:
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            return True

        if start and file_date < start:
            return False
        if end and file_date > end:
            return False
        return True

    def _convert_timestamp(self, series):
        import pandas as pd
        from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

        if is_datetime64_any_dtype(series):
            timestamp = series
        elif is_numeric_dtype(series):
            timestamp = pd.to_datetime(series, unit=self.timestamp_unit, utc=self.timestamp_utc)
        else:
            timestamp = pd.to_datetime(series, utc=self.timestamp_utc)

        if getattr(timestamp.dt, "tz", None) is not None:
            timestamp = timestamp.dt.tz_convert(None)

        if self.normalize_timestamp:
            timestamp = timestamp.dt.normalize()

        return timestamp
