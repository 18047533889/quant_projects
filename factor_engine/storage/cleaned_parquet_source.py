from __future__ import annotations

from pathlib import Path

from .parquet_source import ParquetSource


class CleanedParquetSource(ParquetSource):
    """面向 cleaned parquet 的标准数据源。

    默认读取 raw_data_cleaning 清洗层统一输出的 `align_time` 和 `ticker`，
    因此接 cleaned 数据时不需要每次重复写 timestamp/instrument 配置。
    """

    def __init__(
        self,
        root: str | Path,
        *,
        timestamp_column: str | None = None,
        timestamp_col: str | None = None,
        instrument_column: str | None = None,
        instrument_col: str | None = None,
        fields: dict[str, str] | None = None,
        max_files: int | None = None,
        timestamp_unit: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        recursive: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            timestamp_column=timestamp_col or timestamp_column or "align_time",
            instrument_column=instrument_col or instrument_column or "ticker",
            fields=fields,
            max_files=max_files,
            timestamp_unit=timestamp_unit,
            start_date=start_date,
            end_date=end_date,
            recursive=recursive,
        )
