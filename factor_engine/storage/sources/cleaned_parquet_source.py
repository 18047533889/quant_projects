from __future__ import annotations

from pathlib import Path

from .parquet_source import ParquetSource


class CleanedParquetSource(ParquetSource):
    """面向 cleaned parquet 的 ParquetSource 子类。
    
    参数:
        root: 根目录路径
        timestamp_column: 见函数签名（可选）
        timestamp_col: 见函数签名（可选）
        instrument_column: 见函数签名（可选）
        instrument_col: 见函数签名（可选）
        fields: 逻辑列到物理列映射（可选）
        max_files: 见函数签名（可选）
        timestamp_unit: 见函数签名（可选）
        start_date: 起始日期（可选）
        end_date: 结束日期（可选）
        recursive: 见函数签名（可选）
    
    
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
        """初始化实例。
        
        参数:
            root: 根目录路径
            timestamp_column: 见函数签名（可选）
            timestamp_col: 见函数签名（可选）
            instrument_column: 见函数签名（可选）
            instrument_col: 见函数签名（可选）
            fields: 逻辑列到物理列映射（可选）
            max_files: 见函数签名（可选）
            timestamp_unit: 见函数签名（可选）
            start_date: 起始日期（可选）
            end_date: 结束日期（可选）
            recursive: 见函数签名（可选）
        
        返回:
            无
        """
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
