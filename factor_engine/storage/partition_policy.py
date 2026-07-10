"""Hive 分区策略：从 datetime 派生分区列、目录路径与 checkpoint 键。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("year",)
SUPPORTED_DERIVED_COLUMNS = frozenset({"year", "month", "day"})


@dataclass(frozen=True)
class PartitionPolicy:
    """Hive 分区与存储格式策略配置。
    
    参数:
        无
    """

    columns: tuple[str, ...] = DEFAULT_PARTITION_COLUMNS
    storage_format: str = "long"  # long | wide
    data_filename: str = "data.parquet"

    @classmethod
    def from_config(
        cls,
        *,
        partition_columns: Iterable[str] | None = None,
        storage_format: str | None = None,
    ) -> "PartitionPolicy":
        """from_config。
        
        参数:
            partition_columns: 见函数签名（可选）
            storage_format: 见函数签名（可选）
        
        返回:
            'PartitionPolicy'
        """
        cols = tuple(str(c) for c in partition_columns) if partition_columns else DEFAULT_PARTITION_COLUMNS
        fmt = str(storage_format or "long").lower()
        if fmt not in {"long", "wide"}:
            raise ValueError(f"storage_format 必须是 long|wide，收到 {fmt!r}")
        filename = "panel.parquet" if fmt == "wide" else "data.parquet"
        for col in cols:
            if col not in SUPPORTED_DERIVED_COLUMNS:
                raise ValueError(
                    f"暂不支持的分区列 {col!r}；当前仅支持 {sorted(SUPPORTED_DERIVED_COLUMNS)}"
                )
        return cls(columns=cols, storage_format=fmt, data_filename=filename)

    @property
    def is_wide(self) -> bool:
        """is_wide。
        
        参数:
            无
        
        返回:
            bool
        """
        return self.storage_format == "wide"


def attach_partition_columns(
    df: pd.DataFrame,
    policy: PartitionPolicy,
    *,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """从 datetime 列派生 hive 分区列。
    
    参数:
        df: 输入 DataFrame
        policy: 分区策略
        datetime_col: 见函数签名（可选）
    
    返回:
        pd.DataFrame
    """
    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    for col in policy.columns:
        if col in out.columns:
            continue
        if col == "year":
            out[col] = dt.dt.year.astype("int32")
        elif col == "month":
            out[col] = dt.dt.month.astype("int32")
        elif col == "day":
            out[col] = dt.dt.day.astype("int32")
    return out


def partition_path_segments(part_values: dict[str, Any], *, column_order: tuple[str, ...] | None = None) -> Path:
    """生成分区目录路径段（year=YYYY/...）。
    
    参数:
        part_values: 见函数签名
        column_order: 见函数签名（可选）
    
    返回:
        Path
    """
    if column_order:
        keys = [k for k in column_order if k in part_values]
    else:
        keys = sorted(part_values.keys())
    return Path("/".join(f"{k}={part_values[k]}" for k in keys))


def partition_key(part_values: dict[str, Any]) -> str:
    """生成 checkpoint 用的稳定分区键字符串。
    
    参数:
        part_values: 见函数签名
    
    返回:
        str
    """
    return "|".join(f"{k}={part_values[k]}" for k in sorted(part_values.keys()))


def parse_partition_key(key: str) -> dict[str, int]:
    """解析分区键字符串为字典。
    
    参数:
        key: 缓存键
    
    返回:
        dict[str, int]
    """
    out: dict[str, int] = {}
    for segment in key.split("|"):
        if "=" not in segment:
            continue
        name, raw = segment.split("=", 1)
        out[name] = int(raw)
    return out


def checkpoint_year(part_values: dict[str, Any]) -> int:
    """将分区值编码为 catalog partition_year 整数。
    
    参数:
        part_values: 见函数签名
    
    返回:
        int
    """
    year = int(part_values.get("year", 0))
    if "month" in part_values:
        return year * 100 + int(part_values["month"])
    return year


def iter_partition_groups(
    df: pd.DataFrame,
    policy: PartitionPolicy,
) -> Iterator[tuple[dict[str, Any], pd.DataFrame]]:
    """按分区列 groupby 迭代产出分区 DataFrame。
    
    参数:
        df: 输入 DataFrame
        policy: 分区策略
    
    返回:
        Iterator[tuple[dict[str, Any], pd.DataFrame]]
    """
    group_cols = list(policy.columns)
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少分区列: {missing}")
    grouped = df.groupby(group_cols, sort=True)
    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        part_values = {col: keys[i] for i, col in enumerate(group_cols)}
        drop_cols = [c for c in group_cols if c in grp.columns]
        yield part_values, grp.drop(columns=drop_cols).reset_index(drop=True)
