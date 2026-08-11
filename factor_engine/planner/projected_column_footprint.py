# -*- coding: utf-8 -*-
"""R39-P0-PERF-006：ProjectedColumnFootprint —— 真实投影列 footprint。

ReadWave 的内存记账从固定「500k 行 × 8B/列」升级为逐列真实 footprint：
decoded（解码后驻留）、compressed（物理扫描）、null bitmap / offsets /
dictionary（Arrow/Polars 元数据开销）。

数据来源优先级（R39 §4.6）：
    1. manifest column stats；
    2. parquet row-group metadata；
    3. ScanCost（``projection_bytes`` / ``selected_bytes``）；
    4. 最后才 fallback 到 ``_default_scan_bytes`` 的启发式估算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ProjectedColumnFootprint:
    """单列的真实投影 footprint（R39-P0-PERF-006）。"""

    name: str
    decoded_bytes: int = 0          # 解码后驻留字节（wave memory 用）
    compressed_scan_bytes: int = 0  # 物理扫描字节（IO token / ScanCost 用）
    null_bitmap_bytes: int = 0      # null bitmap（Arrow/Polars 元数据）
    offsets_bytes: int = 0          # 变长列 offsets（string / list）
    dictionary_bytes: int = 0       # 字典编码字典区

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        for f in (
            "decoded_bytes",
            "compressed_scan_bytes",
            "null_bitmap_bytes",
            "offsets_bytes",
            "dictionary_bytes",
        ):
            object.__setattr__(self, f, max(0, int(getattr(self, f))))

    @property
    def metadata_bytes(self) -> int:
        return self.null_bitmap_bytes + self.offsets_bytes + self.dictionary_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decoded_bytes": self.decoded_bytes,
            "compressed_scan_bytes": self.compressed_scan_bytes,
            "null_bitmap_bytes": self.null_bitmap_bytes,
            "offsets_bytes": self.offsets_bytes,
            "dictionary_bytes": self.dictionary_bytes,
        }

    @classmethod
    def from_scan_bytes(
        cls,
        name: str,
        *,
        decoded_bytes: int = 0,
        compressed_scan_bytes: int = 0,
    ) -> "ProjectedColumnFootprint":
        """从 ScanCost 语义构造：decoded 用 projection，compressed 用 selected。"""
        return cls(
            name=name,
            decoded_bytes=decoded_bytes,
            compressed_scan_bytes=compressed_scan_bytes,
        )


class ColumnFootprintProvider:
    """footprint 来源协议（duck-typed，不强依赖 DataAccess manifest API）。

    ``column_footprints(dataset, snapshot_id, columns)`` 返回
    ``dict[str, ProjectedColumnFootprint]``（缺失列不出现）。实现可以是
    manifest column stats / parquet row-group metadata / 缓存。
    """

    def column_footprints(
        self,
        dataset: str,
        snapshot_id: str | None,
        columns: Iterable[str],
    ) -> dict[str, ProjectedColumnFootprint]:
        raise NotImplementedError  # pragma: no cover


def union_footprint_bytes(
    footprints: Mapping[str, ProjectedColumnFootprint],
    columns: Iterable[str],
    *,
    per_column_fallback_bytes: int,
) -> int:
    """union 列的 decoded 驻留字节：已知列用真实 footprint，未知列用逐列 fallback。

    R39-P0-PERF-006：wave memory = axis + union(decoded) + metadata + reserves。
    """
    total = 0
    for col in columns:
        fp = footprints.get(col)
        if fp is not None:
            total += fp.decoded_bytes
        else:
            total += max(0, per_column_fallback_bytes)
    return total


def union_scan_bytes(
    footprints: Mapping[str, ProjectedColumnFootprint],
    columns: Iterable[str],
    *,
    per_column_fallback_bytes: int,
) -> int:
    """union 列的一次物理扫描字节（baseline sum 用 physical_union 的对照）。"""
    total = 0
    for col in columns:
        fp = footprints.get(col)
        if fp is not None:
            total += fp.compressed_scan_bytes or fp.decoded_bytes
        else:
            total += max(0, per_column_fallback_bytes)
    return total
