"""列缓存作用域键：避免跨 dataset / 时间窗 / filter 污染。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from storage.data_scope import compute_data_scope


@dataclass(frozen=True)
class ColumnCacheScope:
    """一次 batch 读列的稳定作用域。"""

    data_scope: str
    columns: tuple[str, ...]
    data_snapshot_id: str | None = None

    @property
    def key(self) -> str:
        """稳定字符串键：data_scope + 列名 + 可选 snapshot id。"""
        base = f"{self.data_scope}:{','.join(self.columns)}"
        if self.data_snapshot_id:
            return f"{base}:snap={self.data_snapshot_id}"
        return base


def column_cache_scope(
    data_source: Any,
    columns: Iterable[str],
    *,
    data_snapshot_id: str | None = None,
) -> ColumnCacheScope:
    """从 data_source 推导列缓存作用域，避免跨 dataset/时间窗污染。

    参数：
        data_source: 实现 ``compute_data_scope`` 的 DataSource
        columns: 本次 batch 要读的列名
        data_snapshot_id: 可选数据快照 id（COS 镜像版本等）
    """
    cols = tuple(sorted(set(str(c) for c in columns if c)))
    snap = data_snapshot_id
    if snap is None:
        snap = getattr(data_source, "data_snapshot_id", None)
    return ColumnCacheScope(
        data_scope=compute_data_scope(data_source),
        columns=cols,
        data_snapshot_id=str(snap) if snap else None,
    )
