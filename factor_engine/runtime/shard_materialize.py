"""稳定哈希分片：config 路径、factor_id、任意键。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def stable_bucket(key: str, shard_count: int) -> int:
    """SHA256 前 8 位 hex → ``[0, shard_count)``。"""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(shard_count))


def shard_by_hash(
    items: Iterable[T],
    *,
    shard_index: int,
    shard_count: int,
    key_fn: Callable[[T], str],
) -> list[T]:
    """按 ``key_fn(item)`` 稳定哈希分片。"""
    if shard_count <= 1:
        return list(items)
    index = int(shard_index)
    count = int(shard_count)
    if index < 0 or index >= count:
        raise ValueError(f"shard_index={index} 必须在 [0, {count}) 内")
    selected: list[T] = []
    for item in items:
        if stable_bucket(key_fn(item), count) == index:
            selected.append(item)
    return selected


def shard_factor_ids(
    factor_ids: Iterable[str],
    *,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    """物化/落盘分片：按 ``factor_id`` 哈希。"""
    return shard_by_hash(
        factor_ids,
        shard_index=shard_index,
        shard_count=shard_count,
        key_fn=str,
    )
