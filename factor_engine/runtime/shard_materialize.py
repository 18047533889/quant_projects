"""稳定哈希分片：config 路径、factor_id、bucket、time_month。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


def shard_merge_mode_for(contract: Any, chunks: list[Any]) -> Any:
    """R39-PERF-026：物化/合并路径选择 merge 模式（惰性委托给 shard_executor）。

    返回 :class:`~runtime.shard_execution_plan.ShardMergeMode`：
    DIRECT_DURABLE_APPEND / CONCAT_ONLY / ORDERED_MERGE / REDUCE_STATE。
    物化端（materialize_sharded 等批量落盘路径）据此决定：能直接落盘 → 只提交
    manifest；已证明排序 + 不重叠 → 单次 concat；否则 ordered merge。
    """
    from factor_engine.runtime.shard_executor import choose_shard_merge_mode

    return choose_shard_merge_mode(contract, chunks)


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


def shard_bucket_values(
    bucket_count: int,
    *,
    shard_index: int,
    shard_count: int,
) -> list[int]:
    """asset_bucket 分片：按 bucket 编号取模分配。"""
    count = max(1, int(bucket_count))
    index = int(shard_index)
    shards = max(1, int(shard_count))
    if index < 0 or index >= shards:
        raise ValueError(f"shard_index={index} 必须在 [0, {shards}) 内")
    return [b for b in range(count) if b % shards == index]


def month_keys_between(
    since: str | None,
    end_date: str | None,
) -> list[str]:
    """``YYYY-MM-DD`` 区间 → ``YYYY-MM`` 列表。"""
    if since is None and end_date is None:
        return []
    start = datetime.strptime(str(since or end_date)[:10], "%Y-%m-%d")
    end = datetime.strptime(str(end_date or since)[:10], "%Y-%m-%d")
    if end < start:
        start, end = end, start
    months: list[str] = []
    cursor = datetime(start.year, start.month, 1)
    end_marker = datetime(end.year, end.month, 1)
    while cursor <= end_marker:
        months.append(f"{cursor.year:04d}-{cursor.month:02d}")
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return months


def month_date_bounds(month_keys: Iterable[str]) -> tuple[str, str]:
    """多个月份键 → 覆盖全体月份的起止日期。"""
    import calendar

    keys = sorted(set(month_keys))
    if not keys:
        raise ValueError("month_keys 不能为空")
    first = datetime.strptime(keys[0], "%Y-%m")
    last = datetime.strptime(keys[-1], "%Y-%m")
    last_day = calendar.monthrange(last.year, last.month)[1]
    return (
        f"{first.year:04d}-{first.month:02d}-01",
        f"{last.year:04d}-{last.month:02d}-{last_day:02d}",
    )


def shard_time_months(
    month_keys: Iterable[str],
    *,
    shard_index: int,
    shard_count: int,
) -> list[str]:
    """time_month 分片：按 ``YYYY-MM`` 稳定哈希。"""
    return shard_by_hash(
        month_keys,
        shard_index=shard_index,
        shard_count=shard_count,
        key_fn=str,
    )
