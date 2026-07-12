"""Hive bucket 分区策略：symbol hash → bucket 剪枝。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BucketLayoutPolicy:
    column: str = "bucket"
    count: int = 64


@dataclass(frozen=True)
class LayoutPolicy:
    bucket: BucketLayoutPolicy | None = None


def parse_layout_policy(raw: Any) -> LayoutPolicy | None:
    """解析 datasets.yaml ``layout_policy`` 块。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"layout_policy 必须是 mapping，收到 {type(raw).__name__}")
    bucket_raw = raw.get("bucket")
    if not bucket_raw:
        return None
    if not isinstance(bucket_raw, dict):
        raise ValueError("layout_policy.bucket 必须是 mapping")
    return LayoutPolicy(
        bucket=BucketLayoutPolicy(
            column=str(bucket_raw.get("column", "bucket")),
            count=max(1, int(bucket_raw.get("count", 64))),
        )
    )


def stable_bucket(key: str, bucket_count: int) -> int:
    """稳定哈希分桶 ``[0, bucket_count)``。"""
    count = max(1, int(bucket_count))
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % count


def instrument_buckets(
    instruments: Sequence[str],
    *,
    bucket_count: int = 64,
) -> set[int]:
    """标的列表 → 需读取的 bucket 集合。"""
    return {stable_bucket(str(inst), bucket_count) for inst in instruments}


def bucket_values_for_instruments(
    instruments: Sequence[str] | None,
    layout_policy: LayoutPolicy | None,
    *,
    partition_columns: Sequence[str],
) -> list[int] | None:
    """若数据集启用 bucket 分区且给定 instrument_filter，返回 bucket 值列表。"""
    if not instruments or layout_policy is None or layout_policy.bucket is None:
        return None
    bucket_col = layout_policy.bucket.column
    if bucket_col not in partition_columns:
        return None
    return sorted(
        instrument_buckets(instruments, bucket_count=layout_policy.bucket.count)
    )


def prune_glob_paths_for_buckets(
    paths: Sequence[str],
    bucket_column: str,
    buckets: Sequence[int],
) -> list[str]:
    """把 glob 中 ``bucket=*`` 展开为具体 bucket 分区路径。"""
    needle = f"{bucket_column}=*"
    out: list[str] = []
    for path in paths:
        if needle in path:
            for bucket in buckets:
                out.append(path.replace(needle, f"{bucket_column}={bucket}", 1))
        else:
            out.append(path)
    return out


def bucket_partition_predicate(
    instruments: Sequence[str],
    *,
    bucket_column: str = "bucket",
    bucket_count: int = 64,
) -> tuple[str, list[int]]:
    """生成 hive bucket 剪枝谓词片段与 bucket 值列表。"""
    buckets = sorted(instrument_buckets(instruments, bucket_count=bucket_count))
    if not buckets:
        return "", []
    in_list = ", ".join(str(b) for b in buckets)
    return f"{bucket_column} IN ({in_list})", buckets
