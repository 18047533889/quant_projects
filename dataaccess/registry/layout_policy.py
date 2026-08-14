"""Hive bucket 分区策略：symbol hash → bucket 剪枝。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class BucketLayoutPolicy:
    column: str = "bucket"
    count: int = 64
    # #P1-76 记录 hash 算法与版本，写/读端共用同一实现；算法/版本变化必须
    # 显式 bump（否则读端 bucket prune 会漏文件）。
    hash_algorithm: str = "sha256"
    hash_version: int = 1


@dataclass(frozen=True)
class LayoutPolicy:
    bucket: BucketLayoutPolicy | None = None


def _strict_count(value: Any, *, context: str) -> int:
    """#P1-75 count 严格正整数：bool/float/负数/字符串都不接受静默修正。"""
    if isinstance(value, bool):
        raise ValidationError(f"{context}: count 必须是正整数，不能是 bool")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(f"{context}: count 必须是正整数，收到 {value!r}")
        if int(value) <= 0:
            raise ValidationError(f"{context}: count 必须 > 0，收到 {value!r}")
        return int(value)
    raise ValidationError(
        f"{context}: count 必须是正整数，收到 {value!r}（类型 {type(value).__name__}）"
    )


def parse_layout_policy(raw: Any) -> LayoutPolicy | None:
    """解析 datasets.yaml ``layout_policy`` 块。

    #P1-75 strict config：unknown key 拒绝、count 严格正整数、column 非空——
    非法配置直接报错，不静默修正（``count=0 → 1`` 会让 bucket 布局语义漂移）。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidationError(
            f"layout_policy 必须是 mapping，收到 {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - {"bucket"})
    if unknown:
        raise ValidationError(
            f"layout_policy 含未知 key {unknown}；应为 bucket"
        )
    bucket_raw = raw.get("bucket")
    if not bucket_raw:
        return None
    if not isinstance(bucket_raw, dict):
        raise ValidationError("layout_policy.bucket 必须是 mapping")
    b_unknown = sorted(set(bucket_raw) - {"column", "count", "hash_algorithm", "hash_version"})
    if b_unknown:
        raise ValidationError(
            f"layout_policy.bucket 含未知 key {b_unknown}；应为 "
            "column/count/hash_algorithm/hash_version"
        )
    column = bucket_raw.get("column", "bucket")
    if not isinstance(column, str) or not column.strip():
        raise ValidationError(f"layout_policy.bucket.column 必须是非空字符串，收到 {column!r}")
    count = _strict_count(bucket_raw.get("count", 64), context="layout_policy.bucket")
    algo = str(bucket_raw.get("hash_algorithm", "sha256")).strip().lower()
    if algo not in {"sha256", "xxhash64", "md5"}:
        raise ValidationError(
            f"layout_policy.bucket.hash_algorithm 必须 sha256/xxhash64/md5，收到 {algo!r}"
        )
    version = bucket_raw.get("hash_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValidationError(
            f"layout_policy.bucket.hash_version 必须 >=1 的整数，收到 {version!r}"
        )
    return LayoutPolicy(
        bucket=BucketLayoutPolicy(
            column=column,
            count=count,
            hash_algorithm=algo,
            hash_version=version,
        )
    )


def stable_bucket(
    key: str,
    bucket_count: int,
    *,
    algorithm: str = "sha256",
) -> int:
    """稳定哈希分桶 ``[0, bucket_count)``。

    #P1-76 算法来自 BucketLayoutPolicy（写/读共用），散列实现与 bucket_count
    一起构成存储契约；换算法必须 bump hash_version。
    """
    count = max(1, int(bucket_count))
    data = str(key).encode("utf-8")
    if algorithm == "sha256":
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:8], 16) % count
    if algorithm == "md5":
        # MD5 used only for consistent hash bucketing, not cryptographic security
        digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
        return int(digest[:8], 16) % count
    if algorithm == "xxhash64":
        try:
            import xxhash
        except ImportError:
            raise ValidationError(
                "bucket hash_algorithm=xxhash64 需要安装 xxhash 库"
            ) from None
        return int(xxhash.xxh64(data).hexdigest()[:8], 16) % count
    raise ValidationError(f"未知 bucket hash_algorithm={algorithm!r}")


def instrument_buckets(
    instruments: Sequence[str],
    *,
    bucket_count: int = 64,
    algorithm: str = "sha256",
) -> set[int]:
    """标的列表 → 需读取的 bucket 集合。"""
    return {
        stable_bucket(str(inst), bucket_count, algorithm=algorithm) for inst in instruments
    }


def bucket_values_for_instruments(
    instruments: Sequence[str] | None,
    layout_policy: LayoutPolicy | None,
    *,
    partition_columns: Sequence[str],
) -> list[int] | None:
    """若数据集启用 bucket 分区且给定 instrument_filter，返回 bucket 值列表。

    #P1-76 用 layout_policy 声明的 hash 算法/版本，写/读端共享同一实现。
    """
    if not instruments or layout_policy is None or layout_policy.bucket is None:
        return None
    bucket_col = layout_policy.bucket.column
    if bucket_col not in partition_columns:
        return None
    return sorted(
        instrument_buckets(
            instruments,
            bucket_count=layout_policy.bucket.count,
            algorithm=layout_policy.bucket.hash_algorithm,
        )
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
