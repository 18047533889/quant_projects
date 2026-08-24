"""Hive bucket 分区策略：symbol hash → bucket 剪枝。

DA-P0-008: BucketHashRegistry 单一权威，version 冻结算法与字节编码。
DA-P0-009: write assignment 与 read predicate 绑定同一 validated policy。
DA-P1-027: 拒绝非法 bucket_count，不静默修正。
DA-P1-028: MD5 不得作为生产正确性身份选项。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class BucketLayoutPolicy:
    column: str = "bucket"
    count: int = 64
    # DA-P0-008: version 真正控制 hash 实现，冻结算法与字节编码
    hash_algorithm: str = "sha256"
    hash_version: int = 1


@dataclass(frozen=True)
class LayoutPolicy:
    bucket: BucketLayoutPolicy | None = None


# DA-P0-008: 版本化 bucket hash 注册表，version 冻结算法与字节编码
_BucketHashFn = Callable[[str, int], int]


class BucketHashRegistry:
    """单一权威 bucket hash 注册表。

    version 冻结算法与 canonical byte encoding，确保 write assignment
    与 read pruning 使用同一实现。
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[int, str], _BucketHashFn] = {}
        self._register_builtin_implementations()

    def _register_builtin_implementations(self) -> None:
        """注册内置实现。

        DA-P1-028: MD5 仅用于 legacy 兼容读取，不得用于生产写入或新配置。
        """
        # v1: sha256/xxhash64 生产可用；md5 仅 legacy 兼容读取
        self.register(1, "sha256", self._v1_sha256)
        self.register(1, "md5", self._v1_md5_legacy_readonly)
        self.register(1, "xxhash64", self._v1_xxhash64)

    def register(
        self,
        version: int,
        algorithm: str,
        impl: _BucketHashFn,
    ) -> None:
        """注册 (version, algorithm) → 实现。"""
        if version < 1:
            raise ValidationError(f"hash_version 必须 >=1，收到 {version}")
        key = (version, algorithm.lower().strip())
        if key in self._registry:
            raise ValidationError(
                f"bucket hash (v{version}, {algorithm}) 已注册，禁止覆盖"
            )
        self._registry[key] = impl

    def compute_bucket(
        self, key: str, bucket_count: int, version: int, algorithm: str
    ) -> int:
        """计算 bucket: (version, algorithm) 决定实现，bucket_count 必须已验证。

        DA-P1-027: bucket_count 必须是已验证的正整数，不接受 0/-1/float/bool。
        """
        # DA-P1-027: bool 是 int 的子类，必须显式拒绝
        if isinstance(bucket_count, bool):
            raise ValidationError(
                f"bucket_count 必须是正整数，不能是 bool (收到 {bucket_count!r})"
            )
        if not isinstance(bucket_count, int) or bucket_count <= 0:
            raise ValidationError(
                f"bucket_count 必须是正整数，收到 {bucket_count!r} "
                f"(type={type(bucket_count).__name__})"
            )
        key_normalized = (version, algorithm.lower().strip())
        impl = self._registry.get(key_normalized)
        if impl is None:
            raise ValidationError(
                f"bucket hash (v{version}, {algorithm}) 未注册；"
                f"已注册: {sorted(self._registry.keys())}"
            )
        return impl(key, bucket_count)

    @staticmethod
    def _v1_sha256(key: str, bucket_count: int) -> int:
        """v1 + sha256: UTF-8 编码 → SHA-256 → 前8位hex → int → mod."""
        data = key.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:8], 16) % bucket_count

    @staticmethod
    def _v1_md5_legacy_readonly(key: str, bucket_count: int) -> int:
        """v1 + md5: LEGACY 兼容读取旧数据，禁止用于生产写入或新配置。

        DA-P1-028: MD5 不得作为生产正确性身份选项。
        此实现仅用于读取历史 MD5-bucketed 数据，不得用于新写入。
        """
        data = key.encode("utf-8")
        digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
        return int(digest[:8], 16) % bucket_count

    @staticmethod
    def _v1_xxhash64(key: str, bucket_count: int) -> int:
        """v1 + xxhash64: 需要 xxhash 库。"""
        try:
            import xxhash
        except ImportError:
            raise ValidationError(
                "bucket hash_algorithm=xxhash64 需要安装 xxhash 库"
            ) from None
        data = key.encode("utf-8")
        return int(xxhash.xxh64(data).hexdigest()[:8], 16) % bucket_count


# 全局单例注册表
_BUCKET_HASH_REGISTRY = BucketHashRegistry()


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

    DA-P1-028: 拒绝 MD5 作为生产配置选项，仅 sha256/xxhash64 可用。
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
    # DA-P1-028: MD5 不得出现在生产配置中
    if algo not in {"sha256", "xxhash64"}:
        raise ValidationError(
            f"layout_policy.bucket.hash_algorithm 必须为 sha256/xxhash64，收到 {algo!r}"
            f"（MD5 已禁用于生产配置）"
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
    version: int = 1,
) -> int:
    """稳定哈希分桶 ``[0, bucket_count)``。

    DA-P0-008: version + algorithm 决定实现，通过 BucketHashRegistry。
    DA-P1-027: bucket_count 必须是正整数，不接受 0/-1/float/bool。

    .. warning::
        LEGACY/RESEARCH API: 生产代码应使用 ``stable_bucket_from_policy``
        从 validated BucketLayoutPolicy 计算，确保 write/read 对称性。
        直接调用此函数绕过 policy 验证，可能导致 DA-P0-009 违规。
    """
    return _BUCKET_HASH_REGISTRY.compute_bucket(
        str(key), bucket_count, version, algorithm
    )


def stable_bucket_from_policy(key: str, policy: BucketLayoutPolicy) -> int:
    """DA-P0-009: 从 validated policy 计算 bucket，write/read 共用。"""
    return stable_bucket(
        key,
        policy.count,
        algorithm=policy.hash_algorithm,
        version=policy.hash_version,
    )


def instrument_buckets(
    instruments: Sequence[str],
    *,
    bucket_count: int = 64,
    algorithm: str = "sha256",
    version: int = 1,
) -> set[int]:
    """标的列表 → 需读取的 bucket 集合。

    DA-P1-027: bucket_count 必须是正整数。

    .. warning::
        LEGACY/RESEARCH API: 生产代码应使用 ``instrument_buckets_from_policy``
        从 validated BucketLayoutPolicy 计算，确保 write/read 对称性。
        直接调用此函数绕过 policy 验证，可能导致 DA-P0-009 违规。
    """
    return {
        stable_bucket(str(inst), bucket_count, algorithm=algorithm, version=version)
        for inst in instruments
    }


def instrument_buckets_from_policy(
    instruments: Sequence[str], policy: BucketLayoutPolicy
) -> set[int]:
    """DA-P0-009: 从 validated policy 计算 buckets，write/read 共用。"""
    return instrument_buckets(
        instruments,
        bucket_count=policy.count,
        algorithm=policy.hash_algorithm,
        version=policy.hash_version,
    )


def bucket_values_for_instruments(
    instruments: Sequence[str] | None,
    layout_policy: LayoutPolicy | None,
    *,
    partition_columns: Sequence[str],
) -> list[int] | None:
    """若数据集启用 bucket 分区且给定 instrument_filter，返回 bucket 值列表。

    DA-P0-009: 使用 layout_policy 的 version + algorithm，write/read 共享同一实现。
    """
    if not instruments or layout_policy is None or layout_policy.bucket is None:
        return None
    bucket_col = layout_policy.bucket.column
    if bucket_col not in partition_columns:
        return None
    return sorted(instrument_buckets_from_policy(instruments, layout_policy.bucket))


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
    algorithm: str = "sha256",
    version: int = 1,
) -> tuple[str, list[int]]:
    """生成 hive bucket 剪枝谓词片段与 bucket 值列表。

    DA-P1-027: bucket_count 必须是正整数，不接受 0/-1/float/bool。

    .. warning::
        LEGACY/RESEARCH API: 生产代码应使用 ``bucket_partition_predicate_from_policy``
        从 validated BucketLayoutPolicy 计算，确保 write/read 对称性。
        直接调用此函数绕过 policy 验证，可能导致 DA-P0-009 违规。
    """
    buckets = sorted(
        instrument_buckets(
            instruments, bucket_count=bucket_count, algorithm=algorithm, version=version
        )
    )
    if not buckets:
        return "", []
    in_list = ", ".join(str(b) for b in buckets)
    return f"{bucket_column} IN ({in_list})", buckets


def bucket_partition_predicate_from_policy(
    instruments: Sequence[str], policy: BucketLayoutPolicy
) -> tuple[str, list[int]]:
    """DA-P0-009: 从 validated policy 生成剪枝谓词，write/read 共用。"""
    buckets = sorted(instrument_buckets_from_policy(instruments, policy))
    if not buckets:
        return "", []
    in_list = ", ".join(str(b) for b in buckets)
    return f"{policy.column} IN ({in_list})", buckets
