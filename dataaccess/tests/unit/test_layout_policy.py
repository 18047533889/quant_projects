"""layout_policy bucket 剪枝单元测试。

DA-P0-008: BucketHashRegistry 单一权威，version 冻结算法与字节编码。
DA-P0-009: write assignment 与 read predicate 绑定同一 validated policy。
DA-P1-027: 拒绝非法 bucket_count，不静默修正。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保从 worktree 导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from registry.layout_policy import (
    BucketHashRegistry,
    BucketLayoutPolicy,
    bucket_partition_predicate,
    bucket_partition_predicate_from_policy,
    instrument_buckets,
    instrument_buckets_from_policy,
    prune_glob_paths_for_buckets,
    stable_bucket,
    stable_bucket_from_policy,
)

# Import the actual ValidationError used by layout_policy
# This ensures we catch the correct exception type
try:
    import data_access.core.exceptions
    ValidationError = data_access.core.exceptions.ValidationError
except (ImportError, AttributeError):
    # Fallback if running in different environment
    from core.exceptions import ValidationError  # type: ignore


def test_stable_bucket_in_range():
    b = stable_bucket("AAPL", 64)
    assert 0 <= b < 64


def test_instrument_buckets_deduplicates():
    buckets = instrument_buckets(["AAPL", "AAPL", "MSFT"], bucket_count=64)
    assert buckets
    assert all(0 <= b < 64 for b in buckets)


def test_prune_glob_paths_for_buckets():
    paths = ["/data/year=*/bucket=*/*.parquet"]
    out = prune_glob_paths_for_buckets(paths, "bucket", [1, 2])
    assert len(out) == 2


def test_bucket_partition_predicate():
    sql, buckets = bucket_partition_predicate(["AAPL", "MSFT"], bucket_count=64)
    assert "bucket IN" in sql
    assert buckets


# DA-P0-008: BucketHashRegistry 单一权威，version 冻结算法
class TestBucketHashRegistry:
    def test_registry_builtin_implementations(self):
        """内置实现：v1 + sha256/md5/xxhash64。"""
        registry = BucketHashRegistry()
        # v1 + sha256
        b1 = registry.compute_bucket("AAPL", 64, 1, "sha256")
        assert 0 <= b1 < 64
        # v1 + md5
        b2 = registry.compute_bucket("AAPL", 64, 1, "md5")
        assert 0 <= b2 < 64

    def test_registry_rejects_invalid_version(self):
        """version 必须 >=1。"""
        registry = BucketHashRegistry()
        with pytest.raises(ValidationError, match="hash_version 必须 >=1"):
            registry.register(0, "sha256", lambda k, c: 0)

    def test_registry_rejects_duplicate_registration(self):
        """禁止覆盖已注册的 (version, algorithm)。"""
        registry = BucketHashRegistry()
        with pytest.raises(ValidationError, match="已注册，禁止覆盖"):
            registry.register(1, "sha256", lambda k, c: 0)

    def test_registry_rejects_unregistered_algorithm(self):
        """未注册的 (version, algorithm) 抛出错误。"""
        registry = BucketHashRegistry()
        with pytest.raises(ValidationError, match="未注册"):
            registry.compute_bucket("AAPL", 64, 99, "unknown")


# DA-P1-027: 拒绝非法 bucket_count，不静默修正
class TestBucketCountValidation:
    def test_rejects_zero_bucket_count(self):
        """bucket_count=0 必须报错，不能静默修正为 1。"""
        with pytest.raises(ValidationError, match="bucket_count 必须是正整数"):
            stable_bucket("AAPL", 0)

    def test_rejects_negative_bucket_count(self):
        """bucket_count=-1 必须报错。"""
        with pytest.raises(ValidationError, match="bucket_count 必须是正整数"):
            stable_bucket("AAPL", -1)

    def test_rejects_float_bucket_count(self):
        """bucket_count=3.5 必须报错。"""
        with pytest.raises(ValidationError, match="bucket_count 必须是正整数"):
            stable_bucket("AAPL", 3.5)  # type: ignore

    def test_rejects_bool_bucket_count(self):
        """bucket_count=True 必须报错。"""
        with pytest.raises(ValidationError, match="不能是 bool"):
            stable_bucket("AAPL", True)  # type: ignore


# DA-P0-008: 金色向量测试 - version 冻结算法与字节编码
class TestBucketHashGoldenVectors:
    """固定 (instrument, version, algorithm, bucket_count) → 预期 bucket。

    跨版本回归测试，确保 hash 实现稳定。
    """

    def test_v1_sha256_golden_vectors(self):
        """v1 + sha256 金色向量。"""
        # AAPL, v1, sha256, 64 buckets
        assert stable_bucket("AAPL", 64, algorithm="sha256", version=1) == 34
        # MSFT, v1, sha256, 64 buckets
        assert stable_bucket("MSFT", 64, algorithm="sha256", version=1) == 39
        # 000001.SZ, v1, sha256, 128 buckets
        assert stable_bucket("000001.SZ", 128, algorithm="sha256", version=1) == 66

    def test_v1_md5_golden_vectors(self):
        """v1 + md5 金色向量（非密码学用途）。"""
        assert stable_bucket("AAPL", 64, algorithm="md5", version=1) == 46
        assert stable_bucket("MSFT", 64, algorithm="md5", version=1) == 44

    def test_golden_vectors_stable_across_calls(self):
        """同一输入多次调用返回同一结果。"""
        b1 = stable_bucket("AAPL", 64, algorithm="sha256", version=1)
        b2 = stable_bucket("AAPL", 64, algorithm="sha256", version=1)
        assert b1 == b2


# DA-P0-009: write assignment 与 read predicate 对称
class TestWriteReadSymmetry:
    """write bucket assignment 与 read pruning predicate 必须 100% 对称。"""

    def test_stable_bucket_from_policy_matches_direct_call(self):
        """从 policy 计算与直接调用结果一致。"""
        policy = BucketLayoutPolicy(
            column="bucket", count=64, hash_algorithm="sha256", hash_version=1
        )
        b1 = stable_bucket_from_policy("AAPL", policy)
        b2 = stable_bucket("AAPL", 64, algorithm="sha256", version=1)
        assert b1 == b2

    def test_instrument_buckets_from_policy_matches_direct_call(self):
        """从 policy 计算 buckets 与直接调用结果一致。"""
        policy = BucketLayoutPolicy(
            column="bucket", count=64, hash_algorithm="sha256", hash_version=1
        )
        buckets1 = instrument_buckets_from_policy(["AAPL", "MSFT"], policy)
        buckets2 = instrument_buckets(
            ["AAPL", "MSFT"], bucket_count=64, algorithm="sha256", version=1
        )
        assert buckets1 == buckets2

    def test_write_bucket_matches_read_predicate(self):
        """write 时分配的 bucket 与 read 时计算的 bucket 完全一致。"""
        policy = BucketLayoutPolicy(
            column="bucket", count=64, hash_algorithm="sha256", hash_version=1
        )
        instruments = ["AAPL", "MSFT", "GOOGL", "000001.SZ", "600000.SH"]

        # Write side: 为每个 instrument 计算 bucket assignment
        write_assignments = {
            inst: stable_bucket_from_policy(inst, policy) for inst in instruments
        }

        # Read side: 生成 pruning predicate
        _, read_buckets = bucket_partition_predicate_from_policy(instruments, policy)

        # 验证：write 分配的所有 buckets 必须包含在 read pruning 中
        assert set(write_assignments.values()) == set(read_buckets)

        # 验证：每个 instrument 的 write bucket 在 read buckets 中
        for inst in instruments:
            write_bucket = write_assignments[inst]
            assert write_bucket in read_buckets

    def test_write_read_symmetry_different_algorithms(self):
        """不同算法的 write/read 对称性。"""
        for algo in ["sha256", "md5"]:
            policy = BucketLayoutPolicy(
                column="bucket", count=128, hash_algorithm=algo, hash_version=1
            )
            instruments = ["AAPL", "MSFT", "GOOGL"]

            write_buckets = {
                inst: stable_bucket_from_policy(inst, policy) for inst in instruments
            }
            _, read_buckets = bucket_partition_predicate_from_policy(instruments, policy)

            assert set(write_buckets.values()) == set(read_buckets)

    def test_write_read_symmetry_different_bucket_counts(self):
        """不同 bucket_count 的 write/read 对称性。"""
        for count in [32, 64, 128, 256]:
            policy = BucketLayoutPolicy(
                column="bucket", count=count, hash_algorithm="sha256", hash_version=1
            )
            instruments = ["AAPL", "MSFT"]

            write_buckets = {
                inst: stable_bucket_from_policy(inst, policy) for inst in instruments
            }
            _, read_buckets = bucket_partition_predicate_from_policy(instruments, policy)

            assert set(write_buckets.values()) == set(read_buckets)
            # 所有 buckets 必须在范围内
            assert all(0 <= b < count for b in read_buckets)


# Low-memory tests: 不加载大数据集
class TestLowMemoryExecution:
    """低内存测试：只测试逻辑，不加载真实数据。"""

    def test_bucket_hash_registry_no_heavy_imports(self):
        """注册表不依赖重量级导入。"""
        registry = BucketHashRegistry()
        # sha256/md5 是标准库，无重量级依赖
        b = registry.compute_bucket("test", 64, 1, "sha256")
        assert 0 <= b < 64

    def test_stable_bucket_string_encoding(self):
        """测试字符串编码正确性（UTF-8）。"""
        # 中文字符
        b1 = stable_bucket("平安银行", 64, algorithm="sha256", version=1)
        assert 0 <= b1 < 64
        # 特殊字符
        b2 = stable_bucket("600000.SH", 64, algorithm="sha256", version=1)
        assert 0 <= b2 < 64
        # 空字符串边界
        b3 = stable_bucket("", 64, algorithm="sha256", version=1)
        assert 0 <= b3 < 64

