# -*- coding: utf-8 -*-
"""factor_storage_layout —— 纯函数单测（确定性 / 分片范围 / 预算判定）。"""
from __future__ import annotations

import pytest

from data_access.benchmarks.factor_storage_layout import (
    DEFAULT_BUCKET_COUNT,
    OBJECT_COUNT_BUDGET,
    OBJECT_COUNT_WARN,
    SCALES,
    check_object_count_budget,
    compaction_decision,
    factor_bucket_dir,
    layout_a_object_count,
    layout_b_object_count,
    partition_key,
    shard_key,
    target_parquet_size,
)


class TestShardKey:
    def test_deterministic(self):
        a = shard_key("F_000001")
        b = shard_key("F_000001")
        assert a == b

    def test_in_range(self):
        for fid in (f"F_{i:06d}" for i in range(2000)):
            s = shard_key(fid)
            assert 0 <= s < DEFAULT_BUCKET_COUNT

    def test_distinct_ids_not_all_same_shard(self):
        shards = {shard_key(f"F_{i:06d}") for i in range(2000)}
        # 2000 个因子分布在 1024 分片，不可能全落在同一个。
        assert len(shards) > 100

    def test_bucket_count_scale(self):
        assert shard_key("F_000001", 8) < 8
        assert shard_key("F_000001", 1024) == shard_key("F_000001", 1024)

    def test_bucket_count_floor(self):
        assert shard_key("X", 0) == 0
        assert shard_key("X", -3) == 0


class TestPartitionKey:
    def test_shape(self):
        assert partition_key("ashare", "cn", "1d", 2026, 1) == "ashare/cn/1d/year=2026/month=01"

    def test_zero_pad(self):
        assert "month=01" in partition_key("a", "b", "c", 2026, 1)
        assert "month=12" in partition_key("a", "b", "c", 2026, 12)

    def test_no_slashes_injection(self):
        # 拒绝路径注入：region/asset 内含 / 会产生额外层级，但不会 escape 根。
        assert "asset=.." not in partition_key("..", "cn", "1d", 2026, 1)


class TestFactorBucketDir:
    def test_bucket_dir(self):
        s = shard_key("F_000001")
        assert factor_bucket_dir("F_000001") == f"factor_bucket={s:04d}"

    def test_deterministic(self):
        assert factor_bucket_dir("F_000001") == factor_bucket_dir("F_000001")


class TestObjectCount:
    def test_layout_a(self):
        assert layout_a_object_count(100) == 100 * 250

    def test_layout_a_days_override(self):
        assert layout_a_object_count(100, days=10) == 1000

    def test_layout_b(self):
        # bucket × months × 2（delta+compacted）
        assert layout_b_object_count(100_000) == 1024 * 12 * 2

    def test_layout_b_single_variant(self):
        assert layout_b_object_count(100_000, delta_and_compacted=False) == 1024 * 12

    def test_layout_b_independent_of_factor_count(self):
        assert layout_b_object_count(10_000) == layout_b_object_count(100_000)

    def test_layout_b_does_not_explode(self):
        assert layout_b_object_count(100_000) < OBJECT_COUNT_BUDGET


@pytest.mark.parametrize("factor_count", [10_000, 50_000, 100_000])
class TestBudgetCheck:
    def test_layout_a_over_budget(self, factor_count):
        r = check_object_count_budget(factor_count, "A")
        assert r["layout"] == "A"
        assert r["object_count"] == layout_a_object_count(factor_count)
        assert r["verdict"] == "OVER_BUDGET"

    def test_layout_b_under_budget(self, factor_count):
        r = check_object_count_budget(factor_count, "B")
        assert r["layout"] == "B"
        assert r["object_count"] == layout_b_object_count(factor_count)
        assert r["verdict"] == "UNDER_BUDGET"

    def test_unknown_layout_raises(self, factor_count):
        with pytest.raises(ValueError):
            check_object_count_budget(factor_count, "Z")

    def test_scale_enum(self, factor_count):
        assert SCALES == (10_000, 50_000, 100_000)


class TestCompactionDecision:
    def test_should_compact_when_delta_present(self):
        d = compaction_decision(delta_object_count=50)
        assert d["should_compact"] is True
        assert d["grace_period_days"] == 7
        assert "raw_factor_compacted" in d["lineage"]

    def test_no_compact_when_zero_delta(self):
        d = compaction_decision(delta_object_count=0)
        assert d["should_compact"] is False
        assert d["delete_after_grace"] is False

    def test_grace_period_override(self):
        assert compaction_decision(10, grace_period_days=30)["grace_period_days"] == 30

    def test_min_delta_override(self):
        assert compaction_decision(2, min_delta_objects_before_compact=3)["should_compact"] is False
        assert compaction_decision(3, min_delta_objects_before_compact=3)["should_compact"] is True


class TestTargetParquetSize:
    def test_positive(self):
        assert target_parquet_size() > 0

    def test_rough_order(self):
        # 5000 × 100 × 4 = 2MB
        assert target_parquet_size() == 2_000_000

    def test_parts(self):
        assert target_parquet_size(1000, 10) == 40_000