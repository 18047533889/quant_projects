"""layout_policy bucket 剪枝单元测试。"""

from __future__ import annotations

from data_access.registry.layout_policy import (
    bucket_partition_predicate,
    instrument_buckets,
    prune_glob_paths_for_buckets,
    stable_bucket,
)


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
