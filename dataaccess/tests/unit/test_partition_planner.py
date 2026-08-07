"""read/partition_planner.py：时间分区路径裁剪单测。"""
from __future__ import annotations

from data_access.read.partition_planner import (
    PartitionSpec,
    TimePartitionSpec,
    parse_partitioning,
    prune_paths_for_time_range,
)


def test_parse_partitioning():
    spec = parse_partitioning(
        {
            "time": {
                "source": "filename",
                "field": "date",
                "frequency": "daily",
                "pattern": "{date}.parquet",
            }
        }
    )
    assert spec is not None
    assert spec.time.pattern == "{date}.parquet"
    assert parse_partitioning(None) is None


def test_hive_date_wildcard_prune():
    paths = ["/root/date=*/*.parquet"]
    out = prune_paths_for_time_range(paths, ("2024-01-02", "2024-01-03"))
    assert out == [
        "/root/date=2024-01-02/*.parquet",
        "/root/date=2024-01-03/*.parquet",
    ]


def test_hive_year_month_prune():
    paths = ["/root/year=*/month=*/data.parquet"]
    out = prune_paths_for_time_range(paths, ("2024-01-02", "2024-02-01"))
    assert any("year=2024" in p and "month=01" in p for p in out)
    assert any("year=2024" in p and "month=02" in p for p in out)
    assert len(out) == 2


def test_daily_pattern_prune():
    spec = PartitionSpec(
        time=TimePartitionSpec(
            source="filename", field="date", frequency="daily", pattern="{date}.parquet"
        )
    )
    paths = ["/root/*.parquet"]
    out = prune_paths_for_time_range(
        paths, ("2024-01-04", "2024-01-04"), partitioning=spec
    )
    assert out == ["/root/2024-01-04.parquet"]


def test_open_range_not_pruned_by_planner():
    # 开区间无法安全枚举 → 原样返回（交给 manifest / DuckDB）
    spec = PartitionSpec(
        time=TimePartitionSpec(frequency="daily", pattern="{date}.parquet")
    )
    paths = ["/root/*.parquet"]
    out = prune_paths_for_time_range(paths, ("2024-01-04", None), partitioning=spec)
    assert out == paths


def test_no_time_range_returns_as_is():
    paths = ["/root/**/*.parquet"]
    assert prune_paths_for_time_range(paths, None) == paths
