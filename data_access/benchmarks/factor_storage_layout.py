# -*- coding: utf-8 -*-
"""Factor 值存储布局的纯函数助手（可被真实 writer 复用）。

对应平台总规范 §21（COS 数据湖布局）与 §55（"每个日期一个 factor 文件 → object
explosion → bucket/partition"）。本模块只提供**纯函数**：shard/bucket 分配、
partition key、对象数预算检查、delta→compacted 决策。不依赖任何 I/O，便于单测
与真实 writer 直接 import 复用。

核心结论（由 benchmark_factor_storage_layout.py 实证）：
    * 永久 "100k 因子 × 每天 × 每因子一个小 parquet" 布局 = 25M 对象/年（评估
      再翻倍），远超对象数预算 → 禁止。
    * 生产布局 = 日频 delta（factor_bucket 1024 分片 + year/month 分片）+ 定期
      compaction（raw_factor_delta → raw_factor_compacted，old→new lineage，
      原子发布，grace period 后删除旧对象）。对象数被 1024 分片 × 12 月 × 2
      （delta+compacted）≈ 24.6k/年 封顶。
"""

from __future__ import annotations

import hashlib
from typing import Any

# ---- 分片规模（规范 §11-13 建议 ~1024 分片）--------------------------------
DEFAULT_BUCKET_COUNT = 1024

# ---- 对象数预算（COS/S3 单 prefix 对象数上限的经验值）-----------------------
# 超过 OBJECT_COUNT_BUDGET 即判定 over-budget（LIST 延迟随对象数线性恶化）。
OBJECT_COUNT_BUDGET = 1_000_000  # 硬上限：单 factor_values prefix 对象数
OBJECT_COUNT_WARN = 100_000  # 预警线：超过即建议 compaction / 换布局

# ---- 规模预设（因子数）------------------------------------------------------
SCALES = (10_000, 50_000, 100_000)

# ---- 现实行数假设（A 股 ~5000 标的/日，~250 交易日/年）-----------------------
DEFAULT_INSTRUMENTS_PER_DAY = 5000
DEFAULT_TRADING_DAYS_PER_YEAR = 250
DEFAULT_MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# shard / bucket 分配（规范 §11-13）
# ---------------------------------------------------------------------------
def shard_key(factor_id: str, bucket_count: int = DEFAULT_BUCKET_COUNT) -> int:
    """把 factor_id 确定性散列到 ``[0, bucket_count)`` 分片。

    用 blake2b 前 4 字节取模，保证同一 factor_id 永远落在同一分片（增量写入
    与 compaction 都按分片聚合，避免跨分片扫描）。纯函数、确定性、可单测。
    """
    bucket_count = max(1, int(bucket_count))
    digest = hashlib.blake2b(str(factor_id).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % bucket_count


def partition_key(
    asset: str,
    region: str,
    freq: str,
    year: int,
    month: int,
) -> str:
    """构造 ``asset/region/freq/year=YYYY/month=MM`` 分区路径。

    与规范 §21 canonical factor-major 布局对齐：``market=ashare/freq=1d/...``。
    返回 POSIX 相对路径（不含 factor_bucket，bucket 由调用方拼接）。
    """
    return f"{asset}/{region}/{freq}/year={int(year):04d}/month={int(month):02d}"


def factor_bucket_dir(factor_id: str, bucket_count: int = DEFAULT_BUCKET_COUNT) -> str:
    """factor_id → ``factor_bucket=<shard:04d>`` 目录名（规范 §21）。"""
    return f"factor_bucket={shard_key(factor_id, bucket_count):04d}"


# ---------------------------------------------------------------------------
# 对象数估算（布局 A vs 布局 B）
# ---------------------------------------------------------------------------
def layout_a_object_count(
    factor_count: int,
    days: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> int:
    """布局 A（禁止）：每因子 × 每天一个小 parquet。对象数 = factors × days。"""
    return int(factor_count) * int(days)


def layout_b_object_count(
    factor_count: int,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    months: int = DEFAULT_MONTHS_PER_YEAR,
    *,
    delta_and_compacted: bool = True,
) -> int:
    """布局 B（生产）：日频 delta + 分片 compaction。

    每个 (shard, month) 一个 delta parquet；compaction 后每个 (shard, month) 一个
    compacted parquet。对象数 ≈ bucket_count × months × (1 + compacted)。
    与因子数无关（分片数封顶），因此 100k 因子下仍被 1024×12×2 封顶。
    """
    bucket_count = max(1, int(bucket_count))
    months = max(1, int(months))
    per_month = bucket_count * (2 if delta_and_compacted else 1)
    return per_month * months


def object_count_budget() -> dict[str, int]:
    """返回对象数预算（硬上限 / 预警线）。"""
    return {"budget": OBJECT_COUNT_BUDGET, "warn": OBJECT_COUNT_WARN}


def check_object_count_budget(
    factor_count: int,
    layout: str = "B",
    *,
    days: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    months: int = DEFAULT_MONTHS_PER_YEAR,
) -> dict[str, Any]:
    """纯函数：判定某布局在某规模下是否超对象数预算。

    返回 dict：``{layout, factor_count, object_count, budget, warn, verdict}``。
    verdict ∈ {"UNDER_BUDGET", "OVER_BUDGET", "WARN"}。100k 因子布局 A 必为
    OVER_BUDGET（25M >> 1M），布局 B 必为 UNDER_BUDGET（~24.6k << 1M）。
    """
    layout = str(layout).upper()
    if layout == "A":
        count = layout_a_object_count(factor_count, days=days)
    elif layout == "B":
        count = layout_b_object_count(
            factor_count, bucket_count=bucket_count, months=months
        )
    else:
        raise ValueError(f"未知布局: {layout!r}（仅支持 A/B）")
    if count > OBJECT_COUNT_BUDGET:
        verdict = "OVER_BUDGET"
    elif count > OBJECT_COUNT_WARN:
        verdict = "WARN"
    else:
        verdict = "UNDER_BUDGET"
    return {
        "layout": layout,
        "factor_count": int(factor_count),
        "object_count": int(count),
        "budget": OBJECT_COUNT_BUDGET,
        "warn": OBJECT_COUNT_WARN,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# delta → compacted 决策（规范 §14-15）
# ---------------------------------------------------------------------------
def compaction_decision(
    delta_object_count: int,
    *,
    compacted_object_count: int | None = None,
    grace_period_days: int = 7,
    min_delta_objects_before_compact: int = 1,
) -> dict[str, Any]:
    """纯函数：决定某 (shard, month) 是否该 compaction。

    规则：
      * 若 delta 对象数 >= min_delta_objects_before_compact 且已过 grace period，
        则 compaction（raw_factor_delta → raw_factor_compacted）。
      * 原子发布：先写 compacted 对象 + lineage，再删 delta 旧对象（grace 后）。
      * 绝不在 compaction 完成前删除旧对象。

    返回 dict：``{should_compact, reason, grace_period_days, lineage}``。
    """
    should = int(delta_object_count) >= int(min_delta_objects_before_compact)
    return {
        "should_compact": bool(should),
        "reason": (
            f"delta={delta_object_count} >= min={min_delta_objects_before_compact}"
            if should
            else f"delta={delta_object_count} < min={min_delta_objects_before_compact}"
        ),
        "grace_period_days": int(grace_period_days),
        "lineage": "raw_factor_delta -> raw_factor_compacted (old->new, atomic publish)",
        "delete_after_grace": bool(should),
    }


def target_parquet_size(
    instruments_per_day: int = DEFAULT_INSTRUMENTS_PER_DAY,
    factors_per_shard: int = 100,
    *,
    bytes_per_cell: int = 4,
) -> int:
    """估算一个分片 parquet 的目标字节数（float32 契约，~5000 行 × 100 列）。

    用于校准 bucket_count / compaction cadence：目标 parquet 应落在
    8MB-128MB 区间（COS multipart 阈值 8MB 之上，避免小对象爆炸）。
    """
    return max(1, int(instruments_per_day) * int(factors_per_shard) * int(bytes_per_cell))


__all__ = [
    "DEFAULT_BUCKET_COUNT",
    "OBJECT_COUNT_BUDGET",
    "OBJECT_COUNT_WARN",
    "SCALES",
    "shard_key",
    "partition_key",
    "factor_bucket_dir",
    "layout_a_object_count",
    "layout_b_object_count",
    "object_count_budget",
    "check_object_count_budget",
    "compaction_decision",
    "target_parquet_size",
]
