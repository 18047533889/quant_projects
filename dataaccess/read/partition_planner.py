"""
data_access.read.partition_planner —— 时间/分区 → 具体文件路径的裁剪

职责
    1. 解析 registry ``partitioning:`` 声明（时间分区 source/field/frequency/pattern）
    2. 在把路径交给 DuckDB 之前，按 ``time_range`` 把 hive 通配（``date=*`` /
       ``year=*`` / ``month=*``）和显式 pattern（``{date}.parquet``）展开成
       具体文件路径，避免 ``**/*.parquet`` 全量 glob

设计要点
    1. 纯函数：输入 glob 列表 + time_range，输出展开后的路径列表。
       不认识 / 无法裁剪的路径原样保留（回退 DuckDB 自身 glob）。
    2. 与 Manifest 互补：partition_planner 做「路径模板级」裁剪，
       manifest 做「文件级」min/max 裁剪。两者都在进 DuckDB 前完成。
    3. hive 通配替换是通用安全操作：只把已知的 date=/year=/month= 目录段
       换成具体值，绝不改动其它路径段。

非职责
    不读文件 footer；不做数据过滤（那是 DuckDB 的事）；不构建 manifest。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence


@dataclass(frozen=True)
class TimePartitionSpec:
    """时间分区声明。``pattern`` 支持 {date}/{year}/{month} 占位符。"""

    source: str = "filename"       # "filename" | "path"
    field: str = "date"
    frequency: str = "daily"       # daily | monthly | yearly
    pattern: str | None = None     # 如 "{date}.parquet"、"date={date}/data.parquet"

    @property
    def is_valid(self) -> bool:
        return self.frequency in {"daily", "monthly", "yearly"}


@dataclass(frozen=True)
class PartitionSpec:
    time: TimePartitionSpec | None = None
    hive: tuple[str, ...] = ()     # hive 分区列（year/month/date...）


def parse_partitioning(raw: Any) -> PartitionSpec | None:
    """解析 YAML ``partitioning:`` 块。未知结构保守返回 None（不裁剪）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    time_spec: TimePartitionSpec | None = None
    time_raw = raw.get("time")
    if isinstance(time_raw, dict):
        pattern = time_raw.get("pattern")
        time_spec = TimePartitionSpec(
            source=str(time_raw.get("source", "filename")) or "filename",
            field=str(time_raw.get("field", "date")) or "date",
            frequency=str(time_raw.get("frequency", "daily")) or "daily",
            pattern=str(pattern) if pattern else None,
        )
    hive_raw = raw.get("hive") or raw.get("partition_columns")
    hive: tuple[str, ...] = ()
    if isinstance(hive_raw, (list, tuple)):
        hive = tuple(str(c) for c in hive_raw if str(c))
    elif isinstance(hive_raw, str) and hive_raw:
        hive = (hive_raw,)
    if time_spec is None and not hive:
        return None
    return PartitionSpec(time=time_spec, hive=hive)


def parse_date(value: Any) -> date | None:
    """解析 time_range 端点；无法解析返回 None。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _iter_dates(start: date, end: date) -> list[date]:
    from datetime import timedelta

    if start > end:
        return []
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _date_segments(d: date, frequency: str) -> dict[str, str]:
    return {
        "date": d.isoformat(),
        "year": f"{d.year:04d}",
        "month": f"{d.month:02d}",
        "day": f"{d.day:02d}",
    }


def _substitute_segment(path: str, key: str, value: str) -> str | None:
    """把路径里形如 ``{key}=*`` 的 hive 段替换为 ``{key}={value}``。"""
    needle = f"{key}=*"
    if needle in path:
        return path.replace(needle, f"{key}={value}", 1)
    return None


def _expand_pattern(pattern: str, segments: dict[str, str]) -> str:
    out = pattern
    for key, val in segments.items():
        out = out.replace("{" + key + "}", val)
    return out


def prune_paths_for_time_range(
    glob_paths: Sequence[str],
    time_range: tuple[Any, Any] | None,
    *,
    partitioning: PartitionSpec | None = None,
    time_column: str | None = None,
) -> list[str]:
    """按 time_range 裁剪 glob 路径列表。

    - time_range 为空 → 原样返回。
    - hive 通配（date=/year=/month=）→ 展开为具体值。
    - partitioning.time.pattern → 按 pattern 展开每日/每月/每年路径。
    无法裁剪的路径段原样保留。
    """
    if time_range is None:
        return list(glob_paths)
    start = parse_date(time_range[0])
    end = parse_date(time_range[1])
    if start is None or end is None:
        return list(glob_paths)

    days = _iter_dates(start, end)
    years = sorted({d.year for d in days})
    months = sorted({(d.year, d.month) for d in days})

    time_spec = partitioning.time if partitioning else None
    frequency = time_spec.frequency if time_spec and time_spec.is_valid else "daily"

    out: list[str] = []
    for path in glob_paths:
        # 1) hive 通配替换
        replaced = _substitute_hive_wildcards(path, days, years, months, frequency)
        # 2) 显式 pattern 展开（pattern 与路径尾部匹配时）
        if time_spec and time_spec.pattern:
            pattern_expanded = _expand_pattern_paths(
                path, days, years, months, time_spec, frequency
            )
            if pattern_expanded is not None:
                out.extend(pattern_expanded)
                continue
        if replaced is not None:
            out.extend(replaced)
        else:
            out.append(path)
    return _dedup(out)


def _substitute_hive_wildcards(
    path: str,
    days: list[date],
    years: list[int],
    months: list[tuple[int, int]],
    frequency: str,
) -> list[str] | None:
    """替换 date=*/year=*/month=* 通配；没有任何通配返回 None。"""
    has_date = "date=*" in path
    has_year = "year=*" in path
    has_month = "month=*" in path
    if not (has_date or has_year or has_month):
        return None
    out: list[str] = []
    if has_date:
        for d in days:
            out.append(_substitute_segment(path, "date", d.isoformat()))
    elif has_month and has_year:
        for y, m in months:
            p = _substitute_segment(path, "year", f"{y:04d}")
            p = _substitute_segment(p, "month", f"{m:02d}")
            out.append(p)
    elif has_month:
        # 没有 year 前缀的 month 段：按 (year, month) 里出现的 month 值替换
        seen_months = sorted({m for _, m in months})
        for m in seen_months:
            out.append(_substitute_segment(path, "month", f"{m:02d}"))
    elif has_year:
        for y in years:
            out.append(_substitute_segment(path, "year", f"{y:04d}"))
    return [p for p in out if p is not None]


def _expand_pattern_paths(
    path: str,
    days: list[date],
    years: list[int],
    months: list[tuple[int, int]],
    time_spec: TimePartitionSpec,
    frequency: str,
) -> list[str] | None:
    """显式 pattern 展开。pattern 是 glob 尾部时替换通配后的列表。"""
    pattern = time_spec.pattern or ""
    pattern_clean = pattern.rstrip("/")
    # pattern 形如 "{date}.parquet" → 与 glob 的尾部比对
    # 我们把 glob 里可能的通配段剥掉再比尾部，尽量保守：
    # 只当 path 尾部与 pattern 的静态部分结构吻合时才展开。
    if "{date}" not in pattern and "{year}" not in pattern and "{month}" not in pattern:
        return None
    out: list[str] = []
    for d in days:
        segs = _date_segments(d, frequency)
        rendered = _expand_pattern(pattern, segs)
        # 把 path 末尾的 glob 通配（如 **/*.parquet / *.parquet / data.parquet）
        # 替换为 rendered。若 pattern 是纯文件名，替换路径最后一个 "/" 之后。
        new_path = _replace_tail(path, rendered)
        if new_path is not None:
            out.append(new_path)
    return out or None


def _replace_tail(path: str, rendered: str) -> str | None:
    """把 path 末尾的通配部分替换为 rendered。无法安全替换返回 None。"""
    # 找最后一个 '/'；尾部是 filename/glob
    idx = path.rfind("/")
    head = path[: idx + 1]
    tail = path[idx + 1 :]
    if "*" not in tail and "?" not in tail:
        # 没有通配的文件名（如 data.parquet）：若 pattern 是纯文件名则替换
        return None
    return head + rendered


def _dedup(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def describe_partitioning(ds: Any) -> str:
    """数据集的分区描述（日志/诊断用）。"""
    partitioning = getattr(ds, "partitioning", None)
    spec = parse_partitioning(partitioning)
    if spec is None:
        return "none"
    parts: list[str] = []
    if spec.time:
        parts.append(
            f"time({spec.time.source},{spec.time.field},{spec.time.frequency})"
        )
    if spec.hive:
        parts.append("hive=" + ",".join(spec.hive))
    return ";".join(parts)
