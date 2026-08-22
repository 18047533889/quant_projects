"""因子湖版本摘要、diff 与归档回滚。"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _iter_parquet_files(root: Path) -> list[Path]:
    """递归列举目录下所有 Parquet 文件。
    
    参数:
        root: 根目录路径
    
    返回:
        list[Path]
    """
    if not root.exists():
        return []
    return sorted(root.rglob("*.parquet"))


def _datetime_bounds_from_file(pq_file: Path) -> tuple[Any, Any] | None:
    """从 Parquet 列统计或全列读取 datetime 范围。
    
    参数:
        pq_file: 见函数签名
    
    返回:
        tuple[Any, Any] | None
    """
    pf = pq.ParquetFile(pq_file)
    schema = pf.schema_arrow
    if "datetime" not in schema.names:
        return None

    col_idx = schema.get_field_index("datetime")
    meta = pf.metadata
    mins: list[Any] = []
    maxs: list[Any] = []
    for rg_idx in range(meta.num_row_groups):
        col_meta = meta.row_group(rg_idx).column(col_idx)
        stats = col_meta.statistics
        if stats is not None and stats.has_min_max:
            mins.append(stats.min)
            maxs.append(stats.max)
    if mins and maxs:
        return min(mins), max(maxs)

    table = pq.read_table(pq_file, columns=["datetime"])
    if table.num_rows == 0:
        return None
    col = table.column("datetime").to_pylist()
    return min(col), max(col)


def _snapshot_ids_from_file(pq_file: Path) -> set[str]:
    """从 Parquet 文件收集 data_snapshot_id 集合。
    
    参数:
        pq_file: 见函数签名
    
    返回:
        set[str]
    """
    values: set[str] = set()
    try:
        table = pq.read_table(pq_file, columns=["data_snapshot_id"])
    except Exception:
        return values
    if "data_snapshot_id" not in table.column_names:
        return values
    for item in table.column("data_snapshot_id").to_pylist():
        text = str(item or "").strip()
        if text:
            values.add(text)
    return values


def summarize_factor_tree(root: Path) -> dict[str, Any]:
    """摘要因子目录的行数、年份、时间范围等。
    
    参数:
        root: 根目录路径
    
    返回:
        dict[str, Any]
    """
    files = _iter_parquet_files(root)
    total_rows = 0
    years: set[int] = set()
    min_dt = None
    max_dt = None
    snapshots: set[str] = set()

    for pq_file in files:
        total_rows += pq.read_metadata(pq_file).num_rows
        for part in pq_file.parts:
            if part.startswith("year="):
                try:
                    years.add(int(part.split("=", 1)[1]))
                except ValueError:
                    pass
        try:
            bounds = _datetime_bounds_from_file(pq_file)
        except Exception:
            bounds = None
        if bounds is not None:
            cur_min, cur_max = bounds
            min_dt = cur_min if min_dt is None else min(min_dt, cur_min)
            max_dt = cur_max if max_dt is None else max(max_dt, cur_max)
        snapshots.update(_snapshot_ids_from_file(pq_file))

    return {
        "path": str(root),
        "files": len(files),
        "rows": total_rows,
        "years": sorted(years),
        "datetime_min": None if min_dt is None else str(min_dt),
        "datetime_max": None if max_dt is None else str(max_dt),
        "snapshot_ids": sorted(snapshots),
    }


def diff_factor_trees(path_a: Path, path_b: Path) -> dict[str, Any]:
    """对比两个因子版本目录的差异。
    
    参数:
        path_a: 见函数签名
        path_b: 见函数签名
    
    返回:
        dict[str, Any]
    """
    left = summarize_factor_tree(path_a)
    right = summarize_factor_tree(path_b)
    return {
        "left": left,
        "right": right,
        "rows_delta": int(right["rows"]) - int(left["rows"]),
        "years_only_left": sorted(set(left["years"]) - set(right["years"])),
        "years_only_right": sorted(set(right["years"]) - set(left["years"])),
        "snapshot_only_left": sorted(set(left["snapshot_ids"]) - set(right["snapshot_ids"])),
        "snapshot_only_right": sorted(set(right["snapshot_ids"]) - set(left["snapshot_ids"])),
    }


def list_publish_archives(factor_id: str, lake_root: str | Path) -> list[Path]:
    """列出因子的 publish 归档目录。
    
    参数:
        factor_id: 因子唯一标识
        lake_root: 因子湖根目录
    
    返回:
        list[Path]
    """
    archive_root = Path(lake_root) / "factors" / "_archive"
    if not archive_root.exists():
        return []
    prefix = f"{factor_id}_"
    archives = [p for p in archive_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(archives, key=lambda p: p.name)


def rollback_factor_publish(
    *,
    factor_id: str,
    lake_root: str | Path,
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """将 published 因子目录回滚到归档版本。
    
    参数:
        factor_id: 因子唯一标识（可选）
        lake_root: 因子湖根目录（可选）
        archive_path: 见函数签名（可选）
    
    返回:
        dict[str, Any]
    """
    root = Path(lake_root)
    target_dir = root / "factors" / factor_id
    archives = list_publish_archives(factor_id, root)
    if archive_path is None:
        if not archives:
            raise FileNotFoundError(f"因子 '{factor_id}' 没有可回滚归档")
        chosen = archives[-1]
    else:
        chosen = Path(archive_path)
        if not chosen.exists():
            raise FileNotFoundError(f"归档不存在: {chosen}")

    before = summarize_factor_tree(target_dir) if target_dir.exists() else None
    backup = None
    if target_dir.exists():
        backup = target_dir.parent / f".rollback_backup_{factor_id}_{uuid.uuid4().hex[:8]}"
        os.rename(str(target_dir), str(backup))
    try:
        shutil.copytree(chosen, target_dir, symlinks=False)
    except Exception:
        if backup and backup.exists() and not target_dir.exists():
            os.rename(str(backup), str(target_dir))
        raise

    after = summarize_factor_tree(target_dir)
    return {
        "factor_id": factor_id,
        "archive_path": str(chosen),
        "backup_path": None if backup is None else str(backup),
        "before": before,
        "after": after,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
    }
