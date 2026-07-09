"""因子湖 Parquet schema 迁移：为旧分区补全 metadata 列。"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from storage.factor_schema import FACTOR_METADATA_COLUMNS

_DEFAULTS: dict[str, Any] = {
    "calc_time": "",
    "factor_version": "",
    "data_snapshot_id": "",
    "is_valid": 1,
    "invalid_reason": "",
}


def migrate_factor_parquet_file(
    path: Path,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """单文件补全 metadata 列；有变更且非 dry_run 时原子覆盖。"""
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "changed": False, "reason": "missing"}

    table = pq.read_table(path)
    df = table.to_pandas()
    missing = [c for c in FACTOR_METADATA_COLUMNS if c not in df.columns]
    if not missing:
        return {"path": str(path), "changed": False, "rows": len(df)}

    for col in missing:
        df[col] = _DEFAULTS[col]
    if dry_run:
        return {
            "path": str(path),
            "changed": True,
            "dry_run": True,
            "columns_added": missing,
            "rows": len(df),
        }

    tmp = path.parent / f".migrate.tmp.{uuid.uuid4().hex[:8]}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    os.replace(tmp, path)
    return {
        "path": str(path),
        "changed": True,
        "dry_run": False,
        "columns_added": missing,
        "rows": len(df),
    }


def migrate_factor_lake_tree(
    root: str | Path,
    *,
    factor_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """扫描 ``factors/{factor_id}/year=*/*.parquet`` 并迁移 metadata 列。"""
    root = Path(root)
    base = root / "factors"
    if factor_id:
        bases = [base / factor_id]
    else:
        bases = [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []

    files_changed = 0
    files_scanned = 0
    details: list[dict[str, Any]] = []

    for factor_dir in bases:
        for parquet_path in sorted(factor_dir.rglob("*.parquet")):
            files_scanned += 1
            report = migrate_factor_parquet_file(parquet_path, dry_run=dry_run)
            if report.get("changed"):
                files_changed += 1
            details.append(report)

    return {
        "root": str(root),
        "factor_id": factor_id,
        "dry_run": dry_run,
        "files_scanned": files_scanned,
        "files_changed": files_changed,
        "details": details[:50],
    }
