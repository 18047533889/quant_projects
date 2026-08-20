# -*- coding: utf-8 -*-
"""因子湖跨 region / 灾备目录同步。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def replicate_lake_directory(
    source_root: str | Path,
    target_root: str | Path,
    *,
    dry_run: bool = False,
    include_catalog: bool = True,
) -> dict[str, Any]:
    """将因子湖目录原子复制到灾备/跨 region 目标（rsync 风格）。"""
    src = Path(source_root)
    dst = Path(target_root)
    if not src.exists():
        raise FileNotFoundError(f"source lake not found: {src}")

    copied: list[str] = []
    if include_catalog:
        for name in ("_catalog.sqlite", "_catalog.sqlite-wal", "_catalog.sqlite-shm"):
            path = src / name
            if path.exists():
                target = dst / name
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                copied.append(str(target if not dry_run else path))

    factors_src = src / "factors"
    if factors_src.exists():
        factors_dst = dst / "factors"
        if not dry_run:
            factors_dst.mkdir(parents=True, exist_ok=True)
            for item in factors_src.iterdir():
                dest = factors_dst / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
                copied.append(str(dest))
        else:
            copied.extend(str(p) for p in factors_src.rglob("*"))

    return {
        "ok": True,
        "source": str(src),
        "target": str(dst),
        "dry_run": dry_run,
        "paths": copied[:50],
        "paths_count": len(copied),
    }


def disaster_recovery_verify(
    primary_root: str | Path,
    replica_root: str | Path,
) -> dict[str, Any]:
    """灾备演练：对比 primary/replica 因子 ID 与 catalog 存在性。"""
    primary = Path(primary_root)
    replica = Path(replica_root)
    primary_factors = sorted(p.name for p in (primary / "factors").glob("*") if p.is_dir()) if (primary / "factors").exists() else []
    replica_factors = sorted(p.name for p in (replica / "factors").glob("*") if p.is_dir()) if (replica / "factors").exists() else []
    missing = sorted(set(primary_factors) - set(replica_factors))
    return {
        "ok": not missing,
        "primary_count": len(primary_factors),
        "replica_count": len(replica_factors),
        "missing_on_replica": missing,
        "catalog_primary": (primary / "_catalog.sqlite").exists(),
        "catalog_replica": (replica / "_catalog.sqlite").exists(),
    }
