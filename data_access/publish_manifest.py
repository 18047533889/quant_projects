# -*- coding: utf-8
"""publish 完成后写入 target 目录的原子 manifest（审计 / 回滚 / 对账）。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .read_contract import FileVersion


MANIFEST_NAME = ".publish_manifest.json"


def _parquet_files_under(root: Path) -> tuple[FileVersion, ...]:
    files: list[FileVersion] = []
    if not root.exists():
        return ()
    for p in sorted(root.rglob("*.parquet")):
        if p.name.startswith(".") or p.is_symlink():
            continue
        try:
            st = p.stat()
            files.append(
                FileVersion(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
            )
        except OSError:
            files.append(FileVersion(path=str(p)))
    return tuple(files)


def write_publish_manifest(
    target_dir: Path,
    *,
    staging_name: str,
    target_name: str,
    params: Mapping[str, Any] | None,
    rows: int,
    archive_path: Path | None,
    elapsed_ms: float,
) -> Path:
    """原子写入 ``{target_dir}/.publish_manifest.json``（tmp → replace）。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    files = _parquet_files_under(target_dir)
    payload = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "staging_dataset": staging_name,
        "target_dataset": target_name,
        "params": dict(params or {}),
        "rows": int(rows),
        "archive_path": str(archive_path) if archive_path else None,
        "elapsed_ms": float(elapsed_ms),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "mtime_ns": f.mtime_ns,
            }
            for f in files
        ],
        "file_count": len(files),
        "manifest_version": 1,
    }
    tmp = target_dir / f"{MANIFEST_NAME}.tmp"
    text = json.dumps(payload, sort_keys=True, indent=2, default=str)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target_dir / MANIFEST_NAME)
    return target_dir / MANIFEST_NAME


def read_publish_manifest(target_dir: Path) -> dict[str, Any] | None:
    """读取已发布目录的 manifest；不存在返回 None。"""
    path = target_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
