# -*- coding: utf-8
"""publish 完成后写入 target 目录的原子 manifest（审计 / 回滚 / 对账）。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from data_access.read.read_contract import FileVersion


MANIFEST_NAME = ".publish_manifest.json"


def _content_hash(file_rows: list[Any]) -> str:
    """#P0-30 文件清单内容指纹（path, size, mtime_ns 序列）。"""
    text = json.dumps(file_rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
    file_entries = [
        {
            "path": f.path,
            "size": f.size,
            "mtime_ns": f.mtime_ns,
        }
        for f in files
    ]
    payload = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "staging_dataset": staging_name,
        "target_dataset": target_name,
        "params": dict(params or {}),
        "rows": int(rows),
        "archive_path": str(archive_path) if archive_path else None,
        "elapsed_ms": float(elapsed_ms),
        "files": file_entries,
        "file_count": len(files),
        # #P0-30 发布代次的 content hash：文件清单指纹，供下游/审计验证该
        # generation 是否被改动（行数/大小/mtime 变化 → hash 变化）。
        "content_hash": _content_hash(
            [(e["path"], e["size"], e["mtime_ns"]) for e in file_entries]
        ),
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
