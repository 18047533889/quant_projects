# -*- coding: utf-8
"""publish 完成后写入 target 目录的原子 manifest（审计 / 回滚 / 对账）。"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MANIFEST_NAME = ".publish_manifest.json"


def _content_hash(file_rows: list[Any]) -> str:
    """#P0-30 文件清单内容指纹（path, size, mtime_ns 序列）。"""
    text = json.dumps(file_rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parquet_files_under(root: Path) -> tuple[tuple[str, int, int], ...]:
    """#P0-28 返回 ``(rel_path, size, mtime_ns)`` —— **相对路径**。

    manifest 在 candidate 目录里生成，candidate 随后 rename 成 target；绝对
    candidate 路径上线后即失效。相对路径对 target 目录恒有效。
    """
    entries: list[tuple[str, int, int]] = []
    if not root.exists():
        return ()
    for p in sorted(root.rglob("*.parquet")):
        if p.name.startswith(".") or p.is_symlink():
            continue
        try:
            st = p.stat()
            rel = str(p.relative_to(root))
            entries.append((rel, st.st_size, st.st_mtime_ns))
        except OSError:
            rel = str(p.relative_to(root))
            entries.append((rel, 0, 0))
    return tuple(entries)


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
    """原子写入 ``{target_dir}/.publish_manifest.json``（tmp → replace）。

    #P0-28 ``files[].path`` 一律存**相对路径**（相对 target_dir）——manifest 在
    candidate 里生成、随 candidate rename 上线，绝对 candidate 路径上线后已失效。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    files = _parquet_files_under(target_dir)
    file_entries = [
        {
            "path": rel,
            "size": size,
            "mtime_ns": mtime_ns,
        }
        for rel, size, mtime_ns in files
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
        # #P0-4 base_dir 一律 "."（相对本 manifest 所在目录）。manifest 在
        # candidate 目录生成、随 rename 上线，绝对 candidate 路径上线后已失效；
        # files[].path 相对 "." 恒有效。
        "base_dir": ".",
        # #P0-30 发布代次的 content hash：文件清单指纹，供下游/审计验证该
        # generation 是否被改动（行数/大小/mtime 变化 → hash 变化）。
        "content_hash": _content_hash(
            [(e["path"], e["size"], e["mtime_ns"]) for e in file_entries]
        ),
        "manifest_version": 3,
    }
    # #P1-final closure 19：统一 atomic durable-write（tmp→fsync(fd)→replace→fsync(dir)）
    from data_access.core.atomic import atomic_write_text

    atomic_write_text(
        target_dir / MANIFEST_NAME,
        json.dumps(payload, sort_keys=True, indent=2, default=str),
    )
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
