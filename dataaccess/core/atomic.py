"""Atomic durable file writes（#P1-final closure 18/19）。

正确 durable commit 是 **write tmp → flush/fsync(tmp fd) → replace → fsync(parent)**：
    - fsync(tmp fd) 保证 tmp 文件**内容**落盘（旧代码只对目录 fsync，rename 后
      崩溃可能留下空/截断的目标文件）；
    - os.replace 原子替换目标（旧内容一直存在直到替换成功）；
    - fsync(parent) 保证 rename 的目录项落盘。

Manifest parquet/json/rowgroups、PIT index、publish manifest 以及权威
catalog/sidecar 统一走这里，不再各自拼 tmp+fsync。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def fsync_dir(path: Path) -> None:
    """对 ``path`` 的父目录做 fsync（rename 后目录项落盘）。"""
    try:
        fd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_tmp(tmp: Path) -> None:
    """对 tmp 文件内容 fsync（只读打开即可刷内核缓存）。"""
    try:
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """原子 durable 写字节：tmp → fsync(fd) → replace → fsync(parent)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    fsync_dir(path)
    return path


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8"
) -> Path:
    """原子 durable 写文本。"""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """原子 durable 写 JSON（``ensure_ascii=False`` + ``sort_keys`` 稳定序列化）。"""
    import json

    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return atomic_write_bytes(path, data)


def atomic_write_file(path: str | Path, writer: Callable[[Path], Any]) -> Path:
    """原子 durable 写由 ``writer`` 产生的文件（parquet/feather 等 pyarrow 写库）。

    ``writer(tmp_path)`` 负责把内容写到 tmp；完成后统一 fsync tmp 内容 →
    os.replace → fsync(parent)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    writer(tmp)
    _fsync_tmp(tmp)
    os.replace(str(tmp), str(path))
    fsync_dir(path)
    return path
