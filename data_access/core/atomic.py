"""Atomic durable file writes（#P1-final closure 18/19）。

正确 durable commit 是 **write tmp → flush/fsync(tmp fd) → replace → fsync(parent)**：
    - fsync(tmp fd) 保证 tmp 文件**内容**落盘（旧代码只对目录 fsync，rename 后
      崩溃可能留下空/截断的目标文件）；
    - os.replace 原子替换目标（旧内容一直存在直到替换成功）；
    - fsync(parent) 保证 rename 的目录项落盘。

Manifest parquet/json/rowgroups、PIT index、publish manifest 以及权威
catalog/sidecar 统一走这里，不再各自拼 tmp+fsync。

#P0 收官（0.9.5，main @ 0cf8e18）：**并发 writer 安全**。
    - 临时文件必须是唯一的 ``.<name>.tmp.<pid>.<uuid8>`` + ``O_EXCL``——两个进程
      同时写同一个 catalog/manifest/sidecar 时，旧代码共用 ``.<name>.tmp`` 会互相
      ``O_TRUNC`` 对方的半成品（各自在对方写了一半的文件上继续）；
    - ``os.write`` 不保证一次写完，必须 **full-write loop**；
    - ``durable=True`` 路径的 fsync 错误**向上传播**——函数名叫 durable，就不能
      「落盘失败但返回成功」；
    - 异常时清理**自己的** tmp（唯一名，不会误删别的 writer 的）。
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Callable


def _unique_tmp(path: Path) -> Path:
    """唯一的临时文件名：``.<name>.tmp.<pid>.<uuid8>``。

    两个 writer 同时写同一目标时各拿各的 tmp，互相不 O_TRUNC；最后只有一个
    ``os.replace`` 能赢（后到者覆盖），但**绝不存在「A 在 B 写了一半的文件上
    继续写」**的损坏窗口。
    """
    return path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )


def _cleanup_tmp(tmp: Path) -> None:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    """Full-write loop：``os.write`` 可能短写，必须循环到写完。

    短写/写 0 字节都视为 I/O 错误——否则函数返回成功但内容不完整。
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"os.write 短写/零写: {written} 字节（文件 {fd}）")
        view = view[written:]


def _fsync_fd(fd: int, *, durable: bool) -> None:
    """对已打开 fd 的内容 fsync；``durable`` 决定失败是否向上传播。"""
    try:
        os.fsync(fd)
    except OSError:
        if durable:
            raise


def fsync_dir(path: Path, *, durable: bool = True) -> None:
    """对 ``path`` 的父目录做 fsync（rename 后目录项落盘）。

    ``durable=True``（默认，权威写入）：fsync 失败向上传播——rename 的目录项没
    落盘就是没有 durable commit，不能静默返回成功。``durable=False``（非权威
    best-effort）才允许吞掉 OSError。
    """
    try:
        fd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        if durable:
            raise
        pass


def _fsync_tmp(tmp: Path, *, durable: bool = True) -> None:
    """对 tmp 文件内容 fsync（只读打开即可刷内核缓存）。"""
    try:
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        if durable:
            raise
        pass


def atomic_write_bytes(
    path: str | Path, data: bytes, *, durable: bool = True
) -> Path:
    """原子 durable 写字节：唯一 tmp → full-write → fsync(fd) → replace → fsync(parent)。

    失败（写/fsync/replace 任意一步）清理自己的 tmp 后向上抛——调用方得到的是
    「要么目标完整更新，要么目标不变」，绝不返回「半成品或未落盘但看似成功」。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)
    try:
        try:
            _write_all(fd, data)
            _fsync_fd(fd, durable=durable)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(path))
        fsync_dir(path, durable=durable)
    except Exception:
        _cleanup_tmp(tmp)
        raise
    return path


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8", durable: bool = True
) -> Path:
    """原子 durable 写文本。"""
    return atomic_write_bytes(path, text.encode(encoding), durable=durable)


def atomic_write_json(
    path: str | Path, payload: Any, *, durable: bool = True
) -> Path:
    """原子 durable 写 JSON（``ensure_ascii=False`` + ``sort_keys`` 稳定序列化）。"""
    import json

    data = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return atomic_write_bytes(path, data, durable=durable)


def atomic_write_file(
    path: str | Path,
    writer: Callable[[Path], Any],
    *,
    durable: bool = True,
) -> Path:
    """原子 durable 写由 ``writer`` 产生的文件（parquet/feather 等 pyarrow 写库）。

    ``writer(tmp_path)`` 负责把内容写到 tmp；完成后统一 fsync tmp 内容 →
    os.replace → fsync(parent)。唯一 tmp + 异常清理，与 ``atomic_write_bytes``
    相同并发 writer 安全语义。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    try:
        writer(tmp)
        _fsync_tmp(tmp, durable=durable)
        os.replace(str(tmp), str(path))
        fsync_dir(path, durable=durable)
    except Exception:
        _cleanup_tmp(tmp)
        raise
    return path
