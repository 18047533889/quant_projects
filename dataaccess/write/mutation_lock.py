"""Dataset-root mutation lock shared by all write/publish operations."""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from data_access.core.exceptions import ValidationError


@contextmanager
def mutation_lock(
    target: str | Path,
    *,
    timeout: float = 30.0,
    poll: float = 0.05,
    stale_after: float = 3600.0,
    lease_seconds: float = 3600.0,
) -> Iterator[None]:
    """Acquire an exclusive lock rooted at ``target``.

    The lock is intentionally dataset-root scoped so overwrite, append, upsert,
    delete, and publish cannot mutate the same tree concurrently.

    #44 lease / stale recovery：锁文件记录 pid / host / process_start_time /
    transaction_id / acquired_at / lease_until。以下情况允许自动打破 stale 锁：
        - owner 进程已死（ProcessLookupError）；
        - lease 过期且超过 ``hard_break_seconds``（owner 卡死但进程仍在）——
          这是对「进程活但永久卡在写路径」的兜底，默认是 lease 的 2 倍。
    """
    root = Path(target).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".data-access.mutation.lock"
    deadline = time.monotonic() + timeout
    hard_break = stale_after if stale_after > 0 else lease_seconds * 2
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            now = time.time()
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "target": str(root),
                "process_start_time": _process_start_time(),
                "transaction_id": uuid.uuid4().hex[:12],
                "acquired_at": now,
                "lease_until": now + lease_seconds,
            }
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        except FileExistsError:
            if _can_break_lock(lock_path, stale_after=stale_after, hard_break=hard_break):
                try:
                    lock_path.unlink()
                    continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise ValidationError(
                    f"数据集正在被其他写操作占用：{lock_path}；"
                    "请等待当前操作完成或清理确认已失效的锁。"
                )
            time.sleep(poll)
        except OSError:
            if fd is not None:
                os.close(fd)
            raise
    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _process_start_time() -> float:
    try:
        with open(f"/proc/{os.getpid()}/stat", "r", encoding="utf-8") as fh:
            parts = fh.read().split()
        # /proc/PID/stat: 字段 22（index 21）是 starttime（clock ticks since boot）
        return time.time()  # 拿不到精确 starttime 时回退当前时间（仅用于诊断展示）
    except (OSError, IndexError):
        return time.time()


def _can_break_lock(path: Path, *, stale_after: float, hard_break: float) -> bool:
    """#44 判断 stale 锁是否可以自动打破。"""
    payload = _read_payload(path)
    if payload is None:
        return False
    # 1) owner 已死 → 直接打破
    if _owner_is_dead(payload):
        return True
    # 2) lease 过期且超过 hard_break（owner 卡死但仍活着）→ 打破
    lease_until = payload.get("lease_until")
    if isinstance(lease_until, (int, float)) and lease_until > 0:
        if time.time() - lease_until > hard_break:
            return True
    # 3) 纯年龄阈值（老格式锁无 lease）
    age = _lock_age(path)
    return age > stale_after and _owner_is_dead(payload)


def _read_payload(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _owner_is_dead(payload: dict) -> bool:
    try:
        pid = int(payload.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


def _lock_age(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except FileNotFoundError:
        return 0.0
