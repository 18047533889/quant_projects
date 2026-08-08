"""Dataset-root mutation lock shared by all write/publish operations."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from data_access.core.exceptions import ValidationError

# #P1-final closure 1：mutation_lock 改为**同线程同 root 可重入**。
# ``_dataset_mutation`` 在「bump epoch → mutate → rebuild manifest」整个事务外
# 持一把 dataset-root 级锁，而 write_arrow / upsert_table 正文里还会再对同一个
# target_dir 调一次 ``mutation_lock``（二者正常情形下是同一个 root）。可重入后，
# 内层调用直接 yield，不产生 FileExistsError 自锁；不同 root 的嵌套调用仍各自
# 持锁（锁序固定：事务 root → body target，无死锁）。
_held_local = threading.local()


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

    同一线程已持有同一个 root 的锁时直接通过（可重入），供 ``_dataset_mutation``
    的整事务锁与正文的 target 锁在 root 相同时共用。
    """
    root = Path(target).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".data-access.mutation.lock"

    held = getattr(_held_local, "roots", None)
    if held is None:
        held = _held_local.roots = set()
    if lock_path in held:
        # 同线程同 root 已持有 → 同一事务内的重入，直接放行。
        yield
        return

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
    held.add(lock_path)
    try:
        yield
    finally:
        held.discard(lock_path)
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
    """#P0-31 判断锁 owner 进程是否已死。

    跨机器共享文件系统（NFS/Lustre）上，锁可能来自另一台 host：
    - owner host != 本机 → **绝不能用本机 os.kill(pid, 0) 判定死亡**（本机恰好
      没有该 PID 不代表 owner 死了），视为"活着"，只能靠 lease 过期回收。
    - owner host == 本机 → 用 os.kill(pid, 0) 探测本地进程。
    """
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    owner_host = str(payload.get("host") or "")
    if owner_host and owner_host != socket.gethostname():
        # 锁来自别的机器：本地没有这个 PID 是正常现象，不能据此打破锁。
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
