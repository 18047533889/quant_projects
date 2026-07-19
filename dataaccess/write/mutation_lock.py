"""Dataset-root mutation lock shared by all write/publish operations."""
from __future__ import annotations

import json
import os
import socket
import time
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
) -> Iterator[None]:
    """Acquire an exclusive lock rooted at ``target``.

    The lock is intentionally dataset-root scoped so overwrite, append, upsert,
    delete, and publish cannot mutate the same tree concurrently. Metadata makes
    stale-lock diagnosis possible without exposing application secrets.
    """
    root = Path(target).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".data-access.mutation.lock"
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "target": str(root),
                "created_at": time.time(),
            }
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
            os.fsync(fd)
        except FileExistsError:
            if _is_stale(lock_path, stale_after) and _owner_is_dead(lock_path):
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


def _owner_is_dead(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
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


def _is_stale(path: Path, stale_after: float) -> bool:
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except FileNotFoundError:
        return False
    return age > stale_after
