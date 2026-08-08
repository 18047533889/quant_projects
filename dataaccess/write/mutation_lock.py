"""Dataset-root mutation lock shared by all write/publish operations.

#P1-final closure 1：mutation_lock 同线程同 root 可重入。``_dataset_mutation``
在「bump epoch → mutate → rebuild manifest」整个事务外持一把 dataset-root 级锁，
而 write_arrow / upsert_table 正文里还会再对同一个 target_dir 调一次
``mutation_lock``（二者正常情形下是同一个 root）。可重入后内层调用直接 yield，
不产生 FileExistsError 自锁；不同 root 的嵌套调用仍各自持锁。

#P1-final closure 10（Core Freeze fencing）：
  - **lease 续租（heartbeat）**：持有期间后台线程按 lease/3 周期重写 lease_until。
    一个正常运行的 writer 写多久都不会被 hard-break 破锁——「长任务续租」取代
    「超时即破」。锁被外部打破后心跳线程检测到 owner 变化即自动停止。
  - **owner-only release**：release 前读回 payload，只有 transaction_id 仍属于
    自己才 ``unlink``——旧 writer 的锁被打破、新 writer 重建后，旧 writer 的
    ``finally`` 绝不能把**新 writer** 的锁删掉。
  - **PID reuse 检测**：payload 记录 owner 的 /proc/PID/stat starttime ticks；
    ``_owner_is_dead`` 同 host 时对比当前进程 starttime——PID 被复用（不同进程
    拿到同一 PID）视为 owner 已死，不再被 os.kill(pid,0) 误判为活着。
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from data_access.core.exceptions import ValidationError

_held_local = threading.local()


@dataclass
class _LockState:
    """一个已持有的锁的运行时状态。"""

    lock_path: Path
    fd: int
    transaction_id: str
    depth: int = 1
    stop: threading.Event | None = None
    renew_thread: threading.Thread | None = None


def _current_state() -> dict:
    held = getattr(_held_local, "locks", None)
    if held is None:
        held = _held_local.locks = {}
    return held


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
        - owner 进程已死（ProcessLookupError，或 PID 已被复用）；
        - lease 过期且超过 ``hard_break_seconds``（owner 卡死但进程仍在）——
          这是对「进程活但永久卡在写路径」的兜底，默认是 lease 的 2 倍。

    #P1-final closure 10：lease 期间后台心跳续租（``lease/3`` 周期），正常运行的
    writer 永不因超时被破锁；release 只允许删除 transaction_id 仍属于自己的锁。
    """
    root = Path(target).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".data-access.mutation.lock"

    held = _current_state()
    existing = held.get(lock_path)
    if existing is not None:
        # 同线程同 root 已持有 → 同一事务内的重入，直接放行（计数进出）。
        existing.depth += 1
        try:
            yield
        finally:
            existing.depth -= 1
        return

    deadline = time.monotonic() + timeout
    hard_break = stale_after if stale_after > 0 else lease_seconds * 2
    fd: int | None = None
    txid = uuid.uuid4().hex[:12]
    starttime_ticks = _proc_starttime_ticks(os.getpid())
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            now = time.time()
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "target": str(root),
                "process_start_time": time.time(),
                "starttime_ticks": starttime_ticks,
                "transaction_id": txid,
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
    state = _LockState(lock_path=lock_path, fd=fd, transaction_id=txid)
    held[lock_path] = state

    # 后台心跳续租：正常运行的 writer 永不因 lease 过期被 hard-break 破锁。
    # 心跳每次重读 payload——锁已被打破、被新 writer 重建时 transaction_id 变化，
    # 立即停止续租（不能给别人的锁续命）。
    if lease_seconds and lease_seconds > 0:
        interval = max(lease_seconds / 3.0, 0.5)
        stop = threading.Event()
        state.stop = stop

        def _renew() -> None:
            while not stop.wait(interval):
                if not _renew_lease(lock_path, txid, lease_seconds):
                    return  # 锁已不属于我们 → 停止续租

        t = threading.Thread(
            target=_renew, daemon=True, name=f"da-lock-renew-{txid[:8]}"
        )
        t.start()
        state.renew_thread = t
    try:
        yield
    finally:
        if state.stop is not None:
            state.stop.set()
        if state.renew_thread is not None:
            state.renew_thread.join(timeout=2.0)
        held.pop(lock_path, None)
        try:
            os.close(fd)
        finally:
            # owner-only release：只有锁仍是自己的（transaction_id 匹配）才删。
            _release_lease(lock_path, txid)


def _proc_starttime_ticks(pid: int) -> float | None:
    """读取 /proc/PID/stat 字段 22（starttime，boot 以来 clock ticks）。

    comm 可能含空格/括号，必须从最后一个 ``)`` 之后取字段：其后第 1 个字段是
    state（字段 3），starttime 是字段 22 → 切片索引 19。
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read().decode("ascii", "replace")
        rp = data.rfind(")")
        if rp < 0:
            return None
        fields = data[rp + 2:].split()
        if len(fields) <= 19:
            return None
        return float(fields[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _renew_lease(path: Path, txid: str, lease_seconds: float) -> bool:
    """续租：只有锁的 transaction_id 仍属于我们才改写 lease_until。

    返回 False 表示锁已不属于我们（被打破/重建），调用方应停止续租。
    """
    payload = _read_payload(path)
    if payload is None:
        return False
    if str(payload.get("transaction_id") or "") != txid:
        return False
    try:
        now = time.time()
        body = json.dumps(
            {**payload, "acquired_at": now, "lease_until": now + lease_seconds},
            sort_keys=True,
        )
        fd = os.open(str(path), os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(fd, (body + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def _release_lease(path: Path, txid: str) -> None:
    """owner-only release：读回 payload，transaction_id 匹配才 unlink。"""
    payload = _read_payload(path)
    if payload is not None and str(payload.get("transaction_id") or "") != txid:
        return  # 锁已被打破并重建，是别人的锁——绝不删除
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _can_break_lock(path: Path, *, stale_after: float, hard_break: float) -> bool:
    """#44 判断 stale 锁是否可以自动打破。"""
    payload = _read_payload(path)
    if payload is None:
        return False
    # 1) owner 已死（含 PID reuse）→ 直接打破
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
    """#P0-31 判断锁 owner 进程是否已死（#P1-final closure 10 加入 PID reuse 检测）。

    跨机器共享文件系统（NFS/Lustre）上，锁可能来自另一台 host：
    - owner host != 本机 → **绝不能用本机 os.kill(pid, 0) 判定死亡**（本机恰好
      没有该 PID 不代表 owner 死了），视为"活着"，只能靠 lease 过期回收。
    - owner host == 本机 → 用 os.kill(pid, 0) 探测本地进程；若 payload 带
      ``starttime_ticks``，再对比当前进程的 starttime——PID 被复用（同 PID 但
      starttime 不同）说明 owner 进程已死，换了个新进程占住 PID。
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
    # PID reuse 检测：payload 记录的 starttime 与当前进程不一致 → owner 已死
    saved_ticks = payload.get("starttime_ticks")
    if isinstance(saved_ticks, (int, float)) and saved_ticks > 0:
        cur_ticks = _proc_starttime_ticks(pid)
        if cur_ticks is not None and cur_ticks != float(saved_ticks):
            return True  # 同 PID 但不同进程 → owner 死，PID 被复用
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
