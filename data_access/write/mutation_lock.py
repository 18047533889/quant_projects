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
            # O_RDWR：续租/release 需要经 fd 读回 payload（_read_payload_fd）。
            # O_WRONLY 下 os.read(fd) 会 EBADF，fencing 的 fd 身份校验全部失效。
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            now = time.time()
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "target": str(root),
                "process_start_time": time.time(),
                "starttime_ticks": starttime_ticks,
                "transaction_id": txid,
                "acquired_at": now,
                # R55（P0-08/09/10）：lease 显式记进 ls-listed 文件头部，运维/人工
                # 排查一眼可见「锁是否仍活跃」，不再靠 stats 猜。
                "lease_seconds": lease_seconds,
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
                # #7 续租经持有的 fd（inode 身份），绝不能经路径 O_TRUNC——
                # 路径上的锁被打破/重建后，经路径截断会毁掉新 writer 的锁。
                if not _renew_lease(fd, lock_path, txid, lease_seconds):
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
            # #7 owner-only release：先经 fd 写 released 标记（见 _release_lease），
            # 再关 fd——release 需要活 fd 做 inode 身份校验。
            _release_lease(fd, lock_path, txid)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


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


def _renew_lease(fd: int, path: Path, txid: str, lease_seconds: float) -> bool:
    """续租：只改写**自己持有的 inode**（fd 即所有权），绝不动路径上的新锁。

    #7 TOCTOU 修复：旧代码「读 path payload → 判断 txid → os.open(path, O_TRUNC)」
    在判断与 truncate 之间锁可能已被打破/重建，旧 writer 会把**新 writer 的锁**
    truncate 掉。现在：
        - 所有权判定 = ``os.stat(path)`` 与 ``os.fstat(fd)`` 的 **inode 身份**一致，
          且 fd payload 的 transaction_id 仍属于我们；
        - 写入只经 fd（``lseek + ftruncate + write + fsync``）——fd 永远指向创建时
          的 inode；即使路径已换成新 inode（我们的 fd 变成孤儿），也碰不到它。

    返回 False 表示锁已不属于我们（路径 inode 与 fd 不一致 / 文件消失 / txid 不符），
    调用方应停止续租。
    """
    try:
        st = os.stat(path)
        if st.st_ino != os.fstat(fd).st_ino:
            return False  # 路径上的锁已被替换成新 inode → 我们失去所有权
        payload = _read_payload_fd(fd)
        if payload is None or str(payload.get("transaction_id") or "") != txid:
            return False
        now = time.time()
        body = json.dumps(
            {**payload, "acquired_at": now, "lease_until": now + lease_seconds},
            sort_keys=True,
        )
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, (body + "\n").encode())
        os.fsync(fd)
        return True
    except OSError:
        return False


def _release_lease(fd: int, path: Path, txid: str) -> None:
    """owner-only release：经 fd 写 ``released`` 标记，**绝不 unlink**。

    #7 TOCTOU 修复：旧代码「读 path → 判断 txid → unlink(path)」check 与 unlink
    之间锁被替换，旧 writer 会删掉新 writer 的锁。现在旧 owner 永远不对路径做
    任何 unlink/truncate——release 只在**自己的 inode** 上写 released 标记（fd
    指向的 inode），路径上的新锁原样保留。released 标记由 ``_can_break_lock``
    视为立即可打破，下次 acquisition 会 unlink+重建完成清理。

    代价：release 后锁文件保留在目录里（写一次该 root 就会被清理），换来了
    「旧 owner 不可能删除后来创建的新 inode」这一 provable 保证。
    """
    try:
        payload = _read_payload_fd(fd)
        if payload is None or str(payload.get("transaction_id") or "") != txid:
            return  # 锁已不属于我们（txid 不符）——绝不碰
        now = time.time()
        body = json.dumps(
            {**payload, "released": True, "lease_until": now},
            sort_keys=True,
        )
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, (body + "\n").encode())
        os.fsync(fd)
    except OSError:
        pass


def _read_payload_fd(fd: int) -> dict | None:
    """从持有的 fd 读 payload（自己的 inode），与路径无关。"""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 65536).decode("utf-8", "replace")
        payload = json.loads(data)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _can_break_lock(path: Path, *, stale_after: float, hard_break: float) -> bool:
    """#44 判断 stale 锁是否可以自动打破。

    R55（P0-08/09/10）修复「陈旧 lease 无 owner 判定时永久阻塞」：
       - lease 已过期（``lease_until < now``）即视为 stale 可打破——即使 owner
         进程无法本地判定（跨 host / PID reuse）也不再等 hard_break 的 2×lease
         兜底窗口；
       - 未过期 lease 但 owner 已死（PID reuse / kill 探测）→ 立即可打破（跨
         host 无法判定 → 只能靠 lease 过期回收）；
       - 老格式锁（无 lease 字段）保留纯年龄回退：``mtime 年龄 > stale_after``
         且 owner 已死。
    """
    payload = _read_payload(path)
    if payload is None:
        return False
    # #7 released 标记：owner 已显式释放（release 不再 unlink，改为写标记）。
    # 新 writer 看到 released 立即打破——正是 release 不删锁换来的安全：旧 owner
    # 没有 unlink 路径，绝无「删掉新锁」的可能；清理交给下次 acquisition。
    if payload.get("released") is True:
        return True
    # 1) lease 已过期 → stale，直接打破（owner 死活都无所谓——lease 语义过期即失权）。
    lease_until = payload.get("lease_until")
    if isinstance(lease_until, (int, float)) and lease_until > 0:
        return time.time() > lease_until
    # 2) owner 已死（含 PID reuse）→ 直接打破
    if _owner_is_dead(payload):
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
