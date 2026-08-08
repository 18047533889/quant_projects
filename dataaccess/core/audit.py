"""
data_access.audit —— 审计日志（JSONL，append-only）

职责：
    1. 每次写入操作记录一行 JSON（ts / operator / namespace / dataset / op / rows / elapsed_ms / ok / error）
    2. 读操作默认不记（太吵），可通过 QUANT_AUDIT_READS=true 打开

非职责：
    - 不做聚合、告警、限流（那是 PR5 的事，基于这份 JSONL 消费即可）
    - 不做日志轮转（交给外部 logrotate；本模块只 append）

落盘位置（按优先级）：
    1. 环境变量 QUANT_AUDIT_LOG（绝对路径）
    2. ${QUANTSOCIETY_WORKSPACE_DATA_ROOT:-<repo>/workspace_data}/logs/data_access_audit.jsonl

线程安全：单进程内用一把 Lock 串行追加；JSONL 格式天然 append-safe，
跨进程并发追加在 Linux 上对 <PIPE_BUF(4KB) 的 write 也是原子的，够用。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import enum
import json
import math
import os
import threading
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .namespace import resolve_namespace, resolve_operator


_DEFAULT_AUDIT_SUBPATH = "logs/data_access_audit.jsonl"
_write_lock = threading.Lock()


def _resolve_audit_path() -> Path:
    """按优先级算审计日志路径。第一次调用时创建父目录。"""
    explicit = os.environ.get("QUANT_AUDIT_LOG")
    if explicit:
        path = Path(explicit)
    else:
        workspace = os.environ.get(
            "QUANTSOCIETY_WORKSPACE_DATA_ROOT",
            str(Path(__file__).resolve().parent.parent / "workspace_data"),
        )
        path = Path(workspace) / _DEFAULT_AUDIT_SUBPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_default(value: Any) -> Any:
    """#P1-65 审计 canonical serializer：datetime/date/Path/Decimal/Enum/numpy
    等非 JSON 对象统一转成可序列化表示——一个坏值不能让整条审计日志丢失。"""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    try:
        import pyarrow as pa

        if isinstance(value, pa.Scalar):
            return value.as_py()
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # NaN/Inf 不是合法 JSON 数
    return str(value)


def _should_audit_reads() -> bool:
    flag = os.environ.get("QUANT_AUDIT_READS", "").lower()
    if flag in {"0", "false", "no"}:
        return False
    if flag in {"1", "true", "yes"}:
        return True
    # #P1-66 production 判定与 query_budget/telemetry 统一（不能只看
    # QUANT_PRODUCTION_MODE，FACTOR_ENGINE_RUN_MODE=production 也要开 read audit）。
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        return os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
            "1",
            "true",
            "yes",
        }


def record(
    *,
    op: str,
    dataset: str,
    ok: bool,
    rows: int | None = None,
    mode: str | None = None,
    paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
    elapsed_ms: float | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    durable: bool = False,
) -> None:
    """记一条审计日志。

    - ``durable=False``（默认，observability）：失败**吞掉异常不影响主路径**
      （审计不应该把业务查询搞挂）。
    - ``durable=True``（#P1-final closure 22，published write / publish）：
      业务成功但审计写失败 ⇒ 抛 ``AuditWriteError``——权威发布必须有可证明的
      审计落盘（flush + fsync），不允许「发布了、审计悄悄没记」。

    参数：
        op: "read" | "write" | "publish"（read 默认不记录，需 QUANT_AUDIT_READS=true）
        dataset: 数据集注册名
        ok: 操作是否成功
        rows: 写/读的行数
        mode: write 模式（overwrite | append | upsert）
        paths: 实际落盘 / 读取的路径
        params: 参数化数据集的参数
        elapsed_ms: 耗时
        error: 失败时的错误消息
        extra: 任意额外字段（比如 upsert_on 列）
        durable: 是否要求 durable acknowledgement（权威写入/发布用）
    """
    if op == "read" and not _should_audit_reads():
        return

    record_obj: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "operator": resolve_operator(),
        "namespace": resolve_namespace(),
        "op": op,
        "dataset": dataset,
        "ok": ok,
    }
    if mode is not None:
        record_obj["mode"] = mode
    if rows is not None:
        record_obj["rows"] = rows
    if paths:
        record_obj["paths"] = paths
    if params:
        record_obj["params"] = params
    if elapsed_ms is not None:
        record_obj["elapsed_ms"] = round(elapsed_ms, 2)
    if error:
        # 审计日志里不放 traceback，避免泄露路径/栈；只记一行错误摘要
        record_obj["error"] = str(error)[:500]
    if extra:
        record_obj["extra"] = extra

    try:
        path = _resolve_audit_path()
        # #P1-65 canonical serializer：datetime/Path/Decimal/numpy/Enum 等非 JSON
        # 对象不再让 json.dumps 抛异常（否则整条 audit 静默丢失）。
        line = json.dumps(
            record_obj,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_canonical_default,
        )
        with _write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                if durable:
                    f.flush()
                    os.fsync(f.fileno())
    except Exception as exc:
        if durable:
            # #P1-final closure 22：权威写入/发布的审计必须 durable——业务成功但
            # 审计静默失败 = 合规证据缺失，向上抛而不是吞。
            from .exceptions import AuditWriteError

            raise AuditWriteError(
                f"审计日志写失败（op={op} dataset={dataset!r} ok={ok}），"
                "权威写入/发布需要 durable audit acknowledgement。"
                f"请检查 QUANT_AUDIT_LOG 路径可写。原因: {type(exc).__name__}: {exc}"
            ) from exc
        # 审计失败不能影响业务（observability）。PR5 会加告警路径。
        pass


class AuditTimer:
    """with AuditTimer() as t: ... t.elapsed_ms 拿耗时。

    用法（见 store.write_arrow）：
        with AuditTimer() as timer:
            do_the_work()
        record(..., elapsed_ms=timer.elapsed_ms, ok=True)
    """

    def __enter__(self) -> "AuditTimer":
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
