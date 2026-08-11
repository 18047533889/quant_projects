# -*- coding: utf-8 -*-
"""R38 P0-032/033（§13）：SpillStore —— 真实 spill（write / checksum / reload / delete）。

R36 的 ``GovernedBufferStore.spill()`` 只是 ``release(key)``（drop，不是 spill）。
本模块实现真正的 spill：
    - :meth:`spill`：把 buffer 写 parquet，返回 :class:`SpillRef`（path + checksum +
      bytes + dtype/schema + generation + source_identity，§P0-032 必须字段）；
    - :meth:`reload`：读回（带 checksum 校验，§R38_SPILL_CHECKSUM 失败检测）；
    - :meth:`delete` / :meth:`cleanup_execution`：按 execution 清理；
    - :meth:`should_spill`：spill-vs-recompute 成本决策（§P0-033：低成本 elementwise
      倾向 recompute，高成本 PCA/GARCH 倾向 spill）。
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpillRef:
    """一次 spill 的完整引用（§P0-032 必须字段）。"""

    path: str
    checksum: str
    bytes: int
    dtype: str = ""
    schema: str = ""
    generation_id: str = ""
    source_identity: str = ""
    execution_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "checksum": self.checksum[:12],
            "bytes": self.bytes,
            "dtype": self.dtype,
            "generation_id": self.generation_id,
            "source_identity": self.source_identity,
            "execution_id": self.execution_id,
        }


@dataclass(frozen=True)
class SpillDecision:
    """spill-vs-recompute 成本决策（§P0-033）。"""

    should_spill: bool
    recompute_cost: float
    reload_cost: float
    io_pressure_cost: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_spill": self.should_spill,
            "recompute_cost": round(self.recompute_cost, 3),
            "reload_cost": round(self.reload_cost, 3),
            "io_pressure_cost": round(self.io_pressure_cost, 3),
            "reason": self.reason,
        }


class SpillStore:
    """真实 spill 存储（parquet + sha256 checksum）。"""

    def __init__(self, *, root_dir: str | None = None) -> None:
        self._lock = threading.RLock()
        self._root = root_dir or os.path.join(
            os.environ.get("FACTOR_ENGINE_CACHE_DIR", "").strip() or ".",
            "_r38_spill_store",
        )
        os.makedirs(self._root, exist_ok=True)
        self._refs: dict[str, SpillRef] = {}
        self._events: list[str] = []

    @property
    def root_dir(self) -> str:
        return self._root

    def spill(
        self,
        value: Any,
        *,
        key: str = "",
        generation_id: str = "",
        source_identity: str = "",
        execution_id: str = "",
    ) -> SpillRef:
        """把 buffer 写盘并返回 SpillRef（真实 spill）。"""
        import pandas as pd

        # 估算字节 + dtype/schema（写前记录，供 reload 校验）。
        try:
            from runtime.resource_governor import estimate_object_bytes

            n_bytes = max(0, int(estimate_object_bytes(value)))
        except Exception:
            n_bytes = 0
        dtype = "float64"
        schema = ""
        if hasattr(value, "dtype"):
            try:
                dtype = str(value.dtype)
            except Exception:
                dtype = "unknown"
        if hasattr(value, "index") and hasattr(value, "name"):
            schema = f"series:{value.index.nlevels}:{getattr(value, 'name', '')}"
        path = os.path.join(self._root, f"{key or 'buf'}_{uuid.uuid4().hex[:10]}.parquet")
        try:
            if hasattr(value, "to_frame"):
                value.to_frame("__r38_spill__").to_parquet(path)
            else:
                pd.DataFrame(value).to_parquet(path)
        except Exception as exc:  # noqa: BLE001
            self._events.append(f"spill_write_failed:{type(exc).__name__}")
            raise
        checksum = self._checksum_file(path)
        ref = SpillRef(
            path=path,
            checksum=checksum,
            bytes=n_bytes,
            dtype=dtype,
            schema=schema,
            generation_id=generation_id,
            source_identity=source_identity,
            execution_id=execution_id,
        )
        with self._lock:
            self._refs[ref.path] = ref
        self._events.append(f"spilled:{os.path.basename(path)}:{n_bytes}")
        return ref

    def reload(self, ref: SpillRef) -> Any:
        """读回 buffer 并校验 checksum（§R38_SPILL_CHECKSUM 失败检测）。"""
        import pandas as pd

        if not os.path.exists(ref.path):
            raise FileNotFoundError(f"spill reload: missing {ref.path}")
        actual = self._checksum_file(ref.path)
        if actual != ref.checksum:
            raise RuntimeError(
                f"spill checksum mismatch: {os.path.basename(ref.path)} "
                f"expected {ref.checksum[:12]} got {actual[:12]} (corrupted spill)"
            )
        frame = pd.read_parquet(ref.path)
        if frame.shape[1] == 1:
            return frame.iloc[:, 0]
        return frame

    def delete(self, ref: SpillRef) -> bool:
        """删除 spill 文件。"""
        try:
            if os.path.exists(ref.path):
                os.remove(ref.path)
            with self._lock:
                self._refs.pop(ref.path, None)
            return True
        except OSError:
            return False

    def cleanup_execution(self, execution_id: str) -> int:
        """清理某次 execution 的全部 spill 文件。"""
        removed = 0
        with self._lock:
            for path, ref in list(self._refs.items()):
                if ref.execution_id == execution_id:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    self._refs.pop(path, None)
                    removed += 1
        return removed

    def verify(self, ref: SpillRef) -> bool:
        try:
            return os.path.exists(ref.path) and self._checksum_file(ref.path) == ref.checksum
        except Exception:
            return False

    def should_spill(
        self,
        *,
        recompute_cost_ms: float,
        reload_cost_ms: float,
        io_pressure_score: float = 0.0,
        reuse_count: int = 1,
    ) -> SpillDecision:
        """§P0-033：spill if reload_cost + io_pressure_cost < recompute_cost。

        - 低成本 elementwise（recompute 便宜）→ drop + recompute；
        - 高成本（PCA / GARCH fit / 大 source block）→ spill。
        """
        io_cost = reload_cost_ms * (1.0 + 2.0 * io_pressure_score)
        should = (io_cost + 1.0) < (recompute_cost_ms * 0.8 * max(1, reuse_count))
        reason = (
            f"spill: io_cost={io_cost:.0f}ms < recompute*0.8={recompute_cost_ms * 0.8:.0f}ms"
            if should
            else f"recompute: io_cost={io_cost:.0f}ms >= recompute*0.8={recompute_cost_ms * 0.8:.0f}ms"
        )
        return SpillDecision(
            should_spill=should,
            recompute_cost=recompute_cost_ms,
            reload_cost=reload_cost_ms,
            io_pressure_cost=io_cost - reload_cost_ms,
            reason=reason,
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "root_dir": self._root,
                "refs": len(self._refs),
                "bytes": sum(r.bytes for r in self._refs.values()),
                "events": self._events[-20:],
            }

    @staticmethod
    def _checksum_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


def _cleanup_root(root_dir: str) -> None:
    try:
        for name in os.listdir(root_dir):
            os.remove(os.path.join(root_dir, name))
        os.rmdir(root_dir)
    except OSError:
        pass
