"""R26-P0-004/017 —— 统一读执行链（ReadPipeline）。

把 R25 设计出来但未接线的机制（SourceSnapshotResolver / SnapshotVerifier /
GlobalResourceGovernor / QueryBudget v2）真正变成**一条不可绕过的执行链**：

    prepare:
        auth(调用方已做) → contract gate(调用方已做)
        → snapshot resolve（exact objects）
        → budget enforce（objects/bytes/remote/memory）
        → governor admit（reservation）
    execute:
        verify_before → backend execute → verify_after
    release:
        成功 / 异常 / 生成器关闭 都必须 release exactly once

``ReadPipeline`` 持有 instrumentation counters，供 T-R26-PIPE-001 断言每个
public path 各阶段 exactly once。生成器（stream / polars lazy）通过
``streaming_reservation`` 在生成器 finally 释放，杜绝 disconnect 泄漏。
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from data_access.core.exceptions import ResourceAdmissionError
from data_access.runtime.resource_governor import (
    GlobalResourceGovernor,
    ResourceReservation,
    get_global_governor,
)
from data_access.snapshot.source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)
from data_access.snapshot.verifier import SnapshotVerifier

logger = logging.getLogger("data_access.read_pipeline")


@dataclass
class PipelineCounters:
    """每个 public read path 的阶段计数（T-R26-PIPE-001 断言 exactly once）。"""

    auth: int = 0
    contract: int = 0
    snapshot: int = 0
    budget: int = 0
    governor: int = 0
    verify_before: int = 0
    execute: int = 0
    verify_after: int = 0
    release: int = 0

    def snapshot_dict(self) -> dict[str, int]:
        return {
            "auth": self.auth,
            "contract": self.contract,
            "snapshot": self.snapshot,
            "budget": self.budget,
            "governor": self.governor,
            "verify_before": self.verify_before,
            "execute": self.execute,
            "verify_after": self.verify_after,
            "release": self.release,
        }

    def assert_all_exactly_once(self) -> None:
        for key, val in self.snapshot_dict().items():
            assert val == 1, (
                f"pipeline counter {key}={val}，期望 exactly 1"
                "（R26 T-R26-PIPE-001：每个 public path 必须走同一执行链）"
            )


def resolved_snapshot_from_files(
    dataset: str,
    files: Sequence[Any],
    *,
    source: str = "file_manifest",
) -> ResolvedSourceSnapshot:
    """把 exact file manifest 转成 ResolvedSourceSnapshot（P0-004）。

    FileVersion（path/size/mtime_ns/etag/version_id/content_length）→
    ResolvedObject。这是本地/local-mirror 场景的 exact object set；remote
    unresolved wildcard 会保持 etag=None → production snapshot 校验拒绝。
    """
    objects: list[ResolvedObject] = []
    for f in files or ():
        path = str(getattr(f, "path", ""))
        mtime_ns = getattr(f, "mtime_ns", None)
        last_modified = None
        if mtime_ns:
            from datetime import datetime, timezone

            try:
                last_modified = datetime.fromtimestamp(
                    int(mtime_ns) / 1e9, tz=timezone.utc
                )
            except (OverflowError, OSError, ValueError):
                last_modified = None
        objects.append(
            ResolvedObject(
                uri=path,
                etag=getattr(f, "etag", None) or None,
                version_id=getattr(f, "version_id", None) or None,
                content_length=(
                    getattr(f, "content_length", None)
                    or getattr(f, "size", None)
                ) or None,
                last_modified=last_modified,
                # R28-4：原始纳秒 mtime 直接带上——执行后校验按 size+mtime_ns
                # 精确比较，不再走 datetime→float→ns 的二次换算与 2s 容差。
                mtime_ns=int(mtime_ns) if mtime_ns else None,
                source=source,
            )
        )
    digest = content_digest_of_objects(objects)
    return ResolvedSourceSnapshot(
        dataset=dataset,
        source_generation=None,
        objects=tuple(objects),
        content_digest=digest,
    )


class ReadPipeline:
    """统一读执行链（R26-P0-004）。Store 持有一个实例，所有 public read path
    复用。``governor`` / ``verifier`` / ``resolver`` 可注入（测试/部署）。"""

    def __init__(
        self,
        *,
        governor: GlobalResourceGovernor | None = None,
        verifier: SnapshotVerifier | None = None,
        resolver: Any = None,
    ) -> None:
        self._governor = governor or get_global_governor()
        self._verifier = verifier or SnapshotVerifier()
        self._resolver = resolver
        self.counters = PipelineCounters()

    # ---- prepare ----

    def resolve_snapshot(
        self,
        dataset: str,
        *,
        files: Sequence[Any] | None = None,
        paths: Sequence[str] | None = None,
        strict: bool | None = None,
        policy: str = "latest",
        pin_snapshot_id: str | None = None,
    ) -> ResolvedSourceSnapshot:
        """resolve exact source snapshot（P0-004）。优先 injected resolver。

        R29-P0 #199：resolver 是**主读链**——FileVersion snapshot 作为 resolver 的
        fallback 分支（``files`` 透传给 resolver 的 fallback_fn），不再在 pipeline
        层与 resolver 并行两套世界。
        """
        self.counters.snapshot += 1
        if self._resolver is not None:
            snap = self._resolver.resolve(
                dataset,
                paths=paths,
                files=files,
                policy=policy,
                pin_snapshot_id=pin_snapshot_id,
            )
            if snap is not None:
                return snap
        if files is not None:
            return resolved_snapshot_from_files(dataset, files)
        return ResolvedSourceSnapshot(
            dataset=dataset, objects=(), content_digest=""
        )

    def enforce_budget(
        self,
        budget: Any,
        *,
        snapshot: ResolvedSourceSnapshot | None = None,
        paths: Sequence[str] | None = None,
    ) -> None:
        """QueryBudget v2 强制（objects/bytes/remote，P0-010/018）。"""
        self.counters.budget += 1
        from data_access.read.query_budget import (
            enforce_remote_request_budget,
            enforce_scan_byte_budget,
            enforce_scan_object_budget,
        )

        if snapshot is not None:
            object_count = snapshot.object_count
            scan_bytes = snapshot.total_bytes
            remote_count = sum(
                1
                for o in snapshot.objects
                if str(o.uri).startswith(("s3://", "cos://"))
            )
        else:
            object_count = len(paths or ())
            scan_bytes = 0
            remote_count = 0
        enforce_scan_object_budget(budget, object_count=object_count)
        enforce_scan_byte_budget(budget, scan_bytes=scan_bytes)
        enforce_remote_request_budget(budget, remote_requests=remote_count)

    # ---- governor ----

    def admit(
        self,
        *,
        request_identity: str,
        principal_id: str = "unknown",
        estimated_scan_bytes: int = 0,
        estimated_memory: int = 0,
        remote_requests: int = 0,
    ) -> ResourceReservation:
        """Governor admission（prepare 阶段）。返回 reservation，execute 阶段 release。

        R26-P0-017：成功/异常/超时/客户端断开都必须 release exactly once。
        """
        self.counters.governor += 1
        res = ResourceReservation(
            query_id=request_identity,
            principal_id=principal_id,
            estimated_scan_bytes=estimated_scan_bytes,
            estimated_memory=estimated_memory,
            remote_requests=remote_requests,
        )
        self._governor.admit(res)
        return res

    def release_reservation(self, res: ResourceReservation | None) -> None:
        if res is None or res.released:
            return
        self._governor.release(res.query_id)
        res.released = True
        self.counters.release += 1

    @contextmanager
    def reservation(
        self,
        *,
        dataset: str,
        request_identity: str,
        principal_id: str = "unknown",
        estimated_scan_bytes: int = 0,
        estimated_memory: int = 0,
        remote_requests: int = 0,
    ):
        """Governor admission → yield reservation → 释放 exactly once。

        成功 / 异常都必须 release；``released`` 防重复 release。
        """
        res = self.admit(
            request_identity=request_identity,
            principal_id=principal_id,
            estimated_scan_bytes=estimated_scan_bytes,
            estimated_memory=estimated_memory,
            remote_requests=remote_requests,
        )
        try:
            yield res
        finally:
            self.release_reservation(res)

    def streaming_reservation(self, **kwargs) -> "StreamingReservation":
        """Streaming 用的 reservation 句柄（生成器 close 时释放，T-R26-RES-003）。"""
        self.counters.governor += 1
        res = ResourceReservation(
            query_id=kwargs.get("request_identity", f"stream-{id(kwargs)}"),
            principal_id=kwargs.get("principal_id", "unknown"),
            estimated_scan_bytes=kwargs.get("estimated_scan_bytes", 0),
            estimated_memory=kwargs.get("estimated_memory", 0),
            remote_requests=kwargs.get("remote_requests", 0),
        )
        self._governor.admit(res)
        return StreamingReservation(pipeline=self, reservation=res)

    # ---- verify ----

    def verify_before(self, snapshot: ResolvedSourceSnapshot) -> None:
        self.counters.verify_before += 1
        if snapshot is not None and snapshot.objects:
            self._verifier.verify_before_execute(snapshot)

    def verify_after(self, snapshot: ResolvedSourceSnapshot) -> None:
        self.counters.verify_after += 1
        if snapshot is not None and snapshot.objects:
            self._verifier.verify_after_execute(snapshot)

    # ---- instrumentation ----

    def reset_counters(self) -> None:
        self.counters = PipelineCounters()

    def counter_snapshot(self) -> dict[str, int]:
        return self.counters.snapshot_dict()


class StreamingReservation:
    """流式读的 governor reservation 句柄。

    用 ``@contextmanager`` 包住生成器：生成器被完整消费、中途 break、或
    客户端断开触发 GC/close 时，``finally`` 都会 release——active governor
    / remote slot 归零，不泄漏。
    """

    def __init__(self, *, pipeline: ReadPipeline, reservation: ResourceReservation) -> None:
        self._pipeline = pipeline
        self._reservation = reservation
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pipeline._governor.release(self._reservation.query_id)
        self._reservation.released = True
        self._pipeline.counters.release += 1

    @property
    def reservation(self) -> ResourceReservation:
        return self._reservation

    @contextmanager
    def guard(self):
        """包裹生成器主体：退出（含 close/异常）必 release。"""
        try:
            yield
        finally:
            self.release()
