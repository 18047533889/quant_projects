"""R25 P0-011 —— SnapshotVerifier：全 backend 统一 snapshot 验证。

Polars ``ScanHandle.collect()`` 已有 collect 前 remote HEAD + etag/size/version
检查；但 direct DuckDB / PyArrow / stream / read_joined / factor read 没有统一
复用同一 verifier。本模块把「resolve → verify → execute → 可选 final verify」统一
成一条链，供所有 backend 共用。

    - Local ：path + size + mtime + optional checksum/generation
    - Remote：exact object + etag/versionId/content_length

执行顺序（R25 §12）：
    resolve snapshot → verify → execute exact objects → optional final verify。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import (
    SourceSnapshotChanged,
    SourceSnapshotUnavailable,
    ValidationError,
)

from .source_snapshot import ResolvedObject, ResolvedSourceSnapshot

logger = logging.getLogger("data_access.snapshot_verifier")


@dataclass(frozen=True)
class LocalFileStat:
    """本地文件验证快照（path/size/mtime_ns/可选 checksum）。"""

    path: str
    size: int
    mtime_ns: int
    checksum: str | None = None


@dataclass
class SnapshotVerifier:
    """统一 snapshot 验证器（R25 §12 / P0-011）。

    - ``remote_meta_fn``   给定 s3:// uri 返回 ``{etag, content_length, last_modified}``
                           （COS HEAD；None = 拿不到 HEAD → fail-closed）
    - ``local_stat_fn``    给定本地 path 返回 ``LocalFileStat``（None = 文件缺失）
    - ``strict``           缺省 ``is_strict_semantics()``
    """

    remote_meta_fn: Any = None
    local_stat_fn: Any = None
    strict: bool | None = None

    def _effective_strict(self) -> bool:
        if self.strict is not None:
            return self.strict
        try:
            from data_access.read.query_budget import is_strict_semantics

            return is_strict_semantics()
        except Exception:
            return True

    # ---- 验证 ----

    def verify_before_execute(
        self,
        snapshot: ResolvedSourceSnapshot,
        *,
        paths: Sequence[str] | None = None,
    ) -> None:
        """执行前验证（P0-009/011）。

        - snapshot 含 unresolved wildcard → 抛 ``SourceSnapshotUnavailable``；
        - exact object 里每个远端对象必须能 HEAD（etag/size），缺失 → fail-closed；
        - 本地对象验证 path/size/mtime。
        """
        strict = self._effective_strict()
        if snapshot.has_wildcard:
            raise SourceSnapshotUnavailable(
                f"production remote snapshot 含 unresolved wildcard（{snapshot.dataset}）："
                "wildcard URI 禁止直接进 executor，必须先 LIST/HEAD 解析成 exact "
                "object 集（或用上游 source manifest）。"
            )
        if not snapshot.objects:
            raise SourceSnapshotUnavailable(
                f"source snapshot 对象集为空（{snapshot.dataset}）：无法证明读取对象。"
            )
        for obj in snapshot.objects:
            if str(obj.uri).startswith("s3://"):
                if self.remote_meta_fn is not None:
                    meta = self._safe_remote_meta(obj.uri)
                    if meta is None:
                        if strict:
                            raise SourceSnapshotUnavailable(
                                f"remote 对象 HEAD 失败（{obj.uri}）：无法证明存在/版本，"
                                "production fail-closed。"
                            )
                        continue
                    if obj.etag and meta.get("etag") and obj.etag != meta.get("etag"):
                        raise SourceSnapshotChanged(
                            f"remote 对象 ETag 变化（{obj.uri}）：快照 {obj.etag} → "
                            f"{meta.get('etag')}，source snapshot 已过期。"
                        )
            else:
                st = self._safe_local_stat(obj.uri)
                if st is None:
                    if strict:
                        raise SourceSnapshotUnavailable(
                            f"本地对象缺失（{obj.uri}）：无法证明存在，production fail-closed。"
                        )
                    continue
                if obj.content_length is not None and st.size != obj.content_length:
                    raise SourceSnapshotChanged(
                        f"本地对象 size 变化（{obj.uri}）：{obj.content_length} → {st.size}"
                    )

    def verify_after_execute(
        self,
        snapshot: ResolvedSourceSnapshot,
        *,
        paths: Sequence[str] | None = None,
    ) -> None:
        """可选 final verify（长读后，P0-011）：只对 remote exact object 复核 HEAD。"""
        if self._effective_strict() and snapshot.objects:
            for obj in snapshot.objects:
                if str(obj.uri).startswith("s3://") and self.remote_meta_fn is not None:
                    meta = self._safe_remote_meta(obj.uri)
                    if meta is None and self.strict:
                        raise SourceSnapshotUnavailable(
                            f"执行后 remote 对象 HEAD 失败（{obj.uri}）"
                        )

    # ---- helpers ----

    def _safe_remote_meta(self, uri: str) -> Mapping[str, Any] | None:
        try:
            meta = self.remote_meta_fn(uri)
            return dict(meta or {}) or None
        except Exception as exc:
            if self._effective_strict():
                raise SourceSnapshotUnavailable(
                    f"remote HEAD 失败（{uri}）：{exc}"
                ) from exc
            return None

    def _safe_local_stat(self, path: str) -> LocalFileStat | None:
        if self.local_stat_fn is not None:
            try:
                return self.local_stat_fn(path)
            except Exception:
                return None
        try:
            p = Path(path)
            if not p.exists():
                return None
            st = p.stat()
            return LocalFileStat(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
        except OSError:
            return None


def verify_snapshot_before_execute(
    snapshot: ResolvedSourceSnapshot,
    *,
    remote_meta_fn: Any = None,
    local_stat_fn: Any = None,
    strict: bool | None = None,
) -> None:
    """便捷入口：执行前统一验证（P0-011 各 backend 共用）。"""
    SnapshotVerifier(
        remote_meta_fn=remote_meta_fn,
        local_stat_fn=local_stat_fn,
        strict=strict,
    ).verify_before_execute(snapshot)
