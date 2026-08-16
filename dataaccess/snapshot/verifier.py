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

    def _remote_identity_of(
        self, obj: ResolvedObject, meta: Mapping[str, Any]
    ) -> tuple[Any, Any, Any]:
        """从 object + 实时 HEAD 提取 (etag, version_id, content_length) 身份三元组。"""
        return (
            meta.get("etag") or obj.etag,
            meta.get("version_id") or obj.version_id,
            int(meta.get("content_length") or obj.content_length or 0),
        )

    def verify_before_execute(
        self,
        snapshot: ResolvedSourceSnapshot,
        *,
        paths: Sequence[str] | None = None,
    ) -> None:
        """执行前验证（P0-009/011 + R26-P0-015/016）。

        - snapshot 含 unresolved wildcard → 抛 ``SourceSnapshotUnavailable``；
        - exact object 里每个远端对象必须能 HEAD（etag/size/version_id），缺失 → fail-closed；
        - **production 下 exact object 必须携带可证明身份**（version_id OR
          etag+content_length OR 本地 size+mtime）——只有 URI 没有身份证据 →
          ``SourceSnapshotUnavailable``（R26-P0-015：URI+""+""+0 不是 content identity）；
        - remote 逐字段比较 etag / version_id / content_length；
        - 本地对象验证 path/size/mtime_ns。
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
            uri = str(obj.uri)
            if uri.startswith(("s3://", "cos://")):
                has_identity = bool(
                    obj.version_id or (obj.etag and obj.content_length is not None)
                )
                if strict and not has_identity:
                    raise SourceSnapshotUnavailable(
                        f"remote 对象 {uri} 无可证明身份（version_id/etag+content_length "
                        "均缺）：URI 不是 content identity（R26-P0-015，production fail-closed）。"
                    )
                if self.remote_meta_fn is not None:
                    meta = self._safe_remote_meta(uri)
                    if meta is None:
                        if strict:
                            raise SourceSnapshotUnavailable(
                                f"remote 对象 HEAD 失败（{uri}）：无法证明存在/版本，"
                                "production fail-closed。"
                            )
                        continue
                    # R2-P1：strict 身份验证不能把 HEAD 中缺失的已记录字段
                    # 当作相等。缺失 ETag/VersionId/size 无法证明对象没有变化。
                    meta_etag = meta.get("etag")
                    meta_version = meta.get("version_id")
                    meta_len = meta.get("content_length")
                    missing_identity = (
                        meta_version is None
                        if obj.version_id is not None
                        else (
                            (obj.etag is not None and meta_etag is None)
                            or (
                                obj.content_length is not None
                                and meta_len is None
                            )
                        )
                    )
                    if strict and missing_identity:
                        raise SourceSnapshotUnavailable(
                            f"remote 对象 HEAD 身份不完整（{uri}）：无法验证快照中"
                            "记录的 ETag/VersionId/content_length，production fail-closed。"
                        )
                    if obj.etag is not None and meta_etag is not None and obj.etag != meta_etag:
                        raise SourceSnapshotChanged(
                            f"remote 对象 ETag 变化（{uri}）：快照 {obj.etag} → "
                            f"{meta_etag}，source snapshot 已过期。"
                        )
                    if (
                        obj.version_id is not None
                        and meta_version is not None
                        and obj.version_id != meta_version
                    ):
                        raise SourceSnapshotChanged(
                            f"remote 对象 VersionId 变化（{uri}）：快照 {obj.version_id} → "
                            f"{meta_version}，source snapshot 已过期。"
                        )
                    if (
                        obj.content_length is not None
                        and meta_len is not None
                        and int(obj.content_length) != int(meta_len)
                    ):
                        raise SourceSnapshotChanged(
                            f"remote 对象 size 变化（{uri}）：{obj.content_length} → "
                            f"{meta_len}，source snapshot 已过期。"
                        )
            else:
                st = self._safe_local_stat(uri)
                if st is None:
                    if strict:
                        raise SourceSnapshotUnavailable(
                            f"本地对象缺失（{uri}）：无法证明存在，production fail-closed。"
                        )
                    continue
                # R26-P0-016：本地身份 = path + size + mtime_ns。
                if obj.content_length is not None and st.size != obj.content_length:
                    raise SourceSnapshotChanged(
                        f"本地对象 size 变化（{uri}）：{obj.content_length} → {st.size}"
                    )
                # R28-4：优先原始 ``mtime_ns`` **精确**比较（消除 2s 容差下
                # 「同大小文件替换」的漏网）；仅老构造（无 mtime_ns）回退
                # last_modified 的 2s 容差。
                if obj.mtime_ns is not None:
                    if st.mtime_ns != int(obj.mtime_ns):
                        raise SourceSnapshotChanged(
                            f"本地对象 mtime 变化（{uri}）：快照 mtime_ns="
                            f"{obj.mtime_ns} → 当前 {st.mtime_ns}，source snapshot 已过期。"
                        )
                elif obj.last_modified is not None and hasattr(
                    obj.last_modified, "timestamp"
                ):
                    # tz-aware UTC → epoch 秒 → 纳秒；与 stat().st_mtime_ns 比较。
                    # 绝不用本地时区 mktime（会把 UTC 解释成 local，产生假变化）。
                    recorded_ns = int(obj.last_modified.timestamp() * 1e9)
                    if abs(recorded_ns - st.mtime_ns) > 2_000_000_000:
                        raise SourceSnapshotChanged(
                            f"本地对象 mtime 变化（{uri}）：快照 {obj.last_modified} "
                            "→ 当前文件，source snapshot 已过期。"
                        )

    def verify_after_execute(
        self,
        snapshot: ResolvedSourceSnapshot,
        *,
        paths: Sequence[str] | None = None,
    ) -> None:
        """可选 final verify（长读后，P0-011 + R26-P0-016）。

        **重新比较** etag / version_id / content_length（不是只确认 HEAD 存在），
        并统一用 ``_effective_strict()``（不是原始 ``self.strict``）。
        """
        strict = self._effective_strict()
        if not strict or not snapshot.objects:
            return
        for obj in snapshot.objects:
            uri = str(obj.uri)
            if uri.startswith(("s3://", "cos://")):
                if self.remote_meta_fn is None:
                    continue
                meta = self._safe_remote_meta(uri)
                if meta is None:
                    raise SourceSnapshotUnavailable(
                        f"执行后 remote 对象 HEAD 失败（{uri}）"
                    )
                meta_etag = meta.get("etag")
                meta_version = meta.get("version_id")
                meta_len = meta.get("content_length")
                missing_identity = (
                    meta_version is None
                    if obj.version_id is not None
                    else (
                        (obj.etag is not None and meta_etag is None)
                        or (
                            obj.content_length is not None
                            and meta_len is None
                        )
                    )
                )
                if missing_identity:
                    raise SourceSnapshotUnavailable(
                        f"执行后 remote 对象 HEAD 身份不完整（{uri}）：无法验证快照中"
                        "记录的 ETag/VersionId/content_length。"
                    )
                if obj.etag is not None and meta_etag is not None and obj.etag != meta_etag:
                    raise SourceSnapshotChanged(
                        f"执行后 remote 对象 ETag 变化（{uri}）：{obj.etag} → {meta_etag}"
                    )
                if (
                    obj.version_id is not None
                    and meta_version is not None
                    and obj.version_id != meta_version
                ):
                    raise SourceSnapshotChanged(
                        f"执行后 remote 对象 VersionId 变化（{uri}）："
                        f"{obj.version_id} → {meta_version}"
                    )
                if (
                    obj.content_length is not None
                    and meta_len is not None
                    and int(obj.content_length) != int(meta_len)
                ):
                    raise SourceSnapshotChanged(
                        f"执行后 remote 对象 size 变化（{uri}）："
                        f"{obj.content_length} → {meta_len}"
                    )
            else:
                st = self._safe_local_stat(uri)
                if st is None:
                    raise SourceSnapshotUnavailable(
                        f"执行后本地对象缺失（{uri}）"
                    )
                if obj.content_length is not None and st.size != obj.content_length:
                    raise SourceSnapshotChanged(
                        f"执行后本地对象 size 变化（{uri}）："
                        f"{obj.content_length} → {st.size}"
                    )
                # R28-4：执行后同样精确比较 mtime_ns（不只 size）——否则
                # 「同大小文件替换」在 final verify 里也漏网。
                if obj.mtime_ns is not None:
                    if st.mtime_ns != int(obj.mtime_ns):
                        raise SourceSnapshotChanged(
                            f"执行后本地对象 mtime 变化（{uri}）：快照 mtime_ns="
                            f"{obj.mtime_ns} → 当前 {st.mtime_ns}，source snapshot 已过期。"
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
