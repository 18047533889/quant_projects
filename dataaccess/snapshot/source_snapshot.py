"""R25 P0-009/010/013 —— ResolvedObject / ResolvedSourceSnapshot。

production remote 物理规划必须先得到 **exact objects**（COS LIST/HEAD 或上游
source manifest），而不是把 wildcard URI 当 1 个 FileVersion。预算（max_scan_objects /
max_scan_bytes）在 execution 前按真实对象统计。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from data_access.snapshot.fidelity import SnapshotFidelity


@dataclass(frozen=True)
class ObjectIdentity:
    """R32-P0-039: Formally model checksum/content hash.

    Object identity based on content hash/checksum when available.

    - ``kind``: "md5", "sha256", "etag", "version_id", "mtime_ns"
    - ``algorithm``: hash algorithm if applicable
    - ``value``: hash/checksum value
    - ``size``: object size in bytes
    """

    kind: str
    value: str
    size: int | None = None
    algorithm: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "size": self.size,
            "algorithm": self.algorithm,
        }


@dataclass(frozen=True)
class ResolvedObject:
    """解析后的单个远程/本地对象（R25 §11）。

    - ``uri``            canonical s3:// / 本地 path
    - ``etag``           COS/S3 ETag（同 key overwrite 检测）
    - ``version_id``     COS 对象版本（VersionId，若 bucket 开启版本控制）
    - ``content_length`` 对象字节数
    - ``last_modified``  最后修改时间
    - ``mtime_ns``       **原始纳秒 mtime**（R28-4：不再 datetime→float→ns 绕一圈，
                         执行前/后直接按 ``size + mtime_ns`` 精确比较，消除 2s 容差
                         下「同大小文件替换」的漏网）
    - ``source``         exact_list（COS LIST/HEAD）| source_manifest（上游权威）
    - ``_identity``      R32-P0-039: ObjectIdentity（checksum/content hash）- internal storage
    - ``checksum``       Content checksum (e.g., MD5, SHA256)
    - ``checksum_algorithm`` Algorithm used for checksum
    """

    uri: str
    etag: str | None = None
    version_id: str | None = None
    content_length: int | None = None
    last_modified: datetime | None = None
    mtime_ns: int | None = None
    source: str | None = None
    _identity: ObjectIdentity | None = field(default=None, repr=False)
    checksum: str | None = None
    checksum_algorithm: str | None = None

    @property
    def identity(self) -> ObjectIdentity | None:
        """R32-P0-039: Extract ObjectIdentity from available fields.

        Priority: checksum > etag/version_id > mtime_ns
        """
        if self._identity:
            return self._identity

        # Content checksum (highest priority for local files)
        if self.checksum and self.checksum_algorithm:
            return ObjectIdentity(
                kind="checksum",
                value=self.checksum,
                size=self.content_length,
                algorithm=self.checksum_algorithm,
            )

        # Remote object with etag
        if self.etag:
            return ObjectIdentity(
                kind="etag",
                value=self.etag,
                size=self.content_length,
            )

        # Remote object with version_id
        if self.version_id:
            return ObjectIdentity(
                kind="version_id",
                value=self.version_id,
                size=self.content_length,
            )

        # Local object with mtime_ns
        if self.mtime_ns is not None:
            return ObjectIdentity(
                kind="mtime_ns",
                value=str(self.mtime_ns),
                size=self.content_length,
            )

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "etag": self.etag,
            "version_id": self.version_id,
            "content_length": self.content_length,
            "last_modified": (
                self.last_modified.isoformat()
                if getattr(self.last_modified, "isoformat", None)
                else self.last_modified
            ),
            "mtime_ns": self.mtime_ns,
            "source": self.source,
            "identity": self.identity.to_dict() if self.identity else None,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
        }


@dataclass(frozen=True)
class ResolvedSourceSnapshot:
    """一个可证明的 source snapshot（R25 §11 / INV-02）。

    - ``dataset``            数据集名
    - ``source_generation``  上游 generation（publisher manifest 提供；None = 未提供）
    - ``objects``            **exact** object set（不允许 unresolved wildcard 占位）
    - ``content_digest``     ``hash(sorted(object_key, etag/version_id, content_length))``
                             —— reproducibility identity（R25 §31）
    - ``resolved_at``        解析时间
    """

    dataset: str
    source_generation: str | None = None
    objects: tuple[ResolvedObject, ...] = ()
    content_digest: str = ""
    # R40 #52：证据保真度层级（PUBLISHER_MANIFEST > REMOTE_VERSION_ID >
    # CONTENT_HASH > LOCAL_STAT > FALLBACK > UNKNOWN）。production 拒绝低于
    # REMOTE_VERSION_ID 的 snapshot。
    fidelity: SnapshotFidelity = SnapshotFidelity.UNKNOWN
    resolved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def total_bytes(self) -> int | None:
        """R32-P0-040: Distinguish unknown from 0.

        If any object has unknown content_length (None), return None instead of
        treating it as 0. Resource admission must use conservative bounds for
        unknown sizes.
        """
        if any(o.content_length is None for o in self.objects):
            return None
        return sum(int(o.content_length) for o in self.objects)

    @property
    def has_wildcard(self) -> bool:
        """对象集里是否含未解析通配（* / ? / [ ] / {..} / **）——production 禁止。

        R26-P0-015：``[0-9]`` 字符类也是 unresolved glob，必须被 gate 拒绝。
        """
        return any(
            any(m in str(o.uri) for m in ("*", "?", "[", "]", "{"))
            for o in self.objects
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_generation": self.source_generation,
            "object_count": self.object_count,
            "total_bytes": self.total_bytes,
            "content_digest": self.content_digest,
            "fidelity": getattr(self.fidelity, "name", str(self.fidelity)),
            "objects": [o.to_dict() for o in self.objects],
            "resolved_at": self.resolved_at.isoformat(),
        }


def content_digest_of_objects(objects: Sequence[ResolvedObject]) -> str:
    """R25 §31 + R32-P0-031/038: content_set_digest with proper identity.

    ``hash(sorted(object_key, etag/version_id, content_length, mtime_ns))``

    - Local immutable generation uses path+checksum (or size+mtime_ns)
    - Remote uses etag/versionId+length
    - Empty object set has canonical non-empty digest (R32-P0-031)
    - Local mtime_ns included for same-path replacement detection (R32-P0-038)

    This is reproducibility identity: same key overwrite (etag change) →
    digest change → snapshot change.
    """
    entries = []
    for o in sorted(objects, key=lambda x: str(x.uri)):
        # R32-P0-038: Include mtime_ns for local files
        ident = (
            str(o.uri),
            str(o.etag or ""),
            str(o.version_id or ""),
            str(int(o.content_length or 0)),
            str(int(o.mtime_ns or 0)),
        )
        entries.append("|".join(ident))
    # R32-P0-031: Empty set has canonical digest, not magic empty string
    if not entries:
        empty_payload = json.dumps([], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(empty_payload.encode("utf-8")).hexdigest()[:32]
    text = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
