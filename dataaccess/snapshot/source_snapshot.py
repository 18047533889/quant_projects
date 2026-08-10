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


@dataclass(frozen=True)
class ResolvedObject:
    """解析后的单个远程/本地对象（R25 §11）。

    - ``uri``            canonical s3:// / 本地 path
    - ``etag``           COS/S3 ETag（同 key overwrite 检测）
    - ``version_id``     COS 对象版本（VersionId，若 bucket 开启版本控制）
    - ``content_length`` 对象字节数
    - ``last_modified``  最后修改时间
    - ``source``         exact_list（COS LIST/HEAD）| source_manifest（上游权威）
    """

    uri: str
    etag: str | None = None
    version_id: str | None = None
    content_length: int | None = None
    last_modified: datetime | None = None
    source: str | None = None

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
            "source": self.source,
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
    resolved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def total_bytes(self) -> int:
        return sum(int(o.content_length or 0) for o in self.objects)

    @property
    def has_wildcard(self) -> bool:
        """对象集里是否含未解析通配（* / ? / {..}）——production 禁止。"""
        return any(
            ("*" in str(o.uri) or "?" in str(o.uri) or "{" in str(o.uri))
            for o in self.objects
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_generation": self.source_generation,
            "object_count": self.object_count,
            "total_bytes": self.total_bytes,
            "content_digest": self.content_digest,
            "objects": [o.to_dict() for o in self.objects],
            "resolved_at": self.resolved_at.isoformat(),
        }


def content_digest_of_objects(objects: Sequence[ResolvedObject]) -> str:
    """R25 §31：content_set_digest。

    ``hash(sorted(object_key, etag/version_id, content_length))``——本地 immutable
    generation 用 path+checksum；远程用 etag/versionId+length。这是 reproducibility
    identity：同 key 被 overwrite（etag 变）→ digest 变 → snapshot 变。
    """
    entries = []
    for o in sorted(objects, key=lambda x: str(x.uri)):
        ident = (
            str(o.uri),
            str(o.etag or ""),
            str(o.version_id or ""),
            str(int(o.content_length or 0)),
        )
        entries.append("|".join(ident))
    if not entries:
        return ""
    text = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
