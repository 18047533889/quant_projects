"""R25 Source Snapshot —— 远程对象解析 / snapshot 验证 / mirror 源身份。

    - ``source_snapshot``  ResolvedObject / ResolvedSourceSnapshot（P0-009/010/013）
    - ``resolver``         SourceSnapshotResolver（P0-012/013/030）
    - ``verifier``         SnapshotVerifier（P0-011 全 backend 统一验证）

核心原则（INV-02 / INV-11）：
    - production remote 不允许 unresolved wildcard 直接进 executor；
    - local mirror freshness 必须比较远端 source identity（ETag/generation），
      不只本地 checksum；
    - auto hybrid 必须同 source generation，无法证明则全部 remote。
"""

from .source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)
from .resolver import SourceSnapshotResolver, resolve_source_snapshot
from .verifier import SnapshotVerifier, verify_snapshot_before_execute

__all__ = [
    "ResolvedObject",
    "ResolvedSourceSnapshot",
    "content_digest_of_objects",
    "SourceSnapshotResolver",
    "resolve_source_snapshot",
    "SnapshotVerifier",
    "verify_snapshot_before_execute",
]
