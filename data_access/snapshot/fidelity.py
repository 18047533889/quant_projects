"""R40 #52 —— SnapshotFidelity：source snapshot 的证据保真度层级。

problem
-------
``ResolvedSourceSnapshot`` 之前没有任何「保真度」概念——一个从 ``FALLBACK``
（本地 FileVersion 兜底）解析出的 snapshot 与从 ``PUBLISHER_MANIFEST``（上游权威
清单）解析出的 snapshot 在类型上完全一样。production 读路径无法区分「我读的是
上游权威版本」还是「我读的是本地兜底文件」，无法对数据身份做出强度断言。

design
------
:class:`SnapshotFidelity` 是有序层级（数值越大越强）：

    PUBLISHER_MANIFEST > REMOTE_VERSION_ID > CONTENT_HASH
        > LOCAL_STAT > FALLBACK > UNKNOWN

- ``PUBLISHER_MANIFEST``  上游 publisher 权威 manifest（generation + exact objects）；
- ``REMOTE_VERSION_ID``   远端 LIST/HEAD 精确对象 + etag/version_id（可证明远端身份）；
- ``CONTENT_HASH``        本地对象 content digest（checksum 可证明内容身份）；
- ``LOCAL_STAT``          本地文件 stat（size/mtime_ns）；
- ``FALLBACK``            FileVersion snapshot 兜底（无远端身份证明）；
- ``UNKNOWN``             无法判定。

production 语义（#52）：production 模式拒绝 fidelity 低于 ``REMOTE_VERSION_ID``
的 snapshot——本地 stat / fallback 无法证明远端数据身份，不能作为 production 的
权威来源。
"""
from __future__ import annotations

import enum
from typing import Any


class SnapshotFidelity(enum.Enum):
    """source snapshot 保真度层级（数值越大越强）。"""

    PUBLISHER_MANIFEST = 5
    REMOTE_VERSION_ID = 4
    CONTENT_HASH = 3
    LOCAL_STAT = 2
    FALLBACK = 1
    UNKNOWN = 0

    def __ge__(self, other: "SnapshotFidelity") -> bool:
        if isinstance(other, SnapshotFidelity):
            return self.value >= other.value
        return NotImplemented

    def __lt__(self, other: "SnapshotFidelity") -> bool:
        if isinstance(other, SnapshotFidelity):
            return self.value < other.value
        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        return {"fidelity": self.name, "rank": self.value}


#: production 模式要求的最低保真度。
PRODUCTION_MIN_FIDELITY = SnapshotFidelity.REMOTE_VERSION_ID


def production_snapshot_fidelity_ok(fidelity: SnapshotFidelity) -> bool:
    """production 是否接受该保真度（≥ REMOTE_VERSION_ID 才通过）。"""
    if not isinstance(fidelity, SnapshotFidelity):
        return False
    return fidelity >= PRODUCTION_MIN_FIDELITY


def assert_production_snapshot_fidelity(
    fidelity: SnapshotFidelity,
    *,
    dataset: str = "",
    production: bool | None = None,
) -> None:
    """production 下 fidelity 低于 REMOTE_VERSION_ID → raise ``SnapshotBuildError``。

    ``production`` 缺省按 ``is_strict_semantics()`` 判定。research 模式只记录
    warning（不拒绝）。
    """
    from data_access.core.exceptions import SnapshotBuildError
    from data_access.read.query_budget import is_strict_semantics

    if production is None:
        production = is_strict_semantics()
    if not production:
        return
    if not production_snapshot_fidelity_ok(fidelity):
        raise SnapshotBuildError(
            f"snapshot fidelity 不足（R40 #52）：dataset={dataset or '<unknown>'} "
            f"fidelity={getattr(fidelity, 'name', fidelity)!r} 低于 production 要求 "
            f"{PRODUCTION_MIN_FIDELITY.name}。本地 stat / fallback / UNKNOWN 身份不能"
            "作为 production 权威 source snapshot。"
        )


__all__ = [
    "SnapshotFidelity",
    "PRODUCTION_MIN_FIDELITY",
    "production_snapshot_fidelity_ok",
    "assert_production_snapshot_fidelity",
]
