"""R25 P0-012/013/030 —— SourceSnapshotResolver。

把「读某个 dataset 的 snapshot」统一成一条解析链：
    - policy：``latest`` / ``pin(snapshot_id)`` / ``fail_if_changed``
    - 优先 publisher source manifest（generation + exact object set）；
    - 次选 exact object list + etag（COS LIST/HEAD）；
    - 禁止 wildcard-only（production 下 unresolved wildcard → fail-closed）。

auto hybrid 先 ``resolve()`` 得到 target snapshot，再按对象身份选 local/remote
（P0-013：local 对象匹配 target 则 local，否则 remote exact target；最后
``assert all(parts.source_generation == target.generation)``）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from data_access.core.exceptions import (
    SourceSnapshotUnavailable,
    ValidationError,
)

from .source_snapshot import (
    ResolvedObject,
    ResolvedSourceSnapshot,
    content_digest_of_objects,
)

logger = logging.getLogger("data_access.source_snapshot")


@dataclass(frozen=True)
class SourceManifest:
    """上游 publisher 的 source manifest（R25 §13 / §31）。

    ``{source_generation, objects: [{key, etag, size}]}``。DataAccess mirror 下载
    generation G，本地 manifest 绑定 G。缺失时 DataAccess 必须明确
    ``external_source_identity_unverified``（§32）。
    """

    source_generation: str
    objects: tuple[ResolvedObject, ...] = ()


def parse_source_manifest(raw: Any) -> SourceManifest | None:
    """解析 publisher source manifest（dict 或 JSON 字符串）。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    gen = raw.get("source_generation") or raw.get("generation")
    if not gen:
        return None
    objs: list[ResolvedObject] = []
    for entry in raw.get("objects") or ():
        if isinstance(entry, str):
            objs.append(ResolvedObject(uri=entry))
            continue
        if isinstance(entry, dict):
            objs.append(
                ResolvedObject(
                    uri=str(entry.get("key") or entry.get("uri") or ""),
                    etag=entry.get("etag"),
                    version_id=entry.get("version_id"),
                    content_length=entry.get("size") or entry.get("content_length"),
                    source="source_manifest",
                )
            )
    return SourceManifest(source_generation=str(gen), objects=tuple(objs))


class SourceSnapshotResolver:
    """解析 dataset 的权威 source snapshot（R25 §30）。

    - ``list_objects_fn``  ：给定 s3:// 前缀 → ``list[ResolvedObject]``（COS LIST，
                            返回 exact objects）。None = 不执行 LIST。
    - ``head_object_fn``   ：给定 s3:// uri → ResolvedObject（COS HEAD）。None = 不 HEAD。
    - ``source_manifest_fn``：给定 dataset → SourceManifest | None（publisher 提供）。
    """

    def __init__(
        self,
        *,
        list_objects_fn: Callable[[str], Sequence[ResolvedObject]] | None = None,
        head_object_fn: Callable[[str], ResolvedObject | None] | None = None,
        source_manifest_fn: Callable[[str], Any | None] | None = None,
        strict: bool | None = None,
    ) -> None:
        self._list_objects_fn = list_objects_fn
        self._head_object_fn = head_object_fn
        self._source_manifest_fn = source_manifest_fn
        self._strict = strict

    def _effective_strict(self) -> bool:
        if self._strict is not None:
            return self._strict
        try:
            from data_access.read.query_budget import is_strict_semantics

            return is_strict_semantics()
        except Exception:
            return True

    def resolve(
        self,
        dataset: str,
        *,
        policy: str = "latest",
        pin_snapshot_id: str | None = None,
        paths: Sequence[str] | None = None,
    ) -> ResolvedSourceSnapshot:
        """解析 dataset 的 source snapshot。

        解析顺序（R25 §30）：
            1. publisher source manifest（最权威）；
            2. exact object list（COS LIST，若提供 list_objects_fn）；
            3. HEAD exact objects（paths 不含通配时逐对象 HEAD）；
            4. 都无法解析 exact object set 且 strict → ``SourceSnapshotUnavailable``
               （production 禁止 wildcard-only snapshot）。
        """
        strict = self._effective_strict()

        # 1) publisher source manifest（最优）。
        manifest = None
        if self._source_manifest_fn is not None:
            try:
                manifest = parse_source_manifest(self._source_manifest_fn(dataset))
            except Exception as exc:
                if strict:
                    raise SourceSnapshotUnavailable(
                        f"source manifest 解析失败（{dataset}）：{exc}"
                    ) from exc

        objects: tuple[ResolvedObject, ...] = ()
        generation = None
        if manifest is not None:
            objects = manifest.objects
            generation = manifest.source_generation
            logger.info(
                "source_snapshot: %s via source_manifest gen=%s objects=%d",
                dataset, generation, len(objects),
            )

        # 2) exact object list（COS LIST）。
        if not objects and self._list_objects_fn is not None and paths:
            prefix = _common_prefix(paths)
            if prefix:
                try:
                    listed = list(self._list_objects_fn(prefix))
                    objects = tuple(listed)
                except Exception as exc:
                    if strict:
                        raise SourceSnapshotUnavailable(
                            f"COS LIST 失败（{prefix}）：{exc}"
                        ) from exc

        # 3) HEAD exact objects（paths 不含通配）。
        if not objects and self._head_object_fn is not None and paths:
            head_objs: list[ResolvedObject] = []
            for p in paths:
                if "*" in p or "?" in p or "{" in p:
                    continue  # 通配交给 LIST / manifest
                try:
                    h = self._head_object_fn(p)
                except Exception as exc:
                    if strict:
                        raise SourceSnapshotUnavailable(
                            f"COS HEAD 失败（{p}）：{exc}"
                        ) from exc
                    h = None
                if h is not None:
                    head_objs.append(h)
            objects = tuple(head_objs)

        # 4) 仍无法得到 exact object set：strict 下 wildcard-only 禁止进 executor。
        if not objects:
            if strict:
                raise SourceSnapshotUnavailable(
                    f"dataset={dataset!r} 无法解析 exact source snapshot（无 publisher "
                    "manifest、无 exact object list、无 HEAD 结果）。production "
                    "fail-closed：wildcard URI 不构成可证明 snapshot。"
                )
            return ResolvedSourceSnapshot(
                dataset=dataset,
                source_generation=None,
                objects=(),
                content_digest="",
            )

        digest = content_digest_of_objects(objects)
        snap = ResolvedSourceSnapshot(
            dataset=dataset,
            source_generation=generation,
            objects=objects,
            content_digest=digest,
        )
        if policy == "pin" and pin_snapshot_id and pin_snapshot_id != digest:
            raise ValidationError(
                f"dataset={dataset!r} 当前 snapshot digest={digest} != pinned "
                f"{pin_snapshot_id}（policy=pin）——拒绝读取不一致版本。"
            )
        return snap


def _common_prefix(paths: Sequence[str]) -> str | None:
    """从一组 s3:// path 提取公共前缀（去掉尾部 glob）。"""
    cleaned = []
    for p in paths:
        s = str(p)
        for marker in ("*", "? ", "{"):
            idx = s.find(marker)
            if idx >= 0:
                s = s[:idx]
        cleaned.append(s.rstrip("/"))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    common = cleaned[0]
    for c in cleaned[1:]:
        while not c.startswith(common):
            common = common[:-1]
            if not common:
                return None
    return common


def resolve_source_snapshot(
    dataset: str,
    *,
    policy: str = "latest",
    pin_snapshot_id: str | None = None,
    paths: Sequence[str] | None = None,
    list_objects_fn: Callable[[str], Sequence[ResolvedObject]] | None = None,
    head_object_fn: Callable[[str], ResolvedObject | None] | None = None,
    source_manifest_fn: Callable[[str], Any | None] = None,
    strict: bool | None = None,
) -> ResolvedSourceSnapshot:
    """便捷入口：解析 dataset 的权威 source snapshot。"""
    return SourceSnapshotResolver(
        list_objects_fn=list_objects_fn,
        head_object_fn=head_object_fn,
        source_manifest_fn=source_manifest_fn,
        strict=strict,
    ).resolve(
        dataset,
        policy=policy,
        pin_snapshot_id=pin_snapshot_id,
        paths=paths,
    )
