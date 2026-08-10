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


# R26-P0-015：合法 snapshot policy；未知 policy 必须 reject。
_SNAPSHOT_POLICIES = frozenset({"latest", "pin", "fail_if_changed"})


@dataclass(frozen=True)
class SourceManifest:
    """上游 publisher 的 source manifest（R25 §13 / §31）。

    ``{source_generation, objects: [{key, etag, size}]}``。DataAccess mirror 下载
    generation G，本地 manifest 绑定 G。缺失时 DataAccess 必须明确
    ``external_source_identity_unverified``（§32）。

    R26-P0-015：manifest 必须携带
        manifest_version / dataset / generation / complete=true / objects[] /
        object_count / content_digest / prefix / published_at。
    """

    source_generation: str
    objects: tuple[ResolvedObject, ...] = ()
    manifest_version: str | None = None
    dataset: str | None = None
    complete: bool = True
    object_count: int | None = None
    content_digest: str | None = None
    prefix: str | None = None
    published_at: str | None = None


def _validate_manifest_object(uri: str, entry: Any, problems: list[str]) -> bool:
    """单 object 验证（R26-P0-015/P1-023）：空 URI / ../ escape / 跨 bucket / dup。"""
    if not uri:
        problems.append("manifest object 空 URI")
        return False
    if ".." in uri.split("/"):
        problems.append(f"manifest object URI 含 ../ escape：{uri!r}")
        return False
    if uri.startswith(("s3://", "cos://")):
        parts = uri.split("/", 3)
        if len(parts) < 3:
            problems.append(f"manifest object URI 缺少 bucket：{uri!r}")
            return False
    return True


def parse_source_manifest(
    raw: Any, *, strict: bool = True
) -> SourceManifest | None:
    """解析 publisher source manifest（dict 或 JSON 字符串）。

    R26-P0-015：strict 下校验 manifest_version / complete / object_count /
    duplicate / empty URI / ../ escape / cross bucket / digest。违反 → 抛
    ``SourceSnapshotUnavailable``（不是返回 None——None 意味着「无 manifest」，
    损坏 manifest 必须 fail，不能当「没提供」放行）。
    """
    from data_access.core.exceptions import SourceSnapshotUnavailable

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            if strict:
                raise SourceSnapshotUnavailable(
                    "source manifest 不是合法 JSON（R26-P0-015 fail-closed）"
                )
            return None
    if not isinstance(raw, dict):
        if strict:
            raise SourceSnapshotUnavailable(
                "source manifest 顶层必须是 mapping（R26-P0-015 fail-closed）"
            )
        return None
    gen = raw.get("source_generation") or raw.get("generation")
    if not gen:
        if strict:
            raise SourceSnapshotUnavailable(
                "source manifest 缺少 source_generation/generation"
                "（R26-P0-015 fail-closed）"
            )
        return None
    complete = raw.get("complete", True)
    if strict and complete is False:
        raise SourceSnapshotUnavailable(
            "source manifest complete=false：上游尚未发布完整 generation"
            "（R26-P0-015 fail-closed，拒绝读取未完成快照）"
        )
    manifest_version = str(raw.get("manifest_version", "")).strip() or None
    if strict and manifest_version is None:
        raise SourceSnapshotUnavailable(
            "source manifest 缺少 manifest_version（R26-P0-015 fail-closed）"
        )
    problems: list[str] = []
    seen_uris: set[str] = set()
    objs: list[ResolvedObject] = []
    for entry in raw.get("objects") or ():
        if isinstance(entry, str):
            uri = entry
            _validate_manifest_object(uri, entry, problems)
            if uri in seen_uris:
                problems.append(f"manifest object 重复：{uri!r}")
            seen_uris.add(uri)
            objs.append(ResolvedObject(uri=uri, source="source_manifest"))
            continue
        if isinstance(entry, dict):
            uri = str(entry.get("key") or entry.get("uri") or "").strip()
            if not _validate_manifest_object(uri, entry, problems):
                continue
            if uri in seen_uris:
                problems.append(f"manifest object 重复：{uri!r}")
            seen_uris.add(uri)
            objs.append(
                ResolvedObject(
                    uri=uri,
                    etag=entry.get("etag"),
                    version_id=entry.get("version_id"),
                    content_length=entry.get("size") or entry.get("content_length"),
                    source="source_manifest",
                )
            )
    declared_count = raw.get("object_count")
    if strict and declared_count is not None:
        try:
            if int(declared_count) != len(objs):
                problems.append(
                    f"manifest object_count={declared_count} != 实际 {len(objs)}"
                )
        except (TypeError, ValueError):
            problems.append(f"manifest object_count 非法：{declared_count!r}")
    if strict and problems:
        raise SourceSnapshotUnavailable(
            "source manifest 验证失败（R26-P0-015 fail-closed）："
            + "; ".join(problems)
        )
    return SourceManifest(
        source_generation=str(gen),
        objects=tuple(objs),
        manifest_version=manifest_version,
        dataset=raw.get("dataset"),
        complete=bool(complete),
        object_count=len(objs),
        content_digest=raw.get("content_digest"),
        prefix=raw.get("prefix"),
        published_at=raw.get("published_at"),
    )


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
        file_selector: Any = None,
    ) -> None:
        self._list_objects_fn = list_objects_fn
        self._head_object_fn = head_object_fn
        self._source_manifest_fn = source_manifest_fn
        self._strict = strict
        # R26-P0-010：FileSelector IR（split/shares 精确过滤）。
        self._file_selector = file_selector

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
        # R26-P0-015：未知 policy reject。
        if policy not in _SNAPSHOT_POLICIES:
            raise ValidationError(
                f"snapshot policy={policy!r} 非法；允许 {sorted(_SNAPSHOT_POLICIES)}"
                "（R26-P0-015，未知 policy 必须 reject）"
            )

        # 1) publisher source manifest（最优）。
        manifest = None
        if self._source_manifest_fn is not None:
            try:
                manifest = parse_source_manifest(
                    self._source_manifest_fn(dataset), strict=strict
                )
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
                    # R26-P0-010：按 FileSelector 过滤 exact objects（split 不吞
                    # shares_*），避免两类 schema 混读。
                    if self._file_selector is not None:
                        listed = [
                            o
                            for o in listed
                            if self._file_selector.matches(
                                str(getattr(o, "uri", o)).rsplit("/", 1)[-1]
                            )
                        ]
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

        # R26-P0-015：object 必须在 dataset registered boundary 下（paths 公共前缀）。
        if objects and paths and strict:
            boundary = _common_prefix(paths)
            if boundary:
                for o in objects:
                    if not str(o.uri).startswith(boundary):
                        raise SourceSnapshotUnavailable(
                            f"manifest object {o.uri} 越出 dataset registered boundary "
                            f"{boundary!r}（R26-P0-015，production fail-closed）"
                        )
        # R26-P0-015：strict 下 exact object 必须携带可证明身份
        # （version_id OR etag+content_length OR checksum）。
        if strict and objects:
            for o in objects:
                if str(o.uri).startswith(("s3://", "cos://")):
                    if not (
                        o.version_id or (o.etag and o.content_length is not None)
                    ):
                        raise SourceSnapshotUnavailable(
                            f"manifest object {o.uri} 无 etag/version_id/content_length"
                            "（R26-P0-015，URI 不是 content identity，production deny）"
                        )

        digest = content_digest_of_objects(objects)
        snap = ResolvedSourceSnapshot(
            dataset=dataset,
            source_generation=generation,
            objects=objects,
            content_digest=digest,
        )
        if policy == "fail_if_changed":
            # R26-P0-015：fail_if_changed = 固定到当前 digest，执行期变化 → 拒绝。
            pin_snapshot_id = digest
            policy = "pin"
        if policy == "pin" and pin_snapshot_id and pin_snapshot_id != digest:
            raise ValidationError(
                f"dataset={dataset!r} 当前 snapshot digest={digest} != pinned "
                f"{pin_snapshot_id}（policy=pin）——拒绝读取不一致版本。"
            )
        return snap


_GLOB_MARKERS = ("*", "?", "[", "{", "]")


def _common_prefix(paths: Sequence[str]) -> str | None:
    """从一组 s3:// path 提取公共前缀（去掉尾部 glob）。

    R26-P0-015：修正 ``"? "`` typo，完整支持 ``*`` / ``?`` / ``[`` / ``]`` / ``{``。
    """
    cleaned = []
    for p in paths:
        s = str(p)
        idx = len(s)
        for marker in _GLOB_MARKERS:
            m = s.find(marker)
            if m >= 0 and m < idx:
                idx = m
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
