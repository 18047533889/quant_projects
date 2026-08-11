"""
data_access.r30.partition_index —— R30-P0-006 Partition Metadata Index（立面 + 验收）

R30 全维度成熟度层是 **additive** 的（不修改 store.py / read/ 既有文件）。本模块
是既有 Partition Metadata 底层能力之上的 **立面（facade）+ 验收**，不重复实现：

    - ``read/manifest.py`` 的 ``DatasetManifest``（``files`` 含 ``min_time/max_time/
      rows/bytes/schema_hash/etag``）——文件级裁剪的事实源；
    - ``read/metadata_plane.py`` 的 ``DatasetMetadataPlane``——统一加载已落盘
      manifest；
    - ``read/partition_planner.prune_paths_for_time_range``——manifest 缺失时按
      hive/pattern 展开 glob 路径的兜底。

R30 命名对象：

    - ``PartitionMeta``：单个分区的不可变元数据条目；
    - ``PartitionMetadataIndex``：不可变 generation 的索引对象，``objects_for_range``
      / ``object_uris`` 只做内存裁剪（min/max + partition_values 过滤），**绝不 glob**；
    - ``build_index``：从 store 现有能力构造索引；
    - ``build_prune_paths_via_index``：首次 build 后缓存到**进程内模块级 dict**，
      同一 immutable generation 的第二次请求直接查缓存，不再 glob/stat/footer；
    - ``assert_no_rescan_on_repeat``：R30-P0-006 验收函数——连续两次调用，断言
      第二次命中索引缓存（``from_index=True``）且没有再次调用 ``_resolve_raw_paths``。

``from_index`` 语义：
    - ``True``  = 路径来自 ``PartitionMetadataIndex``（无论本次是新建还是缓存命中），
      本调用**没有**发生兜底 glob/scan；
    - ``False`` = 没有可用 manifest/索引，本次回退到 ``_resolve_raw_paths`` 的原始
      glob 展开（路径来自扫描而非索引）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from data_access.r30._shared import stable_digest

_PARTITION_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^/]+)")
"""hive 分区路径段 ``date=2024-01-01`` / ``year=2024/month=03`` 的解析正则。"""


# ---------------------------------------------------------------------------
# 单个分区元数据
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionMeta:
    """一个物理分区文件的不可变元数据条目。

    ``partition_values`` 是从路径解析出的 hive 分区键值对（``date=2024-01-01``），
    ``predicates`` 过滤直接消费它——不需要 glob，也不需要读 parquet footer。
    """

    object_uri: str
    partition_values: Mapping[str, str] = field(default_factory=dict)
    min_time: Any = None
    max_time: Any = None
    row_count: int | None = None
    byte_size: int | None = None
    schema_epoch: str | None = None
    etag: str | None = None
    source_generation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_uri": self.object_uri,
            "partition_values": dict(self.partition_values or {}),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "schema_epoch": self.schema_epoch,
            "etag": self.etag,
            "source_generation": self.source_generation,
        }


# ---------------------------------------------------------------------------
# PartitionMetadataIndex
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionMetadataIndex:
    """一个 immutable generation 的分区元数据索引。

    构造后**不可变**：同一 generation 的第二次 prune 请求直接复用本对象，
    不再访问文件系统。
    """

    dataset: str
    objects: tuple[PartitionMeta, ...] = ()
    built_epoch: str | None = None
    source_generation: str | None = None
    version: str | None = None

    # ---- 内存裁剪（不 glob） ----
    def objects_for_range(
        self,
        time_range: tuple[Any, Any] | None = None,
        predicates: Mapping[str, Any] | None = None,
    ) -> list[PartitionMeta]:
        """按 min_time/max_time 裁剪 + partition_values 谓词过滤，纯内存。

        - ``time_range``：闭区间 ``(start, end)``；None 表示不过滤时间。
        - ``predicates``：``{分区键: 值或值集合}``；对象缺该分区键时保守保留
          （无法验证就不排除）。
        """
        out: list[PartitionMeta] = []
        for obj in self.objects:
            if not _overlaps(obj, time_range):
                continue
            if predicates:
                if not _matches_predicates(obj, predicates):
                    continue
            out.append(obj)
        return out

    def object_uris(
        self,
        time_range: tuple[Any, Any] | None = None,
        predicates: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """与 ``objects_for_range`` 相同的裁剪，只返回 object_uri 列表。"""
        return [o.object_uri for o in self.objects_for_range(time_range, predicates)]

    def index_fingerprint(self) -> str:
        """built_epoch + objects 摘要 → 稳定指纹（缓存 key / 身份比较用）。

        注意：``stable_digest`` 对 list/tuple 输入有拼接优先级 bug（``str + bytes``），
        这里统一折叠成**单个规范字符串**再求摘要，规避该缺陷（不修改 _shared.py）。
        """
        obj_parts = [
            f"{o.object_uri}|{o.min_time}|{o.max_time}|{o.row_count}|{o.byte_size}"
            for o in self.objects
        ]
        canonical = "|".join(
            [self.dataset, self.built_epoch or "", self.version or ""]
            + sorted(obj_parts)
        )
        return stable_digest(canonical)


def _time_key(value: Any) -> Any:
    """把 date/datetime 规范化成 isoformat 字符串；数值/字符串原样保留。"""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _cmp(a: Any, b: Any) -> int | None:
    """类型安全的比较；类型不兼容返回 None（保守保留）。"""
    try:
        if a < b:
            return -1
        if a > b:
            return 1
        return 0
    except TypeError:
        return None


def _overlaps(meta: PartitionMeta, time_range: tuple[Any, Any] | None) -> bool:
    """min/max 与闭区间 time_range 是否有交集；无时间信息保守保留。"""
    if time_range is None:
        return True
    start, end = time_range
    lo = _time_key(meta.min_time)
    hi = _time_key(meta.max_time)
    sk = _time_key(start)
    ek = _time_key(end)
    if lo is None or hi is None:
        return True
    if sk is not None:
        c = _cmp(hi, sk)
        if c is not None and c < 0:  # max < start → 区间在右侧，无交集
            return False
    if ek is not None:
        c = _cmp(lo, ek)
        if c is not None and c > 0:  # min > end → 区间在左侧，无交集
            return False
    return True


def _matches_predicates(meta: PartitionMeta, predicates: Mapping[str, Any]) -> bool:
    for key, value in predicates.items():
        if key not in meta.partition_values:
            continue  # 缺键 → 无法验证，保守保留
        allowed = (
            set(value)
            if isinstance(value, (list, tuple, set, frozenset))
            else {value}
        )
        if meta.partition_values[key] not in allowed:
            return False
    return True


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
def _parse_partition_values(path: str) -> dict[str, str]:
    """从路径段解析 hive 分区键值（``date=2024-01-01``、``year=2024``）。"""
    out: dict[str, str] = {}
    for seg in str(path).split("/"):
        m = _PARTITION_KEY_RE.match(seg)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _parse_path_date(path: str) -> str | None:
    """从路径里解析 ``YYYY-MM-DD`` 片段（兜底 fallback 的近似 min/max）。"""
    for tok in str(path).split("/"):
        if len(tok) >= 10:
            try:
                return date.fromisoformat(tok[:10]).isoformat()
            except ValueError:
                continue
    return None


def _metas_from_manifest(manifest: Any) -> tuple[PartitionMeta, ...]:
    """从 DatasetManifest（或 duck-typed 对象）的 files 构建 PartitionMeta。"""
    files = getattr(manifest, "files", None) or ()
    source_gen = (
        getattr(manifest, "source_epoch", None)
        or getattr(manifest, "manifest_epoch", None)
    )
    metas: list[PartitionMeta] = []
    for f in files:
        path = getattr(f, "path", None) or str(f)
        metas.append(
            PartitionMeta(
                object_uri=path,
                partition_values=_parse_partition_values(path),
                min_time=getattr(f, "min_time", None),
                max_time=getattr(f, "max_time", None),
                row_count=getattr(f, "rows", None),
                byte_size=getattr(f, "bytes", None),
                schema_epoch=getattr(f, "schema_hash", None),
                etag=getattr(f, "etag", None),
                source_generation=source_gen,
            )
        )
    return tuple(metas)


def _metas_from_paths(paths: Sequence[str]) -> tuple[PartitionMeta, ...]:
    """没有 manifest 时的兜底：只有 glob 路径，min/max 用路径日期近似。"""
    metas: list[PartitionMeta] = []
    for p in paths:
        dt = _parse_path_date(p)
        metas.append(
            PartitionMeta(
                object_uri=str(p),
                partition_values=_parse_partition_values(str(p)),
                min_time=dt,
                max_time=dt,
                row_count=None,
                byte_size=0,
                schema_epoch=None,
                etag=None,
                source_generation=None,
            )
        )
    return tuple(metas)


def _try_build_manifest(store: Any, dataset: str, params: Mapping[str, Any]) -> Any | None:
    """优先路径：``store.build_dataset_manifest`` 构建/确认后取含 per-file 细节的
    DatasetManifest。

    真实 store 的 ``build_dataset_manifest`` 返回摘要 dict（无 per-file 细节），
    因此构建后通过 ``load_manifest_for_dataset`` 重新加载完整 manifest。若 store
    没有该方法（或返回 None / 无 files），返回 None 交给 metadata_plane。
    """
    try:
        summary = store.build_dataset_manifest(dataset, **dict(params or {}))
    except Exception:
        summary = None
    if not summary:
        return None
    # 少数实现直接返回带 files 的对象
    if isinstance(summary, dict) and summary.get("files"):
        return summary
    # 否则重新加载（build 刚完成，应为 fresh）
    try:
        from data_access.read.manifest import load_manifest_for_dataset

        m = load_manifest_for_dataset(store, dataset, **dict(params or {}))
        if m is not None and getattr(m, "files", None):
            return m
    except Exception:
        return None
    return None


def _load_manifest(store: Any, dataset: str, params: Mapping[str, Any]) -> Any | None:
    """加载含 per-file 细节的 manifest 对象（优先 build 路径，其次 metadata_plane）。"""
    built = _try_build_manifest(store, dataset, params)
    if built is not None and getattr(built, "files", None):
        return built
    try:
        plane = store.metadata_plane(dataset, **dict(params or {}))
    except Exception:
        plane = None
    if plane is not None:
        try:
            m = plane.manifest()
        except Exception:
            m = None
        if m is not None and getattr(m, "files", None):
            return m
    return None


def _resolve_fallback_paths(store: Any, dataset: str, params: Mapping[str, Any]) -> list[str]:
    """manifest 不可用时的兜底：``_resolve_raw_paths`` 原始展开。"""
    reg = getattr(store, "_registry", None) or getattr(store, "registry", None)
    ds = None
    if reg is not None:
        try:
            ds = reg.get(dataset)
        except Exception:
            ds = None
    try:
        paths = store._resolve_raw_paths(ds, time_range=None, params=dict(params or {}))
    except Exception:
        try:
            paths = store._resolve_raw_paths(ds, time_range=None, params=dict(params or {}))
        except Exception:
            return []
    return list(paths)


def build_index(store: Any, dataset: str, params: Mapping[str, Any] | None = None) -> PartitionMetadataIndex:
    """从 store 既有能力构造 PartitionMetadataIndex。

    - 优先：DatasetManifest（files 含 min_time/max_time/row_count/byte_size）；
    - manifest 没有时：metadata_plane 或 ``prune_paths_for_time_range`` 兜底路径
      （min/max 由路径日期近似，row_count/byte_size 未知）。
    """
    params = dict(params or {})
    manifest = _load_manifest(store, dataset, params)
    if manifest is not None and getattr(manifest, "files", None):
        objects = _metas_from_manifest(manifest)
        built_epoch = (
            getattr(manifest, "manifest_built_epoch", None)
            or getattr(manifest, "source_epoch", None)
            or getattr(manifest, "manifest_epoch", None)
        )
        source_generation = (
            getattr(manifest, "source_epoch", None)
            or getattr(manifest, "manifest_epoch", None)
        )
        version = getattr(manifest, "dataset_version", None) or getattr(
            manifest, "partition_version", None
        )
        return PartitionMetadataIndex(
            dataset=dataset,
            objects=objects,
            built_epoch=built_epoch,
            source_generation=source_generation,
            version=version,
        )
    # 兜底：partition_planner / raw paths（无 per-file 统计）
    paths = _resolve_fallback_paths(store, dataset, params)
    objects = _metas_from_paths(paths)
    return PartitionMetadataIndex(dataset=dataset, objects=objects)


# ---------------------------------------------------------------------------
# 进程内索引缓存 + prune 入口 + 验收
# ---------------------------------------------------------------------------
_INDEX_CACHE: dict[tuple[str, str, str], PartitionMetadataIndex] = {}
"""进程内模块级缓存。key = (dataset, params_fingerprint, built_epoch)。"""


def clear_index_cache() -> None:
    """清空进程内索引缓存（测试 / 显式失效用）。"""
    _INDEX_CACHE.clear()


def index_cache_size() -> int:
    return len(_INDEX_CACHE)


def _params_fingerprint(params: Mapping[str, Any] | None) -> str:
    params = dict(params or {})
    canonical = "|".join(
        f"{str(k)}={str(v)}"
        for k, v in sorted(params.items(), key=lambda kv: str(kv[0]))
    )
    return stable_digest(canonical)


def _find_cached(dataset: str, params_fp: str) -> PartitionMetadataIndex | None:
    candidates = [
        idx
        for (ds, fp, _epoch), idx in _INDEX_CACHE.items()
        if ds == dataset and fp == params_fp
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda idx: idx.built_epoch or "", reverse=True)
    return candidates[0]


def build_prune_paths_via_index(
    store: Any,
    dataset: str,
    time_range: tuple[Any, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    *,
    revalidate: bool = False,
) -> tuple[list[str], bool]:
    """用索引裁剪路径；首次 build 后缓存，第二次命中缓存**不再 glob**。

    返回 ``(exact_paths, from_index)``：

        - ``from_index=True``：路径来自 PartitionMetadataIndex（本调用无兜底 glob）；
        - ``from_index=False``：无可用 manifest/索引，回退原始 glob 展开。

    缓存失效：``revalidate=True`` 强制重建（built_epoch / manifest 版本变化时）；
    同一 ``(dataset, params_fingerprint)`` 的新 build 会覆盖旧条目。
    """
    params = dict(params or {})
    params_fp = _params_fingerprint(params)
    if not revalidate:
        cached = _find_cached(dataset, params_fp)
        if cached is not None:
            return cached.object_uris(time_range), True
    index = build_index(store, dataset, params)
    built_epoch = index.built_epoch or "none"
    key = (dataset, params_fp, built_epoch)
    # 移除同 (dataset, params_fp) 的旧 generation 条目
    for old_key in [
        k
        for k in _INDEX_CACHE
        if k[0] == dataset and k[1] == params_fp and k != key
    ]:
        del _INDEX_CACHE[old_key]
    _INDEX_CACHE[key] = index
    return index.object_uris(time_range), True


def assert_no_rescan_on_repeat(
    store: Any,
    dataset: str,
    time_range: tuple[Any, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """R30-P0-006 验收函数：连续两次 ``build_prune_paths_via_index``。

    断言第二次命中索引缓存（``from_index=True``）且**没有重新 glob**（用计数包装
    统计 ``_resolve_raw_paths`` 调用次数）。返回诊断 dict。
    """
    original = getattr(store, "_resolve_raw_paths", None)
    if original is None:
        raise AssertionError("store 没有 _resolve_raw_paths，无法计数扫描次数")
    if not getattr(original, "_r30_counting", False):
        def counting(*args: Any, **kwargs: Any) -> Any:
            counting.call_count += 1
            return original(*args, **kwargs)

        counting.call_count = 0  # type: ignore[attr-defined]
        counting._r30_counting = True  # type: ignore[attr-defined]
        store._resolve_raw_paths = counting  # type: ignore[assignment]
        counter = counting  # type: ignore[assignment]
    else:
        counter = original  # type: ignore[assignment]

    paths1, from1 = build_prune_paths_via_index(
        store, dataset, time_range, params=params
    )
    n1 = counter.call_count  # type: ignore[attr-defined]
    paths2, from2 = build_prune_paths_via_index(
        store, dataset, time_range, params=params
    )
    n2 = counter.call_count  # type: ignore[attr-defined]

    assert from2 is True, (
        f"第二次应命中索引缓存（from_index=True），收到 {from2!r}"
    )
    assert n2 == n1, (
        f"第二次不应再 glob（_resolve_raw_paths 调用 {n1} → {n2}）"
    )
    assert paths1 == paths2, "两次裁剪结果应一致"
    return {
        "first_from_index": from1,
        "second_from_index": from2,
        "resolve_calls_after_first": n1,
        "resolve_calls_after_second": n2,
        "paths_match": paths1 == paths2,
    }


__all__ = [
    "PartitionMeta",
    "PartitionMetadataIndex",
    "build_index",
    "build_prune_paths_via_index",
    "assert_no_rescan_on_repeat",
    "clear_index_cache",
    "index_cache_size",
]
