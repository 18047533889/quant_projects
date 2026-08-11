# -*- coding: utf-8 -*-
"""R30-P0-011 (DATA_CHANGE_MIN_RECOMPUTE)：数据变化事实 + 最小重算触发。

本模块是 **DataAccess 侧** 的「数据变化事实」生产者：

    1. ``DataChangeSet`` —— 一次数据集变化的完整描述（对象 / 分区 / 时间窗 /
       标的 / 列 / 变化形态 / revision 可用性），是 DA 与 FE 之间传递「哪些
       数据变了」的最小事实单位。
    2. ``source_snapshot_fingerprint`` —— 廉价的 query-scoped 指纹，用来判断
       「数据集内容在两次调用之间是否变化」，不假设物理文件日期 = knowledge time。
    3. ``detect_changes`` —— before/after 指纹（或显式文件快照）差异 → 精确的
       ``DataChangeSet``（change_kind 判定：append / revision / delete /
       schema_change…）。
    4. ``revision_availability`` —— 合并物理文件日期与 knowledge time，回答
       「一个在 ``knowledge_date`` 时刻运行的因子，能看见这条 dataset 上最新
       的 revision 吗」——财务 revision 绝不能只看物理文件日期（避免 look-ahead）。

R30-P0-011 验收：**单公司一条财报修订不允许默认触发「全 A 股全历史所有因子」
重算**。本模块只负责描述 *到底哪些数据变了*；「哪些因子要重算」由
``change_impact.plan_minimal_recompute`` 消费本模块的 ``DataChangeSet`` 完成。
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.r30._shared import stable_digest_full


class ChangeKind(Enum):
    """数据集变化形态（最小重算按形态选择传播策略）。"""

    append = "append"
    correction = "correction"
    revision = "revision"
    delete = "delete"
    schema_change = "schema_change"
    calendar_change = "calendar_change"
    universe_change = "universe_change"


def _coerce_change_kind(value: Any) -> ChangeKind:
    if isinstance(value, ChangeKind):
        return value
    if isinstance(value, str):
        try:
            return ChangeKind(value)
        except ValueError:
            # 未知字符串 → 保守地视为 revision（内容已变但形态未知）。
            return ChangeKind.revision
    return ChangeKind.revision


def _utc_now_iso() -> str:
    return datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _freeze_sequence(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, set, frozenset)):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    return ()


@dataclass(frozen=True)
class DataChangeSet:
    """一次数据集变化的事实描述。

    所有序列字段冻结为 tuple；``change_kind`` 归一化为 :class:`ChangeKind`。
    ``changed_time_range`` 是 **knowledge time 意义上的变化区间**（从 changed
    files 的 min/max 推得），**不是**物理文件写入时间。``changed_instruments``
    为空 tuple 表示「无法判定哪些标的变了」——消费方应保守处理。
    """

    dataset: str
    source_before: str | None = None
    source_after: str | None = None
    changed_objects: tuple = ()
    changed_partitions: tuple = ()
    changed_time_range: tuple | None = None  # (start, end) knowledge-time 窗口
    changed_instruments: tuple = ()
    changed_columns: tuple = ()
    change_kind: ChangeKind | str = ChangeKind.append
    revision_availability: dict = field(default_factory=dict)
    detected_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_objects", _freeze_sequence(self.changed_objects))
        object.__setattr__(self, "changed_partitions", _freeze_sequence(self.changed_partitions))
        object.__setattr__(self, "changed_instruments", _freeze_sequence(self.changed_instruments))
        object.__setattr__(self, "changed_columns", _freeze_sequence(self.changed_columns))
        object.__setattr__(self, "change_kind", _coerce_change_kind(self.change_kind))
        if isinstance(self.changed_time_range, (list, tuple)) and len(self.changed_time_range) == 2:
            object.__setattr__(self, "changed_time_range", tuple(self.changed_time_range))
        if not isinstance(self.revision_availability, dict):
            object.__setattr__(self, "revision_availability", dict(self.revision_availability or {}))
        if self.detected_at is None:
            object.__setattr__(self, "detected_at", _utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        ck = self.change_kind
        return {
            "dataset": self.dataset,
            "source_before": self.source_before,
            "source_after": self.source_after,
            "changed_objects": list(self.changed_objects),
            "changed_partitions": list(self.changed_partitions),
            "changed_time_range": (
                None if self.changed_time_range is None else list(self.changed_time_range)
            ),
            "changed_instruments": list(self.changed_instruments),
            "changed_columns": list(self.changed_columns),
            "change_kind": ck.value if isinstance(ck, ChangeKind) else str(ck),
            "revision_availability": dict(self.revision_availability),
            "detected_at": self.detected_at,
        }

    def is_empty(self) -> bool:
        """没有任何实际数据变化。

        判定：对象/分区/时间窗/标的/列都为空，**且** source_before/source_after
        不指向一次内容变化（二者相等或都为空 = 无历史可比 → 视为空）。
        """
        if (
            self.changed_objects
            or self.changed_partitions
            or self.changed_time_range
            or self.changed_instruments
            or self.changed_columns
        ):
            return False
        if self.source_before is None and self.source_after is None:
            return True
        if self.source_before is not None and self.source_before == self.source_after:
            return True
        return False


# ---------------------------------------------------------------------------
# 快照指纹
# ---------------------------------------------------------------------------
_MANIFEST_TOKEN_KEYS = (
    "dataset_version",
    "partition_version",
    "source_epoch",
    "manifest_built_epoch",
    "file_count",
    "manifest_generation_id",
)


def source_snapshot_fingerprint(store: Any, dataset: str, params: Any = None) -> str | None:
    """折叠数据集当前内容的廉价指纹；失败返回 ``None``（调用方按不可判定处理）。

    优先 ``store.manifest_version(dataset, **params)``（只读 ``_manifest.json``
    sidecar，不扫全量 footer）。无 manifest / 失败时回退
    ``store.build_dataset_manifest``；再失败返回 ``None``。
    """
    params = dict(params or {})
    token: Any = None
    try:
        token = store.manifest_version(dataset, **params)
    except Exception:
        token = None
    if isinstance(token, dict) and token.get("has_manifest"):
        payload = {"dataset": dataset, "kind": "manifest"}
        for key in _MANIFEST_TOKEN_KEYS:
            value = token.get(key)
            if value is not None:
                payload[key] = value
        return stable_digest_full(payload)

    # 回退：build_dataset_manifest（一次性扫描，更贵）。
    built: Any = None
    try:
        built = store.build_dataset_manifest(dataset, **params)
    except Exception:
        built = None
    if isinstance(built, dict):
        summary = {key: built.get(key) for key in ("files", "rows", "bytes", "format")}
        if summary.get("files") is not None:
            summary["dataset"] = dataset
            summary["kind"] = "built"
            return stable_digest_full(summary)
    return None


# ---------------------------------------------------------------------------
# manifest 加载 / 文件信息抽取（防御性，绝不因 manifest 缺失而抛）
# ---------------------------------------------------------------------------
def _load_manifest(store: Any, dataset: str, params: Mapping[str, Any] | None) -> Any:
    """加载 ``DatasetManifest``；真实 store 与 fake store 都能工作，失败返回 None。"""
    params = dict(params or {})
    loader = getattr(store, "load_manifest", None)
    if callable(loader):
        try:
            manifest = loader(dataset, **params)
            if manifest is not None:
                return manifest
        except Exception:
            pass
    try:
        from data_access.read.manifest import load_manifest_for_dataset

        return load_manifest_for_dataset(store, dataset, **params)
    except Exception:
        return None


def _files_from_manifest(manifest: Any) -> dict[str, dict[str, Any]] | None:
    """DatasetManifest → ``{path: {bytes, mtime_ns, schema_hash, min_time, ...}}``。"""
    if manifest is None:
        return None
    files = getattr(manifest, "files", None)
    if not files:
        return None
    out: dict[str, dict[str, Any]] = {}
    for f in files:
        path = str(getattr(f, "path", ""))
        if not path:
            continue
        out[path] = {
            "bytes": getattr(f, "bytes", None),
            "mtime_ns": getattr(f, "mtime_ns", None),
            "schema_hash": getattr(f, "schema_hash", None),
            "min_time": getattr(f, "min_time", None),
            "max_time": getattr(f, "max_time", None),
            "min_instrument": getattr(f, "min_instrument", None),
            "max_instrument": getattr(f, "max_instrument", None),
        }
    return out or None


def _extract_file_map(value: Any) -> dict[str, dict[str, Any]] | None:
    """从 before/after 输入里抽 ``{path: info}``。

    接受 str 指纹（无文件级信息 → None）、或 Mapping（``files`` 为 ``{path:
    info}`` 或 ``[{path, ...}]`` 列表）。
    """
    if not isinstance(value, Mapping):
        return None
    files = value.get("files")
    if isinstance(files, Mapping):
        return {str(p): dict(i) if isinstance(i, Mapping) else {} for p, i in files.items()}
    if isinstance(files, (list, tuple)):
        out: dict[str, dict[str, Any]] = {}
        for item in files:
            if not isinstance(item, Mapping):
                continue
            path = item.get("path")
            if path:
                out[str(path)] = dict(item)
        return out or None
    return None


def _file_sig(info: dict[str, Any] | None) -> tuple:
    if not info:
        return ()
    return (
        info.get("bytes"),
        info.get("mtime_ns"),
        info.get("schema_hash"),
    )


def _schema_epochs(value: Any) -> dict | None:
    if isinstance(value, Mapping):
        se = value.get("schema_epochs")
        if isinstance(se, Mapping):
            return se
    return None


def _schema_fields(schema_epochs: dict | None) -> set[str] | None:
    if not schema_epochs:
        return None
    fields: set[str] = set()
    for epoch_hash, fields_map in schema_epochs.items():
        if isinstance(fields_map, Mapping):
            for key in fields_map:
                fields.add(str(key))
        elif isinstance(epoch_hash, str):
            fields.add(epoch_hash)
    return fields or None


def _schema_changed(
    before_files: dict | None,
    after_files: dict | None,
    before: Any,
    after: Any,
) -> bool:
    if before_files and after_files:
        for path in set(before_files) & set(after_files):
            bh = (before_files.get(path) or {}).get("schema_hash")
            ah = (after_files.get(path) or {}).get("schema_hash")
            if bh is not None and ah is not None and bh != ah:
                return True
    bf = _schema_fields(_schema_epochs(before))
    af = _schema_fields(_schema_epochs(after))
    if bf is not None and af is not None and bf != af:
        return True
    return False


def _changed_columns(before: Any, after: Any) -> tuple[str, ...]:
    bf = _schema_fields(_schema_epochs(before))
    af = _schema_fields(_schema_epochs(after))
    if bf is None or af is None:
        return ()
    return tuple(sorted(bf ^ af))


def _partitions_from_paths(paths: Sequence[str]) -> tuple[str, ...]:
    parts: set[str] = set()
    for p in paths:
        path = str(p).replace("\\", "/")
        for segment in path.split("/"):
            if "=" in segment:
                parts.add(segment)
    if not parts:
        for p in paths:
            parent = str(Path(str(p)).parent)
            if parent and parent != ".":
                parts.add(parent)
    return tuple(sorted(parts))


def _infer_change_kind(
    *,
    added: set[str],
    removed: set[str],
    modified: set[str],
    schema_changed: bool,
    before_known: bool,
    before_is_none: bool,
) -> ChangeKind:
    if schema_changed:
        return ChangeKind.schema_change
    if removed and not added and not modified:
        return ChangeKind.delete
    if added and not removed and not modified:
        return ChangeKind.append
    if modified:
        return ChangeKind.revision
    if not before_known:
        # 无法与 before 精确 diff → 首快照视为 append，其余保守视为 revision。
        return ChangeKind.append if before_is_none else ChangeKind.revision
    return ChangeKind.revision


def _changed_time_range_from_files(
    changed_paths: Sequence[str],
    before_files: dict | None,
    after_files: dict | None,
) -> tuple[str, str] | None:
    times: list[tuple[str, str]] = []
    for path in changed_paths:
        info = None
        if after_files:
            info = after_files.get(path) or info
        if before_files:
            info = before_files.get(path) or info
        if not info:
            continue
        if info.get("min_time"):
            times.append(("min", str(info["min_time"])))
        if info.get("max_time"):
            times.append(("max", str(info["max_time"])))
    if not times:
        return None
    return (min(t for _, t in times), max(t for _, t in times))


def _changed_instruments_from_files(
    changed_paths: Sequence[str],
    before_files: dict | None,
    after_files: dict | None,
) -> tuple[str, ...]:
    insts: set[str] = set()
    for path in changed_paths:
        info = None
        if after_files:
            info = after_files.get(path) or info
        if before_files:
            info = before_files.get(path) or info
        if not info:
            continue
        if info.get("min_instrument"):
            insts.add(str(info["min_instrument"]))
        if info.get("max_instrument"):
            insts.add(str(info["max_instrument"]))
    return tuple(sorted(insts))


def detect_changes(
    store: Any,
    dataset: str,
    before: str | None,
    after: str | None = None,
    params: Any = None,
) -> DataChangeSet:
    """比较 before/after，产出一次 ``DataChangeSet``。

    ``before`` / ``after`` 支持两种形态（防御性）：

      - ``str``：``source_snapshot_fingerprint`` 的指纹。两者相等 → 空变化集；
        不等 → 从 ``store`` 加载当前 manifest，把当前文件集视为潜在变化对象
        （无法与指纹恢复 before 文件集，因此 change_kind 保守为 revision）。
      - ``Mapping``：带 ``files``（``{path: info}`` 或 ``[{path, ...}]` 列表）
        与可选 ``schema_epochs`` 的显式快照 → 精确 diff（新增→append、文件变
        →revision、缺失→delete、schema 变→schema_change）。

    不假设物理文件日期 = knowledge time：``changed_time_range`` 取自 changed
    files 的 ``min_time/max_time``（数据时间），不是 mtime。
    """
    params = dict(params or {})
    before_fp = before if isinstance(before, str) else (
        before.get("fingerprint") if isinstance(before, Mapping) else None
    )
    after_fp = after if isinstance(after, str) else (
        after.get("fingerprint") if isinstance(after, Mapping) else None
    )
    if after is None:
        # 未提供显式 after 快照 → 从 store 现算指纹（after 是 Mapping 时 caller
        # 已给出文件级快照，不强制依赖 store 的指纹可用性）。
        after_fp = source_snapshot_fingerprint(store, dataset, params)

    detected_at = _utc_now_iso()

    # 完全一致 → 无变化。
    if before_fp is not None and after_fp is not None and before_fp == after_fp:
        return DataChangeSet(
            dataset=dataset,
            source_before=before_fp,
            source_after=after_fp,
            change_kind=ChangeKind.append,
            detected_at=detected_at,
        )

    before_files = _extract_file_map(before)
    after_files = _extract_file_map(after)
    if after_files is None:
        after_files = _files_from_manifest(_load_manifest(store, dataset, params))

    before_known = before_files is not None
    before_is_none = before is None

    if after_files is None and before_files is None:
        # 无任何文件级信息，但指纹确实不同 → 保守：全数据集变化，窗口未绑定。
        return DataChangeSet(
            dataset=dataset,
            source_before=before_fp,
            source_after=after_fp,
            changed_objects=(),
            changed_partitions=(),
            changed_time_range=None,
            changed_instruments=(),
            changed_columns=(),
            change_kind=_infer_change_kind(
                added=set(),
                removed=set(),
                modified=set(),
                schema_changed=_schema_changed(None, None, before, after),
                before_known=before_known,
                before_is_none=before_is_none,
            ),
            revision_availability={},
            detected_at=detected_at,
        )

    after_keys = set(after_files) if after_files else set()
    before_keys = set(before_files) if before_files else set()
    added = after_keys - before_keys
    removed = before_keys - after_keys
    modified = {
        p
        for p in (after_keys & before_keys)
        if _file_sig((before_files or {}).get(p)) != _file_sig((after_files or {}).get(p))
    }
    changed_paths = tuple(sorted(added | removed | modified))

    schema_changed = _schema_changed(before_files, after_files, before, after)
    change_kind = _infer_change_kind(
        added=added,
        removed=removed,
        modified=modified,
        schema_changed=schema_changed,
        before_known=before_known,
        before_is_none=before_is_none,
    )

    # 有 after 文件集但没精确 diff（before 是 str 指纹）→ 全量保守。
    if before_files is None and after_files is not None and not changed_paths:
        changed_paths = tuple(sorted(after_keys))

    return DataChangeSet(
        dataset=dataset,
        source_before=before_fp,
        source_after=after_fp,
        changed_objects=changed_paths,
        changed_partitions=_partitions_from_paths(changed_paths),
        changed_time_range=_changed_time_range_from_files(
            changed_paths, before_files, after_files
        ),
        changed_instruments=_changed_instruments_from_files(
            changed_paths, before_files, after_files
        ),
        changed_columns=_changed_columns(before, after),
        change_kind=change_kind,
        revision_availability={},
        detected_at=detected_at,
    )


def dataset_change_since(
    store: Any,
    dataset: str,
    reference_fingerprint: str | None,
    params: Any = None,
) -> DataChangeSet:
    """便捷：与一个历史引用指纹对比，判断该 dataset 自引用以来是否变化。"""
    current = source_snapshot_fingerprint(store, dataset, params)
    return detect_changes(store, dataset, reference_fingerprint, current, params)


# ---------------------------------------------------------------------------
# revision 可用性（合并物理文件日期与 knowledge time）
# ---------------------------------------------------------------------------
def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).date()
        except (ValueError, OSError, OverflowError):
            return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _mtime_ns_to_date(mtime_ns: Any) -> date | None:
    try:
        return datetime.fromtimestamp(int(mtime_ns) / 1_000_000_000).date()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def revision_availability(
    store: Any,
    dataset: str,
    instrument: str,
    knowledge_date: Any,
) -> dict[str, Any]:
    """合并物理文件日期与 knowledge time，判定 revision 在 ``knowledge_date`` 可见性。

    返回 ``{latest_revision_date, period_end, knowledge_aware}``。

    - ``latest_revision_date``：一个在 ``knowledge_date`` 运行的消费方实际能用到
      的最新 revision 日期。当给了 knowledge_date 且物理最新写入晚于它时，钳制到
      knowledge_date（**杜绝 look-ahead**——财务 revision 不能只看物理文件日期）。
    - ``period_end``：changed 文件 ``max_time`` 的最大值（财报所属期间）。
    - ``knowledge_aware``：True = 应用了 knowledge time 校正；False = 只有物理
      日期（调用方未给 knowledge_date）。
    """
    params: dict[str, Any] = {}
    manifest = _load_manifest(store, dataset, params)
    physical_dates: list[date] = []
    period_ends: list[str] = []
    if manifest is not None:
        files = getattr(manifest, "files", ()) or ()
        for f in files:
            if instrument:
                min_inst = getattr(f, "min_instrument", None)
                max_inst = getattr(f, "max_instrument", None)
                if min_inst is not None and max_inst is not None:
                    if not (str(min_inst) <= str(instrument) <= str(max_inst)):
                        continue
            mtime_ns = getattr(f, "mtime_ns", None)
            if mtime_ns:
                d = _mtime_ns_to_date(mtime_ns)
                if d is not None:
                    physical_dates.append(d)
            max_time = getattr(f, "max_time", None)
            if max_time:
                period_ends.append(str(max_time))

    physical_latest = max(physical_dates) if physical_dates else None
    period_end = max(period_ends) if period_ends else None
    kd = _to_date(knowledge_date)

    if kd is not None:
        knowledge_aware = True
        if physical_latest is not None:
            latest = min(kd, physical_latest)
        else:
            latest = kd
    else:
        knowledge_aware = False
        latest = physical_latest

    return {
        "latest_revision_date": (
            latest.isoformat() if hasattr(latest, "isoformat") else latest
        ),
        "period_end": period_end,
        "knowledge_aware": bool(knowledge_aware),
    }


__all__ = [
    "ChangeKind",
    "DataChangeSet",
    "dataset_change_since",
    "detect_changes",
    "revision_availability",
    "source_snapshot_fingerprint",
]
