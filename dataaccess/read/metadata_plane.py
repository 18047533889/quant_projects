"""
data_access.read.metadata_plane —— 数据集元数据统一访问层（#38）

职责
    把散落在各处的 sidecar / 推导结果统一成一个查询入口：
        - manifest（``_manifest.parquet`` + ``_manifest.json``）
        - row-group stats（``_manifest_rowgroups.parquet``）
        - coverage（声明 vs 观测，max_staleness 判定）
        - PIT 事件索引（``_pit_event_index.parquet`` + json）
        - schema（registry 声明）
        - contract（COS 契约 / ContractIR 视图）
        - source epoch（manifest 双 epoch）
        - mirror inventory（cos/mirror 的下载记录）

设计
    1. 每个 accessor 惰性求值并缓存（按 (dataset, params) 键）；数据变了会因
       source_epoch 变化而失效。
    2. planner 只访问本 plane，不再到处 ``glob / read json / read manifest /
       read PIT sidecar / read schema``。

非职责
    不做数据读取；不判定查询预算（那是 query_budget.py）；不裁剪路径（那是
    partition_planner / manifest.prune）。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class DatasetMetadataPlane:
    """一个数据集在当前 params 下的统一元数据视图。"""

    store: Any
    dataset: str
    params: dict[str, Any] = field(default_factory=dict)
    _manifest: Any = field(default=None, repr=False)
    _manifest_fresh: bool = False
    _coverage: Any = field(default=None, repr=False)
    _pit_index: Any = field(default=None, repr=False)
    _contract: Any = field(default=None, repr=False)
    _source_epoch: str | None = None

    # ---- manifest ----

    def manifest(self) -> Any | None:
        """加载数据集 manifest（仅当双 epoch fresh）。"""
        if self._manifest is None:
            from data_access.read.manifest import (
                DatasetManifest,
                is_manifest_fresh,
                manifest_root_for_paths,
            )

            try:
                raw_paths = self.store._resolve_raw_paths(
                    self.store._registry.get(self.dataset),
                    time_range=None,
                    params=self.params,
                )
            except Exception:
                return None
            root = manifest_root_for_paths(raw_paths)
            if root is not None:
                try:
                    m = DatasetManifest.load(root)
                except Exception:
                    m = None
                if m is not None and m.dataset in {"", self.dataset}:
                    fresh = False
                    try:
                        fresh = is_manifest_fresh(m, raw_paths)
                    except Exception:
                        fresh = False
                    if fresh:
                        self._manifest = m
                        self._manifest_fresh = True
        return self._manifest

    @property
    def manifest_fresh(self) -> bool:
        return self._manifest_fresh

    def source_epoch(self) -> str | None:
        """当前 source_epoch（数据版本；mutation 后递增）。"""
        if self._source_epoch is None:
            try:
                token = self.store.manifest_version(self.dataset, **self.params)
                self._source_epoch = token.get("manifest_epoch")
            except Exception:
                self._source_epoch = None
        return self._source_epoch

    def schema(self) -> dict[str, str]:
        """registry 声明的 schema。"""
        try:
            ds = self.store._registry.get(self.dataset)
            return dict(getattr(ds, "schema", None) or {})
        except Exception:
            return {}

    # ---- coverage ----

    def coverage(self) -> Any:
        """覆盖/完整性/陈旧度报告（含 max_staleness 判定）。"""
        if self._coverage is None:
            from data_access.read.coverage import compute_coverage

            self._coverage = compute_coverage(
                self.store, self.dataset, params=self.params
            )
        return self._coverage

    # ---- PIT 事件索引 ----

    def pit_event_index(self) -> Any | None:
        """PIT 事件索引（仅当 authoritative：complete + 源匹配）。"""
        if self._pit_index is None:
            from data_access.read.pit_event_index import _index_path_for, load_pit_event_index

            try:
                path = _index_path_for(self.store, self.dataset)
            except Exception:
                path = None
            if path is None:
                return None
            try:
                idx = load_pit_event_index(path)
            except Exception:
                return None
            if not idx.metadata.is_authoritative:
                return None
            if (
                idx.metadata.manifest_epoch is not None
                and idx.metadata.manifest_epoch != self.source_epoch()
            ):
                return None
            self._pit_index = idx
        return self._pit_index

    # ---- contract ----

    def contract(self) -> Any | None:
        """COS 契约（表级时间语义）。"""
        if self._contract is None:
            from data_access.cos_contract import get_cos_contract

            try:
                self._contract = get_cos_contract(self.dataset)
            except Exception:
                self._contract = None
        return self._contract

    # ---- mirror inventory ----

    def mirror_inventory(self) -> dict[str, Any]:
        """本地 mirror 的下载清单（remote_key/etag/bytes/verified...）。"""
        from data_access.cos.mirror import load_mirror_inventory

        try:
            return load_mirror_inventory(self.store, self.dataset)
        except Exception:
            return {}

    def to_dict(self) -> dict[str, Any]:
        """把本层能拿到的元数据序列化成 dict（审计/看板用）。"""
        out: dict[str, Any] = {"dataset": self.dataset}
        out["source_epoch"] = self.source_epoch()
        out["schema_version"] = None
        try:
            out["schema_version"] = getattr(
                self.store._registry.get(self.dataset), "schema_version", None
            )
        except Exception:
            pass
        m = self.manifest()
        if m is not None:
            out["manifest"] = {
                "file_count": m.file_count,
                "rows": m.total_rows,
                "bytes": m.total_bytes,
                "fresh": self._manifest_fresh,
                "source_epoch": m.source_epoch,
                "built_epoch": m.manifest_built_epoch,
            }
        try:
            cov = self.coverage()
            out["coverage"] = {
                "status": cov.status,
                "observed_start": cov.observed_start,
                "observed_end": cov.observed_end,
                "observed_files": cov.observed_files,
                "max_staleness": cov.max_staleness,
            }
        except Exception:
            pass
        c = self.contract()
        if c is not None:
            out["contract"] = {
                "temporal_model": c.temporal_model,
                "pit_policy": c.pit_policy,
                "calendar_domain": c.calendar_domain,
            }
        return out


__all__ = ["DatasetMetadataPlane"]
