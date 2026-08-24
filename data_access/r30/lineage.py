# -*- coding: utf-8 -*-
"""R30-P1-019 (LINEAGE_STORE)：读取血缘查询。

记录「谁（run / principal）在哪个代码构建下、基于哪个 source snapshot /
experiment snapshot，读了哪个 dataset / concept 的哪些列」，并支持三组典型问题：

    - 某 factor 用过哪些 dataset？      -> ``datasets_for_factor``
    - 某 dataset revision 影响哪些 factor？-> ``factors_for_dataset``
    - 某 premium source 被谁读过？      -> ``readers_of_source``

实现：优先 DuckDB 建表持久化（duckdb 缺失 / path 不可写 → 降级内存模式，记录
``mode``）。行结构统一为：

    run_id, factor_id, dataset, concept, source_snapshot, experiment_snapshot,
    principal, build_sha, timestamp, extra(dict)

``extra`` 存任意附加字段（内存模式直接存 dict；DuckDB 模式序列化为 JSON）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

_LINEAGE_COLUMNS = (
    "run_id",
    "factor_id",
    "dataset",
    "concept",
    "source_snapshot",
    "experiment_snapshot",
    "principal",
    "build_sha",
    "timestamp",
    "extra",
)

_DDL = (
    "CREATE TABLE IF NOT EXISTS lineage ("
    "run_id VARCHAR, factor_id VARCHAR, dataset VARCHAR, concept VARCHAR, "
    "source_snapshot VARCHAR, experiment_snapshot VARCHAR, principal VARCHAR, "
    "build_sha VARCHAR, timestamp VARCHAR, extra VARCHAR)"
)


class LineageStore:
    """读取血缘存储（DuckDB 优先，内存降级）。"""

    def __init__(self, path: str | None = None) -> None:
        self.path: str | None = None
        self.mode: str = "memory"
        self._duckdb = None
        self._rows: list[dict[str, Any]] = []  # 始终维护的内存镜像（to_parquet/count）
        if path is not None:
            self._connect_duckdb(path)

    # -- 连接 -------------------------------------------------------------
    def _connect_duckdb(self, path: str) -> None:
        try:
            import duckdb

            # 防御：目录不可写 / path 无效时静默降级内存模式。
            parent = Path(path).parent
            if str(parent) and not os.access(str(parent), os.W_OK):
                self.mode = "memory"
                return
            con = duckdb.connect(path)
            con.execute(_DDL)
            self._duckdb = con
            self.path = str(path)
            self.mode = "duckdb"
        except Exception:
            self._duckdb = None
            self.path = None
            self.mode = "memory"

    # -- 写 ---------------------------------------------------------------
    def record(self, **row: Any) -> dict[str, Any]:
        """记录一行血缘；返回规范化后的行 dict。"""
        now = row.get("timestamp") or _now_iso()
        norm: dict[str, Any] = {
            "run_id": str(row.get("run_id") or ""),
            "factor_id": str(row.get("factor_id") or ""),
            "dataset": str(row.get("dataset") or ""),
            "concept": str(row.get("concept") or ""),
            "source_snapshot": str(row.get("source_snapshot") or ""),
            "experiment_snapshot": str(row.get("experiment_snapshot") or ""),
            "principal": str(row.get("principal") or ""),
            "build_sha": str(row.get("build_sha") or ""),
            "timestamp": str(now),
        }
        extra = row.get("extra") or {}
        norm["extra"] = dict(extra) if isinstance(extra, Mapping) else {}
        self._rows.append(norm)
        if self.mode == "duckdb" and self._duckdb is not None:
            self._duckdb.execute(
                "INSERT INTO lineage VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    norm["run_id"],
                    norm["factor_id"],
                    norm["dataset"],
                    norm["concept"],
                    norm["source_snapshot"],
                    norm["experiment_snapshot"],
                    norm["principal"],
                    norm["build_sha"],
                    norm["timestamp"],
                    json.dumps(norm["extra"], ensure_ascii=False, sort_keys=True),
                ],
            )
        return dict(norm)

    # -- 查 ---------------------------------------------------------------
    def factors_for_dataset(self, dataset: str) -> list[dict[str, Any]]:
        """该 dataset 被哪些 factor 读过（去重，按时间倒序）。"""
        rows = self.search(dataset=dataset)
        seen: dict[str, dict[str, Any]] = {}
        for r in rows:
            fid = r.get("factor_id") or ""
            if fid and fid not in seen:
                seen[fid] = r
        return sorted(seen.values(), key=lambda r: str(r.get("timestamp") or ""), reverse=True)

    def datasets_for_factor(self, factor_id: str) -> list[dict[str, Any]]:
        """该 factor 读过哪些 dataset（去重）。"""
        rows = self.search(factor_id=factor_id)
        seen: dict[str, dict[str, Any]] = {}
        for r in rows:
            ds = r.get("dataset") or ""
            if ds and ds not in seen:
                seen[ds] = r
        return sorted(seen.values(), key=lambda r: str(r.get("timestamp") or ""), reverse=True)

    def readers_of_source(self, source_snapshot: str) -> list[dict[str, Any]]:
        """某个 source snapshot 被谁（run / factor / principal）读过。"""
        return self.search(source_snapshot=source_snapshot)

    def search(
        self,
        factor_id: str | None = None,
        dataset: str | None = None,
        principal: str | None = None,
        experiment_snapshot: str | None = None,
        source_snapshot: str | None = None,
    ) -> list[dict[str, Any]]:
        """按任意组合过滤；返回行 dict 列表（extra 反序列化为 dict）。"""
        filters: list[tuple[str, str]] = []
        for key, value in (
            ("factor_id", factor_id),
            ("dataset", dataset),
            ("principal", principal),
            ("experiment_snapshot", experiment_snapshot),
            ("source_snapshot", source_snapshot),
        ):
            if value is not None:
                filters.append((key, str(value)))
        if self.mode == "duckdb" and self._duckdb is not None and filters:
            clause = " AND ".join(f"{key} = ?" for key, _ in filters)
            params = [value for _, value in filters]
            try:
                rel = self._duckdb.execute(
                    f"SELECT * FROM lineage WHERE {clause} ORDER BY timestamp DESC",
                    params,
                ).fetchall()
                return [self._row_to_dict(t) for t in rel]
            except Exception:
                pass  # 查询失败 → 回退内存镜像
        out = [dict(r) for r in self._rows]
        for key, value in filters:
            out = [r for r in out if str(r.get(key) or "") == value]
        out.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
        return out

    def _row_to_dict(self, values: Sequence) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        raw_extra = values[9] if len(values) > 9 else None
        if raw_extra:
            try:
                extra = json.loads(raw_extra)
            except (TypeError, ValueError):
                extra = {}
        return {
            "run_id": values[0],
            "factor_id": values[1],
            "dataset": values[2],
            "concept": values[3],
            "source_snapshot": values[4],
            "experiment_snapshot": values[5],
            "principal": values[6],
            "build_sha": values[7],
            "timestamp": values[8],
            "extra": extra,
        }

    def count(self) -> int:
        return len(self._rows)

    # -- 导出 -------------------------------------------------------------
    def to_parquet(self, path: str) -> str:
        """把全部血缘导出为 parquet 文件（DuckDB 模式也走内存镜像，结果一致）。"""
        table = self._to_arrow()
        out = str(path)
        pq.write_table(table, out)
        return out

    def _to_arrow(self) -> pa.Table:
        arrays: dict[str, list[Any]] = {col: [] for col in _LINEAGE_COLUMNS}
        for r in self._rows:
            for col in _LINEAGE_COLUMNS:
                if col == "extra":
                    arrays[col].append(
                        json.dumps(r.get("extra") or {}, ensure_ascii=False, sort_keys=True)
                    )
                else:
                    arrays[col].append(r.get(col, ""))
        return pa.table(arrays)

    def __len__(self) -> int:
        return self.count()


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 读取路径血缘摄取
# ---------------------------------------------------------------------------
def _lineage_of(result: Any) -> Any:
    """从 ReadResult / ReadHandle 上取 lineage（防御性）。"""
    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        return lineage
    # 部分 fake / 聚合结果可能把 lineage 塞在 ``_lineage``。
    return getattr(result, "_lineage", None)


def ingest_read_lineage(
    store: Any,
    lineage_store: LineageStore,
    result: Any,
    *,
    run_id: str,
    factor_id: str | None = None,
    principal: str | None = None,
    experiment_snapshot: str | None = None,
) -> dict[str, Any]:
    """从一次读取结果抽取血缘并记录。

    抽取 ``ReadResult`` / ``ReadHandle`` 的 ``lineage``（dataset / columns /
    snapshot / build_sha）与 ``snapshot``（snapshot_id），落一行：
    ``{run_id, factor_id, dataset, concept, source_snapshot,
    experiment_snapshot, principal, build_sha, timestamp, extra}``。
    """
    lineage = _lineage_of(result)
    snapshot = getattr(result, "snapshot", None)

    dataset = getattr(lineage, "dataset", None) or getattr(snapshot, "dataset", None)
    if not dataset:
        dataset = getattr(result, "dataset", None)

    columns = tuple(getattr(lineage, "columns", ()) or ())
    build_sha = getattr(lineage, "build_sha", None) or getattr(snapshot, "build_sha", None)
    source_snapshot = getattr(snapshot, "snapshot_id", None) or build_sha or dataset

    if principal is None and store is not None:
        sp = getattr(store, "security_principal", None)
        if sp is not None:
            try:
                principal = str(getattr(sp, "name", None) or sp)
            except Exception:
                principal = None

    extra: dict[str, Any] = {
        "time_range": getattr(lineage, "time_range", None),
        "instrument_filter": getattr(lineage, "instrument_filter", None),
        "params": list(getattr(lineage, "params", ()) or ()),
    }
    concept = ",".join(columns) if columns else (str(dataset) if dataset else "")

    row = lineage_store.record(
        run_id=run_id,
        factor_id=factor_id,
        dataset=str(dataset or ""),
        concept=concept,
        source_snapshot=str(source_snapshot or ""),
        experiment_snapshot=experiment_snapshot,
        principal=principal,
        build_sha=str(build_sha or ""),
        extra=extra,
    )
    return row


__all__ = [
    "LineageStore",
    "ingest_read_lineage",
]
