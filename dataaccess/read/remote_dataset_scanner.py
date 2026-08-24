# -*- coding: utf-8 -*-
"""R44-P0: 远程数据集扫描器 —— 节点级增量 FactorEngine 的对象存储读平面。

在 ObjectStore（``dataaccess.read.object_store``）之上做**不落本地磁盘**的
parquet/arrow 扫描：

1. ``resolve_snapshot``：从 snapshot manifest JSON
   ``{dataset, snapshot_id, objects: [{key, etag, size}]}`` 解析对象 key 集。
2. ``scan``：对每个对象做投影（fields）+ 谓词下推（date 列的 start/end、
   asset 列的 universe）并产出 ``pa.RecordBatch``，累计 ``RemoteScanStats``。
3. ``scan_to_tables``：把全部 batch 聚合成 ``pa.Table`` 列表。

谓词下推语义（R44-P0）：
    - ``start`` / ``end`` 作用于 **date 列**（字符串 ``YYYY-MM-DD`` 或 date32）；
    - ``universe`` 作用于 **asset 列**（``isin``）；
    - parquet row-group / 列裁剪是 best-effort：**失败一律降级为全量读**，
      最终结果以内存 pyarrow 过滤为准 —— 正确性绝不依赖裁剪。

包名注意：本仓库源码包是 ``dataaccess``（无下划线；``data_access`` 是
site-packages 里的已安装拷贝）。本模块一律用 ``dataaccess`` 前缀或相对导入。
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .object_store import ObjectStore

__all__ = [
    "RemoteScanPlan",
    "RemoteScanStats",
    "RemoteDatasetScanner",
    "load_default_snapshot_manifest",
]

_DEFAULT_BATCH_ROWS = 65536
_MANIFEST_KEY_PREFIX = "manifests/"
#: date 列候选（谓词下推 / row-group 裁剪用）。
_DATE_COLUMNS = ("date", "trade_date", "dt")
#: asset 列候选（universe 过滤 / row-group 裁剪用）。
_ASSET_COLUMNS = ("asset", "symbol", "ticker", "code")


# ---------------------------------------------------------------------------
# 快照 manifest 加载
# ---------------------------------------------------------------------------
def _default_manifest_loader(
    store: ObjectStore, dataset: str, snapshot_id: str
) -> dict[str, Any]:
    """从 store 读 ``manifests/{dataset}/{snapshot_id}.json``（R44-P0 约定路径）。

    manifest 缺失（对象不存在）→ 抛 ``ValueError``（fail-closed，调用方按
    领域错误处理，而不是裸 ``FileNotFoundError``）。
    """
    key = f"{_MANIFEST_KEY_PREFIX}{dataset}/{snapshot_id}.json"
    try:
        blob = store.range_read(key, offset=0, length=10 ** 9)
    except Exception as exc:
        raise ValueError(
            f"快照 manifest 不存在或不可读（R44-P0）: {key} "
            f"（dataset={dataset!r} snapshot_id={snapshot_id!r}）"
        ) from exc
    payload = json.loads(blob.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"快照 manifest 非法（非对象）: {key}")
    return payload


def load_default_snapshot_manifest(
    store: ObjectStore, dataset: str, snapshot_id: str
) -> dict[str, Any]:
    """public 入口：默认 manifest 加载器（测试与本地跑批用）。"""
    return _default_manifest_loader(store, dataset, snapshot_id)


# ---------------------------------------------------------------------------
# Plan / Stats
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteScanPlan:
    """一次远程扫描的计划（R44-P0，不可变）。

    ``object_keys`` 由 manifest 解析得到；``to_dict`` 供 planner / 成本模型
    （``estimate_remote_io``）序列化。
    """

    dataset: str
    snapshot_id: str
    fields: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    universe: tuple[str, ...] | None = None
    object_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "fields": list(self.fields),
            "start": self.start,
            "end": self.end,
            "universe": list(self.universe) if self.universe is not None else None,
            "object_keys": list(self.object_keys),
        }


@dataclass
class RemoteScanStats:
    """一次扫描的累计统计（R44-P0）。"""

    rows_read: int = 0
    bytes_read: int = 0
    object_count: int = 0
    batches: int = 0


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
class RemoteDatasetScanner:
    """在 ObjectStore 上执行投影 + 谓词下推的 parquet/arrow 扫描器（R44-P0）。

    构造参数：
        - ``store``：ObjectStore 实现（LocalObjectStore / COSObjectStore）。
        - ``manifest_loader``：``(store, dataset, snapshot_id) -> dict``；缺省用
          ``_default_manifest_loader``（``manifests/{dataset}/{snapshot_id}.json``）。
        - ``scan_fn``：可选注入的对象级扫描函数
          ``(store, key, *, fields, start, end, universe, batch_rows) -> Iterable[pa.RecordBatch]``；
          缺省用本类 ``_scan_object``（parquet row-group/列裁剪 + 全量降级）。
    """

    def __init__(
        self,
        store: ObjectStore,
        *,
        manifest_loader: Callable[..., dict[str, Any]] | None = None,
        scan_fn: Callable[..., Iterable[pa.RecordBatch]] | None = None,
    ) -> None:
        self.store = store
        self.manifest_loader = manifest_loader or _default_manifest_loader
        self.scan_fn = scan_fn
        self.last_stats = RemoteScanStats()

    # ---- manifest 解析 ----
    def resolve_snapshot(self, dataset: str, snapshot_id: str) -> list[str]:
        """解析 snapshot 的对象 key 列表（R44-P0）。

        manifest 格式: ``{dataset, snapshot_id, objects: [{key, etag, size}]}``。
        dataset/snapshot_id 不匹配 → 抛 ValueError（fail-closed）。
        """
        payload = self.manifest_loader(self.store, dataset, snapshot_id)
        if str(payload.get("dataset", "")) != dataset:
            raise ValueError(
                f"manifest dataset 不匹配: {payload.get('dataset')!r} != {dataset!r}"
            )
        if str(payload.get("snapshot_id", "")) != snapshot_id:
            raise ValueError(
                f"manifest snapshot_id 不匹配: {payload.get('snapshot_id')!r} != {snapshot_id!r}"
            )
        objects = payload.get("objects") or []
        keys: list[str] = []
        for obj in objects:
            key = obj.get("key") if isinstance(obj, dict) else None
            if not key:
                raise ValueError(f"manifest 含缺 key 的对象: {obj!r}")
            keys.append(str(key))
        return keys

    # ---- 顶层扫描 ----
    def scan(
        self,
        *,
        dataset: str,
        snapshot_id: str,
        fields: Sequence[str],
        start: str | None = None,
        end: str | None = None,
        universe: Sequence[str] | None = None,
        batch_rows: int = _DEFAULT_BATCH_ROWS,
    ) -> Iterable[pa.RecordBatch]:
        """扫描 snapshot 全部对象，产出投影 + 过滤后的 RecordBatch（R44-P0）。

        - 谓词：``start``/``end`` 作用于 date 列，``universe`` 作用于 asset 列；
        - ``fields`` 是输出列（date/asset 过滤列可不在其中）。
        - 每次 yield 前调用 ``assert_no_local_persistent_write``（STRICT_REMOTE
          下任何 prior 落盘都会 fail-closed）。
        - 统计累计到 ``last_stats``。
        """
        from .local_disk_policy import assert_no_local_persistent_write

        fields = tuple(fields)
        self.last_stats = RemoteScanStats()
        scan_fn = self.scan_fn or self._scan_object
        for key in self.resolve_snapshot(dataset, snapshot_id):
            for batch in scan_fn(
                self.store,
                key,
                fields=fields,
                start=start,
                end=end,
                universe=tuple(universe) if universe is not None else None,
                batch_rows=int(batch_rows),
            ):
                assert_no_local_persistent_write()
                self.last_stats.rows_read += batch.num_rows
                self.last_stats.batches += 1
                self.last_stats.object_count += 1
                yield batch

    def scan_to_tables(
        self,
        *,
        dataset: str,
        snapshot_id: str,
        fields: Sequence[str],
        start: str | None = None,
        end: str | None = None,
        universe: Sequence[str] | None = None,
        batch_rows: int = _DEFAULT_BATCH_ROWS,
    ) -> list[pa.Table]:
        """扫描并把全部 batch 聚合成 ``pa.Table`` 列表（R44-P0）。"""
        batches = list(
            self.scan(
                dataset=dataset,
                snapshot_id=snapshot_id,
                fields=fields,
                start=start,
                end=end,
                universe=universe,
                batch_rows=batch_rows,
            )
        )
        return [pa.Table.from_batches([b]) for b in batches]

    # ---- 对象级扫描 ----
    def _scan_object(
        self,
        store: ObjectStore,
        key: str,
        *,
        fields: tuple[str, ...],
        start: str | None,
        end: str | None,
        universe: tuple[str, ...] | None,
        batch_rows: int,
    ) -> Iterable[pa.RecordBatch]:
        """扫描单个对象（R44-P0）。

        路径：
            1) ``head`` + tail range-read 拿 parquet footer 元数据；失败 → 全量读；
            2) 有 footer：按 date 列统计裁剪 row-group；store 提供可 seek 的
               ``open_reader``（LocalObjectStore）时用
               ``pq.ParquetFile.read_row_groups(ids, columns=needed)`` 真裁剪；
            3) 否则（COS 无 seek reader）→ 全量 ``range_read`` 后内存过滤；
            4) 内存 pyarrow 过滤（严格谓词，string/date32 归一化）。
        """
        head = store.head_object(key)
        size = int(head["size"]) if head and head.get("size") is not None else None
        meta = self._try_parquet_footer(store, key) if size else None

        if meta is not None:
            names = list(meta.schema.names)
            needed = [n for n in names if n in set(fields) or n in _DATE_COLUMNS or n in _ASSET_COLUMNS]
            row_group_ids = self._prune_row_groups(meta, start=start, end=end, universe=universe)
            if row_group_ids is None:
                row_group_ids = list(range(meta.num_row_groups))
            reader = store.open_reader(key)
            if reader is not None and getattr(reader, "seek", None) is not None:
                try:
                    pf = pq.ParquetFile(reader)
                    table = pf.read_row_groups(
                        list(row_group_ids), columns=needed or None
                    )
                    self.last_stats.bytes_read += self._approx_read_bytes(
                        meta, row_group_ids, needed, names
                    )
                    try:
                        reader.close()
                    except Exception:
                        pass
                    yield from self._apply_and_yield(
                        table, fields, start, end, universe, batch_rows
                    )
                    return
                except Exception:
                    try:
                        reader.close()
                    except Exception:
                        pass
                    # 回退到全量读（不丢正确性）。

        data = self._full_read(store, key)
        table = _table_from_bytes(data)
        yield from self._apply_and_yield(table, fields, start, end, universe, batch_rows)

    # ---- parquet footer / row-group 裁剪 ----
    def _try_parquet_footer(self, store: ObjectStore, key: str) -> Any | None:
        """读对象尾部 1MB 尝试解析 parquet footer（R44-P0 best-effort）。"""
        head = store.head_object(key)
        size = (head or {}).get("size")
        if size is None:
            return None
        tail_len = min(int(size), 1 << 20)
        try:
            tail = store.range_read(key, offset=int(size) - tail_len, length=tail_len)
        except Exception:
            return None
        try:
            return pq.read_metadata(io.BytesIO(tail))
        except Exception:
            return None

    def _prune_row_groups(
        self,
        meta: Any,
        *,
        start: str | None,
        end: str | None,
        universe: tuple[str, ...] | None,
    ) -> list[int] | None:
        """用 row-group 列统计裁剪（R44-P0 best-effort）。

        返回命中的 row_group id 列表；无法裁剪（缺 date 列 / 统计缺失）返回 None
        → 调用方全量读。**只做优化，正确性由内存过滤兜底。**
        """
        names = list(meta.schema.names)
        date_col = next((n for n in _DATE_COLUMNS if n in names), None)
        asset_col = next((n for n in _ASSET_COLUMNS if n in names), None)
        has_time_pred = start is not None or end is not None
        has_universe = bool(universe)
        if not has_time_pred and not has_universe:
            return list(range(meta.num_row_groups))
        if has_time_pred and date_col is None:
            return None  # 无法裁剪 → 全量
        if has_universe and asset_col is None:
            # universe 无对应列 → 裁剪不了，读全量由内存过滤。
            return None
        s_lo = _to_cmp_str(start)
        s_hi = _to_cmp_str(end)
        out: list[int] = []
        for i in range(meta.num_row_groups):
            rg = meta.row_group(i)
            keep = True
            if has_time_pred:
                cm = _column_stats(rg, names, date_col)
                if cm is None:
                    return None  # 统计缺失 → 无法证明 → 全量
                lo = _to_cmp_str(getattr(cm, "min", None))
                hi = _to_cmp_str(getattr(cm, "max", None))
                if lo is None or hi is None:
                    return None
                if s_lo is not None and hi < s_lo:
                    keep = False
                if keep and s_hi is not None and lo > s_hi:
                    keep = False
            if keep:
                out.append(i)
        return out

    @staticmethod
    def _approx_read_bytes(
        meta: Any, row_group_ids: Sequence[int], needed: Sequence[str], names: Sequence[str]
    ) -> int:
        """近似统计实际读取字节（命中 row-group × 所需列压缩大小之和）。"""
        try:
            idxs = [names.index(n) for n in needed]
        except ValueError:
            return 0
        total = 0
        for i in row_group_ids:
            rg = meta.row_group(i)
            for j in idxs:
                ci = rg.column(j)
                total += int(ci.total_compressed_size or 0)
        return total

    def _full_read(self, store: ObjectStore, key: str) -> bytes:
        data = store.range_read(key, offset=0, length=10 ** 9)
        self.last_stats.bytes_read += len(data)
        return data

    # ---- 内存谓词应用 ----
    def _apply_and_yield(
        self,
        table: pa.Table,
        fields: tuple[str, ...],
        start: str | None,
        end: str | None,
        universe: tuple[str, ...] | None,
        batch_rows: int,
    ) -> Iterable[pa.RecordBatch]:
        expr = self._build_filter(table, start, end, universe)
        if expr is not None:
            try:
                table = table.filter(expr)
            except Exception:
                # 类型不匹配（string vs date32 等）→ 逐行 Python 兜底过滤。
                table = _python_filter(table, start, end, universe)
        if fields:
            table = table.select(list(fields))
        for batch in table.to_batches(max_chunksize=max(1, batch_rows)):
            yield batch

    def _build_filter(
        self,
        table: pa.Table,
        start: str | None,
        end: str | None,
        universe: tuple[str, ...] | None,
    ) -> Any | None:
        parts: list[Any] = []
        names = table.column_names
        if start is not None or end is not None:
            date_col = next((n for n in _DATE_COLUMNS if n in names), None)
            if date_col is not None:
                col = table[date_col]
                field = pc.field(date_col)
                if start is not None and end is not None:
                    parts.append(
                        pc.and_(
                            field >= _scalar_for(col, start),
                            field <= _scalar_for(col, end),
                        )
                    )
                elif start is not None:
                    parts.append(field >= _scalar_for(col, start))
                elif end is not None:
                    parts.append(field <= _scalar_for(col, end))
        if universe:
            asset_col = next((n for n in _ASSET_COLUMNS if n in names), None)
            if asset_col is not None:
                parts.append(pc.field(asset_col).isin(list(universe)))
        if not parts:
            return None
        acc = parts[0]
        for p in parts[1:]:
            acc = pc.and_(acc, p)
        return acc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _table_from_bytes(data: bytes) -> pa.Table:
    """bytes → pa.Table（parquet 优先，arrow/ipc 兜底；都不行则空表）。"""
    if data[:4] == b"PAR1":
        return pq.read_table(io.BytesIO(data))
    try:
        return pa.ipc.open_stream(io.BytesIO(data)).read_all()
    except Exception:
        try:
            return pq.read_table(io.BytesIO(data))
        except Exception:
            return pa.table({})


def _column_stats(rg: Any, names: Sequence[str], col_name: str) -> Any | None:
    try:
        idx = names.index(col_name)
    except ValueError:
        return None
    cm = rg.column(idx)
    st = cm.statistics
    if st is None:
        return None
    return st


def _to_cmp_str(value: Any) -> str | None:
    """把 date/datetime/str/bytes 归一化成 ISO 字符串（可跨类型比较）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return None
    if hasattr(value, "as_py"):
        try:
            return _to_cmp_str(value.as_py())
        except Exception:
            return None
    return str(value)


def _to_date(value: Any) -> date | None:
    """把 str/datetime/date 解析成 date（date 列 scalar 用）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "as_py"):
        try:
            return _to_date(value.as_py())
        except Exception:
            return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _scalar_for(col: Any, value: Any) -> Any:
    """按列类型把比较值转成 pa.scalar（date32 用 date；string 用 str）。"""
    typ = col.type
    if pa.types.is_date32(typ) or pa.types.is_date64(typ):
        return pa.scalar(_to_date(value), type=pa.date32())
    if pa.types.is_timestamp(typ):
        return pa.scalar(value)
    return pa.scalar(str(value))


def _python_filter(
    table: pa.Table,
    start: str | None,
    end: str | None,
    universe: tuple[str, ...] | None,
) -> pa.Table:
    """兜底 Python 过滤（R44-P0：任何 pushdown 失败都不丢正确性）。"""
    names = table.column_names
    date_col = next((n for n in _DATE_COLUMNS if n in names), None)
    asset_col = next((n for n in _ASSET_COLUMNS if n in names), None)
    n = table.num_rows
    keep = [True] * n
    if (start is not None or end is not None) and date_col is not None:
        values = table[date_col].to_pylist()
        lo = _to_cmp_str(start)
        hi = _to_cmp_str(end)
        for i, v in enumerate(values):
            cv = _to_cmp_str(v)
            if lo is not None and cv is not None and cv < lo:
                keep[i] = False
            elif hi is not None and cv is not None and cv > hi:
                keep[i] = False
    if universe and asset_col is not None:
        values = table[asset_col].to_pylist()
        wanted = set(universe)
        for i, v in enumerate(values):
            if keep[i] and v not in wanted:
                keep[i] = False
    if all(keep):
        return table
    mask = pa.array(keep, type=pa.bool_())
    return table.filter(mask)
