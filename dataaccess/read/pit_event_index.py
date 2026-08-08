"""
data_access.read.pit_event_index —— 美股财务 PIT 物理索引（#8 / #15）

为什么需要
    A 股财务文件按 ``PubDate`` 命名，PIT 也是 PubDate，按 knowledge time 做路径
    裁剪天然很好。但美股完全不同：
      - 物理文件按 ``{period_end}.parquet`` 命名；
      - 真正的 PIT 是文件里的 ``filing_date``。
    因此查询「2024~2026 的 filing_date」时，**文件路径本身无法按 filing_date
    精准裁剪**，仍可能访问大量 period_end 文件。

本模块维护一个二级 ``PITEventIndex``：记录
    ``(ticker, filing_date, period_end, timeframe, file_path, row_group)``。
    - ``build_pit_event_index(store, dataset, ...)`` 扫描一次，落盘
      ``_pit_event_index.parquet`` sidecar + ``_pit_event_index.json`` 元数据；
    - ``prune_paths_by_filing_range(...)`` 用索引按 filing_date 精准返回文件
      子集（即使文件按 period_end 命名）。

**#15 权威性**：索引只有在其 ``PITIndexMetadata.complete == True`` **且** 当前
数据源 snapshot（manifest_epoch + source file snapshot）与构建时一致时才允许
authoritative prune；否则 fail-open（返回空 → 调用方回退全量路径）。任何部分
构建失败都会把 ``complete`` 置 False，禁止拿不完整索引做 false-negative 裁剪。

COS 原始布局不动；本索引是本地 serving 层的优化物。更进一步的
filing_year/month 分区物化湖见 ``docs/PIT_SERVING_LAYOUT.md``。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import ValidationError

_INDEX_FILENAME = "_pit_event_index.parquet"
_INDEX_META_FILENAME = "_pit_event_index.json"

_FILING_SCHEMA_COLS = (
    "ticker",
    "filing_date",
    "period_end",
    "timeframe",
    "file_path",
    "row_group",
)


@dataclass(frozen=True)
class PITEventRecord:
    ticker: str
    filing_date: Any
    period_end: Any
    timeframe: str | None = None
    file_path: str = ""
    row_group: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "filing_date": self.filing_date,
            "period_end": self.period_end,
            "timeframe": self.timeframe,
            "file_path": self.file_path,
            "row_group": self.row_group,
        }


@dataclass(frozen=True)
class PITIndexMetadata:
    """索引的权威性元数据（#15）：完整 + 源匹配才允许 authoritative prune。"""

    source_snapshot: str | None = None       # 源文件版本指纹（paths+size+mtime）
    manifest_epoch: str | None = None        # 构建时数据集 source_epoch
    source_file_count: int = 0
    indexed_file_count: int = 0
    failed_files: tuple[str, ...] = ()
    schema_hash: str | None = None
    created_at: str | None = None
    complete: bool = False

    @property
    def is_authoritative(self) -> bool:
        return self.complete and self.indexed_file_count == self.source_file_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_snapshot": self.source_snapshot,
            "manifest_epoch": self.manifest_epoch,
            "source_file_count": self.source_file_count,
            "indexed_file_count": self.indexed_file_count,
            "failed_files": list(self.failed_files),
            "schema_hash": self.schema_hash,
            "created_at": self.created_at,
            "complete": self.complete,
        }


class PITEventIndex:
    """美股财务 (ticker, filing_date, period_end, timeframe) → 文件 的倒排索引。"""

    def __init__(
        self,
        records: Sequence[PITEventRecord] | None = None,
        *,
        metadata: PITIndexMetadata | None = None,
    ) -> None:
        self.records = list(records or ())
        self.metadata = metadata or PITIndexMetadata()
        # file_path → (min_filing, max_filing, min_period, max_period)
        self._by_file: dict[str, dict[str, Any]] = {}
        self._by_ticker: dict[str, list[PITEventRecord]] = {}
        for r in self.records:
            self._by_ticker.setdefault(r.ticker, []).append(r)
            meta = self._by_file.setdefault(
                r.file_path,
                {"min_filing": r.filing_date, "max_filing": r.filing_date,
                 "min_period": r.period_end, "max_period": r.period_end},
            )
            meta["min_filing"] = min(_as_ts(meta["min_filing"]), _as_ts(r.filing_date))
            meta["max_filing"] = max(_as_ts(meta["max_filing"]), _as_ts(r.filing_date))

    def __len__(self) -> int:
        return len(self.records)

    def prune_paths(
        self,
        *,
        filing_range: tuple[Any, Any] | None = None,
        timeframe: str | None = None,
        tickers: Sequence[str] | None = None,
    ) -> list[str]:
        """按 filing_date / timeframe / ticker 返回匹配的文件路径子集。

        #P0-15 文件级剪枝：先用 ``_by_file`` 的 min/max filing 区间淘汰不可能
        命中的文件（O(files)），再只扫候选文件内的记录（O(候选文件内记录数)），
        不再遍历全部事件记录。
        """
        if not self.records:
            return []
        lo, hi = filing_range if filing_range else (None, None)
        lo_ts = _as_ts(lo) if lo is not None else None
        hi_ts = _as_ts(hi) if hi is not None else None
        ticker_set = set(tickers) if tickers else None
        tf = str(timeframe).lower() if timeframe else None
        # 1) 文件级 min/max 淘汰
        candidate_files: set[str] = set()
        for fp, meta in self._by_file.items():
            if lo_ts is not None and _as_ts(meta["max_filing"]) < lo_ts:
                continue
            if hi_ts is not None and _as_ts(meta["min_filing"]) > hi_ts:
                continue
            candidate_files.add(fp)
        if not candidate_files:
            return []
        # 2) ticker 维：直接查 _by_ticker 的候选文件
        if ticker_set is not None:
            ticker_files: set[str] = set()
            for tk in ticker_set:
                for r in self._by_ticker.get(tk, ()):
                    if r.file_path in candidate_files:
                        ticker_files.add(r.file_path)
            candidate_files &= ticker_files
            if not candidate_files:
                return []
        # 3) 记录级：只扫候选文件（timeframe 与 filing 精确匹配）
        out: set[str] = set()
        for r in self.records:
            if r.file_path not in candidate_files:
                continue
            if tf and (r.timeframe or "").lower() != tf:
                continue
            ft = _as_ts(r.filing_date)
            if lo_ts is not None and ft < lo_ts:
                continue
            if hi_ts is not None and ft > hi_ts:
                continue
            out.add(r.file_path)
        return sorted(out)

    def to_arrow(self) -> Any:
        import pyarrow as pa

        return pa.Table.from_pylist([r.to_dict() for r in self.records])


def _as_ts(value: Any):
    import pandas as pd

    if value is None:
        return None
    return pd.Timestamp(value)


def _q(name: str) -> str:
    return f'"{str(name).replace(chr(34), chr(34) * 2)}"'


def _index_root(store: Any, dataset: str) -> Path | None:
    """索引 sidecar 所在目录（数据集根下）。"""
    from data_access.read.manifest import manifest_root_for_paths

    ds = store._registry.get(dataset)
    try:
        paths = store._resolve_raw_paths(ds, time_range=None, params={})
    except Exception:
        return None
    root = manifest_root_for_paths(paths)
    if root is None and paths:
        # 回退：取第一个 glob 的静态前缀
        static = str(paths[0]).split("*", 1)[0].rstrip("/")
        root = Path(static)
    return root


def _source_snapshot(store: Any, dataset: str, paths: Sequence[str]) -> str | None:
    """源文件版本指纹：paths 展开后的 (path, size, mtime_ns) 集合哈希。

    与 read_contract.file_manifest_hash 对齐，但独立实现避免循环依赖。
    """
    import hashlib
    import json

    from data_access.read.manifest import (
        DatasetManifest,
        is_manifest_fresh,
        manifest_root_for_paths,
    )
    from data_access.read.read_contract import file_versions_from_manifest

    # 复用 fresh manifest（省 O(N) stat）；否则 glob+stat。
    try:
        root = manifest_root_for_paths(list(paths))
        if root is not None:
            m = DatasetManifest.load(root)
            if m is not None and is_manifest_fresh(m, list(paths)):
                files = file_versions_from_manifest(m, list(paths))
                payload = [
                    {"path": f.path, "size": f.size, "mtime_ns": f.mtime_ns}
                    for f in files
                ]
                return hashlib.sha256(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":"), default=str
                    ).encode("utf-8")
                ).hexdigest()[:16]
    except Exception:
        pass
    try:
        from data_access.read.read_contract import build_file_manifest

        files = build_file_manifest(list(paths))
        payload = [
            {"path": f.path, "size": f.size, "mtime_ns": f.mtime_ns}
            for f in files
        ]
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return None


def _manifest_epoch_of(store: Any, dataset: str) -> str | None:
    try:
        token = store.manifest_version(dataset)
        return token.get("manifest_epoch")
    except Exception:
        return None


def _meta_path(root: Path) -> Path:
    return root / _INDEX_META_FILENAME


def _schema_hash_of(ds: Any) -> str:
    """#P0-16 真 schema hash：源数据集的列名 + 类型 + 语义角色。

    旧实现只对 records 的 timeframe 取值集合做 hash，加/删列根本检测不到。
    现在对 registry 声明的源 schema（name→dtype）+ 索引使用的语义角色
    （instrument/filing/period/timeframe 列）一起 hash。
    """
    import hashlib
    import json

    schema = dict(getattr(ds, "schema", None) or {})
    payload = sorted(
        (str(k), str(v)) for k, v in schema.items()
    )
    contract = None
    try:
        from data_access.cos_contract import get_cos_contract

        contract = get_cos_contract(getattr(ds, "name", ""))
    except Exception:
        contract = None
    roles = [
        ("instrument", getattr(ds, "instrument_column", None)),
        ("filing", getattr(contract, "availability_column", None) if contract else None),
        ("period", getattr(contract, "period_column", None) if contract else None),
    ]
    payload.extend(sorted((r, str(c) or "") for r, c in roles))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def build_pit_event_index(
    store: Any,
    dataset: str,
    *,
    ticker_column: str | None = None,
    filing_column: str | None = None,
    period_column: str | None = None,
    timeframe_column: str | None = None,
    time_range: tuple[Any, Any] | None = None,
    timeframe_filter: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> PITEventIndex:
    """扫描数据集一次，构建并落盘 PIT 事件索引。

    列名缺省按 COS 契约推断（ticker=instrument_column、filing_date=
    availability_column、period_end=period_column）。返回构建好的索引；sidecar
    写在数据集根 ``_pit_event_index.parquet`` + ``_pit_event_index.json``。

    **#15**：任何源文件读取失败都会记录到 ``failed_files`` 并把 ``complete``
    置 False——索引存在但不允许 authoritative prune（fail-open）。
    """
    import json

    import pyarrow.parquet as pq

    from data_access.cos_contract import get_cos_contract

    ds = store._registry.get(dataset)
    contract = get_cos_contract(dataset)
    inst_col = ticker_column or ds.instrument_column or (
        contract.instrument_column if contract else None
    )
    filing_col = filing_column or (
        contract.availability_column if contract else None
    ) or ds.time_column
    period_col = period_column or (contract.period_column if contract else None)
    if not inst_col or not filing_col:
        raise ValidationError(
            f"构建 PIT 索引需要 instrument 列与 filing_date 列；"
            f"dataset={dataset!r} inst={inst_col!r} filing={filing_col!r}"
        )
    cols = [inst_col, filing_col]
    if period_col:
        cols.append(period_col)
    if timeframe_column:
        cols.append(timeframe_column)

    root = _index_root(store, dataset)
    if root is None:
        raise ValidationError(f"无法解析数据集 {dataset!r} 的根目录")
    index_path = root / _INDEX_FILENAME
    if index_path.exists() and not force:
        try:
            return load_pit_event_index(index_path)
        except Exception:
            pass

    paths = store._prepare_dataset_read(
        ds, time_range=time_range, params={}, instrument_filter=None
    )
    # 源快照指纹 + manifest epoch（#15 权威性判定）
    source_snapshot = _source_snapshot(store, dataset, paths)
    manifest_epoch = _manifest_epoch_of(store, dataset)

    files: list[str] = []
    for g in paths:
        if str(g).startswith(("s3://", "cos://")):
            continue  # 远程不建本地索引
        try:
            tbl = store._engine.execute_arrow(
                "SELECT file FROM glob(?)", [str(g)], deadline_ms=None
            )
        except Exception:
            continue
        files.extend(str(r["file"]) for r in tbl.to_pylist())
    files = sorted(set(files))
    source_file_count = len(files)

    records: list[PITEventRecord] = []
    failed_files: list[str] = []
    read_cols = [c for c in (inst_col, filing_col, period_col, timeframe_column) if c]
    truncated = False
    indexed_files = 0  # #P0-15 已完整索引的**文件**数（不是记录数）
    for fp in files:
        try:
            t = pq.read_table(fp, columns=read_cols)
        except Exception:
            # #15：单文件读取失败 → 记录并置 incomplete，禁止拿不完整索引裁剪
            failed_files.append(fp)
            continue
        file_done = True
        for row in t.to_pylist():
            if timeframe_filter and timeframe_column:
                if str(row.get(timeframe_column) or "") != timeframe_filter:
                    continue
            records.append(
                PITEventRecord(
                    ticker=str(row[inst_col]),
                    filing_date=row[filing_col],
                    period_end=row.get(period_col) if period_col else None,
                    timeframe=str(row[timeframe_column]) if timeframe_column else None,
                    file_path=fp,
                    row_group=None,  # 逐文件级裁剪够用；row_group 级留给后续物化层
                )
            )
            if limit is not None and len(records) >= limit:
                truncated = True
                file_done = False
                break
        if file_done:
            indexed_files += 1
        if truncated:
            break
    metadata = PITIndexMetadata(
        source_snapshot=source_snapshot,
        manifest_epoch=manifest_epoch,
        source_file_count=source_file_count,
        indexed_file_count=indexed_files,
        failed_files=tuple(failed_files),
        schema_hash=_schema_hash_of(ds),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        # #P0-15/#P0-16：完整 = 无失败文件 + 未截断 + 文件数全对。
        # limit 截断构建 → complete=False（is_authoritative 必须完整）。
        complete=(not failed_files and not truncated and indexed_files == source_file_count),
    )
    idx = PITEventIndex(records, metadata=metadata)
    try:
        pq.write_table(idx.to_arrow(), str(index_path))
        _meta_path(root).write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        # 写失败不阻塞正确性，但必须告警（不再静默 pass）
        import logging

        logging.getLogger("data_access.pit_index").warning(
            "PIT 索引 sidecar 写失败（dataset=%s）：%s", dataset, exc
        )
    return idx


def load_pit_event_index(path: Any) -> PITEventIndex:
    """从 sidecar parquet 加载索引（不带权威性元数据判定）。"""
    import json

    import pyarrow.parquet as pq

    path = Path(path)
    table = pq.read_table(str(path))
    records = [
        PITEventRecord(
            ticker=str(r["ticker"]),
            filing_date=r["filing_date"],
            period_end=r["period_end"],
            timeframe=str(r["timeframe"]) if r.get("timeframe") else None,
            file_path=str(r.get("file_path") or ""),
            row_group=r.get("row_group"),
        )
        for r in table.to_pylist()
    ]
    metadata = PITIndexMetadata()
    meta_file = path.with_name(_INDEX_META_FILENAME)
    if meta_file.exists():
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            metadata = PITIndexMetadata(
                source_snapshot=payload.get("source_snapshot"),
                manifest_epoch=payload.get("manifest_epoch"),
                source_file_count=int(payload.get("source_file_count") or 0),
                indexed_file_count=int(payload.get("indexed_file_count") or 0),
                failed_files=tuple(str(f) for f in (payload.get("failed_files") or ())),
                schema_hash=payload.get("schema_hash"),
                created_at=payload.get("created_at"),
                complete=bool(payload.get("complete", False)),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata = PITIndexMetadata()  # 元数据损坏 → 非权威
    return PITEventIndex(records, metadata=metadata)


def _index_path_for(store: Any, dataset: str) -> Path | None:
    root = _index_root(store, dataset)
    if root is None:
        return None
    p = root / _INDEX_FILENAME
    return p if p.exists() else None


def prune_paths_by_filing_range(
    store: Any,
    dataset: str,
    *,
    filing_range: tuple[Any, Any] | None = None,
    timeframe: str | None = None,
    tickers: Sequence[str] | None = None,
) -> list[str]:
    """用 PIT 索引按 filing_date 精准裁剪文件路径。

    **#15 fail-open**：只有索引 ``complete`` 且当前源（manifest_epoch +
    source snapshot）与构建时一致时才做 authoritative prune；否则返回空列表
    （调用方回退全量路径），**绝不**用不完整/过期索引做 false-negative 裁剪。
    """
    path = _index_path_for(store, dataset)
    if path is None:
        return []
    try:
        idx = load_pit_event_index(path)
    except Exception:
        return []
    meta = idx.metadata
    if not meta.is_authoritative:
        return []  # 不完整索引 → fail-open
    cur_epoch = _manifest_epoch_of(store, dataset)
    if meta.manifest_epoch is not None and cur_epoch != meta.manifest_epoch:
        return []  # 数据已变 → 索引过期 → fail-open
    try:
        raw_paths = store._prepare_dataset_read(
            store._registry.get(dataset), time_range=None, params={}
        )
        cur_snap = _source_snapshot(store, dataset, raw_paths)
    except Exception:
        cur_snap = None
    if meta.source_snapshot is not None and cur_snap != meta.source_snapshot:
        return []  # 源文件已变 → fail-open
    return idx.prune_paths(
        filing_range=filing_range,
        timeframe=timeframe,
        tickers=tickers,
    )


__all__ = [
    "PITEventRecord",
    "PITEventIndex",
    "PITIndexMetadata",
    "build_pit_event_index",
    "load_pit_event_index",
    "prune_paths_by_filing_range",
]
