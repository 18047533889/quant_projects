"""
data_access.read.pit_event_index —— 美股财务 PIT 物理索引（#8）

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
      ``_pit_event_index.parquet`` sidecar；
    - ``prune_paths_by_filing_range(...)`` 用索引按 filing_date 精准返回文件
      子集（即使文件按 period_end 命名）。

COS 原始布局不动；本索引是本地 serving 层的优化物。更进一步的
filing_year/month 分区物化湖见 ``docs/PIT_SERVING_LAYOUT.md``。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import ValidationError

_INDEX_FILENAME = "_pit_event_index.parquet"


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


class PITEventIndex:
    """美股财务 (ticker, filing_date, period_end, timeframe) → 文件 的倒排索引。"""

    def __init__(self, records: Sequence[PITEventRecord] | None = None) -> None:
        self.records = list(records or ())
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
        """按 filing_date / timeframe / ticker 返回匹配的文件路径子集。"""
        if not self.records:
            return []
        lo, hi = filing_range if filing_range else (None, None)
        lo_ts = _as_ts(lo) if lo is not None else None
        hi_ts = _as_ts(hi) if hi is not None else None
        ticker_set = set(tickers) if tickers else None
        tf = str(timeframe).lower() if timeframe else None
        out: set[str] = set()
        for r in self.records:
            if ticker_set is not None and r.ticker not in ticker_set:
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
    availability_column、period_end=period_column）。美股财务 E2 契约自带这些。
    返回构建好的索引；sidecar 写在数据集根 ``_pit_event_index.parquet``。
    """
    import pyarrow.parquet as pq

    from data_access.cos_contract import get_cos_contract

    ds = store._registry.get(dataset)
    contract = get_cos_contract(dataset)
    inst_col = ticker_column or ds.instrument_column or (
        contract.instrument_column if contract else None
    )
    filing_col = filing_column or (
        contract.availability_column
        if contract
        else None
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

    # 枚举具体文件（glob() 表函数，本地 serving 优化层），再逐文件 pyarrow 读列，
    # 精确记录每行源文件（file_path）。构建期间不走 read_result 的 snapshot/budget
    # 语义（索引是 serving 层元数据）。
    paths = store._prepare_dataset_read(
        ds, time_range=time_range, params={}, instrument_filter=None
    )
    import pyarrow.parquet as pq

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

    records: list[PITEventRecord] = []
    read_cols = [c for c in (inst_col, filing_col, period_col, timeframe_column) if c]
    for fp in files:
        try:
            t = pq.read_table(fp, columns=read_cols)
        except Exception:
            continue
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
                    row_group=None,
                )
            )
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    idx = PITEventIndex(records)
    try:
        pq.write_table(idx.to_arrow(), str(index_path))
    except Exception:
        pass  # sidecar 写失败不阻塞（只影响优化，不影响正确性）
    return idx


def load_pit_event_index(path: Any) -> PITEventIndex:
    """从 sidecar parquet 加载索引。"""
    import pyarrow.parquet as pq

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
    return PITEventIndex(records)


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

    无索引时回退全量路径（调用方自行处理）。返回按文件聚合的路径集合。
    """
    path = _index_path_for(store, dataset)
    if path is None:
        return []
    idx = load_pit_event_index(path)
    return idx.prune_paths(
        filing_range=filing_range,
        timeframe=timeframe,
        tickers=tickers,
    )


__all__ = [
    "PITEventRecord",
    "PITEventIndex",
    "build_pit_event_index",
    "load_pit_event_index",
    "prune_paths_by_filing_range",
]
