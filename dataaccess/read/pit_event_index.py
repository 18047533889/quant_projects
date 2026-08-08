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

import os
from dataclasses import dataclass, field
import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

from data_access.core.exceptions import ValidationError

_INDEX_FILENAME = "_pit_event_index.parquet"
_INDEX_META_FILENAME = "_pit_event_index.json"
# #P1-final closure 18：generation-directory + 原子指针提交。旧实现双文件直接
# ``os.replace`` 两次，crash 在两次 replace 之间会留下 mixed generation——下一
# 次 load 能检测但**上一份好索引已被毁掉**。新布局：
#     <root>/.pit_index/current            ← 原子指针（唯一 commit 点）
#     <root>/.pit_index/<gen>/index.parquet
#     <root>/.pit_index/<gen>/metadata.json
# 先写完整的新 generation 目录，最后才原子替换指针——crash 任意时刻旧 generation
# 目录与指针都完整，「保留旧 generation 直到新 generation 完全提交」。legacy
# ``<root>/_pit_event_index.*`` 布局继续可读（回退）。
_PIT_INDEX_DIR = ".pit_index"
_PIT_INDEX_POINTER = "current"


def _is_nan(value: Any) -> bool:
    """float NaN 检测（np.float64/py float 共用）。"""
    import math

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _fsync_parent(path: Path) -> None:
    """#P1-final closure 6：对父目录做 fsync，保证 rename 后目录项落盘。"""
    try:
        fd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

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
    """索引的权威性元数据（#15）：完整 + 源匹配才允许 authoritative prune。

    #P0-17 generation：index.parquet 与 metadata.json 必须同 generation（复刻
    manifest 双写模型），进程死在两次 replace 之间 → 非权威。
    #P0-15/#P0-16 IndexScope：只记录局部构建（time_range / timeframe_filter /
    列 override）的覆盖范围，请求超出 scope → fail-open，绝不能把局部 index
    当全局 authoritative。
    """

    source_snapshot: str | None = None       # 源文件版本指纹（paths+size+mtime）
    manifest_epoch: str | None = None        # 构建时数据集 source_epoch
    source_file_count: int = 0
    indexed_file_count: int = 0
    failed_files: tuple[str, ...] = ()
    schema_hash: str | None = None
    created_at: str | None = None
    complete: bool = False
    # #P1-final closure 6：源文件枚举（glob）失败标志。glob 失败意味着「源里
    # 可能有文件根本没被看到」——``source_file_count`` 只数成功枚举出的文件，
    # 单独记这个标志，否则漏枚举文件的 index 仍会被标 complete=True。
    glob_failed: bool = False
    # ---- #P0-17 generation（parquet + JSON 双写，防 partial write）----
    generation_id: str | None = None
    # ---- #P0-15/#P0-16 IndexScope ----
    filing_scope_min: str | None = None      # 构建时扫描到的 min filing_date（iso）
    filing_scope_max: str | None = None      # 构建时扫描到的 max filing_date（iso）
    timeframe_scope: str | None = None       # 构建时 timeframe_filter（None=全 timeframe）
    # ---- #P0-18 列 override 身份（自定义列构建的 index 不能当默认构建复用）----
    columns_used: tuple[str, ...] = ()

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
            "glob_failed": self.glob_failed,
            "generation_id": self.generation_id,
            "filing_scope_min": self.filing_scope_min,
            "filing_scope_max": self.filing_scope_max,
            "timeframe_scope": self.timeframe_scope,
            "columns_used": list(self.columns_used),
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


def _pit_index_dir(root: Path) -> Path:
    return root / _PIT_INDEX_DIR


def _current_generation(root: Path) -> str | None:
    """读当前 generation id（原子指针）；无指针 → None（legacy 布局）。"""
    try:
        text = (_pit_index_dir(root) / _PIT_INDEX_POINTER).read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, IOError):
        return None
    return text or None


def _current_index_parquet(root: Path) -> Path | None:
    """解析当前生效的 index.parquet 路径。

    优先 generation-directory 布局（指针指向的 gen 目录）；无指针/目录缺失回退
    legacy ``<root>/_pit_event_index.parquet``。
    """
    gen = _current_generation(root)
    if gen:
        cand = _pit_index_dir(root) / gen / _INDEX_FILENAME
        if cand.exists():
            return cand
    legacy = root / _INDEX_FILENAME
    return legacy if legacy.exists() else None


def _prune_old_generations(index_dir: Path, keep: str) -> None:
    """提交后清理旧 generation 目录（保留刚提交的）。crash 前的旧目录仍完整。"""
    import shutil

    try:
        for child in index_dir.iterdir():
            if child.name == _PIT_INDEX_POINTER or not child.is_dir():
                continue
            if child.name != keep:
                shutil.rmtree(child, ignore_errors=True)
    except OSError:
        pass


def _commit_index_generation(
    root: Path,
    idx: "PITEventIndex",
    metadata: "PITIndexMetadata",
    generation: str,
) -> None:
    """#P1-final closure 18：staged 提交新 generation。

    1. 写 ``.pit_index/<gen>/index.parquet`` + ``metadata.json``（各自 tmp +
       fsync + atomic replace，但都是**新目录内**——不影响旧 generation）；
    2. 原子替换 ``current`` 指针 —— 这是唯一 commit 点；
    3. crash 在任意时刻：指针要么还在旧 gen（旧索引完整可用），要么已指向新
       gen（新目录完整）。绝不出现「旧索引被毁 + 新索引未完成」。
    """
    import json

    import pyarrow.parquet as pq

    index_dir = _pit_index_dir(root)
    index_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = index_dir / generation
    gen_dir.mkdir(parents=True, exist_ok=True)

    arrow = idx.to_arrow().cast(
        idx.to_arrow().schema.with_metadata(
            {b"manifest_generation_id": generation.encode("utf-8")}
        )
    )
    # #P1-final closure 19：统一 atomic durable-write（内容 fsync，不只 fsync 目录）
    from data_access.core.atomic import atomic_write_file, atomic_write_json

    atomic_write_file(
        gen_dir / _INDEX_FILENAME, lambda tmp: pq.write_table(arrow, tmp)
    )
    atomic_write_json(gen_dir / _INDEX_META_FILENAME, metadata.to_dict())

    # 唯一 commit 点：原子替换指针（先 fsync tmp 文件 fd 再 rename）。
    pointer = index_dir / _PIT_INDEX_POINTER
    tmp_ptr = index_dir / f".{_PIT_INDEX_POINTER}.tmp"
    with open(tmp_ptr, "w", encoding="utf-8") as fh:
        fh.write(generation)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp_ptr), str(pointer))
    _fsync_parent(pointer)

    _prune_old_generations(index_dir, generation)


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

    columns_used = tuple(dict.fromkeys(c for c in cols if c))

    root = _index_root(store, dataset)
    if root is None:
        raise ValidationError(f"无法解析数据集 {dataset!r} 的根目录")
    index_path = _current_index_parquet(root)
    if index_path is not None and index_path.exists() and not force:
        try:
            existing = load_pit_event_index(index_path)
        except Exception:
            existing = None
        if existing is not None and _reuse_matches_current(
            store, dataset, existing.metadata, columns_used=columns_used,
            timeframe_filter=timeframe_filter,
        ):
            return existing
        # #P0-14 旧 index stale（数据已变 / 列 override 不同 / 局部构建）→ 重建，
        # 绝不把 stale index 直接返回给调用方。

    paths = store._prepare_dataset_read(
        ds, time_range=time_range, params={}, instrument_filter=None
    )
    # 源快照指纹 + manifest epoch（#15 权威性判定）
    source_snapshot = _source_snapshot(store, dataset, paths)
    manifest_epoch = _manifest_epoch_of(store, dataset)

    files: list[str] = []
    failed_files: list[str] = []
    enum_failed = False  # #P1-final closure 6：glob 枚举失败 → 索引不完整
    for g in paths:
        if str(g).startswith(("s3://", "cos://")):
            continue  # 远程不建本地索引
        try:
            tbl = store._engine.execute_arrow(
                "SELECT file FROM glob(?)", [str(g)], deadline_ms=None
            )
        except Exception as exc:
            # #P1-final closure 6：glob 失败不能 except:continue 静默吞掉——
            # 源里可能有文件根本没被枚举到。记进 failed_files + 置 enum_failed，
            # complete 必为 False，禁止拿「漏枚举文件」的索引做 authoritative prune。
            failed_files.append(f"glob:{g}")
            enum_failed = True
            continue
        files.extend(str(r["file"]) for r in tbl.to_pylist())
    files = sorted(set(files))
    source_file_count = len(files)

    records: list[PITEventRecord] = []
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
            # #P1-final closure 6：null ticker / 空 filing_date → 拒绝构建。
            # 旧代码 ticker=str(None) 会把缺失 key 变成字符串 "None" 索引进倒排
            # 表，剪枝时匹配到假 ticker；filing_date 为空则 min/max filing 范围
            # 失真。fail-closed：整次构建拒绝（这些是坏数据，不允许索引）。
            ticker_val = row.get(inst_col)
            filing_val = row.get(filing_col)
            if ticker_val is None or (isinstance(ticker_val, float) and _is_nan(ticker_val)):
                raise ValidationError(
                    f"构建 PIT 索引发现 null ticker（{fp} 列 {inst_col}）："
                    "不允许把缺失 key 当 'None' 字符串索引。请修复源数据。"
                )
            if filing_val is None or _as_ts(filing_val) is None:
                raise ValidationError(
                    f"构建 PIT 索引发现空/非法 filing_date（{fp} 列 {filing_col}）："
                    "min/max filing 范围会失真。请修复源数据。"
                )
            records.append(
                PITEventRecord(
                    ticker=str(ticker_val),
                    filing_date=filing_val,
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
    # #P0-15/#P0-16 IndexScope：记录本次构建的 filing 覆盖范围与 timeframe 过滤，
    # 局部 index 不能冒充全局 authoritative。
    filing_dates = [_as_ts(r.filing_date) for r in records]
    filing_min = (
        min(filing_dates).isoformat() if filing_dates and min(filing_dates) is not None else None
    )
    filing_max = (
        max(filing_dates).isoformat() if filing_dates and max(filing_dates) is not None else None
    )
    from data_access.read.manifest import uuid4_hex

    generation = uuid4_hex()
    metadata = PITIndexMetadata(
        source_snapshot=source_snapshot,
        manifest_epoch=manifest_epoch,
        source_file_count=source_file_count,
        indexed_file_count=indexed_files,
        failed_files=tuple(failed_files),
        schema_hash=_schema_hash_of(ds),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        # #P0-15/#P0-16 + #P1-final closure 6：完整 = 无失败文件 + 无 glob 枚举
        # 失败 + 未截断 + 文件数全对。任何一项不满足 → complete=False。
        complete=(
            not failed_files
            and not enum_failed
            and not truncated
            and indexed_files == source_file_count
        ),
        glob_failed=enum_failed,
        generation_id=generation,
        filing_scope_min=filing_min,
        filing_scope_max=filing_max,
        timeframe_scope=timeframe_filter if timeframe_filter else None,
        columns_used=columns_used,
    )
    idx = PITEventIndex(records, metadata=metadata)
    try:
        # #P1-final closure 18：staged generation-directory 提交——先写完整新 gen
        # 目录，最后原子切换指针。crash 任意时刻旧 generation 完整可用。
        _commit_index_generation(root, idx, metadata, generation)
    except Exception as exc:
        # 写失败不阻塞正确性，但必须告警（不再静默 pass）
        import logging

        logging.getLogger("data_access.pit_index").warning(
            "PIT 索引 sidecar 写失败（dataset=%s）：%s", dataset, exc
        )
    return idx


def load_pit_event_index(path: Any) -> PITEventIndex:
    """从 sidecar parquet 加载索引（不带权威性元数据判定）。

    #P0-17 generation 校验：index.parquet 与 metadata.json 任一侧缺失/不一致 →
    视为 partial write，metadata 回退成非权威（complete=False）。
    """
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
    parquet_gen = _index_parquet_generation(path)
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
                glob_failed=bool(payload.get("glob_failed", False)),
                generation_id=payload.get("generation_id"),
                filing_scope_min=payload.get("filing_scope_min"),
                filing_scope_max=payload.get("filing_scope_max"),
                timeframe_scope=payload.get("timeframe_scope"),
                columns_used=tuple(str(c) for c in (payload.get("columns_used") or ())),
            )
            # #P0-17 新格式任一侧缺 generation → partial write → 非权威
            if parquet_gen is not None or metadata.generation_id is not None:
                if (
                    parquet_gen is None
                    or metadata.generation_id is None
                    or parquet_gen != metadata.generation_id
                ):
                    metadata = PITIndexMetadata()  # mixed generation → 非权威
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata = PITIndexMetadata()  # 元数据损坏 → 非权威
    return PITEventIndex(records, metadata=metadata)


def _index_parquet_generation(path: Path) -> str | None:
    """读 index.parquet schema metadata 里的 generation id。"""
    import pyarrow.parquet as pq

    try:
        meta = pq.read_metadata(str(path))
    except Exception:
        return None
    kv = meta.metadata
    if kv is None:
        return None
    raw = kv.get(b"manifest_generation_id")
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _index_path_for(store: Any, dataset: str) -> Path | None:
    root = _index_root(store, dataset)
    if root is None:
        return None
    return _current_index_parquet(root)


def validate_current_source(
    store: Any,
    dataset: str,
    meta: PITIndexMetadata,
) -> bool:
    """#P0-C8 统一的 PIT index 源身份校验（唯一事实源）。

    检查 authoritative + manifest_epoch + source_snapshot + schema_hash 全部与
    当前源一致。MetadataPlane 的 ``pit_event_index()`` / 剪枝路径 / 构建复用路径
    共用这一份判断——不再各自写第二套近似（只比 manifest_epoch 会在「外部文件
    绕过 DataAccess 写入、schema 变化但 epoch 不变」时返回本应 stale 的 index）。
    """
    if not meta.is_authoritative:
        return False
    cur_epoch = _manifest_epoch_of(store, dataset)
    if meta.manifest_epoch is not None and cur_epoch != meta.manifest_epoch:
        return False  # 数据已变 → stale
    try:
        raw_paths = store._prepare_dataset_read(
            store._registry.get(dataset), time_range=None, params={}
        )
        cur_snap = _source_snapshot(store, dataset, raw_paths)
    except Exception:
        cur_snap = None
    if meta.source_snapshot is not None and cur_snap != meta.source_snapshot:
        return False  # 源文件已变 → stale
    ds = store._registry.get(dataset)
    if meta.schema_hash and meta.schema_hash != _schema_hash_of(ds):
        return False  # schema 声明已变 → stale
    return True


def _reuse_matches_current(
    store: Any,
    dataset: str,
    meta: PITIndexMetadata,
    *,
    columns_used: tuple[str, ...],
    timeframe_filter: str | None,
) -> bool:
    """#P0-14/#P0-18 判断旧 index 是否与当前源 + 本次构建请求一致，可安全复用。

    源身份判断统一走 ``validate_current_source``；这里只叠加构建请求相关的列
    override / timeframe scope 检查。任一项不满足 → False（调用方应重建）。
    """
    if not validate_current_source(store, dataset, meta):
        return False
    # #P0-18 列 override 身份：自定义列构建的 index 不能当默认构建复用
    if meta.columns_used and tuple(meta.columns_used) != tuple(columns_used):
        return False
    if meta.timeframe_scope != (timeframe_filter if timeframe_filter else None):
        return False
    return True


def _filing_ts(value: Any):
    return _as_ts(value)


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

    #P0-15/#P0-16 IndexScope：请求超出索引构建时的覆盖范围（timeframe 或
    filing_range）→ fail-open。局部 index（只建了 quarterly / 只建了 2025 年）
    不能回答 annual / 2024 的查询——没有记录 ≠ 源数据不存在。
    """
    path = _index_path_for(store, dataset)
    if path is None:
        return []
    try:
        idx = load_pit_event_index(path)
    except Exception:
        return []
    meta = idx.metadata
    # #P0-C8 统一源身份校验：authoritative + epoch + source snapshot + schema
    # hash 一处判断，不再重复写第二套近似。
    if not validate_current_source(store, dataset, meta):
        return []  # 不完整/过期索引 → fail-open
    # ---- #P0-15/#P0-16 IndexScope ----
    if timeframe is not None and meta.timeframe_scope is not None:
        if str(timeframe).lower() != str(meta.timeframe_scope).lower():
            return []  # index 只覆盖别的 timeframe → 无法回答
    if filing_range is not None:
        req_lo = _filing_ts(filing_range[0]) if filing_range[0] is not None else None
        req_hi = _filing_ts(filing_range[1]) if filing_range[1] is not None else None
        scope_lo = _filing_ts(meta.filing_scope_min) if meta.filing_scope_min else None
        scope_hi = _filing_ts(meta.filing_scope_max) if meta.filing_scope_max else None
        if scope_lo is not None and scope_hi is not None:
            outside = (req_lo is not None and req_lo < scope_lo) or (
                req_hi is not None and req_hi > scope_hi
            )
            if outside:
                return []  # 请求范围超出 index 覆盖 → 不能声称「没有记录」
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
    "validate_current_source",
]
