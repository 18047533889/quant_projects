"""
data_access.read.manifest —— 数据集物理元数据清单（_manifest.parquet）

职责
    1. 每个数据集根放一个 ``_manifest.parquet``，记录每个文件：
       path / rows / bytes / min_time / max_time / min_instrument /
       max_instrument / schema_hash / mtime_ns / etag
    2. ``prune_by_time`` / ``prune_by_instruments``：读前按 manifest 的
       min/max 直接裁剪文件列表，避免 ``**/*.parquet`` 全量 glob + 逐文件
       ``pq.read_metadata()``
    3. 可扩展 row-group 级 statistics（column min/max/null_count）

设计要点
    1. min/max 存为规范化字符串（date/datetime→isoformat，int→str），
       同格式下字符串序 == 时间序，可直接比较。
    2. 仅在 manifest 存在且新鲜时使用；``is_fresh`` 用文件名级 glob 做
       数量比对（不读 footer），计数不一致视为过期。
    3. 构建是显式操作（``store.build_dataset_manifest`` / CLI），读路径只消费。

非职责
    不做谓词过滤（那是 DuckDB）；不负责时间分区 pattern 展开
    （partition_planner.py）；不读全表数据。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import glob as glob_module
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core.exceptions import DataError


def uuid4_hex() -> str:
    """#P0-30 manifest generation id（完整 uuid4 hex）。"""
    return uuid.uuid4().hex


def _fsync_parent(path: Path) -> None:
    """#P1-23 关键文件替换后 fsync 父目录，保证 power-loss 下 rename 可见。"""
    try:
        fd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass

MANIFEST_FILENAME = "_manifest.parquet"
_MANIFEST_META_FILENAME = "_manifest.json"

_MANIFEST_COLUMNS = [
    "path",
    "rows",
    "bytes",
    "min_time",
    "max_time",
    "min_instrument",
    "max_instrument",
    "schema_hash",
    "mtime_ns",
    "etag",
    "footer_bytes",
]


def _norm_value(value: Any) -> Any:
    """把 parquet 统计值规范化为可比较键（数值保留数值，时间保留全精度 iso）。

    注意：不再把 int/float 字符串化（避免 ``"9" > "10"`` 字典序 bug），
    也不再对字符串 ``[:10]`` 截断（保留 intraday 全精度）。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _time_key(value: Any) -> Any:
    """查询 time_range 端点的规范化键（与 _norm_value 对齐）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return value
    return str(value)


@dataclass(frozen=True)
class ManifestFile:
    """R32-P0-064: 区分 unknown (None) 与 0。

    rows=None 表示未知行数（manifest 构建失败/footer 缺失），与 rows=0（真实空文件）严格区分。
    bytes=None 表示未知大小，与 bytes=0（零字节文件）严格区分。
    成本估算/资源准入对 None 必须用保守上界，绝不当成 0（避免低估）。
    """
    path: str
    rows: int | None = None
    bytes: int | None = None
    min_time: str | None = None
    max_time: str | None = None
    min_instrument: str | None = None
    max_instrument: str | None = None
    schema_hash: str | None = None
    mtime_ns: int | None = None
    etag: str | None = None
    # #P0-29 footer_bytes 单独记（ParquetFileMetadata.serialized_size 是 footer 的
    # 大小，不是整个物理文件）；``bytes`` 一律用 stat().st_size。
    footer_bytes: int | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "bytes": self.bytes,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "min_instrument": self.min_instrument,
            "max_instrument": self.max_instrument,
            "schema_hash": self.schema_hash,
            "mtime_ns": self.mtime_ns,
            "etag": self.etag,
            "footer_bytes": self.footer_bytes,
        }


@dataclass(frozen=True)
class ManifestRowGroup:
    path: str
    row_group: int
    column: str
    min: Any | None = None
    max: Any | None = None
    null_count: int | None = None
    rows: int | None = None


@dataclass
class DatasetManifest:
    """数据集物理元数据清单。"""

    dataset: str
    time_column: str | None = None
    instrument_column: str | None = None
    format: str = "parquet"
    files: tuple[ManifestFile, ...] = ()
    row_groups: tuple[ManifestRowGroup, ...] | None = None
    created_at: str | None = None
    time_dtype: str | None = None       # date / timestamp / int64 / float64 / string（typed 比较用）
    # 双 epoch（#1）：source_epoch 由写路径每次 mutation 递增；manifest_built_epoch
    # 只在 manifest 真正重建/增量更新后追上 source_epoch。仅当
    # ``source_epoch == manifest_built_epoch`` 时允许用本 manifest 做 prune——
    # 否则 ``_manifest.parquet`` 里的 min/max/rows 可能是旧数据，prune 会返回错。
    source_epoch: str | None = None
    manifest_built_epoch: str | None = None
    # 兼容：老 sidecar 只有 manifest_epoch；load 时映射成 source==built（trust）。
    manifest_epoch: str | None = None
    # #P0-30 manifest generation id：parquet + JSON sidecar 都带同一 generation。
    # 两份不一致（进程死在两次 replace 之间）→ 视为 mixed generation → 不 fresh。
    manifest_generation_id: str | None = None
    # R28-8：schema epoch 摘要 ``{schema_hash: {field: dtype}}``——publish/manifest
    # 构建时算一次，query-time ``SchemaEpochGate`` 直接从摘要分组，不再逐文件开
    # parquet footer（O(N footer) → O(schema epochs)，FactorEngine 热路径关键）。
    schema_epochs: dict[str, dict[str, str]] | None = None

    @property
    def is_fresh_epoch(self) -> bool:
        """双 epoch 新鲜判定：source == built 才新鲜（fail-closed on partial state）。

        #P0-FRESHNESS-AUDIT：partial dual-epoch state (任一为 None、另一非 None) 必须
        fail-closed。只有两者都存在时才比较；都缺失时退回老单 epoch 语义（兼容）。
        例：source_epoch="42" 但 manifest_built_epoch=None → 部分写入/损坏 → False。
        """
        src_exists = self.source_epoch is not None
        built_exists = self.manifest_built_epoch is not None
        if src_exists != built_exists:
            # Partial dual-epoch state: 必须 fail-closed（一侧存在另一侧缺失 → 损坏）
            return False
        if src_exists and built_exists:
            # 完整双 epoch：按相等判定
            return self.source_epoch == self.manifest_built_epoch
        # 两者都缺失：老单 epoch 格式（存在即 trust）
        return self.manifest_epoch is not None

    def _typed(self, value: str | None, *, ints: bool = False) -> Any:
        """把 manifest 里存的 min/max 按 time_dtype 还原成可比较类型。"""
        if value is None:
            return None
        td = (self.time_dtype or "").lower()
        if "int" in td or ints:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if "float" in td:
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        return value

    @property
    def total_rows(self) -> int:
        return sum(f.rows or 0 for f in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def schema_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({f.schema_hash for f in self.files if f.schema_hash}))

    @property
    def dataset_version(self) -> str:
        """数据集内容版本：schema + 列定义 + 文件集合（结构变 → 版本变）。"""
        payload = {
            "format": self.format,
            "time_column": self.time_column,
            "instrument_column": self.instrument_column,
            "schema_hashes": list(self.schema_hashes),
            "file_count": self.file_count,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def partition_version(self) -> str:
        """物理分区版本：文件数 + 字节 + 最大 mtime（新文件/替换 → 版本变）。"""
        max_mtime = max((f.mtime_ns or 0) for f in self.files) if self.files else 0
        payload = {
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "max_mtime_ns": max_mtime,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_table(self) -> pa.Table:
        rows = [f.to_row() for f in self.files]
        arrays: dict[str, list[Any]] = {c: [] for c in _MANIFEST_COLUMNS}
        for row in rows:
            for col in _MANIFEST_COLUMNS:
                arrays[col].append(row.get(col))
        return pa.table(arrays)

    def save(self, root: Path) -> Path:
        """写入 ``{root}/_manifest.parquet`` + ``_manifest.json``（meta，原子替换）。

        #P0-30 两份文件都写同一 ``manifest_generation_id``（parquet 走 schema
        metadata，JSON 走字段）。进程死在两次 replace 之间 → generation 不一致 →
        load 判 mixed → 不 fresh（fail-closed，planner 回退 glob 全文件列表）。
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        table = self.to_table()
        # #P0-final closure 1：每次 logical manifest commit 都必须 mint **全新**
        # generation id——即使本对象是从旧 manifest load 后重建，也绝不复用旧
        # generation。否则进程死在 ``_manifest.parquet`` replace 之后、
        # ``_manifest.json`` replace 之前，若新旧两代恰巧共用同一 generation，
        # load() 的双 generation 一致性检查仍会通过，把「新 parquet + 旧 JSON」
        # 误认为同一代（generation 机制就是为了防这个 crash 窗口）。
        gen = uuid4_hex()
        self.manifest_generation_id = gen  # 内存对象与落盘一致（后续 bump 保留）
        schema = table.schema.with_metadata(
            {b"manifest_generation_id": gen.encode("utf-8")}
        )
        out = root / MANIFEST_FILENAME
        # #P1-final closure 19：统一 atomic durable-write（tmp→fsync(fd)→replace→fsync(dir)）
        from data_access.core.atomic import atomic_write_file

        atomic_write_file(out, lambda tmp: pq.write_table(table.cast(schema), tmp))
        if self.row_groups:
            _save_row_groups(root, self.row_groups, generation=gen)
        else:
            # #P0-13 重建不生成 row-group sidecar 时删掉旧的——否则新 manifest
            # （generation B，无 rowgroups）会 load 到旧 sidecar（generation A）。
            _remove_stale_row_group_sidecar(root)
        old_epoch = _read_epoch(root)
        src = self.source_epoch or self.manifest_epoch or _next_epoch(old_epoch)
        built = self.manifest_built_epoch or src
        meta_path = root / _MANIFEST_META_FILENAME
        # #P1-final closure 19：统一 atomic durable-write
        from data_access.core.atomic import atomic_write_json

        atomic_write_json(
            meta_path,
            {
                "dataset": self.dataset,
                "time_column": self.time_column,
                "instrument_column": self.instrument_column,
                "format": self.format,
                "time_dtype": self.time_dtype,
                # 双 epoch：source 是数据版本，built 是 manifest 同步到的版本。
                "source_epoch": src,
                "manifest_built_epoch": built,
                "manifest_epoch": src,  # 兼容老读取方（等价 source_epoch）
                "manifest_generation_id": gen,
                "created_at": self.created_at,
                "file_count": self.file_count,
                "dataset_version": self.dataset_version,
                "partition_version": self.partition_version,
                # R28-8：schema epoch 摘要（query-time O(1) 分组，避免逐文件 footer）。
                "schema_epochs": self.schema_epochs or {},
            },
        )
        return out

    @classmethod
    def load(cls, root: Path) -> "DatasetManifest | None":
        path = Path(root) / MANIFEST_FILENAME
        if not path.exists():
            return None
        try:
            table = pq.read_table(str(path))
        except Exception:
            return None
        data = table.to_pydict()
        files: list[ManifestFile] = []
        paths = data.get("path", [])
        for i in range(len(paths)):
            files.append(
                ManifestFile(
                    path=str(paths[i]),
                    rows=_int_or_none(_at(data, "rows", i)),
                    bytes=_int_or_none(_at(data, "bytes", i)),
                    min_time=_str_or_none(_at(data, "min_time", i)),
                    max_time=_str_or_none(_at(data, "max_time", i)),
                    min_instrument=_str_or_none(_at(data, "min_instrument", i)),
                    max_instrument=_str_or_none(_at(data, "max_instrument", i)),
                    schema_hash=_str_or_none(_at(data, "schema_hash", i)),
                    mtime_ns=_int_or_none(_at(data, "mtime_ns", i)),
                    etag=_str_or_none(_at(data, "etag", i)),
                    footer_bytes=_int_or_none(_at(data, "footer_bytes", i)),
                )
            )
        meta = _read_manifest_meta_json(root)
        # #P0-30 parquet 与 JSON sidecar 的 generation 必须一致——进程死在两次
        # replace 之间会留下 mixed generation，绝不能当权威 manifest 用。
        # #P0-12 新格式任一侧缺 generation 也算 mixed（partial write）→ 不 fresh。
        parquet_gen = _parquet_generation_id(path)
        sidecar_gen = meta.get("manifest_generation_id")
        if parquet_gen is not None or sidecar_gen is not None:
            if parquet_gen is None or sidecar_gen is None or parquet_gen != sidecar_gen:
                return None  # 单侧缺失 / 不一致 → mixed generation
            if not _valid_generation_id(parquet_gen):
                return None  # malformed identity → fail closed
        generation = parquet_gen or sidecar_gen
        src = meta.get("source_epoch")
        built = meta.get("manifest_built_epoch")
        legacy = meta.get("manifest_epoch")
        if src is None and built is None and legacy is not None:
            # 老格式：manifest 建好后未 mutation 即 trust（source==built）。
            src = built = legacy
        return cls(
            dataset=meta.get("dataset", ""),
            time_column=meta.get("time_column"),
            instrument_column=meta.get("instrument_column"),
            format=meta.get("format", "parquet"),
            files=tuple(files),
            row_groups=_load_row_groups(root, expected_generation=generation),
            created_at=meta.get("created_at"),
            time_dtype=meta.get("time_dtype"),
            source_epoch=src,
            manifest_built_epoch=built,
            manifest_epoch=legacy,
            manifest_generation_id=generation,
            schema_epochs=meta.get("schema_epochs") or None,
        )

    def prune_by_time(
        self, time_range: tuple[Any, Any] | None
    ) -> list[str]:
        """返回与 time_range 有交集的文件路径（按 min/max_time 裁剪）。

        min/max 比较是类型化的（#19）：int64 时间列按数值比较，避免
        ``"9" > "10"`` 字典序 bug；date/timestamp 按 isoformat 字符串比较。
        """
        if time_range is None or self.time_column is None:
            return [f.path for f in self.files]
        start_key = _time_key(time_range[0])
        end_key = _time_key(time_range[1])
        ints = "int" in (self.time_dtype or "").lower()
        if isinstance(start_key, (int, float)):
            ints = True
        if isinstance(end_key, (int, float)):
            ints = True
        out: list[str] = []
        for f in self.files:
            if f.min_time is None or f.max_time is None:
                out.append(f.path)
                continue
            lo = self._typed(f.min_time, ints=ints)
            hi = self._typed(f.max_time, ints=ints)
            if start_key is not None and hi is not None:
                try:
                    if hi < start_key:
                        continue
                except TypeError:
                    pass
            if end_key is not None and lo is not None:
                try:
                    if lo > end_key:
                        continue
                except TypeError:
                    pass
            out.append(f.path)
        return out

    def prune_by_instruments(
        self, instrument_filter: Sequence[str] | None
    ) -> list[str]:
        """按 instrument 的字典序 min/max 裁剪（仅当字符串列统计可用）。"""
        if not instrument_filter:
            return [f.path for f in self.files]
        if self.instrument_column is None:
            return [f.path for f in self.files]
        lo = min(instrument_filter)
        hi = max(instrument_filter)
        out: list[str] = []
        for f in self.files:
            if f.min_instrument is None or f.max_instrument is None:
                out.append(f.path)
                continue
            if f.max_instrument < lo or f.min_instrument > hi:
                continue
            out.append(f.path)
        return out

    def prune(
        self,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """time + instrument 联合裁剪。"""
        paths = set(self.prune_by_time(time_range))
        if instrument_filter:
            paths &= set(self.prune_by_instruments(instrument_filter))
        return sorted(paths)

    def epoch_summary_for_paths(
        self, paths: Sequence[str]
    ) -> tuple[dict[str, str], dict[str, dict[str, str]]] | None:
        """R28-8：从 manifest schema epoch 摘要构建 (path→schema_hash, hash→fields)。

        返回 None 表示 manifest 没有 schema_epochs 摘要（老 manifest / 未构建）——
        调用方回退逐文件 footer。path 未在 manifest 文件清单中的也会回退 footer。
        这是 query-time O(files) 的纯内存映射（无 parquet footer I/O）。
        """
        if not self.schema_epochs or not self.files:
            return None
        hash_by_path: dict[str, str] = {}
        for f in self.files:
            if f.schema_hash and f.path in hash_by_path:
                # 同一 path 不会重复出现；去重防御。
                continue
            if f.schema_hash:
                hash_by_path[f.path] = f.schema_hash
        return hash_by_path, dict(self.schema_epochs)


def _at(data: Mapping[str, list[Any]], col: str, i: int) -> Any:
    arr = data.get(col)
    if arr is None or i >= len(arr):
        return None
    return arr[i]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _valid_generation_id(value: Any) -> bool:
    """接受完整 UUID；16 hex 仅用于读取本格式既有 legacy manifest。"""
    return isinstance(value, str) and bool(
        re.fullmatch(r"(?:[0-9a-f]{16}|[0-9a-f]{32})", value)
    )


def _parquet_generation_id(path: Path) -> str | None:
    """#P0-30 读 ``_manifest.parquet`` schema metadata 里的 generation id。"""
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


def _read_manifest_meta_json(root: Path) -> dict[str, str]:
    """读 ``_manifest.json`` sidecar（dataset/time_column/format/created_at）。"""
    path = Path(root) / _MANIFEST_META_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "dataset": str(payload.get("dataset", "")),
        "time_column": payload.get("time_column"),
        "instrument_column": payload.get("instrument_column"),
        "format": str(payload.get("format", "parquet")),
        "created_at": payload.get("created_at"),
        "time_dtype": payload.get("time_dtype"),
        "source_epoch": payload.get("source_epoch"),
        "manifest_built_epoch": payload.get("manifest_built_epoch"),
        "manifest_epoch": payload.get("manifest_epoch"),
        # #P0-12 必须把 generation 完整带回来——之前读回时丢字段，load() 的
        # parquet vs JSON mismatch 检测无法可靠生效。
        "manifest_generation_id": payload.get("manifest_generation_id"),
        # R28-8：schema epoch 摘要随 sidecar 带回（query-time O(1) 分组）。
        "schema_epochs": payload.get("schema_epochs"),
    }


def manifest_root_for_paths(paths: Sequence[str]) -> Path | None:
    """从解析出的 glob 路径推断 manifest 根目录。

    #P0-31 三种形态分开处理：
        - 精确单文件路径（无 glob）→ 返回**父目录**（manifest 不可能以数据文件
          自身为根）；
        - glob 路径 → 返回**通配符前最长静态目录前缀**（glob 的包含目录）——
          文件 ``d/part-*.parquet`` / ``d/year=2024/part-*.parquet`` 都活在
          ``d``（或 ``d/year=2024``）下，manifest 根是那个目录，而不是把
          ``d/part-`` 这种文件名前缀当目录（会在不存在的地方建 phantom 目录）；
        - 目录 → 返回自身。
    """
    for p in paths:
        if not p:
            continue
        text = str(p)
        if "*" in text or "?" in text or "[" in text:
            import re as _re

            m = _re.search(r"[*?[]", text)
            static = text[: m.start()] if m else text
            # 静态前缀可能停在文件名中间（如 part-*.parquet / year=*.parquet），
            # 取到最后一个 '/' 之前才是真正目录；无 '/' 的相对 glob → 当前目录。
            idx = static.rfind("/")
            if idx >= 0:
                static = static[:idx]
            elif static:
                static = "."
            if static:
                return Path(static)
            continue
        path = Path(text)
        if path.suffix.lower() in {".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".feather"}:
            return path.parent if path.name else path
        return path
    return None


def _count_data_files(glob_paths: Sequence[str]) -> int | None:
    """文件名级 glob 计数（只算数据文件，排除 manifest 自身）。

    返回 None 表示无法本地计数（远程 glob）→ 调用方应信任 manifest。
    """
    count = 0
    for pattern in glob_paths:
        if str(pattern).startswith("s3://"):
            # 远程无法本地计数 → 信任 manifest（COS 对象按日不变）
            return None
        matches = []
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matches = glob_module.glob(pattern, recursive=True)
        else:
            p = Path(pattern)
            if p.is_dir():
                matches = [str(x) for x in p.rglob("*.parquet")]
            elif p.suffix in {".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".feather"} and p.exists():
                matches = [str(p)]
        for m in matches:
            # manifest 自身（_manifest.parquet / _manifest.json）不算数据文件
            name = str(m).split("/")[-1]
            if name == MANIFEST_FILENAME or name == _MANIFEST_META_FILENAME:
                continue
            count += 1
    return count


def is_manifest_fresh(manifest: DatasetManifest, glob_paths: Sequence[str]) -> bool:
    """新鲜度检查（默认 O(1)）：双 epoch 判定 ``source_epoch == manifest_built_epoch``。

    mutation 后 ``source_epoch`` 递增但 ``manifest_built_epoch`` 不动 → 立刻 dirty，
    读路径不会再拿旧 ``_manifest.parquet`` 做 prune（避免 min/max 过期的错误裁剪）。

    #P0-FRESHNESS-AUDIT：partial dual-epoch state (任一为 None、另一非 None) 必须
    fail-closed → False。只有完全无 epoch 的极老 ``_manifest.json`` 才退回文件名
    glob 比对。单 epoch 老格式（无 split）视为 source==built（建好后未 mutation）。
    """
    root = manifest_root_for_paths(list(glob_paths) if glob_paths else [])
    if root is not None:
        meta = manifest_version_token(root)
        if meta is not None:
            src = meta.get("source_epoch")
            built = meta.get("manifest_built_epoch")
            src_exists = src is not None
            built_exists = built is not None
            # Partial dual-epoch state 必须 fail-closed
            if src_exists != built_exists:
                return False
            if src_exists and built_exists:
                return src == built
            # 完全无双 epoch：退回单 epoch 老格式（存在即 trust）
            if meta.get("manifest_epoch") is not None:
                return True
    # 完全无 epoch metadata：退回文件名 glob 比对（极老格式）
    count = _count_data_files(glob_paths)
    if count is None:
        return True
    return count == manifest.file_count


def manifest_version_token(root: Path) -> dict[str, Any] | None:
    """廉价版本 token：只读 ``_manifest.json`` sidecar，不读 manifest parquet。

    供 query-scoped snapshot 的 ``store.manifest_version()`` 使用——几十微秒级，
    避免每次 describe 整个 dataset 再执行一次实际 read。
    """
    meta_path = Path(root) / _MANIFEST_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


_ROW_GROUPS_FILENAME = "_manifest_rowgroups.parquet"


def _next_epoch(existing: str | None) -> str:
    try:
        return str(int(existing or 0) + 1)
    except ValueError:
        return "1"


def _read_epoch(root: Path) -> str | None:
    meta = manifest_version_token(root)
    if meta is None:
        return None
    return meta.get("source_epoch") or meta.get("manifest_epoch")


def bump_source_epoch(root: Path) -> str | None:
    """写路径 mutation 后调用：只递增 ``_manifest.json`` 的 ``source_epoch``。

    ``manifest_built_epoch`` 保持不动 → manifest 自动 dirty，读路径不会再用旧
    ``_manifest.parquet`` 的 min/max 做 prune。O(1)，不重建 manifest、不 glob。
    返回新 source_epoch；无 sidecar 返回 None（该数据集没有 manifest 可失效）。
    """
    meta_path = Path(root) / _MANIFEST_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    src = _next_epoch(
        payload.get("source_epoch") or payload.get("manifest_epoch")
    )
    payload["source_epoch"] = src
    payload["manifest_epoch"] = src  # 兼容读取方（等价 source_epoch）
    # manifest_built_epoch 保持原值 → source != built，manifest 变 dirty。
    # #P0-30 保留 generation，避免 bump 后与 parquet 侧不一致被判 mixed。
    payload.setdefault("manifest_generation_id", _parquet_generation_id(meta_path.parent / MANIFEST_FILENAME))
    # #P1-final closure 19：统一 atomic durable-write
    from data_access.core.atomic import atomic_write_json

    atomic_write_json(meta_path, payload)
    return src


def bump_manifest_epoch(root: Path) -> str | None:
    """兼容别名：递增 source_epoch（老调用方）。"""
    return bump_source_epoch(root)


def rebuild_manifest_for_dataset(
    store: Any, dataset: str, **params: Any
) -> DatasetManifest | None:
    """mutation commit 后重建 manifest，使 ``manifest_built_epoch == source_epoch``。

    - 数据集没有 manifest sidecar → 不建（数据集主人才决定是否启用 manifest）。
    - **重建失败 → 向上抛**（本轮 closure 改动）：write caller 必须知道
      manifest 没有重建成，不能再静默吞掉让调用方以为提交完整。读路径的安全
      性不受影响——manifest 保持 dirty 时读路径照旧回退 glob + 全文件列表。
    - 成功后 ``manifest_built_epoch`` 追上 ``source_epoch``，prune 恢复。
    """
    from data_access.read.manifest import (
        build_manifest_for_dataset as _build,
    )
    from data_access.read.manifest import manifest_root_for_paths, manifest_version_token

    ds = store._registry.get(dataset)
    try:
        raw_paths = store._resolve_raw_paths(ds, time_range=None, params=dict(params))
    except Exception:
        return None
    root = manifest_root_for_paths(raw_paths)
    if root is None:
        return None
    token = manifest_version_token(root)
    if token is None or not (
        token.get("source_epoch") is not None or token.get("manifest_epoch") is not None
    ):
        return None  # 没有启用 manifest
    return _build(store, dataset, params=params or None, include_row_groups=False)


def _remove_stale_row_group_sidecar(root: Path) -> None:
    """#P0-13 删除过期 row-group sidecar（重建不生成 rowgroups 时必须调用）。"""
    path = Path(root) / _ROW_GROUPS_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _row_groups_generation(path: Path) -> str | None:
    """读 row-group sidecar parquet schema metadata 里的 manifest generation。"""
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


def _save_row_groups(
    root: Path, row_groups: Sequence[ManifestRowGroup], *, generation: str | None = None
) -> Path:
    """把 row-group 级统计持久化到 ``{root}/_manifest_rowgroups.parquet``。

    #P0-13 把 manifest generation 写入 sidecar 的 schema metadata——旧 sidecar
    不能被新 generation 的 manifest load（文件级 metadata 与 rowgroup 统计必须同代）。
    """
    root = Path(root)
    # #P0-14 ``pa.table(list_of_dicts)`` 在 pyarrow 25 不支持（"Must pass names or
    # schema"），旧代码 rowgroup sidecar 实际从未成功持久化过。改成列数组字典。
    table = pa.table(
        {
            "path": [rg.path for rg in row_groups],
            "row_group": [rg.row_group for rg in row_groups],
            "column": [rg.column for rg in row_groups],
            "min": [_norm_value(rg.min) for rg in row_groups],
            "max": [_norm_value(rg.max) for rg in row_groups],
            "null_count": [rg.null_count for rg in row_groups],
            "rows": [rg.rows for rg in row_groups],
        }
    )
    if generation:
        table = table.cast(
            table.schema.with_metadata(
                {b"manifest_generation_id": generation.encode("utf-8")}
            )
        )
    out = root / _ROW_GROUPS_FILENAME
    # #P1-final closure 19：统一 atomic durable-write（tmp→fsync(fd)→replace→fsync(dir)）
    from data_access.core.atomic import atomic_write_file

    atomic_write_file(out, lambda tmp: pq.write_table(table, tmp))
    return out


def _load_row_groups(
    root: Path, *, expected_generation: str | None = None
) -> tuple[ManifestRowGroup, ...] | None:
    """读取持久化的 row-group 级统计；无则返回 None。

    #P0-13 只接受与 manifest 同 generation 的 sidecar；generation 不一致或
    单侧缺失（旧 sidecar 配新 manifest）→ 忽略（返回 None，读路径回退文件级）。
    """
    path = Path(root) / _ROW_GROUPS_FILENAME
    if not path.exists():
        return None
    try:
        table = pq.read_table(str(path))
    except Exception:
        return None
    sidecar_gen = _row_groups_generation(path)
    if expected_generation is not None:
        if sidecar_gen is None or sidecar_gen != expected_generation:
            return None  # 旧 sidecar / 异代 sidecar 不能配新 manifest
    elif sidecar_gen is not None:
        return None  # 新 sidecar 配无 generation 的旧 manifest → 也拒（fail-closed）
    data = table.to_pydict()
    data = table.to_pydict()
    out: list[ManifestRowGroup] = []
    paths = data.get("path", [])
    for i in range(len(paths)):
        out.append(
            ManifestRowGroup(
                path=str(paths[i]),
                row_group=int(_at(data, "row_group", i) or 0),
                column=str(_at(data, "column", i) or ""),
                min=_at(data, "min", i),
                max=_at(data, "max", i),
                null_count=_int_or_none(_at(data, "null_count", i)),
                rows=_int_or_none(_at(data, "rows", i)),
            )
        )
    return tuple(out) if out else None


def build_manifest_for_dataset(
    store: Any,
    dataset: str,
    *,
    params: Mapping[str, Any] | None = None,
    include_row_groups: bool = False,
    force: bool = False,
) -> DatasetManifest | None:
    """为数据集构建 manifest 并落盘。返回 None 表示没有可枚举文件。"""
    import glob as _glob

    from data_access.read.stats import expand_parquet_paths

    ds = store._registry.get(dataset)
    paths = store._resolve_raw_paths(ds, time_range=None, params=dict(params or {}))
    root = manifest_root_for_paths(paths)
    if root is None:
        return None

    # 只处理本层能读 footer 的格式（parquet 原生；csv 等无 footer → 略过）
    if getattr(ds, "format", "parquet") not in {"parquet", "pq"}:
        return None

    files = expand_parquet_paths(paths)
    # 排除 manifest 自身与 row-group sidecar——它们匹配数据 glob（*.parquet）但
    # 不是数据文件；重建时若不排除会把 _manifest.parquet 自己写进自己的文件清单。
    files = [
        fp
        for fp in files
        if fp.name not in {
            MANIFEST_FILENAME,
            _MANIFEST_META_FILENAME,
            _ROW_GROUPS_FILENAME,
        }
    ]
    if not files:
        return None

    time_col = ds.time_column
    inst_col = ds.instrument_column
    file_rows: list[ManifestFile] = []
    row_groups: list[ManifestRowGroup] = []
    schema_epochs: dict[str, dict[str, str]] = {}
    for fp in files:
        try:
            meta = pq.read_metadata(str(fp))
        except Exception as exc:
            # #P0-28 构建权威 manifest 时任意 parquet footer 读失败 → 整个构建失败。
            # 静默跳过 + 保存成 fresh manifest 会让 planner 误以为坏文件不存在。
            raise DataError(
                f"构建 {dataset} 的 manifest 时读取 {fp} 的 parquet footer 失败: {exc}；"
                "坏文件必须 fail-closed，禁止静默跳过并保存成 fresh manifest"
            ) from exc
        rows = meta.num_rows
        # #P0-29 bytes 用真实物理文件大小（serialized_size 只是 footer 大小，会让
        # scan cost / CBO / QueryBudget 系统性低估）；footer 大小单独记 footer_bytes。
        fbytes = _safe_stat(fp)
        footer_bytes = meta.serialized_size
        schema_names = list(meta.schema.names)
        schema_hash = _schema_hash(meta.schema)
        # R28-8：记录 schema epoch 摘要（{field: logical dtype}）。同一 fingerprint
        # 的物理文件 logical schema 相同——从 schema_arrow 取（与 schema_epoch
        # 的 _normalize_dtype 归一化基准一致）。
        if schema_hash not in schema_epochs:
            arrow_schema = getattr(meta.schema, "schema_arrow", None)
            if arrow_schema is not None:
                schema_epochs[schema_hash] = {
                    str(arrow_schema.names[i]): str(arrow_schema.field(i).type)
                    for i in range(len(arrow_schema.names))
                }
            else:
                schema_epochs[schema_hash] = {
                    str(meta.schema.column(i).name): str(meta.schema.column(i).physical_type)
                    for i in range(len(meta.schema))
                }
        min_t = max_t = min_i = max_i = None
        t_idx = schema_names.index(time_col) if time_col in schema_names else None
        i_idx = schema_names.index(inst_col) if inst_col in schema_names else None
        for rg in range(meta.num_row_groups):
            rg_meta = meta.row_group(rg)
            if include_row_groups and t_idx is not None:
                stats = rg_meta.column(t_idx).statistics
                if stats is not None:
                    row_groups.append(
                        ManifestRowGroup(
                            path=str(fp),
                            row_group=rg,
                            column=str(time_col),
                            min=_norm_value(stats.min),
                            max=_norm_value(stats.max),
                            null_count=stats.null_count,
                            rows=rg_meta.num_rows,
                        )
                    )
            if t_idx is not None:
                stats = rg_meta.column(t_idx).statistics
                if stats is not None:
                    lo, hi = _norm_value(stats.min), _norm_value(stats.max)
                    min_t = lo if (min_t is None or (lo is not None and lo < min_t)) else min_t
                    max_t = hi if (max_t is None or (hi is not None and hi > max_t)) else max_t
            if i_idx is not None:
                stats = rg_meta.column(i_idx).statistics
                if stats is not None:
                    lo, hi = _norm_value(stats.min), _norm_value(stats.max)
                    min_i = lo if (min_i is None or (lo is not None and lo < min_i)) else min_i
                    max_i = hi if (max_i is None or (hi is not None and hi > max_i)) else max_i
        try:
            st = fp.stat()
            mtime_ns = st.st_mtime_ns
        except OSError:
            mtime_ns = None
        file_rows.append(
            ManifestFile(
                path=str(fp),
                rows=rows,
                bytes=fbytes,
                min_time=min_t,
                max_time=max_t,
                min_instrument=min_i,
                max_instrument=max_i,
                schema_hash=schema_hash,
                mtime_ns=mtime_ns,
                footer_bytes=footer_bytes,
            )
        )

    if not file_rows:
        return None
    time_dtype = None
    if time_col is not None:
        td = str((ds.schema or {}).get(time_col, "")).lower()
        if td:
            time_dtype = "timestamp" if ("timestamp" in td or "datetime" in td) else td
    manifest = DatasetManifest(
        dataset=dataset,
        time_column=time_col,
        instrument_column=inst_col,
        format=str(getattr(ds, "format", "parquet")),
        files=tuple(file_rows),
        row_groups=tuple(row_groups) if row_groups else None,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time_dtype=time_dtype,
        schema_epochs=schema_epochs or None,
    )
    manifest.save(root)
    return manifest


def _safe_stat(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _schema_hash(schema: Any) -> str:
    # schema_arrow 是 pa.Schema；没有时退回 parquet 物理类型
    arrow_schema = getattr(schema, "schema_arrow", None)
    if arrow_schema is not None:
        names = list(arrow_schema.names)
        types = [str(arrow_schema.field(i).type) for i in range(len(names))]
    else:
        names = list(schema.names)
        types = [str(schema.column(i).physical_type) for i in range(len(names))]
    text = json.dumps(list(zip(names, types)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest_for_dataset(store: Any, dataset: str, **params: Any) -> DatasetManifest | None:
    """按数据集加载 manifest（存在且新鲜时返回）。"""
    ds = store._registry.get(dataset)
    paths = store._resolve_raw_paths(ds, time_range=None, params=params)
    root = manifest_root_for_paths(paths)
    if root is None:
        return None
    manifest = DatasetManifest.load(root)
    if manifest is None:
        return None
    if manifest.dataset and manifest.dataset != dataset:
        return None
    if not is_manifest_fresh(manifest, paths):
        return None
    return manifest
