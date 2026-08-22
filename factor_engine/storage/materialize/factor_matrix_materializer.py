"""多因子宽矩阵物化：训练/回测零 pivot 读取路径。

输出布局::

    {matrix_root}/
      universe={universe}/
        freq={freq}/
          manifest.json      # P0-23 factor_version 绑定 + P1-15 CAS + generation 指针
          .matrix.lock       # 整个 (universe,freq) publish 的 flock
          generation/<gid>/  # 一次性不可变 generation（全部分区 validate 后才可见）
            year=2025/
              month=01/
                data.parquet   # datetime, asset, factor_a, factor_b, ...

治理闭环（#收官轮 + R13 + R14）：
  * P0-21 read-merge-write 区分「本次没有该 key」与「本次显式 NaN/tombstone」；
  * P0-22 旧分区读失败 → 移动 ``.quarantine/`` 并 hard fail，绝不按空分区覆盖；
  * P0-23 factor_version 绑定：semantic_digest 变化拒绝混列；
  * P1-15 manifest CAS：并发写同 universe/freq 用版本号检测；
  * P1-16 load 后重复 ``(datetime, asset)`` key 抛错，不再静默 keep="last"；
  * R14 #1 generation-atomic publish：整个 ``(universe, frequency)`` 当一个
    generation——先完整写入 ``generation/<gid>/`` 并全部 validate，再把
    ``manifest.json``（含 ``generation`` 指针）用 ``os.replace`` 原子切一次。
    reader 只读 ``manifest.generation`` 指向的那一代，任何崩溃点都只能看到
    完整 old generation 或完整 new generation，**绝无半新半旧**（旧实现的
    「先写 manifest、再逐分区 os.replace」会在切换一半时崩溃产生 mixed 状态）。
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import threading as _threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from logging_utils import get_logger
from storage.matrix_block_layout import (
    block_columns,
    block_file_name,
    block_mode_enabled,
    build_matrix_block,
    checksum_proof_enabled,
    compare_checksums,
    compute_matrix_checksums,
    matrix_join_count,
    matrix_rewrite_amplification,
    overlay_block,
    parse_block_id,
    partition_overlaps_time_range,
    read_parquet_checksums,
    record_rewrite_amplification,
    resolve_factor_blocks,
    wide_to_merged,
)
from storage.partition_object_ref import (
    PartitionObjectRef,
    build_partition_inventory,
    inventory_from_dicts,
    inventory_parquet_paths,
    inventory_to_dicts,
    materialize_generation_from_inventory,
    ref_from_frame,
)
from storage.partition_policy import (
    PartitionPolicy,
    attach_partition_columns,
    iter_partition_groups,
    partition_path_segments,
)

logger = get_logger("storage.factor_matrix_materializer")


class FactorMatrixCorruptionError(ValueError):
    """factor_matrix 分区已存在但无法读取/校验（损坏 / IO / schema 异常）。"""


class FactorMatrixReadError(FactorMatrixCorruptionError):
    """P0-22: 分区 parquet 读失败 → 已隔离到 quarantine 且 hard fail。

    旧实现把读失败当作空分区继续覆盖，会把一整月历史在增量运行时无声丢掉。
    读失败唯一处理路径 = 把原文件移动进 ``<part_dir>/.quarantine/`` 并抛本异常，
    绝不继续 publish。
    """


class FactorMatrixVersionMismatchError(ValueError):
    """P0-23: 同一 factor 的 semantic_digest 变化 → 禁止新旧月份混列。"""


class FactorMatrixConcurrentWriteError(ValueError):
    """P1-15: manifest CAS 失败 → 并发写同一 universe/freq（flock 不是分布式事务）。"""


class DuplicateMatrixKeyError(ValueError):
    """P1-16: 加载后 ``(datetime, asset)`` 重复 → partition 契约已损坏。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def _partition_write_lock(lock_path: Path):
    """分区级写互斥（flock）：并发 matrix 写同一分区时 serial 化 read-merge-write。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


_held_manifest_locks: set[Path] = set()
_held_manifest_locks_guard = _threading.Lock()


@contextlib.contextmanager
def _manifest_write_lock(base: Path):
    """``manifest.json`` 级写互斥：P1-15 CAS 的串行化点。

    R14 #1：整个 (universe,freq) publish（generation 重建 + manifest 切换）在
    base 级 flock 内串行化；``_update_manifest`` 在内部也会再进同一把锁——同一
    进程内**可重入**（flock 对同一文件的第二把 fd 锁会 self-deadlock，因此用
    进程内持有集合跳过重复获取）。
    """
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / ".matrix.lock"
    with _held_manifest_locks_guard:
        reentrant = lock_path in _held_manifest_locks
        if not reentrant:
            _held_manifest_locks.add(lock_path)
    if reentrant:
        yield
        return
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
            with _held_manifest_locks_guard:
                _held_manifest_locks.discard(lock_path)


def _read_manifest(base: Path) -> dict[str, Any] | None:
    mpath = base / "manifest.json"
    if not mpath.exists():
        return None
    with open(mpath, encoding="utf-8") as f:
        return json.load(f)


def _write_manifest_atomic(base: Path, manifest: dict[str, Any]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    mpath = base / "manifest.json"
    tmp = base / f".manifest.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(mpath))


def _link_or_copy(src: Path, dst: Path) -> None:
    """把旧 generation 的不可变分区文件链接/复制进新 generation（copy-on-write）。

    generation 内的 parquet 一旦写入不再原地修改，因此 ``os.link`` 共享 inode
    是安全的；不支持硬链接的文件系统回退 ``shutil.copy2``。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def _merge_matrix_frames(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    *,
    value_dtype: str,
) -> pd.DataFrame:
    """外连接合并已有分区宽表与本次增量（R11 #7 + R13 P0-21）。

    之前直接 ``os.replace`` 覆盖整月 ``data.parquet``——增量写子集（部分因子/
    部分日期）会丢掉原分区里其它因子与其它日期的值。改为 read-merge-write：
    旧因子列与旧日期保留，同 ``(datetime, asset)`` key 上新值覆盖旧值。

    merge 语义必须区分「本次增量没有这个 key」与「本次增量显式写了 NaN /
    tombstone」两个状态：

    * key 在 new 中（merge indicator ∈ {``both``, ``right_only``}）→ 新值覆盖
      旧值，**含显式 NaN/tombstone**（读取端映射为 invalid/deleted）；
    * key 只在 old 中（indicator == ``left_only``）→ 保留旧值。

    不能简单 ``combine_first()``：那会把「新 row 显式 NaN」误填回旧值。因此用
    merge ``indicator`` 精确追踪 key 来源。合并前规范化 key dtype 并去重输入，
    避免 parquet round-trip 的 object/string 差异或重复 key 造成 merge 失真。
    """
    if existing is None or existing.empty:
        return new
    if new is None or new.empty:
        return existing
    key_cols = [
        c
        for c in ("datetime", "asset")
        if c in existing.columns and c in new.columns
    ]
    if not key_cols:
        return new
    existing = existing.copy()
    new = new.copy()
    if "datetime" in key_cols:
        existing["datetime"] = pd.to_datetime(existing["datetime"])
        new["datetime"] = pd.to_datetime(new["datetime"])
    if "asset" in key_cols:
        existing["asset"] = existing["asset"].astype("string")
        new["asset"] = new["asset"].astype("string")
    # 输入 key 去重：现有/增量分区内部契约应为唯一 key，重复时 merge 会产生
    # 笛卡尔积行，先消除再从 merge 结果兜底去重。
    existing = existing.drop_duplicates(subset=key_cols, keep="first")
    new = new.drop_duplicates(subset=key_cols, keep="first")
    # ``indicator`` 记录每个 key 的来源列 ``_fe_src``：
    #   left_only  = 仅旧分区有（本次没重算 → 保留旧值）
    #   both       = 新旧都有（新值覆盖，含显式 NaN/tombstone）
    #   right_only = 仅本次增量有（直接取新值）
    merged = existing.merge(
        new, on=key_cols, how="outer", suffixes=("_old", ""), indicator="_fe_src"
    )
    new_cols = set(new.columns) - set(key_cols)
    src = merged["_fe_src"]
    drop: list[str] = ["_fe_src"]
    for c in sorted(new_cols):
        old_c = f"{c}_old"
        if old_c not in merged.columns:
            continue
        only_old = src == "left_only"
        if only_old.any():
            merged.loc[only_old, c] = merged.loc[only_old, old_c]
        drop.append(old_c)
    if drop:
        merged = merged.drop(columns=drop)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    for c in sorted(new_cols):
        if c in merged.columns:
            merged[c] = merged[c].astype(str(value_dtype or "float32"))
    cols = [c for c in key_cols if c in merged.columns] + sorted(
        c for c in merged.columns if c not in key_cols
    )
    return merged[cols].sort_values(key_cols).reset_index(drop=True)


@dataclass(frozen=True)
class FactorMatrixLayout:
    """factor_matrix 宽表布局路径规则。

    参数:
        无
    """
    universe: str
    frequency: str = "1d"
    partition_columns: tuple[str, ...] = ("year", "month")

    def base_dir(self, matrix_root: Path) -> Path:
        """base_dir。

        参数:
            matrix_root: factor_matrix 根目录

        返回:
            Path
        """
        return (
            matrix_root
            / f"universe={self.universe}"
            / f"freq={self.frequency}"
        )


class FactorMatrixMaterializer:
    """多因子矩阵宽表物化器。

    参数:
        matrix_root: factor_matrix 根目录（可选）
    """

    def __init__(self, matrix_root: str | Path | None = None) -> None:
        """初始化实例。

        参数:
            matrix_root: factor_matrix 根目录（可选）

        返回:
            无
        """
        if matrix_root is None:
            env_root = os.environ.get("FACTOR_MATRIX_ROOT")
            matrix_root = env_root or "factor_matrix"
        self._matrix_root = Path(matrix_root)
        self._matrix_root.mkdir(parents=True, exist_ok=True)

    @property
    def matrix_root(self) -> Path:
        """matrix_root。

        参数:
            无

        返回:
            Path
        """
        return self._matrix_root

    @staticmethod
    def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
        """原子写 parquet（tmp + os.replace），崩溃残留不命中固定 .tmp。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        df.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(str(tmp), str(path))

    @staticmethod
    def _read_existing_or_quarantine(
        parquet_path: Path, *, immutable: bool = False
    ) -> pd.DataFrame:
        """P0-22: 旧分区读失败 → 隔离 quarantine + 抛 ``FactorMatrixReadError``。

        绝不把损坏 / IO 抖动 / 权限问题当作空分区继续覆盖——那会把一整月历史
        在增量运行时无声丢掉。原文件先移入 ``<part_dir>/.quarantine/<ts>-data.parquet``
        再 hard fail，数据保留待人工恢复。

        R14 #2：``immutable=True`` 时读的是**已发布 generation** 里的文件——该代
        由 ``manifest.generation`` 引用，在途 reader 可能正在读它。此时**只复制**
        到 quarantine 留证、绝不 ``os.replace`` 原文件（把 active generation 文件
        移走会让在途 reader 读到半套、且新 generation 构造失败后 manifest 仍指向
        这个残缺旧代），直接把该 generation 判 corrupt + hard fail。
        """
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            qdir = parquet_path.parent / ".quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            qpath = qdir / f"{ts}-{uuid.uuid4().hex[:8]}-data.parquet"
            try:
                if immutable:
                    # 已发布 generation immutable：留证副本，不动原文件。
                    shutil.copy2(str(parquet_path), str(qpath))
                else:
                    os.replace(str(parquet_path), str(qpath))
            except Exception:
                qpath = parquet_path
            tag = "（已发布 generation，immutable：仅留证副本，原文件未移动）" if immutable else ""
            raise FactorMatrixReadError(
                f"factor_matrix 分区读取失败 {parquet_path}{tag}，已隔离至 "
                f"quarantine={qpath}，拒绝继续 publish（历史可能丢失）: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _validate_staging(
        staging_path: Path,
        expected: pd.DataFrame,
    ) -> pd.DataFrame:
        """P1-14: 读回校验 staging（行数 / 列数 / 无全 NaN 列 / 值一致性）。

        R39 PERF-064：``FACTOR_ENGINE_MATRIX_CHECKSUM_PROOF=1`` 时改用 checksum
        proof —— writer 先计算 ``key_order_checksum`` / per-column
        ``finite_mask_checksum`` / ``numeric_checksum`` / ``row_count`` /
        ``schema_hash``，read-back 用 Arrow 列式读取后比对 checksum（不做两个
        巨大矩阵的 pandas 全量 sort + 逐列 ``np.allclose``）。否则保持原
        value-compare 作为 reference 兜底路径。

        校验失败抛 ``FactorMatrixReadError``，staging 不会进入正式位置。
        """
        if checksum_proof_enabled():
            try:
                expected_checks = compute_matrix_checksums(expected)
                actual_checks = read_parquet_checksums(
                    staging_path, expected.columns
                )
                errors = compare_checksums(expected_checks, actual_checks)
            except Exception as exc:
                raise FactorMatrixReadError(
                    f"staging checksum 校验失败 {staging_path}（读取/解析异常）: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if errors:
                raise FactorMatrixReadError(
                    f"staging checksum 校验失败 {staging_path}: " + "; ".join(errors)
                )
            return pd.read_parquet(staging_path)
        rb = pd.read_parquet(staging_path)
        if len(rb) != len(expected):
            raise FactorMatrixReadError(
                f"staging 校验失败 {staging_path}: 行数 {len(rb)} != 期望 {len(expected)}"
            )
        if set(rb.columns) != set(expected.columns):
            raise FactorMatrixReadError(
                f"staging 校验失败 {staging_path}: 列集合不一致 "
                f"{sorted(rb.columns)} != {sorted(expected.columns)}"
            )
        key_cols = ["datetime", "asset"]
        exp_cols = [c for c in expected.columns if c not in key_cols]
        for c in exp_cols:
            if expected[c].isna().all():
                continue
            if rb[c].isna().all():
                raise FactorMatrixReadError(
                    f"staging 校验失败 {staging_path}: 列 {c} 读回为全 NaN"
                )
        exp_sorted = expected.sort_values(key_cols).reset_index(drop=True)
        rb_sorted = rb.sort_values(key_cols).reset_index(drop=True)
        for c in exp_cols:
            ev = exp_sorted[c].astype("float64")
            rv = rb_sorted[c].astype("float64")
            if not (ev.isna() == rv.isna()).all():
                raise FactorMatrixReadError(
                    f"staging 校验失败 {staging_path}: 列 {c} NaN 掩码不一致"
                )
            mask = ~ev.isna()
            if mask.any() and not np.allclose(ev[mask], rv[mask]):
                raise FactorMatrixReadError(
                    f"staging 校验失败 {staging_path}: 列 {c} 数值不一致"
                )
        return rb

    def _current_manifest_version(self, base: Path) -> int:
        manifest = _read_manifest(base)
        return int((manifest or {}).get("manifest_version", 0))

    def _update_manifest(
        self,
        base: Path,
        universe: str,
        frequency: str,
        factor_ids: list[str],
        factor_versions: dict[str, str] | None,
        expected_manifest_version: int | None,
        generation: str | None = None,
        factor_blocks: dict[str, dict[str, Any]] | None = None,
        certificates: dict[str, Any] | None = None,
        factor_block_refs: list[dict[str, Any]] | None = None,
        partition_inventory: list[PartitionObjectRef] | None = None,
    ) -> dict[str, Any]:
        """P0-23 + P1-15: 写 manifest（semantic_digest 校验 + CAS 版本比对）。

        ``factor_versions``（factor_id → semantic_digest/version）缺省为 None 时
        只推进 ``updated_at``/``manifest_version``，不做 digest 绑定。

        R39 PERF-061/067：``factor_blocks``（factor_id → {block_id, column}）在
        column-factor block layout 下把 ``factor_id → block_id`` 记入 manifest，
        ``load_matrix`` 据此只扫需要的 block；``certificates``（R39 PERF-067
        MaterializationIdentityCertificate）把 compile-time 构建的身份字段直接
        落入 manifest，writer 不再为每 factor 重建 metadata。

        R39 PERF-030：``factor_block_refs``（可序列化 ``FactorBlockRef`` 元数据，
        见 ``runtime.factor_block_ref.block_ref_meta``）随 manifest 记录——多个因子
        共享一份 axis 的证据（axis_key / row_count / factor_count / block_id）。

        R14 #1：``generation`` 是本次已完整 validate 的新 generation id。写入后
        manifest 的 ``os.replace`` 就是全矩阵的**唯一原子切换点**——reader 从
        ``manifest.generation`` 解析读取目录，任何崩溃窗口都看不到半新半旧。
        """
        factor_versions = factor_versions or {}
        with _manifest_write_lock(base):
            manifest = _read_manifest(base)
            if manifest is None:
                manifest = {
                    "universe": universe,
                    "frequency": frequency,
                    "factors": {},
                    "updated_at": _now_iso(),
                    "manifest_version": 0,
                }
            if (
                manifest.get("universe") != universe
                or manifest.get("frequency") != frequency
            ):
                raise FactorMatrixCorruptionError(
                    f"manifest 与当前 universe/freq 不符: "
                    f"{manifest.get('universe')}/{manifest.get('frequency')}"
                )
            factors = manifest.setdefault("factors", {})
            # P0-23: 已有 digest 与本次不同 → 禁止混列
            for fid, digest in factor_versions.items():
                prev = (factors.get(fid) or {}).get("semantic_digest")
                if prev and prev != digest:
                    raise FactorMatrixVersionMismatchError(
                        f"factor {fid} semantic_digest 变化 {prev} -> {digest}，"
                        f"禁止新旧月份混列（请全历史重算，或写独立 versioned matrix）"
                    )
            # P1-15: manifest CAS —— 期望版本缺省为物化开始时读到的当前值
            current_version = manifest.get("manifest_version", 0)
            if (
                expected_manifest_version is not None
                and current_version != expected_manifest_version
            ):
                raise FactorMatrixConcurrentWriteError(
                    f"manifest CAS 失败: 期望版本 {expected_manifest_version}，"
                    f"当前 {current_version}（并发写同一 universe/freq，请读新值后"
                    f"重新 merge 再发布）"
                )
            for fid in factor_ids:
                digest = factor_versions.get(fid)
                if digest is None:
                    continue
                entry = factors.setdefault(
                    fid,
                    {
                        "version": None,
                        "semantic_digest": None,
                        "operator_manifest_hash": None,
                        "field_catalog_hash": None,
                        "source_snapshot": None,
                    },
                )
                entry["version"] = digest
                entry["semantic_digest"] = digest
                for field in (
                    "operator_manifest_hash",
                    "field_catalog_hash",
                    "source_snapshot",
                ):
                    entry.setdefault(field, None)
            # R39 PERF-061: factor_id -> {block_id, column} for column-factor blocks.
            if factor_blocks:
                manifest["layout"] = "block"
                for fid, fb in factor_blocks.items():
                    entry = factors.setdefault(fid, {})
                    entry["block_id"] = str(fb["block_id"])
                    entry["column"] = str(fb.get("column", fid))
            # R39 PERF-067: consume compile-time MaterializationIdentityCertificate.
            if certificates:
                for fid, cert in certificates.items():
                    entry = factors.setdefault(str(fid), {})
                    if cert.semantic_digest:
                        entry["semantic_digest"] = cert.semantic_digest
                        entry["version"] = cert.semantic_digest
                    for fld, val in (
                        ("source_snapshot", cert.source_snapshot),
                        ("calendar", cert.calendar),
                        ("universe", cert.universe),
                        ("frequency", cert.frequency),
                        ("storage_precision", cert.storage_precision),
                        ("operator_manifest_hash", cert.operator_manifest_hash),
                    ):
                        if val is not None:
                            entry[fld] = val
            # R39 PERF-030: 多因子共享 axis 的 FactorBlockRef 元数据（可序列化）。
            if factor_block_refs:
                manifest["factor_block_refs"] = factor_block_refs
            # R39 PERF-063: generation 的完整 partition inventory（COW 精确 materialize，
            # 不再每次 rglob 全目录发现）。
            if partition_inventory is not None:
                manifest["partition_inventory"] = inventory_to_dicts(partition_inventory)
            manifest["updated_at"] = _now_iso()
            manifest["manifest_version"] = current_version + 1
            if generation:
                manifest["generation"] = generation
            _write_manifest_atomic(base, manifest)
            return manifest

    def materialize(
        self,
        results: dict[str, pd.Series],
        *,
        universe: str,
        frequency: str = "1d",
        partition_columns: Iterable[str] | None = None,
        value_dtype: str = "float32",
        production: bool | None = None,
        recovery: bool = False,
        factor_versions: dict[str, str] | None = None,
        expected_manifest_version: int | None = None,
        certificates: dict[str, Any] | None = None,
        layout: str | None = None,
    ) -> dict[str, Any]:
        """``factor_id -> Series`` 合并为宽表并按 hive 分区落盘。

        参数:
            results: 见函数签名
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            partition_columns: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
            production: 显式 production 标志；缺省从运行模式解析（可选）。
                production 对每个分区写后再读回 validate，validate 全部通过才
                原子切换 manifest generation 指针。
            recovery: **已弃用**（P0-22）：分区读失败一律 quarantine + hard fail，
                不再支持按空分区破坏性重建；保留参数仅为兼容旧调用方。
            factor_versions: factor_id → semantic_digest/version 绑定（可选，
                P0-23）；缺省时 manifest 只记录本次时间，不做 digest 校验。
            expected_manifest_version: manifest CAS 期望版本（可选，P1-15）；
                缺省为物化开始时读到的当前值。
            certificates: R39 PERF-067 —— ``fid -> MaterializationIdentityCertificate``
                编译期缓存，writer 直接消费（可选）。
            layout: R39 PERF-061 —— ``"legacy"``（默认单宽文件）或 ``"block"``
                （column-factor block）；缺省 ``"auto"`` 从环境变量
                ``FACTOR_ENGINE_MATRIX_BLOCK_LAYOUT=1`` 解析为 block，否则 legacy。

        返回:
            dict[str, Any]

        R39 PERF-060：
        宽表装配不再按 factor 逐个 pandas outer merge。所有结果 axis 相同时
        直接 column-stack（``build_matrix_block`` 的零 join 路径，
        ``matrix_join_count == 0``）；axis 不同时只做**一次** canonical row-key
        index + 逐因子 vectorized reindex（``matrix_join_count == 1``）。

        R14 #1（crash-atomic publish）：
        整个 ``(universe, frequency)`` 是一个 generation。publish 顺序是
        「完整写入 ``generation/<gid>/``（含未触达分区的 copy-on-write）→ 全部
        validate → 一次 ``os.replace`` 切 ``manifest.json`` 的 ``generation``
        指针」。reader 只读 ``manifest.generation`` 指向的那一代，任何崩溃窗口
        只能看到完整 old generation 或完整 new generation，绝无半新半旧。
        """
        # #收官轮 P0：production 由调用方（matrix_service 从 engine.run_mode）
        # 显式传入，缺省才回退到运行模式解析——下游不再自行猜运行模式。
        effective_production = self._resolve_production(production)
        if not results:
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": [],
                "matrix_root": str(self._matrix_root),
            }

        effective_layout = str(layout or "auto").lower()
        if effective_layout == "auto":
            effective_layout = "block" if block_mode_enabled() else "legacy"
        if effective_layout == "block":
            return self._materialize_block(
                results,
                universe=universe,
                frequency=frequency,
                partition_columns=partition_columns,
                value_dtype=value_dtype,
                production=effective_production,
                factor_versions=factor_versions,
                expected_manifest_version=expected_manifest_version,
                certificates=certificates,
            )

        factor_ids = sorted(results.keys())
        # R39 PERF-060: replace the per-factor pairwise outer merge with a zero-join
        # column stack (equal axis) or a single canonical-reindex (different axis).
        _axis, wide, _joins = build_matrix_block(
            {fid: results[fid] for fid in factor_ids}
        )
        merged = wide_to_merged(wide, factor_ids, value_dtype=value_dtype)

        if merged is None or merged.empty:
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": factor_ids,
                "matrix_root": str(self._matrix_root),
            }

        layout = FactorMatrixLayout(
            universe=universe,
            frequency=frequency,
            partition_columns=tuple(partition_columns or ("year", "month")),
        )
        policy = PartitionPolicy.from_config(
            partition_columns=layout.partition_columns,
            storage_format="long",
        )
        work = attach_partition_columns(merged, policy)
        base = layout.base_dir(self._matrix_root)
        base.mkdir(parents=True, exist_ok=True)

        # R14 #1: 整个 (universe,freq) 在 base 级 flock 内完成「generation 重建 +
        # manifest 原子切换」。同 host 并发 writer serial 化；跨 host 由 manifest
        # CAS（expected_manifest_version）兜底。
        with _manifest_write_lock(base):
            snapshot_version = self._current_manifest_version(base)
            expected = (
                expected_manifest_version
                if expected_manifest_version is not None
                else snapshot_version
            )
            manifest = _read_manifest(base)
            old_gid = (manifest or {}).get("generation")
            old_gen_dir = base / "generation" / old_gid if old_gid else None
            # R14 #2 fail-closed：manifest 声明了 generation 但目录缺失 = 当前发布代
            # 损坏。绝不能静默当「无旧代」从零重建——read-merge-write 会丢掉全部未触达
            # 分区的历史。
            if old_gid and old_gen_dir is not None and not old_gen_dir.is_dir():
                raise FactorMatrixCorruptionError(
                    f"manifest.generation={old_gid!r} 指向的目录缺失（{old_gen_dir}），"
                    f"当前发布代损坏。拒绝增量 publish（会静默丢历史），请人工恢复该代"
                    f"或清除 manifest 后全量重建。"
                )

            # R39 PERF-063: 旧 generation 的 partition inventory。manifest 已带
            # inventory 时直接复用（精确 materialize，不 rglob）；缺 inventory
            # （pre-R39 legacy 代首次遇到）才做一次性 rglob 构建（计数 +1）。
            old_inventory: list[PartitionObjectRef] = []
            if old_gen_dir is not None:
                manifest_inv = (manifest or {}).get("partition_inventory")
                if manifest_inv is not None:
                    old_inventory = inventory_from_dicts(manifest_inv)
                else:
                    old_inventory = build_partition_inventory(old_gen_dir)

            new_gid = uuid.uuid4().hex
            new_gen_dir = base / "generation" / new_gid
            new_gen_dir.mkdir(parents=True, exist_ok=True)

            partitions_written: list[str] = []
            written_rel: set[str] = set()
            written_refs: list[PartitionObjectRef] = []
            try:
                # 1) 本次触达的分区：read-merge-write 进新 generation（production
                #    写后再读回 validate，validate 全部通过才允许切指针）。
                for part_values, partition_df in iter_partition_groups(work, policy):
                    drop_cols = [c for c in policy.columns if c in partition_df.columns]
                    out_df = partition_df.drop(columns=drop_cols).reset_index(drop=True)
                    rel = (
                        partition_path_segments(
                            part_values, column_order=policy.columns
                        )
                        / "data.parquet"
                    )
                    new_path = new_gen_dir / rel
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path = old_gen_dir / rel if old_gen_dir is not None else None
                    if old_path is not None and old_path.exists():
                        # R14 #2：读已发布 generation 文件 → immutable，损坏只留证
                        # 副本并 hard fail，不移动被 manifest 引用的原文件。
                        existing = self._read_existing_or_quarantine(
                            old_path, immutable=True
                        )
                        merged_out = _merge_matrix_frames(
                            existing, out_df, value_dtype=value_dtype
                        )
                    else:
                        merged_out = out_df
                    self._write_parquet_atomic(new_path, merged_out)
                    if effective_production:
                        self._validate_staging(new_path, merged_out)
                    written_rel.add(rel.as_posix())
                    written_refs.append(ref_from_frame(rel.as_posix(), merged_out))
                    pkey = "|".join(
                        f"{k}={part_values[k]}" for k in sorted(part_values)
                    )
                    partitions_written.append(pkey)
                    logger.info(
                        "factor_matrix 分区 upsert universe=%s partition=%s "
                        "rows=%d cols=%d generation=%s",
                        universe,
                        pkey,
                        len(merged_out),
                        len(merged_out.columns),
                        new_gid,
                    )

                # 2) 未触达的分区：按 inventory 精确 materialize（exact rel_path，
                #    不再 rglob 全目录发现）。partial 更新不用把整个矩阵重写一遍。
                if old_gen_dir is not None and old_inventory:
                    materialize_generation_from_inventory(
                        old_gen_dir,
                        old_inventory,
                        new_gen_dir,
                        skip_rel=written_rel,
                    )

                # 2.5) 新 generation 的完整 partition inventory = 本次写入的 refs
                #     + 未触达（inventory 精确 COW）的旧 refs，随 manifest 落盘，
                #     供下一次 generation 切换精确 materialize。
                new_inventory = list(written_refs)
                for ref in old_inventory:
                    if ref.rel_path not in written_rel:
                        new_inventory.append(ref)

                # 3) manifest：digest 校验 + CAS + generation 指针 + inventory，
                #    一次 os.replace 原子切换——这是 reader 唯一能感知新数据的边界。
                manifest = self._update_manifest(
                    base,
                    universe,
                    frequency,
                    factor_ids,
                    factor_versions,
                    expected,
                    generation=new_gid,
                    certificates=certificates,
                    partition_inventory=new_inventory,
                )

                # 4) GC：只保留当前与上一代。在途 reader 可能仍按旧 manifest 读
                #    上一代文件，因此上一代不能马上删。
                for gdir in (base / "generation").glob("*"):
                    if gdir.is_dir() and gdir.name not in {new_gid, old_gid or ""}:
                        shutil.rmtree(gdir, ignore_errors=True)
            except Exception:
                # 新 generation 尚未被 manifest 引用（孤儿），失败即清理，不残留。
                shutil.rmtree(new_gen_dir, ignore_errors=True)
                raise

        logger.info(
            "factor_matrix manifest updated universe=%s version=%s factors=%d "
            "generation=%s",
            universe,
            manifest.get("manifest_version"),
            len(manifest.get("factors", {})),
            new_gid,
        )
        return {
            "universe": universe,
            "frequency": frequency,
            "rows_written": len(merged),
            "partitions": partitions_written,
            "factor_ids": factor_ids,
            "matrix_root": str(self._matrix_root),
            "columns": ["datetime", "asset", *factor_ids],
            "manifest_version": manifest.get("manifest_version"),
            "generation": new_gid,
            "layout": "legacy",
            "matrix_join_count": matrix_join_count,
        }

    def materialize_from_iterable(
        self,
        results_iter: Iterable[tuple[str, pd.Series]],
        *,
        universe: str,
        frequency: str = "1d",
        partition_columns: Iterable[str] | None = None,
        value_dtype: str = "float32",
        production: bool | None = None,
        recovery: bool = False,
        factor_versions: dict[str, str] | None = None,
        expected_manifest_version: int | None = None,
        certificates: dict[str, Any] | None = None,
        layout: str | None = "block",
    ) -> dict[str, Any]:
        """R39 PERF-066: consume ``(fid, series)`` incrementally (streaming adapter).

        The engine boundary is documented: ``engine.run_many`` returns a full
        results dict, so this buffers *references* (no data copy) and delegates to
        ``materialize``.  In the column-factor block layout the writer never builds
        one monolithic wide frame — each block is assembled and written before the
        next, bounding writer memory to one block of factors at a time.
        """
        buffered = dict(results_iter)
        return self.materialize(
            buffered,
            universe=universe,
            frequency=frequency,
            partition_columns=partition_columns,
            value_dtype=value_dtype,
            production=production,
            recovery=recovery,
            factor_versions=factor_versions,
            expected_manifest_version=expected_manifest_version,
            certificates=certificates,
            layout=layout,
        )

    # ------------------------------------------------------------------
    # R39 PERF-061/062: column-factor block layout (opt-in)
    # ------------------------------------------------------------------
    def _assign_blocks(
        self,
        existing_blocks: dict[str, str],
        factor_ids: list[str],
        manifest_block_ids: set[str] | None = None,
    ) -> dict[str, str]:
        """Assign ``fid -> block_id``.

        Factors already present in the manifest keep their existing block; new
        factors are grouped into *fresh* blocks of up to ``block_columns()`` factors
        so adding new factors never rewrites existing column blocks.

        ``manifest_block_ids`` is the full block inventory from the manifest (all
        factors, not just this run's), so a fresh block never collides with an
        existing block even when the run touches only a subset of factors.
        """
        assignment: dict[str, str] = {}
        for fid in factor_ids:
            bid = existing_blocks.get(fid)
            if bid is not None:
                assignment[fid] = bid
        known = manifest_block_ids or set(existing_blocks.values())
        known_ints = [int(b) for b in known if str(b).isdigit()]
        next_block = max(known_ints) + 1 if known_ints else 0
        new_fids = [fid for fid in factor_ids if fid not in assignment]
        cap = block_columns()
        for i in range(0, len(new_fids), cap):
            chunk = new_fids[i : i + cap]
            bid = f"{next_block:04d}"
            next_block += 1
            for fid in chunk:
                assignment[fid] = bid
        return assignment

    def _materialize_block(
        self,
        results: dict[str, pd.Series],
        *,
        universe: str,
        frequency: str,
        partition_columns: Iterable[str] | None,
        value_dtype: str,
        production: bool,
        factor_versions: dict[str, str] | None,
        expected_manifest_version: int | None,
        certificates: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """R39 PERF-061/062: column-factor block materialization.

        Layout: ``year=YYYY/month=MM/block=NNNN.parquet``; manifest records
        ``factor_id -> {block_id, column}``.

        Incremental semantics (PERF-062): only *touched* blocks are read-merge-
        written (via ``overlay_block``, never the whole-month wide table);
        untouched blocks are copy-on-write hardlinked; brand-new factors get fresh
        block files so existing column blocks are never rewritten.  The
        ``matrix_rewrite_amplification`` metric accumulates historical-rewrite-bytes
        vs changed-logical-bytes.
        """
        factor_ids = sorted(results.keys())
        # 与 legacy 路径一致：空结果（全空 axis）不发布 generation。
        if not any(len(series) for series in results.values()):
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": factor_ids,
                "matrix_root": str(self._matrix_root),
                "layout": "block",
                "matrix_rewrite_amplification": matrix_rewrite_amplification,
                "factor_block_refs": [],
            }
        layout = FactorMatrixLayout(
            universe=universe,
            frequency=frequency,
            partition_columns=tuple(partition_columns or ("year", "month")),
        )
        policy = PartitionPolicy.from_config(
            partition_columns=layout.partition_columns,
            storage_format="long",
        )
        base = layout.base_dir(self._matrix_root)
        base.mkdir(parents=True, exist_ok=True)

        with _manifest_write_lock(base):
            snapshot_version = self._current_manifest_version(base)
            expected = (
                expected_manifest_version
                if expected_manifest_version is not None
                else snapshot_version
            )
            manifest = _read_manifest(base)
            old_gid = (manifest or {}).get("generation")
            old_gen_dir = base / "generation" / old_gid if old_gid else None
            if old_gid and (old_gen_dir is None or not old_gen_dir.is_dir()):
                raise FactorMatrixCorruptionError(
                    f"manifest.generation={old_gid!r} 指向的目录缺失（{old_gen_dir}），"
                    f"当前发布代损坏。拒绝增量 publish（会静默丢历史），请人工恢复该代"
                    f"或清除 manifest 后全量重建。"
                )

            # R39 PERF-063: 旧 generation 的 partition inventory（manifest 优先，
            # 缺 inventory 才一次性 rglob 构建，计数 +1）。
            old_inventory: list[PartitionObjectRef] = []
            if old_gen_dir is not None:
                manifest_inv = (manifest or {}).get("partition_inventory")
                if manifest_inv is not None:
                    old_inventory = inventory_from_dicts(manifest_inv)
                else:
                    old_inventory = build_partition_inventory(old_gen_dir)

            existing_blocks = resolve_factor_blocks(manifest, factor_ids)
            manifest_block_ids: set[str] = set()
            for _entry in ((manifest or {}).get("factors", {}) or {}).values():
                _bid = _entry.get("block_id") if isinstance(_entry, dict) else None
                if _bid is not None:
                    manifest_block_ids.add(str(_bid))
            assignment = self._assign_blocks(
                existing_blocks, factor_ids, manifest_block_ids
            )
            from collections import defaultdict

            groups: dict[str, list[str]] = defaultdict(list)
            for fid in factor_ids:
                groups[assignment[fid]].append(fid)

            new_gid = uuid.uuid4().hex
            new_gen_dir = base / "generation" / new_gid
            new_gen_dir.mkdir(parents=True, exist_ok=True)

            partitions_written: list[str] = []
            written_rel: set[str] = set()
            written_refs: list[PartitionObjectRef] = []
            rows_written = 0
            block_refs_meta: list[dict[str, Any]] = []
            try:
                for block_id in sorted(groups):
                    bfids = sorted(groups[block_id])
                    _axis, wide, _j = build_matrix_block(
                        {fid: results[fid] for fid in bfids}
                    )
                    # R39-PERF-030: 同 axis 因子（column-stack 完成、零 join）构造
                    # FactorBlockRef——多因子共享一份 axis 的观测/载体。记录计数与
                    # 可序列化元数据（随 manifest/summary），不改变写入文件。
                    if len(bfids) >= 2 and _j == 0:
                        from runtime.factor_block_ref import (
                            block_ref_meta,
                            build_factor_block,
                            record_block_ref_used,
                        )

                        fbr = build_factor_block(
                            bfids,
                            wide.to_numpy(dtype=value_dtype),
                            index=_axis,
                            dtype=value_dtype,
                        )
                        record_block_ref_used(fbr)
                        block_refs_meta.append(
                            block_ref_meta(fbr, block_id=block_id)
                        )
                    block_merged = wide_to_merged(wide, bfids, value_dtype=value_dtype)
                    work = attach_partition_columns(block_merged, policy)
                    for part_values, partition_df in iter_partition_groups(work, policy):
                        drop_cols = [
                            c for c in policy.columns if c in partition_df.columns
                        ]
                        out_df = partition_df.drop(columns=drop_cols).reset_index(drop=True)
                        rel = (
                            partition_path_segments(
                                part_values, column_order=policy.columns
                            )
                            / block_file_name(block_id)
                        )
                        new_path = new_gen_dir / rel
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        old_path = old_gen_dir / rel if old_gen_dir is not None else None
                        rewritten_bytes = 0
                        if old_path is not None and old_path.exists():
                            existing = self._read_existing_or_quarantine(
                                old_path, immutable=True
                            )
                            merged_out = overlay_block(
                                existing, out_df, value_dtype=value_dtype
                            )
                            rewritten_bytes = old_path.stat().st_size
                        else:
                            merged_out = out_df
                        self._write_parquet_atomic(new_path, merged_out)
                        if production:
                            self._validate_staging(new_path, merged_out)
                        changed_logical = int(
                            len(out_df) * len(out_df.columns) * 4
                        )
                        record_rewrite_amplification(rewritten_bytes, changed_logical)
                        rows_written += len(merged_out)
                        written_rel.add(rel.as_posix())
                        written_refs.append(ref_from_frame(rel.as_posix(), merged_out))
                        pkey = "|".join(
                            f"{k}={part_values[k]}" for k in sorted(part_values)
                        )
                        partitions_written.append(pkey)
                        logger.info(
                            "factor_matrix block upsert universe=%s partition=%s "
                            "block=%s rows=%d cols=%d generation=%s",
                            universe,
                            pkey,
                            block_id,
                            len(merged_out),
                            len(merged_out.columns),
                            new_gid,
                        )

                # 2) 未触达的 block：按 inventory 精确 COW materialize（新增因子
                #    绝不重写既有 block；不再 rglob 全目录发现）。
                if old_gen_dir is not None and old_inventory:
                    materialize_generation_from_inventory(
                        old_gen_dir,
                        old_inventory,
                        new_gen_dir,
                        skip_rel=written_rel,
                    )

                # 2.5) 新 generation 的完整 partition inventory（本次写入 + 未触达
                #     COW 的旧 refs），随 manifest 落盘供下次精确 materialize。
                new_inventory = list(written_refs)
                for ref in old_inventory:
                    if ref.rel_path not in written_rel:
                        new_inventory.append(ref)

                # 3) manifest：factor_id -> block_id/column + digest + CAS + pointer
                #    + partition inventory。
                factor_blocks = {
                    fid: {"block_id": assignment[fid], "column": fid}
                    for fid in factor_ids
                }
                manifest = self._update_manifest(
                    base,
                    universe,
                    frequency,
                    factor_ids,
                    factor_versions,
                    expected,
                    generation=new_gid,
                    factor_blocks=factor_blocks,
                    certificates=certificates,
                    factor_block_refs=block_refs_meta,
                    partition_inventory=new_inventory,
                )

                # 4) GC：只保留当前与上一代。
                for gdir in (base / "generation").glob("*"):
                    if gdir.is_dir() and gdir.name not in {new_gid, old_gid or ""}:
                        shutil.rmtree(gdir, ignore_errors=True)
            except Exception:
                shutil.rmtree(new_gen_dir, ignore_errors=True)
                raise

        logger.info(
            "factor_matrix block manifest updated universe=%s version=%s factors=%d "
            "generation=%s",
            universe,
            manifest.get("manifest_version"),
            len(manifest.get("factors", {})),
            new_gid,
        )
        return {
            "universe": universe,
            "frequency": frequency,
            "rows_written": rows_written,
            "partitions": partitions_written,
            "factor_ids": factor_ids,
            "matrix_root": str(self._matrix_root),
            "columns": ["datetime", "asset", *factor_ids],
            "manifest_version": manifest.get("manifest_version"),
            "generation": new_gid,
            "layout": "block",
            "matrix_rewrite_amplification": matrix_rewrite_amplification,
            "factor_block_refs": block_refs_meta,
        }

    @staticmethod
    def _resolve_production(production: bool | None) -> bool:
        """解析 production 标志：显式值优先，否则从运行模式推断。"""
        if production is not None:
            return bool(production)
        try:
            from runtime.production_policy import is_production_mode

            return is_production_mode()
        except Exception:  # pragma: no cover - 解析失败按 research 处理
            return False

    @staticmethod
    def _parquet_available_columns(path: Path) -> list[str]:
        """Column names present in a parquet file (footer-only, cheap metadata read)."""
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(str(path)).schema_arrow.names)

    @staticmethod
    def _frames_from_legacy(
        dir_path: Path,
        manifest: dict[str, Any] | None,
        *,
        factor_ids: Iterable[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Iterable[str] | None,
    ) -> list[pd.DataFrame]:
        """R39 PERF-065: legacy data.parquet read with pushdown.

        * ``columns=`` pushdown to the parquet reader for the requested factors;
        * partition pruning by ``time_range`` (year/month hive dirs);
        * ``instrument_filter`` applied as a row filter after column pushdown;
        * R39 PERF-063: manifest 已有 ``partition_inventory`` 时优先用其精确
          ``rel_path`` 集合枚举文件（不 ``rglob`` 全目录发现），无 inventory 才
          回退 rglob。
        """
        frames: list[pd.DataFrame] = []
        need_cols: list[str] | None = None
        if factor_ids is not None:
            need_cols = ["datetime", "asset", *sorted(set(str(f) for f in factor_ids))]
        inventory = (manifest or {}).get("partition_inventory")
        if inventory is not None:
            parquet_paths = inventory_parquet_paths(
                dir_path, inventory_from_dicts(inventory), suffix="data.parquet"
            )
        else:
            parquet_paths = [
                p
                for p in sorted(dir_path.rglob("data.parquet"))
                if ".staging" not in p.parts
                and ".quarantine" not in p.parts
                and not p.name.startswith(".")
            ]
        for pq_path in parquet_paths:
            if not partition_overlaps_time_range(pq_path.parent, time_range):
                continue
            read_cols: list[str] | None = None
            if need_cols is not None:
                try:
                    avail = FactorMatrixMaterializer._parquet_available_columns(pq_path)
                    read_cols = [c for c in need_cols if c in avail]
                except Exception:
                    read_cols = None  # footer read failure -> full read, fail loud below
            try:
                if read_cols is not None:
                    frame = pd.read_parquet(pq_path, columns=read_cols)
                else:
                    frame = pd.read_parquet(pq_path)
            except Exception as exc:
                # R14 #2：当前 generation 内文件损坏 = 发布代损坏，fail loud（
                # 绝不当作空分区静默跳过——那会把一整月历史在读取侧无声丢掉）。
                raise FactorMatrixReadError(
                    f"factor_matrix 分区读取失败 {pq_path}（当前 generation 损坏，"
                    f"拒绝继续）: {type(exc).__name__}: {exc}"
                ) from exc
            if instrument_filter is not None:
                frame = frame[frame["asset"].isin(set(str(x) for x in instrument_filter))]
            frames.append(frame)
        return frames

    @staticmethod
    def _frames_from_block(
        dir_path: Path,
        manifest: dict[str, Any] | None,
        *,
        factor_ids: Iterable[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Iterable[str] | None,
    ) -> list[pd.DataFrame]:
        """R39 PERF-061/065: column-factor block read with pushdown.

        Uses the manifest ``factor_id -> block_id`` mapping to scan only the needed
        ``block=NNNN.parquet`` files, passes ``columns=`` to the reader, prunes
        partitions by ``time_range``, and outer-joins the blocks of each partition
        back into a wide frame (only when multiple blocks are involved).
        """
        from collections import defaultdict

        need_fids: list[str] | None = None
        if factor_ids is not None:
            need_fids = [str(f) for f in factor_ids]
        else:
            need_fids = list((manifest or {}).get("factors", {}).keys()) or None

        factor_block_map = resolve_factor_blocks(manifest, need_fids or [])
        needed_blocks: set[str] = set()
        if need_fids is not None:
            for fid in need_fids:
                bid = factor_block_map.get(fid)
                if bid is not None:
                    needed_blocks.add(bid)
        else:
            for bid in factor_block_map.values():
                needed_blocks.add(bid)

        # Group needed block files by partition directory.
        # R39 PERF-063: manifest 已有 inventory 时优先用精确 rel_path 集合枚举
        # block 文件（不再 rglob 全目录发现）。
        partitions: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
        inventory = (manifest or {}).get("partition_inventory")
        if inventory is not None:
            block_paths = inventory_parquet_paths(
                dir_path, inventory_from_dicts(inventory), suffix=".parquet"
            )
            block_paths = [
                p for p in block_paths if parse_block_id(p.name) is not None
            ]
        else:
            block_paths = [
                p
                for p in sorted(dir_path.rglob("block=*.parquet"))
                if ".staging" not in p.parts
                and ".quarantine" not in p.parts
                and not p.name.startswith(".")
            ]
        for pq_path in block_paths:
            bid = parse_block_id(pq_path.name)
            if bid is None:
                continue
            if needed_blocks and bid not in needed_blocks:
                continue
            if not partition_overlaps_time_range(pq_path.parent, time_range):
                continue
            partitions[pq_path.parent].append((bid, pq_path))

        frames: list[pd.DataFrame] = []
        for part_path in sorted(partitions):
            part_frames: list[pd.DataFrame] = []
            for bid, pq_path in sorted(partitions[part_path]):
                read_cols: list[str] | None = None
                if need_fids is not None:
                    try:
                        avail = FactorMatrixMaterializer._parquet_available_columns(pq_path)
                        read_cols = ["datetime", "asset"] + [
                            c for c in need_fids if c in avail
                        ]
                    except Exception:
                        read_cols = None
                try:
                    if read_cols is not None:
                        frame = pd.read_parquet(pq_path, columns=read_cols)
                    else:
                        frame = pd.read_parquet(pq_path)
                except Exception as exc:
                    raise FactorMatrixReadError(
                        f"factor_matrix block 读取失败 {pq_path}（当前 generation "
                        f"损坏，拒绝继续）: {type(exc).__name__}: {exc}"
                    ) from exc
                if instrument_filter is not None:
                    frame = frame[
                        frame["asset"].isin(set(str(x) for x in instrument_filter))
                    ]
                part_frames.append(frame)
            if not part_frames:
                continue
            if len(part_frames) == 1:
                frames.append(part_frames[0])
            else:
                # Outer-join the column blocks of one partition back into a wide
                # frame (read-side only; the write side never N-way merges).
                merged_part = part_frames[0]
                for extra in part_frames[1:]:
                    merged_part = merged_part.merge(
                        extra, on=["datetime", "asset"], how="outer"
                    )
                frames.append(merged_part)
        return frames

    @staticmethod
    def load_matrix(
        matrix_root: str | Path,
        *,
        universe: str,
        frequency: str = "1d",
        factor_ids: Iterable[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读取 universe 下全部或指定因子列宽表。

        R14 #1：只读取 ``manifest.generation`` 指向的那一代（通过 generation
        指针隔离发布中的 half-published 状态）。无 ``generation`` 字段的旧布局
        按 legacy 全局 glob 兜底。

        R39 PERF-065（pushdown）：支持 ``time_range``（``(start, end)``，含端点）
        按 year/month hive 分区裁剪，``factor_ids`` 通过 ``columns=`` 下推到
        parquet reader（不先读全表再选列）；column-factor block layout 下只扫
        ``factor_id -> block_id`` 命中的 block 文件。``instrument_filter`` 作为
        row 过滤在列下推后应用。

        参数:
            matrix_root: factor_matrix 根目录
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            factor_ids: 因子 ID 列表（可选）
            time_range: (start, end) 时间窗口（可选，含端点）
            instrument_filter: 标的过滤集合（可选）

        返回:
            pd.DataFrame
        """
        base = (
            Path(matrix_root)
            / f"universe={universe}"
            / f"freq={frequency}"
        )
        if not base.exists():
            raise FileNotFoundError(f"factor_matrix 不存在: {base}")

        manifest = _read_manifest(base)
        gen_id = (manifest or {}).get("generation")
        gen_dir = base / "generation" / gen_id if gen_id else None
        if gen_id and (gen_dir is None or not gen_dir.is_dir()):
            # R14 #2 fail-closed：manifest 声明了 generation 但目录缺失 = 当前发布代
            # 损坏。绝不 legacy fallback——那会 glob 到 previous generation / 孤儿
            # generation / legacy 数据，把损坏掩盖成「可读」。
            raise FactorMatrixCorruptionError(
                f"manifest.generation={gen_id!r} 指向的目录缺失（{gen_dir}），当前"
                f"发布代损坏。拒绝读取——修复该代或清除 manifest 后全量重建。"
            )

        is_block = (manifest or {}).get("layout") == "block"
        scan_dir = gen_dir if (gen_dir is not None and gen_dir.is_dir()) else base
        if is_block:
            frames = FactorMatrixMaterializer._frames_from_block(
                scan_dir,
                manifest,
                factor_ids=factor_ids,
                time_range=time_range,
                instrument_filter=instrument_filter,
            )
        else:
            frames = FactorMatrixMaterializer._frames_from_legacy(
                scan_dir,
                manifest,
                factor_ids=factor_ids,
                time_range=time_range,
                instrument_filter=instrument_filter,
            )
        if not frames:
            raise FileNotFoundError(f"factor_matrix 无 parquet: {base}")

        merged = pd.concat(frames, ignore_index=True)
        # P1-16: 重复 (datetime, asset) key = partition 契约损坏，绝不静默 keep="last"
        dup = merged.duplicated(subset=["datetime", "asset"])
        if dup.any():
            sample = merged.loc[dup, ["datetime", "asset"]].head(5).to_dict("records")
            raise DuplicateMatrixKeyError(
                f"factor_matrix 加载后出现 {int(dup.sum())} 个重复 (datetime, asset)"
                f" key（partition 契约已损坏），示例: {sample}"
            )
        if factor_ids is not None:
            keep = ["datetime", "asset", *sorted(set(str(f) for f in factor_ids))]
            missing = [c for c in keep if c not in merged.columns]
            if missing:
                raise ValueError(f"factor_matrix 缺少列: {missing}")
            merged = merged[keep]
        return merged.sort_values(["datetime", "asset"]).reset_index(drop=True)
