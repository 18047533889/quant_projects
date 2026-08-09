"""多因子宽矩阵物化：训练/回测零 pivot 读取路径。

输出布局::

    {matrix_root}/
      universe={universe}/
        freq={freq}/
          year=2025/
            month=01/
              data.parquet   # datetime, asset, factor_a, factor_b, ...
          manifest.json      # P0-23: universe/freq 级 factor_version 绑定 + P1-15 CAS

治理闭环（#收官轮 + R13）：
  * P0-21 read-merge-write 区分「本次没有该 key」与「本次显式 NaN/tombstone」；
  * P0-22 旧分区读失败 → 移动 ``.quarantine/`` 并 hard fail，绝不按空分区覆盖；
  * P0-23 factor_version 绑定：semantic_digest 变化拒绝混列；
  * P1-14 production 走 ``.staging/<part>/`` → validate → manifest(CAS) → publish；
  * P1-15 manifest CAS：并发写同 universe/freq 用版本号检测；
  * P1-16 load 后重复 ``(datetime, asset)`` key 抛错，不再静默 keep="last"。
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from logging_utils import get_logger
from storage.factor_format import series_to_long_table
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


@contextlib.contextmanager
def _manifest_write_lock(base: Path):
    """``manifest.json`` 级写互斥：P1-15 CAS 的串行化点。"""
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / ".manifest.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


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
    def _read_existing_or_quarantine(parquet_path: Path) -> pd.DataFrame:
        """P0-22: 旧分区读失败 → 移动 quarantine + 抛 ``FactorMatrixReadError``。

        绝不把损坏 / IO 抖动 / 权限问题当作空分区继续覆盖——那会把一整月历史
        在增量运行时无声丢掉。原文件先移入 ``<part_dir>/.quarantine/<ts>-data.parquet``
        再 hard fail，数据保留待人工恢复。
        """
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            qdir = parquet_path.parent / ".quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            qpath = qdir / f"{ts}-{uuid.uuid4().hex[:8]}-data.parquet"
            try:
                os.replace(str(parquet_path), str(qpath))
            except Exception:
                qpath = parquet_path
            raise FactorMatrixReadError(
                f"factor_matrix 分区读取失败 {parquet_path}，已隔离至 quarantine="
                f"{qpath}，拒绝继续 publish（历史可能丢失）: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _validate_staging(
        staging_path: Path,
        expected: pd.DataFrame,
    ) -> pd.DataFrame:
        """P1-14: 读回校验 staging（行数 / 列数 / 无全 NaN 列 / 值一致性）。

        校验失败抛 ``FactorMatrixReadError``，staging 不会进入正式位置。
        """
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
    ) -> dict[str, Any]:
        """P0-23 + P1-15: 写 manifest（semantic_digest 校验 + CAS 版本比对）。

        ``factor_versions``（factor_id → semantic_digest/version）缺省为 None 时
        只推进 ``updated_at``/``manifest_version``，不做 digest 绑定。
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
            manifest["updated_at"] = _now_iso()
            manifest["manifest_version"] = current_version + 1
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
    ) -> dict[str, Any]:
        """``factor_id -> Series`` 合并为宽表并按 hive 分区落盘。

        参数:
            results: 见函数签名
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            partition_columns: 见函数签名（可选）
            value_dtype: 见函数签名（可选）
            production: 显式 production 标志；缺省从运行模式解析（可选）。
                production 走 ``.staging/<part>/`` → validate → manifest(CAS) →
                原子 publish；research 保持直写但复用 merge 逻辑。
            recovery: **已弃用**（P0-22）：分区读失败一律 quarantine + hard fail，
                不再支持按空分区破坏性重建；保留参数仅为兼容旧调用方。
            factor_versions: factor_id → semantic_digest/version 绑定（可选，
                P0-23）；缺省时 manifest 只记录本次时间，不做 digest 校验。
            expected_manifest_version: manifest CAS 期望版本（可选，P1-15）；
                缺省为物化开始时读到的当前值。

        返回:
            dict[str, Any]
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

        factor_ids = sorted(results.keys())
        # Phase 5 P1-12：按因子列分块 merge，限制中间 merge 工作集
        # （5000×2500×2000 列的训练矩阵不可能一次成形）。
        try:
            block = max(
                16, int(os.environ.get("FACTOR_ENGINE_MATRIX_COLUMNS_PER_BLOCK", "256"))
            )
        except ValueError:
            block = 256
        merged: pd.DataFrame | None = None
        block_ids: list[str] = []
        block_frame: pd.DataFrame | None = None

        def _flush_block() -> None:
            nonlocal block_frame, merged
            if block_frame is None:
                return
            if merged is None:
                merged = block_frame
            else:
                merged = merged.merge(block_frame, on=["datetime", "asset"], how="outer")
            block_frame = None

        for fid in factor_ids:
            series = results[fid]
            long_df = series_to_long_table(series)
            long_df = long_df.rename(columns={"value": fid})
            block_ids.append(fid)
            if block_frame is None:
                block_frame = long_df
            else:
                block_frame = block_frame.merge(
                    long_df, on=["datetime", "asset"], how="outer"
                )
            long_df = None  # 释放该因子长表引用
            if len(block_ids) >= block:
                _flush_block()
                block_ids = []
        _flush_block()

        if merged is None or merged.empty:
            return {
                "universe": universe,
                "frequency": frequency,
                "rows_written": 0,
                "partitions": [],
                "factor_ids": factor_ids,
                "matrix_root": str(self._matrix_root),
            }

        for fid in factor_ids:
            merged[fid] = merged[fid].astype(str(value_dtype or "float32"))

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
        # P1-15: manifest 版本快照；CAS 期望值缺省 = 读到的当前值
        snapshot_version = self._current_manifest_version(base)
        expected = (
            expected_manifest_version
            if expected_manifest_version is not None
            else snapshot_version
        )
        staged: list[tuple[Path, Path, Path]] = []
        partitions_written: list[str] = []

        try:
            for part_values, partition_df in iter_partition_groups(work, policy):
                drop_cols = [c for c in policy.columns if c in partition_df.columns]
                out_df = partition_df.drop(columns=drop_cols).reset_index(drop=True)
                part_dir = base / partition_path_segments(
                    part_values, column_order=policy.columns
                )
                part_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = part_dir / "data.parquet"
                lock_path = part_dir / ".data.parquet.lock"
                # R11 #7: read-merge-write 在 flock 互斥内完成——并发 matrix 写
                # 同一分区不再互相覆盖；tmp 文件名带 pid+uuid，崩溃残留不命中固定 .tmp。
                with _partition_write_lock(lock_path):
                    if parquet_path.exists():
                        existing = self._read_existing_or_quarantine(parquet_path)
                        merged_out = _merge_matrix_frames(
                            existing, out_df, value_dtype=value_dtype
                        )
                    else:
                        merged_out = out_df
                    if effective_production:
                        # P1-14: production 先写 staging + validate，manifest CAS
                        # 通过后再统一原子 publish。
                        staging_path = part_dir / ".staging" / "data.parquet"
                        self._write_parquet_atomic(staging_path, merged_out)
                        self._validate_staging(staging_path, merged_out)
                        staged.append((part_dir, parquet_path, staging_path))
                    else:
                        self._write_parquet_atomic(parquet_path, merged_out)
                pkey = "|".join(f"{k}={part_values[k]}" for k in sorted(part_values))
                partitions_written.append(pkey)
                logger.info(
                    "factor_matrix 分区 upsert universe=%s partition=%s rows=%d cols=%d",
                    universe,
                    pkey,
                    len(merged_out),
                    len(merged_out.columns),
                )

            # P0-23 + P1-15: manifest（校验 digest + CAS），失败则不发布任何 staging
            manifest = self._update_manifest(
                base,
                universe,
                frequency,
                factor_ids,
                factor_versions,
                expected,
            )
            if effective_production:
                for _part_dir, parquet_path, staging_path in staged:
                    with _partition_write_lock(_part_dir / ".data.parquet.lock"):
                        os.replace(str(staging_path), str(parquet_path))
                    try:
                        staging_path.parent.rmdir()
                    except OSError:
                        pass
            logger.info(
                "factor_matrix manifest updated universe=%s version=%s factors=%d",
                universe,
                manifest.get("manifest_version"),
                len(manifest.get("factors", {})),
            )
        except Exception:
            for _part_dir, _parquet_path, staging_path in staged:
                try:
                    staging_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        return {
            "universe": universe,
            "frequency": frequency,
            "rows_written": len(merged),
            "partitions": partitions_written,
            "factor_ids": factor_ids,
            "matrix_root": str(self._matrix_root),
            "columns": ["datetime", "asset", *factor_ids],
            "manifest_version": manifest.get("manifest_version"),
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
    def load_matrix(
        matrix_root: str | Path,
        *,
        universe: str,
        frequency: str = "1d",
        factor_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读取 universe 下全部或指定因子列宽表。

        参数:
            matrix_root: factor_matrix 根目录
            universe: 标的池标识（可选）
            frequency: 因子频率（可选）
            factor_ids: 因子 ID 列表（可选）

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

        frames: list[pd.DataFrame] = []
        for pq in sorted(base.rglob("data.parquet")):
            if ".staging" in pq.parts or ".quarantine" in pq.parts:
                continue
            if pq.name.startswith("."):
                continue
            df = pd.read_parquet(pq)
            frames.append(df)
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
