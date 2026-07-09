"""
data_access.upsert —— 读-合-写的幂等合并写入

职责：
    1. 对 namespaced/staging 数据集按 upsert_on 键做"后写覆盖先写"的合并
    2. hive 分区场景（partition_by=[...]）按分区列 group 后逐分区合并
    3. 跨进程文件锁，防止同一分区并发 upsert 互相吞数据
    4. 单分区 tmp+rename 原子写，避免半写状态
    5. 审计日志（成功/失败）

非职责：
    - 不做 schema 演进（靠 pyarrow.concat_tables 的 union；列不齐就 raise）
    - 不做业务层校验（值合理性是调用方的事）
    - 不做跨分区事务（每个分区独立原子，跨分区不保证整体原子）

设计参考：
    复刻自 factor_layer/factor_engine/storage/materializer.py::_upsert_partition 的
    语义（[datetime,asset] 双键 / keep=last / tmp+rename / 按 year 分区），
    但通用化：upsert_on 由调用方声明，partition_by 对齐 write_arrow，
    加跨进程文件锁（materializer 现状是无锁的，多 worker 会有 lost update）。

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from . import audit
from .exceptions import DataError, ValidationError
from .namespace import is_namespace_explicit, resolve_namespace
from .paths import PathAuthorizer
from .registry import Dataset


logger = logging.getLogger("data_access.upsert")


def upsert_table(
    *,
    ds: Dataset,
    authorizer: PathAuthorizer,
    target_dir: Path,
    new_table: pa.Table,
    upsert_on: Sequence[str],
    partition_by: Sequence[str] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """对已登记的 namespaced/staging 数据集做 upsert。

    参数：
        ds: registry 里的 Dataset 对象（access_mode 已经由调用方校验过）
        authorizer: 路径白名单（target_dir 必须在内）
        target_dir: 合并写的目标根目录（由 store._resolve_write_dir 算好）
        new_table: 要合入的新数据；列里必须包含 upsert_on + partition_by
        upsert_on: 合并键，按这组列去重；后写覆盖先写
        partition_by: 分区列；为 None 时整表单文件合并
        params: 参数化数据集的参数（只用来写审计日志）

    返回：
        {"rows": int, "path": str, "partitions": list[str], "elapsed_ms": float}
    """
    start = time.perf_counter()

    if not upsert_on:
        raise ValidationError("upsert_on 不能为空；必须明确声明合并键")
    if not isinstance(new_table, pa.Table):
        raise ValidationError(
            f"upsert 只接受 pyarrow.Table，收到 {type(new_table).__name__}"
        )
    if new_table.num_rows == 0:
        raise DataError("upsert 传入的 table 为空；不允许空 upsert（避免无意义的磁盘搅动）")

    _require_columns(new_table, upsert_on, label="upsert_on")
    if partition_by:
        _require_columns(new_table, list(partition_by), label="partition_by")

    authorizer.resolve_and_authorize(str(target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)

    err_msg: str | None = None
    ok = False
    partitions_written: list[str] = []
    rows_written = 0
    elapsed_ms = 0.0

    try:
        if partition_by:
            for part_values, part_table in _split_by_partitions(new_table, partition_by):
                partition_dir = target_dir
                for col, val in zip(partition_by, part_values):
                    partition_dir = partition_dir / f"{col}={val}"
                _upsert_single_dir(
                    partition_dir=partition_dir,
                    new_table=part_table,
                    upsert_on=upsert_on,
                    data_filename="data.parquet",
                )
                partitions_written.append(str(partition_dir))
                rows_written += part_table.num_rows
        else:
            _upsert_single_dir(
                partition_dir=target_dir,
                new_table=new_table,
                upsert_on=upsert_on,
                data_filename="data.parquet",
            )
            partitions_written.append(str(target_dir))
            rows_written = new_table.num_rows
        ok = True
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        audit.record(
            op="upsert",
            dataset=ds.name,
            ok=ok,
            mode="upsert",
            rows=rows_written if ok else new_table.num_rows,
            paths=partitions_written if partitions_written else [str(target_dir)],
            params=params or None,
            elapsed_ms=elapsed_ms,
            error=err_msg,
            extra={
                "upsert_on": list(upsert_on),
                "partition_by": list(partition_by) if partition_by else None,
                "namespace_explicit": is_namespace_explicit(),
                "namespace": resolve_namespace(),
            },
        )

    logger.info(
        "upsert dataset=%s rows=%d partitions=%d elapsed_ms=%.1f",
        ds.name, rows_written, len(partitions_written), elapsed_ms,
    )
    return {
        "rows": rows_written,
        "path": str(target_dir),
        "partitions": partitions_written,
        "elapsed_ms": elapsed_ms,
    }


# ---- 核心：单目录的 read-merge-write 原子合并 --------------------------------


def _upsert_single_dir(
    *,
    partition_dir: Path,
    new_table: pa.Table,
    upsert_on: Sequence[str],
    data_filename: str,
) -> None:
    """
    单个分区目录（或无分区的整个 target_dir）内做 read-merge-write。

    锁：在 partition_dir 的**父目录**下开 O_EXCL 锁文件，防止同 partition_dir
    被并发 upsert（lost update）。选父目录是因为 partition_dir 可能还不存在。
    """
    partition_dir.mkdir(parents=True, exist_ok=True)
    final_path = partition_dir / data_filename

    with _upsert_lock(partition_dir):
        # 1. 读已有数据 + concat；全程走 pandas，省得处理 pyarrow 版本间
        #    concat_tables(promote=...) / (promote_options=...) 的 API 漂移
        #
        # partitioning=None 很关键：pq.read_table 默认会从 hive 风格路径
        # （.../year=2024/data.parquet）反推分区列，把 year 注入回 Table，
        # 导致 existing vs new 的列数对不上。我们写 parquet 时已经把分区列
        # drop 掉了（靠目录名体现），读回来也要同样"无 year"，所以显式关掉。
        new_df = new_table.to_pandas()
        if final_path.exists():
            existing_df = pq.read_table(
                str(final_path), partitioning=None,
            ).to_pandas()
            combined_df = _safe_concat(existing_df, new_df)
        else:
            combined_df = new_df

        # 2. 按 upsert_on 去重；keep="last" 等价于新覆盖旧（new 在 concat 后面）
        merged_df = combined_df.drop_duplicates(
            subset=list(upsert_on), keep="last",
        ).reset_index(drop=True)
        merged = pa.Table.from_pandas(merged_df, preserve_index=False)

        # 3. tmp + rename 原子落盘
        tmp_path = partition_dir / f".{data_filename}.tmp.{uuid.uuid4().hex[:8]}"
        pq.write_table(merged, str(tmp_path))
        os.replace(str(tmp_path), str(final_path))


def _safe_concat(existing_df, new_df):
    """pandas concat，防止列序/缺列时的诡异行为。

    缺列直接 raise：upsert 的语义前提是 new + existing 的 schema 一致。
    若调用方真的想演进 schema，走 overwrite（write_arrow）而不是 upsert。
    """
    import pandas as pd

    missing_in_new = set(existing_df.columns) - set(new_df.columns)
    missing_in_existing = set(new_df.columns) - set(existing_df.columns)
    if missing_in_new or missing_in_existing:
        raise ValidationError(
            f"upsert 的 new 和已有数据 schema 不一致；"
            f"new 缺列 {sorted(missing_in_new)}，existing 缺列 {sorted(missing_in_existing)}。"
            f"schema 演进请走 overwrite 而非 upsert。"
        )
    # 对齐列序（取 existing 的列序），防止 concat 后列顺序乱
    new_df = new_df[list(existing_df.columns)]
    return pd.concat([existing_df, new_df], ignore_index=True)


def _require_columns(table: pa.Table, cols: Sequence[str], *, label: str) -> None:
    missing = [c for c in cols if c not in table.column_names]
    if missing:
        raise ValidationError(
            f"upsert 的 {label} 声明了 {list(cols)}，但 table 里缺列 {missing}；"
            f"实际列：{table.column_names}"
        )


def _split_by_partitions(
    table: pa.Table, partition_by: Sequence[str],
) -> Iterator[tuple[tuple[Any, ...], pa.Table]]:
    """
    按 partition_by 列把 table 拆成 (part_values, part_table) 序列。

    走 pandas.groupby —— 比自己拼 pyarrow.compute mask 稳、好读；
    upsert 批次一般不大（一次 flush 几百 ~ 几万行），性能差距可忽略。
    """
    df = table.to_pandas()
    part_cols = list(partition_by)
    for part_values, group_df in df.groupby(part_cols, sort=False):
        # 单列 groupby 给的是标量，多列是 tuple，统一成 tuple
        if not isinstance(part_values, tuple):
            part_values = (part_values,)
        # 丢掉分区列本身——它已经体现在目录名里，不用再存一份
        payload_df = group_df.drop(columns=part_cols).reset_index(drop=True)
        yield part_values, pa.Table.from_pandas(payload_df, preserve_index=False)


# ---- 并发锁 -----------------------------------------------------------------


@contextmanager
def _upsert_lock(partition_dir: Path) -> Iterator[None]:
    """O_EXCL 文件锁，fail-fast；锁放在 partition_dir 的父目录下。

    和 publish 的锁不同：upsert 会频繁发生（streaming 场景每几秒一次），
    直接 raise 太吵；这里改成**短暂轮询**（最多 5s），轮询失败再 raise。
    """
    lock_parent = partition_dir.parent
    lock_parent.mkdir(parents=True, exist_ok=True)
    lock_name = f".upsert.{partition_dir.name}.lock"
    lock_path = lock_parent / lock_name

    deadline = time.monotonic() + 5.0
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise ValidationError(
                    f"upsert 并发冲突：锁文件 {lock_path} 已存在超过 5s。"
                    f"若确认是遗留文件请手工删除；否则说明另一 writer 卡住了"
                )
            time.sleep(0.05)

    try:
        os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode())
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("删除 upsert 锁文件 %s 失败：%s", lock_path, exc)


def delete_rows_from_dataset(
    *,
    ds: Dataset,
    authorizer: PathAuthorizer,
    target_dir: Path,
    time_column: str,
    start: Any | None = None,
    end: Any | None = None,
    after: Any | None = None,
    params: dict[str, Any] | None = None,
    dry_run: bool = False,
    max_rows: int | None = None,
    reason: str | None = None,
    ticket_id: str | None = None,
) -> dict[str, Any]:
    """从 staging/namespaced hive 分区删除指定时间范围内的行。"""
    import pandas as pd

    from .exceptions import ValidationError

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    after_ts = pd.Timestamp(after) if after is not None else None

    authorizer.resolve_and_authorize(str(target_dir))
    if not target_dir.exists():
        return {
            "rows_deleted": 0,
            "partitions": [],
            "elapsed_ms": 0.0,
            "dry_run": dry_run,
        }

    rows_deleted = 0
    partitions: list[str] = []
    t0 = time.perf_counter()

    for parquet_path in sorted(target_dir.rglob("*.parquet")):
        try:
            df = pq.read_table(str(parquet_path), partitioning=None).to_pandas()
        except Exception as exc:
            logger.warning("delete_rows 跳过 %s: %s", parquet_path, exc)
            continue
        if df.empty or time_column not in df.columns:
            continue

        dt = pd.to_datetime(df[time_column])
        if after_ts is not None and start_ts is None and end_ts is None:
            delete_mask = dt > after_ts
        else:
            delete_mask = pd.Series(True, index=df.index)
            if start_ts is not None:
                delete_mask &= dt >= start_ts
            if end_ts is not None:
                delete_mask &= dt <= end_ts
            if after_ts is not None:
                delete_mask &= dt > after_ts

        removed = int(delete_mask.sum())
        if removed <= 0:
            continue

        if max_rows is not None and rows_deleted + removed > max_rows:
            raise ValidationError(
                f"delete_rows 将删除 {rows_deleted + removed} 行，超过 max_rows={max_rows}。"
                "请缩小时间范围或提高 max_rows。"
            )

        rows_deleted += removed
        partitions.append(str(parquet_path.parent))

        if dry_run:
            continue

        kept = df.loc[~delete_mask]

        if kept.empty:
            parquet_path.unlink(missing_ok=True)
            continue

        with _upsert_lock(parquet_path.parent):
            tmp_path = parquet_path.parent / f".delete.tmp.{uuid.uuid4().hex[:8]}.parquet"
            pq.write_table(pa.Table.from_pandas(kept, preserve_index=False), str(tmp_path))
            os.replace(str(tmp_path), str(parquet_path))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    audit_extra: dict[str, Any] = {"time_column": time_column}
    if reason:
        audit_extra["reason"] = reason[:500]
    if ticket_id:
        audit_extra["ticket_id"] = ticket_id
    if dry_run:
        audit_extra["dry_run"] = True

    audit.record(
        op="delete_rows",
        dataset=ds.name,
        ok=True,
        mode="delete",
        rows=rows_deleted,
        paths=partitions[:20],
        params=params or None,
        elapsed_ms=elapsed_ms,
        extra=audit_extra,
    )
    return {
        "rows_deleted": rows_deleted,
        "partitions": partitions,
        "elapsed_ms": elapsed_ms,
        "dry_run": dry_run,
    }
