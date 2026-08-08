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

from data_access.core import audit
from data_access.core.exceptions import DataError, ValidationError
from data_access.core.namespace import is_namespace_explicit, resolve_namespace
from data_access.registry.paths import PathAuthorizer
from data_access.registry import Dataset
from data_access.write.mutation_lock import mutation_lock


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

    # #P0-34 new_table 自身必须对 upsert_on 唯一：重复 key 的 tie-break 只依赖
    # __src 且两条 new rows 优先级相同，最终选哪条取决于执行顺序——拒绝。
    _assert_unique_keys(new_table, list(upsert_on), label="upsert new_table")

    target_dir = _authorize_write_path(target_dir, authorizer)
    target_dir.mkdir(parents=True, exist_ok=True)

    err_msg: str | None = None
    ok = False
    partitions_written: list[str] = []
    staged: list[tuple[Path, Path]] = []  # (final_path, tmp_path) —— 全部 merge 成功后才 promote
    rows_written = 0
    elapsed_ms = 0.0
    transaction_id = uuid.uuid4().hex[:12]

    try:
        with mutation_lock(target_dir):
            # ---- #P0-28 Phase 1（prepare）：逐分区 read-merge-write 到 staging tmp。
            # 任何分区 merge 失败（schema 不一致 / DuckDB 错）→ 全部不 promote，
            # 已存在的分区文件保持原样，不会「前几个分区已提交」。
            if partition_by:
                # #P0-35 upsert_on ∩ partition_by 重叠时，partition key 已由目录固定，
                # 局部合并键 = upsert_on - partition_by（否则引用已被 drop 的列）。
                local_merge_key = [c for c in upsert_on if c not in partition_by]
                if not local_merge_key:
                    raise ValidationError(
                        "upsert_on 完全被 partition_by 覆盖：合并键为空，无法去重。"
                        "请去掉 upsert_on 中已被分区固定的列，或减少 partition_by。"
                    )
                for part_values, part_table in _split_by_partitions(new_table, partition_by):
                    partition_dir = target_dir
                    for col, val in zip(partition_by, part_values):
                        _validate_partition_component(col, val)
                        partition_dir = partition_dir / f"{col}={val}"
                    partition_dir = _authorize_write_path(
                        partition_dir, authorizer, expected_root=target_dir
                    )
                    final_path, tmp_path = _stage_upsert(
                        partition_dir=partition_dir,
                        new_table=part_table,
                        upsert_on=local_merge_key,
                        data_filename="data.parquet",
                    )
                    staged.append((final_path, tmp_path))
                    rows_written += part_table.num_rows
            else:
                final_path, tmp_path = _stage_upsert(
                    partition_dir=target_dir,
                    new_table=new_table,
                    upsert_on=upsert_on,
                    data_filename="data.parquet",
                )
                staged.append((final_path, tmp_path))
                rows_written = new_table.num_rows
            # ---- Phase 2（promote）：全部 merge 成功后统一原子晋级。
            # os.replace 单文件原子；中途失败会留下可检测的部分状态
            # （.transactions.jsonl status=failed + manifest 保持 dirty）。
            for final_path, tmp_path in staged:
                os.replace(str(tmp_path), str(final_path))
                partitions_written.append(str(final_path.parent))
            ok = True
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        for _, tmp_path in staged:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        # #42 / #P0-28 TransactionManifest：全部分区成功才记 committed；任一失败
        # 记 failed（含 staged partitions）。供 audit / lineage / watermark 判定。
        _record_transaction(
            target_dir,
            transaction_id=transaction_id,
            partitions=partitions_written,
            staged_partitions=[str(fp.parent) for fp, _ in staged],
            rows=rows_written,
            ok=ok,
        )
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


def _stage_upsert(
    *,
    partition_dir: Path,
    new_table: pa.Table,
    upsert_on: Sequence[str],
    data_filename: str,
) -> tuple[Path, Path]:
    """
    #P0-28 单目录 read-merge-write 的 **prepare 阶段**：把合并结果写到 staging
    tmp 文件，**不落最终路径**。调用方在所有分区 merge 成功后统一 promote
    （os.replace），保证「前几个分区已提交、后面失败」不发生。

    锁：在 partition_dir 的**父目录**下开 O_EXCL 锁文件，防止同 partition_dir
    被并发 upsert（lost update）。选父目录是因为 partition_dir 可能还不存在。

    **#43**：不再全量 pandas merge——DuckDB COW（UNION ALL + row_number 去重，
    new 优先覆盖 old）。schema 不一致仍报错（schema 演进请走 overwrite）。

    返回 ``(final_path, tmp_path)``。
    """
    partition_dir.mkdir(parents=True, exist_ok=True)
    final_path = partition_dir / data_filename

    with _upsert_lock(partition_dir):
        # 1. 读已有数据（partitioning=None：hive 分区列在目录名里，不注入回表）
        existing = None
        if final_path.exists():
            existing = pq.read_table(str(final_path), partitioning=None)

        # 2. DuckDB COW：new 先、existing 后 → 每 key 保留第一条（= new 覆盖 old）
        if existing is not None:
            merged = _duckdb_merge(existing, new_table, upsert_on)
        else:
            merged = new_table

        # 3. 只写 staging tmp，不 replace 到 final（promote 由调用方统一做）
        tmp_path = partition_dir / f".{data_filename}.staging.{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(merged, str(tmp_path))
    return final_path, tmp_path


def _duckdb_merge(
    existing: pa.Table,
    new_table: pa.Table,
    upsert_on: Sequence[str],
) -> pa.Table:
    """#43 用 DuckDB 做 read-merge-write：new 覆盖 old，不经过 pandas。

    列不一致直接报错（upsert 语义前提是 schema 一致；schema 演进走 overwrite）。
    返回合并后的 Arrow Table。
    """
    import duckdb

    missing_in_new = set(existing.column_names) - set(new_table.column_names)
    missing_in_existing = set(new_table.column_names) - set(existing.column_names)
    if missing_in_new or missing_in_existing:
        raise ValidationError(
            f"upsert 的 new 和已有数据 schema 不一致；"
            f"new 缺列 {sorted(missing_in_new)}，existing 缺列 {sorted(missing_in_existing)}。"
            f"schema 演进请走 overwrite 而非 upsert。"
        )
    key_list = ", ".join(f'"{c}"' for c in upsert_on)
    con = duckdb.connect()
    try:
        con.register("__existing", existing)
        con.register("__new", new_table)
        sql = (
            f"SELECT * EXCLUDE (__src, __rn) FROM ("
            f"  SELECT *, row_number() OVER (PARTITION BY {key_list} "
            f"    ORDER BY __src) AS __rn "
            f"  FROM (SELECT 0 AS __src, * FROM __new "
            f"        UNION ALL "
            f"        SELECT 1 AS __src, * FROM __existing)"
            f") WHERE __rn = 1"
        )
        result = con.execute(sql)
        if hasattr(result, "to_arrow_table"):
            table = result.to_arrow_table()
        else:  # 老版本 DuckDB
            table = result.fetch_arrow_table()
        return table if isinstance(table, pa.Table) else pa.Table.from_batches(table)
    finally:
        con.close()


def _authorize_write_path(
    path: Path,
    authorizer: PathAuthorizer,
    *,
    expected_root: Path | None = None,
) -> Path:
    """Authorize a write path and reject symlink components."""
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    raw_root = Path(expected_root).expanduser() if expected_root is not None else None
    current_raw = raw_path
    while True:
        if current_raw.is_symlink():
            raise ValidationError(f"upsert 不允许通过软链接写入：{current_raw}")
        if raw_root is not None and current_raw == raw_root:
            break
        if current_raw.parent == current_raw:
            break
        current_raw = current_raw.parent

    resolved = authorizer.resolve_and_authorize(raw_path)
    root = authorizer.resolve_and_authorize(expected_root) if expected_root is not None else None
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidationError(
                f"upsert 分区路径越界：{resolved} 不在目标目录 {root} 下"
            ) from exc
    current = resolved
    while True:
        if current.is_symlink():
            raise ValidationError(f"upsert 不允许通过软链接写入：{current}")
        if root is not None and current == root:
            break
        if current.parent == current:
            break
        current = current.parent
    return resolved


def _validate_partition_component(column: Any, value: Any) -> None:
    """Reject partition components that can alter filesystem path semantics."""
    text = str(value)
    if not isinstance(column, str) or not column or column in {".", ".."}:
        raise ValidationError(f"upsert 分区列名非法：{column!r}")
    if any(ch in text for ch in ("/", "\\", "\x00", "\n", "\r")):
        raise ValidationError(f"upsert 分区值包含非法路径字符：{value!r}")
    if text in {"", ".", ".."} or Path(text).is_absolute():
        raise ValidationError(f"upsert 分区值非法：{value!r}")

def _record_transaction(
    target_dir: Path,
    *,
    transaction_id: str,
    partitions: list[str],
    rows: int,
    ok: bool,
    staged_partitions: list[str] | None = None,
) -> None:
    """#42 追加一条 TransactionManifest 记录（``{root}/.transactions.jsonl``）。

    ``committed`` 表示全部分区写入成功；``failed`` 表示部分/全部失败（此时
    dataset 级 watermark / snapshot 不应前进）。``staged_partitions``（#P0-28）
    记录 prepare 阶段完成、但 promote 未完成的候选分区，供修复/审计定位。
    """
    import json as _json

    try:
        log_path = target_dir / ".transactions.jsonl"
        record = {
            "transaction_id": transaction_id,
            "status": "committed" if ok else "failed",
            "partitions": partitions,
            "rows": rows,
            "affected_partition_count": len(partitions),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if staged_partitions is not None:
            record["staged_partitions"] = staged_partitions
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("upsert 写 TransactionManifest 失败：%s", exc)


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


def _assert_unique_keys(table: pa.Table, cols: Sequence[str], *, label: str) -> None:
    """#P0-34 断言 table 对 cols 唯一；重复 → ValidationError。"""
    if not cols:
        return
    import duckdb

    key_list = ", ".join(f'"{c}"' for c in cols)
    con = duckdb.connect()
    try:
        con.register("__u", table)
        row = con.execute(
            f"SELECT COUNT(*) FROM ("
            f"SELECT {key_list} FROM __u GROUP BY {key_list} HAVING COUNT(*) > 1"
            f")"
        ).fetchone()
        dup_groups = int(row[0]) if row else 0
    except Exception as exc:
        raise ValidationError(
            f"{label} 无法校验 {cols} 唯一性（DuckDB 失败: {exc}）"
        ) from exc
    finally:
        con.close()
    if dup_groups:
        raise ValidationError(
            f"{label} 在合并键 {list(cols)} 上存在 {dup_groups} 组重复 key；"
            "upsert 语义要求 new_table 对 upsert_on 唯一，不允许依赖输入/执行顺序"
            "的 tie-break。业务确实允许多修订时请显式声明 revision_order。"
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
    best_effort: bool = False,
) -> dict[str, Any]:
    """从 staging/namespaced hive 分区删除指定时间范围内的行。

    #P0-37 **two-phase preflight**：
        PHASE 1 —— 扫描每个 candidate 文件的 footer + 谓词，构建完整 delete plan
        （path → kept rows），累计行数并先验证 ``max_rows``（超限 → 抛错，**0 行
        实际删除**）；
        PHASE 2 —— plan 全部通过后统一 commit（删除/重写）。

    #P0-36 任意 candidate 文件读取失败：production fail-closed（整个 delete abort）；
    ``best_effort=True``（研究/手动）才允许跳过，并返回 ``failed_files`` /
    ``remaining_unverified_rows``。
    """
    import pandas as pd

    from data_access.core.exceptions import ValidationError

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

    t0 = time.perf_counter()
    # ---- PHASE 1：构建 delete plan（不落盘、不删） ----
    plan: list[tuple[Path, pd.DataFrame]] = []  # (parquet_path, kept_rows)
    failed_files: list[str] = []
    rows_deleted = 0
    partitions: list[str] = []
    for parquet_path in sorted(target_dir.rglob("*.parquet")):
        try:
            df = pq.read_table(str(parquet_path), partitioning=None).to_pandas()
        except Exception as exc:
            if not best_effort:
                raise DataError(
                    f"delete_rows 读取 {parquet_path} 失败: {exc}；"
                    "production fail-closed：整个 delete 中止，未删除任何行。"
                ) from exc
            logger.warning("delete_rows best_effort 跳过 %s: %s", parquet_path, exc)
            failed_files.append(str(parquet_path))
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
        # max_rows 是 safety guard：超限必须在**删除任何行之前**失败（#P0-37）。
        if max_rows is not None and rows_deleted + removed > max_rows:
            raise ValidationError(
                f"delete_rows 将删除 {rows_deleted + removed} 行，超过 max_rows={max_rows}。"
                "已 abort（0 行实际删除）。请缩小时间范围或提高 max_rows。"
            )
        rows_deleted += removed
        partitions.append(str(parquet_path.parent))
        kept = df.loc[~delete_mask]
        plan.append((parquet_path, kept))

    remaining_unverified = rows_deleted  # best_effort 下跳过文件的行数未知
    # ---- PHASE 2：commit plan ----
    if dry_run:
        committed = 0
    else:
        for parquet_path, kept in plan:
            if kept.empty:
                parquet_path.unlink(missing_ok=True)
                continue
            with _upsert_lock(parquet_path.parent):
                tmp_path = parquet_path.parent / f".delete.tmp.{uuid.uuid4().hex[:8]}.parquet"
                pq.write_table(pa.Table.from_pandas(kept, preserve_index=False), str(tmp_path))
                os.replace(str(tmp_path), str(parquet_path))
        committed = rows_deleted

    elapsed_ms = (time.perf_counter() - t0) * 1000
    audit_extra: dict[str, Any] = {"time_column": time_column}
    if reason:
        audit_extra["reason"] = reason[:500]
    if ticket_id:
        audit_extra["ticket_id"] = ticket_id
    if dry_run:
        audit_extra["dry_run"] = True
    if best_effort:
        audit_extra["best_effort"] = True
    if failed_files:
        audit_extra["failed_files"] = failed_files[:20]

    audit.record(
        op="delete_rows",
        dataset=ds.name,
        ok=not failed_files or best_effort,
        mode="delete",
        rows=rows_deleted,
        paths=partitions[:20],
        params=params or None,
        elapsed_ms=elapsed_ms,
        extra=audit_extra,
    )
    result: dict[str, Any] = {
        "rows_deleted": committed if not dry_run else rows_deleted,
        "partitions": partitions,
        "elapsed_ms": elapsed_ms,
        "dry_run": dry_run,
    }
    if failed_files:
        result["failed_files"] = failed_files
        result["remaining_unverified_rows"] = remaining_unverified
    return result
