"""
data_access.publish —— staging → published 的原子发布

职责：
    1. 校验 staging 和 target 数据集的配对合法性（access_mode、schema、params_schema）
    2. 把 staging 目录的 parquet 内容原子晋升为 published 版本
    3. 保留旧 published（归档到 _archive/），便于回滚和审计
    4. 失败时回滚，保证 published 目录处于一致状态
    5. 审计日志：publish 开始 / 成功 / 失败

非职责：
    - 不做数据校验（因子值是否合理、分布是否符合预期）—— 那是业务层的事
    - 不做审批门（谁能发布、什么情况下能发布）—— 走人工流程 + operator 审计
    - 不做自动归档清理 —— 手工运维清，PR5 会加 TTL 脚本

设计要点：
    - 跨文件系统兼容：staging 和 published 可能在不同 FS（workspace vs 共享盘），
      先 copytree 到 published 父目录下的候选目录（同 FS），再用 rename 做原子切换。
    - 三段式原子切换（全部 rename，同 FS 保证原子）：
        1. rename old_published → _archive/...       （旧版本归档）
        2. rename candidate → published              （新版本上线）
        3. 失败时各自回滚
    - 并发锁：published 父目录下 .publish.<name>.lock 文件 O_EXCL 创建，
      冲突直接 raise 不等待（发布是低频人为动作，撞了说明操作协调出问题）

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.core import audit
from data_access.core.exceptions import DataError, ValidationError
from data_access.core.namespace import is_namespace_explicit, resolve_namespace
from data_access.registry.paths import PathAuthorizer
from .publish_manifest import write_publish_manifest
from data_access.registry import Dataset, DatasetRegistry, ParametricDataset, StaticDataset
from data_access.write.mutation_lock import mutation_lock


logger = logging.getLogger("data_access.publish")


# 归档保留策略：不自动清。_archive/ 下按时间排序，运维脚本按 TTL 清。
_ARCHIVE_SUBDIR = "_archive"


def publish_from_staging(
    registry: DatasetRegistry,
    authorizer: PathAuthorizer,
    staging_name: str,
    target_name: str,
    **params: Any,
) -> dict[str, Any]:
    """把 staging 数据集当前内容晋升为 target（published）数据集。

    参数：
        registry: DatasetRegistry（由 store 注入）
        authorizer: PathAuthorizer（由 store 注入）
        staging_name: staging 数据集注册名，access_mode 必须是 staging
        target_name: 目标数据集注册名，access_mode 必须是 published
        **params: 参数化数据集的参数（两边必须接受同一组参数）

    返回：
        {"source": {...}, "target_path": str, "archive_path": str | None,
         "rows": int, "elapsed_ms": float}

    raise：
        ValidationError: access_mode 不匹配 / schema 不兼容 / 并发锁冲突
        DataError: staging 目录不存在或为空
        OSError: 拷贝 / rename 失败（尽量回滚到一致状态后再抛）
    """
    start = time.perf_counter()
    # 先把 audit 用得上的字段预置成「未知」。这里尽量别抛——抛了连目标名都拿不到，
    # 审计就没法记。登记表缺 key 之类的异常仍然保留原样（那种错不配进审计）。
    staging_ds = registry.get(staging_name)
    target_ds = registry.get(target_name)

    source_rows = 0
    staging_dir: Path | None = None
    target_dir: Path | None = None
    archive_path: Path | None = None
    candidate_dir: Path | None = None
    err_msg: str | None = None
    ok = False
    elapsed_ms = 0.0

    try:
        _validate_publish_pair(staging_ds, target_ds, params)

        staging_dir = _resolve_dataset_dir(staging_ds, params)
        target_dir = _resolve_dataset_dir(target_ds, params)

        authorizer.resolve_and_authorize(str(staging_dir))
        authorizer.resolve_and_authorize(str(target_dir))

        # #P0-33 单一 inventory snapshot：行数/字节/schema 全部来自同一次
        # _file_inventory（不再先 _count_parquet_rows 再 inventory——二者之间
        # staging 可能变化，source_rows 会来自上一版本）。
        source_inv = _file_inventory(staging_dir)
        if source_inv is None or not source_inv:
            raise DataError(
                f"staging 数据集 '{staging_name}' 在 {staging_dir} 没有可发布内容；"
                f"是不是没写 / 参数 {params} 不对？"
            )
        source_rows = sum(rows for rows, _bytes, _hash in source_inv.values())

        target_parent = target_dir.parent
        target_parent.mkdir(parents=True, exist_ok=True)
        candidate_dir = target_parent / f".publish_candidate.{uuid.uuid4().hex[:12]}"

        with mutation_lock(target_dir.parent):
            with _publish_lock(target_parent, target_dir.name):
                # 第 1 步：copy staging → candidate（跨 FS 兼容的慢路径）
                _copy_tree(staging_dir, candidate_dir)

                # #40 校验 source 在 copy 期间未被并发修改（文件清单不变）
                if _file_inventory(staging_dir) != source_inv:
                    shutil.rmtree(candidate_dir, ignore_errors=True)
                    raise DataError(
                        f"staging 数据集 '{staging_name}' 在发布 copy 期间被并发写入，"
                        f"拒绝发布（避免 mixed generation）。请重试。"
                    )

                # 第 2 步：校验 candidate 与 staging 完全一致（#41 不止 row count：
                # 文件清单 + 行数 + schema hash + 分区清单）
                candidate_inv = _file_inventory(candidate_dir)
                if source_inv is None or candidate_inv != source_inv:
                    raise DataError(
                        f"candidate 与 staging 文件清单/行数/schema 不一致；"
                        f"拷贝可能中断或 staging 被并发修改，拒绝发布"
                    )

                # #P0-30 key 唯一性发布 gate：契约声明 unique_key 时必须无重复。
                _validate_unique_key(target_ds, candidate_dir)

                # #P0-3 candidate 实际数据必须符合 target 声明 schema 契约。
                # ``_file_inventory`` 证明的是 "candidate == staging"（两边 hash
                # 一致）；这里是 "candidate == target contract"——target 声明列
                # 缺失 / 类型不符都拒绝发布，禁止靠 DuckDB 隐式 cast 蒙混过关。
                _validate_candidate_contract(target_ds, candidate_dir)

                # #P0-32 第 3 步：publish manifest 属于 commit metadata，必须**在
                # 原子切换之前**生成并写进 candidate——切换失败/写失败都 abort，
                # 绝不允许「新 target 已上线但写 manifest 失败 → caller 抛异常、
                # 却留下已发布版本」的不一致状态。
                # #P0-C13 只有「真有上一代发布内容」才算旧版本要归档。``_dataset_mutation``
                # 的事务锁（mutation_lock）会 ``mkdir`` 出空 target 目录，旧代码
                # ``if target_dir.exists()`` 把这种空目录也当旧版本归档——首次发布
                # 也会生成 archive_path，产生垃圾归档目录。
                if _target_has_published_content(target_dir):
                    archive_path = _archive_path(target_parent, target_dir.name)
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path = write_publish_manifest(
                    candidate_dir,
                    staging_name=staging_name,
                    target_name=target_name,
                    params=params,
                    rows=source_rows,
                    archive_path=archive_path,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )

                # 第 4 步：同 FS 原子 rename —— old_published → archive，candidate → final
                # （manifest 已随 candidate 一起原子切换上线）
                if _target_has_published_content(target_dir):
                    os.rename(str(target_dir), str(archive_path))
                elif target_dir.exists():
                    # #P0-C13 mutation 事务锁 mkdir 出的空 target（仅含锁文件，无
                    # 发布内容）：先清掉，否则 candidate rename 会撞「目录非空」。
                    # mutation_lock 的 finally 对缺失锁文件容错（FileNotFoundError）。
                    shutil.rmtree(str(target_dir), ignore_errors=True)

                try:
                    os.rename(str(candidate_dir), str(target_dir))
                except OSError:
                    if archive_path and archive_path.exists():
                        os.rename(str(archive_path), str(target_dir))
                        archive_path = None
                    raise

                try:
                    post_inv = _file_inventory(target_dir)
                except Exception as exc:
                    _rollback_final(target_dir, archive_path)
                    raise DataError(
                        f"publish 后读 {target_dir} 失败：{exc}；已回滚到归档版本"
                    ) from exc
                if post_inv is None or post_inv != source_inv:
                    _rollback_final(target_dir, archive_path)
                    raise DataError(
                        f"publish 后文件清单与 staging 不一致（行数/文件/schema）；已回滚"
                    )
                # 切换后验证 publish manifest 确实随数据一起上线。
                if not (target_dir / ".publish_manifest.json").exists():
                    _rollback_final(target_dir, archive_path)
                    raise DataError(
                        f"publish 后 {target_dir}/.publish_manifest.json 缺失；"
                        "commit metadata 未随原子切换上线，已回滚"
                    )

                ok = True
                candidate_dir = None
                logger.info(
                    "publish manifest committed with target: %s",
                    target_dir / ".publish_manifest.json",
                )
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # 清理 candidate（如果还存在，说明 rename 前失败）
        if candidate_dir and candidate_dir.exists():  #当 candidate_dir 存在时，就把他删掉，代表变成 published失败，所以可以清空临时路径。
            #因为正常运行 candidatedir 会变成 published 路径，不会留着临时路径
            shutil.rmtree(candidate_dir, ignore_errors=True)

        elapsed_ms = (time.perf_counter() - start) * 1000
        audit.record(
            op="publish",
            dataset=target_name,
            ok=ok,
            rows=source_rows,
            # target_dir 在早期校验阶段失败时可能还没解析出来；不记就是不记
            paths=[str(target_dir)] if target_dir is not None else None,
            params=params or None,
            elapsed_ms=elapsed_ms,
            error=err_msg,
            # #P1-final closure 22：发布是权威动作——审计必须 durable 落盘（flush
            # + fsync）。业务成功但审计写失败 ⇒ AuditWriteError 向上抛，不允许
            # 「发布了、审计悄悄没记」。
            durable=True,
            extra={
                "source": staging_name,
                "archive_path": str(archive_path) if archive_path else None,
                "namespace_explicit": is_namespace_explicit(),
                "namespace": resolve_namespace(),
            },
        )

    logger.info(
        "publish staging=%s → target=%s rows=%d archive=%s elapsed_ms=%.1f",
        staging_name, target_name, source_rows,
        archive_path if archive_path else "(none)",
        elapsed_ms,
    )
    return {
        "source": {"dataset": staging_name, "path": str(staging_dir)},
        "target_path": str(target_dir),
        "archive_path": str(archive_path) if archive_path else None,
        "rows": source_rows,
        "elapsed_ms": elapsed_ms,
        "manifest_path": str(target_dir / ".publish_manifest.json"),
    }


# ---- 校验 ------------------------------------------------------------------


def _validate_publish_pair(
    staging_ds: Dataset,
    target_ds: Dataset,
    params: dict[str, Any],
) -> None:
    """校验 staging + target 是一对合法的发布配对。

    规则：
        - staging_ds.access_mode 必须是 "staging"
        - target_ds.access_mode 必须是 "published"
        - time_column / instrument_column 一致（语义对齐，避免发错库）
        - 如果是 parametric，params_schema 必须一致（同一 factor_id 对应得上）
        - layout / hive_partitioning 一致（避免读取侧行为不一致）
    """
    if staging_ds.access_mode != "staging":
        raise ValidationError(
            f"源数据集 '{staging_ds.name}' access_mode={staging_ds.access_mode!r}，"
            f"publish_from_staging 只接受 staging 来源"
        )
    if target_ds.access_mode != "published":
        raise ValidationError(
            f"目标数据集 '{target_ds.name}' access_mode={target_ds.access_mode!r}，"
            f"publish_from_staging 只能晋升到 published"
        )

    if staging_ds.time_column != target_ds.time_column:
        raise ValidationError(
            f"time_column 不一致：staging={staging_ds.time_column!r} "
            f"vs target={target_ds.time_column!r}，拒绝发布"
        )
    if staging_ds.instrument_column != target_ds.instrument_column:
        raise ValidationError(
            f"instrument_column 不一致：staging={staging_ds.instrument_column!r} "
            f"vs target={target_ds.instrument_column!r}，拒绝发布"
        )
    if staging_ds.layout != target_ds.layout:
        raise ValidationError(
            f"layout 不一致：staging={staging_ds.layout!r} "
            f"vs target={target_ds.layout!r}，拒绝发布"
        )
    if staging_ds.hive_partitioning != target_ds.hive_partitioning:
        raise ValidationError(
            f"hive_partitioning 不一致：staging={staging_ds.hive_partitioning} "
            f"vs target={target_ds.hive_partitioning}，拒绝发布"
        )

    # #P0-31 staging 与 target 契约对齐：schema_version / 声明 schema /
    # partition_columns / 物理格式。仅校验「双方都显式声明」的字段——空声明
    # （未启用 schema 校验）不代表契约冲突。
    if staging_ds.schema_version != target_ds.schema_version:
        raise ValidationError(
            f"schema_version 不一致：staging={staging_ds.schema_version!r} "
            f"vs target={target_ds.schema_version!r}，拒绝发布"
        )
    _staging_schema = dict(getattr(staging_ds, "schema", None) or {})
    _target_schema = dict(getattr(target_ds, "schema", None) or {})
    if _staging_schema and _target_schema and _staging_schema != _target_schema:
        raise ValidationError(
            f"声明 schema 不一致：staging={_staging_schema} "
            f"vs target={_target_schema}，拒绝发布（发布的 schema 必须与 target 契约一致）"
        )
    if tuple(getattr(staging_ds, "partition_columns", ()) or ()) != tuple(
        getattr(target_ds, "partition_columns", ()) or ()
    ):
        raise ValidationError(
            f"partition_columns 不一致：staging={getattr(staging_ds, 'partition_columns', ())} "
            f"vs target={getattr(target_ds, 'partition_columns', ())}，拒绝发布"
        )
    _sfmt = str(getattr(staging_ds, "format", "parquet") or "parquet")
    _tfmt = str(getattr(target_ds, "format", "parquet") or "parquet")
    if _sfmt != _tfmt:
        raise ValidationError(
            f"物理格式不一致：staging={_sfmt} vs target={_tfmt}，拒绝发布"
        )

    # 参数化数据集：两边参数 schema 必须一致，调用方提供的 params 也要覆盖
    staging_params = _params_schema(staging_ds)
    target_params = _params_schema(target_ds)
    if staging_params != target_params:
        raise ValidationError(
            f"params_schema 不一致：staging={dict(staging_params)} "
            f"vs target={dict(target_params)}，拒绝发布"
        )
    if staging_params:
        missing = [k for k in staging_params if k not in params]
        if missing:
            raise ValidationError(
                f"publish_from_staging 缺参数 {missing}；"
                f"参数化数据集必须两边用同一组参数定位"
            )


def _params_schema(ds: Dataset) -> dict[str, str]:
    if isinstance(ds, ParametricDataset):
        return dict(ds.params_schema)
    return {}


# ---- 路径 / 行数辅助 ---------------------------------------------------------


def _resolve_dataset_dir(ds: Dataset, params: dict[str, Any]) -> Path:
    """跟 store._resolve_write_dir 语义一致：取 glob 的静态前缀段作为数据目录根。"""
    glob_paths = ds.resolve_paths(**params)
    if len(glob_paths) != 1:
        raise ValidationError(
            f"数据集 '{ds.name}' 解析出 {len(glob_paths)} 个 glob，"
            f"publish 只支持唯一目标目录的数据集"
        )
    segments = glob_paths[0].split("/")
    clean: list[str] = []
    for seg in segments:
        if "*" in seg or "?" in seg:
            break
        clean.append(seg)
    if not clean:
        raise ValidationError(f"数据集 '{ds.name}' 的 glob 无静态前缀")
    return Path("/".join(clean))


def _count_parquet_rows(root: Path) -> int:
    """把 root 下所有 .parquet 文件加起来数行数（不读 payload，只读 footer）。"""
    if not root.exists() or not root.is_dir():
        return 0
    total = 0
    for p in root.rglob("*.parquet"):
        if p.name.startswith(".") or p.is_symlink():
            continue
        try:
            meta = pq.read_metadata(str(p))
            total += meta.num_rows
        except Exception as exc:
            # 发布前读不了 parquet footer 是坏事，直接抛让上游停
            raise DataError(
                f"读 {p} 的 parquet footer 失败：{exc}；staging 目录可能损坏"
            ) from exc
    return total


def _validate_unique_key(ds: Dataset, root: Path) -> None:
    """#P0-30 key 唯一性发布 gate：契约/registry 声明 unique_key 时必须无重复。

    重复 key（Hive 分区目录 + 同 key 多行）在发布时就拒绝，而不是等读取端
    fan-out / 去重。只有声明了 unique_key 的数据集才做（无声明跳过）。
    """
    unique_key = getattr(ds, "unique_key", None)
    if not unique_key:
        try:
            from data_access.cos_contract import get_cos_contract

            contract = get_cos_contract(getattr(ds, "name", ""))
            unique_key = tuple(contract.unique_key or ()) if contract else None
        except Exception:
            unique_key = None
    if not unique_key:
        return
    keys = [str(k) for k in unique_key]
    parquet_files = sorted(
        str(p) for p in root.rglob("*.parquet")
        if not p.name.startswith(".") and not p.is_symlink()
    )
    if not parquet_files:
        return
    import duckdb

    key_list = ", ".join(f'"{k}"' for k in keys)
    # #P0-30 多文件用 list 参数绑定 read_parquet(?)，不再字符串拼路径（路径含
    # 单引号/特殊字符会直接 SQL 语法错）。
    sql = (
        f"SELECT {key_list}, COUNT(*) AS _n FROM read_parquet(?) "
        f"GROUP BY {key_list} HAVING COUNT(*) > 1 LIMIT 1"
    )
    # #P0-29 unique-key 是 production publish gate，必须 **fail-closed**：DuckDB
    # 坏 / key 列缺失 / parquet 读失败 / SQL 错 → 一律拒绝发布（不能吞掉放行）。
    con = duckdb.connect()
    try:
        rows = con.execute(sql, [parquet_files]).fetchall()
    except Exception as exc:
        raise DataError(
            f"publish 唯一键校验无法执行（fail-closed）：{type(exc).__name__}: {exc}。"
            f"数据集 '{getattr(ds, 'name', '')}' 声明了 unique_key={keys}，但验证失败——"
            "不能在不验证唯一性的情况下发布。请检查 key 列存在性与 parquet 完整性。"
        ) from exc
    finally:
        try:
            con.close()
        except Exception:
            pass
    if rows:
        raise DataError(
            f"publish 唯一键校验失败：{keys} 存在重复（示例 {rows[0][:-1]}）。"
            "发布数据不能违反 unique_key 契约。"
        )


_SCHEMA_TYPE_NORMALIZED = {
    "double": "double", "float": "double", "float64": "double",
    "float32": "float",
    "int": "int64", "integer": "int64", "long": "int64",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "short": "int16",
    "string": "string", "str": "string", "text": "string", "varchar": "string",
    "bool": "bool", "boolean": "bool",
    "date": "date",
    "timestamp": "timestamp", "datetime": "timestamp",
    "decimal": "decimal",
}


def _normalize_schema_type(raw: Any) -> str | None:
    """把 datasets.yaml 声明的类型字符串归一成可比较的 token。

    无法识别的声明类型（object/null/任意）→ None（只查列存在，不查类型）。
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"object", "null", "any", "list", "dict"}:
        return None
    return _SCHEMA_TYPE_NORMALIZED.get(text)


def _normalize_arrow_type(t: Any) -> str | None:
    """把 pyarrow DataType 归一成与声明类型同尺度的 token。"""
    if not isinstance(t, pa.DataType):
        return None
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_decimal(t):
        return "decimal"
    if pa.types.is_floating(t):
        return "double" if t.bit_width == 64 else "float"
    if pa.types.is_integer(t):
        return {8: "int8", 16: "int16", 32: "int32", 64: "int64"}.get(t.bit_width)
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return None


def _validate_candidate_contract(ds: Dataset, root: Path) -> None:
    """#P0-3 candidate 实际数据 vs target 声明 schema 的正式发布门。

    publish pair 只比较 staging/target 双方**声明**的 schema；本门验证 candidate
    里**实际** parquet 数据的列/类型符合 target 契约：
        - target 声明 schema 为空（未启用 schema 校验）→ 跳过
        - 声明列必须全部出现在 candidate 数据里（缺列 fail-closed）
        - 声明类型可识别时，与 candidate 实际类型必须一致（不允许隐式 cast）
        - 同一列跨文件类型不一致 → fail-closed（mixed schema）
    Hive 分区列在目录名里、payload 未必有，不做 payload 要求。
    """
    declared = dict(getattr(ds, "schema", None) or {})
    if not declared:
        return
    fields: dict[str, pa.DataType] = {}
    for p in root.rglob("*.parquet"):
        if p.name.startswith(".") or p.is_symlink():
            continue
        try:
            schema = pq.read_schema(str(p))
        except Exception as exc:
            raise DataError(
                f"publish 契约校验读 {p} 失败（fail-closed）：{exc}"
            ) from exc
        for i in range(len(schema)):
            fld = schema.field(i)
            existing = fields.get(fld.name)
            if existing is not None and existing != fld.type:
                raise DataError(
                    f"publish 契约校验失败：列 {fld.name!r} 在不同文件类型不一致 "
                    f"（{existing} vs {fld.type}），mixed schema 拒绝发布"
                )
            fields[fld.name] = fld.type
    missing = [c for c in declared if c not in fields]
    if missing:
        raise DataError(
            f"publish 契约校验失败：target 声明 schema 列 {missing} 在 candidate "
            f"数据里缺失（实际列：{sorted(fields)}）。拒绝发布。"
        )
    for col, declared_type in declared.items():
        actual = fields[col]
        want = _normalize_schema_type(str(declared_type))
        got = _normalize_arrow_type(actual)
        if want and got and want != got:
            raise DataError(
                f"publish 契约校验失败：列 {col!r} target 声明类型 "
                f"{declared_type!r}（归一 {want}），candidate 实际 {actual}（归一 "
                f"{got}）。隐式 cast 不允许发生在发布门；请修正 staging 数据。"
            )


def _file_inventory(root: Path) -> dict[str, tuple[int, int, str]] | None:
    """#41 发布内容强校验指纹：{相对路径: (rows, bytes, schema_hash)}。

    只比较 .parquet 数据文件（跳过 manifest/临时/symlink）。读 footer 失败
    返回 None（发布中止）。用于 candidate/源/最终 三方一致性比较——比 row count
    更能发现「两表同行数但内容不同」。
    """
    if not root.exists() or not root.is_dir():
        return None
    inv: dict[str, tuple[int, int, str]] = {}
    for p in root.rglob("*.parquet"):
        if p.name.startswith(".") or p.is_symlink():
            continue
        rel = str(p.relative_to(root))
        try:
            meta = pq.read_metadata(str(p))
        except Exception as exc:
            raise DataError(
                f"读 {p} 的 parquet footer 失败：{exc}；目录可能损坏"
            ) from exc
        inv[rel] = (
            meta.num_rows,
            p.stat().st_size,
            _schema_hash_str(meta),
        )
    return inv


def _schema_hash_str(meta: Any) -> str:
    import hashlib
    import json

    arrow_schema = getattr(meta, "schema_arrow", None)
    if arrow_schema is not None:
        pairs = [
            (arrow_schema.field(i).name, str(arrow_schema.field(i).type))
            for i in range(len(arrow_schema))
        ]
    else:
        pairs = [
            (meta.schema.names[i], str(meta.schema.column(i).physical_type))
            for i in range(len(meta.schema.names))
        ]
    text = json.dumps(pairs, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _archive_path(parent: Path, target_name: str) -> Path:
    """算归档位置：{parent}/_archive/{target_name}_{YYYYMMDD_HHMMSS}_{shortuuid}/"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return parent / _ARCHIVE_SUBDIR / f"{target_name}_{ts}_{short}"


def _target_has_published_content(target_dir: Path) -> bool:
    """target 是否真有上一代发布内容（#P0-C13）。

    ``_dataset_mutation`` 事务锁（mutation_lock）会 ``mkdir`` 出空 target，锁文件
    ``.data-access.mutation.lock`` 也落在这里——所以「目录存在」不等于「有旧版本」。
    只认实际数据 parquet（含 hive 分区 year=*/*.parquet 与 manifest parquet）：
    空目录 / 只有锁文件 → 不是旧版本（首次发布不归档）。
    """
    if not target_dir.exists():
        return False
    try:
        return any(target_dir.rglob("*.parquet"))
    except OSError:
        return False


# ---- 拷贝 / 回滚 ------------------------------------------------------------


def _copy_tree(src: Path, dst: Path) -> None:
    """跨 FS 安全的目录拷贝；dst 预期不存在（我们自己生成的唯一名）。

    #P0-32 ``shutil.copytree(symlinks=False)`` 的语义是**跟随** symlink 并把
    目标内容复制过来（不是"不跟"）。发布前先逐层拒绝 staging 里任何 symlink
    组件——否则 copytree 可能把白名单外的文件带进 candidate。
    """
    if dst.exists():
        raise RuntimeError(f"内部错误：candidate 目录已存在 {dst}")
    _assert_no_symlinks(src)
    shutil.copytree(src, dst, symlinks=False)


def _assert_no_symlinks(root: Path) -> None:
    """#P0-32 递归拒绝 staging 目录里任何 symlink 文件/目录组件。"""
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValidationError(
                f"publish 拒绝包含 symlink 的 staging：{p} 是软链接。"
                "发布源必须全是真实文件（避免把白名单外内容带进 candidate）。"
            )


def _rollback_final(target_dir: Path, archive_path: Path | None) -> None:
    """verify 失败时回滚：把刚发布的 target rename 走，归档 rename 回来。"""
    try:
        if target_dir.exists():
            failed = target_dir.parent / f".publish_failed.{uuid.uuid4().hex[:8]}"
            os.rename(str(target_dir), str(failed))
            logger.error(
                "publish 验证失败，失败版本留存在 %s 供排查（未自动删除）", failed,
            )
    except OSError as exc:
        logger.error("publish 回滚：rename target→failed 出错：%s", exc)

    if archive_path and archive_path.exists():
        try:
            os.rename(str(archive_path), str(target_dir))
            logger.info("publish 回滚：归档版本已 rename 回 %s", target_dir)
        except OSError as exc:
            logger.error(
                "publish 回滚：rename archive→target 失败：%s；"
                "target_dir 现在可能缺失，请手工从 %s 恢复",
                exc, archive_path,
            )


# ---- 并发锁 ------------------------------------------------------------------


@contextmanager
def _publish_lock(parent: Path, target_name: str) -> Iterator[None]:
    """fail-fast 锁：O_EXCL 创建锁文件，已存在就直接 raise。

    WHY fail-fast：publish 是低频人为动作（一天个位数），两个人同时发同一个 target
    说明操作没协调好，让第二个人先停下来比排队更合理（排队容易出双发）。
    """
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".publish.{target_name}.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        # 关键：O_EXCL 失败说明锁是别人的，绝对不能在 finally 里 unlink 掉。
        raise ValidationError(
            f"publish 并发冲突：锁文件 {lock_path} 已存在。"
            f"可能有别的 publish 正在跑；如果确认是遗留文件，手工删除该锁文件再重试。"
        ) from exc
    try:
        # 在锁文件里留点追溯信息
        os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode())
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        # 只有我们自己成功 O_EXCL 出来的锁文件才由我们删
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("删除锁文件 %s 失败：%s", lock_path, exc)
