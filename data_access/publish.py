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

import pyarrow.parquet as pq

from . import audit
from .exceptions import DataError, ValidationError
from .namespace import is_namespace_explicit, resolve_namespace
from .paths import PathAuthorizer
from .registry import Dataset, DatasetRegistry, ParametricDataset, StaticDataset


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

        # 校验 staging 真的有东西发
        source_rows = _count_parquet_rows(staging_dir)
        if source_rows == 0:
            raise DataError(
                f"staging 数据集 '{staging_name}' 在 {staging_dir} 没有可发布内容；"
                f"是不是没写 / 参数 {params} 不对？"
            )

        target_parent = target_dir.parent
        target_parent.mkdir(parents=True, exist_ok=True)
        candidate_dir = target_parent / f".publish_candidate.{uuid.uuid4().hex[:12]}"

        with _publish_lock(target_parent, target_dir.name):
            # 第 1 步：copy staging → candidate（跨 FS 兼容的慢路径）
            _copy_tree(staging_dir, candidate_dir)

            # 第 2 步：校验 candidate 确实有数据
            candidate_rows = _count_parquet_rows(candidate_dir)
            if candidate_rows != source_rows:
                raise DataError(
                    f"candidate 拷贝后行数 {candidate_rows} != staging {source_rows}，"
                    f"拷贝可能中断；拒绝发布"
                )

            # 第 3 步：同 FS 原子 rename —— old_published → archive，candidate → final
            if target_dir.exists():
                archive_path = _archive_path(target_parent, target_dir.name)
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(target_dir), str(archive_path))

            try:
                os.rename(str(candidate_dir), str(target_dir))
            except OSError:
                # candidate → final 失败，把归档 rename 回来
                if archive_path and archive_path.exists():
                    os.rename(str(archive_path), str(target_dir))
                    archive_path = None
                raise

            # 第 4 步：发布后验证 —— 真的能从 published 路径读出数据
            try:
                post_rows = _count_parquet_rows(target_dir)
            except Exception as exc:
                # 发布后读失败是大事，尽力回滚
                _rollback_final(target_dir, archive_path)
                raise DataError(
                    f"publish 后读 {target_dir} 失败：{exc}；已回滚到归档版本"
                ) from exc
            if post_rows != source_rows:
                _rollback_final(target_dir, archive_path)
                raise DataError(
                    f"publish 后读出行数 {post_rows} != source {source_rows}；已回滚"
                )

            ok = True             
            # publish流程会先建个临时目录 candidate_dir,把 staging 数据复制进去，校验通过后再rename为 published 目标路径。若candidatedir 仍没改路径，程序结束就在 finally 后面把临时名字 candidate_dir删掉，不留在磁盘上。
            candidate_dir = None  #这行的错误，原来是 candidate_dir = Path()，本意是随便给个哨兵值，让他无参从而是空值，让后面 if 判断为 false， 这样 finally 就会识别 false 从而跳过清理 candidatedir
            # 但无参数Path() 等价于 Path('.')，代表了当前工作目录，是非空而且 exists()==True 恒为真，所以会错误的让 finally 里的 shutil.rmtree(candidate_dir)
            #然后在做pytest时会执行这段命令然后误删当前工作目录

            # 注意：这里 **必须** 是 None 而不是 Path()。Path() 等价于 PoixPath('.')，
            # 既非空也 exists()==True，会让 finally 里的 shutil.rmtree(candidate_dir)
            # 把当前工作目录整个删掉——pytest 下 CWD 就是仓库根。
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


def _archive_path(parent: Path, target_name: str) -> Path:
    """算归档位置：{parent}/_archive/{target_name}_{YYYYMMDD_HHMMSS}_{shortuuid}/"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return parent / _ARCHIVE_SUBDIR / f"{target_name}_{ts}_{short}"


# ---- 拷贝 / 回滚 ------------------------------------------------------------


def _copy_tree(src: Path, dst: Path) -> None:
    """跨 FS 安全的目录拷贝；dst 预期不存在（我们自己生成的唯一名）。"""
    if dst.exists():
        raise RuntimeError(f"内部错误：candidate 目录已存在 {dst}")
    # copytree 不跟 symlink，避免拷贝到白名单外的文件
    shutil.copytree(src, dst, symlinks=False)


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
