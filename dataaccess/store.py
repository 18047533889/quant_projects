"""
data_access.store —— 对外唯一数据读写入口

对外 API（PR1 读 + PR2 写 + PR3 publish/upsert/sql）：
    read_arrow(dataset, columns=, time_range=, instrument_filter=, **params) -> pa.Table
    read_frame(dataset, ...) -> pd.DataFrame
    load_columns(dataset, keys, columns, time_range=, instrument_filter=, **params) -> dict[str, pd.Series]
    write_arrow(dataset, table, mode="overwrite"|"append", partition_by=None, **params) -> dict
    upsert(dataset, table, *, upsert_on=, partition_by=None, **params) -> dict          [PR3]
    publish_from_staging(staging, target, **params) -> dict                              [PR3]
    sql(query, *, read_datasets, read_params=None, params=None) -> pa.Table              [PR3, 有限]

仍未开放：
    流式 open_writer / 任意 DDL — PR4+

非职责：
    - 不做清洗（raw_data_layer 的事）
    - 不做因子计算（factor_engine 的事）
    - 不解析 YAML（registry.py 的事）

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa

if TYPE_CHECKING:
    import polars as pl

from data_access.core import audit
from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from data_access.core.exceptions import DataError, ValidationError
from data_access.registry.params_validation import ParamSpec, params_fingerprint, validate_params
from data_access.read.query_budget import (
    QueryBudget,
    enforce_arrow_budget,
    enforce_scan_file_budget,
    enforce_stream_budget,
    merge_dataset_policies,
    merge_dataset_policy,
    resolve_query_budget,
    validate_query_request,
    _production_mode,
    _strict_read_mode,
)
from data_access.read.read_contract import (
    DataSnapshot,
    FileVersion,
    ReadLineage,
    ReadResult,
    ReadStats,
    SqlReadLineage,
    SqlReadResult,
    build_data_snapshot,
    build_file_manifest,
    file_manifest_hash,
    merge_sql_data_snapshots,
)
from data_access.read.scan_handle import ScanHandle
from data_access.core.namespace import is_namespace_explicit, resolve_namespace
from data_access.registry.paths import (
    PathAuthorizer,
    canonicalize,
    dataset_env_root,
    extra_allowed_roots_from_env,
)
from data_access.read.predicate import Predicate, compile_predicate
from data_access.registry import (
    Dataset,
    DatasetRegistry,
    ParametricDataset,
    StaticDataset,
    load_registry,
)
from data_access.registry.schema_validation import (
    check_schema,
    enforce_schema_or_raise,
    mark_validated,
    reset_validated_cache,
    schema_cache_key,
)
from data_access.read.telemetry import record_polars_scan
from data_access.write.mutation_lock import mutation_lock


logger = logging.getLogger("data_access.store")


_VALID_WRITE_MODES = {"overwrite", "append"}

# 读写路径元参数：不进 params_schema，不进 snapshot params
_READ_PATH_META_KEYS = ("read_root", "_read_root", "bucket_values", "read_auto", "lazy_scan")
_WRITE_PATH_META_KEYS = ("write_root", "_write_root", "write_dir", "_write_dir")


def _assert_instrument_filter_supported(
    ds: Dataset,
    instrument_filter: Sequence[str] | None,
) -> None:
    """schema 已声明但缺少 instrument 列时，禁止 instrument_filter。"""
    if not instrument_filter:
        return
    schema = getattr(ds, "schema", None) or {}
    if schema and ds.instrument_column not in schema:
        raise ValidationError(
            f"数据集 '{ds.name}' 的 instrument_column='{ds.instrument_column}' "
            f"不在 schema 中，不支持 instrument_filter；请去掉 filter 或补 schema。"
        )


class DataAccessStore:
    """统一读取入口。进程内应只有一个实例（由 get_store() 管理）。"""

    def __init__(
        self,
        registry: DatasetRegistry,
        engine: DuckDBEngine,
    ) -> None:
        self._registry = registry
        self._engine = engine
        # 登记表根 + 环境额外根（其他服务器自选读/写目录时用）
        self._authorizer = PathAuthorizer(
            list(registry.allowed_roots()) + extra_allowed_roots_from_env()
        )
        # PR8 + P0：首访 schema 自检缓存 key = dataset + params + manifest
        self._schema_checked: set[str] = set()
        self._schema_check_lock = threading.Lock()
        self._registry_hash = _compute_registry_hash(registry)

    @property
    def registry(self) -> DatasetRegistry:
        """已加载的数据集登记表（只读）。"""
        return self._registry

    def get_dataset(self, name: str) -> Dataset:
        """按名获取已注册数据集元数据。"""
        return self._registry.get(name)

    def registry_fingerprint(self) -> str:
        """登记表内容指纹（用于 data_snapshot_id）。"""
        return self._registry_hash

    def describe_dataset(
        self,
        dataset: str,
        *,
        params: Mapping[str, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
    ) -> DataSnapshot:
        """解析路径并构建数据快照（不读数据）。"""
        ds = self._registry.get(dataset)
        raw_params = dict(params or {})
        paths = self._resolve_paths(
            ds,
            raw_params,
            instrument_filter=instrument_filter,
        )
        snap_params = self._split_read_params(raw_params)
        return build_data_snapshot(
            dataset=dataset,
            registry_hash=self._registry_hash,
            schema=getattr(ds, "schema", None),
            paths=paths,
            params=snap_params if isinstance(ds, ParametricDataset) else None,
        )

    def build_sql_snapshot(
        self,
        read_datasets: Sequence[str],
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
    ) -> DataSnapshot:
        """为 sql() 多 dataset 读路径构建合并 DataSnapshot（含 COS remote）。"""
        params_map = dict(read_params or {})
        time_ranges = dict(read_time_ranges or {})
        snapshots: list[DataSnapshot] = []
        for name in sorted(read_datasets):
            ds = self._registry.get(name)
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_ranges.get(name),
                params=dict(params_map.get(name, {})),
            )
            snapshots.append(
                self._build_snapshot(
                    dataset=name,
                    ds=ds,
                    paths=paths,
                    params=dict(params_map.get(name, {})),
                )
            )
        return merge_sql_data_snapshots(
            snapshots,
            registry_hash=self._registry_hash,
        )

    def _resolve_read_budget(
        self,
        ds: Dataset,
        query_budget: QueryBudget | None,
    ) -> QueryBudget:
        base = resolve_query_budget(query_budget)
        return merge_dataset_policy(base, ds.query_policy)

    @staticmethod
    def _split_read_params(params: dict[str, Any]) -> dict[str, Any]:
        """剥离读/写路径元参数（不进 params_schema / snapshot params）。"""
        out = dict(params)
        for key in _READ_PATH_META_KEYS + _WRITE_PATH_META_KEYS:
            out.pop(key, None)
        return out

    @staticmethod
    def _split_write_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """拆出写路径元参数与数据集参数。

        返回 ``(path_meta, clean_params)``：
            - ``write_root``：替换数据集根（参数化数据集仍拼接 factor_id 等后缀）
            - ``write_dir``：最终写入目录（完全覆盖模板解析结果）
        """
        out = dict(params)
        meta: dict[str, Any] = {}
        wr = out.pop("write_root", None) or out.pop("_write_root", None)
        wd = out.pop("write_dir", None) or out.pop("_write_dir", None)
        if wr is not None:
            meta["write_root"] = wr
        if wd is not None:
            meta["write_dir"] = wd
        return meta, out

    def _pop_read_root(self, ds: Dataset, params: dict[str, Any]) -> str | None:
        """从 params / 环境变量取出可选读根覆盖。"""
        read_root = params.pop("read_root", None) or params.pop("_read_root", None)
        if read_root is None:
            read_root = dataset_env_root(ds.name, "read")
        return str(read_root) if read_root is not None else None

    def _assert_path_override_allowed(self, ds: Dataset, *, kind: str, value: str) -> None:
        """production/严格模式下禁止路径覆盖（写目录 write_dir/write_root 例外：staging 需要可选落盘）。"""
        if kind == "read" and (_production_mode() or _strict_read_mode()):
            raise ValidationError(
                f"production/严格读模式禁止 read_root 覆盖数据集 '{ds.name}' 根路径；"
                f"收到 read_root={value!r}。"
                "灰度切读请仅在开发环境使用，或通过独立 registry 数据集登记。"
            )

    @staticmethod
    def _rewrite_root_prefix(
        glob_paths: list[str],
        *,
        old_static_root: Path,
        new_root: Path,
    ) -> list[str]:
        """把解析出的 glob 路径中 static_root 前缀替换为 new_root。"""
        old = str(canonicalize(old_static_root)).rstrip("/")
        new = str(canonicalize(new_root)).rstrip("/")
        out: list[str] = []
        for g in glob_paths:
            if g == old or g.startswith(old + "/"):
                out.append(new + g[len(old):])
            else:
                # 解析结果不在 static_root 下时，退化为 new_root + 相对尾缀
                # （例如模板展开后路径略有差异）
                rel = Path(g)
                try:
                    suffix = rel.relative_to(old_static_root)
                    out.append(str(new_root / suffix))
                except ValueError:
                    out.append(str(new_root / rel.name))
        return out

    def _build_snapshot(
        self,
        *,
        dataset: str,
        ds: Dataset,
        paths: list[str],
        params: dict[str, Any],
        files: Sequence[FileVersion] | None = None,
    ) -> DataSnapshot:
        read_params = self._split_read_params(params)
        return build_data_snapshot(
            dataset=dataset,
            registry_hash=self._registry_hash,
            schema=getattr(ds, "schema", None),
            paths=paths,
            params=read_params if isinstance(ds, ParametricDataset) else None,
            files=files,
        )

    def _schema_fingerprint(
        self,
        ds: Dataset,
        paths: list[str],
        params: dict[str, Any],
        *,
        files: Sequence[FileVersion] | None = None,
    ) -> str:
        read_params = self._split_read_params(params)
        pf = params_fingerprint(read_params if isinstance(ds, ParametricDataset) else None)
        file_versions = files if files is not None else build_file_manifest(paths)
        manifest = file_manifest_hash(file_versions)
        return schema_cache_key(ds.name, params_fingerprint=pf, manifest_hash=manifest)

    def read_result(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> ReadResult:
        """读取数据集并返回带 ``DataSnapshot`` 的 ``ReadResult``（审计/lineage 用）。"""
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        paths = self._prepare_dataset_read(
            ds,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
        )
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)
        self._ensure_schema(ds, paths, params, files=files)
        snapshot = self._build_snapshot(
            dataset=dataset, ds=ds, paths=paths, params=params, files=files
        )
        lineage = ReadLineage(
            dataset=dataset,
            columns=tuple(columns) if columns else (),
            time_range=time_range,
            instrument_filter=tuple(instrument_filter) if instrument_filter else (),
            params=snapshot.params,
        )

        sql, sql_params = self._build_select_sql(
            ds=ds,
            paths=paths,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            limit=limit,
        )

        start = time.perf_counter()
        ok = False
        err_msg: str | None = None
        table: pa.Table | None = None
        try:
            table = self._engine.execute_arrow(sql, sql_params)
            elapsed_ms = (time.perf_counter() - start) * 1000
            enforce_arrow_budget(budget, table, elapsed_ms=elapsed_ms)
            ok = True
            stats = ReadStats(
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                paths=tuple(paths[:20]),
            )
            logger.info(
                "read_result dataset=%s rows=%d cols=%d elapsed_ms=%.1f snapshot=%s",
                dataset,
                table.num_rows,
                table.num_columns,
                elapsed_ms,
                snapshot.snapshot_id,
            )
            return ReadResult(
                table=table,
                snapshot=snapshot,
                stats=stats,
                lineage=lineage,
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            audit.record(
                op="read",
                dataset=dataset,
                ok=ok,
                rows=table.num_rows if table is not None else None,
                paths=paths[:5] if paths else None,
                params=params or None,
                elapsed_ms=elapsed_ms,
                error=err_msg,
                extra={
                    "columns": list(columns) if columns else None,
                    "snapshot_id": snapshot.snapshot_id,
                },
            )

    def read_asof(
        self,
        dataset: str,
        *,
        as_of: Any,
        columns: Sequence[str] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> ReadResult:
        """Point-in-time 读：``time_column <= as_of``（闭区间上界）。

        返回带 ``DataSnapshot`` 的 ``ReadResult``，供 lineage / 回测复现使用。
        """
        if as_of is None:
            raise ValidationError("read_asof 必须指定 as_of（时间戳或日期字符串）")
        return self.read_result(
            dataset,
            columns=columns,
            time_range=(None, as_of),
            instrument_filter=instrument_filter,
            limit=limit,
            query_budget=query_budget,
            **params,
        )

    def _resolve_sql_budget(
        self,
        read_datasets: Sequence[str],
        query_budget: QueryBudget | None,
    ) -> QueryBudget:
        base = resolve_query_budget(query_budget)
        policies = [self._registry.get(name).query_policy for name in read_datasets]
        return merge_dataset_policies(base, policies)

    # ---- 对外主 API ----

    def read_arrow(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> pa.Table:
        """读取已注册数据集，返回 Arrow Table（零拷贝，推荐路径）。

        参数：
            dataset: datasets.yaml 里注册的数据集名
            columns: 只读指定列（强烈推荐指定，避免宽表全扫）
            time_range: (start, end) 闭区间，按 registry 里的 time_column 过滤
            instrument_filter: 标的白名单
            **params: 参数化数据集的参数（如 factor_id=）

        返回：
            pa.Table；列顺序与 columns 一致；若 columns=None 则为 SELECT *。

        常见错误：
            ValidationError: 数据集未注册 / 参数缺失 / 路径越界
            DataError: 读出的结果为空（time_range 筛没了 → 通常是上游 bug）
            EngineError: DuckDB 内部错

        示例：
            >>> store.read_arrow(
            ...     "us_stocks_sip_day_aggs",
            ...     columns=["align_time", "ticker", "close"],
            ...     time_range=("2024-01-01", "2024-12-31"),
            ...     instrument_filter=["AAPL", "MSFT"],
            ... )
        """
        return self.read_result(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            limit=limit,
            query_budget=query_budget,
            **params,
        ).table

    def read_arrow_stream(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        batch_size: int = 100_000,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> Iterator[pa.RecordBatch]:
        """流式读取（PR7）：返回一个 Arrow RecordBatch 生成器。

        使用场景：
            - 单次查询结果大到不愿全部 materialize 到内存（几十 GB 回测输出）
            - ETL / 逐 batch 聚合后直接写下游（中间结果不留内存）
            - 处理慢但是连续可分：生成信号 → 发下游 → 丢 batch 继续读

        与 read_arrow 的取舍：
            read_arrow 一次全拿 = 零拷贝直接给 pandas/polars，延迟高但吞吐好；
            read_arrow_stream 按 batch = 延迟低、内存峰值平，但每 batch 有
            少量 overhead（cursor 状态 + batch 拼装）。百 MB 级别以内用 read_arrow，
            多 GB 扫描用 read_arrow_stream。

        参数：
            batch_size: 每个 RecordBatch 目标行数；默认 100k（与 DuckDB vector
                size 对齐）。太小会浪费向量化收益，太大内存波动大。

        注意：
            - 生成器用 for 循环消费；迭代完自动释放底层 cursor。
            - 中途 break 会让 Python GC 清理 reader，cursor 也会跟着释放。
            - 不要跨线程传 reader；DuckDB 的 record batch reader 线程不安全。

        示例：
            >>> total = 0
            >>> for batch in store.read_arrow_stream("factor_lake", factor_id="mom_3d"):
            ...     total += batch.num_rows
        """
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        paths = self._prepare_dataset_read(
            ds,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
        )
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)
        self._ensure_schema(ds, paths, params, files=files)
        sql, sql_params = self._build_select_sql(
            ds=ds,
            paths=paths,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            limit=limit,
        )
        reader = self._engine.execute_reader(sql, sql_params, batch_size=batch_size)
        start = time.perf_counter()
        total_rows = 0
        total_bytes = 0
        ok = False
        err_msg: str | None = None

        def _iter_batches() -> Iterator[pa.RecordBatch]:
            nonlocal total_rows, total_bytes, ok, err_msg
            try:
                for batch in reader:
                    total_rows += batch.num_rows
                    total_bytes += batch.nbytes
                    enforce_stream_budget(
                        budget,
                        total_rows=total_rows,
                        total_bytes=total_bytes,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                    )
                    yield batch
                ok = True
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                audit.record(
                    op="read",
                    dataset=dataset,
                    ok=ok,
                    rows=total_rows,
                    paths=paths[:5] if paths else None,
                    params=params or None,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=err_msg,
                    extra={"stream": True, "columns": list(columns) if columns else None},
                )

        logger.info(
            "read_arrow_stream dataset=%s batch_size=%d params=%s",
            dataset, batch_size, params or "{}",
        )
        return _iter_batches()

    def _scan_polars_with_paths(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> tuple[Any, list[str]]:
        """内部：构建 Polars LazyFrame，同时返回已解析 paths（避免 scan 二次准备）。"""
        try:
            import polars as pl_mod
        except ImportError as exc:
            raise ImportError(
                "scan_polars 需要 polars。pip install polars 后重试。"
            ) from exc

        scan_start = time.perf_counter()
        ds = self._registry.get(dataset)
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        paths = self._prepare_dataset_read(
            ds,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
        )
        # scan_polars 也走 schema 自检：发现声明漂移尽早报。
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)
        self._ensure_schema(ds, paths, params, files=files)

        # pl.scan_parquet 可以接 list[str]，也支持 glob。我们传 list 给它。
        # 跨年份列不一致时靠 union_by_name：
        #   - Polars 1.30+：参数是 missing_columns='insert'/'raise'，老的 allow_missing_columns 被弃用
        #   - Polars 1.28 ~ 1.29：参数名是 allow_missing_columns
        # 用 try/except 兜两边，避免硬编码版本号判断。
        scan_kwargs: dict[str, Any] = {"hive_partitioning": ds.hive_partitioning}
        try:
            lf = pl_mod.scan_parquet(
                paths,
                missing_columns="insert" if ds.union_by_name else "raise",
                **scan_kwargs,
            )
        except TypeError:
            lf = pl_mod.scan_parquet(
                paths,
                allow_missing_columns=ds.union_by_name,
                **scan_kwargs,
            )

        # 仅选列：给 Polars optimizer 做 column pushdown
        # time/instrument 列如果 columns 里没包含但有过滤条件，先留着让 filter
        # 用上，最后 Polars 自己会剪掉
        needed_cols: list[str] = []
        if columns:
            needed_cols.extend(columns)
            if time_range is not None and ds.time_column not in needed_cols:
                needed_cols.append(ds.time_column)
            if instrument_filter is not None and ds.instrument_column not in needed_cols:
                needed_cols.append(ds.instrument_column)
            lf = lf.select([pl_mod.col(c) for c in needed_cols])

        # Lazy filter：time_range + instrument_filter
        # 注意 Polars 不像 DuckDB 能把 '2024-01-01' 这种字符串自动 cast 成 datetime，
        # 对 datetime 列比较字符串会抛 InvalidOperationError。统一用 pandas.Timestamp
        # 把字符串 / datetime / date 归一化到 python datetime，再交给 pl.lit——
        # 这样无论下层是 Datetime 还是 Date 列，Polars 自己都能 coerce。
        if time_range is not None:
            start_val, end_val = time_range
            if start_val is not None:
                lf = lf.filter(
                    pl_mod.col(ds.time_column) >= pl_mod.lit(_to_pydatetime(start_val))
                )
            if end_val is not None:
                lf = lf.filter(
                    pl_mod.col(ds.time_column) <= pl_mod.lit(_to_pydatetime(end_val))
                )
        if instrument_filter is not None:
            lf = lf.filter(pl_mod.col(ds.instrument_column).is_in(list(instrument_filter)))

        # 如果 columns 指定了但我们追加过 time/instrument，在 filter 之后再把
        # 原始 columns 剪回来（filter 用过了就可以丢）
        if columns and needed_cols != list(columns):
            lf = lf.select([pl_mod.col(c) for c in columns])

        record_polars_scan(
            dataset=dataset,
            elapsed_ms=(time.perf_counter() - scan_start) * 1000,
            paths_count=len(paths),
        )
        audit.record(
            op="read",
            dataset=dataset,
            ok=True,
            paths=paths[:5] if paths else None,
            params=params or None,
            elapsed_ms=(time.perf_counter() - scan_start) * 1000,
            extra={"scan_polars": True, "columns": list(columns) if columns else None},
        )
        return lf, paths

    def scan_polars(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ):
        """返回 Polars LazyFrame，惰性扫描已注册数据集（PR7）。

        为什么提供：Polars 的 lazy engine 是 Rust 实现的 pushdown（列/谓词/
        partition 剪枝全套），对「多表 join + 窗口函数 + 大量复杂变换」远优于
        DuckDB 的 Arrow 结果 → pandas → polars 路径。尤其是跑复杂因子 pipeline
        时用 LazyFrame 能让 Polars 自己决定什么时候物化。

        示例：
            >>> lf = store.scan_polars("factor_lake", factor_id="mom_3d",
            ...                         time_range=("2024-01-01", None))
            >>> df = lf.filter(pl.col("asset").is_in(["AAPL"])).collect()

        依赖：
            需要 `polars` 已安装；没装会 raise ImportError 并提示 `pip install polars`。
        """
        lf, _paths = self._scan_polars_with_paths(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            query_budget=query_budget,
            **params,
        )
        return lf

    def scan(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> ScanHandle:
        """受控 Polars 扫描：``collect()`` 强制 budget + snapshot（production 推荐）。"""
        ds = self._registry.get(dataset)
        budget = self._resolve_read_budget(ds, query_budget)
        lf, paths = self._scan_polars_with_paths(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            query_budget=query_budget,
            **params,
        )
        snapshot = self._build_snapshot(
            dataset=dataset, ds=ds, paths=paths, params=params
        )
        lineage = ReadLineage(
            dataset=dataset,
            columns=tuple(columns) if columns else (),
            time_range=time_range,
            instrument_filter=tuple(instrument_filter) if instrument_filter else (),
            params=snapshot.params,
        )
        return ScanHandle(
            _lf=lf,
            snapshot=snapshot,
            budget=budget,
            lineage=lineage,
            _store=self,
        )

    def read_frame(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        **params: Any,
    ):
        """同 read_arrow，但返回 pandas DataFrame。

        WHY 单独提供 read_frame：最末端要画图/落 CSV 的场景很多，每次都让
        调用方 .to_pandas() 也是重复劳动；但 pipeline 中段仍推荐 read_arrow。
        """
        table = self.read_arrow(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            **params,
        )
        return table.to_pandas(self_destruct=True, split_blocks=True)

    def read_auto(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        mode: str = "auto",
        prefer_polars: bool = False,
        batch_size: int = 100_000,
        **params: Any,
    ) -> pa.Table:
        """按数据集规模自动路由 read_arrow / read_arrow_stream / scan_polars。

        ``mode="auto"`` 时用 parquet footer 估算行数；``arrow`` / ``stream`` /
        ``polars`` 可强制指定路径。最终统一 materialize 为 Arrow Table。

        大结果且需保持低内存峰值时，优先 ``read_auto_stream()``。
        """
        from data_access.read.read_auto_router import resolve_read_auto_mode

        ds = self._registry.get(dataset)
        budget = self._resolve_read_budget(ds, query_budget)
        resolved_mode, _stats = resolve_read_auto_mode(
            self,
            dataset,
            columns=list(columns) if columns else None,
            time_range=time_range,
            query_budget=query_budget,
            mode=mode,
            prefer_polars=prefer_polars,
            budget=budget,
            **params,
        )
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )

        read_kwargs = dict(
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            query_budget=query_budget,
            **params,
        )
        if resolved_mode == "polars":
            from data_access.read.query_budget import collect_polars_with_budget

            lf = self.scan_polars(dataset, **read_kwargs)
            table = collect_polars_with_budget(lf, query_budget=budget)
            if limit is not None and table.num_rows > limit:
                table = table.slice(0, limit)
            return table
        if resolved_mode == "stream":
            if mode == "auto":
                logger.warning(
                    "read_auto(auto) estimated stream-sized result but API requires "
                    "Arrow Table; falling back to read_arrow with budget enforcement. "
                    "Use read_auto_stream/read_arrow_stream/sql_stream for true streaming."
                )
                resolved_mode = "arrow"
            else:
                batches = list(
                    self.read_arrow_stream(
                        dataset,
                        batch_size=batch_size,
                        limit=limit,
                        **read_kwargs,
                    )
                )
                if not batches:
                    return pa.table({})
                return pa.Table.from_batches(batches)
        return self.read_arrow(dataset, limit=limit, **read_kwargs)

    def read_auto_stream(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        mode: str = "auto",
        prefer_polars: bool = False,
        batch_size: int = 100_000,
        **params: Any,
    ) -> Iterator[pa.RecordBatch]:
        """按规模自动路由，**始终**以 RecordBatch 流式返回（不物化全表）。

        ``mode="auto"`` 仅影响路由日志；polars 路由会 fallback 到
        ``read_arrow_stream`` 并打 warning。需要 LazyFrame 时用 ``scan_polars``。
        """
        from data_access.read.read_auto_router import resolve_read_auto_mode

        ds = self._registry.get(dataset)
        budget = self._resolve_read_budget(ds, query_budget)
        resolved_mode, _stats = resolve_read_auto_mode(
            self,
            dataset,
            columns=list(columns) if columns else None,
            time_range=time_range,
            query_budget=query_budget,
            mode=mode,
            prefer_polars=prefer_polars,
            budget=budget,
            **params,
        )
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        if resolved_mode == "polars":
            logger.warning(
                "read_auto_stream dataset=%s resolved_mode=polars; "
                "fallback to read_arrow_stream for true batch streaming",
                dataset,
            )
        logger.info(
            "read_auto_stream dataset=%s resolved_mode=%s batch_size=%d",
            dataset,
            resolved_mode,
            batch_size,
        )
        read_kwargs = dict(
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            query_budget=query_budget,
            **params,
        )
        yield from self.read_arrow_stream(
            dataset,
            batch_size=batch_size,
            limit=limit,
            **read_kwargs,
        )

    def dataset_read_stats(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        prefer_polars: bool = False,
        **params: Any,
    ):
        """返回 ``DatasetReadStats``（footer 行数估算 + 建议读路径）。"""
        from data_access.read.stats import dataset_read_stats as _dataset_read_stats

        return _dataset_read_stats(
            self,
            dataset,
            columns=list(columns) if columns else None,
            time_range=time_range,
            prefer_polars=prefer_polars,
            **params,
        )

    def load_columns(
        self,
        dataset: str,
        *,
        columns: Sequence[str],
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        output_names: dict[str, str] | None = None,
        normalize_timestamp: bool | None = None,
        timestamp_unit: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """批量读多列，每列转成 `(timestamp, instrument)` MultiIndex Series。

        这是 ParquetSource.load_column 的「批量版」，是 PR1 性能关键 API：
        老代码每列都重开 parquet、重读 ts/instrument，这里一次 SQL 拿齐，
        再在内存里按列拆分。

        参数：
            columns: 要读取的值列（不包括 time/instrument，它们 registry 里有）
            output_names: 可选列名映射 {源列: 目标 Series name}
            normalize_timestamp / timestamp_unit: 传给 adapter
            **params: 参数化数据集参数

        返回：
            {output_name: pd.Series}；Series 的 name 是 output_name 或源列名

        示例：
            >>> cols = store.load_columns(
            ...     "us_stocks_sip_day_aggs",
            ...     columns=["close", "volume", "open"],
            ...     time_range=("2024-01-01", "2024-12-31"),
            ... )
            >>> cols["close"].head()
        """
        ds = self._registry.get(dataset)
        output_names = output_names or {}
        adapter_opts = adapter_options_for_dataset(ds)
        if normalize_timestamp is None:
            normalize_timestamp = bool(adapter_opts.get("normalize_timestamp", False))
        if timestamp_unit is None:
            timestamp_unit = adapter_opts.get("timestamp_unit")

        # 一次 SQL 同时选所有需要的列 + 时间列 + 标的列
        all_cols = list(dict.fromkeys(
            [ds.time_column, ds.instrument_column, *columns]
        ))
        from data_access.read.key_policy import resolve_key_policy

        read_result = self.read_result(
            dataset,
            columns=all_cols,
            time_range=time_range,
            instrument_filter=instrument_filter,
            **params,
        )
        table = read_result.table

        if table.num_rows == 0:
            raise DataError(
                f"数据集 '{dataset}' 在给定条件下读出 0 行；"
                f"检查 time_range / instrument_filter 或数据是否真的存在"
            )

        # 批量路径：一次 Arrow→pandas，多列拆分（避免每列重复转换）
        reverse_names = {src: tgt for src, tgt in output_names.items()}
        result = arrow_table_to_multiindex_columns(
            table,
            timestamp_column=ds.time_column,
            instrument_column=ds.instrument_column,
            value_columns=list(columns),
            output_names=reverse_names or None,
            normalize_timestamp=normalize_timestamp,
            timestamp_unit=timestamp_unit,
            key_policy=resolve_key_policy(),
        )
        # output_names 映射的是 physical→logical，批量函数 key 用 target name
        if output_names:
            return {
                output_names.get(src, src): result[output_names.get(src, src)]
                for src in columns
            }
        return result

    # ---- 写入 API（PR2） ----

    def write_arrow(
        self,
        dataset: str,
        table: pa.Table,
        *,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """把 Arrow Table 写到已注册的数据集。只允许 namespaced/staging 两种 access_mode。

        参数：
            dataset: datasets.yaml 里登记的数据集名
            table: 要写入的 Arrow Table
            mode: "overwrite"（先清目标目录再写）| "append"（加新文件不动旧的）
            partition_by: 分区列，如 ["year"]。传了之后用 hive 布局落盘
            **params: 参数化数据集参数（如 factor_id=）；可选路径覆盖：
                - write_root=：替换数据集根，参数后缀仍保留
                - write_dir=：最终写入目录（完全指定落盘位置）
                也可设环境变量 DATA_ACCESS_WRITE_ROOT_<数据集大写名>

        返回：
            {"rows": int, "path": str, "mode": str}，同时写一行审计日志

        拒绝规则：
            - published 数据集：直写永远 raise ValidationError，必须走 publish 流程（PR3）
            - namespace 兜底：写 namespaced/staging 时若未显式 export QUANT_RUN_NAMESPACE，
              只 warning 不拦（避免阻塞脚本调试），但日志里标记 namespace_explicit=false

        示例：
            >>> import pyarrow as pa
            >>> tbl = pa.table({"ts": [...], "symbol": [...], "pnl": [...]})
            >>> store.write_arrow(
            ...     "single_asset_backtest_runs",
            ...     tbl,
            ...     strategy_id="mom_3d",
            ...     version="v1",
            ...     mode="overwrite",
            ... )
            >>> # 自定义落盘目录（需在 DATA_ACCESS_EXTRA_ALLOWED_ROOTS 白名单内）
            >>> store.write_arrow(
            ...     "factor_lake_staging", tbl, factor_id="x",
            ...     write_dir="/data/my_out/factor_x",
            ... )
        """
        if mode not in _VALID_WRITE_MODES:
            raise ValidationError(
                f"write_arrow mode 必须是 {_VALID_WRITE_MODES}，收到 {mode!r}"
            )
        if not isinstance(table, pa.Table):
            raise ValidationError(
                f"write_arrow 只接受 pyarrow.Table，收到 {type(table).__name__}；"
                f"如有 DataFrame 请先 pyarrow.Table.from_pandas(df)"
            )

        ds = self._registry.get(dataset)

        if ds.access_mode == "published":
            raise ValidationError(
                f"数据集 '{dataset}' 是 published，不允许直写。"
                f"请写对应的 staging 数据集，再用 publish_from_staging 晋升（PR3）"
            )
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' 的 access_mode={ds.access_mode!r} 不支持写入"
            )

        if not is_namespace_explicit():
            logger.warning(
                "写入 namespaced/staging 数据集 '%s' 但 QUANT_RUN_NAMESPACE 未显式设置，"
                "用的是兜底 namespace=%s。生产脚本请 export。",
                dataset, resolve_namespace(),
            )

        target_dir = self._resolve_write_dir(ds, params)
        self._authorizer.resolve_and_authorize(str(target_dir))

        ok = False
        err_msg: str | None = None
        files_written: list[Path] = []
        with audit.AuditTimer() as timer:
            try:
                with mutation_lock(target_dir):
                    if mode == "overwrite":
                        self._clear_dir(target_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    files_written = self._write_table_to_dir(
                        table, target_dir, partition_by=partition_by,
                    )
                ok = True
                err_msg = None
            except Exception as exc:
                ok = False
                err_msg = f"{type(exc).__name__}: {exc}"
                files_written = []
                # 出错仍然先记审计再抛，方便事后追查
                raise
            finally:
                audit.record(
                    op="write",
                    dataset=dataset,
                    ok=ok,
                    mode=mode,
                    rows=table.num_rows,
                    paths=[str(p) for p in files_written] if files_written else [str(target_dir)],
                    params=params or None,
                    elapsed_ms=timer.elapsed_ms,
                    error=err_msg if not ok else None,
                    extra={"partition_by": list(partition_by)} if partition_by else None,
                )

        logger.info(
            "write_arrow dataset=%s rows=%d mode=%s files=%d elapsed_ms=%.1f",
            dataset, table.num_rows, mode, len(files_written), timer.elapsed_ms,
        )
        return {
            "rows": table.num_rows,
            "path": str(target_dir),
            "files": [str(p) for p in files_written],
            "mode": mode,
        }

    def upsert(
        self,
        dataset: str,
        table: pa.Table,
        *,
        upsert_on: Sequence[str],
        partition_by: Sequence[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """读-合-写的幂等合并写入。只允许 namespaced/staging。

        参数：
            dataset: datasets.yaml 里登记的 namespaced/staging 数据集
            table: 新增/更新的数据；列里必须包含 upsert_on + partition_by
            upsert_on: 合并键（必填）。同 upsert_on 组合已存在 → 新值覆盖旧值
            partition_by: 分区列。传了之后按 hive 布局逐分区合并（每分区一个
                data.parquet，合并时做 read-merge-tmp-rename 原子写）
            **params: 参数化数据集的参数（如 factor_id=）

        返回：
            {"rows": int, "path": str, "partitions": list[str], "elapsed_ms": float}

        与 write_arrow 的区别：
            - write_arrow(overwrite) 把目标目录清空再整写；upsert 保留旧数据，
              按 upsert_on 合并
            - write_arrow(append) 单纯加文件，不处理重复；upsert 会 dedup
            - upsert 按分区做读-合-写，比 overwrite 贵，但避免了全量重算

        示例：
            >>> store.upsert(
            ...     "factor_lake_staging", tbl,
            ...     factor_id="mom_3d",
            ...     upsert_on=["datetime", "asset"],
            ...     partition_by=["year"],
            ... )
        """
        if not isinstance(table, pa.Table):
            raise ValidationError(
                f"upsert 只接受 pyarrow.Table，收到 {type(table).__name__}；"
                f"DataFrame 请先 pyarrow.Table.from_pandas(df)"
            )

        ds = self._registry.get(dataset)

        if ds.access_mode == "published":
            raise ValidationError(
                f"数据集 '{dataset}' 是 published，不允许 upsert。"
                f"请 upsert 对应的 staging 数据集，再 publish_from_staging 晋升"
            )
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' 的 access_mode={ds.access_mode!r} 不支持 upsert"
            )

        if not is_namespace_explicit():
            logger.warning(
                "upsert 到 namespaced/staging 数据集 '%s' 但 QUANT_RUN_NAMESPACE 未显式设置，"
                "用的是兜底 namespace=%s。生产脚本请 export。",
                dataset, resolve_namespace(),
            )

        target_dir = self._resolve_write_dir(ds, params)

        from data_access.write import upsert as upsert_mod

        return upsert_mod.upsert_table(
            ds=ds,
            authorizer=self._authorizer,
            target_dir=target_dir,
            new_table=table,
            upsert_on=upsert_on,
            partition_by=partition_by,
            params=params,
        )

    def delete_rows(
        self,
        dataset: str,
        *,
        start: Any | None = None,
        end: Any | None = None,
        after: Any | None = None,
        time_column: str | None = None,
        dry_run: bool = False,
        max_rows: int | None = None,
        reason: str | None = None,
        ticket_id: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """从 namespaced/staging 数据集删除时间范围内的行（行级补偿删除）。"""
        ds = self._registry.get(dataset)
        if ds.access_mode not in {"namespaced", "staging"}:
            raise ValidationError(
                f"数据集 '{dataset}' access_mode={ds.access_mode!r} 不支持 delete_rows"
            )
        tc = time_column or ds.time_column
        target_dir = self._resolve_write_dir(ds, params)
        from data_access.write import upsert as upsert_mod

        return upsert_mod.delete_rows_from_dataset(
            ds=ds,
            authorizer=self._authorizer,
            target_dir=target_dir,
            time_column=tc,
            start=start,
            end=end,
            after=after,
            params=params,
            dry_run=dry_run,
            max_rows=max_rows,
            reason=reason,
            ticket_id=ticket_id,
        )

    def resolve_dataset_path(self, dataset: str, **params: Any) -> Path:
        """解析已登记数据集在当前 params 下的物理目录。"""
        from data_access.write.publish import _resolve_dataset_dir

        ds = self._registry.get(dataset)
        return _resolve_dataset_dir(ds, params)

    def dataset_axis_columns(self, dataset: str) -> tuple[str, str]:
        """返回数据集的时间列与标的列名（供 factor_engine SQL 下推使用）。"""
        ds = self._registry.get(dataset)
        return ds.time_column, ds.instrument_column

    def publish_from_staging(
        self,
        staging_dataset: str,
        target_dataset: str,
        **params: Any,
    ) -> dict[str, Any]:
        """把 staging 数据集的内容晋升为 published 版本（PR3 实装）。

        参数：
            staging_dataset: 源 staging 数据集（access_mode=staging）
            target_dataset:  目标 published 数据集（access_mode=published）
            **params:        参数化数据集的参数（两边必须接受同一组）

        返回：
            {"source": {...}, "target_path": str, "archive_path": str | None,
             "rows": int, "elapsed_ms": float}

        行为：
            1. 校验两数据集的 access_mode / schema / params_schema 匹配
            2. 把 staging 的内容 copytree 到 published 父目录下的候选目录
            3. 加并发锁；把旧 published（如存在）rename 到 _archive/...
            4. rename 候选 → published，做发布后读取验证
            5. 任何阶段失败都尽力回滚到一致状态；全程写审计日志

        示例：
            >>> store.write_arrow("factor_lake_staging", tbl, factor_id="mom_3d")
            >>> store.publish_from_staging(
            ...     "factor_lake_staging", "factor_lake",
            ...     factor_id="mom_3d",
            ... )
        """
        from data_access.write import publish

        return publish.publish_from_staging(
            registry=self._registry,
            authorizer=self._authorizer,
            staging_name=staging_dataset,
            target_name=target_dataset,
            **params,
        )

    def sql(
        self,
        query: str,
        *,
        read_datasets: Sequence[str],
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
        view_columns: Mapping[str, Sequence[str]] | None = None,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
    ) -> pa.Table:
        """有限 SQL 逃生口：只允许 SELECT，FROM 的表必须是 read_datasets 里预声明的数据集。

        参数：
            query: 用户提供的 SELECT（可含 WITH / JOIN / GROUP BY 等）。
                禁用 INSERT/UPDATE/DELETE/COPY/ATTACH/PRAGMA/read_parquet 等关键字。
            read_datasets: 本次查询要访问的数据集名列表（必填，非空）。
                每个数据集会以其注册名作为 TEMP VIEW 暴露给 query 的 FROM。
            read_params: 参数化数据集的参数 {dataset_name: {param: value}}，
                例如 {"factor_lake": {"factor_id": "mom_3d"}}
            read_time_ranges: {dataset_name: (start, end)}，COS remote 按日选文件
                并在 view 上做 time_column 过滤；强烈建议对行情表传入。
            params: query 自身的 ? 绑定参数（用户层面的查询参数，不是路径）

        返回：
            pa.Table —— 用户 SQL 的结果

        示例：
            >>> tbl = store.sql(
            ...     "SELECT asset, AVG(value) FROM {{factor_lake}} "
            ...     "WHERE datetime >= ? GROUP BY asset",
            ...     read_datasets=["factor_lake"],
            ...     read_params={"factor_lake": {"factor_id": "mom_3d"}},
            ...     params=["2024-01-01"],
            ... )

        为什么要这一层：
            大部分临时分析需求（GROUP BY / window / 多表 JOIN）用标准 read_arrow
            表达不了，但我们又不想开放裸 read_parquet —— 路径白名单/审计/未来的
            配额限流就没着落了。这里通过 TEMP VIEW 收敛：你能读的表仅限 registry
            注册的，路径照旧走 PathAuthorizer，每次执行都记审计。
        """
        from data_access.read import sql_escape

        merged_budget = self._resolve_sql_budget(read_datasets, query_budget)
        return sql_escape.run_sql(
            registry=self._registry,
            authorizer=self._authorizer,
            engine=self._engine,
            query=query,
            read_datasets=read_datasets,
            read_params=read_params,
            read_time_ranges=read_time_ranges,
            view_columns=view_columns,
            params=params,
            query_budget=merged_budget,
            build_select_sql=self._build_select_sql,
            resolve_paths=self._resolve_paths_for_sql,
        )

    def compute_and_write(
        self,
        query: str,
        *,
        read_datasets: Sequence[str],
        write_dataset: str,
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
        view_columns: Mapping[str, Sequence[str]] | None = None,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
        **write_params: Any,
    ) -> dict[str, Any]:
        """读源数据（可 COS remote）→ SQL 运算 → 写入 staging/namespaced。

        **不修改** published / COS 源数据；结果只能落到 ``access_mode`` 为
        ``staging`` 或 ``namespaced`` 的登记数据集（例如 ``factor_lake_staging``）。

        典型用法（COS 直读 + 聚合 + 落 staging）::

            export DATA_ACCESS_COS_READ_MODE=remote
            # + COS 凭证 / endpoint

            store.compute_and_write(
                '''
                SELECT TradeDate AS datetime, Symbol AS asset, Close AS value
                FROM {{ashare_stock_daily}}
                ''',
                read_datasets=["ashare_stock_daily"],
                read_time_ranges={"ashare_stock_daily": ("2024-01-01", "2024-01-31")},
                write_dataset="factor_lake_staging",
                factor_id="close_raw_v1",
                mode="overwrite",
                partition_by=["year"],
                # 可选：自定义结果落盘根（或 write_dir= 指定最终目录）
                # write_root="/data/my_workspace/staging/factors",
            )
        """
        target = self._registry.get(write_dataset)
        if target.access_mode == "published":
            raise ValidationError(
                f"compute_and_write 禁止写 published 数据集 '{write_dataset}'；"
                f"请写 staging/namespaced（如 factor_lake_staging），"
                f"需要正式发布时再 publish_from_staging。"
            )
        table = self.sql(
            query,
            read_datasets=read_datasets,
            read_params=read_params,
            read_time_ranges=read_time_ranges,
            view_columns=view_columns,
            params=params,
            query_budget=query_budget,
        )
        result = self.write_arrow(
            write_dataset,
            table,
            mode=mode,
            partition_by=partition_by,
            **write_params,
        )
        result = dict(result)
        result["source_datasets"] = list(read_datasets)
        result["rows_computed"] = table.num_rows
        return result

    def sql_result(
        self,
        query: str,
        *,
        read_datasets: Sequence[str],
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
        view_columns: Mapping[str, Sequence[str]] | None = None,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
    ) -> SqlReadResult:
        """与 ``sql()`` 相同，但返回带合并 ``DataSnapshot`` 的 ``SqlReadResult``。"""
        from data_access.read import sql_escape

        params_map = dict(read_params or {})
        time_ranges = dict(read_time_ranges or {})
        path_by_ds: dict[str, list[str]] = {}
        snapshots: list[DataSnapshot] = []
        for name in sorted(read_datasets):
            ds = self._registry.get(name)
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_ranges.get(name),
                params=dict(params_map.get(name, {})),
            )
            path_by_ds[name] = paths
            snapshots.append(
                self._build_snapshot(
                    dataset=name,
                    ds=ds,
                    paths=paths,
                    params=dict(params_map.get(name, {})),
                )
            )
        snapshot = merge_sql_data_snapshots(
            snapshots,
            registry_hash=self._registry_hash,
        )

        def _resolve_cached(
            ds: Dataset,
            ds_params: dict[str, Any],
            *,
            time_range: tuple[Any, Any] | None = None,
            instrument_filter: Sequence[str] | None = None,
        ) -> list[str]:
            cached = path_by_ds.get(ds.name)
            if cached is not None:
                return cached
            return self._prepare_dataset_read(
                ds,
                time_range=time_range,
                params=ds_params,
                instrument_filter=instrument_filter,
            )

        merged_budget = self._resolve_sql_budget(read_datasets, query_budget)
        start = time.perf_counter()
        table = sql_escape.run_sql(
            registry=self._registry,
            authorizer=self._authorizer,
            engine=self._engine,
            query=query,
            read_datasets=read_datasets,
            read_params=read_params,
            read_time_ranges=read_time_ranges,
            view_columns=view_columns,
            params=params,
            query_budget=merged_budget,
            build_select_sql=self._build_select_sql,
            resolve_paths=_resolve_cached,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        lineage = SqlReadLineage(
            datasets=tuple(sorted(read_datasets)),
            read_params=tuple(
                sorted(
                    (name, tuple(sorted(params_map.get(name, {}).items())))
                    for name in read_datasets
                )
            ),
            query_preview=query[:500],
        )
        stats = ReadStats(
            rows=table.num_rows,
            bytes=table.nbytes,
            elapsed_ms=elapsed_ms,
        )
        return SqlReadResult(
            table=table,
            snapshot=snapshot,
            stats=stats,
            lineage=lineage,
        )

    def sql_stream(
        self,
        query: str,
        *,
        read_datasets: Sequence[str],
        read_params: Mapping[str, Mapping[str, Any]] | None = None,
        read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
        view_columns: Mapping[str, Sequence[str]] | None = None,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
        batch_size: int = 100_000,
    ) -> Iterator[pa.RecordBatch]:
        """流式有限 SQL：与 ``sql()`` 相同约束，按 RecordBatch 返回。"""
        from data_access.read import sql_escape

        merged_budget = self._resolve_sql_budget(read_datasets, query_budget)
        return sql_escape.run_sql_stream(
            registry=self._registry,
            authorizer=self._authorizer,
            engine=self._engine,
            query=query,
            read_datasets=read_datasets,
            read_params=read_params,
            read_time_ranges=read_time_ranges,
            view_columns=view_columns,
            params=params,
            query_budget=merged_budget,
            batch_size=batch_size,
            build_select_sql=self._build_select_sql,
            resolve_paths=self._resolve_paths_for_sql,
        )

    # ---- 内部 helpers ----

    def _resolve_paths_for_sql(
        self,
        ds: Dataset,
        params: dict[str, Any],
        *,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """sql()/sql_stream 路径解析：走 COS remote/mirror 与白名单。"""
        return self._prepare_dataset_read(
            ds,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
        )

    def _prepare_dataset_read(
        self,
        ds: Dataset,
        *,
        time_range: tuple[Any, Any] | None,
        params: dict[str, Any],
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """COS 镜像 / 远程直读 + 路径解析（read_arrow / stream / scan 共用）。

        若调用方显式传了 ``read_root``（或环境变量 DATA_ACCESS_READ_ROOT_*），
        则优先用本地覆盖路径，跳过 COS remote（适合「数据已在自选目录」）。
        """
        from data_access.cos.mirror import ensure_local_mirror_for_dataset
        from data_access.cos.remote import should_read_cos_remote

        peek = dict(params)
        explicit_read_root = (
            peek.get("read_root")
            or peek.get("_read_root")
            or dataset_env_root(ds.name, "read")
        )
        if explicit_read_root:
            return self._resolve_paths(ds, params, instrument_filter=instrument_filter)

        if should_read_cos_remote(ds, time_range=time_range):
            from data_access.cos.remote import prepare_cos_remote_paths

            paths, backend = prepare_cos_remote_paths(ds.name, time_range=time_range)
            read_params = dict(params)
            explicit_buckets = read_params.pop("bucket_values", None)
            for k in _READ_PATH_META_KEYS + _WRITE_PATH_META_KEYS:
                read_params.pop(k, None)
            hive_filters = self._bucket_hive_filters(
                ds,
                instrument_filter,
                bucket_values=explicit_buckets,
            )
            if hive_filters:
                from data_access.registry.layout_policy import prune_glob_paths_for_buckets

                bucket_col = next(iter(hive_filters))
                paths = prune_glob_paths_for_buckets(
                    paths,
                    bucket_col,
                    hive_filters[bucket_col],
                )
            paths = self._authorize_read_paths(paths)
            if backend == "httpfs":
                self._engine.ensure_s3_configured()
            logger.info(
                "cos_remote: dataset=%s paths=%d backend=%s",
                ds.name,
                len(paths),
                backend,
            )
            return paths

        ensure_local_mirror_for_dataset(ds, time_range=time_range)
        return self._resolve_paths(ds, params, instrument_filter=instrument_filter)

    def _authorize_read_paths(self, glob_paths: list[str]) -> list[str]:
        """本地/远程读路径白名单校验。"""
        from data_access.cos.remote import authorize_s3_path, cos_cache_root
        from data_access.registry.paths import path_is_under

        cache_root = cos_cache_root()
        for g in glob_paths:
            static_part = g.split("*", 1)[0].rstrip("/")
            if not static_part:
                continue
            if static_part.startswith("s3://"):
                authorize_s3_path(static_part)
                continue
            resolved = canonicalize(static_part)
            if path_is_under(resolved, cache_root):
                continue
            self._authorizer.resolve_and_authorize(resolved)
        return glob_paths

    @staticmethod
    def _bucket_hive_filters(
        ds: Dataset,
        instrument_filter: Sequence[str] | None,
        bucket_values: Sequence[int] | None = None,
    ) -> dict[str, list[Any]] | None:
        from data_access.registry.layout_policy import bucket_values_for_instruments

        layout_policy = getattr(ds, "layout_policy", None)
        if bucket_values is not None and layout_policy is not None and layout_policy.bucket is not None:
            return {layout_policy.bucket.column: sorted(int(b) for b in bucket_values)}
        buckets = bucket_values_for_instruments(
            instrument_filter,
            layout_policy,
            partition_columns=getattr(ds, "partition_columns", ()),
        )
        if not buckets or layout_policy is None or layout_policy.bucket is None:
            return None
        return {layout_policy.bucket.column: buckets}

    def _enforce_scan_files(
        self,
        budget: QueryBudget,
        paths: list[str],
        *,
        files: Sequence[FileVersion] | None = None,
    ) -> tuple[FileVersion, ...]:
        file_versions = tuple(files) if files is not None else build_file_manifest(paths)
        enforce_scan_file_budget(budget, file_count=len(file_versions))
        return file_versions

    def _ensure_schema(
        self,
        ds: Dataset,
        paths: list[str],
        params: dict[str, Any] | None = None,
        *,
        files: Sequence[FileVersion] | None = None,
    ) -> None:
        """首次访问 (dataset, params, manifest) 时做 schema 对齐校验。"""
        if not getattr(ds, "schema", None):
            return
        cache_key = self._schema_fingerprint(ds, paths, params or {}, files=files)
        with self._schema_check_lock:
            if cache_key in self._schema_checked:
                return

            result = check_schema(self._engine, ds, paths)
            if result.ok:
                self._schema_checked.add(cache_key)
                mark_validated(cache_key)
                return

            from data_access.registry.schema_validation import _resolve_mode

            mode = _resolve_mode()
            enforce_schema_or_raise(result, mode=mode)
            self._schema_checked.add(cache_key)
            if mode == "warn":
                mark_validated(cache_key)

    def _resolve_paths(
        self,
        ds: Dataset,
        params: dict[str, Any],
        *,
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """把 dataset + params 解析成传给 DuckDB 的 path glob 列表，并做白名单校验。

        可选 ``read_root`` / ``DATA_ACCESS_READ_ROOT_<NAME>``：覆盖数据集根路径。
        - static：``read_root / glob``
        - parametric：用 read_root 替换 ``static_root`` 前缀，保留参数后缀
        """
        from data_access.registry.layout_policy import prune_glob_paths_for_buckets

        read_params = dict(params)
        explicit_buckets = read_params.pop("bucket_values", None)
        for k in _WRITE_PATH_META_KEYS:
            read_params.pop(k, None)
        read_root = self._pop_read_root(ds, read_params)
        if read_root is not None:
            self._assert_path_override_allowed(ds, kind="read", value=read_root)

        if isinstance(ds, StaticDataset):
            root = Path(read_root) if read_root else ds.root
            glob_paths = [str(root / ds.glob)]
        else:
            glob_paths = ds.resolve_paths(**read_params)
            if read_root is not None:
                glob_paths = self._rewrite_root_prefix(
                    glob_paths,
                    old_static_root=ds.static_root,
                    new_root=Path(read_root),
                )

        hive_filters = self._bucket_hive_filters(
            ds,
            instrument_filter,
            bucket_values=explicit_buckets,
        )
        if hive_filters:
            bucket_col = next(iter(hive_filters))
            glob_paths = prune_glob_paths_for_buckets(
                glob_paths,
                bucket_col,
                hive_filters[bucket_col],
            )
        return self._authorize_read_paths(glob_paths)

    def _resolve_write_dir(self, ds: Dataset, params: dict[str, Any]) -> Path:
        """解析写入目标目录。

        优先级：
            1. ``write_dir`` / ``_write_dir`` —— 最终目录，完全覆盖
            2. ``write_root`` / ``DATA_ACCESS_WRITE_ROOT_<NAME>`` —— 替换数据集根
            3. datasets.yaml 模板默认路径

        例子：
            '/xx/runs/mom_3d/**/*.parquet'       → '/xx/runs/mom_3d'
            '/xx/factors/mom_3d/year=*/*.parquet' → '/xx/factors/mom_3d'
            write_dir='/data/out/x'               → '/data/out/x'
            write_root='/data/alt' + factor_id    → '/data/alt/<factor_id>'
        """
        path_meta, clean = self._split_write_params(dict(params))
        write_dir = path_meta.get("write_dir")
        write_root = path_meta.get("write_root") or dataset_env_root(ds.name, "write")

        if write_dir is not None:
            # write_dir 跳过路径模板，但仍校验 params_schema（如 factor_id）
            if isinstance(ds, ParametricDataset):
                specs = ds.param_specs or {
                    k: ParamSpec(name=k, type=t) for k, t in ds.params_schema.items()
                }
                validate_params(ds.name, specs, clean)
            return canonicalize(write_dir)

        if isinstance(ds, StaticDataset):
            if write_root is not None:
                return canonicalize(write_root)
            glob_paths = ds.resolve_paths()
        else:
            glob_paths = ds.resolve_paths(**clean)
            if write_root is not None:
                glob_paths = self._rewrite_root_prefix(
                    glob_paths,
                    old_static_root=ds.static_root,
                    new_root=Path(write_root),
                )

        if len(glob_paths) != 1:
            raise ValidationError(
                f"数据集 '{ds.name}' 解析出 {len(glob_paths)} 个 glob，"
                f"write_arrow 只支持唯一目标目录的数据集"
            )
        glob = glob_paths[0]
        segments = glob.split("/")
        clean_segments: list[str] = []
        for seg in segments:
            if "*" in seg or "?" in seg:
                break
            clean_segments.append(seg)
        if not clean_segments:
            raise ValidationError(f"数据集 '{ds.name}' 的 glob 无静态前缀: {glob}")
        return Path("/".join(clean_segments))

    @staticmethod
    def _clear_dir(path: Path) -> None:
        """overwrite 模式下清空目标目录。不存在就跳过，不递归到白名单外。"""
        if not path.exists():
            return
        if not path.is_dir():
            raise ValidationError(f"目标路径不是目录，不能 overwrite: {path}")
        # 逐项删（比 rmtree 稍慢但更安全，避免 symlink 逃逸）
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    @staticmethod
    def _write_table_to_dir(
        table: pa.Table,
        target_dir: Path,
        *,
        partition_by: Sequence[str] | None,
    ) -> list[Path]:
        """把 Arrow Table 写成 parquet；带分区用 pyarrow.dataset。

        不带 partition_by：写成 `{dir}/part-{uuid}.parquet` 单文件。
            append 时 uuid 保证不撞，overwrite 时目录已清空。
        带 partition_by：用 pyarrow.dataset.write_dataset 生成 hive 分区。
        """
        import pyarrow.parquet as pq

        target_dir.mkdir(parents=True, exist_ok=True)

        if partition_by:
            # 写分区：用 pyarrow.dataset 的 hive partitioning
            import pyarrow.dataset as pads

            # basename_template 必须含 "{i}"；用 uuid 前缀避免 append 时同分区下撞文件
            prefix = uuid.uuid4().hex[:8]
            pads.write_dataset(
                table,
                base_dir=str(target_dir),
                format="parquet",
                partitioning=list(partition_by),
                partitioning_flavor="hive",
                existing_data_behavior="overwrite_or_ignore",
                basename_template=f"part-{prefix}-{{i}}.parquet",
            )
            # 只返回本轮写入的文件（按 uuid 前缀），避免 append 时把旧文件算进来
            return sorted(target_dir.rglob(f"part-{prefix}-*.parquet"))

        # 单文件写：先写 .tmp 再 rename，保证原子
        out_name = f"part-{uuid.uuid4().hex[:8]}.parquet"
        out_path = target_dir / out_name
        tmp_path = target_dir / f".{out_name}.tmp"
        pq.write_table(table, tmp_path)
        os.replace(str(tmp_path), str(out_path))
        return [out_path]

    def _build_select_sql(
        self,
        *,
        ds: Dataset,
        paths: list[str],
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        limit: int | None = None,
    ) -> tuple[str, list[Any]]:
        """组装 SELECT 语句。路径用 ? 参数绑定，列名/谓词用 registry 控制。"""
        if columns:
            col_clause = ", ".join(_quote_ident(c) for c in columns)
        else:
            col_clause = "*"

        # read_parquet 的 path 参数：单路径直接 ?，多路径用 list
        # DuckDB 支持 read_parquet([...], hive_partitioning=..., union_by_name=...)
        # 注意 hive_partitioning / union_by_name 是 read_parquet 的命名参数，
        # 不能用 ? 占位符传（DuckDB SQL 语法限制），必须直接拼到 SQL 里
        # —— 这些值来自 registry，可信。
        path_param = paths if len(paths) > 1 else paths[0]

        read_parquet_opts = []
        if ds.hive_partitioning:
            read_parquet_opts.append("hive_partitioning=true")
        if ds.union_by_name:
            read_parquet_opts.append("union_by_name=true")
        opts_suffix = (", " + ", ".join(read_parquet_opts)) if read_parquet_opts else ""

        from_clause = f"read_parquet(?{opts_suffix})"

        time_col_type = str((ds.schema or {}).get(ds.time_column, "")).lower()
        predicate = Predicate(
            time_range=time_range,
            instrument_filter=instrument_filter,
            hive_filters=self._bucket_hive_filters(ds, instrument_filter),
            time_column_is_timestamp=("timestamp" in time_col_type or "datetime" in time_col_type),
        )
        compiled = compile_predicate(
            predicate,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
        )

        sql = f"SELECT {col_clause} FROM {from_clause} {compiled.where_sql}".strip()
        if limit is not None:
            sql = f"{sql} LIMIT {int(limit)}"
        params: list[Any] = [path_param, *compiled.params]
        return sql, params


# ---- 进程单例管理 ----

_store: DataAccessStore | None = None
_store_lock = threading.Lock()


def _compute_registry_hash(registry: DatasetRegistry) -> str:
    """登记表稳定指纹（dataset 名 + schema 声明）。"""
    payload: dict[str, Any] = {}
    for name in registry.names():
        ds = registry.get(name)
        payload[name] = {
            "kind": ds.kind,
            "schema": dict(getattr(ds, "schema", None) or {}),
        }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_store() -> DataAccessStore:
    """获取进程级 DataAccessStore 实例。第一次调用会初始化 registry + engine。

    线程安全：双重检查 + Lock。
    """
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        registry = load_registry()
        # 共享 DuckDBEngine：ParquetSource 也用同一个，buffer pool 复用
        engine = get_shared_engine()
        _store = DataAccessStore(registry=registry, engine=engine)
        logger.info(
            "DataAccessStore 初始化：%d 个数据集注册",
            len(registry.names()),
        )
        return _store


def adapter_options_for_dataset(ds: Dataset) -> dict[str, Any]:
    """按 datasets.yaml schema 推断 Arrow→MultiIndex 适配参数。

    - ``date`` 时间列：归一化到日（A 股日频）。
    - ``timestamp`` 时间列：保留完整精度（分钟/逐笔等多条同日内记录），
      否则会折叠成每 (日期, 标的) 一行。需要按日聚合的调用方自行
      ``normalize_timestamp=True``。
    """
    schema = getattr(ds, "schema", None) or {}
    if not schema:
        return {}
    time_dtype = str(schema.get(ds.time_column, "")).lower()
    opts: dict[str, Any] = {}
    if time_dtype == "date":
        opts["normalize_timestamp"] = True
    return opts


def reset_store() -> None:
    """主要给测试用。重置进程单例（会丢弃 DuckDB 连接里的缓存）。"""
    global _store
    with _store_lock:
        _store = None
    reset_shared_engine()
    reset_validated_cache()


def _quote_ident(name: str) -> str:
    """本地复制一份列名引用，避免 store 依赖 predicate 的私有函数。"""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _to_pydatetime(value: Any):
    """把 time_range 的端点统一成 python `datetime.datetime`，方便 Polars coerce。

    为什么：Polars 对 Datetime/Date 列不做 str->timestamp 自动 cast，
    直接 `pl.lit("2024-01-01")` 会报 InvalidOperationError。DuckDB 则会自动 cast，
    所以 read_arrow 这条路走字符串没问题，只有 scan_polars 需要转。

    支持的输入：
        - 字符串（走 pd.Timestamp 解析）
        - `datetime.datetime` / `pandas.Timestamp`：原样返回 .to_pydatetime()
        - `datetime.date`：直接返回（polars Date 列可以直接比）
        - int / float：视为 unix 秒或毫秒时戳是 DuckDB 专属语义，
          scan_polars 里不做猜测，原样丢回让 Polars 自己报错
    """
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        import pandas as _pd  # 本地 import：pd 只在这个分支用，避免拉顶层依赖

        return _pd.Timestamp(value).to_pydatetime()
    # 其它类型交给 polars，它会给一个更有信息量的错误
    return value
