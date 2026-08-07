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
    collect_polars_with_budget,
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
from data_access.read.read_handle import ReadHandle
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
    if ds.instrument_column is None:
        raise ValidationError(
            f"数据集 '{ds.name}' 未声明 instrument_column，不支持 instrument_filter；"
            "请在 datasets.yaml 声明 instrument_column 或 roles.instrument。"
        )
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

    def _files_for_snapshot(
        self, dsobj: Any, dataset: str, paths: list[str]
    ) -> tuple[Any, ...]:
        """构建 snapshot 的文件版本列表：优先复用 fresh manifest（省 O(N) stat）。

        只有 manifest 存在、新鲜且数据集名匹配时才走复用；否则回退
        ``build_file_manifest``（glob + stat）。``file_versions_from_manifest``
        只省 stat，路径展开仍需一次 glob。
        """
        from data_access.read.manifest import (
            DatasetManifest,
            is_manifest_fresh,
            manifest_root_for_paths,
        )
        from data_access.read.read_contract import file_versions_from_manifest

        root = manifest_root_for_paths(paths)
        if root is not None:
            try:
                manifest = DatasetManifest.load(root)
            except Exception:
                manifest = None
            if manifest is not None and manifest.dataset in {"", dataset}:
                try:
                    if is_manifest_fresh(manifest, paths):
                        return file_versions_from_manifest(manifest, paths)
                except Exception:
                    pass
        return build_file_manifest(paths)

    def _enforce_required_filters(
        self,
        fields_meta: Sequence[Any],
        params_by_dataset: Mapping[str, Mapping[str, Any]],
        *,
        filters: Any = None,
        filters_by_dataset: Mapping[str, Any] | None = None,
    ) -> None:
        """#8 ``required_filters`` 强制执行。

        catalog 字段声明了读取时必须附带的条件（如 index_weight 必须指定
        IndexSymbol、industry 字段必须指定 IndustrySource、美股财务必须 filter
        timeframe）时：缺少即 production fail-closed（抛 ValidationError），
        research 只告警。

        满足途径（任一即可）：
            1. params_by_dataset 里带该 key（路径参数）
            2. 全局 filters 或 filters_by_dataset 的过滤 AST 里覆盖该列
               （列过滤，如 timeframe='quarterly'）——#42 扩展
        """
        from data_access.read.predicate_ast import filter_columns, parse_filters

        missing: list[tuple[str, str]] = []
        global_cols = filter_columns(parse_filters(filters))
        per_ds_cols = {
            str(ds): filter_columns(parse_filters(f))
            for ds, f in (filters_by_dataset or {}).items()
        }
        for f in fields_meta:
            required = getattr(f, "required_filters", ()) or ()
            if not required:
                continue
            ds = getattr(f, "dataset", None)
            effective = dict((params_by_dataset or {}).get(ds, {}))
            filter_cols = set(global_cols) | set(per_ds_cols.get(ds, set()))
            for key in required:
                if key in effective:
                    continue
                # params 里大小写宽松匹配（IndexSymbol / index_symbol）
                lower_key = str(key).lower()
                if any(str(k).lower() == lower_key for k in effective):
                    continue
                # filters / filters_by_dataset 里按列覆盖
                if any(str(c).lower() == lower_key for c in filter_cols):
                    continue
                missing.append((getattr(f, "logical_name", "?"), str(key)))
        if not missing:
            return
        detail = "; ".join(f"'{n}' 需参数或过滤 {k}" for n, k in missing)
        if _production_mode():
            raise ValidationError(
                f"required_filters 未满足（production fail-closed）：{detail}"
            )
        logger.warning("required_filters 未满足（research 放行）：%s", detail)

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
        filters: Any = None,
        limit: int | None = None,
        query_budget: QueryBudget | None = None,
        **params: Any,
    ) -> ReadResult:
        """读取数据集并返回带 ``DataSnapshot`` 的 ``ReadResult``（审计/lineage 用）。"""
        ds = self._registry.get(dataset)
        return self._read_dataset_object(
            ds,
            dataset=dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            limit=limit,
            query_budget=query_budget,
            params=params,
        )

    def _read_dataset_object(
        self,
        ds: Dataset,
        *,
        dataset: str,
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        filters: Any,
        limit: int | None,
        query_budget: QueryBudget | None,
        params: dict[str, Any],
    ) -> ReadResult:
        """按已解析的 ``Dataset`` 对象执行一次读（read_result / read_uri 共用）。"""
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
            filters=filters,
            limit=limit,
        )

        start = time.perf_counter()
        ok = False
        err_msg: str | None = None
        table: pa.Table | None = None
        try:
            table = self._engine.execute_arrow(
                sql, sql_params, deadline_ms=budget.max_elapsed_ms
            )
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
        filters: Any = None,
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
            filters=filters,
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
        filters: Any = None,
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
            filters=filters,
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
        filters: Any = None,
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
            if time_range is not None and ds.time_column and ds.time_column not in needed_cols:
                needed_cols.append(ds.time_column)
            if (
                instrument_filter is not None
                and ds.instrument_column
                and ds.instrument_column not in needed_cols
            ):
                needed_cols.append(ds.instrument_column)
            lf = lf.select([pl_mod.col(c) for c in needed_cols])

        # Lazy filter：time_range + instrument_filter
        # 注意 Polars 不像 DuckDB 能把 '2024-01-01' 这种字符串自动 cast 成 datetime，
        # 对 datetime 列比较字符串会抛 InvalidOperationError。统一用 pandas.Timestamp
        # 把字符串 / datetime / date 归一化到 python datetime，再交给 pl.lit——
        # 这样无论下层是 Datetime 还是 Date 列，Polars 自己都能 coerce。
        if time_range is not None:
            if ds.time_column is None:
                raise ValidationError(
                    f"数据集 '{dataset}' 未声明 time_column，无法应用 time_range"
                )
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
            if ds.instrument_column is None:
                raise ValidationError(
                    f"数据集 '{dataset}' 未声明 instrument_column，无法应用 instrument_filter"
                )
            lf = lf.filter(pl_mod.col(ds.instrument_column).is_in(list(instrument_filter)))
        if filters is not None:
            from data_access.read.predicate_ast import compile_filter_polars, parse_filters

            expr = compile_filter_polars(parse_filters(filters), pl=pl_mod)
            if expr is not None:
                lf = lf.filter(expr)

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
        filters: Any = None,
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
            filters=filters,
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
        filters: Any = None,
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
            filters=filters,
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
        filters: Any = None,
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
            filters=filters,
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
        filters: Any = None,
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
        ds = self._registry.get(dataset)
        budget = self._resolve_read_budget(ds, query_budget)

        resolved_mode = str(mode or "auto").lower()
        if resolved_mode == "auto":
            # 成本路由（rows/bytes/columns/files/remote/selectivity），不只行数
            from data_access.read.scan_cost import (
                estimate_scan_cost,
                suggest_read_strategy,
            )

            cost = estimate_scan_cost(
                self,
                dataset,
                columns=list(columns) if columns else None,
                time_range=time_range,
                instrument_filter=instrument_filter,
                prefer_polars=prefer_polars,
                **params,
            )
            engine, result = suggest_read_strategy(
                cost, prefer_polars=prefer_polars, engine="auto", result="auto"
            )
            if engine == "polars":
                resolved_mode = "polars"
            elif result == "stream":
                resolved_mode = "stream"
            else:
                resolved_mode = "arrow"
            log_read_auto(dataset, cost, engine, result)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )

        read_kwargs = dict(
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            query_budget=query_budget,
            **params,
        )
        if resolved_mode == "polars":
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
        filters: Any = None,
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
        ds = self._registry.get(dataset)
        budget = self._resolve_read_budget(ds, query_budget)

        resolved_mode = str(mode or "auto").lower()
        if resolved_mode == "auto":
            from data_access.read.scan_cost import (
                estimate_scan_cost,
                suggest_read_strategy,
            )

            cost = estimate_scan_cost(
                self,
                dataset,
                columns=list(columns) if columns else None,
                time_range=time_range,
                instrument_filter=instrument_filter,
                prefer_polars=prefer_polars,
                **params,
            )
            engine, result = suggest_read_strategy(
                cost, prefer_polars=prefer_polars, engine="auto", result="stream"
            )
            resolved_mode = "polars" if engine == "polars" else "stream"
            log_read_auto(dataset, cost, engine, result)
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
            filters=filters,
            query_budget=query_budget,
            **params,
        )
        yield from self.read_arrow_stream(
            dataset,
            batch_size=batch_size,
            limit=limit,
            **read_kwargs,
        )

    # ---- 统一 read() / read_uri()（返回 ReadHandle） ----

    def read(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        filters: Any = None,
        limit: int | None = None,
        engine: str = "auto",
        result: str = "auto",
        prefer_polars: bool = False,
        batch_size: int = 100_000,
        query_budget: QueryBudget | None = None,
        normalize_units: bool = False,
        **params: Any,
    ) -> ReadHandle:
        """统一读入口：引擎/结果形态自动路由，返回 ``ReadHandle``。

        - ``engine``: auto | duckdb | polars | pyarrow
        - ``result``: auto | arrow | pandas | polars | lazy | stream
        auto 时按 `estimated_scan_cost`（rows/bytes/columns/files/remote/selectivity）
        路由，而不是只看行数。
        - ``normalize_units``: 按 SemanticFieldCatalog 的 scale 做输出层单位归一化
          （percent→ratio 等），默认 False 保持旧行为逐字节不变。

        返回 ``ReadHandle``，支持 ``.to_arrow() / .to_pandas() / .to_polars() /
        .to_lazy() / .stream()``。
        """
        ds = self._registry.get(dataset)
        return self._read_handle(
            ds,
            dataset=dataset,
            registered_name=dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            limit=limit,
            engine=engine,
            result=result,
            prefer_polars=prefer_polars,
            batch_size=batch_size,
            query_budget=query_budget,
            params=params,
            normalize_units=normalize_units,
        )

    def read_uri(
        self,
        uri: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        filters: Any = None,
        limit: int | None = None,
        format: str = "auto",
        engine: str = "auto",
        result: str = "auto",
        prefer_polars: bool = False,
        batch_size: int = 100_000,
        query_budget: QueryBudget | None = None,
        time_column: str | None = None,
        instrument_column: str | None = None,
        normalize_units: bool = False,
        **kwargs: Any,
    ) -> ReadHandle:
        """直接读一个 URI（本地路径或 cos:// / s3://），不必先登记数据集。

        **开发环境**：URI 静态前缀须在 PathAuthorizer 白名单根（已登记数据集根 +
        DATA_ACCESS_EXTRA_ALLOWED_ROOTS + DATA_ACCESS_READ_URI_ROOTS）下。
        **production / strict 读模式**：只允许落在已登记数据集根下，任意 URI 被拒。

        ``format`` 不传时按扩展名推断（parquet/csv/tsv/jsonl/arrow/feather）。
        arrow/feather 自动走 PyArrow 引擎。
        """
        from data_access.read.formats import normalize_format_name

        uri = str(uri)  # 兼容 Path 对象
        fmt = _infer_format_from_uri(uri, format)
        self._assert_uri_allowed(uri, format=fmt)
        ds = self._uri_dataset(
            uri,
            format=fmt,
            time_column=time_column,
            instrument_column=instrument_column,
        )
        return self._read_handle(
            ds,
            dataset=f"<uri:{uri[:80]}>",
            registered_name=None,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            limit=limit,
            engine=engine,
            result=result,
            prefer_polars=prefer_polars,
            batch_size=batch_size,
            query_budget=query_budget,
            params=kwargs,
            normalize_units=normalize_units,
        )

    # ---- 集成层：DataRequest/ReadPlan / read_joined / RelationHandle ----

    def read_joined(
        self,
        anchor: str,
        fields: Any,
        *,
        joins: Mapping[str, Any] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        filters: Any = None,
        filters_by_dataset: Mapping[str, Any] | None = None,
        limit: int | None = None,
        engine: str = "duckdb",
        result: str = "auto",
        normalize_units: bool = False,
        seed_window: bool = True,
        universe: str | None = None,
        time_varying_universe: bool = True,
        query_budget: QueryBudget | None = None,
        params: Mapping[str, Any] | None = None,
        params_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ReadHandle:
        """多数据集批量 join：每张物理表只扫一次，DuckDB 内 exact / PIT-asof join。

        参数：
            anchor: 锚定数据集（其时间/标的列决定输出行空间）
            fields: ``{dataset: [物理列]}`` mapping，或 ``"col"`` / ``"dataset.col"`` 序列
            joins: ``{dataset: 规格}``；规格可以是 ``exact|asof|pit_asof`` 字符串、
                dict 或 ``TemporalJoinSpec``。缺省 exact。
                - exact   ：按 (time, instrument) 等值对齐（日频面板）
                - pit_asof：对每个 anchor 行取该标的最新可见记录（asof）
                    - ``TemporalJoinSpec(knowledge_time=..., availability=...,
                      revision_order=...)`` 给出语义级 PIT（PubDate 可见性、
                      下一交易日可用、按版本去重）
            filters_by_dataset: ``{dataset: filters}`` 每数据集独立过滤
            normalize_units: 输出层按 SemanticFieldCatalog scale 归一化单位
            seed_window: asof/pit_asof 右表走 seed+window（默认 True），替代全历史
                扫描——窗口 [start,end] + 每标的 start 前最后一条可见记录，UNION
                后 ASOF，语义与全历史逐字节一致
            params: anchor 数据集的参数；params_by_dataset 可为每个数据集分别指定

        一次 DuckDB 查询完成：projection pushdown + 分区裁剪 + join，返回 ReadHandle。
        预算/审计/snapshot 与普通 read 一致（多数据集合并 snapshot）。
        """
        from data_access.read.read_contract import (
            ReadStats,
            SqlReadLineage,
            file_versions_from_manifest,
            merge_sql_data_snapshots,
        )
        from data_access.read.read_handle import ReadHandle
        from data_access.read.semantic_catalog import normalize_table_units
        from data_access.read.temporal_join import parse_join_spec

        if engine in {"pyarrow", "polars"}:
            raise ValidationError(
                "read_joined 在 DuckDB 内完成 join，engine 必须为 duckdb"
            )
        self._registry.get(anchor)
        per_ds, fields_meta, default_joins = self._normalize_joined_fields(anchor, fields)
        # 显式 joins 优先；未指定时用 catalog 字段推导的语义 join（#3）
        effective_joins = dict(joins or {})
        for ds, spec in default_joins.items():
            effective_joins.setdefault(ds, spec)
        joins_map: dict[str, str] = {}
        for ds in per_ds:
            joins_map[ds] = parse_join_spec(
                effective_joins.get(ds)
            ).policy
        joins_map[anchor] = "exact"

        pbd: dict[str, dict[str, Any]] = {}
        for ds in per_ds:
            pbd[ds] = dict((params_by_dataset or {}).get(ds, {}))
        pbd.setdefault(anchor, dict(params or {}))

        # #8 required_filters 强制执行（production fail-closed，research warning）
        # #42：同时检查 params 与 filters/filters_by_dataset 列覆盖（美股财务
        # timeframe、行业 IndustrySource 是列过滤不是路径参数）。
        self._enforce_required_filters(
            fields_meta,
            pbd,
            filters=filters,
            filters_by_dataset=filters_by_dataset,
        )

        merged = self._resolve_sql_budget(list(per_ds), query_budget)
        all_cols = [c for cols in per_ds.values() for c in cols]
        validate_query_request(merged, columns=all_cols or None, time_range=time_range)

        sql, sql_params, datasets, per_ds_paths = self._read_joined_sql(
            anchor,
            per_ds,
            effective_joins,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            filters_by_dataset=filters_by_dataset,
            params_by_dataset=pbd,
            limit=limit,
            seed_window=seed_window,
            universe=(universe if time_varying_universe else None),
        )

        snapshots = []
        for ds in datasets:
            dsobj = self._registry.get(ds)
            try:
                paths = per_ds_paths.get(ds) or []
                files = self._files_for_snapshot(dsobj, ds, paths)
                self._enforce_scan_files(merged, paths, files=files)
                snapshots.append(
                    self._build_snapshot(
                        dataset=ds,
                        ds=dsobj,
                        paths=paths,
                        params=pbd.get(ds, {}),
                        files=files,
                    )
                )
            except Exception:
                continue
        snapshot = (
            merge_sql_data_snapshots(snapshots, registry_hash=self.registry_fingerprint())
            if snapshots
            else None
        )
        lineage = SqlReadLineage(datasets=tuple(datasets), query_preview=sql[:200])

        start = time.perf_counter()
        ok = False
        err_msg: str | None = None
        table: pa.Table | None = None
        try:
            table = self._engine.execute_arrow(
                sql, sql_params, deadline_ms=merged.max_elapsed_ms
            )
            if normalize_units and fields_meta:
                table = normalize_table_units(table, fields_meta)
            elapsed_ms = (time.perf_counter() - start) * 1000
            enforce_arrow_budget(merged, table, elapsed_ms=elapsed_ms)
            ok = True
            stats = ReadStats(
                rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms
            )
            logger.info(
                "read_joined anchor=%s datasets=%s rows=%d cols=%d elapsed_ms=%.1f",
                anchor,
                datasets,
                table.num_rows,
                table.num_columns,
                elapsed_ms,
            )
            return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            audit.record(
                op="read_joined",
                dataset=anchor,
                ok=ok,
                rows=table.num_rows if table is not None else None,
                params=dict(params or {}),
                elapsed_ms=elapsed_ms,
                error=err_msg,
                extra={
                    "datasets": list(datasets),
                    "columns": all_cols,
                    "joins": joins_map,
                },
            )

    def sql_relation(
        self,
        sql: str,
        *,
        params: Sequence[Any] | None = None,
        query_budget: QueryBudget | None = None,
        snapshot_datasets: Sequence[str] | None = None,
        snapshot_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Any:
        """受控 SQL Relation 句柄：在 DataAccess scan 上追加表达式，collect 时强制治理。

        返回 ``RelationHandle``：
            - ``.sql("SELECT AVG(x) FROM _sub GROUP BY _sub.asset")`` 追加表达式
            - ``.arrow() / .collect() / .pandas()`` 强制 QueryBudget(deadline) + audit
            - ``.relation`` 只读底层 DuckDB Relation（schema/explain 检查用）

        ``snapshot_datasets`` 提供后，collect 绑定合并的 DataSnapshot（lineage 复现用）。
        """
        from data_access.read.relation_handle import RelationHandle

        snapshot = lineage = None
        if snapshot_datasets:
            from data_access.read.read_contract import (
                SqlReadLineage,
                merge_sql_data_snapshots,
            )

            snapshots = []
            for ds in snapshot_datasets:
                try:
                    dsobj = self._registry.get(ds)
                    ds_params = dict((snapshot_params or {}).get(ds, {}))
                    paths = self._prepare_dataset_read(
                        dsobj,
                        time_range=None,
                        params=ds_params,
                        instrument_filter=None,
                    )
                    files = build_file_manifest(paths)
                    snapshots.append(
                        self._build_snapshot(
                            dataset=ds,
                            ds=dsobj,
                            paths=paths,
                            params=ds_params,
                            files=files,
                        )
                    )
                except Exception:
                    continue
            if snapshots:
                snapshot = merge_sql_data_snapshots(
                    snapshots, registry_hash=self.registry_fingerprint()
                )
                lineage = SqlReadLineage(
                    datasets=tuple(snapshot_datasets), query_preview=sql[:200]
                )
        return RelationHandle(
            self,
            sql,
            params=params,
            query_budget=query_budget,
            snapshot=snapshot,
            lineage=lineage,
        )

    def resolve_fields(
        self,
        names: Sequence[str],
        *,
        dataset: str | None = None,
    ) -> list[Any]:
        """逻辑字段 → 物理字段解析（SemanticFieldCatalog 单一事实源 + registry 回退）。

        每个名字先查 SemanticFieldCatalog（含 aliases）；不在 catalog 就回退：
        显式 dataset 的 schema，否则全 registry 里第一个含该列的 dataset；都找不到
        抛 ValidationError。
        """
        from data_access.read.semantic_catalog import SemanticField, get_semantic_catalog

        catalog = get_semantic_catalog()
        out: list[Any] = []
        for name in names:
            # #41 跨市场：传 dataset 让 catalog 按市场消歧（A股 return_bp 的 alias
            # 'ret' 与美股独立字段 'ret' 不再串味）。
            f = catalog.resolve_one(name, dataset=dataset)
            if f is not None:
                out.append(f)
                continue
            if dataset:
                # 调用方传物理列时，反查 catalog 拿到单位/语义定义（scale 归一化）
                by_phys = catalog.resolve_by_physical(dataset, name)
                if by_phys is not None:
                    out.append(by_phys)
                    continue
                ds = self._registry.get(dataset)
                schema = getattr(ds, "schema", None) or {}
                if name in schema:
                    out.append(
                        SemanticField(
                            logical_name=name,
                            dataset=dataset,
                            physical_name=name,
                            dtype=schema[name],
                        )
                    )
                    continue
            found: tuple[str, str] | None = None
            for dsn in self._registry.names():
                d = self._registry.get(dsn)
                schema = getattr(d, "schema", None) or {}
                if name in schema:
                    found = (dsn, schema[name])
                    break
            if found is None:
                raise ValidationError(
                    f"字段 '{name}' 未在 SemanticFieldCatalog，也不在任何数据集 schema 中。"
                    f"请先在 config/semantic_fields.yaml 登记逻辑字段。"
                )
            out.append(
                SemanticField(
                    logical_name=name, dataset=found[0], physical_name=name, dtype=found[1]
                )
            )
        return out

    def plan(self, request: Any) -> Any:
        """把 DataRequest 编译成 ReadPlan（不执行）。``ReadPlan.explain()`` 看计划，
        ``ReadPlan.execute()`` 执行（单数据集走 read，多数据集走 read_joined）。

        ``request`` 可以是 ``DataRequest`` 实例或等价 dict。
        """
        from data_access.core.storage import storage_description
        from data_access.read.data_request import (
            DataRequest,
            ReadPlan,
            normalize_join_policy,
        )
        from data_access.read.scan_cost import (
            estimate_scan_cost,
            suggest_read_strategy,
        )
        from data_access.read.semantic_catalog import SemanticField

        if isinstance(request, dict):
            request = DataRequest(**request)
        if not isinstance(request, DataRequest):
            raise ValidationError("plan 需要 DataRequest 实例或等价 dict")

        fields: list[SemanticField] = []
        for raw in request.fields:
            if isinstance(raw, str) and "." in raw:
                ds, col = raw.split(".", 1)
                fields.append(
                    SemanticField(logical_name=col, dataset=ds, physical_name=col)
                )
            else:
                fields.extend(self.resolve_fields([raw], dataset=request.anchor))

        anchor = request.anchor
        if anchor is None:
            dsets = {f.dataset for f in fields}
            if len(dsets) == 1:
                anchor = next(iter(dsets))
            else:
                raise ValidationError("多数据集 DataRequest 必须显式指定 anchor")
        request.anchor = anchor

        per_ds: dict[str, list[str]] = {}
        for f in fields:
            per_ds.setdefault(f.dataset, [])
            if f.physical_name not in per_ds[f.dataset]:
                per_ds[f.dataset].append(f.physical_name)
        datasets = [anchor] + [d for d in per_ds if d != anchor]

        from data_access.read.temporal_join import parse_join_spec

        raw_joins: dict[str, Any] = {}
        raw_joins.update(dict(request.joins or {}))
        raw_joins.update(dict(request.join_specs or {}))
        joins: dict[str, str] = {}
        for ds in datasets:
            if ds == anchor:
                joins[ds] = "exact"
            elif ds in raw_joins:
                joins[ds] = parse_join_spec(raw_joins[ds]).policy
            else:
                joins[ds] = "exact"

        scan_costs: dict[str, Any] = {}
        storage: dict[str, str] = {}
        snapshot_info: dict[str, dict[str, Any]] = {}
        for ds in datasets:
            dsobj = self._registry.get(ds)
            try:
                cost = estimate_scan_cost(
                    self,
                    ds,
                    columns=per_ds.get(ds) or None,
                    time_range=request.time_range,
                    instrument_filter=request.instruments,
                )
            except Exception:
                cost = None
            scan_costs[ds] = cost
            storage[ds] = storage_description(dsobj)
            try:
                snapshot_info[ds] = self.manifest_version(ds)
            except Exception:
                snapshot_info[ds] = {"has_manifest": False}

        engine, result = request.engine, request.result
        if len(datasets) > 1:
            engine = "duckdb"
            if result == "auto":
                result = "arrow"
        elif engine == "auto" or result == "auto":
            cost = scan_costs.get(anchor)
            if cost is not None:
                engine, result = suggest_read_strategy(
                    cost, engine=request.engine, result=request.result
                )
            else:
                engine = "duckdb"
                if result == "auto":
                    result = "arrow"

        return ReadPlan(
            request=request,
            datasets=datasets,
            fields=fields,
            per_dataset_columns=per_ds,
            join_policies=joins,
            scan_costs=scan_costs,
            storage=storage,
            snapshot_info=snapshot_info,
            engine=engine,
            result=result,
            time_range=request.time_range,
            instruments=request.instruments,
            universe=request.universe,
            _store=self,
        )

    def manifest_version(self, dataset: str, **params: Any) -> dict[str, Any]:
        """廉价的 query-scoped snapshot token：只读 ``_manifest.json`` sidecar，
        不做全量 footer 扫描，也不做 O(N) 文件 glob（#17）。

        freshness 由写路径维护的 ``manifest_epoch`` 决定：sidecar 存在即信任
        （数据有变更时写路径会 bump epoch）。返回 ``{has_manifest, fresh,
        dataset_version, partition_version, file_count, manifest_epoch,
        created_at}``。
        """
        from data_access.read.manifest import (
            manifest_root_for_paths,
            manifest_version_token,
        )

        ds = self._registry.get(dataset)
        try:
            paths = self._resolve_raw_paths(ds, time_range=None, params=params)
        except Exception:
            return {"dataset": dataset, "has_manifest": False}
        root = manifest_root_for_paths(paths)
        if root is None:
            return {"dataset": dataset, "has_manifest": False}
        token = manifest_version_token(root)
        if token is None:
            return {"dataset": dataset, "has_manifest": False}
        return {
            "dataset": dataset,
            "has_manifest": True,
            "fresh": True,
            "dataset_version": token.get("dataset_version"),
            "partition_version": token.get("partition_version"),
            "file_count": token.get("file_count"),
            "manifest_epoch": token.get("manifest_epoch"),
            "created_at": token.get("created_at"),
        }

    def touch_manifest_epoch(self, dataset: str, **params: Any) -> str | None:
        """写路径 mutation 后调用：递增 manifest epoch（O(1)，不重建 manifest）。

        这样 query-scoped snapshot 在数据变更后立刻判定过期，无需在 read path
        做 glob 计数。返回新 epoch；无 manifest 返回 None。
        """
        from data_access.read.manifest import (
            bump_manifest_epoch,
            manifest_root_for_paths,
        )

        ds = self._registry.get(dataset)
        try:
            paths = self._resolve_raw_paths(ds, time_range=None, params=params)
        except Exception:
            return None
        root = manifest_root_for_paths(paths)
        if root is None:
            return None
        return bump_manifest_epoch(root)

    def is_snapshot_stale(
        self,
        dataset: str,
        *,
        dataset_version: str | None = None,
        partition_version: str | None = None,
        manifest_epoch: str | None = None,
        **params: Any,
    ) -> bool:
        """对比 token 判断缓存/快照是否过期（无 manifest → 视为过期）。

        支持 ``manifest_epoch`` 对比（写路径 bump 后立刻失效）；无 epoch 时
        退回 dataset_version/partition_version 对比（老 manifest 兼容）。
        """
        cur = self.manifest_version(dataset, **params)
        if not cur.get("has_manifest"):
            return True
        if manifest_epoch is not None:
            return cur.get("manifest_epoch") != manifest_epoch
        if dataset_version is not None and cur.get("dataset_version") != dataset_version:
            return True
        if partition_version is not None and cur.get("partition_version") != partition_version:
            return True
        return False

    # ---- 私有 helpers ----

    def _normalize_joined_fields(
        self, anchor: str, fields: Any
    ) -> tuple[dict[str, list[str]], list[Any], dict[str, Any]]:
        """read_joined 的 fields 归一化。

        返回 ``( {dataset: [物理列]}, 字段列表, 由 catalog 推导的每数据集默认
        join spec )``。默认 join spec 来自字段的 join_policy/knowledge_time/
        revision_order（#3 语义级 PIT）；多字段冲突时非 exact 优先。
        """
        from data_access.read.semantic_catalog import SemanticField, get_semantic_catalog
        from data_access.read.temporal_join import join_spec_from_field

        catalog = get_semantic_catalog()
        fields_meta: list[Any] = []
        default_joins: dict[str, Any] = {}

        def _note_default(f: Any) -> None:
            if f is None or f.dataset is None:
                return
            spec = join_spec_from_field(f)
            if spec is None or spec.policy == "exact":
                return
            cur = default_joins.get(f.dataset)
            if cur is None or cur.policy != "pit_asof":
                default_joins[f.dataset] = spec

        if isinstance(fields, Mapping):
            per_ds: dict[str, list[str]] = {}
            for ds, cols in fields.items():
                per_ds[str(ds)] = [str(c) for c in (cols or [])]
            for ds, cols in per_ds.items():
                for c in cols:
                    f = catalog.resolve_one(c, dataset=ds)
                    if f is None:
                        f = catalog.resolve_by_physical(ds, c)
                    fields_meta.append(
                        f
                        if f is not None
                        else SemanticField(logical_name=c, dataset=ds, physical_name=c)
                    )
                    _note_default(f)
            return per_ds, fields_meta, default_joins
        per_ds = {anchor: []}
        for raw in fields:
            col = str(raw)
            if "." in col:
                ds, physical = col.split(".", 1)
                per_ds.setdefault(ds, [])
                if physical not in per_ds[ds]:
                    per_ds[ds].append(physical)
                fields_meta.append(
                    SemanticField(logical_name=physical, dataset=ds, physical_name=physical)
                )
                continue
            f = catalog.resolve_one(col, dataset=anchor)
            if f is not None:
                per_ds.setdefault(f.dataset, [])
                if f.physical_name not in per_ds[f.dataset]:
                    per_ds[f.dataset].append(f.physical_name)
                fields_meta.append(f)
                _note_default(f)
            else:
                if col not in per_ds[anchor]:
                    per_ds[anchor].append(col)
                fields_meta.append(
                    SemanticField(logical_name=col, dataset=anchor, physical_name=col)
                )
        return per_ds, fields_meta, default_joins

    def _read_joined_sql(
        self,
        anchor: str,
        per_ds: Mapping[str, Sequence[str]],
        join_specs: Mapping[str, Any],
        *,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        filters: Any,
        filters_by_dataset: Mapping[str, Any] | None = None,
        params_by_dataset: Mapping[str, Mapping[str, Any]],
        limit: int | None = None,
        seed_window: bool = True,
        universe: str | None = None,
    ) -> tuple[str, list[Any], list[str], dict[str, list[str]]]:
        """生成 read_joined 的单条 SQL：每张物理表一个子查询，DuckDB 内 join。

        相对旧实现的关键变化：
            1. **右表 instrument_filter 下推**：join key 同为 instrument 轴的
               右表（exact / asof / pit_asof）都用同一 instrument_filter 剪
               文件 + WHERE，避免「锚点 100 只、右表扫全 A 5000 只」。
            2. **显式 join 列**：用 ``TemporalJoinSpec`` 的 decision_time /
               knowledge_time 分别绑定锚点/右表时间列，不再假设两边列名相同
               （修复 a.ticker = b.ticker 但 a 无 ticker 的跨表列名 bug）。
            3. **PIT seed + window**：asof/pit_asof 右表不再扫全历史——
               ``[start, end]`` 内窗口 + 每标的 start 前最后一条可见记录的
               seed，UNION 后 ASOF，语义与全历史 ASOF 逐字节一致。
            4. **revision 去重**：``TemporalJoinSpec.revision_order`` 提供后，
               join 前按 (instrument, time) QUALIFY 保留最新一版，替代依赖
               parquet 扫描顺序的 keep_last。
        """
        from data_access.read.formats import format_adapter_for_dataset
        from data_access.read.predicate import Predicate, compile_predicate
        from data_access.read.predicate_ast import parse_filters
        from data_access.read.temporal_join import TemporalJoinSpec, parse_join_spec

        datasets = [anchor] + [d for d in per_ds if d != anchor]
        specs: dict[str, Any] = {
            ds: parse_join_spec(join_specs.get(ds) if join_specs else None)
            for ds in datasets
        }
        specs[anchor] = TemporalJoinSpec(policy="exact")
        outer_cols: list[str] = []
        params_list: list[Any] = []
        seen_out: set[str] = set()
        join_clauses: list[str] = []
        anchor_sub: str | None = None
        per_ds_paths: dict[str, list[str]] = {}
        anchor_meta = self._registry.get(anchor)
        anchor_time = anchor_meta.time_column
        anchor_inst = anchor_meta.instrument_column
        for key in (anchor_time, anchor_inst):
            if key and key not in seen_out:
                seen_out.add(key)
                outer_cols.append(f"a.{_quote_ident(key)} AS {_quote_ident(key)}")

        def _build_branch_sub(
            dsobj: Any,
            adapter: Any,
            ds_params: dict[str, Any],
            select_cols: Sequence[str],
            *,
            branch_tr: tuple[Any, Any] | None,
            time_lower_exclusive: bool,
            time_upper_exclusive: bool,
            time_col: str,
            inst_col: str,
            hive_filters: Any,
            ds_filters: Any,
            time_is_ts: bool,
        ) -> tuple[str, list[Any], list[str]]:
            """构建单个分支子查询；返回 (sql, branch_params, paths)。

            instrument_filter 由外层闭包 ``ds_inst`` 提供（锚点/右表一致下推）。
            """
            select_list = ", ".join(_quote_ident(c) for c in select_cols)
            paths = self._prepare_dataset_read(
                dsobj,
                time_range=branch_tr,
                params=ds_params,
                instrument_filter=ds_inst,
            )
            if not paths:
                # #21 manifest 空裁剪：生成带 schema 的空 SELECT，不回退全量扫描
                return _empty_branch_sql(select_cols, dsobj), [], []
            path_param = paths if len(paths) > 1 else paths[0]
            from_clause = adapter.build_from_clause(
                path_param,
                hive_partitioning=dsobj.hive_partitioning,
                union_by_name=dsobj.union_by_name,
            )
            pred = Predicate(
                time_range=branch_tr,
                instrument_filter=ds_inst,
                hive_filters=hive_filters,
                filters=ds_filters,
                time_column_is_timestamp=time_is_ts,
                time_lower_exclusive=time_lower_exclusive,
                time_upper_exclusive=time_upper_exclusive,
            )
            compiled = compile_predicate(
                pred, time_column=time_col, instrument_column=inst_col
            )
            sql = (
                f"SELECT {select_list} FROM {from_clause} {compiled.where_sql}".strip()
            )
            return sql, [path_param, *compiled.params], paths

        for i, ds in enumerate(datasets):
            alias = "a" if i == 0 else chr(ord("b") + (i - 1))
            dsobj = self._registry.get(ds)
            spec = specs[ds]
            policy = spec.policy
            cols = list(per_ds.get(ds, []))
            t_col = dsobj.time_column
            inst_col = dsobj.instrument_column
            # join 用锚点 decision_time 与右表 knowledge_time（可不同于各自 time_column）
            right_time = spec.effective_knowledge_time(t_col)
            decision_time = spec.effective_decision_time(anchor_time)
            if not inst_col:
                raise ValidationError(
                    f"read_joined 数据集 '{ds}' 未声明 instrument_column，无法 join"
                )
            if not t_col and policy != "exact":
                raise ValidationError(
                    f"asof join 需要数据集 '{ds}' 声明 time_column"
                )

            select_cols = list(cols)
            for k in (t_col, inst_col, right_time, *spec.revision_order):
                if k and k not in select_cols:
                    select_cols.append(k)
            select_list = ", ".join(_quote_ident(c) for c in select_cols)

            ds_params = dict(params_by_dataset.get(ds, {}))
            # 锚点通用 filters + 每数据集 filters 合并
            ds_filters = _and_filters(
                parse_filters(filters) if ds == anchor else None,
                parse_filters(
                    (filters_by_dataset or {}).get(ds) if filters_by_dataset else None
                ),
            )
            # instrument_filter 对锚点/右表一律下推（join key 同为 instrument 轴）
            ds_inst = instrument_filter
            hive = self._bucket_hive_filters(dsobj, ds_inst)
            time_type = (
                str((dsobj.schema or {}).get(t_col or "", "")).lower() if t_col else ""
            )
            time_is_ts = "timestamp" in time_type or "datetime" in time_type

            if ds == anchor or policy == "exact":
                branch_tr = time_range
                branch = _build_branch_sub(
                    dsobj,
                    format_adapter_for_dataset(dsobj),
                    ds_params,
                    select_cols,
                    branch_tr=branch_tr,
                    time_lower_exclusive=False,
                    time_upper_exclusive=False,
                    time_col=t_col,
                    inst_col=inst_col,
                    hive_filters=hive,
                    ds_filters=ds_filters,
                    time_is_ts=time_is_ts,
                )
                sub, branch_params, paths = branch
                params_list.extend(branch_params)
                per_ds_paths[ds] = paths
                if (
                    spec.deduplicate
                    and spec.revision_order
                    and inst_col
                    and t_col
                ):
                    sub = _dedup_key_sql(sub, inst_col, t_col, spec.revision_order)
            else:
                # asof / pit_asof：seed + window
                all_paths: list[str] = []
                branches: list[str] = []
                if seed_window and time_range is not None and time_range[0] is not None:
                    start, end = time_range
                    # 窗口分支 [start, end]（按数据集 time_column 裁剪文件）
                    win = _build_branch_sub(
                        dsobj,
                        format_adapter_for_dataset(dsobj),
                        ds_params,
                        select_cols,
                        branch_tr=(start, end),
                        time_lower_exclusive=False,
                        time_upper_exclusive=False,
                        time_col=right_time,
                        inst_col=inst_col,
                        hive_filters=hive,
                        ds_filters=ds_filters,
                        time_is_ts=time_is_ts,
                    )
                    branches.append(win[0])
                    params_list.extend(win[1])
                    all_paths.extend(win[2])
                    # seed 分支：每标的 start 前最后一条可见记录
                    seed = _build_branch_sub(
                        dsobj,
                        format_adapter_for_dataset(dsobj),
                        ds_params,
                        select_cols,
                        branch_tr=(None, start),
                        time_lower_exclusive=False,
                        time_upper_exclusive=True,
                        time_col=right_time,
                        inst_col=inst_col,
                        hive_filters=hive,
                        ds_filters=ds_filters,
                        time_is_ts=time_is_ts,
                    )
                    seed_sql = _seed_qualify_sql(
                        seed[0], inst_col, right_time, spec.revision_order
                    )
                    branches.append(seed_sql)
                    params_list.extend(seed[1])
                    all_paths.extend(seed[2])
                    per_ds_paths[ds] = list(dict.fromkeys(all_paths))
                else:
                    # 无 start（全历史）或显式关闭 seed_window：整段扫描
                    full = _build_branch_sub(
                        dsobj,
                        format_adapter_for_dataset(dsobj),
                        ds_params,
                        select_cols,
                        branch_tr=None,
                        time_lower_exclusive=False,
                        time_upper_exclusive=False,
                        time_col=right_time,
                        inst_col=inst_col,
                        hive_filters=hive,
                        ds_filters=ds_filters,
                        time_is_ts=time_is_ts,
                    )
                    branches.append(full[0])
                    params_list.extend(full[1])
                    per_ds_paths[ds] = full[2]
                sub = " UNION ALL ".join(branches)
                # 跨窗口/seed 统一按 (instrument, knowledge_time) 去重最新 revision
                if spec.deduplicate and spec.revision_order and inst_col and right_time:
                    sub = _dedup_key_sql(sub, inst_col, right_time, spec.revision_order)

            for c in cols:
                if c in seen_out:
                    raise ValidationError(
                        f"read_joined 输出列冲突: '{c}' 出现在多个数据集；"
                        f"请用 'dataset.col' 限定名或对其中一列改名。"
                    )
                seen_out.add(c)
                outer_cols.append(f"{alias}.{_quote_ident(c)} AS {_quote_ident(c)}")

            if i == 0:
                anchor_sub = sub
                continue
            prev = "a"
            if policy in {"asof", "pit_asof"}:
                cond = (
                    f"{prev}.{_quote_ident(anchor_inst)} = "
                    f"{alias}.{_quote_ident(inst_col)} "
                    f"AND {prev}.{_quote_ident(decision_time)} "
                    f"{spec.comparison_operator} {alias}.{_quote_ident(right_time)}"
                )
                join_clauses.append(f"ASOF LEFT JOIN ({sub}) AS {alias} ON {cond}")
            else:
                cond = (
                    f"{prev}.{_quote_ident(t_col)} = {alias}.{_quote_ident(t_col)} "
                    f"AND {prev}.{_quote_ident(anchor_inst)} = "
                    f"{alias}.{_quote_ident(inst_col)}"
                )
                join_clauses.append(f"LEFT JOIN ({sub}) AS {alias} ON {cond}")

        # #15 时变 universe：按 (date, instrument) 精确成员过滤（INNER JOIN），
        # 表达每日成分变化；不再把窗口内成员拍平成静态集合。
        if universe:
            uds = self._registry.get(universe)
            ut, ui = uds.time_column, uds.instrument_column
            if not ut or not ui:
                raise ValidationError(
                    f"universe 数据集 '{universe}' 未声明 time_column/instrument_column，"
                    f"无法做时变成员过滤"
                )
            uni_params = dict(params_by_dataset.get(universe, {}))
            upaths = self._prepare_dataset_read(
                uds,
                time_range=time_range,
                params=uni_params,
                instrument_filter=instrument_filter,
            )
            uadapter = format_adapter_for_dataset(uds)
            upath_param = upaths if len(upaths) > 1 else upaths[0]
            u_from = uadapter.build_from_clause(
                upath_param,
                hive_partitioning=uds.hive_partitioning,
                union_by_name=uds.union_by_name,
            )
            params_list.append(upath_param)
            utype = str((uds.schema or {}).get(ut or "", "")).lower()
            upred = Predicate(
                time_range=time_range,
                instrument_filter=instrument_filter,
                time_column_is_timestamp=("timestamp" in utype or "datetime" in utype),
            )
            ucompiled = compile_predicate(upred, time_column=ut, instrument_column=ui)
            params_list.extend(ucompiled.params)
            u_sub = (
                f"SELECT DISTINCT {_quote_ident(ut)}, {_quote_ident(ui)} "
                f"FROM {u_from} {ucompiled.where_sql}".strip()
            )
            join_clauses.append(
                f"INNER JOIN ({u_sub}) AS _u ON "
                f"a.{_quote_ident(anchor_time)} = _u.{_quote_ident(ut)} "
                f"AND a.{_quote_ident(anchor_inst)} = _u.{_quote_ident(ui)}"
            )
            if universe not in datasets:
                datasets.append(universe)
            per_ds_paths[universe] = upaths

        if anchor_sub is None:
            raise ValidationError("read_joined: anchor 子查询缺失")
        sql = (
            f"SELECT {', '.join(outer_cols)} "
            f"FROM ({anchor_sub}) AS a " + " ".join(join_clauses)
        )
        if limit is not None:
            sql = f"{sql} LIMIT {int(limit)}"
        return sql, params_list, datasets, per_ds_paths

    def _maybe_normalize_units(
        self,
        table: Any,
        *,
        dataset: str,
        columns: Sequence[str],
    ) -> Any:
        """按 SemanticFieldCatalog 做输出层单位归一化（容错：无定义的列跳过）。"""
        from data_access.read.semantic_catalog import normalize_table_units

        if table is None or not columns:
            return table
        fields = []
        for col in columns:
            try:
                fields.append(self.resolve_fields([col], dataset=dataset)[0])
            except Exception:
                continue
        return normalize_table_units(table, fields)

    def _normalize_read_result(
        self,
        rr: ReadResult,
        *,
        dataset: str,
        columns: Sequence[str],
    ) -> ReadResult:
        from data_access.read.read_contract import ReadResult as _RR, ReadStats

        table = self._maybe_normalize_units(rr.table, dataset=dataset, columns=columns)
        return _RR(
            table=table,
            snapshot=rr.snapshot,
            stats=ReadStats(
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=rr.stats.elapsed_ms,
                paths=rr.stats.paths,
            ),
            lineage=rr.lineage,
        )

    def _check_factor_versions(
        self,
        fids: Sequence[str],
        *,
        versions: Mapping[str, str] | None = None,
        require_same_data_snapshot: bool = False,
        require_same_universe: bool = False,
        time_range: tuple[Any, Any] | None = None,
        **params: Any,
    ) -> None:
        """#41 训练矩阵防混版本：读取时核对 factor_version / data_snapshot / universe。

        逐因子读一列样本（limit=1）取 factor_version / data_snapshot_id；违反
        一致性要求抛 ValidationError。
        """
        ds = self._registry.get("factor_lake")
        check_cols = []
        if versions:
            check_cols.append("factor_version")
        if require_same_data_snapshot:
            check_cols.append("data_snapshot_id")
        if require_same_universe:
            check_cols.append("universe" if "universe" in (ds.schema or {}) else "")
        check_cols = [c for c in check_cols if c]

        seen_snapshots: set[str] = set()
        seen_universes: set[str] = set()
        for fid in fids:
            if versions and versions.get(fid) is not None:
                expected = versions[fid]
                if not check_cols:
                    check_cols.append("factor_version")
                try:
                    probe = self.read(
                        "factor_lake",
                        columns=["factor_version"],
                        factor_id=fid,
                        time_range=time_range,
                        limit=1,
                        **params,
                    ).to_arrow()
                    actual = (
                        str(probe.column("factor_version").to_pylist()[0])
                        if probe.num_rows
                        else None
                    )
                except Exception:
                    actual = None
                if actual is not None and actual != str(expected):
                    raise ValidationError(
                        f"factor {fid}: 期望版本 {expected}，实际 {actual}。"
                        "禁止在训练矩阵里混入不同版本。"
                    )
                continue
            if require_same_data_snapshot or require_same_universe:
                try:
                    probe = self.read(
                        "factor_lake",
                        columns=["data_snapshot_id", "universe"],
                        factor_id=fid,
                        time_range=time_range,
                        limit=1,
                        **params,
                    ).to_arrow()
                except Exception:
                    probe = None
                if probe is not None and probe.num_rows:
                    if require_same_data_snapshot:
                        snap = str(probe.column("data_snapshot_id").to_pylist()[0])
                        seen_snapshots.add(snap)
                    if require_same_universe and "universe" in probe.column_names:
                        uni = str(probe.column("universe").to_pylist()[0])
                        seen_universes.add(uni)
        if require_same_data_snapshot and len(seen_snapshots) > 1:
            raise ValidationError(
                f"因子混用不同 data_snapshot：{sorted(seen_snapshots)}。"
                "训练矩阵要求同一数据快照。"
            )
        if require_same_universe and len(seen_universes) > 1:
            raise ValidationError(
                f"因子混用不同 universe：{sorted(seen_universes)}。"
                "训练矩阵要求同一 universe。"
            )

    def _resolve_universe_instruments(
        self,
        universe: str | None,
        time_range: tuple[Any, Any] | None,
        instruments: Sequence[str] | None,
    ) -> Sequence[str] | None:
        """把 universe 数据集解析成 instrument 集合，与显式 instruments 求交集。"""
        if not universe:
            return instruments
        try:
            ds = self._registry.get(universe)
        except ValidationError as exc:
            raise ValidationError(f"universe 数据集 '{universe}' 未注册") from exc
        inst_col = ds.instrument_column
        if inst_col is None:
            raise ValidationError(
                f"universe 数据集 '{universe}' 未声明 instrument_column，无法解析股票池"
            )
        cols = [ds.time_column, inst_col] if ds.time_column else [inst_col]
        try:
            table = self.read_arrow(universe, columns=cols, time_range=time_range)
        except Exception:
            table = None
        members: set[str] = set()
        if table is not None and table.num_rows:
            members = {
                str(m)
                for m in table.column(inst_col).to_pylist()
                if m is not None
            }
        if instruments:
            members &= set(instruments)
        return sorted(members) if members else None

    def _read_handle(
        self,
        ds: Dataset,
        *,
        dataset: str,
        registered_name: str | None,
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        filters: Any,
        limit: int | None,
        engine: str,
        result: str,
        prefer_polars: bool,
        batch_size: int,
        query_budget: QueryBudget | None,
        params: dict[str, Any],
        normalize_units: bool = False,
    ) -> ReadHandle:
        """统一 read 编排：engine 路由 + 结果形态。"""
        from data_access.read.formats import format_adapter_for_dataset
        from data_access.read.read_handle import ReadHandle

        adapter = format_adapter_for_dataset(ds)

        if engine == "auto":
            if not adapter.uses_duckdb:
                engine = "pyarrow"
            elif registered_name is not None:
                try:
                    from data_access.read.scan_cost import (
                        estimate_scan_cost,
                        suggest_read_strategy,
                    )

                    cost = estimate_scan_cost(
                        self,
                        registered_name,
                        columns=columns,
                        time_range=time_range,
                        instrument_filter=instrument_filter,
                        prefer_polars=prefer_polars,
                        **params,
                    )
                    engine, result = suggest_read_strategy(
                        cost,
                        prefer_polars=prefer_polars,
                        engine="auto",
                        result=result,
                    )
                except Exception:
                    engine = "duckdb"
            else:
                engine = "duckdb"

        if engine == "pyarrow":
            return self._read_pyarrow(
                ds,
                dataset=dataset,
                columns=columns,
                time_range=time_range,
                instrument_filter=instrument_filter,
                filters=filters,
                limit=limit,
                query_budget=query_budget,
                params=params,
                batch_size=batch_size,
                normalize_units=normalize_units,
            )
        if engine == "polars":
            budget = self._resolve_read_budget(ds, query_budget)
            lf, paths = self._scan_polars_with_paths(
                dataset,
                columns=columns,
                time_range=time_range,
                instrument_filter=instrument_filter,
                filters=filters,
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
            if result in {"lazy", "polars"}:
                return ReadHandle(
                    lazy=lf, snapshot=snapshot, lineage=lineage, batch_size=batch_size
                )
            table = collect_polars_with_budget(lf, query_budget=budget)
            if normalize_units and columns:
                table = self._maybe_normalize_units(
                    table, dataset=dataset, columns=columns
                )
            stats = ReadStats(
                rows=table.num_rows, bytes=table.nbytes, elapsed_ms=0.0
            )
            return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage)

        # duckdb：走标准 read 路径（含 budget/audit/snapshot）
        rr = self._read_dataset_object(
            ds,
            dataset=dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            limit=limit,
            query_budget=query_budget,
            params=params,
        )
        if normalize_units and columns and rr.table is not None:
            rr = self._normalize_read_result(
                rr, dataset=dataset, columns=columns
            )
        handle = ReadHandle(table=rr.table, snapshot=rr.snapshot, stats=rr.stats, lineage=rr.lineage)
        if result == "stream":
            # 大结果：给一个低内存峰值的流句柄
            return ReadHandle(
                stream=self.read_arrow_stream(
                    dataset,
                    columns=columns,
                    time_range=time_range,
                    instrument_filter=instrument_filter,
                    filters=filters,
                    limit=limit,
                    batch_size=batch_size,
                    query_budget=query_budget,
                    **params,
                ),
                snapshot=rr.snapshot,
                lineage=rr.lineage,
                batch_size=batch_size,
            )
        return handle

    def _read_pyarrow(
        self,
        ds: Dataset,
        *,
        dataset: str,
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        filters: Any,
        limit: int | None,
        query_budget: QueryBudget | None,
        params: dict[str, Any],
        batch_size: int,
        normalize_units: bool = False,
    ) -> ReadHandle:
        """PyArrow 引擎（arrow/feather 格式）：直读文件 + pc 表达式过滤。"""
        import pyarrow.compute as pc

        from data_access.read.formats import pyarrow_engine_read
        from data_access.read.manifest import manifest_root_for_paths
        from data_access.read.predicate_ast import compile_filter_arrow, parse_filters
        from data_access.read.read_handle import ReadHandle

        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        paths = self._prepare_dataset_read(
            ds,
            time_range=None,
            params=params,
            instrument_filter=instrument_filter,
        )
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)

        start = time.perf_counter()
        table = pyarrow_engine_read(
            paths, fmt=str(ds.format), columns=list(columns) if columns else None
        )

        # 过滤：time_range + instrument_filter + filters（pc 表达式）
        exprs: list[Any] = []
        if time_range is not None:
            if ds.time_column is None:
                raise ValidationError(
                    f"'{dataset}' 未声明 time_column，无法应用 time_range"
                )
            field_type = table.schema.field(ds.time_column).type
            start_v, end_v = time_range
            if start_v is not None:
                exprs.append(pc.greater_equal(pc.field(ds.time_column), _cast_scalar(start_v, field_type)))
            if end_v is not None:
                exprs.append(pc.less_equal(pc.field(ds.time_column), _cast_scalar(end_v, field_type)))
        if instrument_filter:
            if ds.instrument_column is None:
                raise ValidationError(
                    f"'{dataset}' 未声明 instrument_column，无法应用 instrument_filter"
                )
            exprs.append(pc.is_in(pc.field(ds.instrument_column), list(instrument_filter)))
        if filters is not None:
            expr = compile_filter_arrow(parse_filters(filters))
            if expr is not None:
                exprs.append(expr)
        if exprs:
            combined = exprs[0]
            for e in exprs[1:]:
                combined = pc.and_kleene(combined, e)
            table = table.filter(combined)
        if limit is not None:
            table = table.slice(0, int(limit))

        if normalize_units and columns:
            table = self._maybe_normalize_units(
                table, dataset=dataset, columns=columns
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        enforce_arrow_budget(budget, table, elapsed_ms=elapsed_ms)
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
        stats = ReadStats(
            rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms,
            paths=tuple(paths[:20]),
        )
        audit.record(
            op="read",
            dataset=dataset,
            ok=True,
            rows=table.num_rows,
            paths=paths[:5] if paths else None,
            params=params or None,
            elapsed_ms=elapsed_ms,
            extra={"engine": "pyarrow", "format": ds.format},
        )
        return ReadHandle(table=table, snapshot=snapshot, stats=stats, lineage=lineage, batch_size=batch_size)

    def _uri_dataset(
        self,
        uri: str,
        *,
        format: str,
        time_column: str | None,
        instrument_column: str | None,
    ) -> Dataset:
        """把 URI 包装成临时 StaticDataset（用于 read_uri，不改 registry）。"""
        from data_access.registry.loader import StaticDataset
        from data_access.read.formats import FormatSpec, default_glob_for_format

        is_remote = uri.startswith("s3://") or uri.startswith("cos://")
        if is_remote:
            root = Path("/")  # 远程路径鉴权走 s3 前缀白名单
            glob = uri
        else:
            p = Path(uri)
            if "*" in uri or "?" in uri or "[" in uri:
                static = uri.split("*", 1)[0].rstrip("/")
                root = canonicalize(static) if static else Path(uri).parent
                glob = uri
            elif p.is_dir():
                root = canonicalize(uri)
                glob = default_glob_for_format(format)
            else:
                root = canonicalize(p.parent)
                glob = p.name
        return StaticDataset(
            name=f"_uri:{format}:{uri[:48]}",
            access_mode="published",
            layout="plain",
            time_column=time_column,
            instrument_column=instrument_column,
            hive_partitioning=False,
            union_by_name=True,
            format_spec=FormatSpec.from_yaml(format),
            root=root,
            glob=glob,
            schema={},
        )

    def _assert_uri_allowed(self, uri: str, *, format: str) -> None:
        """read_uri 白名单：dev 放宽到白名单根 / env 目录；production 收紧到已登记根。"""
        from data_access.read.formats import normalize_format_name

        fmt = normalize_format_name(format)
        if uri.startswith("s3://") or uri.startswith("cos://"):
            from data_access.cos.remote import authorize_s3_path

            authorize_s3_path(uri)
            return

        static = str(uri).split("*", 1)[0].rstrip("/") or str(uri)
        resolved = canonicalize(static)
        strict = _production_mode() or _strict_read_mode()

        for root in self._authorizer.allowed_roots:
            try:
                resolved.relative_to(root)
                return
            except ValueError:
                continue
        if strict:
            raise ValidationError(
                f"read_uri 在 production/strict 模式只允许已登记数据集根下的 URI；"
                f"收到 {uri!r}。临时文件请先登记到 datasets.yaml 或关闭严格读。"
            )
        # dev：允许 env 白名单根
        extra = _uri_allowed_roots_from_env()
        for root in extra:
            try:
                resolved.relative_to(root)
                return
            except ValueError:
                continue
        raise ValidationError(
            f"read_uri 路径不在白名单下：{uri!r}\n"
            "已登记根 + DATA_ACCESS_EXTRA_ALLOWED_ROOTS + DATA_ACCESS_READ_URI_ROOTS 均不匹配。"
        )

    # ---- 因子批量读（read_factors / FactorCatalog） ----

    def read_factors(
        self,
        factor_ids: Sequence[str],
        *,
        time_range: tuple[Any, Any] | None = None,
        universe: str | None = None,
        frequency: str | None = None,
        layout: str = "long",
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        engine: str = "auto",
        result: str = "auto",
        prefer_polars: bool = False,
        batch_size: int = 100_000,
        query_budget: QueryBudget | None = None,
        versions: Mapping[str, str] | None = None,
        require_same_data_snapshot: bool = False,
        require_same_universe: bool = False,
        route_matrix_threshold: int = 100,
        **params: Any,
    ) -> ReadHandle:
        """一次读多个因子（单查询，非逐 factor 循环）。

        - ``layout="long"``：UNION ALL 各因子，输出 ``factor_id, datetime, asset, value, ...``
        - ``layout="wide"``：DuckDB PIVOT 成 ``datetime, asset, f1, f2, ...`` 宽矩阵
        - ``universe`` + ``frequency`` 提供时，wide 优先走 ``factor_matrix`` 物化层
        - ``versions``（#41）：``{factor_id: expected_version}``，读取时逐因子核对
          ``factor_version``，不一致抛 ValidationError（禁止训练矩阵混版本）
        - ``route_matrix_threshold``（#40）：wide 且因子数超过阈值时优先 matrix，
          少量因子仍走 factor-major 树（少开文件）

        返回 ``ReadHandle``，可 ``.to_arrow() / .to_polars() / .to_lazy()``。
        """
        from data_access.read.factors import (
            build_factor_pivot_sql,
            build_factor_union_sql,
        )
        from data_access.read.read_handle import ReadHandle

        if not factor_ids:
            raise ValidationError("read_factors: factor_ids 不能为空")
        fids = [str(f) for f in factor_ids]
        layout = str(layout).lower()
        if layout not in {"long", "wide"}:
            raise ValidationError("read_factors layout 必须是 long|wide")

        # #40 因子路由：matrix 用于「数百/数千因子训练」；少因子走 factor-major 树
        use_matrix = (
            layout == "wide"
            and universe is not None
            and (len(fids) >= max(1, int(route_matrix_threshold)) or prefer_polars)
        )
        if use_matrix:
            try:
                return self._read_factor_matrix(
                    fids,
                    time_range=time_range,
                    universe=universe,
                    frequency=frequency or "daily",
                    columns=columns,
                    limit=limit,
                    engine=engine,
                    result=result,
                    prefer_polars=prefer_polars,
                    batch_size=batch_size,
                    query_budget=query_budget,
                    **params,
                )
            except ValidationError:
                pass  # factor_matrix 未登记 → 回退长表 pivot

        # #41 版本 / data_snapshot / universe 一致性检查（训练矩阵防混版本）
        if versions or require_same_data_snapshot or require_same_universe:
            try:
                self._check_factor_versions(
                    fids,
                    versions=versions,
                    require_same_data_snapshot=require_same_data_snapshot,
                    require_same_universe=require_same_universe,
                    time_range=time_range,
                    **params,
                )
            except DataError:
                raise
            except Exception:
                pass

        ds = self._registry.get("factor_lake")
        branches: list[tuple[str, list[str]]] = []
        all_paths: list[str] = []
        for fid in fids:
            paths = self._prepare_dataset_read(
                ds, time_range=time_range, params=dict(params, factor_id=fid)
            )
            if paths:
                branches.append((fid, paths))
                all_paths.extend(paths)
        if not branches:
            raise DataError(
                f"read_factors: 因子 {fids} 都没有可读文件"
                "（检查 factor_id / time_range / 数据是否存在）"
            )

        union_sql, union_params = build_factor_union_sql(
            branches,
            columns=columns,
            time_range=time_range,
            time_column=ds.time_column or "datetime",
            hive_partitioning=ds.hive_partitioning,
            union_by_name=ds.union_by_name,
            limit=None,
        )
        if layout == "wide":
            sql, sql_params = build_factor_pivot_sql(
                union_sql, union_params, factor_ids=fids
            )
        else:
            sql, sql_params = union_sql, union_params
        if limit is not None:
            sql = f"SELECT * FROM ({sql}) AS __b LIMIT {int(limit)}"

        budget = self._resolve_read_budget(ds, query_budget)
        start = time.perf_counter()
        table = self._engine.execute_arrow(
            sql, sql_params, deadline_ms=budget.max_elapsed_ms
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        enforce_arrow_budget(budget, table, elapsed_ms=elapsed_ms)
        snapshot = self._build_snapshot(
            dataset="factors:" + ",".join(fids),
            ds=ds,
            paths=all_paths,
            params={"factor_ids": fids, "layout": layout},
        )
        lineage = ReadLineage(
            dataset="factors",
            columns=tuple(columns) if columns else (),
            time_range=time_range,
            params=snapshot.params,
        )
        stats = ReadStats(
            rows=table.num_rows, bytes=table.nbytes, elapsed_ms=elapsed_ms
        )
        audit.record(
            op="read",
            dataset="factors",
            ok=True,
            rows=table.num_rows,
            params={"factor_ids": fids, "layout": layout},
            elapsed_ms=elapsed_ms,
            extra={"engine": "duckdb", "multi_factor": True},
        )
        return ReadHandle(
            table=table, snapshot=snapshot, stats=stats, lineage=lineage, batch_size=batch_size
        )

    def _read_factor_matrix(
        self,
        fids: list[str],
        *,
        time_range: tuple[Any, Any] | None,
        universe: str,
        frequency: str,
        columns: Sequence[str] | None,
        limit: int | None,
        engine: str,
        result: str,
        prefer_polars: bool,
        batch_size: int,
        query_budget: QueryBudget | None,
        **params: Any,
    ) -> ReadHandle:
        """走 factor_matrix 物化层读宽矩阵（universe + frequency）。"""
        from data_access.read.read_handle import ReadHandle

        matrix = self._registry.get("factor_matrix")  # 未登记会抛 ValidationError
        matrix_params = dict(params, universe=universe, frequency=frequency)
        handle = self.read(
            "factor_matrix",
            columns=columns,
            time_range=time_range,
            limit=limit,
            engine=engine,
            result=result,
            prefer_polars=prefer_polars,
            batch_size=batch_size,
            query_budget=query_budget,
            **matrix_params,
        )
        audit.record(
            op="read",
            dataset="factor_matrix",
            ok=True,
            rows=handle.rows,
            params={"universe": universe, "frequency": frequency, "factors": fids},
            elapsed_ms=0.0,
            extra={"engine": "matrix", "multi_factor": True},
        )
        return handle

    def get_factor_catalog(
        self,
        dataset: str = "factor_lake",
        *,
        discover: bool = True,
    ):
        """加载因子目录（FactorCatalog）。空目录时可选从因子湖扫描重建。"""
        from data_access.read.factors import FactorCatalog, factor_catalog_root

        root = factor_catalog_root(self, dataset)
        if root is None:
            return FactorCatalog(root=Path("/"), records={})
        catalog = FactorCatalog.load(root)
        if discover and len(catalog) == 0:
            catalog = FactorCatalog.discover(root)
        return catalog

    def refresh_factor_catalog(
        self,
        dataset: str = "factor_lake",
        *,
        factor_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """扫描因子湖重建目录，落盘 ``_factor_catalog.json``。"""
        from data_access.read.factors import FactorCatalog, factor_catalog_root

        root = factor_catalog_root(self, dataset)
        if root is None:
            raise ValidationError(f"无法解析因子湖根目录（dataset={dataset}）")
        catalog = FactorCatalog.discover(root, factor_ids=factor_ids)
        catalog.save(root)
        return {
            "root": str(root),
            "factors": len(catalog),
            "factor_ids": catalog.ids(),
        }

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

    def build_dataset_manifest(
        self,
        dataset: str,
        *,
        include_row_groups: bool = False,
        force: bool = False,
        **params: Any,
    ) -> dict[str, Any] | None:
        """为数据集构建 ``_manifest.parquet``（文件级 min/max 元数据清单）。

        建好后 read 路径会用它按 time_range / instrument_filter 做文件级裁剪，
        避免 ``**/*.parquet`` 全量 glob + 逐文件 footer。返回构建摘要。
        """
        from data_access.read.manifest import build_manifest_for_dataset

        manifest = build_manifest_for_dataset(
            self,
            dataset,
            params=dict(params or {}),
            include_row_groups=include_row_groups,
            force=force,
        )
        if manifest is None:
            return None
        return {
            "dataset": dataset,
            "files": manifest.file_count,
            "rows": manifest.total_rows,
            "bytes": manifest.total_bytes,
            "format": manifest.format,
        }

    def load_columns(
        self,
        dataset: str,
        *,
        columns: Sequence[str],
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        filters: Any = None,
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
        if ds.time_column is None or ds.instrument_column is None:
            raise ValidationError(
                f"数据集 '{dataset}' 需要 time_column + instrument_column 才能 load_columns；"
                "请在 datasets.yaml 声明（或 roles.event_time / roles.instrument）。"
            )
        all_cols = list(dict.fromkeys(
            [ds.time_column, ds.instrument_column, *columns]
        ))
        from data_access.read.key_policy import resolve_key_policy

        read_result = self.read_result(
            dataset,
            columns=all_cols,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
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
        # 写路径 mutation 后 bump manifest epoch，让 query-scoped snapshot 立即失效
        self.touch_manifest_epoch(dataset, **params)
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

        result = upsert_mod.upsert_table(
            ds=ds,
            authorizer=self._authorizer,
            target_dir=target_dir,
            new_table=table,
            upsert_on=upsert_on,
            partition_by=partition_by,
            params=params,
        )
        self.touch_manifest_epoch(dataset, **params)
        return result

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

        result = publish.publish_from_staging(
            registry=self._registry,
            authorizer=self._authorizer,
            staging_name=staging_dataset,
            target_name=target_dataset,
            **params,
        )
        self.touch_manifest_epoch(target_dataset, **params)
        return result

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

    def _resolve_raw_paths(
        self,
        ds: Dataset,
        *,
        time_range: tuple[Any, Any] | None,
        params: dict[str, Any],
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """COS 镜像 / 远程直读 + 路径解析（未做 manifest/partition 裁剪）。

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

    def _prune_read_paths(
        self,
        ds: Dataset,
        paths: list[str],
        *,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        """读前路径裁剪：Partition Planner（路径模板级）+ Manifest（文件级）。

        只对本地 parquet 生效；远程/非 parquet 直接放行。裁剪结果会让
        DuckDB 收到的文件列表大幅缩小（尤其是 ``**/*.parquet`` 大量小文件）。
        """
        from data_access.read.partition_planner import parse_partitioning, prune_paths_for_time_range

        partitioning = parse_partitioning(getattr(ds, "partitioning", None))
        paths = prune_paths_for_time_range(
            paths,
            time_range,
            partitioning=partitioning,
            time_column=ds.time_column,
        )

        if (time_range is not None or instrument_filter) and str(getattr(ds, "format", "parquet")) in {
            "parquet",
            "pq",
        } and not any(str(p).startswith("s3://") for p in paths):
            from data_access.read.manifest import DatasetManifest, is_manifest_fresh, manifest_root_for_paths

            root = manifest_root_for_paths(paths)
            if root is not None:
                try:
                    manifest = DatasetManifest.load(root)
                except Exception:
                    manifest = None
                if manifest is not None and manifest.dataset in {"", ds.name}:
                    raw_paths = self._resolve_raw_paths(
                        ds,
                        time_range=None,
                        params=dict(params or {}),
                        instrument_filter=None,
                    )
                    if is_manifest_fresh(manifest, raw_paths):
                        pruned = manifest.prune(
                            time_range=time_range,
                            instrument_filter=instrument_filter,
                        )
                        if not pruned:
                            # #21 manifest 确认没有匹配文件 → 返回空列表，读路径
                            # 生成空 relation，绝不回退全量扫描。
                            logger.info(
                                "manifest_prune: dataset=%s 无匹配文件，返回空读取",
                                ds.name,
                            )
                            return []
                        logger.info(
                            "manifest_prune: dataset=%s raw_files=%d pruned_files=%d",
                            ds.name,
                            manifest.file_count,
                            len(pruned),
                        )
                        return self._authorize_read_paths(pruned)
        return paths

    def _prepare_dataset_read(
        self,
        ds: Dataset,
        *,
        time_range: tuple[Any, Any] | None,
        params: dict[str, Any],
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """读路径统一入口：raw 解析 + manifest/partition 裁剪。"""
        paths = self._resolve_raw_paths(
            ds,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
        )
        return self._prune_read_paths(
            ds,
            paths,
            time_range=time_range,
            instrument_filter=instrument_filter,
            params=params,
        )

    def _authorize_read_paths(self, glob_paths: list[str]) -> list[str]:
        """本地/远程读路径白名单校验。

        白名单 = PathAuthorizer（已登记根 + DATA_ACCESS_EXTRA_ALLOWED_ROOTS）
        + DATA_ACCESS_READ_URI_ROOTS（read_uri 临时文件目录）。
        """
        from data_access.cos.remote import authorize_s3_path, cos_cache_root
        from data_access.registry.paths import path_is_under

        cache_root = cos_cache_root()
        env_roots = _uri_allowed_roots_from_env()
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
            if _path_under_any(resolved, env_roots):
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
        filters: Any = None,
    ) -> tuple[str, list[Any]]:
        """组装 SELECT 语句。路径用 ? 参数绑定，列名/谓词用 registry 控制。

        文件格式由 registry 的 ``format`` 决定（FormatAdapter），默认 parquet——
        对现有数据集生成的 SQL 与改前逐字节一致。
        """
        from data_access.read.formats import format_adapter_for_dataset
        from data_access.read.predicate_ast import parse_filters

        if columns:
            col_clause = ", ".join(_quote_ident(c) for c in columns)
        else:
            col_clause = "*"

        # #21 manifest 空裁剪 → 返回带 schema 的空 SELECT，避免 read_parquet([]) 报错
        if not paths:
            schema = getattr(ds, "schema", None) or {}
            if columns:
                select_cols = list(columns)
            else:
                select_cols = list(schema.keys())
            for k in (ds.time_column, ds.instrument_column):
                if k and k not in select_cols:
                    select_cols.append(k)
            sql = _empty_branch_sql(select_cols, ds)
            if limit is not None:
                sql = f"{sql} LIMIT {int(limit)}"
            return sql, []

        # path 参数：单路径直接 ?，多路径用 list；命名参数（hive_partitioning/
        # union_by_name 等）由 adapter 直接拼 SQL——这些值来自 registry，可信。
        path_param = paths if len(paths) > 1 else paths[0]
        adapter = format_adapter_for_dataset(ds)
        if not adapter.uses_duckdb:
            raise ValidationError(
                f"数据集 '{ds.name}' 的格式 '{ds.format}' 不能走 DuckDB SQL；"
                f"请用 store.read(..., engine='pyarrow') / read_uri 的 PyArrow 路径。"
            )
        from_clause = adapter.build_from_clause(
            path_param,
            hive_partitioning=ds.hive_partitioning,
            union_by_name=ds.union_by_name,
        )

        if time_range is not None and ds.time_column is None:
            raise ValidationError(
                f"数据集 '{ds.name}' 未声明 time_column，无法应用 time_range；"
                "请在 datasets.yaml 里声明 time_column 或 roles.event_time。"
            )
        if instrument_filter and ds.instrument_column is None:
            raise ValidationError(
                f"数据集 '{ds.name}' 未声明 instrument_column，无法应用 instrument_filter；"
                "请在 datasets.yaml 里声明 instrument_column 或 roles.instrument。"
            )
        time_col_type = str((ds.schema or {}).get(ds.time_column or "", "")).lower()
        predicate = Predicate(
            time_range=time_range,
            instrument_filter=instrument_filter,
            hive_filters=self._bucket_hive_filters(ds, instrument_filter),
            filters=parse_filters(filters),
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

    若引擎已被 ``reset_shared_engine()`` 关闭，自动重建（避免缓存 store 指向
    已关闭连接）。线程安全：双重检查 + Lock。
    """
    global _store
    if _store is not None:
        if _store._engine.is_closed:
            _store = None
        else:
            return _store
    with _store_lock:
        if _store is not None and not _store._engine.is_closed:
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
    if not schema or not ds.time_column:
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


def _and_filters(*filters: Any) -> Any:
    """把多个 Filter AST 合并成 And(...)；None 忽略。"""
    from data_access.read.predicate_ast import And

    parts = [f for f in filters if f is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return And(tuple(parts))


def _dedup_key_sql(sub_sql: str, inst_col: str, time_col: str, revision_order: Sequence[str]) -> str:
    """在（可含 UNION 的）事件子查询上按 (instrument, time) 去重，保留
    revision_order 降序最新一版——替代依赖 parquet 扫描顺序的 keep_last。

    ``QUALIFY`` 在 DuckDB 里对 ``SELECT ... FROM (<subquery>)`` 同样生效。
    """
    order = ", ".join(f"{_quote_ident(c)} DESC" for c in revision_order)
    return (
        f"SELECT * FROM ({sub_sql}) _ev "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)}, "
        f"{_quote_ident(time_col)} ORDER BY {order}) = 1"
    )


def _seed_qualify_sql(sub_sql: str, inst_col: str, time_col: str, revision_order: Sequence[str]) -> str:
    """PIT seed 分支：只取每标的在窗口 start 之前最后一条可见记录
    （time 降序、revision 降序）。"""
    order = ", ".join(
        f"{_quote_ident(c)} DESC" for c in [time_col, *revision_order]
    )
    return (
        f"SELECT * FROM ({sub_sql}) _seed "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
        f"ORDER BY {order}) = 1"
    )


def _duckdb_type_for_schema(dtype: str) -> str:
    """把 registry schema dtype 字符串映射到 DuckDB 类型（空 relation 用）。"""
    d = (dtype or "").strip().lower()
    if "double" in d or "float" in d:
        return "DOUBLE"
    if "date" in d:
        return "DATE"
    if "timestamp" in d or "datetime" in d:
        return "TIMESTAMP"
    if "bool" in d:
        return "BOOLEAN"
    if "int" in d or "long" in d:
        return "BIGINT"
    return "VARCHAR"


def _empty_branch_sql(select_cols: Sequence[str], dsobj: Any) -> str:
    """生成一个带 schema 的空 SELECT（``WHERE FALSE``），供 #21 manifest 空裁剪
    使用——避免 DuckDB ``read_parquet([])`` 直接报错。"""
    schema = getattr(dsobj, "schema", None) or {}
    parts: list[str] = []
    for c in select_cols:
        dtype = str(schema.get(c, "") or "VARCHAR")
        parts.append(f"NULL::{_duckdb_type_for_schema(dtype)} AS {_quote_ident(c)}")
    cols = ", ".join(parts) if parts else "*"
    return f"SELECT {cols} WHERE FALSE"


def _infer_format_from_uri(uri: str, explicit: str) -> str:
    """read_uri 格式推断：显式传入优先，否则按扩展名。"""
    from data_access.read.formats import normalize_format_name

    if explicit not in (None, "", "auto"):
        return normalize_format_name(explicit)
    lower = str(uri).lower()
    for suffix, fmt in (
        (".parquet", "parquet"),
        (".pq", "parquet"),
        (".csv.gz", "csv"),
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".jsonl", "jsonl"),
        (".ndjson", "jsonl"),
        (".json", "jsonl"),
        (".arrow", "arrow"),
        (".ipc", "arrow"),
        (".feather", "feather"),
    ):
        if lower.endswith(suffix):
            return fmt
    return "parquet"


def _cast_scalar(value: Any, field_type: Any):
    """把 time_range 端点 cast 到 Arrow 列类型（pc 比较需要类型匹配）。"""
    import pyarrow as pa

    try:
        return pa.scalar(value).cast(field_type)
    except Exception:
        return pa.scalar(value)


def _uri_allowed_roots_from_env() -> list[Path]:
    """``DATA_ACCESS_READ_URI_ROOTS``：逗号分隔的 read_uri 额外白名单根（dev）。"""
    raw = os.environ.get("DATA_ACCESS_READ_URI_ROOTS", "")
    roots: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            roots.append(canonicalize(part))
    return roots


def _path_under_any(path: Path, roots: Sequence[Path]) -> bool:
    from data_access.registry.paths import path_is_under

    for root in roots:
        if path_is_under(path, root):
            return True
    return False


def log_read_auto(dataset: str, cost: Any, engine: str, result: str) -> None:
    """read_auto 路由日志。"""
    logger.info(
        "read_auto dataset=%s engine=%s result=%s rows=%d files=%d bytes=%s score=%.0f",
        dataset,
        engine,
        result,
        getattr(cost, "estimated_rows", None),
        getattr(cost, "file_count", None),
        getattr(cost, "total_bytes", None),
        getattr(cost, "score", 0.0),
    )


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
