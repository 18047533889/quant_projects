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
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa

if TYPE_CHECKING:
    import polars as pl

from data_access.core import audit
from data_access.read.adapters import arrow_table_to_multiindex_columns
from data_access.core.engine import DuckDBEngine, get_shared_engine, reset_shared_engine
from data_access.core.exceptions import (
    DataError,
    MatrixCoverageMiss,
    MatrixUnavailable,
    ValidationError,
)
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
    is_strict_semantics,
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

# #44 read()/read_result() 的读取模式：普通面板 / 事件 / PIT / 维表 / 稀疏
_VALID_READ_MODES = {"auto", "panel", "event", "pit", "dimension", "sparse"}


def _cos_mode_is_auto() -> bool:
    """COS 读模式是否为 auto（混合 local+remote 的前提）。"""
    return (
        os.environ.get("DATA_ACCESS_COS_READ_MODE", "mirror").strip().lower()
        == "auto"
    )


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
        # #46 注入的市场交易日历（{market: MarketCalendar}），session availability 用
        self._calendars: dict[str, Any] = {}

    @contextmanager
    def _dataset_mutation(self, dataset: str, **params: Any):
        """#P0-27/#10 DatasetTransaction：显式 PREPARED/COMMITTED/ABORTED。

        顺序（#9）：
            1. **PREPARED**（进入 body 前）：先 bump source_epoch——**mark
               generation dirty 在 mutate 之前**。读者立刻判旧，不给「数据已改、
               旧 epoch 仍短暂有效、缓存继续命中旧结果」的窗口。
            2. body 正常返回 → **COMMITTED**：rebuild manifest，
               ``manifest_built_epoch`` 追上 → 数据源成为可信任的新 generation。
            3. body 抛异常 → **ABORTED**：**绝不 build fresh manifest**（#10）——
               失败后的部分状态不能当成新版本；manifest 保持 dirty，读路径安全
               回退 glob。

        业务代码不再自己 ``touch_manifest_epoch``。
        """
        from data_access.read.manifest import (
            bump_source_epoch,
            manifest_root_for_paths,
            rebuild_manifest_for_dataset,
        )

        state = "PREPARED"
        # 关键：finally 里**绝不能 `return`**——body 抛出的异常正在传播时，
        # finally 里的 return 会把它静默吞掉。这里只做 guard，让异常自然传播。
        active_exc = sys.exc_info()[1]

        # #9 mutate 前置失效：先 mark dirty，正文还没写、旧 generation 已不可信
        try:
            ds = self._registry.get(dataset)
        except Exception:
            if active_exc is None:
                raise
            ds = None
        if ds is not None:
            try:
                raw_paths = self._resolve_raw_paths(
                    ds, time_range=None, params=dict(params)
                )
                root = manifest_root_for_paths(raw_paths)
                if root is not None:
                    bump_source_epoch(root)
            except Exception:
                if active_exc is None:
                    raise
        try:
            yield
            state = "COMMITTED"
        except Exception:
            state = "ABORTED"
            raise
        finally:
            # 只有 COMMITTED 才允许 metadata 追平 source generation（#10）
            if state == "COMMITTED" and ds is not None:
                try:
                    rebuild_manifest_for_dataset(self, dataset, **params)
                except Exception:
                    if sys.exc_info()[1] is None:
                        raise

    def set_calendar(self, market: str, calendar: Any) -> None:
        """注入一个市场的交易日历（MarketCalendar），供 session availability 使用。

        部署在不同服务器时，可在这里注入该环境的真实交易所日历（替代默认
        周末休市兜底）。市场名：ashare / us。
        """
        self._calendars[str(market).strip().lower()] = calendar

    def get_calendar(self, market: str | None) -> Any | None:
        """取某市场的交易日历：显式注入优先，否则从 registry 日历数据集惰性加载。"""
        if not market:
            return None
        key = str(market).strip().lower()
        if key in self._calendars:
            return self._calendars[key]
        from data_access.read.session_calendar import get_market_calendar

        return get_market_calendar(key, store=self)

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
        if kind == "read" and (is_strict_semantics()):
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
            2. 全局 filters 或 filters_by_dataset 的过滤 AST **真正限制**该列
               （#14：不是「列名出现在条件里」——``timeframe != 'quarterly'``、
               ``timeframe IS NOT NULL``、``timeframe='quarterly' OR price>0``
               都不能证明结果落在所需维度，必须走 PredicateConstraint）。
        """
        from data_access.read.predicate_ast import filter_restricts_column, parse_filters

        missing: list[tuple[str, str]] = []
        global_ast = parse_filters(filters)
        per_ds_ast = {
            str(ds): parse_filters(f)
            for ds, f in (filters_by_dataset or {}).items()
        }
        for f in fields_meta:
            required = getattr(f, "required_filters", ()) or ()
            if not required:
                continue
            ds = getattr(f, "dataset", None)
            effective = dict((params_by_dataset or {}).get(ds, {}))
            ds_ast = per_ds_ast.get(ds)
            for key in required:
                if key in effective:
                    continue
                # params 里大小写宽松匹配（IndexSymbol / index_symbol）
                lower_key = str(key).lower()
                if any(str(k).lower() == lower_key for k in effective):
                    continue
                # filters / filters_by_dataset 里必须**真正约束**该列
                if filter_restricts_column(global_ast, str(key)):
                    continue
                if filter_restricts_column(ds_ast, str(key)):
                    continue
                missing.append((getattr(f, "logical_name", "?"), str(key)))
        if not missing:
            return
        detail = "; ".join(f"'{n}' 需参数或过滤 {k}" for n, k in missing)
        if is_strict_semantics():
            raise ValidationError(
                f"required_filters 未满足（production fail-closed）：{detail}"
            )
        logger.warning("required_filters 未满足（research 放行）：%s", detail)

    def _dataset_required_filters(self, dataset: str) -> list[str]:
        """#P0-10 数据集级 required filters（COS 契约 required_panel/dimension_filters）。

        无论用户选哪些列都强制执行——不再依赖「恰好选中了某个 catalog 字段」。
        columns=None / 物理直读列 / stream / Polars / PyArrow 全部同样受约束。
        """
        try:
            from data_access.cos_contract import get_cos_contract

            contract = get_cos_contract(dataset)
        except Exception:
            return []
        if contract is None:
            return []
        return list(
            dict.fromkeys(
                list(contract.required_panel_filters or ())
                + list(contract.required_dimension_filters or ())
            )
        )

    def _fields_meta_for_columns(
        self, dataset: str, columns: Sequence[str] | None
    ) -> list[Any]:
        """把单表读的列名解析成 catalog 字段元数据（含市场消歧）。

        required_filters 只对 catalog 已登记语义的列生效；物理直读列（catalog
        未覆盖）返回带空 required_filters 的占位，不触发强制。
        """
        from data_access.read.semantic_catalog import SemanticField, get_semantic_catalog

        if not columns:
            return []
        catalog = get_semantic_catalog()
        out: list[Any] = []
        for col in columns:
            f = catalog.resolve_one(str(col), dataset=dataset)
            if f is not None:
                out.append(f)
            else:
                out.append(
                    SemanticField(
                        logical_name=str(col), dataset=dataset, physical_name=str(col)
                    )
                )
        return out

    def _enforce_read_contract(
        self,
        dataset: str,
        *,
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
    ) -> None:
        """#44 Store 级 temporal model 强制（D1/S1/E1/E2/X0/EMPTY/MINUTE/STATIC/RAW_EVENT 一等公民）。

        只对 ``COS_DATASET_CONTRACTS`` 里登记的数据集生效；factor_lake 等无契约
        数据集跳过（它们没有 panel/PIT 语义约束）。语义：
            - EMPTY               → 一律拒绝（占位表）。
            - X0（mode=panel）    → 必须 allow_sparse=True 才放行。
            - E1/E2/RAW_EVENT（mode=panel/auto）
                → production/strict 拒绝（必须走 mode='event'/'pit' 或 read_joined）；
                  research 告警放行。
            - STATIC（mode=panel/auto）
                → production/strict 拒绝（mode='dimension' 放行）；research 告警。
            - mode='event'/'pit'  → 必须是事件表（pit 会校验 PIT 时钟）。
            - mode='dimension'    → 必须是 STATIC。
            - mode='sparse'       → 必须是 X0。
        """
        from data_access.cos_contract import get_cos_contract, resolve_event_clock

        contract = get_cos_contract(dataset)
        if contract is None:
            return
        effective = str(mode or "auto").strip().lower()
        if effective not in _VALID_READ_MODES:
            raise ValidationError(
                f"mode={mode!r} 非法；合法: {sorted(_VALID_READ_MODES)}"
            )
        if contract.is_empty:
            raise ValidationError(f"数据集 {dataset!r} 是 EMPTY 占位表，禁止作为数据源")
        if effective == "event":
            if not (contract.is_event or contract.is_raw_event):
                raise ValidationError(
                    f"数据集 {dataset!r} 不是事件表，mode='event' 不适用"
                )
            if contract.pit_policy == "effective_time_only":
                resolve_event_clock(
                    dataset, allow_effective_time=allow_effective_time
                )
            return
        if effective == "pit":
            resolve_event_clock(dataset, allow_effective_time=allow_effective_time)
            return
        if effective == "dimension":
            if not contract.is_static:
                raise ValidationError(
                    f"数据集 {dataset!r} 不是 STATIC 维表，mode='dimension' 不适用"
                )
            return
        if effective == "sparse":
            if not contract.is_sparse:
                raise ValidationError(
                    f"数据集 {dataset!r} 不是 X0 稀疏表，mode='sparse' 不适用"
                )
            return
        # mode == auto | panel：普通面板读取
        if contract.is_sparse:
            if allow_sparse:
                return
            raise ValidationError(
                f"数据集 {dataset!r} 是 X0 稀疏表，不是完整日频面板；"
                "如需稀疏读取请显式传 allow_sparse=True"
            )
        if contract.is_event or contract.is_raw_event:
            msg = (
                f"数据集 {dataset!r} 是 {contract.temporal_model} 事件表；普通面板读取"
                "禁止。请用 mode='event'/'pit'（事件/PIT API）或 read_joined 语义 join。"
            )
            if is_strict_semantics():
                raise ValidationError(msg)
            logger.warning("%s（research 放行）", msg)
            return
        if contract.is_static:
            msg = (
                f"数据集 {dataset!r} 是 STATIC 维表，不是时间×标的面板；"
                "如需维表读取请 mode='dimension'。"
            )
            if is_strict_semantics():
                raise ValidationError(msg)
            logger.warning("%s（research 放行）", msg)

    def _validate_allowed_filter_values(
        self,
        fields_meta: Sequence[Any],
        params_by_dataset: Mapping[str, Mapping[str, Any]],
        *,
        filters: Any = None,
        filters_by_dataset: Mapping[str, Any] | None = None,
    ) -> None:
        """#52 allowed_filter_values 值校验：过滤值必须在契约允许集合内。

        只对 catalog 字段挂到的 COS 契约数据集生效；校验 params 里的路径参数值
        与 filters/filters_by_dataset 里过滤列的字面值。production fail-closed。

        #15：``filter_column_values`` 只抽 Eq/In/Between 字面量——对 Ne/NotIn/
        IsNotNull/复杂 Not/Or 返回空集合，旧代码「跳过校验」，等于放行越界值。
        现在用 ``filter_constraint_status`` 区分三种：
            absent   → 未约束该列，跳过（合法）
            positive → 检查提取值 ⊆ allowed
            weak     → 无法证明 result ⊆ allowed → production/strict 拒绝
        """
        from data_access.cos_contract import get_cos_contract
        from data_access.read.predicate_ast import (
            filter_column_values,
            filter_constraint_status,
            parse_filters,
        )

        problems: list[tuple[str, str, list[Any]]] = []
        seen: set[str] = set()
        global_ast = parse_filters(filters)
        for f in fields_meta:
            ds = getattr(f, "dataset", None)
            if not ds:
                continue
            contract = get_cos_contract(ds)
            if contract is None or not contract.allowed_filters:
                continue
            key = f"{ds}:{tuple(sorted(contract.allowed_filters))}"
            if key in seen:
                continue
            seen.add(key)
            effective = dict((params_by_dataset or {}).get(ds, {}))
            per_ds_ast = parse_filters((filters_by_dataset or {}).get(ds))
            for c, choices in contract.allowed_filters.items():
                if c in effective:
                    pv = effective[c]
                    vals = pv if isinstance(pv, (list, tuple, set, frozenset)) else (pv,)
                    bad = [v for v in vals if v not in choices]
                    if bad:
                        problems.append((ds, c, bad))
                # 全局过滤（锚点）与 per-dataset 过滤都看；任一 weak 即不可证明
                g_status = filter_constraint_status(global_ast, str(c))
                p_status = filter_constraint_status(per_ds_ast, str(c))
                if g_status == "positive" or p_status == "positive":
                    vals: set[Any] = set()
                    for src in (global_ast, per_ds_ast):
                        for _c, _v in filter_column_values(src).items():
                            if _c == str(c):
                                vals |= set(_v)
                    bad = [v for v in vals if v not in choices]
                    if bad:
                        problems.append((ds, c, bad))
                elif g_status == "weak" or p_status == "weak":
                    # #15 无法证明子集：Ne/NotIn/IsNotNull/Or 部分分支等
                    problems.append((ds, c, ["<unprovable>"]))
        if not problems:
            return
        detail = "; ".join(f"{ds}.{c}={bad!r}" for ds, c, bad in problems)
        if is_strict_semantics():
            raise ValidationError(
                f"过滤值不在契约允许集合内（production fail-closed）：{detail}"
            )
        logger.warning("过滤值不在契约允许集合内（research 放行）：%s", detail)

    def _validate_joined_contract_filters(
        self,
        per_ds: Mapping[str, Sequence[str]],
        params_by_dataset: Mapping[str, Mapping[str, Any]],
        *,
        filters: Any = None,
        filters_by_dataset: Mapping[str, Any] | None = None,
    ) -> None:
        """read_joined 的每数据集契约校验（P0-1/P0-9）。

        对每个参与 join 的 COS 契约数据集：
            - EMPTY → 拒绝；
            - required_panel_filters / required_dimension_filters 必须被
              params 或 filters 列覆盖（否则 ×N 行或语义错误）。
        fan-out 守卫（P0-10）在 ``_read_joined_sql`` 里按实际 join key 判定。
        """
        from data_access.cos_contract import get_cos_contract
        from data_access.read.predicate_ast import filter_columns, parse_filters

        global_cols = filter_columns(parse_filters(filters))
        per_ds_cols = {
            str(ds): filter_columns(parse_filters(f))
            for ds, f in (filters_by_dataset or {}).items()
        }
        for ds in per_ds:
            contract = get_cos_contract(ds)
            if contract is None:
                continue
            if contract.is_empty:
                raise ValidationError(
                    f"数据集 {ds!r} 是 EMPTY 占位表，禁止参与 join"
                )
            effective = dict((params_by_dataset or {}).get(ds, {}))
            filter_cols = set(global_cols) | set(per_ds_cols.get(ds, set()))
            required = (
                tuple(contract.required_panel_filters or ())
                + tuple(contract.required_dimension_filters or ())
            )
            if required:
                missing = []
                for k in required:
                    if k in effective or any(
                        str(c).lower() == str(k).lower() for c in effective
                    ):
                        continue
                    if any(str(c).lower() == str(k).lower() for c in filter_cols):
                        continue
                    missing.append(k)
                if missing:
                    detail = ", ".join(missing)
                    if is_strict_semantics():
                        raise ValidationError(
                            f"数据集 {ds!r} join 缺少必需过滤 {detail}"
                            "（production fail-closed）"
                        )
                    logger.warning(
                        "数据集 %r join 缺少必需过滤 %s（research 放行）",
                        ds, detail,
                    )

    def _event_cutoff_for_contract(
        self,
        dataset: str,
        *,
        time_range: tuple[Any, Any] | None,
    ) -> None:
        """#16 effective_time_only 事件表未来数据 cutoff（读路径）。"""
        from data_access.cos_contract import enforce_event_cutoff, get_cos_contract

        contract = get_cos_contract(dataset)
        if contract is None:
            return
        enforce_event_cutoff(
            contract,
            time_range=time_range,
            production=is_strict_semantics(),
        )

    def _prepare_read_request(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        filters: Any = None,
        params: Mapping[str, Any] | None = None,
        fields_meta: Sequence[Any] | None = None,
    ) -> list[Any]:
        """#5 统一读前语义门禁：所有 read engine 先过同一组检查。

        顺序：
            1. temporal contract（``_enforce_read_contract``）——mode/稀疏/事件；
            2. event cutoff（``_event_cutoff_for_contract``）；
            3. required_filters（catalog 字段声明，如行业 IndustrySource / 美股
               财务 timeframe / IndexSymbol）——**所有引擎一致**，不再只有
               read_result/read_joined 强制；
            4. allowed_filter_values 值校验。

        返回解析出的 fields_meta（供 normalize 等复用；无 columns 时为空）。
        """
        self._enforce_read_contract(
            dataset,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
        )
        self._event_cutoff_for_contract(dataset, time_range=time_range)
        if fields_meta is None:
            fields_meta = self._fields_meta_for_columns(dataset, columns)
        # #P0-10 数据集级契约门：required_filters / allowed_filter_values 与
        # 用户选了哪些列无关——columns=None 或 catalog 未登记的物理直读列也必须
        # 强制（US 财务 timeframe / 行业 IndustrySource / IndexSymbol）。
        enforce_meta = list(fields_meta or [])
        ds_required = self._dataset_required_filters(dataset)
        if ds_required:
            from data_access.read.semantic_catalog import SemanticField

            covered = {
                k
                for f in enforce_meta
                for k in (getattr(f, "required_filters", ()) or ())
            }
            extra = [k for k in ds_required if k not in covered]
            if extra:
                enforce_meta.append(
                    SemanticField(
                        logical_name="<dataset>",
                        dataset=dataset,
                        physical_name=None,
                        required_filters=tuple(extra),
                    )
                )
        if enforce_meta:
            params_by_dataset = {str(dataset): dict(params or {})}
            self._enforce_required_filters(
                enforce_meta,
                params_by_dataset,
                filters=filters,
            )
            self._validate_allowed_filter_values(
                enforce_meta,
                params_by_dataset,
                filters=filters,
            )
        return list(fields_meta)

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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        **params: Any,
    ) -> ReadResult:
        """读取数据集并返回带 ``DataSnapshot`` 的 ``ReadResult``（审计/lineage 用）。

        ``mode``（#44）：auto|panel|event|pit|dimension|sparse——读取的语义模式，
        决定 COS 契约如何强制（事件表不能当普通面板读等）。
        """
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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        physical_scope: str | Sequence[str] | None = None,
    ) -> ReadResult:
        """按已解析的 ``Dataset`` 对象执行一次读（read_result / read_uri 共用）。"""
        # #5 统一读前语义门禁：temporal contract + event cutoff + required_filters
        # + allowed_filter_values（read_arrow_stream / scan_polars 也走同一入口）
        self._prepare_read_request(
            dataset,
            columns=columns,
            time_range=time_range,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            filters=filters,
            params=params,
        )
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        if physical_scope is not None:
            scope = (
                list(physical_scope)
                if isinstance(physical_scope, (list, tuple))
                else [str(physical_scope)]
            )
            # #6/#17：精确 URI scope 冻结一次，execution 与 snapshot 消费同一份
            paths = self._expand_glob_paths(scope)
        else:
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_range,
                params=params,
                instrument_filter=instrument_filter,
            )
            # #6 冻结 glob → 精确文件列表：DuckDB 不再二次 expand（TOCTOU）
            paths = self._expand_glob_paths(paths)
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        physical_scope: str | Sequence[str] | None = None,
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
        # #5 流式读同样强制 temporal contract + required_filters + allowed values
        self._prepare_read_request(
            dataset,
            columns=columns,
            time_range=time_range,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            filters=filters,
            params=params,
        )
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        if physical_scope is not None:
            scope = (
                list(physical_scope)
                if isinstance(physical_scope, (list, tuple))
                else [str(physical_scope)]
            )
            paths = self._expand_glob_paths(scope)
        else:
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_range,
                params=params,
                instrument_filter=instrument_filter,
            )
            paths = self._expand_glob_paths(paths)
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
        reader = self._engine.execute_reader(
            sql,
            sql_params,
            batch_size=batch_size,
            # #P0-20 deadline 作用在第一批之前：engine 侧 watchdog interrupt
            # cursor，避免「第一批就卡死 5 分钟，根本没机会检查 elapsed」。
            deadline_ms=budget.max_elapsed_ms,
        )
        start = time.perf_counter()
        total_rows = 0
        total_bytes = 0
        ok = False
        err_msg: str | None = None

        def _iter_batches() -> Iterator[pa.RecordBatch]:
            nonlocal total_rows, total_bytes, ok, err_msg
            try:
                # #P0-20 第一批产出前先查一次 elapsed（覆盖 engine watchdog
                # 未触发的边界：预算超时优先于 deadline interrupt 的情形）。
                enforce_stream_budget(
                    budget,
                    total_rows=0,
                    total_bytes=0,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
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
                # #P0-20 用户 break / 异常：显式关闭 reader（interrupt cursor、
                # 归还连接、结束 watchdog），不留悬挂游标。
                try:
                    close = getattr(reader, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        physical_scope: str | Sequence[str] | None = None,
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
        # #5 Polars 扫描同样强制 temporal contract + required_filters + allowed values
        self._prepare_read_request(
            dataset,
            columns=columns,
            time_range=time_range,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            filters=filters,
            params=params,
        )
        _assert_instrument_filter_supported(ds, instrument_filter)
        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        if physical_scope is not None:
            scope = (
                list(physical_scope)
                if isinstance(physical_scope, (list, tuple))
                else [str(physical_scope)]
            )
            paths = self._expand_glob_paths(scope)
        else:
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_range,
                params=params,
                instrument_filter=instrument_filter,
            )
            # #6/#7 冻结一次：LazyFrame 绑定精确文件清单，collect 前不会因新文件
            # 而读到 snapshot 没记录的数据（ScanHandle.collect 再 revalidate mtime）。
            paths = self._expand_glob_paths(paths)
        # scan_polars 也走 schema 自检：发现声明漂移尽早报。
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)
        self._ensure_schema(ds, paths, params, files=files)

        # #29 按 FormatSpec 选 Polars scan 函数：旧代码一律 scan_parquet，
        # CSV/TSV/JSONL 数据集如果路由到 Polars 会被 scan_parquet 处理报错。
        _fmt = str(getattr(ds, "format", "parquet") or "parquet").strip().lower()
        scan_kwargs: dict[str, Any] = {}
        if _fmt in {"parquet", "pq"}:
            # hive_partitioning 只有 parquet 扫描支持（scan_csv/scan_ndjson 没有
            # 这个参数，传了会 TypeError）
            scan_kwargs["hive_partitioning"] = ds.hive_partitioning
            _scan = pl_mod.scan_parquet
        elif _fmt in {"csv"}:
            _scan = pl_mod.scan_csv
        elif _fmt in {"tsv", "tab"}:
            _scan = lambda p, **kw: pl_mod.scan_csv(p, separator="\t", **kw)  # noqa: E731
        elif _fmt in {"jsonl", "ndjson", "json"}:
            _scan = pl_mod.scan_ndjson
        else:
            _scan = pl_mod.scan_parquet
        # pl.scan_* 可以接 list[str]，也支持 glob。我们传 list 给它。
        # 跨年份列不一致时靠 union_by_name（只对 parquet 有意义；scan_csv/
        # scan_ndjson 没有 missing_columns/allow_missing_columns 参数）：
        #   - Polars 1.30+：参数是 missing_columns='insert'/'raise'，老的 allow_missing_columns 被弃用
        #   - Polars 1.28 ~ 1.29：参数名是 allow_missing_columns
        # 用 try/except 兜两边，避免硬编码版本号判断。
        if _fmt in {"parquet", "pq"}:
            try:
                lf = _scan(
                    paths,
                    missing_columns="insert" if ds.union_by_name else "raise",
                    **scan_kwargs,
                )
            except TypeError:
                lf = _scan(
                    paths,
                    allow_missing_columns=ds.union_by_name,
                    **scan_kwargs,
                )
        else:
            lf = _scan(paths, **scan_kwargs)

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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        **params: Any,
    ) -> ReadHandle:
        """统一读入口：引擎/结果形态自动路由，返回 ``ReadHandle``。

        ``mode``（#44）：auto|panel|event|pit|dimension|sparse——语义读取模式，
        COS 契约按此强制（事件表不能当普通面板读；X0 需要 allow_sparse）。

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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
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
        **production / strict 读模式**（#6）：URI 必须**唯一反查到已登记数据集**
        并复用其 Dataset Contract（temporal model / PIT / required_filters），否则
        拒绝——禁止用 ``_uri:parquet:xxx`` 临时 Dataset 绕过财务 PIT / 行业过滤等
        语义门禁。

        ``format`` 不传时按扩展名推断（parquet/csv/tsv/jsonl/arrow/feather）。
        arrow/feather 自动走 PyArrow 引擎。
        """
        from data_access.read.formats import normalize_format_name

        uri = str(uri)  # 兼容 Path 对象
        fmt = _infer_format_from_uri(uri, format)
        self._assert_uri_allowed(uri, format=fmt)
        strict = is_strict_semantics()
        if strict:
            registered = self._resolve_registered_dataset_for_uri(uri, fmt)
            if registered is None:
                raise ValidationError(
                    f"production/strict 下 read_uri 必须唯一对应已登记数据集；"
                    f"{uri!r} 无法唯一匹配。请改用 store.read(<dataset>, ...) 或先"
                    f"在 datasets.yaml 登记该数据集，再走其 Contract 读取。"
                )
            rds = self._registry.get(registered)
            return self._read_handle(
                rds,
                dataset=registered,
                registered_name=registered,
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
                # #17 精确 URI 作为 physical scope：Contract 只提供语义门禁，
                # 物理读取范围仍限定调用方指定的文件——绝不读整个 dataset glob。
                physical_scope=uri,
            )
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

    def _resolve_registered_dataset_for_uri(self, uri: str, fmt: str) -> str | None:
        """#6 production read_uri：URI → 唯一已登记数据集（复用其 Contract）。

        匹配规则：URI 必须落在数据集的 root / static_root 下，且格式一致。
        多个候选 → 返回 None（调用方拒绝，禁止 YAML 顺序决定语义）。

        #18：``s3://``/``cos://`` 对象 URI 不能用 ``pathlib.Path``（那是本地
        文件系统语义，``Path("s3://bucket/key")`` 会被当本地路径解析）——按
        字符串前缀匹配。本地路径才走 Path + resolve。
        #19：格式匹配用 ``fmt == auto or fmt == dformat``。旧代码
        ``fmt not in {"auto", dformat, "parquet"}`` 里 ``"parquet"`` 恒在集合
        中——``read_uri(..., format="parquet")`` 会让 CSV 数据集也成候选。
        """
        from pathlib import Path

        from data_access.registry import ParametricDataset, StaticDataset

        is_object = uri.startswith("s3://") or uri.startswith("cos://")
        candidates: list[str] = []
        for name in self._registry.names():
            dsobj = self._registry.get(name)
            dformat = str(getattr(dsobj, "format", "parquet")).lower()
            if dformat in {"pq"}:
                dformat = "parquet"
            if fmt != "auto" and fmt != dformat:
                continue
            if isinstance(dsobj, StaticDataset):
                root = getattr(dsobj, "root", None)
            elif isinstance(dsobj, ParametricDataset):
                root = getattr(dsobj, "static_root", None)
            else:
                continue
            if root is None:
                continue
            if is_object:
                # 对象 URI：字符串前缀（cos://bucket/prefix 或 s3://bucket/prefix）
                if not uri.startswith(str(root)):
                    continue
            else:
                try:
                    Path(uri).expanduser().resolve().relative_to(Path(root).resolve())
                except (ValueError, OSError):
                    continue
            candidates.append(name)
        return candidates[0] if len(candidates) == 1 else None

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
        order_by: Sequence[str] | None = None,
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
        # #P0-2 统一 effective_join_specs：SemanticField 默认 → COS Contract 默认
        # → 显式 joins/join_specs 覆盖。PIT validator / plan / 组合执行消费同一份。
        effective_joins = self._effective_join_specs(per_ds, fields_meta, joins)
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
        # #52 allowed_filter_values 值校验 + #44 每数据集契约（EMPTY/必需过滤）
        self._validate_allowed_filter_values(
            fields_meta, pbd, filters=filters, filters_by_dataset=filters_by_dataset
        )
        self._validate_joined_contract_filters(
            per_ds, pbd, filters=filters, filters_by_dataset=filters_by_dataset
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
            order_by=order_by,
        )

        snapshots = []
        missing_snapshot: list[str] = []
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
            except Exception as exc:
                missing_snapshot.append(ds)
                if is_strict_semantics():
                    # #14 fail-closed：参与数据集 snapshot 必须完整，否则 lineage/
                    # replay/cache/audit 都不可靠——禁止「查询成功但只记了部分快照」。
                    raise SnapshotBuildError(
                        f"read_joined 参与数据集 '{ds}' snapshot 构建失败："
                        f"{type(exc).__name__}: {exc}。"
                        "查询已中止（production fail-closed）。"
                    ) from exc
        if not snapshots:
            raise SnapshotBuildError(
                "read_joined 没有任何参与数据集成功构建 snapshot；"
                "多数据集读取的 lineage/缓存将不可靠，禁止静默继续。"
            )
        if missing_snapshot:
            logger.warning(
                "read_joined snapshot 不完整（research 放行）：缺失 %s",
                missing_snapshot,
            )
        snapshot = merge_sql_data_snapshots(
            snapshots, registry_hash=self.registry_fingerprint()
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

        **#7 production/strict**：与 ``store.sql()`` 一样走 SQL 沙箱——必须显式声明
        ``snapshot_datasets``（= read_datasets），SQL 用 ``validate_sql_sandbox`` 挡
        掉 read_parquet/COPY/ATTACH/LOAD/INSTALL 等，禁止 ``FROM '/some/path.parquet'``
        绕过 DataAccess。
        """
        from data_access.read.relation_handle import RelationHandle

        if is_strict_semantics():
            from data_access.read import sql_escape

            if not snapshot_datasets:
                raise ValidationError(
                    "production/strict 下 sql_relation 必须显式声明 snapshot_datasets"
                    "（= read_datasets），与 store.sql() 的治理一致。"
                )
            sql_escape.validate_sql_sandbox(sql)
            # #P0-18 snapshot_datasets 是真实数据边界：FROM/JOIN 每一张表都必须在
            # 声明集合内（token 黑名单挡不住 FROM 未声明表绕过 DataAccess）。
            sql_escape.assert_sql_tables_declared(sql, snapshot_datasets)

        snapshot = lineage = None
        if snapshot_datasets:
            from data_access.read.read_contract import (
                SqlReadLineage,
                merge_sql_data_snapshots,
            )

            snapshots = []
            missing_snapshot: list[str] = []
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
                except Exception as exc:
                    missing_snapshot.append(ds)
                    if is_strict_semantics():
                        raise SnapshotBuildError(
                            f"sql_relation 参与数据集 '{ds}' snapshot 构建失败："
                            f"{type(exc).__name__}: {exc}（production fail-closed）。"
                        ) from exc
            if not snapshots:
                raise SnapshotBuildError(
                    "sql_relation 没有任何参与数据集成功构建 snapshot"
                    "（production fail-closed）。"
                )
            if missing_snapshot:
                logger.warning(
                    "sql_relation snapshot 不完整（research 放行）：缺失 %s",
                    missing_snapshot,
                )
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

    def _validate_pit_semantics(
        self,
        fields: Sequence[Any],
        *,
        anchor: str,
        joins: Mapping[str, Any] | None,
    ) -> None:
        """#10 ``DataRequest(pit=True)`` 强语义：每个非 anchor 字段必须可证明 PIT 安全。

        规则（production/strict fail-closed，research 告警放行）：
            - 非 anchor 字段的 join 策略必须是 asof / pit_asof（backward-looking，
              ``decision_time >= knowledge_time`` 天然成立）；exact join 无法证明
              可用性 → 拒绝；
            - 字段必须能解析出 temporal contract（knowledge_time 或 join 规格），
              无任何时间可见性语义 → 拒绝；
            - 显式 ``joins`` 提供的规格优先于字段自身声明的 join 语义。
        """
        from data_access.read.temporal_join import parse_join_spec

        strict = is_strict_semantics()
        problems: list[str] = []
        for f in fields:
            ds = getattr(f, "dataset", None)
            if not ds or ds == anchor:
                continue  # 锚点即决策侧，不需要 join 证明
            spec = parse_join_spec(
                dict(joins or {}).get(ds) if joins and ds in joins else None
            )
            if spec is None or spec.policy == "exact":
                problems.append(
                    f"字段 '{getattr(f, 'logical_name', '?')}'（{ds}.{f.physical_name}）"
                    "用 exact join，无法证明 PIT 可用性（decision_time >= "
                    "availability_time）。pit=True 要求 asof/pit_asof。"
                )
                continue
            # 时间可见性契约：字段声明（knowledge/effective/period）或数据集
            # COS 契约（availability_column/period_column）至少存在一个。
            has_contract = bool(
                getattr(f, "knowledge_time", None)
                or getattr(f, "effective_time", None)
                or getattr(f, "period_time", None)
                or getattr(f, "temporal_model", None)
                or getattr(f, "time_role", None) in {"knowledge_time", "effective_time"}
                or spec.knowledge_time
            )
            if not has_contract:
                try:
                    from data_access.cos_contract import get_cos_contract

                    c = get_cos_contract(ds)
                    has_contract = bool(
                        c is not None
                        and (c.availability_column or c.period_column)
                    )
                except Exception:
                    pass
            if not has_contract:
                problems.append(
                    f"字段 '{getattr(f, 'logical_name', '?')}'（{ds}）无 temporal "
                    "contract（knowledge_time/availability/period 均缺失），无法证明"
                    " PIT 安全。"
                )
        if not problems:
            return
        detail = "; ".join(problems)
        if strict:
            raise ValidationError(
                f"DataRequest(pit=True) 语义不满足（production fail-closed）：{detail}"
            )
        logger.warning("DataRequest(pit=True) 语义不满足（research 放行）：%s", detail)

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
            found: list[tuple[str, str]] = []
            for dsn in self._registry.names():
                d = self._registry.get(dsn)
                schema = getattr(d, "schema", None) or {}
                if name in schema:
                    found.append((dsn, schema[name]))
            if not found:
                raise ValidationError(
                    f"字段 '{name}' 未在 SemanticFieldCatalog，也不在任何数据集 schema 中。"
                    f"请先在 config/semantic_fields.yaml 登记逻辑字段。"
                )
            if len(found) > 1:
                # #33 fail ambiguous：不能「取 registry 第一个」——Close/Symbol/
                # TradeDate 几十张表都有，YAML 顺序决定语义是隐患。
                datasets_with_col = sorted({dsn for dsn, _ in found})
                from data_access.core.exceptions import AmbiguousFieldError

                msg = (
                    f"字段 '{name}' 在多个数据集都有物理列 "
                    f"({datasets_with_col})。请用 'dataset.column' 限定名或传 "
                    f"dataset= 消歧；禁止取 registry 第一个。"
                )
                if is_strict_semantics():
                    raise AmbiguousFieldError(msg)
                logger.warning("%s（research 放行，取第一个）", msg)
            out.append(
                SemanticField(
                    logical_name=name, dataset=found[0][0], physical_name=name, dtype=found[0][1]
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
        # #4 不写回 request.anchor：plan() 不应把调用方的请求改掉（编译后
        # 调用方还能用原 request 做别的事；ReadPlan.anchor 由 datasets[0] 兜底）。

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
        # #P0-2 统一 effective_join_specs（字段语义 → COS 契约 → 显式覆盖），
        # join_policies / PIT validator / 组合执行消费同一份。
        effective_specs = self._effective_join_specs(per_ds, fields, raw_joins)
        joins: dict[str, str] = {}
        for ds in datasets:
            if ds == anchor:
                joins[ds] = "exact"
            else:
                joins[ds] = parse_join_spec(effective_specs.get(ds)).policy

        # #10：DataRequest(pit=True) 强语义——每个非 anchor 字段必须可证明 PIT 安全。
        if getattr(request, "pit", False):
            self._validate_pit_semantics(fields, anchor=anchor, joins=effective_specs)

        scan_costs: dict[str, Any] = {}
        storage: dict[str, str] = {}
        snapshot_info: dict[str, dict[str, Any]] = {}
        plan_snapshot_tokens: dict[str, dict[str, Any]] = {}
        for ds in datasets:
            dsobj = self._registry.get(ds)
            # #27 plan() 按每 dataset 传 source_params：factor lake / model output
            # 这类 ParametricDataset 不传参数会得到错误成本 / 无 manifest。
            ds_params = request.dataset_params(ds)
            try:
                cost = estimate_scan_cost(
                    self,
                    ds,
                    columns=per_ds.get(ds) or None,
                    time_range=request.time_range,
                    instrument_filter=request.instruments,
                    **ds_params,
                )
            except Exception:
                cost = None
            scan_costs[ds] = cost
            storage[ds] = storage_description(dsobj)
            try:
                token = self.manifest_version(ds, **ds_params)
                snapshot_info[ds] = token
                plan_snapshot_tokens[ds] = dict(token)
            except Exception:
                snapshot_info[ds] = {"has_manifest": False}
                plan_snapshot_tokens[ds] = {"has_manifest": False}

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

        # #4 编译物理计划 DAG（Scan→Filter→TemporalJoin→Aggregation→Normalize→Project）
        from data_access.read.physical_plan import build_physical_plan

        physical = build_physical_plan(
            request=request,
            fields=fields,
            datasets=datasets,
            join_policies=joins,
            scan_costs=scan_costs,
        )

        # #4 编译期冻结请求语义（不可变）：execute 只消费 compiled，不读活的 req。
        from data_access.read.data_request import compile_data_request

        compiled = compile_data_request(request)
        snapshot_policy = compiled.snapshot_policy
        if snapshot_policy not in {
            "latest",
            "fail_if_changed",
            "pin",
        }:
            raise ValidationError(
                f"snapshot_policy 必须是 latest|fail_if_changed|pin，收到 {snapshot_policy!r}"
            )

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
            physical=physical,
            join_specs_effective=effective_specs,
            compiled=compiled,
            snapshot_policy=snapshot_policy,
            plan_snapshot_tokens=plan_snapshot_tokens,
            _store=self,
        )

    def pit_event_index(
        self,
        dataset: str,
        *,
        ticker_column: str | None = None,
        filing_column: str | None = None,
        period_column: str | None = None,
        timeframe_column: str | None = None,
        time_range: tuple[Any, Any] | None = None,
        timeframe_filter: str | None = None,
        force: bool = False,
        limit: int | None = None,
    ) -> Any:
        """#8 美股财务 PIT 事件索引（构建或加载 sidecar）。

        记录 ``(ticker, filing_date, period_end, timeframe)``，使文件按
        ``period_end`` 命名时仍能按 ``filing_date`` 裁剪。返回 ``PITEventIndex``。

        列名缺省按 COS 契约推断（us_stock_balance 等 E2 契约自带
        filing_date/period_end）；无契约时可显式传入列名。
        """
        from data_access.read.pit_event_index import build_pit_event_index

        return build_pit_event_index(
            self,
            dataset,
            ticker_column=ticker_column,
            filing_column=filing_column,
            period_column=period_column,
            timeframe_column=timeframe_column,
            time_range=time_range,
            timeframe_filter=timeframe_filter,
            force=force,
            limit=limit,
        )

    def prune_pit_paths(
        self,
        dataset: str,
        *,
        filing_range: tuple[Any, Any] | None = None,
        timeframe: str | None = None,
        tickers: Sequence[str] | None = None,
    ) -> list[str]:
        """#8 用 PIT 索引按 filing_date 精准裁剪文件路径。

        返回匹配文件的路径集合；无索引返回空列表（调用方可回退全量路径）。
        """
        from data_access.read.pit_event_index import prune_paths_by_filing_range

        return prune_paths_by_filing_range(
            self,
            dataset,
            filing_range=filing_range,
            timeframe=timeframe,
            tickers=tickers,
        )

    def materialize_daily_aggregate(
        self,
        *,
        source_dataset: str,
        serving_dataset: str,
        time_column: str,
        instrument_column: str,
        groupby: Sequence[str],
        value_columns: Mapping[str, Sequence[str]],
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        mode: str = "overwrite",
        partition_by: Sequence[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """#30 高扇出源表 → 日频预聚合 serving 数据集。"""
        from data_access.cos.serving import materialize_daily_aggregate as _m

        return _m(
            self,
            source_dataset=source_dataset,
            serving_dataset=serving_dataset,
            time_column=time_column,
            instrument_column=instrument_column,
            groupby=groupby,
            value_columns=value_columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            mode=mode,
            partition_by=partition_by,
            **params,
        )

    @staticmethod
    def route_minute_storage(**kwargs: Any) -> str:
        """#17 分钟级 date-major vs bucket-major 存储路由。"""
        from data_access.cos.serving import route_minute_storage as _r

        return _r(**kwargs)

    def enable_result_cache(self, enabled: bool = True) -> None:
        """进程级开启/关闭查询结果缓存（read_cached 使用）。"""
        from data_access.read.query_cache import set_result_cache_enabled

        set_result_cache_enabled(enabled)

    def read_cached(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        filters: Any = None,
        limit: int | None = None,
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        normalize_units: bool = False,
        **params: Any,
    ) -> Any:
        """带查询结果缓存的 read_arrow（默认关闭，enable_result_cache(True) 开启）。

        key 含 manifest source_epoch（写路径 bump 后自动失效）+ **canonical
        Filter AST hash** + limit + mode + allow_sparse + allow_effective_time +
        normalize_units + 语义 catalog 指纹 + Contract IR 指纹（#3：缺这些会命中
        错误结果）。命中返回 Arrow Table；未命中走 read_arrow 并写入缓存。
        """
        from data_access.read.query_cache import (
            get_query_cache,
            query_cache_key,
            result_cache_enabled,
        )

        if not result_cache_enabled():
            return self.read_arrow(
                dataset,
                columns=columns,
                time_range=time_range,
                instrument_filter=instrument_filter,
                filters=filters,
                limit=limit,
                mode=mode,
                allow_sparse=allow_sparse,
                allow_effective_time=allow_effective_time,
                normalize_units=normalize_units,
                **params,
            )
        cache = get_query_cache()
        try:
            token = self.manifest_version(dataset, **params)
        except Exception:
            token = None
        # #25 无权威 source version → 不缓存：manifest 拿不到就给弱 token 仍缓存
        # 的话，底层数据被外部系统修改（不经 DataAccess epoch）缓存只能等 TTL。
        # 这里没有 authoritative manifest（不存在 / 不 fresh）就完全跳过缓存。
        if token is None or not token.get("has_manifest") or not token.get("fresh"):
            return self.read_arrow(
                dataset,
                columns=columns,
                time_range=time_range,
                instrument_filter=instrument_filter,
                filters=filters,
                limit=limit,
                mode=mode,
                allow_sparse=allow_sparse,
                allow_effective_time=allow_effective_time,
                normalize_units=normalize_units,
                **params,
            )
        try:
            from data_access.read.semantic_catalog import get_semantic_catalog

            cat_fp = get_semantic_catalog().fingerprint()
        except Exception:
            cat_fp = None
        try:
            ir_fp = self.contract_ir_fingerprint()
        except Exception:
            ir_fp = None
        # #P0-36 默认列显式展开：columns=None（全列）的 key 用 registry schema
        # 列清单，schema 变化（新增列）不会命中旧缓存。
        key_columns = columns
        if key_columns is None:
            try:
                ds = self._registry.get(dataset)
                schema_cols = list((getattr(ds, "schema", None) or {}).keys())
                if schema_cols:
                    key_columns = schema_cols
            except Exception:
                key_columns = None
        key = query_cache_key(
            dataset=dataset,
            params=dict(params),
            time_range=time_range,
            instruments=instrument_filter,
            columns=key_columns,
            manifest_token=token,
            filters=filters,
            limit=limit,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            normalize_units=normalize_units,
            catalog_fingerprint=cat_fp,
            contract_ir_fingerprint=ir_fp,
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
        table = self.read_arrow(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            filters=filters,
            limit=limit,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            normalize_units=normalize_units,
            **params,
        )
        if table.num_rows is not None:
            cache.set_with_size(key, table, nbytes=table.nbytes)
        return table

    def coverage(self, dataset: str, **params: Any) -> Any:
        """#14 数据集覆盖/完整性报告：complete/partial/unavailable + 问题列表。"""
        from data_access.read.coverage import compute_coverage

        return compute_coverage(self, dataset, params=params)

    def metadata_plane(self, dataset: str, **params: Any) -> Any:
        """#38 统一元数据访问层：manifest/coverage/PIT index/schema/contract 一处拿。"""
        from data_access.read.metadata_plane import DatasetMetadataPlane

        return DatasetMetadataPlane(self, dataset, params=dict(params))

    def contract_ir(self) -> Any:
        """#12 统一 Contract IR：registry + COS 契约 + 语义字段 的合并视图。"""
        from data_access.read.contract_ir import build_contract_ir
        from data_access.read.semantic_catalog import get_semantic_catalog

        return build_contract_ir(
            self._registry,
            catalog=get_semantic_catalog(),
        )

    def contract_ir_fingerprint(self) -> str:
        """#12 Contract IR 稳定指纹（跨服务器对比语义版本）。"""
        return self.contract_ir().fingerprint()

    def manifest_version(self, dataset: str, **params: Any) -> dict[str, Any]:
        """廉价的 query-scoped snapshot token：只读 ``_manifest.json`` sidecar，
        不做全量 footer 扫描，也不做 O(N) 文件 glob（#17）。

        freshness 由写路径维护的双 epoch 决定：仅当 ``source_epoch ==
        manifest_built_epoch`` 时 manifest 可信（数据变更时写路径 bump
        source_epoch）。返回 ``{has_manifest, fresh, dataset_version,
        partition_version, file_count, source_epoch, manifest_built_epoch,
        manifest_epoch, created_at}``。
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
        src = token.get("source_epoch")
        built = token.get("manifest_built_epoch")
        legacy = token.get("manifest_epoch")
        if src is None and built is None and legacy is not None:
            src = built = legacy
        return {
            "dataset": dataset,
            "has_manifest": True,
            "fresh": (src is not None and src == built),
            "dataset_version": token.get("dataset_version"),
            "partition_version": token.get("partition_version"),
            "file_count": token.get("file_count"),
            "source_epoch": src,
            "manifest_built_epoch": built,
            "manifest_epoch": src if src is not None else legacy,
            "created_at": token.get("created_at"),
        }

    def touch_manifest_epoch(self, dataset: str, **params: Any) -> str | None:
        """兼容别名（新代码请用 ``_dataset_mutation``）：递增 source_epoch。

        只使 manifest 失效（dirty），不重建。返回新 source_epoch；无 manifest
        返回 None。query-scoped snapshot 在数据变更后立刻判定过期。
        """
        from data_access.read.manifest import (
            bump_source_epoch,
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
        return bump_source_epoch(root)

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

        优先比较 source_epoch（写路径 bump 后立刻失效）；当前 manifest 不新鲜
        （source != built）也视为过期。无 epoch 时退回 dataset_version /
        partition_version 对比（老 manifest 兼容）。
        """
        cur = self.manifest_version(dataset, **params)
        if not cur.get("has_manifest"):
            return True
        if not cur.get("fresh"):
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

    def _effective_join_specs(
        self,
        per_ds: Mapping[str, Sequence[str]],
        fields_meta: Sequence[Any],
        explicit: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """#P0-2 生成全链路共享的 effective_join_specs。

        合成顺序（低→高优先级）：
            1. SemanticField 默认（join_policy/knowledge_time/period_time/
               revision_order/availability/period_selection → join_spec_from_field）；
            2. 同 dataset 多个字段合并（非 exact 优先，已由调用方 fields_meta 保序）；
            3. COS Contract 默认（pit_policy 向后 / availability_column /
               period_column / revision_columns → pit_asof 语义）——没有字段语义
               时不再默默退化成 exact；
            4. 显式 request.joins / request.join_specs 覆盖。

        PIT validator / PhysicalPlan / read_joined / 组合执行 / explain 全部消费
        同一份结果，不再各自推导。
        """
        from data_access.cos_contract import get_cos_contract
        from data_access.read.temporal_join import (
            TemporalJoinSpec,
            join_spec_from_field,
            parse_join_spec,
        )

        specs: dict[str, Any] = {}
        for f in fields_meta:
            ds = getattr(f, "dataset", None)
            if not ds or ds not in per_ds:
                continue
            spec = join_spec_from_field(f)
            if spec is None or spec.policy == "exact":
                continue
            cur = specs.get(ds)
            if cur is None or getattr(cur, "policy", "exact") != "pit_asof":
                specs[ds] = spec
        # 数据集级 COS 契约默认：即使字段没给语义，契约声明了 PIT 语义也走 pit_asof。
        for ds in per_ds:
            if ds in specs:
                continue
            try:
                ct = get_cos_contract(ds)
            except Exception:
                ct = None
            if ct is None:
                continue
            pit_backward = bool(
                ct.pit_policy in {"pit_asof", "pit_asof_backward", "asof", "asof_backward"}
                or ct.availability_column
                or ct.period_column
            )
            if pit_backward:
                specs[ds] = TemporalJoinSpec(
                    policy="pit_asof",
                    knowledge_time=ct.availability_column or ct.period_column,
                    period_time=ct.period_column,
                    revision_order=tuple(ct.revision_columns or ()),
                )
        # 显式覆盖（最高优先）
        for ds, raw in (explicit or {}).items():
            if ds not in per_ds:
                continue
            specs[ds] = parse_join_spec(raw)
        return specs

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
        anchor_override: Mapping[str, Any] | None = None,
        order_by: Sequence[str] | None = None,
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
        from data_access.read.predicate_ast import (
            filter_columns,
            filter_column_values,
            parse_filters,
        )
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
        # #P0-1 anchor_override：聚合锚点已物化（临时 parquet），锚点的
        # 行空间/时间列/标的列来自物化结果，不再走 registry 的 time_column。
        if anchor_override is not None:
            anchor_time = anchor_override.get("time_column") or "ts"
            anchor_inst = anchor_override.get("instrument_column") or "inst"
        else:
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
                # #P0-9 查询时钟（PIT 分支是 knowledge_time，可能 != ds.time_column）
                time_column=time_col,
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
            use_session_avail = False
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
            # #45 period_selection 需要 period_time 列参与 running-max
            if spec.needs_period_selection and spec.period_time not in select_cols:
                select_cols.append(spec.period_time)
            select_list = ", ".join(_quote_ident(c) for c in select_cols)

            ds_params = dict(params_by_dataset.get(ds, {}))
            # 锚点通用 filters + 每数据集 filters 合并
            ds_filters = _and_filters(
                parse_filters(filters) if ds == anchor else None,
                parse_filters(
                    (filters_by_dataset or {}).get(ds) if filters_by_dataset else None
                ),
            )
            # #53 join fan-out 守卫：exact join 右表相对面板 one_to_many 时，
            # production 拒绝静默行放大（除非维度过滤已 resolve / join key 覆盖唯一键）。
            if ds != anchor and policy == "exact":
                from data_access.cos_contract import get_cos_contract, validate_join_fanout

                fcontract = get_cos_contract(ds)
                if fcontract is not None:
                    applied_cols = set(filter_columns(parse_filters(filters)))
                    applied_cols |= filter_columns(ds_filters)
                    applied_vals: dict[str, set[Any]] = {}
                    for _c, _v in filter_column_values(parse_filters(filters)).items():
                        applied_vals.setdefault(_c, set()).update(_v)
                    for _c, _v in filter_column_values(ds_filters).items():
                        applied_vals.setdefault(_c, set()).update(_v)
                    validate_join_fanout(
                        fcontract,
                        join_key=tuple(k for k in (t_col, inst_col) if k),
                        production=is_strict_semantics(),
                        applied_filter_columns=sorted(applied_cols),
                        # #P0-14 单值证明：IN 多值不算已 resolve
                        applied_filter_values=applied_vals,
                    )
            # instrument_filter 对锚点/右表一律下推（join key 同为 instrument 轴）
            ds_inst = instrument_filter
            hive = self._bucket_hive_filters(dsobj, ds_inst)
            time_type = (
                str((dsobj.schema or {}).get(t_col or "", "")).lower() if t_col else ""
            )
            time_is_ts = "timestamp" in time_type or "datetime" in time_type

            if anchor_override is not None and ds == anchor:
                # #P0-1 聚合锚点已物化：anchor 子查询 = 临时 parquet（行空间
                # ts/inst + 聚合输出列）。锚点的 filter/instrument 已在聚合时
                # 应用，这里不再走 registry 分支；右表 join 语义照常完整。
                sub = "SELECT * FROM read_parquet(?)"
                params_list.append(anchor_override["path"])
                per_ds_paths[ds] = list(anchor_override.get("source_paths", []))
            elif ds == anchor or policy == "exact":
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
                # #45 period_selection：先选报告期，再去重
                if spec.needs_period_selection:
                    sub, _pp = _period_selection_sql(
                        sub,
                        inst_col=inst_col,
                        period_col=spec.period_time,
                        knowledge_col=t_col,
                        selection=spec.period_selection,
                        period_values=spec.period_values,
                        period_is_text=_column_is_text(dsobj, spec.period_time),
                    )
                    params_list.extend(_pp)
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
                    # #16 future_cutoff：effective_time_only 事件右表不读未来生效
                    # 事件（美股 Dividend 甚至有未来日期文件）。
                    if spec.future_cutoff:
                        from data_access.cos_contract import get_cos_contract as _gcc

                        _ct = _gcc(ds)
                        if _ct is not None and _ct.pit_policy == "effective_time_only":
                            end = _min_date_bound(end)
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
                    seed_sql, seed_params = _seed_qualify_sql(
                        seed[0],
                        inst_col,
                        right_time,
                        spec.revision_order,
                        period_col=spec.period_time,
                        period_selection=spec.period_selection,
                        period_values=spec.period_values,
                        period_is_text=_column_is_text(dsobj, spec.period_time),
                    )
                    branches.append(seed_sql)
                    params_list.extend(seed[1])
                    params_list.extend(seed_params)
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
                # #45 period_selection 必须在 revision 去重之前：running_max 需要
                # 看到全部 revision 才能正确标记「该 period 是否已成为峰值」。
                if spec.needs_period_selection:
                    sub, _pp = _period_selection_sql(
                        sub,
                        inst_col=inst_col,
                        period_col=spec.period_time,
                        knowledge_col=right_time,
                        selection=spec.period_selection,
                        period_values=spec.period_values,
                        period_is_text=_column_is_text(dsobj, spec.period_time),
                    )
                    params_list.extend(_pp)
                # 跨窗口/seed 统一按 (instrument, knowledge_time) 去重最新 revision
                if spec.deduplicate and spec.revision_order and inst_col and right_time:
                    sub = _dedup_key_sql(sub, inst_col, right_time, spec.revision_order)
                # #46 / #P0-6 / #P0-7：用交易日历把 knowledge 编译成 available_from
                # （decision >= available_from）。覆盖 session + 全部粒度 next_* 种类
                # （next_trading_day / next_session_open / next_bar /
                # after_close_next_open）——不再只对旧 "session" 生效。
                if spec.is_calendar_availability and not use_session_avail:
                    market = _market_of_dataset(ds)
                    cal = self.get_calendar(market) if market else None
                    if cal is not None:
                        wrapped, _cp, ok = _session_avail_sql(
                            sub,
                            right_time_col=right_time,
                            calendar=cal,
                            start=time_range[0] if time_range else None,
                            end=time_range[1] if time_range else None,
                        )
                        if ok:
                            sub = wrapped
                            params_list.extend(_cp)
                            use_session_avail = True
                    if not use_session_avail:
                        logger.warning(
                            "calendar availability(%s): 数据集 %r 无可用交易日历，"
                            "回退 >= (same_day)",
                            spec.availability,
                            ds,
                        )

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
                if use_session_avail:
                    # #46 decision >= available_from（日历编译的下一交易日）
                    cond = (
                        f"{prev}.{_quote_ident(anchor_inst)} = "
                        f"{alias}.{_quote_ident(inst_col)} "
                        f"AND {prev}.{_quote_ident(decision_time)} >= "
                        f"{alias}._avail_from"
                    )
                else:
                    cond = (
                        f"{prev}.{_quote_ident(anchor_inst)} = "
                        f"{alias}.{_quote_ident(inst_col)} "
                        f"AND {prev}.{_quote_ident(decision_time)} "
                        f"{spec.comparison_operator} {alias}.{_quote_ident(right_time)}"
                    )
                join_clauses.append(f"ASOF LEFT JOIN ({sub}) AS {alias} ON {cond}")
            else:
                # #8 exact join 也一律显式区分左/右时间列：锚点侧用
                # decision_time（缺省 anchor time_column），右表侧用
                # knowledge_time（缺省右表 time_column）。绝不用
                # ``a.{right_t_col} = b.{right_t_col}``——anchor 可能根本没有
                # 右表的列名（如 anchor.trade_date vs right.date）。
                cond = (
                    f"{prev}.{_quote_ident(decision_time)} = "
                    f"{alias}.{_quote_ident(right_time)} "
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
        # #P1-28 显式 ORDER BY（确定性 panel；raw read 默认无序）。"-col" 表示
        # 降序，"col"/"+col" 升序。
        if order_by:
            ord_parts: list[str] = []
            for c in order_by:
                text = str(c)
                if text.startswith("-"):
                    ord_parts.append(f"{_quote_ident(text[1:])} DESC")
                elif text.startswith("+"):
                    ord_parts.append(_quote_ident(text[1:]))
                else:
                    ord_parts.append(_quote_ident(text))
            sql = f"{sql} ORDER BY {', '.join(ord_parts)}"
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

        #P0-25：不再用 ``limit=1`` 行探针——它只能证明「第一行」的版本，跨分区
        混版本（2024=v1、2025=v2）会漏过。改为窗口内 ``DISTINCT`` 取值集合：
        集合 >1 → 混版本直接拒绝。

        **#12 fail-closed**：production/strict 下无法证明相同 == 不相同——任何
        因子 probe 读取失败（取值集合为空）直接拒绝，禁止静默放行「可能混版本」
        的训练矩阵。
        """
        ds = self._registry.get("factor_lake")
        strict = is_strict_semantics()

        seen_snapshots: set[str] = set()
        seen_universes: set[str] = set()
        for fid in fids:
            if versions and versions.get(fid) is not None:
                expected = versions[fid]
                vals = self._factor_metadata_values(
                    fid, ["factor_version"], time_range=time_range, params=params
                )
                actuals = set(vals.get("factor_version", ())) - {"<null>"}
                if not actuals:
                    if strict:
                        raise DataError(
                            f"factor {fid}: 无法确认 factor_version（窗口内无数据或"
                            " probe 失败）。require_same 无法证明 → 禁止混入训练矩阵。"
                        )
                    continue
                if len(actuals) > 1:
                    raise ValidationError(
                        f"factor {fid}: 窗口内出现多个 factor_version："
                        f"{sorted(actuals)}。禁止在训练矩阵里混入不同版本。"
                    )
                actual = next(iter(actuals))
                if actual != str(expected):
                    raise ValidationError(
                        f"factor {fid}: 期望版本 {expected}，实际 {actual}。"
                        "禁止在训练矩阵里混入不同版本。"
                    )
                continue
            if require_same_data_snapshot or require_same_universe:
                cols: list[str] = []
                if require_same_data_snapshot:
                    cols.append("data_snapshot_id")
                if require_same_universe and "universe" in (ds.schema or {}):
                    cols.append("universe")
                if not cols:
                    continue
                vals = self._factor_metadata_values(
                    fid, cols, time_range=time_range, params=params
                )
                if not vals:
                    if strict:
                        raise DataError(
                            f"factor {fid}: probe 失败，无法证明 data_snapshot/"
                            "universe 一致 → 禁止混入训练矩阵。"
                        )
                    continue
                if require_same_data_snapshot and "data_snapshot_id" in vals:
                    snaps = set(vals["data_snapshot_id"]) - {"<null>"}
                    if not snaps and strict:
                        raise DataError(
                            f"factor {fid}: data_snapshot_id 为空，无法证明一致"
                            " → 禁止混入训练矩阵。"
                        )
                    seen_snapshots.update(snaps)
                if require_same_universe and "universe" in vals:
                    seen_universes.update(set(vals["universe"]) - {"<null>"})
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

    def _factor_metadata_values(
        self,
        fid: str,
        columns: Sequence[str],
        *,
        time_range: tuple[Any, Any] | None,
        params: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        """#P0-25 窗口内某因子的元数据列 DISTINCT 取值集合（跨分区全量）。

        替代 limit=1 探针：``SELECT DISTINCT col FROM factor_lake WHERE factor_id=?``
        覆盖窗口内全部文件，能抓住「2024 分区 v1、2025 分区 v2」的混版本。
        """
        from data_access.read.formats import format_adapter_for_dataset
        from data_access.read.predicate import Predicate, compile_predicate

        ds = self._registry.get("factor_lake")
        paths = self._prepare_dataset_read(
            ds, time_range=time_range, params=dict(params, factor_id=fid)
        )
        if not paths:
            return {}
        path_param = paths if len(paths) > 1 else paths[0]
        adapter = format_adapter_for_dataset(ds)
        from_clause = adapter.build_from_clause(
            path_param,
            hive_partitioning=ds.hive_partitioning,
            union_by_name=ds.union_by_name,
        )
        tcol = ds.time_column or "datetime"
        time_type = str((ds.schema or {}).get(tcol or "", "")).lower()
        pred = Predicate(
            time_range=time_range,
            instrument_filter=None,
            time_column_is_timestamp=("timestamp" in time_type or "datetime" in time_type),
        )
        compiled = compile_predicate(
            pred, time_column=tcol, instrument_column=ds.instrument_column
        )
        col_sql = ", ".join(_quote_ident(c) for c in columns)
        sql = (
            f"SELECT DISTINCT {col_sql} FROM {from_clause} {compiled.where_sql}".strip()
        )
        try:
            tbl = self._engine.execute_arrow(
                sql, [path_param, *compiled.params], deadline_ms=None
            )
        except Exception:
            return {}
        out: dict[str, set[str]] = {c: set() for c in columns}
        for row in tbl.to_pylist():
            for c in columns:
                v = row.get(c)
                out[c].add("<null>" if v is None else str(v))
        return out

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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        physical_scope: str | Sequence[str] | None = None,
    ) -> ReadHandle:
        """统一 read 编排：engine 路由 + 结果形态。

        ``physical_scope``（#17）：read_uri 精确 URI scope，透传给各引擎。
        """
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
                mode=mode,
                allow_sparse=allow_sparse,
                allow_effective_time=allow_effective_time,
                physical_scope=physical_scope,
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
                mode=mode,
                allow_sparse=allow_sparse,
                allow_effective_time=allow_effective_time,
                physical_scope=physical_scope,
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
                # #P0-21 production/strict 不暴露 raw LazyFrame：governed 句柄
                # 的 to_arrow/to_polars/stream 全部走 collect_polars_with_budget。
                return ReadHandle(
                    lazy=lf,
                    snapshot=snapshot,
                    lineage=lineage,
                    batch_size=batch_size,
                    budget=budget,
                    govern_lazy=is_strict_semantics(),
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

        # #P0-19 result="stream"：流式**直接路由**到真正 reader——禁止先
        # materialize 完整表再重扫一遍（旧逻辑把表扫两次，第一次白丢）。
        if result == "stream":
            stream = self.read_arrow_stream(
                dataset,
                columns=columns,
                time_range=time_range,
                instrument_filter=instrument_filter,
                filters=filters,
                limit=limit,
                batch_size=batch_size,
                query_budget=query_budget,
                mode=mode,
                allow_sparse=allow_sparse,
                allow_effective_time=allow_effective_time,
                physical_scope=physical_scope,
                **params,
            )
            try:
                if physical_scope is not None:
                    scope = (
                        list(physical_scope)
                        if isinstance(physical_scope, (list, tuple))
                        else [str(physical_scope)]
                    )
                    stream_paths = self._expand_glob_paths(scope)
                else:
                    stream_paths = self._expand_glob_paths(
                        self._prepare_dataset_read(
                            ds,
                            time_range=time_range,
                            params=params,
                            instrument_filter=instrument_filter,
                        )
                    )
                stream_snapshot = self._build_snapshot(
                    dataset=dataset, ds=ds, paths=stream_paths, params=params
                )
            except Exception:
                stream_snapshot = None
            stream_lineage = ReadLineage(
                dataset=dataset,
                columns=tuple(columns) if columns else (),
                time_range=time_range,
                instrument_filter=tuple(instrument_filter) if instrument_filter else (),
                params=params,
            )
            return ReadHandle(
                stream=stream,
                snapshot=stream_snapshot,
                lineage=stream_lineage,
                batch_size=batch_size,
            )

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
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            physical_scope=physical_scope,
        )
        if normalize_units and columns and rr.table is not None:
            rr = self._normalize_read_result(
                rr, dataset=dataset, columns=columns
            )
        return ReadHandle(table=rr.table, snapshot=rr.snapshot, stats=rr.stats, lineage=rr.lineage)

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
        mode: str = "auto",
        allow_sparse: bool = False,
        allow_effective_time: bool = False,
        physical_scope: str | Sequence[str] | None = None,
    ) -> ReadHandle:
        """PyArrow 引擎（arrow/feather 格式）：直读文件 + pc 表达式过滤。"""
        # #5 PyArrow 引擎同样强制 temporal contract + required_filters + allowed values
        self._prepare_read_request(
            dataset,
            columns=columns,
            time_range=time_range,
            mode=mode,
            allow_sparse=allow_sparse,
            allow_effective_time=allow_effective_time,
            filters=filters,
            params=params,
        )
        import pyarrow.dataset as pa_ds

        from data_access.read.formats import pyarrow_engine_read
        from data_access.read.manifest import manifest_root_for_paths
        from data_access.read.predicate_ast import compile_filter_arrow, parse_filters
        from data_access.read.read_handle import ReadHandle

        budget = self._resolve_read_budget(ds, query_budget)
        validate_query_request(
            budget, columns=list(columns) if columns else None, time_range=time_range
        )
        if physical_scope is not None:
            scope = (
                list(physical_scope)
                if isinstance(physical_scope, (list, tuple))
                else [str(physical_scope)]
            )
            paths = self._expand_glob_paths(scope)
        else:
            paths = self._prepare_dataset_read(
                ds,
                time_range=time_range,
                params=params,
                instrument_filter=instrument_filter,
            )
            # #6 冻结 glob → 精确文件列表，Scanner 与 snapshot 读到完全一致
            paths = self._expand_glob_paths(paths)
        files = build_file_manifest(paths)
        self._enforce_scan_files(budget, paths, files=files)

        # #P0-22 扫描阶段下推：filter + projection 进 Scanner，不再先物化全表。
        # time_range / instrument_filter / filters 编译成 dataset expression。
        exprs: list[Any] = []
        if time_range is not None:
            if ds.time_column is None:
                raise ValidationError(
                    f"'{dataset}' 未声明 time_column，无法应用 time_range"
                )
            start_v, end_v = time_range
            if start_v is not None:
                exprs.append(pa_ds.field(ds.time_column) >= start_v)
            if end_v is not None:
                exprs.append(pa_ds.field(ds.time_column) <= end_v)
        if instrument_filter:
            if ds.instrument_column is None:
                raise ValidationError(
                    f"'{dataset}' 未声明 instrument_column，无法应用 instrument_filter"
                )
            exprs.append(
                pa_ds.field(ds.instrument_column).isin(list(instrument_filter))
            )
        if filters is not None:
            expr = compile_filter_arrow(parse_filters(filters))
            if expr is not None:
                exprs.append(expr)
        combined_expr = None
        if exprs:
            combined_expr = exprs[0]
            for e in exprs[1:]:
                combined_expr = combined_expr & e

        start = time.perf_counter()
        table = pyarrow_engine_read(
            paths,
            fmt=str(ds.format),
            columns=list(columns) if columns else None,
            filters=combined_expr,
            batch_size=batch_size,
        )
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
        strict = is_strict_semantics()

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

        # #41 / #P0-23：版本 / data_snapshot / universe 一致性检查（训练矩阵防混
        # 版本）**先于 matrix 路由**——matrix 一旦成功返回，后面的 gate 根本没机会
        # 跑。先跑 gate 保证任何路由都受同一套一致性约束。
        if versions or require_same_data_snapshot or require_same_universe:
            # #12 fail-closed：production/strict 下 probe 失败直接抛（_check_factor_versions
            # 内部处理）；research 下保持宽容（读不了 probe 就跳过一致性核对）。
            try:
                self._check_factor_versions(
                    fids,
                    versions=versions,
                    require_same_data_snapshot=require_same_data_snapshot,
                    require_same_universe=require_same_universe,
                    time_range=time_range,
                    **params,
                )
            except (DataError, ValidationError):
                raise
            except Exception:
                if is_strict_semantics():
                    raise DataError(
                        "read_factors: 版本一致性检查失败且无法证明一致"
                        "（production fail-closed）。"
                    )
                pass

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
            except (MatrixUnavailable, MatrixCoverageMiss):
                # #13 只捕「矩阵不存在/覆盖不到」——版本错/schema 错/参数错等
                # ValidationError 一律上抛，禁止用 fallback 掩盖真实错误。
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
        """走 factor_matrix 物化层读宽矩阵（universe + frequency）。

        未登记/覆盖不到 → ``MatrixUnavailable``/``MatrixCoverageMiss``（只捕这两
        种，版本/schema/参数错误仍抛 ValidationError——#13）。
        """
        from data_access.read.read_handle import ReadHandle

        try:
            matrix = self._registry.get("factor_matrix")
        except ValidationError as exc:
            raise MatrixUnavailable(
                f"factor_matrix 未登记：{exc}"
            ) from exc
        # #P0-24 精确投影：matrix 必须覆盖全部请求 fids，且只返回请求的因子列。
        # 缺任何一个 → MatrixCoverageMiss，绝不许悄悄返回别的列。
        matrix_cols = self._matrix_available_columns(
            universe=universe, frequency=frequency, **params
        )
        if matrix_cols is not None:
            base = [c for c in ("datetime", "asset") if c in matrix_cols]
            missing = [f for f in fids if f not in matrix_cols]
            if missing:
                raise MatrixCoverageMiss(
                    f"factor_matrix 不覆盖请求因子：{missing}"
                    f"（矩阵可用列 {sorted(matrix_cols) if len(matrix_cols) <= 40 else str(len(matrix_cols)) + ' 列'}）"
                )
            # 请求顺序保留；同时并入用户显式 columns（若给了）
            explicit = [str(c) for c in (columns or []) if str(c) in matrix_cols]
            proj = list(dict.fromkeys(base + fids + explicit))
            if proj:
                columns = proj
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

    def _matrix_available_columns(
        self, *, universe: str, frequency: str, **params: Any
    ) -> list[str] | None:
        """#P0-24 factor_matrix 的可用列名。

        优先用 registry 声明的 schema；否则 DESCRIBE（limit=1 探针，DuckDB 下推）。
        返回 None 表示无法确定（调用方跳过精确投影校验）。
        """
        try:
            ds = self._registry.get("factor_matrix")
        except Exception:
            return None
        schema = dict(getattr(ds, "schema", None) or {})
        if schema:
            return sorted(schema)
        try:
            probe = self.read(
                "factor_matrix",
                columns=None,
                limit=1,
                universe=universe,
                frequency=frequency,
                **dict(params),
            ).to_arrow()
        except Exception:
            return None
        if probe is None or probe.num_columns == 0:
            return None
        return list(probe.column_names)

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
        # 统一写事务：mutation → bump source_epoch → 重建 manifest（成功/失败都失效）
        with self._dataset_mutation(dataset, **params):
            with audit.AuditTimer() as timer:
                try:
                    with mutation_lock(target_dir):
                        if mode == "overwrite":
                            # #39 crash-safe overwrite：写候选目录 → 校验 → 原子
                            # rename 替换。旧逻辑 _clear_dir + 直写目标，写一半崩溃
                            # 会留下损坏数据集。候选目录失败时目标目录完好。
                            files_written = self._crash_safe_overwrite(
                                target_dir, table, partition_by=partition_by
                            )
                        else:
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

    def _crash_safe_overwrite(
        self,
        target_dir: Path,
        table: pa.Table,
        *,
        partition_by: Sequence[str] | None,
    ) -> list[Path]:
        """#39 原子 overwrite：候选目录写入 → 校验 → rename 替换。

        流程：写 ``.write_candidate.*`` → 校验候选行数/文件 → 旧目录 rename 到
        ``.old.*`` → 候选 rename 到 target → 删旧。任何阶段失败都不动旧目标。
        """
        parent = target_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        candidate = parent / f".{target_dir.name}.write_candidate.{uuid.uuid4().hex[:8]}"
        old_dir: Path | None = None
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            written = self._write_table_to_dir(
                table, candidate, partition_by=partition_by,
            )
            if not written:
                raise ValidationError("write_arrow overwrite: 没有写入任何文件")
            # 校验候选（非分区时核对行数；分区时至少确认有非空 parquet）
            import pyarrow.parquet as pq

            total_rows = 0
            for fp in written:
                try:
                    total_rows += pq.read_metadata(str(fp)).num_rows
                except Exception as exc:
                    raise ValidationError(
                        f"overwrite 候选文件 {fp} 读取 footer 失败：{exc}"
                    ) from exc
            if not partition_by and total_rows != table.num_rows:
                raise ValidationError(
                    f"overwrite 候选行数 {total_rows} != 输入 {table.num_rows}"
                )
            # 原子替换：旧目录（存在则）移走 → 候选晋升 → 清理旧目录
            if target_dir.exists():
                old_dir = parent / f".{target_dir.name}.old.{uuid.uuid4().hex[:8]}"
                os.rename(str(target_dir), str(old_dir))
            try:
                os.rename(str(candidate), str(target_dir))
            except OSError:
                if old_dir is not None and old_dir.exists():
                    os.rename(str(old_dir), str(target_dir))
                raise
            if old_dir is not None and old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)
            # 返回最终 target_dir 下的路径
            return [target_dir / p.relative_to(candidate) for p in written]
        finally:
            if candidate.exists():
                shutil.rmtree(candidate, ignore_errors=True)

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

        with self._dataset_mutation(dataset, **params):
            result = upsert_mod.upsert_table(
                ds=ds,
                authorizer=self._authorizer,
                target_dir=target_dir,
                new_table=table,
                upsert_on=upsert_on,
                partition_by=partition_by,
                params=params,
            )
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

        def _do_delete() -> dict[str, Any]:
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

        # #2：delete_rows 也是 mutation —— 必须走统一 manifest 失效事务。
        # dry_run 不产生真实变更，跳过失效（避免无意义的 manifest rebuild）。
        if dry_run:
            return _do_delete()
        with self._dataset_mutation(dataset, **params):
            return _do_delete()

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

        # 发布是 target 的 mutation：统一失效 + 重建 target 的 manifest。
        with self._dataset_mutation(target_dataset, **params):
            result = publish.publish_from_staging(
                registry=self._registry,
                authorizer=self._authorizer,
                staging_name=staging_dataset,
                target_name=target_dataset,
                **params,
            )
        return result

    def _sql_semantic_gate(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        params: Mapping[str, Any] | None,
    ) -> None:
        """#P0-12 sql() 视图构建前的 semantic gate：复用 ``_prepare_read_request``。

        强制 temporal contract / event cutoff / required filters / allowed values，
        使 sql() 不能绕过 generic read() 会拒绝的 E1/E2/RAW_EVENT/X0/
        effective-time-only。
        """
        self._prepare_read_request(
            dataset,
            columns=columns,
            time_range=time_range,
            mode="auto",
            params=params,
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
        _semantic_gate: bool | None = None,
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
        gate = self._sql_semantic_gate if _semantic_gate is not False else None
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
            semantic_gate=gate,
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
            try:
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
            except Exception as exc:
                if is_strict_semantics():
                    raise SnapshotBuildError(
                        f"sql_result 参与数据集 '{name}' snapshot 构建失败："
                        f"{type(exc).__name__}: {exc}（production fail-closed）。"
                    ) from exc
                logger.warning("sql_result snapshot 缺失 %s（research 放行）", name)
                path_by_ds[name] = []
        if not snapshots:
            raise SnapshotBuildError(
                "sql_result 没有任何参与数据集成功构建 snapshot"
                "（production fail-closed）。"
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
            semantic_gate=self._sql_semantic_gate,
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
            semantic_gate=self._sql_semantic_gate,
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

        # #26 local+remote 混合计划：auto 模式下本地已有 partition 用本地、缺失的
        # 用远程，不再「本地不齐就整区间 remote」。
        if _cos_mode_is_auto():
            from data_access.cos.remote import hybrid_cos_read_paths

            hybrid = hybrid_cos_read_paths(ds, time_range=time_range, params=params)
            if hybrid is not None:
                paths = self._authorize_read_paths(hybrid)
                self._engine.ensure_s3_configured()
                logger.info(
                    "cos_multi_location: dataset=%s paths=%d (hybrid)",
                    ds.name,
                    len(paths),
                )
                return paths

        if should_read_cos_remote(ds, time_range=time_range):
            from data_access.cos.remote import prepare_cos_remote_paths

            paths, backend = prepare_cos_remote_paths(
                ds.name, time_range=time_range, ds=ds, params=dict(params)
            )
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
        time_column: str | None = None,
    ) -> list[str]:
        """读前路径裁剪：Partition Planner（路径模板级）+ Manifest（文件级）。

        只对本地 parquet 生效；远程/非 parquet 直接放行。裁剪结果会让
        DuckDB 收到的文件列表大幅缩小（尤其是 ``**/*.parquet`` 大量小文件）。

        #P0-9 多时钟安全：``time_column`` 是本次查询实际裁剪的时钟（PIT join
        时是 knowledge_time，而非 dataset 的 partition time）。manifest /
        partition planner 的 min/max 统计只基于 ``ds.time_column``——查询时钟不同
        时**禁止**基于另一根时间轴裁剪（宁可多扫，不能 false negative）。有
        PIT 索引时按 filing 范围精准裁剪；否则直接放行。
        """
        from data_access.read.partition_planner import parse_partitioning, prune_paths_for_time_range

        prune_clock = time_column or ds.time_column
        if prune_clock != ds.time_column:
            # #P0-9 时钟不匹配：优先用 PIT 索引（filing_date 时钟），无索引则放行
            pit_pruned = self.prune_pit_paths(
                ds.name,
                filing_range=time_range,
                tickers=instrument_filter,
            )
            if pit_pruned:
                logger.info(
                    "pit_index_prune: dataset=%s clock=%s files=%d",
                    ds.name,
                    prune_clock,
                    len(pit_pruned),
                )
                return self._authorize_read_paths(pit_pruned)
            logger.debug(
                "clock_mismatch_overscan: dataset=%s clock=%s (no PIT index → 放行)",
                ds.name,
                prune_clock,
            )
            return paths

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

    @staticmethod
    def _expand_glob_paths(paths: Sequence[str]) -> list[str]:
        """把 glob 路径展开成**精确文件列表**（#6 snapshot 与执行读到完全一致）。

        snapshot 构建后、DuckDB 真正展开 glob 前若新增 parquet，旧流程返回数据
        含新文件但 snapshot 没记录——lineage 不真实。这里在 snapshot 构建的同时
        展开一次，之后读路径只消费这份 frozen list，不再二次 mutable glob。

        - 本地 glob（含 ``*``/``?``/``[``）→ glob.glob(recursive=True) 排序展开
        - 展开为空 → 保留原 pattern（调用方按空分支处理，行为与旧版一致）
        - s3:// / cos:// 无法本地展开 → 原样保留（远程由 etag/version_id 表达版本）
        """
        import glob as glob_mod

        frozen: list[str] = []
        seen: set[str] = set()
        for pattern in paths:
            p = str(pattern)
            if p.startswith("s3://") or p.startswith("cos://"):
                if p not in seen:
                    seen.add(p)
                    frozen.append(p)
                continue
            if not any(ch in p for ch in "*?["):
                if p not in seen:
                    seen.add(p)
                    frozen.append(p)
                continue
            expanded = sorted(glob_mod.glob(p, recursive=True))
            if not expanded:
                if p not in seen:
                    seen.add(p)
                    frozen.append(p)
                continue
            for fp in expanded:
                if fp not in seen:
                    seen.add(fp)
                    frozen.append(fp)
        return frozen

    def _prepare_dataset_read(
        self,
        ds: Dataset,
        *,
        time_range: tuple[Any, Any] | None,
        params: dict[str, Any],
        instrument_filter: Sequence[str] | None = None,
        time_column: str | None = None,
    ) -> list[str]:
        """读路径统一入口：raw 解析 + manifest/partition 裁剪。

        ``time_column``（#P0-9）：本次查询实际裁剪的时钟。缺省 = ds.time_column。
        PIT join 分支传 knowledge_time，禁止用 partition 时钟裁剪。
        """
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
            time_column=time_column,
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


def _column_is_text(ds: Any, column: str | None) -> bool:
    """#P0-27 判断列是否物理声明为文本（string/str/varchar）。

    美股财务 period_end 在 registry runtime patch 里声明为 ``string``——period
    PIT 逻辑必须把它当文本 temporal 处理（严格 CAST 成 ``_period_time`` 再做
    MAX/month），不能依赖 ISO 字典序 + EXTRACT 对脏格式的隐式行为。
    """
    if not column:
        return False
    declared = str((getattr(ds, "schema", None) or {}).get(column, "")).strip().lower()
    return declared in {"string", "str", "varchar"}


def _and_filters(*filters: Any) -> Any:
    """把多个 Filter AST 合并成 And(...)；None 忽略。"""
    from data_access.read.predicate_ast import And

    parts = [f for f in filters if f is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return And(tuple(parts))


def _min_date_bound(end: Any) -> Any:
    """#16 事件 cutoff：把时间窗上界钳到今天（effective_time_only 防未来事件）。"""
    import datetime as _dt

    if end is None:
        return _dt.date.today()
    e = end.date() if isinstance(end, _dt.datetime) else end
    if isinstance(e, _dt.date) and e > _dt.date.today():
        return _dt.date.today()
    return end


def _market_of_dataset(dataset: str) -> str | None:
    """推断数据集所属市场（session availability 的日历查找用）。

    优先级：COS 契约声明的 market → 数据集名前缀。无契约且前缀无法识别
    （测试用临时数据集）返回 None。
    """
    if not dataset:
        return None
    key = str(dataset)
    try:
        from data_access.cos_contract import get_cos_contract

        contract = get_cos_contract(key)
        if contract is not None:
            return contract.market
    except Exception:
        pass
    if key.startswith("us_"):
        return "us"
    if key.startswith("ashare_") or key.startswith("a_share"):
        return "ashare"
    return None


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


def _session_avail_sql(
    sub_sql: str,
    *,
    right_time_col: str,
    calendar: Any,
    start: Any = None,
    end: Any = None,
    max_rows: int = 4000,
) -> tuple[str, list[Any], bool]:
    """#46 session availability：把 knowledge_time 编译成真正的 available_from。

    用交易日历生成 ``(knowledge_date -> next_trading_day)`` 的 VALUES CTE，
    LEFT JOIN 到右表算出 ``_avail_from``；ASOF 改为 ``decision >= _avail_from``。
    相比旧的严格大于，节假日/周末公告能映射到真正的下一交易日。

    返回 (wrapped_sql, params, ok)。窗口为空或超过 max_rows 时返回
    (sub_sql, [], False) 让调用方回退。
    """
    if not calendar or not getattr(calendar, "has_data", False):
        return sub_sql, [], False
    import datetime as _dt

    def _as_date(v: Any) -> _dt.date | None:
        if v is None:
            return None
        if isinstance(v, _dt.datetime):
            return v.date()
        if isinstance(v, _dt.date):
            return v
        if isinstance(v, str):
            try:
                return _dt.date.fromisoformat(v[:10])
            except ValueError:
                return None
        return None

    lo = _as_date(start)
    if lo is not None:
        lo = lo - _dt.timedelta(days=30)
    hi = _as_date(end)
    # #P0-6 前瞻窗口：end 之后的下一交易日必须进映射，否则窗口末条公告会
    # COALESCE 回退到 knowledge 本身（本应下一交易日可用却提前到当天）。
    # 21 天覆盖国庆+周末等连续长假的下一交易日。
    if hi is not None:
        hi = hi + _dt.timedelta(days=21)
    if lo is None:
        lo = calendar.trading_days[0] if calendar.trading_days else None
    if hi is None:
        hi = calendar.trading_days[-1] if calendar.trading_days else None
    if lo is None or hi is None or lo > hi:
        return sub_sql, [], False
    # 把窗口内每个自然日（含周末/节假日）映射到「严格大于它的下一交易日」。
    # 仅交易日有 next；非交易日也要映射（节假日公告 → 下一交易日可见）。
    tds = set(calendar.trading_days)
    next_map: dict[_dt.date, _dt.date] = {}
    nxt_td: _dt.date | None = None
    day = hi
    while day >= lo and len(next_map) < max_rows:
        next_map[day] = nxt_td  # type: ignore[assignment]
        if day in tds:
            nxt_td = day
        day -= _dt.timedelta(days=1)
    pairs: list[tuple[str, str]] = [
        (kd.isoformat(), nxt.isoformat())
        for kd, nxt in next_map.items()
        if nxt is not None
    ]
    if not pairs:
        return sub_sql, [], False
    placeholders = ", ".join("(?, ?)" for _ in pairs)
    vals: list[Any] = []
    for kd, ntd in pairs:
        vals.extend([kd, ntd])
    cte = (
        f"SELECT * FROM (VALUES {placeholders}) AS _cal(_kd, _next_td)"
    )
    # COALESCE 在子查询内完成：日历窗口外的 knowledge（如很早的 seed 记录）
    # 回退到其本身，ASOF 条件保持纯列比较（DuckDB ASOF 不接受表达式）。
    wrapped = (
        f"SELECT _r.*, "
        f"COALESCE(CAST(_cal._next_td AS DATE), "
        f"CAST(_r.{_quote_ident(right_time_col)} AS DATE)) AS _avail_from "
        f"FROM ({sub_sql}) _r LEFT JOIN ({cte}) _cal "
        f"ON CAST(_r.{_quote_ident(right_time_col)} AS DATE) = _cal._kd"
    )
    return wrapped, vals, True


def _period_selection_sql(
    sub_sql: str,
    *,
    inst_col: str,
    period_col: str,
    knowledge_col: str,
    selection: str,
    period_values: Sequence[Any] | None = None,
    period_is_text: bool = False,
) -> tuple[str, list[Any]]:
    """#45 财务报告期选择（PIT 状态更新语义）。

    - latest_period：``period == running_max(period) over (inst, knowledge)``
      ——「在截至 knowledge 已可知的记录中取报告期最新的那版」。旧报告期的晚
      修订（knowledge 更新但 period 更旧）不会让当前财务状态回滚。
    - exact_period：只保留 ``period IN period_values``。
    - annual / quarterly：按 period 月份过滤（12-31 / 3,6,9,12 月末）。
    - ttm：美股有 ``timeframe`` 列时按 'trailing_twelve_months' 过滤；无该列
      则等价 all（并告警）。

    ``period_is_text``（#P0-27）：美股财务 period_end 物理存 string。文本列直接
    ``MAX`` 依赖 ISO 字典序、``EXTRACT(MONTH ...)`` 对脏格式可能失败/错序。统一
    在 scan boundary 生成 ``_period_time = CAST(period_col AS DATE)``（**严格**
    CAST：任何不可解析值直接触发 DuckDB cast 错误 → fail-closed），period 逻辑
    只消费 ``_period_time``，输出前 ``EXCLUDE`` 掉派生列（不改变结果 schema）。

    返回 (sql, extra_params)。调用方把它套在窗口+seed 的 UNION 之后、revision
    去重之前——running_max 需要看到全部 revision 才能正确标记「period 峰值」。
    """
    sel = str(selection or "all").strip().lower()
    if sel in {"", "all"}:
        return sub_sql, []
    if period_is_text:
        inner = (
            f"SELECT *, CAST({_quote_ident(period_col)} AS DATE) AS _period_time "
            f"FROM ({sub_sql}) AS _ptc"
        )
        p = "_period_time"
        exclude = " EXCLUDE (_period_time)"
    else:
        inner = sub_sql
        p = _quote_ident(period_col)
        exclude = ""
    if sel == "latest_period":
        wrapped = (
            f"SELECT *{exclude} FROM ("
            f"SELECT *, MAX({p}) OVER ("
            f"PARTITION BY {_quote_ident(inst_col)} "
            f"ORDER BY {_quote_ident(knowledge_col)} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _pm "
            f"FROM ({inner}) AS _pl) AS _ps "
            f"WHERE _ps.{p} = _ps._pm"
        )
        return wrapped, []
    if sel == "exact_period":
        vals = list(period_values or ())
        if not vals:
            raise ValidationError(
                "period_selection='exact_period' 需要提供 period_values"
            )
        placeholders = ", ".join("?" for _ in vals)
        return (
            f"SELECT *{exclude} FROM ({inner}) AS _pe "
            f"WHERE {p} IN ({placeholders})",
            vals,
        )
    if sel == "annual":
        return (
            f"SELECT *{exclude} FROM ({inner}) AS _pa "
            f"WHERE EXTRACT(MONTH FROM {p}) = 12",
            [],
        )
    if sel == "quarterly":
        return (
            f"SELECT *{exclude} FROM ({inner}) AS _pq "
            f"WHERE EXTRACT(MONTH FROM {p}) IN (3, 6, 9, 12)",
            [],
        )
    if sel == "ttm":
        return (
            f"SELECT *{exclude} FROM ({inner}) AS _pt "
            f"WHERE COALESCE(LOWER({_quote_ident('timeframe')}), '') = 'trailing_twelve_months'",
            [],
        )
    raise ValidationError(f"period_selection={selection!r} 不支持")


def _seed_qualify_sql(
    sub_sql: str,
    inst_col: str,
    time_col: str,
    revision_order: Sequence[str],
    *,
    period_col: str | None = None,
    period_selection: str | None = None,
    period_values: Sequence[Any] | None = None,
    period_is_text: bool = False,
) -> tuple[str, list[Any]]:
    """PIT seed 分支：每标的在窗口 start 之前的 **PIT 状态**，返回 (sql, params)。

    - ``latest_period``（财务默认）：**不是**"start 前最后一条 event"，而是
      "截至 start 已可见报告期最新的一版"——按 (period DESC, revision DESC)
      取行 1。防止 Q3 已发布（10-31）后被晚到的 Q2 修订（11-15）压掉，导致
      seed 只留 Q2、财务状态从 Q3 回滚成 Q2（#P0-3）。
    - ``exact_period``：先按 period_values 过滤，再取该期间最新可见记录。
    - 其他/缺省：start 前最后一条可见记录（knowledge DESC, revision DESC）。

    ``period_is_text``（#P0-27）：文本 period 列同样严格 CAST 成 ``_period_time``
    再排序/过滤（ISO 字典序对脏格式不可靠），输出前 EXCLUDE 派生列。
    """
    sel = str(period_selection or "").strip().lower()
    if period_is_text and period_col:
        inner = (
            f"SELECT *, CAST({_quote_ident(period_col)} AS DATE) AS _period_time "
            f"FROM ({sub_sql}) _stc"
        )
        if sel == "latest_period":
            rev_order = ", ".join(f"{_quote_ident(c)} DESC" for c in revision_order)
            return (
                f"SELECT * EXCLUDE (_period_time) FROM ({inner}) _seed "
                f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
                f"ORDER BY _period_time DESC, {rev_order}) = 1",
                [],
            )
        if sel == "exact_period" and period_values:
            vals = list(period_values or ())
            placeholders = ", ".join("?" for _ in vals)
            filtered = (
                f"SELECT * FROM ({inner}) _es "
                f"WHERE _period_time IN ({placeholders})"
            )
            rev_order = ", ".join(f"{_quote_ident(c)} DESC" for c in revision_order)
            return (
                f"SELECT * EXCLUDE (_period_time) FROM ({filtered}) _seed "
                f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
                f"ORDER BY {_quote_ident(time_col)} DESC, {rev_order}) = 1",
                vals,
            )
        # 其他缺省分支：period 不参与排序，但 CAST 校验仍然生效（fail-closed）
        rev_order = ", ".join(f"{_quote_ident(c)} DESC" for c in revision_order)
        return (
            f"SELECT * EXCLUDE (_period_time) FROM ({inner}) _seed "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
            f"ORDER BY {_quote_ident(time_col)} DESC, {rev_order}) = 1",
            [],
        )
    if sel == "latest_period" and period_col:
        order = ", ".join(
            f"{_quote_ident(c)} DESC" for c in [period_col, *revision_order]
        )
        return (
            f"SELECT * FROM ({sub_sql}) _seed "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
            f"ORDER BY {order}) = 1",
            [],
        )
    if sel == "exact_period" and period_col and period_values:
        vals = list(period_values or ())
        placeholders = ", ".join("?" for _ in vals)
        filtered = (
            f"SELECT * FROM ({sub_sql}) _es "
            f"WHERE {_quote_ident(period_col)} IN ({placeholders})"
        )
        order = ", ".join(
            f"{_quote_ident(c)} DESC" for c in [time_col, *revision_order]
        )
        return (
            f"SELECT * FROM ({filtered}) _seed "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
            f"ORDER BY {order}) = 1",
            vals,
        )
    order = ", ".join(
        f"{_quote_ident(c)} DESC" for c in [time_col, *revision_order]
    )
    return (
        f"SELECT * FROM ({sub_sql}) _seed "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_quote_ident(inst_col)} "
        f"ORDER BY {order}) = 1",
        [],
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
