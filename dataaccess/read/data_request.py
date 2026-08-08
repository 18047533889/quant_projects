"""
data_access.read.data_request —— DataRequest / ReadPlan 统一数据请求接口

职责
    1. ``DataRequest``：FactorEngine Analyzer 一次性提交的「逻辑数据需求」——
       referenced fields、时间区间、instruments/universe、PIT 要求、frequency、
       单位归一化开关、join 策略、engine/result 偏好。
    2. ``ReadPlan``：``store.plan(request)`` 的产物。编译阶段完成字段解析
       （logical → dataset + physical）、多数据集归并（同一物理表只读一次）、
       join 策略、每数据集扫描成本估算与 engine/result 路由，但**不执行**。
       ``ReadPlan.explain()`` 给人看计划；``ReadPlan.execute()`` 真正执行并
       返回 ``ReadHandle``。

设计要点
    1. 字段解析统一走 SemanticFieldCatalog，找不到再回退 registry dataset
       schema（物理列名 == 逻辑名的老代码不受影响）。
    2. 多数据集字段按物理表 coalesce：同一 StockBalance 的多个字段只扫一次。
    3. ``universe`` 命名一个 registry 数据集（如 ashare_universe_daily），
       execute 时用其 instrument 列筛出成分，与 instruments 求交集。
    4. ``frequency`` 目前作为计划元数据声明（为分钟→日聚合 pushdown 预留），
       本层不做 resample——执行引擎决定是否下推。

非职责
    不做文件 IO / 不拼 SQL（read_joined / _read_handle 做）；不做谓词求值。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    import pyarrow as pa

    from data_access.read.semantic_catalog import SemanticField

_VALID_JOIN_POLICIES = {"exact", "asof", "pit_asof"}


@dataclass
class DataRequest:
    """一次统一的逻辑数据请求（Analyzer → DataAccess Planner 的输入）。"""

    fields: Sequence[str]                       # 逻辑字段名，或 "dataset.physical" 限定名
    start: Any = None                           # 闭区间下界（str/date/datetime）
    end: Any = None                             # 闭区间上界
    instruments: Sequence[str] | None = None    # 标的白名单
    universe: str | None = None                 # registry 数据集名，作为股票池来源
    pit: bool = False                           # 是否要求 PIT 语义
    anchor: str | None = None                   # 锚定数据集（无则按字段推导）
    frequency: str | None = None                # daily/minute/...（计划元数据，预留 pushdown）
    normalize_units: bool = False               # 输出层 scale_to_canonical
    engine: str = "auto"                        # auto|duckdb|polars|pyarrow
    result: str = "auto"                        # auto|arrow|pandas|polars|lazy|stream
    limit: int | None = None
    filters: Any = None                         # 锚点级通用过滤
    filters_by_dataset: Mapping[str, Any] | None = None  # 每数据集独立过滤
    joins: Mapping[str, Any] | None = None      # {dataset: exact|asof|pit_asof|dict|TemporalJoinSpec}
    join_specs: Mapping[str, Any] | None = None  # 显式语义 join 规格（同 joins，别名）
    source_params: Mapping[str, Mapping[str, Any]] | None = None  # 每数据集参数（IndexSymbol/IndustrySource...）
    field_params: Mapping[str, Mapping[str, Any]] | None = None  # 每字段变换参数（financial_lag quarters...）
    transforms: Mapping[str, str] | None = None  # 每字段变换（minute_at/financial_lag/...）
    aggregations: Sequence[Any] | None = None    # 分钟→日聚合规格（AggregationSpec，预留 pushdown）
    time_varying_universe: bool = True           # universe 按 (date, instrument) 时变成员过滤
    # #P1-28 deterministic ordering：DataAccess 的 raw read 是**无序**的（DuckDB/
    # Parquet 不保证扫描顺序稳定）。需要确定性 panel（如 FactorEngine 回测面板）
    # 时显式声明 order_by=[time, instrument]，SQL 端加 ORDER BY；ordering 进
    # cache key / lineage / plan，保证同一请求稳定复现。
    order_by: Sequence[str] | None = None

    @property
    def time_range(self) -> tuple[Any, Any] | None:
        if self.start is None and self.end is None:
            return None
        return (self.start, self.end)

    def dataset_params(self, dataset: str, *, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """返回某数据集的生效参数（source_params[dataset] 或 fallback）。"""
        sp = dict(self.source_params or {})
        return dict(sp.get(dataset, fallback or {}) or {})


@dataclass
class ReadPlan:
    """编译好的数据读取计划：可 explain，可 execute。"""

    request: DataRequest
    datasets: list[str]                                  # 有序：anchor 在前
    fields: list["SemanticField"]                        # 已解析字段（去重，保序）
    per_dataset_columns: dict[str, list[str]]            # dataset -> 物理列
    join_policies: dict[str, str]                        # dataset -> join 策略
    scan_costs: dict[str, Any]                           # dataset -> ScanCost
    storage: dict[str, str]                              # dataset -> backend 描述
    snapshot_info: dict[str, dict[str, Any]]             # dataset -> manifest 版本信息
    engine: str = "auto"
    result: str = "auto"
    time_range: tuple[Any, Any] | None = None
    instruments: Sequence[str] | None = None
    universe: str | None = None
    # #P0-2 统一 effective_join_specs（字段语义 → COS 契约 → 显式覆盖）：
    # PIT validator / 组合执行 / explain 消费同一份 join 语义。
    join_specs_effective: dict[str, Any] = field(default_factory=dict)
    # #4 物理计划 DAG（由 store.plan 注入；explain() 渲染，execute() 消费）
    physical: Any = field(default=None, repr=False)
    # 绑定到 store 以便 execute（由 store.plan 注入）
    _store: Any = field(default=None, repr=False)

    @property
    def anchor(self) -> str | None:
        return self.request.anchor or (self.datasets[0] if self.datasets else None)

    def explain(self) -> str:
        """渲染人类可读的计划文本（不执行任何 IO）。"""
        lines: list[str] = []
        lines.append("DataAccess ReadPlan")
        lines.append("=" * 40)
        lines.append("DATASETS")
        for i, ds in enumerate(self.datasets):
            marker = " *anchor" if ds == self.anchor else ""
            join = self.join_policies.get(ds, "exact")
            stg = self.storage.get(ds, "?")
            cost = self.scan_costs.get(ds)
            if cost is None:
                cost_txt = "cost n/a"
            else:
                bytes_txt = (
                    f"{cost.total_bytes:,} bytes" if cost.total_bytes is not None else "bytes n/a"
                )
                cost_txt = (
                    f"~{cost.estimated_rows:,} rows / {cost.file_count} files / {bytes_txt}"
                )
            lines.append(
                f"  [{i}] {ds}{marker}  join={join}  storage={stg}  {cost_txt}"
            )
        lines.append("FIELDS")
        for f in self.fields:
            unit = ""
            if f.is_scale_applicable:
                unit = (
                    f"  ({f.source_unit}->{f.canonical_unit}, x{f.scale})"
                )
            lines.append(f"  {f.logical_name} -> {f.dataset}.{f.physical_name}{unit}")
        lines.append("TIME")
        tr = self.time_range
        lines.append(f"  {tr[0]} ~ {tr[1]}" if tr else "  (全量)")
        lines.append(
            f"INSTRUMENTS  {len(self.instruments or [])}  "
            f"UNIVERSE  {self.universe or '-'}"
        )
        if self.request.frequency:
            lines.append(f"FREQUENCY    {self.request.frequency} (declared)")
        lines.append(f"ENGINE       {self.engine}   RESULT  {self.result}")
        lines.append(f"NORMALIZE    {self.request.normalize_units}")
        snap = self.snapshot_info
        if snap:
            lines.append("SNAPSHOT")
            for ds, info in snap.items():
                lines.append(
                    f"  {ds}: has_manifest={info.get('has_manifest')} "
                    f"dataset_version={info.get('dataset_version')} "
                    f"partition_version={info.get('partition_version')}"
                )
        # #4 物理计划 DAG
        if self.physical is not None:
            lines.append("PHYSICAL PLAN")
            lines.append("  |-- ProjectNode (输出字段)")
            lines.append("  |-- NormalizeNode (单位归一化)")
            lines.append("  |-- AggregationNode (分钟→日聚合)")
            lines.append("  |-- TemporalJoinNode / FilterNode")
            lines.append("  `-- ScanNode (物理扫描)")
            lines.append("")
            lines.append("NODES")
            from data_access.read.physical_plan import render_plan

            lines.append(render_plan(self.physical))
        return "\n".join(lines)

    def execute(self) -> Any:
        """执行计划，返回 ReadHandle（读路径照常走 budget/audit/snapshot）。"""
        if self._store is None:
            raise RuntimeError("ReadPlan 未绑定 DataAccessStore，无法 execute")
        store = self._store
        # #11 节点式执行：聚合+join 组合先交给 PhysicalPlanExecutor；
        # 非组合场景返回 None，走下方现有 read/read_joined/aggregate 路径。
        if self.physical is not None:
            from data_access.read.physical_plan import execute_physical_plan

            composed = execute_physical_plan(store, self)
            if composed is not None:
                return composed
        tr = self.time_range
        insts = self.instruments
        req = self.request

        # #4 AggregationNode：分钟→日聚合一次 scan 多输出（不经过 read_joined）
        if req.aggregations:
            from data_access.read.aggregation import (
                AggregationItem,
                aggregate_minute_bundle,
            )

            items: list[AggregationItem] = []
            for raw in req.aggregations:
                if isinstance(raw, AggregationItem):
                    items.append(raw)
                    continue
                if isinstance(raw, Mapping):
                    fld = str(raw.get("field") or raw.get("column") or "")
                    if not fld:
                        raise ValidationError(
                            f"aggregations 项缺少 field: {raw!r}"
                        )
                    items.append(
                        AggregationItem(
                            field=fld,
                            spec=raw.get("spec"),
                            output_name=(
                                str(raw["output_name"])
                                if raw.get("output_name")
                                else None
                            ),
                        )
                    )
                else:
                    raise ValidationError(
                        "aggregations 项必须是 AggregationItem 或 dict"
                    )
            if not items:
                raise ValidationError("aggregations 为空")
            ds = self.datasets[0]
            ds_params = req.dataset_params(ds)
            market = str(ds_params.get("market") or "") or None
            timezone = str(ds_params.get("timezone") or "") or None
            return aggregate_minute_bundle(
                store,
                ds,
                items,
                time_range=tr,
                instrument_filter=insts,
                params=ds_params,
                market=market,
                timezone=timezone,
            )
        # 时变 universe：把成员过滤下沉到 join（(date, instrument) 精确成员），
        # 否则退化为窗口内静态集合求交（旧行为）。
        time_varying = bool(getattr(req, "time_varying_universe", True))
        if req.universe and not time_varying:
            insts = store._resolve_universe_instruments(req.universe, tr, insts)

        params_by_dataset: dict[str, dict[str, Any]] = {}
        sp = dict(req.source_params or {})
        for ds in self.datasets:
            params_by_dataset[ds] = dict(sp.get(ds, {}) or {})

        if len(self.datasets) == 1 and not (req.universe and time_varying):
            ds = self.datasets[0]
            dsobj = store._registry.get(ds)
            cols = list(self.per_dataset_columns.get(ds) or [])
            # DataRequest 输出是带轴的面板：补齐 anchor 的时间/标的列
            for k in (dsobj.time_column, dsobj.instrument_column):
                if k and k not in cols:
                    cols.append(k)
            ds_params = params_by_dataset.get(ds, {})
            return store.read(
                ds,
                columns=cols or None,
                time_range=tr,
                instrument_filter=insts,
                filters=req.filters,
                limit=req.limit,
                engine=self.engine,
                result=self.result,
                normalize_units=req.normalize_units,
                **ds_params,
            )

        # 多数据集：一次 read_joined，物理表各扫一次，join 在 DuckDB 内完成。
        anchor = self.anchor
        if anchor is None:
            raise ValidationError("多数据集计划缺少 anchor")
        joins: dict[str, Any] = {}
        joins.update(dict(req.joins or {}))
        joins.update(dict(req.join_specs or {}))
        return store.read_joined(
            anchor,
            fields=self.per_dataset_columns,
            joins=joins or None,
            time_range=tr,
            instrument_filter=insts,
            filters=req.filters,
            filters_by_dataset=req.filters_by_dataset,
            limit=req.limit,
            engine=self.engine,
            result=self.result,
            normalize_units=req.normalize_units,
            params_by_dataset=params_by_dataset,
            universe=(req.universe if time_varying else None),
            time_varying_universe=time_varying,
            order_by=req.order_by,
        )

    def __repr__(self) -> str:
        return (
            f"ReadPlan(datasets={self.datasets}, fields={len(self.fields)}, "
            f"engine={self.engine}, result={self.result})"
        )


def normalize_join_policy(policy: str | None) -> str:
    """把 join 策略归一化到合法集合；None 默认 exact。"""
    key = str(policy or "exact").strip().lower()
    if key not in _VALID_JOIN_POLICIES:
        raise ValidationError(
            f"join 策略必须是 {sorted(_VALID_JOIN_POLICIES)}，收到 {policy!r}"
        )
    return key
