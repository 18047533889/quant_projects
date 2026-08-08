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
from data_access.read.predicate import strict_sequence

if TYPE_CHECKING:
    import pyarrow as pa

    from data_access.read.semantic_catalog import SemanticField

_VALID_JOIN_POLICIES = {"exact", "asof", "pit_asof"}
_VALID_SNAPSHOT_POLICIES = {"latest", "fail_if_changed", "pin"}


@dataclass(frozen=True)
class CompiledDataRequest:
    """#4 plan() 时深拷贝的**不可变**请求语义。

    DataRequest 是可变的；``store.plan()`` 曾执行 ``request.anchor = anchor``，
    ReadPlan 又保存原始 request 引用。于是：

        plan = store.plan(req); req.filters = ...; plan.execute()

    实际执行内容会变，但 scan_costs / physical plan / snapshot_info / explain
    还是旧计划——「计划即声明」破裂。plan() 把请求的语义字段冻结成这份对象，
    ReadPlan.execute() / PhysicalPlan 组合执行器只消费它，不再读活的 req。
    """

    fields: tuple[str, ...] = ()
    start: Any = None
    end: Any = None
    instruments: tuple[str, ...] | None = None
    universe: str | None = None
    pit: bool = False
    anchor: str | None = None
    frequency: str | None = None
    normalize_units: bool = False
    engine: str = "auto"
    result: str = "auto"
    limit: int | None = None
    filters: Any = None
    filters_by_dataset: Mapping[str, Any] | None = None
    joins: Mapping[str, Any] | None = None
    join_specs: Mapping[str, Any] | None = None
    source_params: Mapping[str, Mapping[str, Any]] | None = None
    field_params: Mapping[str, Mapping[str, Any]] | None = None
    transforms: Mapping[str, str] | None = None
    aggregations: tuple[Any, ...] | None = None
    time_varying_universe: bool = True
    order_by: tuple[str, ...] | None = None
    snapshot_policy: str = "latest"

    @property
    def time_range(self) -> tuple[Any, Any] | None:
        if self.start is None and self.end is None:
            return None
        return (self.start, self.end)

    def dataset_params(self, dataset: str, *, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
        sp = dict(self.source_params or {})
        return dict(sp.get(dataset, fallback or {}) or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "time_range": list(self.time_range) if self.time_range else None,
            "anchor": self.anchor,
            "universe": self.universe,
            "pit": self.pit,
            "frequency": self.frequency,
            "normalize_units": self.normalize_units,
            "engine": self.engine,
            "result": self.result,
            "limit": self.limit,
            "snapshot_policy": self.snapshot_policy,
            "order_by": list(self.order_by) if self.order_by else None,
        }


def _deep_freeze(value: Any) -> Any:
    """把任意嵌套结构深拷贝成独立对象（彻底脱离活的 request）。

    优先 ``copy.deepcopy``（对 dict/list/tuple/dataclass/Predicate 都正确）；
    万一遇到不可 deepcopy 的奇葩对象（如带 file handle），回退浅拷贝——至少
    顶层容器独立。调用方之后改原始 req 的嵌套 dict / join spec / aggregation
    项都不会再影响编译结果。
    """
    import copy

    try:
        return copy.deepcopy(value)
    except Exception:
        try:
            return copy.copy(value)
        except Exception:
            return value


def _deep_freeze_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(_deep_freeze(value))


def _deep_freeze_nested_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {k: dict(_deep_freeze(v)) for k, v in value.items()}


def compile_data_request(request: Any) -> CompiledDataRequest:
    """把 DataRequest（或等价 dict）深拷贝成不可变 CompiledDataRequest。

    #P1-final closure 2：嵌套结构（filters / joins / join_specs / aggregation 项
    / source_params / field_params / transforms）全部 ``deepcopy``——不再保留
    ``request.filters`` 原对象 / join dict 浅拷贝 / aggregation tuple 里可变的
    dict。ReadPlan.execute()/explain()/anchor 只消费这份冻结 IR，plan() 之后改
    req 的任何嵌套字段都不会造成执行漂移。
    """
    if isinstance(request, dict):
        request = DataRequest(**request)
    fields = tuple(
        list(request.fields) if request.fields is not None else ()
    )
    aggregations = request.aggregations
    if aggregations is not None:
        aggregations = tuple(_deep_freeze(a) for a in aggregations)
    return CompiledDataRequest(
        fields=fields,
        start=request.start,
        end=request.end,
        instruments=(
            tuple(request.instruments) if request.instruments is not None else None
        ),
        universe=request.universe,
        pit=bool(request.pit),
        anchor=request.anchor,
        frequency=request.frequency,
        normalize_units=bool(request.normalize_units),
        engine=request.engine,
        result=request.result,
        limit=request.limit,
        filters=_deep_freeze(request.filters),
        filters_by_dataset=_deep_freeze_mapping(request.filters_by_dataset),
        joins=_deep_freeze_mapping(request.joins),
        join_specs=_deep_freeze_mapping(request.join_specs),
        source_params=_deep_freeze_nested_mapping(request.source_params),
        field_params=_deep_freeze_nested_mapping(request.field_params),
        transforms=_deep_freeze_mapping(request.transforms),
        aggregations=aggregations,
        time_varying_universe=bool(getattr(request, "time_varying_universe", True)),
        order_by=tuple(request.order_by) if request.order_by else None,
        snapshot_policy=str(getattr(request, "snapshot_policy", "latest") or "latest"),
    )


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
    # #5 snapshot_policy：latest（默认）/ fail_if_changed / pin
    #   - latest          ：execute 时读取当时最新文件（现状）
    #   - fail_if_changed ：plan 生成后底层数据版本变化 → execute 拒绝
    #   - pin             ：同 fail_if_changed，且要求有权威 manifest 才可 pin
    # 回测/训练/production 建议 fail_if_changed 或 pin，保证「计划即执行」。
    snapshot_policy: str = "latest"

    def __post_init__(self) -> None:
        """#P0-C12 严格序列边界：fields/instruments/order_by 拒绝裸 str/bytes 与
        非法元素类型。``fields="close"`` 会被逐字符拆成 c/l/o/s/e，是典型的
        silent semantic inversion；非法元素（int/对象）也立即报错而不是 str() 化。
        """
        self.fields = strict_sequence(
            self.fields, name="request.fields", element_type=str, allow_none=False
        )
        self.instruments = strict_sequence(
            self.instruments,
            name="request.instruments",
            element_type=str,
            allow_none=True,
        )
        self.order_by = strict_sequence(
            self.order_by, name="request.order_by", element_type=str, allow_none=True
        )

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
    # #4 不可变编译请求：execute() 只消费它，不读活的 request（防 plan 后篡改）
    compiled: Any = field(default=None, repr=False)
    # #5 snapshot pin：latest / fail_if_changed / pin
    snapshot_policy: str = "latest"
    # #5 plan 时刻每数据集的 manifest token（execute 前对比，变化即拒绝）
    plan_snapshot_tokens: dict[str, dict[str, Any]] = field(default_factory=dict)
    # #P1-final closure 3：snapshot_policy=pin 时 plan 时刻每数据集冻结的
    # 物理文件清单（path + size + mtime_ns / etag / version_id）。execute 必须
    # 逐文件核对——不只看 source_epoch（外部系统直接替换 parquet、没走
    # DataAccess epoch 时不变化，只有物理 pin 能证明）。
    plan_pinned_files: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    # 绑定到 store 以便 execute（由 store.plan 注入）
    _store: Any = field(default=None, repr=False)

    @property
    def anchor(self) -> str | None:
        # #P1-final closure 2：只读 plan() 时冻结的 compiled.anchor，不再读活的
        # request——调用方在 plan() 之后改 req.anchor 不影响执行/explain。
        if self.compiled is not None:
            return self.compiled.anchor or (self.datasets[0] if self.datasets else None)
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
        # #P1-final closure 2：explain 读 plan 时冻结的 compiled（frequency /
        # normalize_units），不再读活的 request——防 plan 后篡改造成 explain 与
        # execute 语义漂移。
        req = self.compiled if self.compiled is not None else self.request
        if req.frequency:
            lines.append(f"FREQUENCY    {req.frequency} (declared)")
        lines.append(f"ENGINE       {self.engine}   RESULT  {self.result}")
        lines.append(f"NORMALIZE    {req.normalize_units}")
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

    def _verify_snapshot_pin(self, store: Any) -> None:
        """#5 snapshot_policy=fail_if_changed/pin：execute 前校验数据版本未变。

        plan 生成后数据可能被改写（上午计划、下午执行）。对比每数据集
        source_epoch / manifest_generation 与 plan 时刻 token；``pin`` 额外
        逐文件核对物理身份（path+size+mtime_ns / etag / version_id）——外部
        系统直接替换 parquet、没走 DataAccess epoch 时也能证明变化。

        #P1-final closure 3 fail-closed：
          - plan 时有 manifest、execute 时 manifest 消失 → 一律视为「已变」，
            不再把「无法证明有没有变化」当成「没变化」（旧逻辑 fail_if_changed
            直接 continue）。
          - pin 必须有权威 manifest 且文件清单与 plan 时刻逐文件一致。
        """
        if self.snapshot_policy == "latest":
            return
        req = self.compiled if self.compiled is not None else self.request
        changed: list[str] = []
        unpinnable: list[str] = []
        for ds in self.datasets:
            try:
                token = store.manifest_version(ds, **req.dataset_params(ds))
            except Exception:
                token = {"has_manifest": False}
            plan_tok = self.plan_snapshot_tokens.get(ds, {}) or {}
            had_manifest = bool(plan_tok.get("has_manifest"))
            if not token.get("has_manifest"):
                if self.snapshot_policy == "pin":
                    unpinnable.append(ds)
                elif had_manifest:
                    # fail_if_changed：plan 时 manifest 存在、现在消失了 →
                    # 数据已被替换/删除，不能当作未变。
                    changed.append(f"{ds}（manifest 消失）")
                # 两者都无 manifest → 无法判断，不误伤。
                continue
            cur_src = token.get("source_epoch") or token.get("manifest_epoch")
            prev_src = plan_tok.get("source_epoch") or plan_tok.get("manifest_epoch")
            cur_gen = token.get("manifest_generation_id")
            prev_gen = plan_tok.get("manifest_generation_id")
            if cur_src != prev_src or cur_gen != prev_gen:
                changed.append(ds)
                continue
            if self.snapshot_policy == "pin":
                # 物理 pin：逐文件对比 plan 冻结的文件清单。大小/mtime（本地）
                # 或 etag/version_id（远程）任一变化 → 数据已变。
                pinned = self.plan_pinned_files.get(ds)
                if pinned is None:
                    unpinnable.append(ds)
                    continue
                try:
                    paths = store._prepare_dataset_read(
                        store._registry.get(ds),
                        time_range=None,
                        params=req.dataset_params(ds),
                        instrument_filter=None,
                    )
                    # 物理 pin 必须**实际 stat**，不能复用 manifest——外部系统直接
                    # 替换 parquet 后 manifest 仍是「新鲜」的（epoch 没 bump），
                    # _files_for_snapshot 会返回 manifest 里的旧 size/mtime，等于
                    # 没核对。用 build_file_manifest 逐文件 stat 当前真实状态。
                    from data_access.read.read_contract import build_file_manifest

                    current = build_file_manifest(paths)
                except Exception:
                    changed.append(f"{ds}（无法重新枚举物理文件）")
                    continue
                if _physical_manifest_changed(pinned, current):
                    changed.append(f"{ds}（物理文件变化）")
        if unpinnable:
            from data_access.core.exceptions import SnapshotBuildError

            raise SnapshotBuildError(
                f"snapshot_policy=pin 需要权威 manifest：{', '.join(unpinnable)} "
                "没有 manifest 或没有 plan 时冻结的物理文件清单，无法 pin 数据版本。"
                "请用 fail_if_changed 或去掉 snapshot_policy=pin。"
            )
        if changed:
            from data_access.core.exceptions import SnapshotBuildError

            raise SnapshotBuildError(
                f"snapshot_policy={self.snapshot_policy}：plan 生成后以下数据集 "
                f"版本已变化（source_epoch/manifest_generation 不一致或物理文件"
                f"变化），拒绝执行：{', '.join(changed)}。请重新 plan() 绑定最新版本。"
            )

    def execute(self) -> Any:
        """执行计划，返回 ReadHandle（读路径照常走 budget/audit/snapshot）。"""
        if self._store is None:
            raise RuntimeError("ReadPlan 未绑定 DataAccessStore，无法 execute")
        store = self._store
        # #4 execute 只消费 plan 编译时冻结的语义（compiled），不读活的 request——
        # 调用方在 plan() 之后改 req.filters/joins/aggregations 不再影响执行。
        req = self.compiled if self.compiled is not None else self.request
        # #5 snapshot pin：fail_if_changed / pin 在 execute 前校验数据版本未变
        self._verify_snapshot_pin(store)
        # #11 节点式执行：聚合+join 组合先交给 PhysicalPlanExecutor；
        # 非组合场景返回 None，走下方现有 read/read_joined/aggregate 路径。
        if self.physical is not None:
            from data_access.read.physical_plan import execute_physical_plan

            composed = execute_physical_plan(store, self)
            if composed is not None:
                return composed
        tr = self.time_range
        insts = self.instruments

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
        # #3 单一事实源：把 plan 阶段编译好的 effective_join_specs 原样交给
        # read_joined（read_joined 内 _effective_join_specs 对已解析 spec 幂等），
        # 不再在 execute 里重新推导——explain 显示的语义 == 真正执行语义。
        return store.read_joined(
            anchor,
            fields=self.per_dataset_columns,
            joins=self.join_specs_effective or None,
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


def _file_identity(fv: Any) -> tuple[str, Any, Any, Any, Any]:
    """FileVersion → 物理身份键 (path, size, mtime_ns, etag, version_id)。

    本地文件用 size/mtime_ns；远程对象用 etag/version_id（size/mtime 不可靠）。
    任一字段为 None 时仍参与对比——两边同 None 视为一致（无法证明变化时在
    pin 场景由 ``_verify_snapshot_pin`` 上层判定不可 pin）。
    """
    return (
        str(getattr(fv, "path", "")),
        getattr(fv, "size", None),
        getattr(fv, "mtime_ns", None),
        getattr(fv, "etag", None),
        getattr(fv, "version_id", None),
    )


def _physical_manifest_changed(pinned: Sequence[Any], current: Sequence[Any]) -> bool:
    """对比 plan 冻结与 execute 时枚举的物理文件清单。

    文件集合、顺序、size/mtime_ns（本地）或 etag/version_id（远程）任一变化
    → True（数据已变）。**注意**：这是 fail-closed 对比——pinned 为空但 current
    非空也判变化；集合内相同文件出现/消失也判变化。
    """
    pinned_keys = [_file_identity(f) for f in pinned]
    current_keys = [_file_identity(f) for f in current]
    # 去重 + 排序后逐条比较；文件集合大小不同直接判定变化。
    if len(pinned_keys) != len(current_keys):
        return True
    return sorted(set(map(str, pinned_keys))) != sorted(set(map(str, current_keys)))
