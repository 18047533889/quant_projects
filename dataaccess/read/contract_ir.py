"""
data_access.read.contract_ir —— 三套声明的统一 Contract IR（#12）

datasets.yaml（路径/schema）、COSDatasetContract（表级时间语义）、
SemanticFieldCatalog（字段级语义）本来是三个独立来源，长期会漂。本模块把
三者编译成一个 ``ContractIR``：

    machine registry + dataset spec + semantic fields
        → ContractCompiler → ContractIR（per-dataset 统一视图）

用途
    - ``store.contract_ir()`` / ``store.contract_ir_fingerprint()``：运行时统一
      视图 + 稳定指纹（部署在任意服务器可对比语义版本）；
    - ``scripts/audit_contract_ir.py``：CI 对齐——每个 COS 契约数据集必须在
      registry（或显式 external）；每个 registry 数据集若有契约，字段级与表级
      语义一致。

非职责
    不替代三个来源（它们仍是单一事实源的输入）；不推断语义（缺失即 None）。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class TemporalAxes:
    """#9 一个数据集的多根时间轴（#P0-9 多时钟）。

    真实数据往往不止一根时间列：
    - ``partition_time``  物理分区/排序时间（registry ``time_column``，manifest 统计基于它）
    - ``event_time``      事件发生时间
    - ``knowledge_time``  数据可见时间（PIT availability 基准，filing/PubDate）
    - ``effective_time``  数据生效时间（如 Dividend/Split 的 ex_date）
    - ``period_time``     会计期间列（财务报告期）
    - ``decision_time``   决策/回测时钟（join 的 anchor 时间）
    - ``storage_timezone`` / ``semantic_timezone`` 存储与语义时区

    用途：PreparedRead 声明 ``predicate_clock / pruning_clock / join_clock``，
    manifest 若没有对应 clock 的统计则禁止基于另一根时间轴 prune（宁可多扫）。
    """

    partition_time: str | None = None
    event_time: str | None = None
    knowledge_time: str | None = None
    effective_time: str | None = None
    period_time: str | None = None
    decision_time: str | None = None
    storage_timezone: str | None = None
    semantic_timezone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContractIRDataset:
    """一个数据集的统一契约视图（编译产物）。

    #34：除表级时间语义外，扩展字段级契约（knowledge/effective/period_time、
    revision_order、required_filters、allowed_filter_values、unique_key、
    duplicate_policy）与存储/查询策略（partitioning、storage_backend、
    query_policy、coverage_policy），让运行时不再分别读三套定义。
    #9：增加 ``temporal_axes`` 多根时间轴（#P0-9）。
    """

    name: str
    in_registry: bool = False
    in_contracts: bool = False
    # #35 契约在、registry 没有，但显式声明为 external（COS remote 才可见）的
    # 数据集——audit() 不再误报「契约数据集不在 registry」。
    external: bool = False
    market: str | None = None
    temporal_model: str | None = None
    panel_policy: str | None = None
    join_policy: str | None = None
    pit_policy: str | None = None
    calendar_domain: str | None = None
    grain: str | None = None
    cardinality: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    storage_layout: str | None = None
    # ---- #34 字段级/查询级契约 ----
    knowledge_time: str | None = None       # 数据可见时间列（availability_column）
    effective_time: str | None = None       # 数据生效时间列
    period_time: str | None = None          # 会计期间列
    revision_order: list[str] = field(default_factory=list)
    availability: str | None = None         # same_day / next_trading_day / session
    required_filters: list[str] = field(default_factory=list)
    allowed_filter_values: dict[str, Any] = field(default_factory=dict)
    unique_key: list[str] = field(default_factory=list)
    duplicate_policy: str | None = None
    partitioning: dict[str, Any] = field(default_factory=dict)
    storage_backend: str | None = None
    file_format: str | None = None
    query_policy: dict[str, Any] = field(default_factory=dict)
    coverage_policy: str | None = None
    # ---- #9 多根时间轴（#P0-9）----
    temporal_axes: TemporalAxes | None = None
    # ---- 审计问题（#P0-33：audit() 引用 d.issues，必须声明）----
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContractIR:
    """统一契约 IR：registry + COS 契约 + 语义字段 的合并视图。"""

    def __init__(self, datasets: Mapping[str, ContractIRDataset]) -> None:
        self.datasets: dict[str, ContractIRDataset] = dict(datasets)

    def get(self, name: str) -> ContractIRDataset | None:
        return self.datasets.get(name)

    def names(self) -> list[str]:
        return sorted(self.datasets)

    def fingerprint(self) -> str:
        payload = {k: v.to_dict() for k, v in self.datasets.items()}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def audit(self) -> list[str]:
        """跨来源一致性审计：返回问题列表（空 = 一致）。"""
        problems: list[str] = []
        for name in self.names():
            d = self.datasets[name]
            problems.extend(d.issues)
            if not d.in_registry and not d.in_contracts:
                continue
            if d.in_contracts and not d.in_registry and not d.external:
                problems.append(
                    f"契约数据集 {name!r} 不在 registry 且未标记 external"
                )
            if (
                d.in_registry
                and d.in_contracts
                and d.temporal_model == "EMPTY"
                and d.schema
            ):
                # 有真实 schema 却被标 EMPTY → 可疑；无 schema 的占位引用是合法状态
                # （Store 读路径已拒绝 EMPTY 数据集）。
                problems.append(
                    f"registry 数据集 {name!r} 有 schema 但契约是 EMPTY 占位（不应可读）"
                )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.datasets.items()}


def build_contract_ir(
    registry: Any,
    contracts: Mapping[str, Any] | None = None,
    catalog: Any = None,
    *,
    external_contract_datasets: Sequence[str] = (),
) -> ContractIR:
    """把 registry + COS 契约 + 语义字段编译成 ContractIR。

    - ``registry``：DatasetRegistry（必填）
    - ``contracts``：COS_DATASET_CONTRACTS（缺省加载全局）
    - ``catalog``：SemanticFieldCatalog（缺省加载全局）
    - ``external_contract_datasets``：契约声明但不在 registry 的数据集白名单
      （如 COS remote 才可见的表）
    """
    from data_access.cos_contract import COS_DATASET_CONTRACTS

    contracts = contracts if contracts is not None else COS_DATASET_CONTRACTS
    external = set(external_contract_datasets or ())
    out: dict[str, ContractIRDataset] = {}

    reg_names = set(registry.names())
    all_names = set(reg_names) | set(contracts.keys())

    for name in sorted(all_names):
        reg_ds = None
        try:
            reg_ds = registry.get(name)
        except Exception:
            reg_ds = None
        contract = contracts.get(name)
        entry = ContractIRDataset(name=name)
        if reg_ds is not None:
            entry.in_registry = True
            entry.schema = dict(getattr(reg_ds, "schema", None) or {})
            entry.market = entry.market or _market_of_name(name)
        if contract is not None:
            entry.in_contracts = True
            entry.market = contract.market
            entry.temporal_model = contract.temporal_model
            entry.panel_policy = contract.panel_policy
            entry.join_policy = contract.join_policy
            entry.pit_policy = contract.pit_policy
            entry.calendar_domain = contract.calendar_domain
            entry.grain = contract.grain
            entry.cardinality = contract.cardinality
            entry.storage_layout = contract.storage_layout
            entry.coverage = {
                k: getattr(contract, k)
                for k in (
                    "coverage_start",
                    "coverage_end",
                    "expected_cadence",
                    "max_staleness",
                    "missing_partition_semantics",
                )
            }
            # #34 字段级契约
            entry.knowledge_time = contract.availability_column
            entry.effective_time = contract.event_column
            entry.period_time = contract.period_column
            entry.revision_order = list(contract.revision_columns or ())
            entry.unique_key = list(contract.unique_key or ())
            # #36 required_filters 完整合并：panel + dimension + event + 字段级
            # required_filters（旧代码只并前两个，事件过滤维度会漏）。
            entry.required_filters = list(
                dict.fromkeys(
                    list(contract.required_panel_filters or ())
                    + list(contract.required_dimension_filters or ())
                    + list(contract.required_event_filters or ())
                )
            )
            entry.allowed_filter_values = dict(contract.allowed_filter_values or ())
            entry.coverage_policy = contract.missing_partition_semantics
        if reg_ds is not None:
            entry.partitioning = dict(getattr(reg_ds, "partitioning", None) or {})
            entry.storage_backend, entry.storage_layout, entry.file_format = (
                _storage_of(reg_ds)
            )
            qp = getattr(reg_ds, "query_policy", None)
            if qp is not None:
                entry.query_policy = {
                    k: getattr(qp, k)
                    for k in (
                        "require_explicit_columns",
                        "require_time_range",
                        "max_rows",
                        "max_result_bytes",
                        "max_elapsed_ms",
                        "max_scan_files",
                    )
                    if getattr(qp, k, None) is not None
                }
        if catalog is not None:
            entry.fields = sorted(
                f.logical_name
                for f in catalog._fields.values()
                if f.dataset == name
            )
            if not entry.availability:
                avail = {
                    getattr(f, "availability", None)
                    for f in catalog._fields.values()
                    if f.dataset == name and getattr(f, "availability", None)
                }
                if len(avail) == 1:
                    entry.availability = next(iter(avail))
                elif len(avail) > 1:
                    # #P2-78 多字段 availability 冲突不能静默留 None（unknown）——
                    # 那是 ContractIR conflict，必须形成 issue 供 audit 暴露。
                    entry.issues.append(
                        f"字段级 availability 冲突 {sorted(avail)}（数据集 {name!r}）："
                        "无法收敛单一 availability，ContractIR 不得视为确定值"
                    )
            if not entry.duplicate_policy:
                dup = {
                    getattr(f, "duplicate_policy", None)
                    for f in catalog._fields.values()
                    if f.dataset == name and getattr(f, "duplicate_policy", None)
                }
                if len(dup) == 1:
                    entry.duplicate_policy = next(iter(dup))
            if entry.temporal_axes is None:
                axes = _temporal_axes_of(
                    reg_ds=reg_ds,
                    contract=contract,
                    fields=[
                        f
                        for f in catalog._fields.values()
                        if f.dataset == name
                    ],
                )
                if any(
                    v is not None
                    for v in (
                        axes.partition_time,
                        axes.event_time,
                        axes.knowledge_time,
                        axes.effective_time,
                        axes.period_time,
                    )
                ):
                    entry.temporal_axes = axes
        # #35 external：契约在、registry 没有但显式声明 → audit() 放行
        entry.external = name in external and not entry.in_registry
        # 一致性 issue：契约在但 registry 没有 → 未标记 external
        if entry.in_contracts and not entry.in_registry and name not in external:
            entry.issues.append(
                f"契约数据集 {name!r} 不在 registry 且未标记 external"
            )
        # #36 冲突检测：契约默认与字段级语义（availability / duplicate_policy /
        # period_selection）在 catalog 全字段一致时仍冲突 → 记 issue。
        if contract is not None and catalog is not None:
            field_avails = {
                getattr(f, "availability", None)
                for f in catalog._fields.values()
                if f.dataset == name and getattr(f, "availability", None)
            }
            if len(field_avails) == 1:
                fav = next(iter(field_avails))
                if fav not in (None, entry.availability):
                    entry.issues.append(
                        f"字段级 availability={fav!r} 与契约默认 "
                        f"{entry.availability!r} 不一致（数据集 {name!r}）"
                    )
            field_dups = {
                getattr(f, "duplicate_policy", None)
                for f in catalog._fields.values()
                if f.dataset == name and getattr(f, "duplicate_policy", None)
            }
            if len(field_dups) == 1:
                fdup = next(iter(field_dups))
                if fdup not in (None, entry.duplicate_policy):
                    entry.issues.append(
                        f"字段级 duplicate_policy={fdup!r} 与契约默认 "
                        f"{entry.duplicate_policy!r} 不一致（数据集 {name!r}）"
                    )
        out[name] = entry

    return ContractIR(out)


def _market_of_name(name: str) -> str | None:
    if name.startswith("us_"):
        return "us"
    if name.startswith("ashare_") or name.startswith("a_share"):
        return "ashare"
    return None


def _storage_of(ds: Any) -> tuple[str | None, str | None, str | None]:
    """#P1-21 严格区分 storage_backend / storage_layout / file_format。

    - backend：数据物理存放位置（local / cos / s3 / oss / httpfs / cli）。
    - layout：记录组织方式（daily_parquet / hive_date / hive_year / long / wide）。
    - format：底层文件格式（parquet / arrow / feather / csv）。

    修复：旧 ``_storage_backend_of`` 会把 ``storage_format``（long/wide，本质是
    layout）当作 backend 回退，导致 ContractIR 语义错位。
    """
    storage = getattr(ds, "storage", None)
    backend: str | None = None
    layout: str | None = None
    fmt: str | None = None
    if isinstance(storage, dict):
        src = storage.get("source")
        if isinstance(src, dict):
            backend = str(src.get("type") or "local")
            layout = src.get("layout")
            fmt = src.get("format")
        elif isinstance(src, str):
            backend = src
        layout = layout or storage.get("layout")
        fmt = fmt or storage.get("format")
    if backend in (None, "", "local") and getattr(ds, "storage_format", None):
        # 无显式 backend 时，storage_format（long/wide/daily…）只当 layout。
        layout = layout or str(ds.storage_format)
    if fmt is None:
        # #P2-77 file_format 统一走 registry 的 FormatSpec（此前 fallback 到
        # 不存在的 ``ds.file_format`` 属性 → 恒 "parquet"，CSV/Arrow/Feather 会报错格式）。
        fmt = getattr(getattr(ds, "format_spec", None), "type", None) or getattr(
            ds, "file_format", None
        ) or "parquet"
    return backend or "local", layout, fmt


def _is_temporal_dtype(f: Any) -> bool:
    """字段 dtype 是否像时间类型（防止把值字段当时间轴列的兜底判定）。"""
    text = str(getattr(f, "dtype", "") or "").lower()
    return any(tok in text for tok in ("time", "date"))


def _temporal_axes_of(reg_ds: Any, contract: Any, fields: Sequence[Any]) -> TemporalAxes:
    """从 registry + 契约 + 语义字段编译一个数据集的多根时间轴（#P0-9）。

    #P0-38 事件时间轴**禁止从值字段的 time_role 推导**：catalog 里 Close/Open/
    Volume/PeRatio 等值字段也标了 ``time_role: event_time``（那是"行级事件时刻"
    标注，不是时间轴列），旧代码会把它们编译成 ``event_time="Close"`` 这类
    无意义的时间轴。时间轴只能来自：
        - 契约时钟列：strict-PIT 事件表用 ``availability_column``（事件时钟），
          effective_time_only 用 ``event_column``；
        - registry roles / ``time_column``：panel 数据集的 bar 时间即事件时间。
    """
    partition_time = None
    if reg_ds is not None:
        partition_time = getattr(reg_ds, "time_column", None)
    knowledge_time = (
        getattr(contract, "availability_column", None)
        if contract is not None
        else None
    )
    effective_time = getattr(contract, "event_column", None) if contract is not None else None
    period_time = getattr(contract, "period_column", None) if contract is not None else None
    # 事件时间轴只来自契约时钟列 / registry time_column，绝不取自值字段。
    event_time = None
    if contract is not None:
        event_time = (
            getattr(contract, "availability_column", None)
            or getattr(contract, "event_column", None)
        )
    if event_time is None and partition_time is not None:
        event_time = partition_time
    decision_time = None
    for f in fields:
        # decision_time 同样不取自值字段：只接受显式标注且 dtype 像时间列。
        if (
            getattr(f, "time_role", None) == "decision_time"
            and decision_time is None
            and _is_temporal_dtype(f)
        ):
            decision_time = getattr(f, "physical_name", None) or getattr(f, "logical_name", None)
        if knowledge_time is None and getattr(f, "knowledge_time", None):
            knowledge_time = getattr(f, "knowledge_time", None)
        if period_time is None and getattr(f, "period_time", None):
            period_time = getattr(f, "period_time", None)
    return TemporalAxes(
        partition_time=partition_time,
        event_time=event_time,
        knowledge_time=knowledge_time,
        effective_time=effective_time,
        period_time=period_time,
        decision_time=decision_time,
        storage_timezone=getattr(contract, "storage_timezone", None)
        if contract is not None
        else None,
        semantic_timezone=getattr(contract, "semantic_timezone", None)
        if contract is not None
        else None,
    )


__all__ = ["ContractIR", "ContractIRDataset", "TemporalAxes", "build_contract_ir"]
