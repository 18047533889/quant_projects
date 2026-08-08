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
class ContractIRDataset:
    """一个数据集的统一契约视图（编译产物）。

    #34：除表级时间语义外，扩展字段级契约（knowledge/effective/period_time、
    revision_order、required_filters、allowed_filter_values、unique_key、
    duplicate_policy）与存储/查询策略（partitioning、storage_backend、
    query_policy、coverage_policy），让运行时不再分别读三套定义。
    """

    name: str
    in_registry: bool = False
    in_contracts: bool = False
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
    query_policy: dict[str, Any] = field(default_factory=dict)
    coverage_policy: str | None = None

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
            if d.in_contracts and not d.in_registry:
                problems.append(
                    f"契约数据集 {name!r} 不在 registry（或未显式标记 external）"
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
            entry.required_filters = list(
                dict.fromkeys(
                    list(contract.required_panel_filters or ())
                    + list(contract.required_dimension_filters or ())
                )
            )
            entry.allowed_filter_values = dict(contract.allowed_filter_values or ())
            entry.coverage_policy = contract.missing_partition_semantics
        if reg_ds is not None:
            entry.partitioning = dict(getattr(reg_ds, "partitioning", None) or {})
            entry.storage_backend = _storage_backend_of(reg_ds)
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
            if not entry.duplicate_policy:
                dup = {
                    getattr(f, "duplicate_policy", None)
                    for f in catalog._fields.values()
                    if f.dataset == name and getattr(f, "duplicate_policy", None)
                }
                if len(dup) == 1:
                    entry.duplicate_policy = next(iter(dup))
        # 一致性 issue：契约在但 registry 没有 → 未标记 external
        if entry.in_contracts and not entry.in_registry and name not in external:
            entry.issues.append(
                f"契约数据集 {name!r} 不在 registry 且未标记 external"
            )
        out[name] = entry

    return ContractIR(out)


def _market_of_name(name: str) -> str | None:
    if name.startswith("us_"):
        return "us"
    if name.startswith("ashare_") or name.startswith("a_share"):
        return "ashare"
    return None


def _storage_backend_of(ds: Any) -> str | None:
    """从 registry 数据集的 storage/engine 声明推断 storage backend（#34）。"""
    storage = getattr(ds, "storage", None)
    if isinstance(storage, dict):
        src = storage.get("source")
        if isinstance(src, dict):
            return str(src.get("type") or "local")
        if isinstance(src, str):
            return src
    if getattr(ds, "storage_format", None):
        return str(ds.storage_format)
    return "local"


__all__ = ["ContractIR", "ContractIRDataset", "build_contract_ir"]
