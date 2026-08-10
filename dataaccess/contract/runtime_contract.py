"""R25 §22/23 —— RuntimeDatasetContract 与 ContractCompiler。

启动时把 datasets.yaml + COSDatasetContract + SemanticFieldCatalog + Mirror/storage
声明 + security 策略编译成一个 ``RuntimeDatasetContract``（per-dataset 运行时单一
事实源）。此后 runtime 只消费它，不再分别调用 registry / get_cos_contract /
semantic_catalog / mirror registry 各自重新解释。

与 ContractIR（read/contract_ir.py）的关系：ContractIR 是只读合并视图（审计用），
本模块是**可执行**运行时契约——物理分区时钟 / 过滤要求 / 时间轴表示 / 单位契约 /
安全分类 全部显式化。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

from .filters import FilterRequirement, build_filter_requirements_from_contract
from .physical_partition import (
    MissingPartitionSemantics,
    PhysicalLayout,
    PhysicalPartitionSpec,
    layout_from_contract,
    layout_from_mirror,
)
from .temporal_axis import TemporalAxisSpec


@dataclass(frozen=True)
class StorageContract:
    """物理存储声明（backend / layout / format / prefix）。"""

    backend: str = "local"        # local / cos / s3 / oss / httpfs / cli
    layout: str | None = None     # 旧字符串布局（daily_parquet / period_files / ...）
    file_format: str = "parquet"
    prefix: str | None = None     # cos:// 或 s3:// 前缀（remote 授权边界）


@dataclass(frozen=True)
class PITContract:
    """PIT 契约（表级）。"""

    pit_policy: str = "not_applicable"       # strict / effective_time_only / unsupported / ...
    availability_column: str | None = None   # knowledge 时钟（filing_date / PubDate）
    event_column: str | None = None          # effective 时钟（ex_dividend_date）
    period_column: str | None = None         # 会计期间列（period_end / ReportPeriodEndDate）
    time_representation: str | None = None   # date_label / instant
    time_precision: str | None = None        # date / timestamp
    semantic_timezone: str | None = None
    storage_timezone: str | None = None
    revision_availability_time: str | None = None  # 历史修订 market-visible 时点
    pit_fidelity: str = "knowledge_date_pit"       # knowledge_date_pit / vintage_pit / ...
    dedup_tiebreaker: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnitContract:
    """字段级单位契约（R24 P0-XM + R25 §36/37/38）。"""

    dimension: str | None = None
    scale: float = 1.0
    currency: str | None = None
    currency_column: str | None = None
    cross_market_comparable: bool = True
    requires_fx: bool = False
    flow_semantics: str | None = None         # cumulative_ytd_flow / single_period_flow / point_in_time_stock
    definition_id: str | None = None
    definition_version: str | None = None


@dataclass(frozen=True)
class CardinalityContract:
    """唯一性/基数契约。"""

    cardinality: str = "one_to_one"
    unique_key: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageContract:
    """覆盖声明。"""

    coverage_start: str | None = None
    coverage_end: str | None = None
    expected_cadence: str | None = None
    max_staleness: str | None = None
    missing_partition_semantics: str = "error"


@dataclass(frozen=True)
class DatasetSecurityContract:
    """数据集级安全分类（R25 §20 / P0-019）。"""

    classification: str = "public"             # public / internal / restricted / premium
    access_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticSchemaContract:
    """字段级语义 schema。"""

    fields: tuple[str, ...] = ()
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeDatasetContract:
    """一个数据集的运行时契约（R25 §22）。"""

    dataset: str
    market: str | None
    storage: StorageContract
    physical_partition: PhysicalPartitionSpec
    temporal_axes: Mapping[str, TemporalAxisSpec]
    pit: PITContract
    filters: tuple[FilterRequirement, ...]
    units: Mapping[str, UnitContract]
    cardinality: CardinalityContract
    coverage: CoverageContract
    security: DatasetSecurityContract
    schema: SemanticSchemaContract
    fingerprint: str

    def filter_requirement_for(self, field: str) -> FilterRequirement | None:
        for req in self.filters:
            if req.field == field:
                return req
        return None

    def temporal_axis(self, name: str) -> TemporalAxisSpec | None:
        return self.temporal_axes.get(name)

    def unit_for(self, field: str) -> UnitContract | None:
        return self.units.get(field)

    @property
    def requires_fx(self) -> bool:
        return any(u.requires_fx for u in self.units.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "market": self.market,
            "storage": asdict(self.storage),
            "physical_partition": self.physical_partition.to_dict(),
            "temporal_axes": {
                k: v.to_dict() for k, v in sorted(self.temporal_axes.items())
            },
            "pit": asdict(self.pit),
            "filters": [asdict(f) for f in self.filters],
            "units": {k: asdict(v) for k, v in sorted(self.units.items())},
            "cardinality": asdict(self.cardinality),
            "coverage": asdict(self.coverage),
            "security": asdict(self.security),
            "schema": asdict(self.schema),
            "fingerprint": self.fingerprint,
        }


class ContractCompiler:
    """把 registry + COS 契约 + semantic catalog + mirror/storage + security 编译成
    RuntimeDatasetContract（R25 §22）。"""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def compile(self, dataset: str) -> RuntimeDatasetContract | None:
        from data_access.cos_contract import get_cos_contract
        from data_access.read.semantic_catalog import get_semantic_catalog

        # R26-P0-011：registry.get 失败不吞——compile 异常由 ``_compile_contract_strict``
        # 在 production 下 hard fail（registry 不可用 ≠ 无 contract ≠ 旧逻辑继续）。
        ds = self._registry.get(dataset)
        contract = get_cos_contract(dataset)
        catalog = get_semantic_catalog()
        if ds is None and contract is None:
            return None

        # ---- storage / physical partition ----
        storage = self._storage_of(ds, contract)
        physical = self._physical_partition_of(dataset, ds, contract)

        # ---- temporal axes ----
        temporal_axes = self._temporal_axes_of(dataset, contract, catalog)

        # ---- PIT ----
        pit = PITContract(
            pit_policy=getattr(contract, "pit_policy", "not_applicable")
            if contract else "not_applicable",
            availability_column=getattr(contract, "availability_column", None)
            if contract else None,
            event_column=getattr(contract, "event_column", None) if contract else None,
            period_column=getattr(contract, "period_column", None) if contract else None,
            time_representation=getattr(contract, "time_representation", None)
            if contract else None,
            time_precision=getattr(contract, "time_precision", None) if contract else None,
            semantic_timezone=getattr(contract, "semantic_timezone", None)
            if contract else None,
            storage_timezone=getattr(contract, "storage_timezone", None)
            if contract else None,
            revision_availability_time=getattr(contract, "revision_availability_time", None)
            if contract else None,
            pit_fidelity=getattr(contract, "pit_fidelity", "knowledge_date_pit")
            if contract else "knowledge_date_pit",
            dedup_tiebreaker=tuple(getattr(contract, "revision_columns", ()) or ())
            if contract else (),
        )

        # ---- filters（含 required_event_filters，P0-003/004）----
        filter_reqs = build_filter_requirements_from_contract(contract)

        # ---- units（catalog 字段级）----
        units = self._units_of(dataset, contract, catalog)

        # ---- cardinality / coverage / security / schema ----
        cardinality = CardinalityContract(
            cardinality=getattr(contract, "cardinality", "one_to_one")
            if contract else "one_to_one",
            unique_key=tuple(getattr(contract, "unique_key", ()) or ())
            if contract else (),
        )
        coverage = CoverageContract(
            coverage_start=getattr(contract, "coverage_start", None) if contract else None,
            coverage_end=getattr(contract, "coverage_end", None) if contract else None,
            expected_cadence=getattr(contract, "expected_cadence", None) if contract else None,
            max_staleness=getattr(contract, "max_staleness", None) if contract else None,
            missing_partition_semantics=getattr(contract, "missing_partition_semantics", "error")
            if contract else "error",
        )
        security = self._security_of(dataset, ds, contract)
        schema = SemanticSchemaContract(
            fields=tuple(getattr(ds, "schema", None) or {} if ds is not None else {}),
            schema=dict(getattr(ds, "schema", None) or {}) if ds is not None else {},
        )

        # ---- fingerprint（编译产物完整哈希）----
        # R26-P1-001：必须包含 temporal_axes 与 schema——date_label→instant /
        # schema 变化必须改 fingerprint。不人工维护字段列表，直接对完整
        # canonical payload 哈希。
        # R26-P1-004：market 优先级 = 显式 contract.market > registry market > name 推断。
        market = _resolve_market(dataset, ds, contract)
        payload = {
            "dataset": dataset,
            "market": market,
            "storage": asdict(storage),
            "physical_partition": physical.to_dict(),
            "pit": asdict(pit),
            "filters": [asdict(f) for f in filter_reqs],
            "units": {k: asdict(v) for k, v in sorted(units.items())},
            "cardinality": asdict(cardinality),
            "coverage": asdict(coverage),
            "security": asdict(security),
            "temporal_axes": {
                k: v.to_dict() for k, v in sorted(temporal_axes.items())
            },
            "schema": asdict(schema),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        # R26-P1-002：deep freeze——frozen dataclass 不等于 immutable，外部原地改
        # temporal_axes/units/schema 会不改 fingerprint。编译后转 MappingProxyType /
        # tuple，禁止原地修改。
        frozen_axes = MappingProxyType(dict(temporal_axes))
        frozen_units = MappingProxyType(dict(units))
        return RuntimeDatasetContract(
            dataset=dataset,
            market=market,
            storage=storage,
            physical_partition=physical,
            temporal_axes=frozen_axes,  # type: ignore[arg-type]
            pit=pit,
            filters=filter_reqs,
            units=frozen_units,  # type: ignore[arg-type]
            cardinality=cardinality,
            coverage=coverage,
            security=security,
            schema=SemanticSchemaContract(
                fields=schema.fields,
                schema=MappingProxyType(dict(schema.schema)),
            ),
            fingerprint=fingerprint,
        )

    # ---- 子编译 ----

    def _storage_of(self, ds: Any, contract: Any) -> StorageContract:
        from data_access.core.storage import (
            declared_storage_layout,
            declared_storage_type,
        )

        backend = declared_storage_type(ds) if ds is not None else None
        layout = declared_storage_layout(ds) if ds is not None else None
        if contract is not None and getattr(contract, "storage_layout", None):
            layout = layout or contract.storage_layout
        fmt = "parquet"
        fmt_spec = getattr(ds, "format_spec", None) if ds is not None else None
        if fmt_spec is not None:
            fmt = getattr(fmt_spec, "type", None) or "parquet"
        prefix = None
        if ds is not None:
            from data_access.core.storage import declared_storage_uri

            prefix = declared_storage_uri(ds)
        return StorageContract(
            backend=backend or "local",
            layout=layout,
            file_format=fmt,
            prefix=prefix,
        )

    def _physical_partition_of(self, dataset: str, ds: Any, contract: Any) -> PhysicalPartitionSpec:
        """physical_partition：契约 storage_layout 优先，其次 mirror layout。

        **P0-001**：US finance contract 声明 ``period_files`` → 必须编译成
        ``PERIOD_END_FILE / partition_clock=period_end / {period_end}.parquet``，
        而**不是** mirror 的 daily_parquet。契约优先于 mirror（mirror 只保留
        deployment/location，业务语义以 RuntimeDatasetContract 为准）。
        """
        # 1) COS 契约 storage_layout（period_files / event_files / daily_parquet ...）
        # R26-P1-006：authoritative 声明 typo 必须 startup fail，不能 fallback 到
        # daily——「契约不可用」≠「旧逻辑继续」。
        contract_layout = None
        if contract is not None:
            raw_layout = getattr(contract, "storage_layout", None)
            if raw_layout:
                contract_layout = layout_from_contract(raw_layout)
            elif contract is not None:
                contract_layout = None

        # 2) mirror layout（退路）
        mirror_layout = None
        try:
            from data_access.cos.mirror import mirror_spec_for_dataset

            spec = mirror_spec_for_dataset(dataset)
            if spec is not None:
                mirror_layout = layout_from_mirror(getattr(spec, "layout", None))
        except Exception:
            mirror_layout = None

        layout = contract_layout or mirror_layout or PhysicalLayout.DAILY_TRADE_DATE
        partition_clock = None
        filename_template = None
        completeness = MissingPartitionSemantics.ERROR
        file_selector = None
        query_clock = None

        if layout == PhysicalLayout.PERIOD_END_FILE:
            # US finance：文件名 = period_end；predicate = filing_date。
            partition_clock = "period_end"
            filename_template = "{period_end}.parquet"
            if contract is not None:
                query_clock = getattr(contract, "availability_column", None) or "filing_date"
                completeness = _missing_semantics_of(contract)
        elif layout == PhysicalLayout.EVENT_DATE_FILE:
            partition_clock = "date"
            filename_template = "{date}.parquet"
            if contract is not None:
                event_col = getattr(contract, "event_column", None)
                partition_clock = event_col or "date"
                filename_template = "{" + (event_col or "date") + "}.parquet"
                query_clock = getattr(contract, "availability_column", None) or "date"
                completeness = _missing_semantics_of(contract)
        elif layout == PhysicalLayout.PREFIXED_DATE_FILE:
            # StockCapital：split={date}.parquet、shares=shares_{date}.parquet。
            # file_selector 区分前缀（由 mirror/registry 声明）。
            partition_clock = "date"
            filename_template = "{date}.parquet"
            if spec is not None and getattr(spec, "file_name", None):
                selector = getattr(spec, "file_selector", None)
                file_selector = selector
                if selector:
                    filename_template = f"{selector}{{date}}.parquet"
            completeness = _missing_semantics_of(contract)
        elif layout in {
            PhysicalLayout.DAILY_TRADE_DATE,
            PhysicalLayout.DAILY_CALENDAR_DATE,
        }:
            partition_clock = "date"
            filename_template = "{date}.parquet"
            completeness = _missing_semantics_of(contract)
        elif layout == PhysicalLayout.HIVE_DATE:
            partition_clock = "date"
            filename_template = "date={date}/data.parquet"
            completeness = _missing_semantics_of(contract)
        elif layout == PhysicalLayout.HIVE_YEAR:
            partition_clock = "year"
            filename_template = "year={year}/data.parquet"
            completeness = _missing_semantics_of(contract)
        elif layout == PhysicalLayout.STATIC_SINGLE:
            completeness = MissingPartitionSemantics.ERROR

        return PhysicalPartitionSpec(
            layout=layout,
            partition_clock=partition_clock,
            filename_template=filename_template,
            file_selector=file_selector,
            query_clock=query_clock,
            completeness=completeness,
            source="contract" if contract_layout else "mirror",
        )

    def _temporal_axes_of(
        self, dataset: str, contract: Any, catalog: Any
    ) -> Mapping[str, TemporalAxisSpec]:
        axes: dict[str, TemporalAxisSpec] = {}
        if contract is None:
            return axes
        tz = getattr(contract, "semantic_timezone", None)
        storage_tz = getattr(contract, "storage_timezone", None)
        repr_ = getattr(contract, "time_representation", None) or "date_label"
        prec_ = getattr(contract, "time_precision", None) or "date"
        precision = "date" if prec_ == "date" else "second"

        for role, col in (
            ("knowledge", getattr(contract, "availability_column", None)),
            ("effective", getattr(contract, "event_column", None)),
            ("period", getattr(contract, "period_column", None)),
        ):
            if col:
                axes[role] = TemporalAxisSpec(
                    column=col,
                    representation=repr_,  # type: ignore[arg-type]
                    precision=precision,  # type: ignore[arg-type]
                    storage_timezone=storage_tz,
                    semantic_timezone=tz,
                )
        return axes

    def _units_of(self, dataset: str, contract: Any, catalog: Any) -> Mapping[str, UnitContract]:
        units: dict[str, UnitContract] = {}
        if catalog is None:
            return units
        for field in catalog._fields.values():
            if getattr(field, "dataset", None) != dataset:
                continue
            units[field.logical_name] = UnitContract(
                dimension=getattr(field, "dimension", None),
                scale=getattr(field, "unit_scale", None) or getattr(field, "scale", 1.0),
                currency=getattr(field, "currency", None),
                currency_column=getattr(field, "currency_column", None),
                cross_market_comparable=getattr(field, "cross_market_comparable", True),
                requires_fx=getattr(field, "requires_fx", False),
                flow_semantics=getattr(field, "flow_semantics", None),
                definition_id=getattr(field, "definition_id", None),
                definition_version=getattr(field, "definition_version", None),
            )
        return units

    def _security_of(self, dataset: str, ds: Any, contract: Any) -> DatasetSecurityContract:
        # R26-P1-007：无显式 policy 时 classification = UNCLASSIFIED（不能乐观默认
        # public）。production/automated mining 对 unknown 应 reject。
        classification = "unclassified"
        tags: tuple[str, ...] = ()
        try:
            policy = self._registry.access_policy_for_dataset(dataset)
        except Exception:
            policy = None
        if policy is not None:
            classification = (
                getattr(policy, "classification", None) or "unclassified"
            )
            tags = tuple(getattr(policy, "access_tags", ()) or ())
        return DatasetSecurityContract(classification=str(classification), access_tags=tags)


def _market_of_name(name: str) -> str | None:
    if name.startswith("us_"):
        return "us"
    if name.startswith("ashare_") or name.startswith("a_share"):
        return "ashare"
    return None


def _resolve_market(dataset: str, ds: Any, contract: Any) -> str | None:
    """R26-P1-004：market 优先级 = 显式 contract.market > registry market > name 推断。"""
    if contract is not None:
        cm = getattr(contract, "market", None)
        if cm:
            return str(cm)
    if ds is not None:
        dm = getattr(ds, "market", None)
        if dm:
            return str(dm)
    return _market_of_name(dataset)


def _missing_semantics_of(contract: Any) -> MissingPartitionSemantics:
    raw = getattr(contract, "missing_partition_semantics", "error") or "error"
    # R26-P1-006：契约声明非法 → fail（不静默默认 ERROR 掩盖配置 typo）。
    return MissingPartitionSemantics(str(raw).lower())


_compilers: dict[int, ContractCompiler] = {}


def get_runtime_contract_compiler(registry: Any) -> ContractCompiler:
    """R26-P1-003：ContractCompiler 按 registry 绑定（不再是 first-registry 全局单例）。

    registry A 绑定的 compiler 绝不用于 registry B / test registry / namespace
    registry。使用方也应在 Store 上显式持有（``self.contract_compiler``）。
    """
    key = id(registry)
    cached = _compilers.get(key)
    if cached is None:
        cached = ContractCompiler(registry)
        _compilers[key] = cached
    return cached


def compile_runtime_contract(dataset: str, registry: Any) -> RuntimeDatasetContract | None:
    """便捷入口：编译单数据集运行时契约。"""
    return get_runtime_contract_compiler(registry).compile(dataset)
