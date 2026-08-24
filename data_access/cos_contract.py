# -*- coding: utf-8 -*-
"""Executable semantic contracts for COS-backed datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Mapping

import numpy as np
import pyarrow as pa

from data_access.core.exceptions import ValidationError
from data_access.core.identity_encoder import CanonicalIdentityEncoder

logger = logging.getLogger("data_access.cos_contract")

_MODELS = {"D1", "E1", "E2", "S1", "X0", "STATIC", "EMPTY", "MINUTE", "RAW_EVENT"}
_PANEL = {"dense", "state_ready", "minute", "event_only", "sparse", "dimension", "forbidden"}
_PIT = {"strict", "effective_time_only", "unsupported", "not_applicable"}
_CALENDAR_DOMAINS = {"trade_day", "calendar_day", "event_time", "static"}
_SOURCE_CLASSES = {"cos_physical", "derived_dataset"}
_CARDINALITY = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}  # 相对股票日频面板
_TIME_REPRESENTATIONS = {None, "date_label", "instant"}
_TIME_PRECISIONS = {None, "date", "timestamp"}
_PIT_FIDELITIES = {
    "knowledge_date_pit",
    "vintage_pit",
    "effective_only",
    "unsupported",
}
_FILTER_CARDINALITIES = {"exactly_one", "one_or_more", "optional"}


@dataclass(frozen=True)
class COSDatasetContract:
    name: str
    market: str
    temporal_model: str
    join_policy: str
    instrument_column: str
    panel_policy: str
    pit_policy: str = "not_applicable"
    availability_column: str | None = None
    period_column: str | None = None
    event_column: str | None = None
    revision_columns: tuple[str, ...] = ()
    event_id_columns: tuple[str, ...] = ()
    required_panel_filters: tuple[str, ...] = ()
    required_event_filters: tuple[str, ...] = ()
    allowed_filter_values: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    return_column: str | None = None
    return_scale: float = 1.0
    adjustment_column: str | None = None
    adjustment_convention: str | None = None
    availability_must_follow_period: bool = False
    storage_layout: str | None = None
    note: str = ""
    # ---- 第四阶段（P0-1/P0-10/P1-13/P1-14/P1-15） ----
    calendar_domain: str = "trade_day"         # trade_day/calendar_day/event_time/static
    grain: str | None = None                   # 描述性 grain（如 "TradeDate+Symbol"）
    unique_key: tuple[str, ...] = ()           # 唯一性契约（相对面板：含时间+标的）
    cardinality: str = "one_to_one"            # one_to_one/one_to_many/many_to_one/many_to_many
    required_dimension_filters: tuple[str, ...] = ()  # 维度过滤（如 IndexSymbol/IndustrySource）
    coverage_start: str | None = None          # 覆盖起始（ISO 日期，声明）
    coverage_end: str | None = None            # 覆盖终止（ISO 日期，声明）
    expected_cadence: str | None = None        # daily/weekly/quarterly/annual/event_driven
    max_staleness: str | None = None           # 如 "7d"（超过告警/查询前提示）
    missing_partition_semantics: str = "error" # error/warn/empty_ok
    # ---- R45 P0-ASHARE：来源类别区分 ----
    # cos_physical = COS lqtp_data/massive_data 真源物理镜像（provenance / snapshot
    # identity / revision 语义均以 COS 物理源为准）；derived_dataset = 派生/本地研究
    # 数据集（如 universe_daily），不是 COS 物理源，**不得**被当作 COS 物理契约使用。
    source_class: str = "cos_physical"         # cos_physical / derived_dataset
    # ---- R24 P0-PIT2 §11：时间表示 / 精度与时区契约 ----
    time_representation: str | None = None     # date_label / instant
    time_precision: str | None = None          # date / timestamp
    storage_timezone: str | None = None        # 物理存储时区（如 UTC）
    semantic_timezone: str | None = None       # 语义时区（如 Asia/Shanghai / America/New_York）
    # ---- R24 P0-PIT4 §13：revision 语义拆分 ----
    # revision_columns 保留 = 去重 tiebreaker（如 UpdateTime）；而
    # revision_availability_time = 历史 revision 的 market-visible 时点（如
    # 尚无 = None/unknown）。A股 UpdateTime 是供应商 freshness，**不是**市场可知
    # 修订时间——绝不能充当 historical revision availability。
    revision_availability_time: str | None = None
    pit_fidelity: str = "knowledge_date_pit"   # knowledge_date_pit / vintage_pit / effective_only / unsupported
    # ---- R24 P0-PIT5 §14：结构化 filter requirement（exactly-one 等）----
    filter_cardinalities: tuple[tuple[str, str], ...] = ()  # (field, cardinality) 如 ("timeframe","exactly_one")

    def __post_init__(self) -> None:
        if self.temporal_model not in _MODELS or self.panel_policy not in _PANEL or self.pit_policy not in _PIT:
            raise ValueError(f"invalid COS contract: {self.name}")
        if self.calendar_domain not in _CALENDAR_DOMAINS:
            raise ValueError(f"invalid calendar_domain {self.calendar_domain!r}: {self.name}")
        if self.source_class not in _SOURCE_CLASSES:
            raise ValueError(f"invalid source_class {self.source_class!r}: {self.name}")
        if self.cardinality not in _CARDINALITY:
            raise ValueError(f"invalid cardinality {self.cardinality!r}: {self.name}")
        if self.pit_policy == "strict" and not self.availability_column:
            raise ValueError(f"strict PIT requires availability_column: {self.name}")
        if self.pit_policy == "effective_time_only" and not self.event_column:
            raise ValueError(f"effective-time PIT requires event_column: {self.name}")
        if self.temporal_model == "RAW_EVENT" and not self.availability_column:
            raise ValueError(f"RAW_EVENT requires availability_column: {self.name}")
        if self.time_representation not in _TIME_REPRESENTATIONS:
            raise ValueError(
                f"invalid time_representation {self.time_representation!r}: {self.name}"
            )
        if self.time_precision not in _TIME_PRECISIONS:
            raise ValueError(
                f"invalid time_precision {self.time_precision!r}: {self.name}"
            )
        if self.pit_fidelity not in _PIT_FIDELITIES:
            raise ValueError(f"invalid pit_fidelity {self.pit_fidelity!r}: {self.name}")
        for field_name, cardinality in self.filter_cardinalities:
            if cardinality not in _FILTER_CARDINALITIES:
                raise ValueError(
                    f"invalid filter_cardinality {cardinality!r} for {field_name}: {self.name}"
                )
            if self.required_event_filters and field_name not in self.required_event_filters and field_name not in self.required_panel_filters and field_name not in self.required_dimension_filters:
                raise ValueError(
                    f"filter_cardinality 声明的 {field_name} 不在 required_*_filters 中: {self.name}"
                )

    @property
    def is_event(self) -> bool:
        return self.temporal_model in {"E1", "E2"}

    @property
    def is_raw_event(self) -> bool:
        return self.temporal_model == "RAW_EVENT"

    @property
    def is_sparse(self) -> bool:
        return self.temporal_model == "X0"

    @property
    def is_empty(self) -> bool:
        return self.temporal_model == "EMPTY"

    @property
    def is_static(self) -> bool:
        return self.temporal_model == "STATIC"

    @property
    def factor_panel_allowed(self) -> bool:
        return self.panel_policy in {"dense", "state_ready", "minute"}

    @property
    def allowed_filters(self) -> dict[str, tuple[Any, ...]]:
        return dict(self.allowed_filter_values)

    @property
    def is_time_panel(self) -> bool:
        """该数据集是否可以（在语义上）作为 time×instrument 面板读取。"""
        return not (self.is_empty or self.is_sparse or self.is_event or self.is_raw_event or self.is_static)


_DEFAULT_PANEL = {"D1": "dense", "S1": "state_ready", "MINUTE": "minute", "E1": "event_only", "E2": "event_only", "RAW_EVENT": "event_only", "X0": "sparse", "STATIC": "dimension", "EMPTY": "forbidden"}
_DEFAULT_CALENDAR = {"D1": "trade_day", "S1": "trade_day", "MINUTE": "trade_day", "E1": "event_time", "E2": "event_time", "RAW_EVENT": "event_time", "X0": "event_time", "STATIC": "static", "EMPTY": "static"}
_DEFAULT_CARDINALITY = {"D1": "one_to_one", "S1": "one_to_one", "MINUTE": "one_to_one", "E1": "many_to_one", "E2": "many_to_one", "RAW_EVENT": "one_to_many", "X0": "many_to_one", "STATIC": "one_to_one", "EMPTY": "one_to_one"}


def _c(name: str, market: str, model: str, join: str, instrument: str, *, panel: str | None = None, pit: str | None = None, **kwargs: Any) -> COSDatasetContract:
    default_panel = _DEFAULT_PANEL[model]
    default_pit = "strict" if model in {"E1", "E2", "RAW_EVENT"} and kwargs.get("availability_column") else "not_applicable"
    # 只对显式 kwargs 提供语义默认；已显式传的优先。
    if "calendar_domain" not in kwargs:
        kwargs["calendar_domain"] = _DEFAULT_CALENDAR[model]
    if "cardinality" not in kwargs:
        kwargs["cardinality"] = _DEFAULT_CARDINALITY[model]
    return COSDatasetContract(name, market, model, join, instrument, panel or default_panel, pit or default_pit, **kwargs)


from .cos_contract_ashare import ASHARE_COS_CONTRACTS  # noqa: E402
from .cos_contract_us import US_COS_CONTRACTS  # noqa: E402

COS_DATASET_CONTRACTS = {**ASHARE_COS_CONTRACTS, **US_COS_CONTRACTS}

# ---- R45 P0-ASHARE：派生 / 本地研究数据集注册表（非 COS 物理源）----
# 与 ASHARE_COS_CONTRACTS / COS_DATASET_CONTRACTS **分离**：这些数据集不是 COS
# lqtp_data 物理镜像（字典 §8「非 COS / 不在本字典范围」），而是本地 clean_data 派生
# 数据集（如 universe_daily）。它们的 provenance / snapshot identity / revision 语义
# 均不同于 COS 物理源——必须单独登记，禁止伪装成 COS 物理契约（source_class=
# "derived_dataset"）。读取走统一的 runtime/read 层，但契约语义不进入 COS 物理注册表。
DERIVED_DATASET_CONTRACTS: dict[str, "COSDatasetContract"] = {
    name: contract
    for name, contract in COS_DATASET_CONTRACTS.items()
    if getattr(contract, "source_class", "cos_physical") == "derived_dataset"
}


def cos_physical_contracts() -> dict[str, "COSDatasetContract"]:
    """COS 物理源契约（source_class == 'cos_physical'）。

    ``COS_DATASET_CONTRACTS`` 同时承载派生/本地研究数据集（如 universe_daily），
    但只有 ``cos_physical`` 才是真 COS 物理源——provenance / snapshot identity /
    revision 语义以 COS 物理源为准。派生数据集见 :data:`DERIVED_DATASET_CONTRACTS`。
    """
    return {
        name: contract
        for name, contract in COS_DATASET_CONTRACTS.items()
        if getattr(contract, "source_class", "cos_physical") == "cos_physical"
    }


def get_cos_contract(dataset: str) -> COSDatasetContract | None:
    return COS_DATASET_CONTRACTS.get(str(dataset))


def require_cos_contract(dataset: str) -> COSDatasetContract:
    contract = get_cos_contract(dataset)
    if contract is None:
        raise ValidationError(f"数据集 {dataset!r} 没有 COS 语义契约；不得推断 panel/PIT 语义")
    if contract.source_class != "cos_physical":
        raise ValidationError(
            f"数据集 {dataset!r} 是 source_class={contract.source_class!r}（派生/本地研究数据集），"
            "不是 COS 物理源；不得按 COS 物理契约（provenance/snapshot/revision 语义）使用。"
        )
    return contract


def _validate_filters(contract: COSDatasetContract, filters: Mapping[str, Any] | None, required: tuple[str, ...], context: str) -> dict[str, Any]:
    supplied = dict(filters or {})
    missing = [name for name in required if name not in supplied or supplied[name] in (None, "")]
    if missing:
        raise ValidationError(f"数据集 {contract.name!r} 的{context}必须显式提供过滤条件 {missing}")
    for name, value in supplied.items():
        choices = contract.allowed_filters.get(name)
        if choices is None:
            continue
        values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        bad = [item for item in values if item not in choices]
        if bad:
            raise ValidationError(f"数据集 {contract.name!r} 的过滤条件 {name}={bad!r} 不在允许值 {choices!r} 中")
    # R24 P0-PIT5 §14：结构化 filter cardinality（如 timeframe exactly-one）。
    for field_name, cardinality in contract.filter_cardinalities:
        present = field_name in supplied and supplied[field_name] not in (None, "")
        if cardinality == "exactly_one":
            if not present:
                raise ValidationError(
                    f"数据集 {contract.name!r} 的过滤条件 {field_name} 必须提供且只能一个值"
                    f"（cardinality=exactly_one，{context}）。missing/multiple/invalid 一律拒绝。"
                )
            values = supplied[field_name]
            count = len(values) if isinstance(values, (list, tuple, set, frozenset)) else 1
            if count != 1:
                raise ValidationError(
                    f"数据集 {contract.name!r} 的过滤条件 {field_name} 只允许一个值"
                    f"（cardinality=exactly_one），收到 {values!r}（{context}）"
                )
    return supplied


def validate_panel_request(dataset: str, *, semantic_filters: Mapping[str, Any] | None = None, allow_sparse: bool = False) -> COSDatasetContract:
    contract = require_cos_contract(dataset)
    if contract.is_empty:
        raise ValidationError(f"数据集 {dataset!r} 是 EMPTY 占位表，禁止作为数据源")
    if contract.is_event:
        raise ValidationError(f"数据集 {dataset!r} 是 {contract.temporal_model} 事件表；必须使用事件/PIT API")
    if contract.is_sparse and not allow_sparse:
        raise ValidationError(f"数据集 {dataset!r} 是 X0 稀疏表，不是完整日频面板")
    if contract.is_static:
        raise ValidationError(f"数据集 {dataset!r} 是 STATIC 维表，不是时间×标的面板")
    if not contract.factor_panel_allowed and not (contract.is_sparse and allow_sparse):
        raise ValidationError(f"数据集 {dataset!r} panel_policy={contract.panel_policy!r}，禁止面板化")
    _validate_filters(contract, semantic_filters, contract.required_panel_filters, " panel 读取")
    return contract


def resolve_event_clock(dataset: str, *, allow_effective_time: bool = False) -> tuple[COSDatasetContract, str]:
    contract = require_cos_contract(dataset)
    if not (contract.is_event or contract.is_raw_event):
        raise ValidationError(f"数据集 {dataset!r} 不是 E1/E2/RAW_EVENT 事件表")
    if contract.pit_policy == "strict" and contract.availability_column:
        return contract, contract.availability_column
    if contract.pit_policy == "effective_time_only" and contract.event_column:
        if not allow_effective_time:
            raise ValidationError(f"数据集 {dataset!r} 没有可靠公告/可知时间；仅可显式按生效日保守使用")
        return contract, contract.event_column
    raise ValidationError(f"数据集 {dataset!r} pit_policy={contract.pit_policy!r}，不支持 PIT")


def validate_event_filters(contract: COSDatasetContract, event_filters: Mapping[str, Any] | None) -> dict[str, Any]:
    return _validate_filters(contract, event_filters, contract.required_event_filters, "事件/PIT 读取")


def enforce_event_cutoff(
    contract: COSDatasetContract,
    *,
    as_of: Any = None,
    time_range: tuple[Any, Any] | None = None,
    production: bool = False,
) -> None:
    """#16 Event 未来数据 cutoff。

    ``effective_time_only``（A股/美股 Dividend、拆分事件）没有可靠公告时点，
    只允许显式按生效日保守使用；且生效日**不得晚于** as_of / 时间窗上界。
    美股 Dividend 甚至可能有未来日期文件——直接当普通 parquet 读会前视。
    """
    if contract.pit_policy != "effective_time_only":
        return
    end = None
    if time_range is not None and time_range[1] is not None:
        end = time_range[1]
    if end is None and as_of is not None:
        end = as_of
    if end is None:
        # #P0-13 无 end 的无界查询：effective_time_only 事件没有可靠公告时点，
        # 无界读取可能读到未来生效事件（前视）。
        import datetime as _dt

        if production:
            name = getattr(contract, "name", None) or getattr(contract, "dataset", None)
            raise ValidationError(
                f"数据集 {name!r} 是 effective_time_only 事件表，读取必须给出"
                "生效日上界（time_range.end 或 as_of）；无界读取可能读到未来生效"
                "事件（前视），production fail-closed。"
            )
        end = _dt.date.today()  # research 默认 cutoff 到今天
    import datetime as _dt

    def _to_date(v: Any) -> _dt.date | None:
        if isinstance(v, _dt.datetime):
            return v.date()
        if isinstance(v, _dt.date):
            return v
        if isinstance(v, str):
            try:
                return _dt.date.fromisoformat(str(v)[:10])
            except ValueError:
                return None
        return None

    end_date = _to_date(end)
    today = _dt.date.today()
    if end_date is None:
        return
    if end_date > today:
        msg = (
            f"数据集 {contract.name!r} 是 effective_time_only 事件表，"
            f"事件窗上界 {end_date.isoformat()} 晚于今天 {today.isoformat()}，"
            "会读到未来生效事件（前视）。请把 time_range / as_of 限制到已发生日期。"
        )
        if production:
            raise ValidationError(msg)
        logger.warning("%s（research 放行）", msg)


def validate_join_fanout(
    contract: COSDatasetContract,
    *,
    join_key: tuple[str, ...],
    production: bool = False,
    applied_filter_columns: Sequence[str] | None = None,
    applied_filter_values: Mapping[str, Sequence[Any]] | None = None,
) -> None:
    """#10 join fan-out 守卫。

    相对股票日频面板，``unique_key`` 不含 (time, instrument) 的数据集在
    exact join 时会产生 fan-out（一行锚点对多行右表）。已知 one-to-many 表
    （TopTen/Industry/美股 capital shares）要求调用方显式声明聚合/去重策略。

    豁免：
        - ``cardinality == one_to_one``；
        - join key 覆盖 unique_key；
        - ``required_dimension_filters`` 已全部出现在 applied_filter_columns
          **且每个 filter 值都是单值**（#P0-14 单值证明）——如
          ``IndustrySource = 'sw_l1'`` 才能证明行业表退化为每日每标的一行；
          ``IndustrySource IN ('sw_l1','sw_l2')`` 每个标的多行业仍 one-to-many。
    """
    if contract.cardinality == "one_to_one":
        return
    # join key 必须覆盖完整 unique_key 才是 1:1 对齐；否则会 fan-out。
    if contract.unique_key and set(contract.unique_key) <= set(join_key):
        return
    if contract.required_dimension_filters:
        applied = set(applied_filter_columns or ())
        if set(contract.required_dimension_filters) <= applied:
            # #P0-14 单值证明：IN 多值 / 范围过滤不能证明退化为一行
            vals = applied_filter_values or {}
            single = all(
                len(tuple(vals.get(col, ()))) == 1
                for col in contract.required_dimension_filters
            )
            if single:
                return
    if contract.cardinality in {"one_to_many", "many_to_many"}:
        msg = (
            f"数据集 {contract.name!r} cardinality={contract.cardinality}，"
            f"相对面板 join key {join_key!r} 会产生行放大（fan-out）。"
            "请在 join 前显式聚合/去重（如按 TopTen 先聚合成每日一行），"
            "或补齐 required_dimension_filters 过滤。"
        )
        if production:
            raise ValidationError(msg)
        logger.warning("%s（research 放行）", msg)


def normalize_return_values(values: Any, dataset: str) -> Any:
    contract = require_cos_contract(dataset)
    if contract.return_column is None:
        raise ValidationError(f"数据集 {dataset!r} 未声明收益字段")
    scale = float(contract.return_scale)
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        import pyarrow.compute as pc
        return pc.multiply(pc.cast(values, pa.float64(), safe=False), pa.scalar(scale, type=pa.float64()))
    return values.astype(float) * scale if hasattr(values, "astype") else np.asarray(values, dtype=float) * scale


def market_scoped_instrument(market: str | None, instrument: str) -> str:
    """R24 P1-XM6 §22：跨市场统一结果中不要只有裸 instrument。

    推荐 ``ashare:000001.SZ`` / ``us:AAPL`` 形式。主键逻辑必须是
    ``(market, instrument)``——避免未来港股 / ETF / index / ADR / 同 ticker
    空间冲突。物理源仍保留原 vendor code（``instrument`` 原样）。

    ``market`` 为 None/未知时保持裸 instrument（不猜市场，避免错误加前缀）。
    """
    if not market:
        return str(instrument)
    return f"{market}:{instrument}"


def semantic_contract_fingerprint() -> str:
    payload = {
        name: asdict(contract)
        for name, contract in sorted(COS_DATASET_CONTRACTS.items())
    }
    return CanonicalIdentityEncoder(strict=True).hash_identity(payload, bits=256)


__all__ = ["COSDatasetContract", "COS_DATASET_CONTRACTS", "DERIVED_DATASET_CONTRACTS", "cos_physical_contracts", "get_cos_contract", "require_cos_contract", "validate_panel_request", "resolve_event_clock", "validate_event_filters", "normalize_return_values", "semantic_contract_fingerprint"]
