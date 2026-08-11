"""
data_access.read.semantic_catalog —— 字段语义单一事实源（SemanticFieldCatalog）

职责
    1. 把「逻辑字段名 → 物理数据集 + 物理列」的映射集中管理，作为 FactorEngine
       FIELD_REGISTRY 与 DataAccess schema/COS 契约之上的统一语义层。
    2. 每个字段声明 dtype / frequency / grain / source_unit / canonical_unit /
       scale（单位归一化乘子）/ 时间语义角色 / PIT·join 策略 / aliases /
       mining_allowed。
    3. 提供 ``scale_to_canonical`` 的输出层归一化（``normalize_table_units``），
       使 Return(bp)、ROE(%)、TurnoverRatio(%) 等字段在 DataAccess 输出层统一
       变成 canonical 单位，FactorEngine 不再二次修正。

设计要点
    1. 配置放 ``config/semantic_fields.yaml``；进程内惰性加载并缓存。
    2. 解析时若逻辑名不在 catalog，store.resolve_fields 会回退到 registry 的
       dataset schema（把物理列名当逻辑名），保证「物理列 == 逻辑名」的老代码
       不受影响。
    3. 单位归一化是显式行为：``normalize_units=True`` 时对浮点列乘 scale；
       默认 False，保持与旧 read 路径逐字节一致。
    4. scale 的语义：从 source_unit 换算到 canonical_unit 的乘子
       （percent→ratio 为 0.01；bp→ratio 为 0.0001；无换算为 None 或 1.0）。

非职责
    不做文件 IO；不解析 datasets.yaml（registry）；不替代 COS PIT 契约（那套
    是单数据集的时间可见性语义，这里是跨数据集字段语义的权威登记处）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

logger = logging.getLogger("data_access.semantic_catalog")

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class SemanticField:
    """一个逻辑字段的完整语义声明（catalog 的原子条目）。"""

    logical_name: str
    dataset: str | None = None         # 物理数据集（registry 名）；#P0-24 derived 字段可无
    physical_name: str | None = None   # 数据集内的物理列名（derived 字段为表达式描述）
    market: str = "any"                # ashare / us / any（any=跨市场通用）
    dtype: str | None = None           # float64 / int64 / string / date ...
    frequency: str | None = None       # daily / minute / tick / quarterly ...
    grain: str | None = None           # instrument / cross_section / snapshot ...
    source_unit: str | None = None     # percent / bp / yuan / yuan_mm ...
    canonical_unit: str | None = None  # ratio / yuan ...
    scale: float | None = None         # source → canonical 的乘子（None 表示不归一化）
    time_role: str | None = None       # event_time / knowledge_time / effective_time ...
    temporal_model: str | None = None  # panel / financial_event / sparse_event ...
    knowledge_time: str | None = None  # 数据可见时间列（如 PubDate）
    effective_time: str | None = None  # 数据生效时间列（如 ReportPeriodEndDate）
    period_time: str | None = None     # 会计期间列（别名 ReportPeriodEndDate）
    join_policy: str | None = None     # exact / pit_asof_backward / latest_period ...
    required_filters: tuple[str, ...] = ()   # 读取时必须附带的条件（如行业过滤）
    revision_order: tuple[str, ...] = ()     # join 前按 (instrument, knowledge_time) 去重的版本排序列
    availability: str = "same_day"           # same_day / next_trading_day（PIT 可见性）
    # #P1-final closure 5 availability_latency：额外可见性延迟（bar 数）。事件
    # helper（read_cos_events_asof）在编译出的 available_from 上叠加它。
    availability_latency: int | None = None
    primary_key: tuple[str, ...] = ()        # exact join 唯一性契约（如 (TradeDate, Symbol)）
    duplicate_policy: str = "latest_revision"  # keep_first / keep_last / latest_revision / error
    aliases: tuple[str, ...] = ()            # 其它叫法（含 FactorEngine 里的别名）
    mining_allowed: bool = True
    period_selection: str = "all"            # latest_period/exact_period/annual/quarterly/ttm/all
    period_values: tuple[Any, ...] = ()      # exact_period 的目标 period 值
    # #P0-24 derived 字段：不是某张物理表的列，而是多表/多列的派生表达式。
    # derived_from 列出依赖的 (dataset.column)；coverage audit 对 derived 字段跳过
    # COLUMN_MISSING 检查（它们不落盘）。
    derived_expression: str | None = None
    derived_from: tuple[str, ...] = ()
    # ---- R24 P0-PIT2/3/4 §11-13：时间表示与 PIT fidelity ----
    time_representation: str | None = None     # date_label / instant
    time_precision: str | None = None          # date / timestamp
    pit_fidelity: str | None = None            # knowledge_date_pit / vintage_pit / effective_only / unsupported
    revision_availability_time: str | None = None  # 历史修订的市场可知时点（A股=unknown）
    # ---- R24 P0/P1-XM3 §19：typed Unit + currency ----
    dimension: str | None = None               # ratio/money/money_per_share/price/shares/identifier/boolean/count
    currency: str | None = None                # CNY/USD/...（None = 无量纲或未声明）
    currency_column: str | None = None         # 动态币种所在物理列（如 US dividend currency）
    cross_market_comparable: bool = True       # 未经 FX/市场内标准化不得跨市场直接联合
    requires_fx: bool = False                  # 需要显式 FX dataset + FX PIT 才能统一
    # ---- R24 P0-XM5 §21：flow semantics（机器可读）----
    flow_semantics: str | None = None          # cumulative_ytd_flow / single_period_flow / point_in_time_stock

    @property
    def is_scale_applicable(self) -> bool:
        """该字段需要输出层单位归一化（scale 存在且 != 1.0）。"""
        return self.scale is not None and self.scale != 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "dataset": self.dataset,
            "physical_name": self.physical_name,
            "market": self.market,
            "dtype": self.dtype,
            "frequency": self.frequency,
            "grain": self.grain,
            "source_unit": self.source_unit,
            "canonical_unit": self.canonical_unit,
            "scale": self.scale,
            "time_role": self.time_role,
            "temporal_model": self.temporal_model,
            "knowledge_time": self.knowledge_time,
            "effective_time": self.effective_time,
            "period_time": self.period_time,
            "join_policy": self.join_policy,
            "required_filters": list(self.required_filters),
            "revision_order": list(self.revision_order),
            "availability": self.availability,
            "primary_key": list(self.primary_key),
            "duplicate_policy": self.duplicate_policy,
            "aliases": list(self.aliases),
            "mining_allowed": self.mining_allowed,
            "period_selection": self.period_selection,
            "period_values": list(self.period_values),
            "derived_expression": self.derived_expression,
            "derived_from": list(self.derived_from),
            "time_representation": self.time_representation,
            "time_precision": self.time_precision,
            "pit_fidelity": self.pit_fidelity,
            "revision_availability_time": self.revision_availability_time,
            "dimension": self.dimension,
            "currency": self.currency,
            "currency_column": self.currency_column,
            "cross_market_comparable": self.cross_market_comparable,
            "requires_fx": self.requires_fx,
            "flow_semantics": self.flow_semantics,
        }


@dataclass(frozen=True)
class UnitSpec:
    """R24 P0/P1-XM3 §19：typed Unit + currency。

    从 scale 升级：除了 ``scale``（单位换算乘子），还必须表达 dimension /
    currency / cross_market_comparable / requires_fx——「A股 Return bp → decimal」
    用 scale 就够，但「CNY money vs USD money」必须显式声明，未经 FX 或市场内
    标准化禁止跨市场直接联合。

    ``from_field(f)`` 从 SemanticField 编译（source_unit/canonical_unit/scale/
    currency/dimension/cross_market_comparable/requires_fx）。
    """

    dimension: str | None = None
    currency: str | None = None
    scale: float | None = None
    source_unit: str | None = None
    canonical_unit: str | None = None
    cross_market_comparable: bool = True
    requires_fx: bool = False

    @classmethod
    def from_field(cls, f: "SemanticField") -> "UnitSpec":
        return cls(
            dimension=getattr(f, "dimension", None),
            currency=getattr(f, "currency", None),
            scale=getattr(f, "scale", None),
            source_unit=getattr(f, "source_unit", None),
            canonical_unit=getattr(f, "canonical_unit", None),
            cross_market_comparable=bool(
                getattr(f, "cross_market_comparable", True)
            ),
            requires_fx=bool(getattr(f, "requires_fx", False)),
        )

    def is_money(self) -> bool:
        return self.dimension == "money" or self.currency is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "currency": self.currency,
            "scale": self.scale,
            "source_unit": self.source_unit,
            "canonical_unit": self.canonical_unit,
            "cross_market_comparable": self.cross_market_comparable,
            "requires_fx": self.requires_fx,
        }


def cross_market_compatible(
    a: "SemanticField | None",
    b: "SemanticField | None",
    *,
    a_units: "UnitSpec | None" = None,
    b_units: "UnitSpec | None" = None,
) -> tuple[bool, str]:
    """R24 P0-XM3 §19 / T-X03：跨市场字段是否允许直接统一。

    允许统一：
        - decimal return / dimensionless ratio（scale 一致且非 money）；
    禁止：
        - CNY money vs USD money（不同 currency、未声明 cross_market_comparable 且
          无 FX）——T-X03 直接 reject；
        - CNY/share vs USD/share；
        - dynamic-currency dividend（requires_fx=True）未经 FX dataset + FX PIT。

    返回 ``(compatible, reason)``；incompatible 时 reason 说明缺什么。
    """
    au = a_units or (UnitSpec.from_field(a) if a is not None else UnitSpec())
    bu = b_units or (UnitSpec.from_field(b) if b is not None else UnitSpec())
    if au.is_money() and bu.is_money():
        if au.currency and bu.currency and au.currency != bu.currency:
            if not (au.cross_market_comparable and bu.cross_market_comparable):
                return False, (
                    f"money 币种不一致（{au.currency} vs {bu.currency}）且未声明 "
                    "cross_market_comparable；需显式 FX dataset + FX PIT 或市场内标准化"
                )
            if au.requires_fx or bu.requires_fx:
                return False, "money 跨市场统一需要显式 FX dataset + FX PIT"
    elif au.is_money() != bu.is_money():
        return False, f"字段维度不一致（{au.dimension} vs {bu.dimension}）"
    return True, ""


def _tuple_of(
    value: Any,
    *,
    name: str = "字段配置",
    element_type: type = str,
) -> tuple:
    """#P0-C12 严格序列解析（与 DataRequest 共用同一套规则）。

    旧实现 fail-open：错误 mapping 类型静默返回空 tuple、list 里的非法对象还会
    str() 化——``required_filters / revision_order / primary_key / derived_from``
    配置写错会静默丢语义。现在：裸 str/bytes 拒绝；元素类型不对立即
    ValidationError；非法容器类型立即 ValidationError。
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValidationError(
            f"{name} 必须是序列（list/tuple），收到裸 {type(value).__name__} "
            f"{value!r}；请写成 [项1, 项2]"
        )
    # #7 只接受 list/tuple（ordered sequence）——``revision_order`` 的顺序本身是
    # 业务语义，set/frozenset/dict/generator 之前被 ``tuple()`` 吞掉：set 迭代序
    # 不确定、dict 退化成键列表，都会让版本排序语义漂移。
    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"{name} 必须是 list/tuple（顺序是语义），收到 {type(value).__name__} "
            f"{value!r}；set/frozenset/dict/generator 会丢顺序"
        )
    out: list[Any] = []
    for v in value:
        if not isinstance(v, element_type):
            raise ValidationError(
                f"{name} 的元素必须是 {element_type.__name__}，"
                f"收到 {type(v).__name__} {v!r}"
            )
        out.append(v)
    return tuple(out)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# #31 合法枚举集合
# R24 P0-PIT3/6：并入 temporal_join 的全部 availability（same_instant / next_bar /
# next_session_open / after_close_next_open / effective_date_only）——US date-only
# filing 需要 next_session_open，不能只允许 same_day/next_trading_day。
_VALID_AVAILABILITY = {
    "same_day",
    "next_trading_day",
    "session",
    "same_instant",
    "next_bar",
    "next_session_open",
    "after_close_next_open",
    "effective_date_only",
}
_VALID_TIME_REPRESENTATIONS = {None, "date_label", "instant"}
_VALID_TIME_PRECISIONS = {None, "date", "timestamp"}
_VALID_PIT_FIDELITIES = {
    None,
    "knowledge_date_pit",
    "vintage_pit",
    "effective_only",
    "unsupported",
}
# #7 join_policy strict enum：canonical 集合与 ``join_spec_from_field`` 支持的
# 一一对应。拼写错（``pit_asof_backword``）之前在 ``join_spec_from_field`` 里
# 静默丢成 None → exact——**丢掉 PIT join 语义**（不是崩，而是可能引入未来数据）。
_VALID_JOIN_POLICIES = frozenset({
    "exact",
    "asof",
    "pit_asof",
    "pit_asof_backward",  # 历史 alias → pit_asof
    "asof_backward",      # 历史 alias → pit_asof
    "latest_period",      # → pit_asof + period_selection=latest_period
})
_VALID_DUPLICATE_POLICIES = {"keep_first", "keep_last", "latest_revision", "error"}
_VALID_PERIOD_SELECTIONS = {
    "latest_period",
    "exact_period",
    "annual",
    "quarterly",
    "ttm",
    "all",
}
_VALID_MARKETS = {"any", "ashare", "us"}


def _strict_bool(value: Any, *, context: str, default: bool) -> bool:
    """#31 严格 bool 解析：``"false"`` 字符串 → False；非法值报错。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off", ""}:
            return False
    raise ValidationError(f"{context}: 非法布尔值 {value!r}（应为 true/false）")


def _strict_scale(value: Any, *, context: str) -> float | None:
    """#31 非法 scale 直接报错，禁止静默变 None（单位换算语义不容丢失）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            pass
    raise ValidationError(f"{context}: 非法 scale {value!r}（应为数值）")


def _strict_enum(
    value: Any,
    *,
    allowed: frozenset[str],
    context: str,
    default: str,
) -> str:
    """#31 枚举字段严格校验；非法值报错。"""
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValidationError(
            f"{context}: {text!r} 不在合法集合 {sorted(allowed)} 内"
        )
    return text


def parse_semantic_field(name: str, raw: dict[str, Any]) -> SemanticField:
    """把 YAML 单条声明解析成 SemanticField。``dataset`` / ``physical_name`` 必填。

    ``logical_name`` 可显式覆盖（用于同一逻辑字段的跨市场条目：key 用
    ``us_total_assets`` 保证 YAML key 唯一，logical_name 声明为 ``total_assets``
    让 catalog 的 _by_name 同时挂 A股/美股两个候选）。
    """
    context = f"semantic field '{name}'"
    dataset = raw.get("dataset")
    physical = raw.get("physical_name") or raw.get("physical") or name
    logical = str(raw.get("logical_name") or name)
    derived_expr = _str_or_none(raw.get("derived_expression"))
    # #P0-24 derived 字段不落在单一物理表：允许无 dataset，但必须有 derived_expression。
    if not dataset and not derived_expr:
        raise ValidationError(f"{context}: 缺少 dataset（逻辑字段必须落到某个数据集）")
    # #P1-final closure 7 + R39 #67：DerivedFieldCompiler 已实现执行链。derived
    # 字段允许 mining_allowed=true（compile 成功即真正可执行）；表达式编译失败时
    # 由 plan() 阶段抛 ``UnsupportedFeatureError``，catalog load 不再一刀切拒绝。
    # 未显式声明 mining_allowed 时保持 fail-closed（默认 False）。
    if derived_expr and raw.get("mining_allowed") is None:
        raw = dict(raw)
        raw["mining_allowed"] = False
    temporal_model = _str_or_none(raw.get("temporal_model"))
    # #9：财务事件字段默认 latest_period——「最新可见会计期 + 该期内最新修订」。
    # 未显式写 period_selection 时，financial_event/event 字段默认 latest_period，
    # 除非用户明确要 exact_period/annual/quarterly/all。
    period_selection = _str_or_none(raw.get("period_selection"))
    if period_selection is None:
        period_selection = (
            "latest_period"
            if temporal_model in {"financial_event", "event", "financial", "e2"}
            else "all"
        )
    # #31 strict schema：未知 key 拒绝、bool/enum/scale 严格校验。
    _KNOWN_KEYS = {
        "dataset",
        "physical_name",
        "physical",
        "logical_name",
        "market",
        "dtype",
        "frequency",
        "grain",
        "source_unit",
        "canonical_unit",
        "scale",
        "time_role",
        "temporal_model",
        "knowledge_time",
        "effective_time",
        "period_time",
        "join_policy",
        "required_filters",
        "revision_order",
        "availability",
        "availability_latency",
        "primary_key",
        "duplicate_policy",
        "aliases",
        "mining_allowed",
        "period_selection",
        "period_values",
        "derived_expression",
        "derived_from",
        "time_representation",
        "time_precision",
        "pit_fidelity",
        "revision_availability_time",
        "dimension",
        "currency",
        "currency_column",
        "cross_market_comparable",
        "requires_fx",
        "flow_semantics",
    }
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        raise ValidationError(
            f"{context}: 未知配置 key {unknown}（应为 {sorted(_KNOWN_KEYS)} 之一）"
        )
    latency_raw = raw.get("availability_latency")
    if latency_raw is not None:
        if isinstance(latency_raw, bool) or not isinstance(latency_raw, int):
            raise ValidationError(
                f"{context}: availability_latency 必须是非负整数（bar 数），"
                f"收到 {latency_raw!r}"
            )
        if latency_raw < 0:
            raise ValidationError(
                f"{context}: availability_latency 必须是非负整数（bar 数）"
            )
    market = _str_or_none(raw.get("market")) or "any"
    if market not in _VALID_MARKETS:
        raise ValidationError(f"{context}: market={market!r} 非法（应为 any/ashare/us）")
    availability = _strict_enum(
        raw.get("availability"),
        allowed=_VALID_AVAILABILITY,
        context=context,
        default="same_day",
    )
    duplicate_policy = _strict_enum(
        raw.get("duplicate_policy"),
        allowed=_VALID_DUPLICATE_POLICIES,
        context=context,
        default="latest_revision",
    )
    period_selection = _strict_enum(
        period_selection,
        allowed=_VALID_PERIOD_SELECTIONS,
        context=context,
        default="all",
    )
    # #7 join_policy 严格枚举：拼写错 fail-closed，绝不在执行链静默退化 exact。
    join_policy = _str_or_none(raw.get("join_policy"))
    if join_policy is not None:
        jp = str(join_policy).strip().lower()
        if jp not in _VALID_JOIN_POLICIES:
            raise ValidationError(
                f"{context}: join_policy={join_policy!r} 不在合法集合 "
                f"{sorted(_VALID_JOIN_POLICIES)} 内"
            )
        join_policy = jp
    # R24 P0-PIT2 §11：time_representation / time_precision / pit_fidelity 严格枚举。
    time_representation = _str_or_none(raw.get("time_representation"))
    if time_representation not in _VALID_TIME_REPRESENTATIONS:
        raise ValidationError(
            f"{context}: time_representation={time_representation!r} 非法"
            "（应为 date_label / instant）"
        )
    time_precision = _str_or_none(raw.get("time_precision"))
    if time_precision not in _VALID_TIME_PRECISIONS:
        raise ValidationError(
            f"{context}: time_precision={time_precision!r} 非法（应为 date / timestamp）"
        )
    pit_fidelity = _str_or_none(raw.get("pit_fidelity"))
    if pit_fidelity not in _VALID_PIT_FIDELITIES:
        raise ValidationError(
            f"{context}: pit_fidelity={pit_fidelity!r} 非法"
            "（应为 knowledge_date_pit / vintage_pit / effective_only / unsupported）"
        )
    return SemanticField(
        logical_name=logical,
        dataset=str(dataset) if dataset else None,
        physical_name=str(physical),
        market=market,
        dtype=_str_or_none(raw.get("dtype")),
        frequency=_str_or_none(raw.get("frequency")),
        grain=_str_or_none(raw.get("grain")),
        source_unit=_str_or_none(raw.get("source_unit")),
        canonical_unit=_str_or_none(raw.get("canonical_unit")),
        scale=_strict_scale(raw.get("scale"), context=context),
        time_role=_str_or_none(raw.get("time_role")),
        temporal_model=temporal_model,
        knowledge_time=_str_or_none(raw.get("knowledge_time")),
        effective_time=_str_or_none(raw.get("effective_time")),
        period_time=_str_or_none(raw.get("period_time")),
        join_policy=join_policy,
        required_filters=_tuple_of(
            raw.get("required_filters"), name=f"{logical}.required_filters"
        ),
        revision_order=_tuple_of(
            raw.get("revision_order"), name=f"{logical}.revision_order"
        ),
        availability=availability,
        availability_latency=latency_raw if latency_raw is not None else None,
        primary_key=_tuple_of(
            raw.get("primary_key"), name=f"{logical}.primary_key"
        ),
        duplicate_policy=duplicate_policy,
        aliases=_tuple_of(raw.get("aliases"), name=f"{logical}.aliases"),
        mining_allowed=_strict_bool(
            raw.get("mining_allowed"), context=context, default=True
        ),
        period_selection=period_selection,
        period_values=(
            _tuple_of(
                raw.get("period_values"),
                name=f"{logical}.period_values",
                element_type=object,
            )
            if raw.get("period_values")
            else ()
        ),
        derived_expression=_str_or_none(raw.get("derived_expression")),
        derived_from=_tuple_of(
            raw.get("derived_from"), name=f"{logical}.derived_from"
        ),
        time_representation=time_representation,
        time_precision=time_precision,
        pit_fidelity=pit_fidelity,
        revision_availability_time=_str_or_none(
            raw.get("revision_availability_time")
        ),
        dimension=_str_or_none(raw.get("dimension")),
        currency=_str_or_none(raw.get("currency")),
        currency_column=_str_or_none(raw.get("currency_column")),
        cross_market_comparable=_strict_bool(
            raw.get("cross_market_comparable"), context=context, default=True
        ),
        requires_fx=_strict_bool(
            raw.get("requires_fx"), context=context, default=False
        ),
        flow_semantics=_str_or_none(raw.get("flow_semantics")),
    )


class SemanticFieldCatalog:
    """逻辑字段 → 物理字段的权威映射表。"""

    def __init__(
        self,
        fields: Mapping[str, SemanticField] | None = None,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        self._fields: dict[str, SemanticField] = dict(fields or {})
        # _by_name：logical_name / YAML key / alias → 候选字段列表。同一逻辑名可被
        # 多个市场占用（如 A股 total_assets 与美股 us_total_assets 的 logical_name
        # 都是 total_assets），resolve 时按 market 消歧。any 表示跨市场通用。
        self._by_name: dict[str, list[SemanticField]] = {}
        for name, f in self._fields.items():
            for key in {name, f.logical_name}:
                self._by_name.setdefault(key, []).append(f)
            for alias in f.aliases:
                self._by_name.setdefault(alias, []).append(f)
        self._source_path = str(source_path) if source_path is not None else None

    @property
    def source_path(self) -> str | None:
        return self._source_path

    def names(self) -> list[str]:
        return sorted(self._fields)

    @staticmethod
    def _market_of(dataset: str | None) -> str | None:
        """由数据集名推断市场（us_*/ashare_*）；无法判断返回 None。"""
        if not dataset:
            return None
        if dataset.startswith("us_") or dataset.startswith("us_stock") or dataset == "us":
            return "us"
        if dataset.startswith("ashare_") or dataset.startswith("a_share") or dataset == "ashare":
            return "ashare"
        return None

    def resolve_one(
        self, name: str, market: str | None = None, *, dataset: str | None = None
    ) -> SemanticField | None:
        """按逻辑名或别名解析；找不到返回 None（调用方可再回退 registry）。

        market 传入时，优先返回该市场专属字段；没有专属字段时回退 any（跨市场
        通用）。dataset 传入时用数据集名前缀推断 market（更高优先级）。

        **#54 fail-closed**：无 market/dataset 上下文且存在多个非 any 候选时，
        production 抛 ``AmbiguousSemanticFieldError``（禁止 YAML 顺序决定市场）；
        research 告警后取第一个。
        """
        candidates = self._by_name.get(name)
        if not candidates:
            return None
        effective = market or (self._market_of(dataset) if dataset else None)
        if effective and effective != "any":
            for f in candidates:
                if f.market == effective:
                    return f
        # 回退：any 通用字段
        for f in candidates:
            if f.market == "any":
                return f
        # 无上下文且多市场候选 → fail-closed
        ambiguous = [f for f in candidates if f.market != "any"]
        if len(ambiguous) > 1:
            markets = sorted({f.market for f in ambiguous})
            from data_access.core.exceptions import AmbiguousSemanticFieldError
            from data_access.read.query_budget import is_strict_semantics

            msg = (
                f"逻辑字段 '{name}' 跨市场歧义（候选市场: {markets}）。"
                "请显式传 market='ashare'/'us' 或 dataset= 让 catalog 消歧。"
            )
            # #16 strict_read 完全共享 production 的 fail-closed：不能只查
            # _production_mode()（DATA_ACCESS_STRICT_READ=1 也要拦）。
            if is_strict_semantics():
                raise AmbiguousSemanticFieldError(msg)
            logger.warning("%s（research 放行，取 YAML 顺序第一个）", msg)
        return candidates[0]

    def resolve_by_physical(self, dataset: str, physical_name: str) -> SemanticField | None:
        """按 (dataset, 物理列名) 反查逻辑字段（调用方传物理列时的单位归一化用）。

        同一物理列可能被多个逻辑字段引用；优先返回与该数据集同市场的字段
        （防止 A股/美股同名物理列串味）。

        #P1-final closure 8 fail-closed：市场过滤后仍是**多个不同语义身份**
        （不同 logical_name 或不同单位语义）时，production/strict 抛
        ``AmbiguousSemanticFieldError``——单位归一化（``normalize_table_units``）
        必须知道这列到底是哪个字段，YAML 顺序不能当决定因素；research 告警后
        取第一个（确定性）。
        """
        candidates = [
            f
            for f in self._fields.values()
            if f.dataset == dataset and f.physical_name == physical_name
        ]
        if not candidates:
            return None
        effective = self._market_of(dataset)
        pool = list(candidates)
        if effective and effective != "any":
            m = [f for f in candidates if f.market == effective]
            if m:
                pool = m
        else:
            a = [f for f in candidates if f.market == "any"]
            if a:
                pool = a
        # 同一物理列多个不同语义身份（不同 logical_name / 不同单位）→ fail-closed
        distinct: dict[tuple, SemanticField] = {}
        for f in pool:
            key = (
                f.logical_name,
                f.market,
                f.scale,
                f.source_unit,
                f.canonical_unit,
            )
            distinct.setdefault(key, f)
        if len(distinct) > 1:
            identities = sorted({k[0] for k in distinct})
            from data_access.core.exceptions import AmbiguousSemanticFieldError
            from data_access.read.query_budget import is_strict_semantics

            msg = (
                f"(dataset={dataset}, physical={physical_name}) 对应多个语义身份"
                f"（{identities}），无法确定该列属于哪个逻辑字段/单位。"
                "请改用逻辑字段名解析，或消除 catalog 里的同名物理列歧义。"
            )
            if is_strict_semantics():
                raise AmbiguousSemanticFieldError(msg)
            logger.warning("%s（research 取第一个）", msg)
        return pool[0]

    def get(self, name: str, market: str | None = None, *, dataset: str | None = None) -> SemanticField:
        """严格解析；找不到抛 ValidationError。"""
        f = self.resolve_one(name, market=market, dataset=dataset)
        if f is None:
            raise ValidationError(
                f"逻辑字段 '{name}' 不在 SemanticFieldCatalog 中。"
                f"可用: {self.names()}"
            )
        return f

    def resolve(
        self, *names: str, market: str | None = None, dataset: str | None = None
    ) -> list[SemanticField]:
        out: list[SemanticField] = []
        for name in names:
            out.append(self.get(name, market=market, dataset=dataset))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {name: f.to_dict() for name, f in self._fields.items()}

    def fingerprint(self) -> str:
        """语义 catalog 稳定指纹（#3：缓存 key 必须覆盖 catalog 语义版本）。"""
        import hashlib
        import json

        payload = json.dumps(
            {k: f.to_dict() for k, f in self._fields.items()},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def semantic_coverage_audit(self, registry: Any) -> dict[str, Any]:
        """#30 语义覆盖审计：registry 物理字段 vs catalog。

        每个 registry 物理列 → FULL_SEMANTIC / PHYSICAL_ONLY / AMBIGUOUS；
        catalog 引用了 registry 里不存在的 dataset/column → BLOCKED。

        物理列可以读，但进入 PIT/join/因子挖掘前必须有完整 semantic contract——
        本审计是 coverage 的看板，供 CI 对齐。
        """
        from data_access.core.exceptions import ValidationError

        by_phys: dict[tuple[str, str], list[str]] = {}
        for f in self._fields.values():
            by_phys.setdefault((f.dataset, f.physical_name), []).append(f.logical_name)

        fields_report: dict[str, dict[str, Any]] = {}
        for name in registry.names():
            ds = registry.get(name)
            schema = getattr(ds, "schema", None) or {}
            for col in sorted(schema):
                refs = by_phys.get((name, col), [])
                status = "FULL_SEMANTIC"
                if len(refs) > 1:
                    status = "AMBIGUOUS"
                elif not refs:
                    status = "PHYSICAL_ONLY"
                fields_report[f"{name}.{col}"] = {
                    "status": status,
                    "logical": sorted(refs),
                }

        blocked: dict[str, str] = {}
        for f in self._fields.values():
            if f.derived_expression:
                # #P0-24 derived 字段不落盘，不做物理列存在性检查。
                continue
            try:
                ds = registry.get(f.dataset)
            except ValidationError:
                blocked[f"{f.logical_name} -> {f.dataset}.{f.physical_name}"] = (
                    "DATASET_MISSING"
                )
                continue
            schema = getattr(ds, "schema", None) or {}
            if f.physical_name not in schema:
                blocked[f"{f.logical_name} -> {f.dataset}.{f.physical_name}"] = (
                    "COLUMN_MISSING"
                )

        counts = {"FULL_SEMANTIC": 0, "PHYSICAL_ONLY": 0, "AMBIGUOUS": 0}
        for r in fields_report.values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return {
            "fields": fields_report,
            "blocked": blocked,
            "counts": counts,
            "total_physical_columns": len(fields_report),
            "blocked_count": len(blocked),
        }

    @classmethod
    def from_yaml(
        cls, path: str | Path | None = None
    ) -> "SemanticFieldCatalog":
        """从 YAML 加载。path 缺省用 ``data_access/config/semantic_fields.yaml``。

        YAML 顶层是 mapping（逻辑字段名 → 声明），``_`` 前缀的键当作锚点模板跳过。
        """
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent
                / "config"
                / "semantic_fields.yaml"
            )
        path = Path(path)
        if not path.exists():
            raise ValidationError(f"semantic_fields.yaml 不存在：{path}")
        if yaml is None:
            raise ValidationError("缺少 PyYAML 依赖，无法加载 SemanticFieldCatalog")
        with path.open("r", encoding="utf-8") as fh:
            from data_access.registry.yaml_loader import strict_yaml_load

            raw = strict_yaml_load(fh.read(), context=str(path)) or {}
        if not isinstance(raw, dict):
            raise ValidationError(
                f"{path}: 顶层必须是 mapping（字段名 → 声明）"
            )
        fields: dict[str, SemanticField] = {}
        for name, body in raw.items():
            if name.startswith("_"):
                continue  # YAML 锚点模板键
            if body is None:
                body = {}
            if not isinstance(body, dict):
                raise ValidationError(
                    f"{path}: 字段 '{name}' 的声明必须是 mapping"
                )
            fields[name] = parse_semantic_field(name, body)
        return cls(fields, source_path=path)


# ---- 进程内缓存（惰性加载） ----
#
# #P1-79 按**解析后的路径**缓存：默认路径与 env 覆盖路径（DATA_ACCESS_
# SEMANTIC_FIELDS）各自独立 cache，同一进程不同调用不会看到不同文件状态，
# 也不会在不同路径间串。

_catalog: SemanticFieldCatalog | None = None
_catalog_cache: dict[str, SemanticFieldCatalog] = {}
_catalog_lock = threading.Lock()


def get_semantic_catalog(
    path: str | Path | None = None,
) -> SemanticFieldCatalog:
    """进程级 SemanticFieldCatalog 单例。

    默认读 ``config/semantic_fields.yaml``；可用环境变量 ``DATA_ACCESS_SEMANTIC_FIELDS``
    覆盖路径（部署/测试用）。

    #P1-79 默认路径（env 未设置）走全局单例；env/显式覆盖路径按解析后路径缓存——
    长期服务里 override 路径不会反复 load YAML，也不会在同一进程看到文件状态漂移。
    """
    global _catalog
    if path is None:
        env_path = os.environ.get("DATA_ACCESS_SEMANTIC_FIELDS")
        if env_path and env_path.strip():
            path = Path(env_path.strip())
    if path is None:
        if _catalog is not None:
            return _catalog
        with _catalog_lock:
            if _catalog is not None:
                return _catalog
            _catalog = SemanticFieldCatalog.from_yaml(None)
            return _catalog
    # 显式/环境覆盖路径：按解析后路径缓存（冻结，避免文件中途变化）
    key = str(Path(path).resolve())
    cached = _catalog_cache.get(key)
    if cached is not None:
        return cached
    with _catalog_lock:
        cached = _catalog_cache.get(key)
        if cached is not None:
            return cached
        built = SemanticFieldCatalog.from_yaml(path)
        _catalog_cache[key] = built
        return built


def reset_semantic_catalog() -> None:
    """清空缓存（测试用）。"""
    global _catalog, _catalog_cache
    _catalog = None
    _catalog_cache = {}


def normalize_table_units(
    table: Any,
    fields: Sequence[SemanticField],
    *,
    column_of: Sequence[SemanticField] | None = None,
) -> Any:
    """输出层单位归一化：把需要 scale 的字段乘上乘子，返回新 Arrow Table。

    只对浮点（含整型）数值列生效；找不到列的字段静默跳过。``column_of``
    预留：当输出列名与 physical_name 不同时，可传「该列对应哪个 SemanticField」，
    按 logical_name 匹配列名（read_joined 的多表输出就是 physical_name）。

    #P1-final closure 8：按**列位置**重建输出（``Table.from_arrays`` + 名字列表），
    不再用 ``dict[column_name] = arr``——合法重复列名（组合读里同名物理列）会被
    dict 覆盖丢列。重复名 + 重复列都保留。
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if not fields:
        return table
    needs = [f for f in fields if f.is_scale_applicable]
    if not needs:
        return table
    target_names = {f.physical_name for f in needs}
    # ``column_of``（按列位置给字段身份）优先；否则按列名匹配。重复列名时
    # column_of 才能区分哪一列对应哪个字段。
    column_of_list = list(column_of or ())
    out_names: list[str] = []
    out_arrays: list[Any] = []
    for i, col in enumerate(table.column_names):
        arr = table.column(i)
        col_field = column_of_list[i] if i < len(column_of_list) else None
        matches: list[SemanticField] = []
        if col_field is not None:
            if col_field.is_scale_applicable:
                matches = [col_field]
        elif col in target_names:
            matches = [f for f in needs if f.physical_name == col]
        for f in matches:
            if pa.types.is_floating(arr.type):
                arr = pc.multiply(arr, pa.scalar(float(f.scale)))
            elif pa.types.is_integer(arr.type):
                arr = pc.multiply(arr.cast(pa.float64()), pa.scalar(float(f.scale)))
            break
        out_names.append(col)
        out_arrays.append(arr)
    return pa.Table.from_arrays(out_arrays, names=out_names)
