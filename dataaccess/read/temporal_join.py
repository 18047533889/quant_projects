"""
data_access.read.temporal_join —— 语义级时间 join 规格（TemporalJoinSpec）

背景
    read_joined 早期的 ``pit_asof`` 只是普通 ``anchor.time >= right.time`` ASOF，
    没有区分「数据何时可见」与「数据属于哪个会计期间」。本模块把 join 语义
    显式化：decision_time（决策时点 = anchor 时间）、knowledge_time（右表
    可见时点，如财务 PubDate）、period_time（会计期间，如 ReportPeriodEndDate）、
    revision_order（同一可见时点的多版本排序，决定取哪一版）、availability
    （数据何时可用：same_day 或 next_trading_day）。

用法
    ``store.read_joined(..., joins={"ashare_stock_balance": TemporalJoinSpec(
        policy="pit_asof",
        knowledge_time="PubDate",
        period_time="ReportPeriodEndDate",
        revision_order=("PubDate", "UpdateTime"),
        availability="next_trading_day",
    )})``

    joins 值也接受字符串（``"pit_asof"``，语义即旧版普通 ASOF）或 dict。

设计要点
    1. ``availability="next_trading_day"`` 时 ASOF 条件用严格大于
       ``decision_time > knowledge_time``（A 股财报盘后落地：PubDate 当天
       的 bar 不能用，下一交易日才可用；对日频面板等价于严格大于）。
    2. ``revision_order`` 提供后，join 前先按 ``(instrument, knowledge_time)``
       去重、保留 revision_order 降序最新一行（替代依赖 parquet 扫描顺序的
       keep_last），保证确定性。
    3. ``primary_key`` / ``duplicate_policy`` 是 exact join 的唯一性契约声明；
       违反（join 后放大行数）会在执行层审计。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from data_access.core.exceptions import ValidationError

_VALID_POLICIES = frozenset({"exact", "asof", "pit_asof"})
# #P0-7 细粒度 availability：命名如实反映语义，不再把一切归到 session。
#   same_instant          —— knowledge 时刻即可用（decision >= knowledge 同刻）
#   same_day              —— 当日可见（向后看，>=）
#   next_bar              —— 下一根 bar 起可用（日频近似 next_trading_day）
#   next_session_open     —— 下一交易时段开盘起可用（交易日历感知）
#   next_trading_day      —— 下一交易日起可用（交易日历感知）
#   after_close_next_open —— 盘后披露，下一开盘起可用（交易日历感知）
#   effective_date_only   —— 仅生效日语义（>=，且 future_cutoff 生效）
#   session               —— 兼容旧名：按 session 边界映射（同 next_session_open）
_VALID_AVAILABILITY = frozenset({
    "same_instant", "same_day", "next_bar", "next_session_open",
    "next_trading_day", "after_close_next_open", "effective_date_only", "session",
})
# #P0-7 需要交易日历把 knowledge 映射成 available_from 的 availability 种类。
_CALENDAR_AVAILABILITIES = frozenset({
    "next_bar", "next_session_open", "next_trading_day",
    "after_close_next_open", "session",
})
# #P0-7 语义上等价于「严格下一交易日」的种类（比较符 / 日历映射用）。
_STRICT_NEXT_KINDS = frozenset({
    "next_bar", "next_session_open", "next_trading_day", "after_close_next_open",
})
_VALID_DUPLICATE_POLICIES = frozenset({"keep_first", "keep_last", "latest_revision", "error"})
_VALID_PERIOD_SELECTIONS = frozenset(
    {"latest_period", "exact_period", "annual", "quarterly", "ttm", "all"}
)


def availability_uses_calendar(availability: str | None) -> bool:
    """#P0-7 统一 availability 语义：该种类是否需交易日历把 knowledge 映射成
    ``available_from``（next_bar / next_session_open / next_trading_day /
    after_close_next_open / session）。same_instant/same_day/effective_date_only
    直接可见，无需日历。"""
    return (availability or "same_day") in _CALENDAR_AVAILABILITIES


def availability_strict_next(availability: str | None) -> bool:
    """#P0-7 统一 availability 语义：该种类是否等价「严格下一交易日」——
    ASOF 无日历回退时用严格 ``>``（可见 iff knowledge < decision）。"""
    return (availability or "same_day") in _STRICT_NEXT_KINDS


@dataclass(frozen=True)
class TemporalJoinSpec:
    """一次时间 join 的完整语义规格。

    参数:
        policy: exact / asof / pit_asof
        decision_time: 锚点决策时点列（缺省 = 锚点数据集 time_column）
        knowledge_time: 右表数据可见时点列（缺省 = 右表 time_column）
        period_time: 右表会计期间/生效期列（可选，仅标注）
        revision_order: 版本排序列（如 (\"PubDate\", \"UpdateTime\")），
            join 前按 (instrument, knowledge_time) 去重取最新一版
        availability: same_day / next_trading_day
        deduplicate: 是否做 revision 去重（True 且 revision_order 非空时生效）
        primary_key: 唯一性契约（如 (\"TradeDate\", \"Symbol\")）
        duplicate_policy: keep_first/keep_last/latest_revision/error
    """

    policy: str = "exact"
    decision_time: str | None = None
    knowledge_time: str | None = None
    period_time: str | None = None
    revision_order: tuple[str, ...] = ()
    availability: str = "same_day"
    deduplicate: bool = True
    primary_key: tuple[str, ...] = ()
    duplicate_policy: str = "latest_revision"
    # ---- #45 财务 PIT 报告期选择 ----
    period_selection: str = "all"          # latest_period/exact_period/annual/quarterly/ttm/all
    period_values: tuple[Any, ...] = ()    # exact_period 的目标 period 值列表
    # ---- #16 事件未来数据 cutoff ----
    future_cutoff: bool = True             # effective_time_only 事件右表不读未来生效事件
    # ---- #P0-7 availability_latency：额外可见性延迟（如分钟/bar 数）----
    availability_latency: int | None = None

    @property
    def is_asof(self) -> bool:
        return self.policy in {"asof", "pit_asof"}

    @property
    def is_calendar_availability(self) -> bool:
        """是否需要交易日历把 knowledge 映射成 available_from（#P0-7）。"""
        return self.availability in _CALENDAR_AVAILABILITIES

    @property
    def comparison_operator(self) -> str:
        """ASOF 条件比较符：严格下一交易日/时段类用严格大于，其余向后看 >=。"""
        if not self.is_asof:
            raise ValueError("exact join 不使用比较操作符")
        return ">" if self.availability in _STRICT_NEXT_KINDS else ">="

    @property
    def needs_period_selection(self) -> bool:
        """是否需要报告期选择（latest_period 等）。"""
        return bool(self.period_time) and self.period_selection not in {"", "all"}

    def effective_decision_time(self, anchor_time_column: str | None) -> str:
        return self.decision_time or anchor_time_column or ""

    def effective_knowledge_time(self, right_time_column: str | None) -> str:
        return self.knowledge_time or right_time_column or ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "decision_time": self.decision_time,
            "knowledge_time": self.knowledge_time,
            "period_time": self.period_time,
            "revision_order": list(self.revision_order),
            "availability": self.availability,
            "deduplicate": self.deduplicate,
            "primary_key": list(self.primary_key),
            "duplicate_policy": self.duplicate_policy,
            "period_selection": self.period_selection,
            "period_values": list(self.period_values),
            "future_cutoff": self.future_cutoff,
            "availability_latency": self.availability_latency,
        }


def _tuple_of(value: Any) -> tuple[str, ...]:
    """把列名序列归一化成 str tuple；非法类型 fail-closed（#34）。

    原来 int/float/bool/dict 等非法输入会静默变成空 tuple——revision_order=2026
    直接丢掉语义还不报错。现在统一抛 ValidationError，与 Semantic YAML 的
    严格解析方向一致。
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            if not isinstance(v, str):
                raise ValidationError(
                    f"列名序列元素必须是字符串，收到 {type(v).__name__}: {v!r}"
                )
            if v.strip():
                out.append(v)
        return tuple(out)
    raise ValidationError(
        f"列名序列必须是字符串/列表/元组，收到 {type(value).__name__}: {value!r}"
    )


def _as_bool(value: Any, default: bool) -> bool:
    """YAML/JSON bool 解析：字符串 "false"/"0"/"no"/"off" 正确 → False（#33）。

    旧代码 ``bool(raw.get("deduplicate", True))`` 会把配置字符串 ``"false"``
    当成 True——TemporalJoinSpec 的语义和配置脱节。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "off", "n", "f"}:
        return False
    if text in {"true", "1", "yes", "on", "y", "t"}:
        return True
    raise ValidationError(f"bool 值无法识别: {value!r}（合法 true/false/1/0/yes/no）")


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _period_values_of(value: Any) -> tuple[Any, ...]:
    """exact_period 的目标 period 值（保留原始类型，日期不要 stringify）。"""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def normalize_join_policy(policy: Any) -> str:
    """把 join 策略归一化到合法集合；None 默认 exact。"""
    if isinstance(policy, TemporalJoinSpec):
        policy = policy.policy
    key = str(policy or "exact").strip().lower()
    if key not in _VALID_POLICIES:
        raise ValidationError(
            f"join 策略必须是 {sorted(_VALID_POLICIES)}，收到 {policy!r}"
        )
    return key


def join_spec_from_field(field: Any) -> TemporalJoinSpec | None:
    """把 SemanticField 的 join 语义（join_policy / knowledge_time / period_time /
    revision_order / availability）翻译成 TemporalJoinSpec。

    返回 None 表示字段未声明 join 语义（调用方用默认 exact）。
    """
    if field is None:
        return None
    policy = str(getattr(field, "join_policy", None) or "exact")
    availability = str(getattr(field, "availability", "same_day") or "same_day")
    revision = tuple(getattr(field, "revision_order", ()) or ())
    knowledge = getattr(field, "knowledge_time", None)
    period = getattr(field, "period_time", None)
    if policy in {"pit_asof_backward", "pit_asof", "asof_backward", "asof", "latest_period"}:
        period_selection = str(getattr(field, "period_selection", "all") or "all")
        if policy == "latest_period" and period_selection == "all":
            period_selection = "latest_period"
        return TemporalJoinSpec(
            policy="pit_asof",
            knowledge_time=knowledge or None,
            period_time=period or None,
            revision_order=revision,
            availability=availability,
            period_selection=period_selection,
            period_values=tuple(getattr(field, "period_values", ()) or ()),
        )
    if policy == "exact" and (revision or getattr(field, "primary_key", ())):
        return TemporalJoinSpec(
            policy="exact",
            revision_order=revision,
            primary_key=tuple(getattr(field, "primary_key", ()) or ()),
            duplicate_policy=str(getattr(field, "duplicate_policy", "latest_revision") or "latest_revision"),
        )
    return None


def parse_join_spec(raw: Any) -> TemporalJoinSpec:
    """把 read_joined 的 joins 值归一化成 TemporalJoinSpec。

    支持三种输入：
        - ``None`` / 缺失 → exact
        - 字符串 ``"pit_asof"`` / ``"exact"`` / ``"asof"``
        - dict：``{"policy": "pit_asof", "knowledge_time": "PubDate", ...}``
        - TemporalJoinSpec 实例（原样返回）
    """
    if isinstance(raw, TemporalJoinSpec):
        return raw
    if raw is None:
        return TemporalJoinSpec(policy="exact")
    if isinstance(raw, str):
        return TemporalJoinSpec(policy=normalize_join_policy(raw))
    if not isinstance(raw, Mapping):
        raise ValidationError(
            f"join 规格必须是字符串/dict/TemporalJoinSpec，收到 {type(raw).__name__}"
        )
    # dict 没显式 policy 但表达了语义时间（knowledge_time/availability/revision）
    # → 默认 pit_asof（asof 语义）；否则默认 exact。
    has_semantic = any(
        raw.get(k) is not None for k in ("knowledge_time", "availability", "revision_order", "period_time")
    )
    policy = normalize_join_policy(
        raw.get("policy") if "policy" in raw else ("pit_asof" if has_semantic else "exact")
    )
    availability = _str_or_none(raw.get("availability")) or "same_day"
    if availability not in _VALID_AVAILABILITY:
        raise ValidationError(
            f"availability 必须是 {sorted(_VALID_AVAILABILITY)}，收到 {availability!r}"
        )
    dup = _str_or_none(raw.get("duplicate_policy")) or "latest_revision"
    if dup not in _VALID_DUPLICATE_POLICIES:
        raise ValidationError(
            f"duplicate_policy 必须是 {sorted(_VALID_DUPLICATE_POLICIES)}，收到 {dup!r}"
        )
    period_selection = _str_or_none(raw.get("period_selection")) or "all"
    if period_selection not in _VALID_PERIOD_SELECTIONS:
        raise ValidationError(
            f"period_selection 必须是 {sorted(_VALID_PERIOD_SELECTIONS)}，"
            f"收到 {period_selection!r}"
        )
    latency = raw.get("availability_latency")
    if latency is not None:
        try:
            latency = int(latency)
        except (ValueError, TypeError):
            raise ValidationError(
                f"availability_latency 必须是整数，收到 {latency!r}"
            )
    return TemporalJoinSpec(
        policy=policy,
        decision_time=_str_or_none(raw.get("decision_time")),
        knowledge_time=_str_or_none(raw.get("knowledge_time")),
        period_time=_str_or_none(raw.get("period_time")),
        revision_order=_tuple_of(raw.get("revision_order")),
        availability=availability,
        # #33 字符串 "false"/"0" 必须解析成 False，不能 bool("false") == True
        deduplicate=_as_bool(raw.get("deduplicate"), True),
        primary_key=_tuple_of(raw.get("primary_key")),
        duplicate_policy=dup,
        period_selection=period_selection,
        period_values=_period_values_of(raw.get("period_values")),
        future_cutoff=_as_bool(raw.get("future_cutoff"), True),
        availability_latency=latency,
    )
