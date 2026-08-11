"""data_access.r30.specs —— R30-P1-003..010：频率 / 粒度 / 市场 / 价格基准 / 工具身份
/ 修订保真 / 语义列版本 / 聚合配方身份。

R30 全维度成熟度层是 **additive** 的：本模块全部是一等对象，不修改任何既有文件。
它们把「原始数据自身不改、只升级 semantic metadata / compiler」落到实处：

    - ``FrequencySpec``：只描述**频率**（tick/event/minute/daily/weekly/snapshot），
      source→target 转换必须另有 ``AggregationSpec``（``requires_aggregation`` 仅
      探测是否必要，不隐含执行聚合）。
    - ``GrainSpec``：描述「一行代表什么」的粒度键，提供 join 基数推断。
    - ``MarketProfile``：市场级**稳定规则**（时区/默认币种/日历/交易时段/结算），
      不替代 Dataset Contract；``resolve`` 预设 ASHARE/US。
    - ``PriceBasisSpec``：价格调整基准枚举（RAW / FORWARD / BACKWARD / TOTAL_RETURN
      / PIT_ADJUSTED）。
    - ``InstrumentIdentity``：工具身份——``display_symbol`` 与 ``stable_id`` 分离
      （美股 ticker rename 用 stable security_id 保持身份）。
    - ``RevisionFidelity``：修订可见性保真；从既有 ``pit_fidelity`` 字符串映射。
    - ``SemanticColumnVersion``：同一物理列的 semantic epoch（如 % → decimal），
      不被普通 schema union 掩盖。
    - ``AggregationRecipeIdentity``：聚合配方的稳定身份——两个都叫 VWAP 但计算
      时段/停牌处理不同 → identity 不同（用 r30._shared.stable_digest 折叠）。

维护人：quant 基础平台组    最后更新：2026-08-11
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any

from data_access.r30._shared import stable_digest
from data_access.r30.concepts import UnitError


# ---------------------------------------------------------------------------
# FrequencySpec（R30-P1-003）
# ---------------------------------------------------------------------------

_FREQUENCY_UNITS = frozenset({"tick", "event", "minute", "daily", "weekly", "snapshot"})
# 粗粒度排序：数值越大越粗；同 unit 时 multiplier 越大越粗。
_FREQUENCY_RANK = {
    "tick": 0,
    "event": 1,
    "minute": 2,
    "daily": 3,
    "weekly": 4,
    "snapshot": 5,
}


@dataclass(frozen=True)
class FrequencySpec:
    """频率描述（一等对象）。

    只描述「数据天然频率」，不做任何聚合。``unit`` ∈
    {tick, event, minute, daily, weekly, snapshot}；``multiplier`` 表示每根 bar
    的周期数（minute=5 → 5 分钟 bar；daily=1 → 日线）。

    语义契约：**source→target 转换必须有 AggregationSpec**，``requires_aggregation``
    只探测「target 比 source 更粗 → 必要」，不隐含任何执行。
    """

    unit: str = "minute"
    multiplier: int = 1
    session: str = "regular"
    timezone: str | None = None

    def __post_init__(self) -> None:
        if self.unit not in _FREQUENCY_UNITS:
            raise UnitError(
                f"非法频率单位 {self.unit!r}（应为 {sorted(_FREQUENCY_UNITS)}）"
            )
        if (
            not isinstance(self.multiplier, int)
            or isinstance(self.multiplier, bool)
            or self.multiplier < 1
        ):
            raise UnitError(f"multiplier 必须为正整数，收到 {self.multiplier!r}")

    # ---- 输出 / 解析 ----

    def label(self) -> str:
        """人类/机器可读标签：``5m`` / ``1d`` / ``weekly`` / ``tick`` ..."""
        if self.unit == "minute":
            return f"{self.multiplier}m"
        if self.unit == "daily":
            return f"{self.multiplier}d"
        if self.unit == "weekly":
            return "weekly" if self.multiplier == 1 else f"{self.multiplier}w"
        return self.unit  # tick / event / snapshot

    @classmethod
    def parse(cls, expr: str) -> "FrequencySpec":
        """解析 ``"5m"`` / ``"1d"`` / ``"daily"`` / ``"weekly"`` / ``"tick"``。

        数字+单位后缀（5m/15m/1d/2w）或裸频率名（daily/minute/weekly/...）；
        非法抛 ``UnitError``。
        """
        text = str(expr).strip().lower()
        m = re.fullmatch(r"(\d+)\s*([a-z]+)", text)
        if m:
            mult = int(m.group(1))
            u = m.group(2)
            if u in ("m", "min", "minute"):
                unit = "minute"
            elif u in ("d", "day", "daily"):
                unit = "daily"
            elif u in ("w", "wk", "week", "weekly"):
                unit = "weekly"
            else:
                raise UnitError(f"无法解析频率表达式 {expr!r}")
            return cls(unit=unit, multiplier=mult)
        if text in ("minute", "min"):
            return cls(unit="minute")
        if text in ("daily", "day"):
            return cls(unit="daily")
        if text in ("weekly", "week"):
            return cls(unit="weekly")
        if text in _FREQUENCY_UNITS - {"minute", "daily", "weekly"}:
            return cls(unit=text)
        raise UnitError(f"无法解析频率表达式 {expr!r}")

    # ---- 频率转换语义 ----

    def requires_aggregation(self, target: "FrequencySpec | str") -> bool:
        """target 比 self 更粗（或同 unit 周期更长）→ 需要聚合。

        - minute→daily → True；5m→1d → True；5m→15m → True（同 unit 周期更长）。
        - daily→minute → False（这是降频采样，不是聚合）；15m→5m → False。
        - 同 unit 同 multiplier（相等）→ False。
        """
        if not isinstance(target, FrequencySpec):
            target = FrequencySpec.parse(str(target))
        if self.unit == target.unit and self.multiplier == target.multiplier:
            return False
        sr, tr = _FREQUENCY_RANK[self.unit], _FREQUENCY_RANK[target.unit]
        if sr < tr:
            return True
        if sr > tr:
            return False
        return target.multiplier > self.multiplier


# ---------------------------------------------------------------------------
# GrainSpec（R30-P1-004）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class GrainSpec:
    """「一行代表什么」的粒度键集合（一等对象）。

    ``GrainSpec(("trade_date", "instrument"))`` /
    ``("knowledge_date", "instrument", "period_end")``。

    ``join_cardinality`` 返回 ``1:1`` / ``N:1`` / ``1:N`` / ``N:N``：
        - keys 完全相等 → 1:1
        - self.keys ⊂ other.keys → 1:N（self 是粗表，one 行对应 other 多行）
        - other.keys ⊂ self.keys → N:1
        - 互不包含 → N:N
    """

    keys: tuple[str, ...] = field(default_factory=tuple)

    def __init__(self, keys: Any) -> None:
        if isinstance(keys, str):
            keys = tuple(k.strip() for k in keys.split("|") if k.strip())
        else:
            keys = tuple(str(k) for k in keys)
        object.__setattr__(self, "keys", keys)

    def join_cardinality(self, other: "GrainSpec | Any") -> str:
        if isinstance(other, GrainSpec):
            okeys = other.keys
        else:
            okeys = tuple(str(k) for k in other)
        sk, ok = set(self.keys), set(okeys)
        if sk == ok:
            return "1:1"
        if sk < ok:
            return "1:N"
        if ok < sk:
            return "N:1"
        return "N:N"

    def __str__(self) -> str:
        return "|".join(self.keys)

    def __repr__(self) -> str:
        return f"GrainSpec({self.keys!r})"


# ---------------------------------------------------------------------------
# MarketProfile（R30-P1-005）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketProfile:
    """市场级稳定规则（一等对象）。

    注意：**不替代 Dataset Contract**——只承载市场级稳定事实（时区 / 默认币种 /
    日历 / 交易时段 / 结算），粒度/频率/字段语义仍由 Dataset Contract 声明。
    """

    market_id: str
    timezone: str
    default_currency: str
    calendar_id: str
    session_policy: str = "regular"
    instrument_namespace: str | None = None
    settlement_policy: str | None = None

    @classmethod
    def resolve(cls, market_id: str) -> "MarketProfile | None":
        """按 market_id 解析预设 profile；未知市场返回 None。"""
        if market_id == "ashare":
            return ASHARE_PROFILE
        if market_id == "us":
            return US_PROFILE
        return None


ASHARE_PROFILE = MarketProfile(
    market_id="ashare",
    timezone="Asia/Shanghai",
    default_currency="CNY",
    calendar_id="ashare",
    session_policy="regular",
    instrument_namespace="ashare",
    settlement_policy="t+1",
)
"""A股市场预设：Asia/Shanghai / CNY / ashare 日历 / t+1 结算。"""

US_PROFILE = MarketProfile(
    market_id="us",
    timezone="America/New_York",
    default_currency="USD",
    calendar_id="us",
    session_policy="regular",
    instrument_namespace="us",
    settlement_policy="t+1",
)
"""美股市场预设：America/New_York / USD / us 日历。"""


# ---------------------------------------------------------------------------
# PriceBasisSpec（R30-P1-006）
# ---------------------------------------------------------------------------


class PriceBasisSpec(enum.Enum):
    """价格调整基准：决定价格序列的复权语义。"""

    RAW = "raw"
    FORWARD_ADJUSTED = "forward_adjusted"
    BACKWARD_ADJUSTED = "backward_adjusted"
    TOTAL_RETURN = "total_return"
    POINT_IN_TIME_ADJUSTED = "point_in_time_adjusted"

    def label(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# InstrumentIdentity + InstrumentType（R30-P1-007）
# ---------------------------------------------------------------------------


class InstrumentType(enum.Enum):
    """工具类型。"""

    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FUTURE = "future"
    OPTION = "option"
    BOND = "bond"
    FX = "fx"
    CRYPTO = "crypto"


@dataclass(frozen=True)
class InstrumentIdentity:
    """工具身份（一等对象）。

    ``display_symbol()`` 与 ``stable_id()`` 分离：美股 ticker 可能改名
    （FB → META），此时 ``symbol`` 变化但 ``security_id`` 稳定——身份比对 /
    去重必须用 ``stable_id()``，展示用 ``display_symbol()``。
    """

    symbol: str
    security_id: str
    market: str
    instrument_type: InstrumentType = InstrumentType.EQUITY

    def display_symbol(self) -> str:
        """当前展示符号（可随 ticker rename 变化）。"""
        return self.symbol

    def stable_id(self) -> str:
        """稳定身份标识（security_id，跨 rename 不变）。"""
        return self.security_id


# ---------------------------------------------------------------------------
# RevisionFidelity（R30-P1-008）
# ---------------------------------------------------------------------------


class RevisionFidelity(enum.Enum):
    """修订可见性保真：历史修订在读取时有多「忠实」。

    从粗到细：NONE（只看当前）< KNOWLEDGE_DATE（知识日可见）< INGESTION_VINTAGE
    （入库 vintage）< TRUE_VENDOR_VINTAGE（供应商真实 vintage）。
    """

    NONE = "none"
    KNOWLEDGE_DATE = "knowledge_date"
    INGESTION_VINTAGE = "ingestion_vintage"
    TRUE_VENDOR_VINTAGE = "true_vendor_vintage"

    @classmethod
    def from_pit_fidelity(
        cls, pit_fidelity: Any, *, vendor_vintage: bool = False
    ) -> "RevisionFidelity":
        """从既有 ``SemanticField.pit_fidelity`` 字符串映射。

        - ``knowledge_date_pit`` → KNOWLEDGE_DATE
        - ``vintage_pit`` → INGESTION_VINTAGE；``vendor_vintage=True`` → TRUE_VENDOR_VINTAGE
        - ``effective_only`` / ``unsupported`` / 未知 / None → NONE（保守）
        """
        if pit_fidelity is None:
            return cls.NONE
        s = str(pit_fidelity).strip().lower()
        if s == "knowledge_date_pit":
            return cls.KNOWLEDGE_DATE
        if s == "vintage_pit":
            return cls.TRUE_VENDOR_VINTAGE if vendor_vintage else cls.INGESTION_VINTAGE
        if s in ("effective_only", "unsupported"):
            return cls.NONE
        return cls.NONE


# ---------------------------------------------------------------------------
# SemanticColumnVersion（R30-P1-009）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticColumnVersion:
    """同一物理列的 semantic epoch（一等对象）。

    物理列可能在不同时间区间有**不同语义**（如 % → decimal、口径调整）。
    每个 epoch 是一个 ``SemanticColumnVersion``：版本区间内 concept/单位/dtype 的
    语义契约。``is_current_as_of`` 支持 compile-time 判定某日期落在哪个 epoch——
    这类语义变更**不被普通 schema union 掩盖**（union 只看到物理 dtype）。
    """

    concept_id: Any  # ConceptId
    physical_column: str
    dtype: str
    unit: Any  # UnitType
    definition_version: str
    valid_from: str | None = None  # ISO 日期；None = epoch 起点
    valid_to: str | None = None    # ISO 日期（左闭右开）；None = 开放

    def is_current_as_of(self, date_str: str) -> bool:
        """date 是否落在这个 semantic epoch 内（valid_from ≤ date < valid_to）。"""
        if self.valid_from is not None and date_str < self.valid_from:
            return False
        if self.valid_to is not None and date_str >= self.valid_to:
            return False
        return True


# ---------------------------------------------------------------------------
# AggregationRecipeIdentity（R30-P1-010）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregationRecipeIdentity:
    """聚合配方（一等对象）：把 source 频率聚合到目标频率的完整规则。

    ``identity()`` 用 ``stable_digest`` 折叠全部字段——**两个都叫 VWAP 但计算时段
    / 停牌处理不同 → identity 不同**（进缓存 / 复用 key 前必须区分）。
    ``label()`` 人类可读，供日志 / 审计。
    """

    source_frequency: Any  # FrequencySpec
    market_session: str = "regular"
    minute_window: int | None = None
    timezone: str | None = None
    halt_policy: str = "strict"
    missing_bar_policy: str = "ffill"
    early_close_policy: str = "treat_as_regular"
    price_basis: Any = None  # PriceBasisSpec | str
    aggregation_function: str = "VWAP"

    def identity(self) -> str:
        """稳定身份：任一字段（计算时段/停牌策略/价格基准/聚合函数）变化 → 不同。"""
        return stable_digest(
            "AggregationRecipeIdentity",
            self.source_frequency,
            self.market_session,
            self.minute_window,
            self.timezone,
            self.halt_policy,
            self.missing_bar_policy,
            self.early_close_policy,
            self.price_basis,
            self.aggregation_function,
        )

    def label(self) -> str:
        freq = self.source_frequency
        fstr = freq.label() if hasattr(freq, "label") else str(freq)
        basis = self.price_basis
        bstr = basis.label() if hasattr(basis, "label") else str(basis)
        return (
            f"{self.aggregation_function}(freq={fstr}, session={self.market_session}, "
            f"window={self.minute_window}, tz={self.timezone}, halt={self.halt_policy}, "
            f"missing={self.missing_bar_policy}, early_close={self.early_close_policy}, "
            f"basis={bstr})"
        )


__all__ = [
    "ASHARE_PROFILE",
    "AggregationRecipeIdentity",
    "FrequencySpec",
    "GrainSpec",
    "InstrumentIdentity",
    "InstrumentType",
    "MarketProfile",
    "PriceBasisSpec",
    "RevisionFidelity",
    "SemanticColumnVersion",
    "US_PROFILE",
]
