# -*- coding: utf-8 -*-
"""R32-P0-079..086: 增量重算类型系统（typed uncertainty + operator traits）。

本模块定义增量重算的类型化域，解决 R32-P0-079..086 的 under-invalidation bug：
    - R32-P0-079: OperatorDependencyTraits（backward input + forward output horizon）
    - R32-P0-080: TimeAxisKind（TRADING_BARS vs CALENDAR_DAYS）
    - R32-P0-081: AxisEffect（cross-sectional 扩散全截面）
    - R32-P0-082: InstrumentScope（EXACT_SET vs RANGE，manifest min/max 是统计量）
    - R32-P0-083: ColumnChangeKind（typed columns，空 tuple 不再多义）
    - R32-P0-085: ImpactDirection（FORWARD/BACKWARD/BIDIRECTIONAL，split 重写历史）

**不再用空容器同时表示「无变化」和「不知道」** —— 每种不确定性显式类型化，
消费方必须保守传播（unknown → widest scope）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# R32-P0-085: Impact direction（公司行为可重写 HISTORY）
# ---------------------------------------------------------------------------
class ImpactDirection(Enum):
    """增量变化的影响方向（R32-P0-085）。

    POINT_ONLY    单点变化（如 t 日价格修订只影响 t 日输出）
    FORWARD       向前传播（rolling 依赖：input[t] 变化影响 outputs[t..t+N]）
    BACKWARD      向后传播（split/复权：t 生效但重写 [start, t] 历史序列）
    BIDIRECTIONAL 双向（同时向前和向后，如某些状态算子的 rebase）
    """

    POINT_ONLY = "point_only"
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


# ---------------------------------------------------------------------------
# R32-P0-080: Time axis kind（BARS vs calendar days）
# ---------------------------------------------------------------------------
class TimeAxisKind(Enum):
    """时间轴类型（R32-P0-080）。

    TRADING_BARS  交易日 bars（window=20 是 20 个交易日，需 TradingCalendar 映射）
    CALENDAR_DAYS 自然日（window=20 是 20 calendar days）
    INDEX_POINTS  整数索引（与日期无关的序列位置，如 minute bar index）
    """

    TRADING_BARS = "trading_bars"
    CALENDAR_DAYS = "calendar_days"
    INDEX_POINTS = "index_points"


# ---------------------------------------------------------------------------
# R32-P0-081: Axis effect（cross-sectional 扩散）
# ---------------------------------------------------------------------------
class AxisEffect(Enum):
    """算子轴效应（R32-P0-081）。

    ELEMENTWISE            逐元素（单标的单时间点，无扩散）
    TIME_SERIES           时间序列（单标的跨时间，rolling 向前传播）
    CROSS_SECTION         截面（单时间点跨标的，一变全变）
    GROUP_CROSS_SECTION   组内截面（行业内一变组内全变，不扩散到无关组）
    GLOBAL_PANEL          全局面板（跨时间跨标的，如 PCA/factor model）
    """

    ELEMENTWISE = "elementwise"
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"
    GROUP_CROSS_SECTION = "group_cross_section"
    GLOBAL_PANEL = "global_panel"


# ---------------------------------------------------------------------------
# R32-P0-079: Operator dependency traits（backward + forward horizon）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorDependencyTraits:
    """算子依赖特性（R32-P0-079 核心修复）。

    backward_input_horizon   向后读多少个 input（MA20 读过去 20 个 bar）
    forward_output_horizon   向前污染多少个 output（MA20 input[t] 变化影响
                             outputs[t..t+19]，R32-P0-079 旧 bug 未标记此项）
    axis_effect              轴效应（R32-P0-081）
    time_axis                时间轴类型（R32-P0-080）

    示例：
        MA20 = OperatorDependencyTraits(
            backward_input_horizon=20,   # 读过去 20 个 bar
            forward_output_horizon=20,   # 向前污染 20 个 output（R32-P0-079 修复）
            axis_effect=AxisEffect.TIME_SERIES,
            time_axis=TimeAxisKind.TRADING_BARS,
        )
        rank(ROE) = OperatorDependencyTraits(
            backward_input_horizon=0,    # 不读历史（单日截面）
            forward_output_horizon=0,    # 不向前污染
            axis_effect=AxisEffect.CROSS_SECTION,  # 一变全变（R32-P0-081）
            time_axis=TimeAxisKind.TRADING_BARS,
        )
    """

    backward_input_horizon: int = 0
    forward_output_horizon: int = 0
    axis_effect: str | AxisEffect = AxisEffect.ELEMENTWISE
    time_axis: str | TimeAxisKind = TimeAxisKind.TRADING_BARS

    def __post_init__(self):
        # 归一化 enum
        if isinstance(self.axis_effect, str):
            object.__setattr__(self, "axis_effect", AxisEffect(self.axis_effect))
        if isinstance(self.time_axis, str):
            object.__setattr__(self, "time_axis", TimeAxisKind(self.time_axis))


# ---------------------------------------------------------------------------
# R32-P0-082: Instrument scope（EXACT_SET vs RANGE vs unknown）
# ---------------------------------------------------------------------------
class InstrumentScope(Enum):
    """标的范围类型（R32-P0-082）。

    EXACT_SET       精确标的集合（逐行扫描得出，确定性）
    RANGE           min/max 统计量（manifest footer，**不是**精确集合）
    PARTITION_WIDE  整个分区的标的（如 date=2024-01-01 分区全量）
    UNIVERSE_WIDE   整个 universe 的标的（全市场）
    UNKNOWN_ALL     不知道哪些标的变了（保守视为全量）

    **关键**：manifest min/max 只是 RANGE 统计量，消费方不能假设「只有 min 和 max
    两个标的变了」（会 under-invalidate），必须保守处理「[min, max] 区间内任何
    标的都可能变」。
    """

    EXACT_SET = "exact_set"
    RANGE = "range"
    PARTITION_WIDE = "partition_wide"
    UNIVERSE_WIDE = "universe_wide"
    UNKNOWN_ALL = "unknown_all"


@dataclass(frozen=True)
class InstrumentChange:
    """类型化标的变化（R32-P0-082）。

    scope: InstrumentScope
    exact_instruments: tuple（仅当 scope=EXACT_SET）
    min_instrument: str（仅当 scope=RANGE）
    max_instrument: str（仅当 scope=RANGE）
    partition_key: Any（仅当 scope=PARTITION_WIDE）
    universe_id: str（仅当 scope=UNIVERSE_WIDE）
    """

    scope: InstrumentScope
    exact_instruments: tuple[str, ...] = ()
    min_instrument: str | None = None
    max_instrument: str | None = None
    partition_key: Any = None
    universe_id: str | None = None

    def is_conservative(self) -> bool:
        """是否是保守全量（UNKNOWN_ALL / UNIVERSE_WIDE / PARTITION_WIDE）。"""
        return self.scope in {
            InstrumentScope.UNKNOWN_ALL,
            InstrumentScope.UNIVERSE_WIDE,
            InstrumentScope.PARTITION_WIDE,
        }


def coerce_instrument_change(
    legacy: tuple[str, ...] | None,
) -> InstrumentChange:
    """R32-P0-082: 向后兼容 legacy changed_instruments tuple。

    默认保守策略：
        - None / () → UNKNOWN_ALL（不知道哪些变了，保守全量）
        - (min, max) 从 manifest 推导 → RANGE（统计量，不是精确集合）
        - 有 1-2 个元素且看起来是 manifest min/max → RANGE
        - 其他 → EXACT_SET（假设调用方逐行扫描得出）
    """
    if legacy is None or len(legacy) == 0:
        return InstrumentChange(scope=InstrumentScope.UNKNOWN_ALL)

    # 启发式：2 个元素且字典序排列，可能是 manifest min/max（保守视为 RANGE）
    if len(legacy) == 2 and legacy[0] < legacy[1]:
        return InstrumentChange(
            scope=InstrumentScope.RANGE,
            min_instrument=legacy[0],
            max_instrument=legacy[1],
        )

    # 默认：假设是精确集合（若不确定，调用方应显式构造 InstrumentChange）
    return InstrumentChange(
        scope=InstrumentScope.EXACT_SET,
        exact_instruments=legacy,
    )


# ---------------------------------------------------------------------------
# R32-P0-083: Column change kind（typed columns，空 tuple 不再多义）
# ---------------------------------------------------------------------------
class ColumnChangeKind(Enum):
    """列变化类型（R32-P0-083）。

    EXACT_COLUMNS      精确列列表（确定哪些列变了）
    ALL_COLUMNS_UNKNOWN 不知道哪些列变了（保守视为全列）
    SCHEMA_ONLY        schema 变化（列增减 / 类型变化），数据未变
    NO_COLUMN_CHANGE   无列变化（如 partition 级元数据更新）

    **不再用空 tuple 同时表示「无变化」和「不知道」**。
    """

    EXACT_COLUMNS = "exact_columns"
    ALL_COLUMNS_UNKNOWN = "all_columns_unknown"
    SCHEMA_ONLY = "schema_only"
    NO_COLUMN_CHANGE = "no_column_change"


@dataclass(frozen=True)
class ColumnChange:
    """类型化列变化（R32-P0-083）。

    kind: ColumnChangeKind
    exact_columns: tuple（仅当 kind=EXACT_COLUMNS）
    """

    kind: ColumnChangeKind
    exact_columns: tuple[str, ...] = ()


def coerce_column_change(
    legacy: tuple[str, ...] | None,
) -> ColumnChange:
    """R32-P0-083: 向后兼容 legacy changed_columns tuple。

    默认保守策略：
        - None → ALL_COLUMNS_UNKNOWN（不知道哪些列变了）
        - () → NO_COLUMN_CHANGE（显式标记无列变化，**不再**表示 don't-know）
        - 非空 → EXACT_COLUMNS
    """
    if legacy is None:
        return ColumnChange(kind=ColumnChangeKind.ALL_COLUMNS_UNKNOWN)
    if len(legacy) == 0:
        return ColumnChange(kind=ColumnChangeKind.NO_COLUMN_CHANGE)
    return ColumnChange(
        kind=ColumnChangeKind.EXACT_COLUMNS,
        exact_columns=legacy,
    )


# ---------------------------------------------------------------------------
# R32-P0-077/078: Time axis disambiguation（四轴显式）
# ---------------------------------------------------------------------------
class TimeAxisName(Enum):
    """时间轴名称（R32-P0-077）。

    DATA_TIME       数据点的交易日期/时间戳（价格 date、minute bar timestamp）
    PERIOD_TIME     周期归属（财报的 period_end、季度/年度归属）
    KNOWLEDGE_TIME  信息可见时间（revision announcement_date、insider 可见性）
    EFFECTIVE_TIME  生效时间（公司行为 ex_date、split 生效日）

    **关键**：manifest min/max_time 通常是 DATA_TIME 或 PERIOD_TIME 轴，**不是**
    KNOWLEDGE_TIME（revision 可见性需单独字段）。R32-P0-077 修复：不再混淆四轴。
    """

    DATA_TIME = "data_time"
    PERIOD_TIME = "period_time"
    KNOWLEDGE_TIME = "knowledge_time"
    EFFECTIVE_TIME = "effective_time"


@dataclass(frozen=True)
class TimeRangeTyped:
    """类型化时间范围（R32-P0-077）。

    axis: TimeAxisName（明确哪个时间轴）
    start: str | None
    end: str | None
    """

    axis: TimeAxisName
    start: str | None = None
    end: str | None = None


# ---------------------------------------------------------------------------
# R32-P0-078: Revision fidelity（不假设 mtime 是历史 vintage）
# ---------------------------------------------------------------------------
class RevisionFidelity(Enum):
    """Revision 保真度（R32-P0-078）。

    NONE                    无 revision 信息（只有当前版本）
    KNOWLEDGE_DATE          当前版本的可见时间（从 mtime/announcement 推导）
    INGESTION_VINTAGE       ingestion ledger 记录的 vintage（知道何时摄入）
    TRUE_VENDOR_VINTAGE     真实 vendor revision store（可重建历史 decision_time）

    **关键**：从单个物理文件只能判定 KNOWLEDGE_DATE（当前版本何时可见），**无法**
    重建「在历史 decision_time X 时刻，消费方看到的是哪个 revision」（需真实
    vendor revision store）。R32-P0-078 修复：不假装 mtime 是历史 vintage。
    """

    NONE = "none"
    KNOWLEDGE_DATE = "knowledge_date"
    INGESTION_VINTAGE = "ingestion_vintage"
    TRUE_VENDOR_VINTAGE = "true_vendor_vintage"


__all__ = [
    # R32-P0-079
    "OperatorDependencyTraits",
    # R32-P0-080
    "TimeAxisKind",
    # R32-P0-081
    "AxisEffect",
    # R32-P0-082
    "InstrumentScope",
    "InstrumentChange",
    "coerce_instrument_change",
    # R32-P0-083
    "ColumnChangeKind",
    "ColumnChange",
    "coerce_column_change",
    # R32-P0-085
    "ImpactDirection",
    # R32-P0-077
    "TimeAxisName",
    "TimeRangeTyped",
    # R32-P0-078
    "RevisionFidelity",
]
