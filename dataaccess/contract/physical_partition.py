"""R25 P0-001/002/016 —— 物理分区契约（PhysicalPartitionSpec）。

把「物理文件布局」与「逻辑日历」彻底分离：
    - ``PhysicalLayout``       布局枚举（daily / event / period / prefixed / hive / static / glob / generation）
    - ``MissingPartitionSemantics``  缺失 partition 语义（error / warn / empty_ok）
    - ``PhysicalPartitionSpec``     单数据集物理分区声明

**为什么需要**：US finance 契约声明 ``storage_layout=period_files``（物理文件名 =
period_end，如 ``StockIncome/2024-03-31.parquet``），而 mirror/remote planner 过去
用 ``daily_parquet`` 布局按 query/decision 日期展开 ``{request_day}.parquet``——
predicate clock（filing_date）被错误当成 physical filename date，可能漏读真实 filing。
Event/period 数据也不再套 trade-day expected-partition 模型（P0-016）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from data_access.core.exceptions import ValidationError


class PhysicalLayout(Enum):
    """物理文件布局枚举（R25 §4 / §5 / §17）。

    - ``DAILY_TRADE_DATE``   ：每交易日一个文件，文件名 = 交易日（如 StockDailyBar/2024-01-02.parquet）
    - ``DAILY_CALENDAR_DATE``：每自然日一个文件（StockList 周末文件）
    - ``EVENT_DATE_FILE``    ：事件表，文件名 = 事件日期（event_driven，稀疏）
    - ``PERIOD_END_FILE``    ：财务期间文件，文件名 = period_end（US finance；partition_clock=period_end）
    - ``PREFIXED_DATE_FILE`` ：带前缀的日期文件（StockCapital ``{date}.parquet`` /
                              ``shares_{date}.parquet``，file_selector 区分）
    - ``HIVE_DATE``          ：hive ``date=YYYY-MM-DD/data.parquet``
    - ``HIVE_YEAR``          ：hive ``year=YYYY/data.parquet``
    - ``STATIC_SINGLE``      ：单文件静态维表（full.parquet）
    - ``PLAIN_GLOB``         ：未分区 glob（``**/*.parquet``）
    - ``GENERATION_POINTER`` ：generation pointer（未来 immutable 源）
    """

    DAILY_TRADE_DATE = "daily_trade_date"
    DAILY_CALENDAR_DATE = "daily_calendar_date"
    EVENT_DATE_FILE = "event_date_file"
    PERIOD_END_FILE = "period_end_file"
    PREFIXED_DATE_FILE = "prefixed_date_file"
    HIVE_DATE = "hive_date"
    HIVE_YEAR = "hive_year"
    STATIC_SINGLE = "static_single"
    PLAIN_GLOB = "plain_glob"
    GENERATION_POINTER = "generation_pointer"


class MissingPartitionSemantics(Enum):
    """缺失 partition 的处理语义（R25 §16 / P0-015）。

    不能统一「404 → debug 跳过」：
        - dense 行情（StockDailyBar）缺失交易日文件 → ERROR
        - 稀疏事件表某天无 event → EMPTY_OK
        - 中间状态 → WARN（research 告警，production 依据 contract 决定）
    """

    ERROR = "error"
    WARN = "warn"
    EMPTY_OK = "empty_ok"


# 旧 MirrorSpec 字符串布局 → PhysicalLayout 映射（兼容迁移，P0-001）。
_MIRROR_LAYOUT_MAP = {
    "daily_parquet": PhysicalLayout.DAILY_TRADE_DATE,
    "hive_date": PhysicalLayout.HIVE_DATE,
    "hive_year": PhysicalLayout.HIVE_YEAR,
    "single_full": PhysicalLayout.STATIC_SINGLE,
    "root_file": PhysicalLayout.STATIC_SINGLE,
    # R25 P0-001/002：MirrorSpec 直接声明 period/prefixed/event 布局。
    "period_files": PhysicalLayout.PERIOD_END_FILE,
    "prefixed_date_file": PhysicalLayout.PREFIXED_DATE_FILE,
    "event_files": PhysicalLayout.EVENT_DATE_FILE,
}

# COSDatasetContract.storage_layout 字符串 → PhysicalLayout 映射（P0-001）。
_CONTRACT_LAYOUT_MAP = {
    "daily_parquet": PhysicalLayout.DAILY_TRADE_DATE,
    "event_files": PhysicalLayout.EVENT_DATE_FILE,
    "period_files": PhysicalLayout.PERIOD_END_FILE,
    "sparse_files": PhysicalLayout.EVENT_DATE_FILE,
    "hive_date": PhysicalLayout.HIVE_DATE,
    "hive_year": PhysicalLayout.HIVE_YEAR,
}


def layout_from_mirror(layout: str | None) -> PhysicalLayout:
    """旧 MirrorSpec.layout 字符串 → PhysicalLayout（未知 fail-closed）。"""
    key = str(layout or "daily_parquet").strip().lower()
    mapped = _MIRROR_LAYOUT_MAP.get(key)
    if mapped is None:
        raise ValidationError(
            f"未知 mirror layout {layout!r}；必须映射到 PhysicalLayout "
            f"（{sorted(_MIRROR_LAYOUT_MAP)} 或显式 PhysicalLayout）"
        )
    return mapped


def layout_from_contract(layout: str | None) -> PhysicalLayout | None:
    """COSDatasetContract.storage_layout 字符串 → PhysicalLayout（None = 未声明）。"""
    key = str(layout or "").strip().lower()
    if not key:
        return None
    mapped = _CONTRACT_LAYOUT_MAP.get(key)
    if mapped is None:
        raise ValidationError(
            f"未知 storage_layout {layout!r}；必须映射到 PhysicalLayout "
            f"（{sorted(_CONTRACT_LAYOUT_MAP)}）"
        )
    return mapped


def is_calendar_clock_layout(layout: PhysicalLayout) -> bool:
    """布局是否按「日历可枚举」的 partition clock 展开。

    只有这些布局能用 expected dates / years 直接展开文件名；其余（event /
    period / prefixed / static / glob）**禁止**用 request time_range 拼文件名
    （P0-001/016：partition_clock != predicate_clock 时不能瞎映射）。
    """
    return layout in {
        PhysicalLayout.DAILY_TRADE_DATE,
        PhysicalLayout.DAILY_CALENDAR_DATE,
        PhysicalLayout.HIVE_DATE,
        PhysicalLayout.HIVE_YEAR,
    }


@dataclass(frozen=True)
class PhysicalPartitionSpec:
    """单数据集物理分区声明（R25 §4）。

    - ``layout``               ：PhysicalLayout
    - ``partition_clock``      ：物理文件名所表示的时钟（如 period_end / date）
    - ``filename_template``    ：文件名模板（``{period_end}.parquet`` / ``{date}.parquet`` /
                                  ``shares_{date}.parquet``），{date}/{period_end}/{year}/{month}
    - ``file_selector``        ：PREFIXED_DATE_FILE 的区分前缀（``shares_``），保证 split 与
                                  shares 同一目录各自只选自己那类文件
    - ``query_clock``          ：查询侧时钟（如 filing_date），predicate 用它裁剪
    - ``completeness``         ：缺失 partition 语义（MissingPartitionSemantics）
    - ``source``               ：声明来源（mirror / contract / storage），审计用
    """

    layout: PhysicalLayout
    partition_clock: str | None = None
    filename_template: str | None = None
    prefix: str | None = None
    file_selector: str | None = None
    query_clock: str | None = None
    completeness: MissingPartitionSemantics = MissingPartitionSemantics.ERROR
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, PhysicalLayout):
            raise ValidationError(
                f"PhysicalPartitionSpec.layout 必须是 PhysicalLayout，收到 {self.layout!r}"
            )
        if not isinstance(self.completeness, MissingPartitionSemantics):
            raise ValidationError(
                f"completeness 必须是 MissingPartitionSemantics，收到 {self.completeness!r}"
            )
        if self.file_selector and not self.file_selector:
            raise ValidationError("file_selector 不能为空字符串")

    @property
    def calendar_enumerable(self) -> bool:
        """该布局能否用 expected dates/years 展开文件名（P0-016）。"""
        return is_calendar_clock_layout(self.layout)

    @property
    def requires_index_mapping(self) -> bool:
        """predicate_clock != partition_clock 时必须有显式 mapping/index，否则不裁剪。

        对 PERIOD_END_FILE / EVENT_DATE_FILE / PREFIXED_DATE_FILE 成立：
        不能用 request time_range 直接拼文件名。
        """
        return self.layout in {
            PhysicalLayout.PERIOD_END_FILE,
            PhysicalLayout.EVENT_DATE_FILE,
            PhysicalLayout.PREFIXED_DATE_FILE,
            PhysicalLayout.PLAIN_GLOB,
        }

    def filename_for(self, **clock_values: Any) -> str:
        """按 partition_clock 值渲染文件名模板。

        ``filename_template`` 缺失时 fallback 到 ``{date}.parquet``（兼容旧 daily）。
        只接受模板里出现的占位符值。
        """
        template = self.filename_template or "{date}.parquet"
        out = template
        for key, val in (clock_values or {}).items():
            token = "{" + str(key) + "}"
            if token in out:
                out = out.replace(token, str(val))
        if "{" in out or "}" in out:
            raise ValidationError(
                f"PhysicalPartitionSpec.filename_for: 模板 {template!r} 仍有未填充"
                f"占位符（缺 partition_clock 值: {list(clock_values)}）"
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.value,
            "partition_clock": self.partition_clock,
            "filename_template": self.filename_template,
            "prefix": self.prefix,
            "file_selector": self.file_selector,
            "query_clock": self.query_clock,
            "completeness": self.completeness.value,
            "source": self.source,
        }
