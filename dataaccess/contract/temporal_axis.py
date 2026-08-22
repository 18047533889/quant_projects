"""R25 P0-005/017/018 —— 时间轴契约（TemporalAxisSpec）与 AvailabilityResult。

TemporalAxisSpec：
    - ``column``           时间列名
    - ``representation``   date_label（filing_date / PubDate 日期标签）vs instant（真实时刻）
    - ``precision``        date / second / millisecond / nanosecond
    - ``storage_timezone`` 物理存储时区（UTC）
    - ``semantic_timezone``语义时区（America/New_York / Asia/Shanghai）

AvailabilityResult：
    - ``available_from``      编译出的可见时点
    - ``authoritative``      是否权威（true = 可证明；false = degraded/无法证明）
    - ``calendar_snapshot_id``使用的日历快照
    - ``degradation_reason``  降级原因（production 下 authoritative=false 必须拒绝）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from data_access.core.exceptions import ValidationError

_REPRESENTATIONS = frozenset({"date_label", "instant"})
_PRECISIONS = frozenset({"date", "second", "millisecond", "nanosecond"})


@dataclass(frozen=True)
class TemporalAxisSpec:
    """单根时间轴的表示契约（R25 §18 / P0-017）。

    US finance ``filing_date``：representation=date_label、precision=date、
    semantic_timezone=America/New_York——naive 00:00 **不是** UTC instant，
    绝不被转成纽约 2024-05-09 20:00（提前一天）。
    US news ``published_utc``：representation=instant、storage_timezone=UTC、
    semantic_timezone=America/New_York——只有 instant 才做时区换算。
    """

    column: str
    representation: Literal["date_label", "instant"] = "instant"
    precision: Literal["date", "second", "millisecond", "nanosecond"] = "second"
    storage_timezone: str | None = None
    semantic_timezone: str | None = None

    def __post_init__(self) -> None:
        if self.representation not in _REPRESENTATIONS:
            raise ValidationError(
                f"TemporalAxisSpec.representation 必须是 date_label/instant，收到 {self.representation!r}"
            )
        if self.precision not in _PRECISIONS:
            raise ValidationError(
                f"TemporalAxisSpec.precision 必须是 {sorted(_PRECISIONS)}，收到 {self.precision!r}"
            )

    @property
    def is_date_label(self) -> bool:
        return self.representation == "date_label"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "representation": self.representation,
            "precision": self.precision,
            "storage_timezone": self.storage_timezone,
            "semantic_timezone": self.semantic_timezone,
        }


@dataclass(frozen=True)
class AvailabilityResult:
    """AvailabilityCompiler 的结构化结果（R25 §8 / P0-005）。

    production：``authoritative=False``（日历缺失 / 右边界不可证明）→ 调用方必须拒绝。
    research：可以返回 degraded，但 lineage 必须记录 ``degradation_reason``。
    """

    available_from: Any
    authoritative: bool
    calendar_snapshot_id: str | None = None
    degradation_reason: str | None = None

    @property
    def degraded(self) -> bool:
        return not self.authoritative

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_from": self.available_from,
            "authoritative": self.authoritative,
            "calendar_snapshot_id": self.calendar_snapshot_id,
            "degradation_reason": self.degradation_reason,
        }
