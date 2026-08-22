"""
data_access.read.minute_filter —— 分钟→日聚合过滤条件的规范化表示（R39 PERF-034/035）

背景
    分钟→日聚合把 HH:MM 字符串过滤（``strftime('%H:%M') >= '09:31'``）改为整数
    分钟比较（``_minute >= 571``）。本模块提供两个可复用件：

    1. ``hhmm_to_minutes``：HH:MM → 当日分钟数的纯函数，SQL 侧与 pandas 回退侧
       （如 ``lqtp_logical_source_v2.py`` 的 ``dt.strftime`` 路径）都能复用；
    2. ``FilterSignature``：一个 minute-window FILTER 条件的规范化签名。两个
       ``AggregationSpec`` 只要 signature 相同，生成的 SQL 过滤条件就完全相同，
       bundle SQL 可以只生成一次条件列、多个输出列共享（PERF-035）。

    边界语义与旧 ``strftime('%H:%M')`` 字符串比较完全一致：时间戳按**分钟向下取整**
    （09:30:59 → 09:30/570，09:31:00 → 09:31/571），因此 ``09:30:59`` 被排除、
    ``09:31:00`` 被包含。本文件不 import aggregation.py（避免循环依赖）；需要的
    session total bars 由调用方传入。

维护人：quant 基础平台组    最后更新：2026-08-11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError


def hhmm_to_minutes(hhmm: str) -> int:
    """HH:MM → 当日分钟数（'09:31' → 571）。

    ``hhmm`` 必须已通过 HH:MM 严格校验（``^([01]\d|2[0-3]):[0-5]\d$``），因此
    返回值是 0..1439 的整数，可以直接内联进 SQL 字面量（无注入风险）。pandas
    回退路径也可复用：``frame["minute"] = frame["timestamp"].dt.hour * 60 +
    frame["timestamp"].dt.minute`` 后再按整数比较，替代字符串 ``dt.strftime``。
    """
    h, m = (int(p) for p in str(hhmm).split(":", 1))
    return h * 60 + m


@dataclass(frozen=True)
class FilterSignature:
    """一个 minute-window FILTER 条件的规范化表示（PERF-035）。

    字段
        kind: minute_at / minute_range / minute_of_day
        start_minute / end_minute: minute_at 与 minute_range 的整数分钟边界
            （minute_at 时 start == end；闭区间）
        slot_start / slot_end: minute_of_day 的 session-elapsed 桶边界
            （slot_start 含、slot_end 不含）

    两个 AggregationSpec 若 ``canonicalize`` 结果相等，则 ``to_sql`` 产出的过滤
    条件逐字相同——bundle SQL 里可以只生成一次 CTE 布尔列、多列共享。
    """

    kind: str
    start_minute: int | None = None
    end_minute: int | None = None
    slot_start: int | None = None
    slot_end: int | None = None

    @classmethod
    def canonicalize(
        cls,
        spec: Any,
        *,
        session_total_bars: int | None = None,
    ) -> "FilterSignature":
        """把 AggregationSpec 映射成规范化签名（与 ``_spec_filter_sql`` 同语义）。

        ``session_total_bars``：minute_of_day 需要 session 总 bar 数；None 时按
        旧默认 240（A 股）。错误信息与 aggregation.py 原有消息逐字一致。
        """
        if spec.aggregation == "minute_at":
            if not spec.hhmm:
                raise ValidationError("minute_at 需要 hhmm")
            m = hhmm_to_minutes(spec.hhmm)
            return cls(kind="minute_at", start_minute=m, end_minute=m)
        if spec.aggregation == "minute_range":
            if not spec.start or not spec.end:
                raise ValidationError("minute_range 需要 start 和 end")
            if spec.start >= spec.end:
                raise ValidationError("minute_range start 必须早于 end")
            return cls(
                kind="minute_range",
                start_minute=hhmm_to_minutes(spec.start),
                end_minute=hhmm_to_minutes(spec.end),
            )
        if spec.aggregation == "minute_of_day":
            period = max(1, spec.period)
            total = session_total_bars if session_total_bars is not None else 240
            # index=0 → 最新桶；桶号从 session 尾部向前数
            last_slot = max(0, (total - 1) // period)
            slot = max(0, last_slot - max(0, spec.index))
            return cls(
                kind="minute_of_day",
                slot_start=slot * period,
                slot_end=(slot + 1) * period,
            )
        raise ValidationError(f"不支持的聚合: {spec.aggregation}")

    def to_sql(self, *, minute_expr: str, elapsed_expr: str) -> str | None:
        """把签名渲染成 SQL 过滤表达式（整数分钟字面量内联）。

        ``minute_expr``：整数分钟表达式/列名（``_minute``）；``elapsed_expr``：
        session elapsed 表达式/列名（``_el``）。返回 None 表示不过滤（全时段）。
        """
        if self.kind == "minute_at":
            return f"{minute_expr} = {self.start_minute}"
        if self.kind == "minute_range":
            return (
                f"{minute_expr} >= {self.start_minute} AND "
                f"{minute_expr} <= {self.end_minute}"
            )
        if self.kind == "minute_of_day":
            return (
                f"{elapsed_expr} >= {self.slot_start} AND "
                f"{elapsed_expr} < {self.slot_end}"
            )
        raise ValidationError(f"不支持的聚合 signature: {self.kind}")
