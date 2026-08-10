"""R26-P0-010 —— FileSelector IR：物理文件选择器（local/mirror/remote/snapshot 共用）。

StockCapital 目录混放两类 schema：
    - split :  ``2024-01-01.parquet``（schema A）
    - shares:  ``shares_2024-01-01.parquet``（schema B）

绝不能用「空 prefix + *」表达排除逻辑（``*.parquet`` 会同时命中两类）。
``FileSelector`` 把 layout + file_selector + filename_template 编译成：
    - ``glob_pattern(base)`` ：给 DuckDB/远程 planner 的**精确** glob（字符类限定
      date 文件名，不吞 ``shares_*.parquet``）；
    - ``regex(basename)``    ：给 COS LIST 结果做 exact object 过滤；
    - ``matches(basename)``  ：单文件名判定（local mirror 用）。

所有读面（local/remote/mirror/snapshot resolver）消费同一 FileSelector IR，
不再各写各的字符串启发式。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError

from .physical_partition import PhysicalLayout

# date 文件名 glob 字符类（YYYY-MM-DD）
_DATE_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
_DATE_RE = r"\d{4}-\d{2}-\d{2}"


@dataclass(frozen=True)
class FileSelector:
    """物理文件选择器（R26-P0-010）。

    - ``layout``        ：PhysicalLayout
    - ``file_selector`` ：前缀（``shares_``）；None = 无前缀（date-only 文件）
    """

    layout: PhysicalLayout
    file_selector: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, PhysicalLayout):
            raise ValidationError(
                f"FileSelector.layout 必须是 PhysicalLayout，收到 {self.layout!r}"
            )

    @property
    def prefix(self) -> str:
        return str(self.file_selector or "")

    def glob_pattern(self, base: str) -> str:
        """给远程 planner 的精确 glob（不吞其他前缀文件）。"""
        if self.layout == PhysicalLayout.PREFIXED_DATE_FILE:
            return f"{base.rstrip('/')}/{self.prefix}{_DATE_GLOB}.parquet"
        if self.layout == PhysicalLayout.PERIOD_END_FILE:
            return f"{base.rstrip('/')}/{_DATE_GLOB}.parquet"
        if self.layout in {
            PhysicalLayout.DAILY_TRADE_DATE,
            PhysicalLayout.DAILY_CALENDAR_DATE,
            PhysicalLayout.EVENT_DATE_FILE,
        }:
            return f"{base.rstrip('/')}/{self.prefix}{_DATE_GLOB}.parquet"
        # 其他布局用目录 glob（snapshot resolver 再 exact LIST 过滤）。
        return f"{base.rstrip('/')}/*.parquet"

    def regex(self) -> re.Pattern[str]:
        """COS LIST 结果的 basename 精确匹配。"""
        if self.layout == PhysicalLayout.PREFIXED_DATE_FILE:
            return re.compile(rf"^{re.escape(self.prefix)}{_DATE_RE}\.parquet$")
        if self.layout == PhysicalLayout.PERIOD_END_FILE:
            return re.compile(rf"^{_DATE_RE}\.parquet$")
        if self.layout in {
            PhysicalLayout.DAILY_TRADE_DATE,
            PhysicalLayout.DAILY_CALENDAR_DATE,
            PhysicalLayout.EVENT_DATE_FILE,
        }:
            return re.compile(rf"^{re.escape(self.prefix)}{_DATE_RE}\.parquet$")
        return re.compile(r"^.*\.parquet$")

    def matches(self, basename: str) -> bool:
        """单文件名是否属于本 selector 选中的对象。"""
        return bool(self.regex().match(str(basename)))

    def filter_objects(self, uris: Any) -> list[str]:
        """从 LIST 结果（URI 或 ResolvedObject）过滤出 exact objects。"""
        out: list[str] = []
        for obj in uris or ():
            uri = str(getattr(obj, "uri", obj))
            base = uri.rsplit("/", 1)[-1]
            if self.matches(base):
                out.append(uri)
        return out


def file_selector_for_layout(
    layout: PhysicalLayout,
    *,
    file_selector: str | None = None,
) -> FileSelector:
    """从 PhysicalPartitionSpec 编译 FileSelector（单 truth）。"""
    return FileSelector(layout=layout, file_selector=file_selector)
