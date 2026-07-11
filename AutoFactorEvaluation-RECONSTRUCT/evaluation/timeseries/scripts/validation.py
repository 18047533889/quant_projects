"""时序运行验证事件定义。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：集中记录结构化诊断信息，并将 WARNING/ERROR 同步写入运行日志。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationEvent:
    """单条验证事件对象（FID §5.2 validation_report）。"""

    severity: Severity
    code: str
    message: str
    count: int | None = None
    details: dict[str, Any] | None = None


@dataclass
class ValidationBuffer:
    """收集单次运行的验证事件，可选绑定 run logger（FID §6 日志）。"""

    events: list[ValidationEvent] = field(default_factory=list)
    pit_dropped_rows: int = 0
    _logger: logging.Logger | None = field(default=None, repr=False)

    def bind_logger(self, logger: logging.Logger | None) -> None:
        """绑定本次运行的文件 logger；WARNING/ERROR 将同步写入日志。"""
        self._logger = logger

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        count: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """追加一条结构化验证事件。

        入参：
            severity: 事件严重级别。
            code: 机器可读事件码。
            message: 人类可读说明。
            count: 可选计数信息。
            details: 可选结构化上下文。
        """
        self.events.append(
            ValidationEvent(
                severity=severity,
                code=code,
                message=message,
                count=count,
                details=details,
            )
        )
        if self._logger is None or severity == Severity.INFO:
            return
        suffix = f" | count={count}" if count is not None else ""
        if details:
            suffix = f"{suffix} | details={details}"
        log_line = f"{code}: {message}{suffix}"
        if severity == Severity.WARNING:
            self._logger.warning(log_line)
        elif severity == Severity.ERROR:
            self._logger.error(log_line)

    def has_error(self) -> bool:
        """检查是否存在 ERROR 级别事件。"""
        return any(e.severity == Severity.ERROR for e in self.events)

    def to_report_dict(self, *, eval_run_id: str, factor_id: str) -> dict[str, Any]:
        """将验证状态序列化为 `validation_report.json` 所需字典。

        入参：
            eval_run_id: 本次评估运行 ID。
            factor_id: 当前因子 ID。

        出参：
            dict[str, Any]：可 JSON 序列化的验证报告对象。
        """
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eval_run_id": eval_run_id,
            "factor_id": factor_id,
            "pit_dropped_rows": self.pit_dropped_rows,
            "events": [
                {
                    "severity": e.severity.value,
                    "code": e.code,
                    "message": e.message,
                    **({"count": e.count} if e.count is not None else {}),
                    **({"details": e.details} if e.details else {}),
                }
                for e in self.events
            ],
        }
