# -*- coding: utf-8 -*-
"""输入侧数据质量检查：因子执行前校验依赖列可用性。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InputDQThresholds:
    min_rows: int = 1
    min_non_null_ratio: float = 0.01
    min_instruments: int = 1
    max_inf_ratio: float = 0.0


@dataclass
class InputColumnReport:
    column: str
    passed: bool
    row_count: int
    non_null_ratio: float
    instrument_count: int
    message: str


@dataclass
class InputDQReport:
    columns: list[InputColumnReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "columns": [
                {
                    "column": c.column,
                    "passed": c.passed,
                    "row_count": c.row_count,
                    "non_null_ratio": c.non_null_ratio,
                    "instrument_count": c.instrument_count,
                    "message": c.message,
                }
                for c in self.columns
            ],
        }


class InputDQError(RuntimeError):
    def __init__(self, report: InputDQReport):
        self.report = report
        failed = [c.column for c in report.columns if not c.passed]
        super().__init__(f"输入 DQ 未通过，列: {', '.join(failed)}")


def _series_stats(series: pd.Series) -> tuple[int, float, int]:
    n = len(series)
    if n == 0:
        return 0, 0.0, 0
    non_null = int(series.notna().sum())
    ratio = non_null / n
    inst = 0
    if isinstance(series.index, pd.MultiIndex) and series.index.nlevels >= 2:
        inst = int(series.index.get_level_values(-1).nunique())
    return n, ratio, inst


def evaluate_input_columns(
    data_source: Any,
    columns: set[str] | list[str],
    *,
    thresholds: InputDQThresholds | None = None,
) -> InputDQReport:
    """加载依赖列并检查行数/覆盖率/标的数。"""
    th = thresholds or InputDQThresholds()
    cols = sorted(set(columns))
    reports: list[InputColumnReport] = []

    load_columns = getattr(data_source, "load_columns", None)
    fetched: dict[str, Any] = {}
    missing_after_batch: list[str] = list(cols)
    if callable(load_columns) and len(cols) > 1:
        try:
            fetched = load_columns(cols)
            missing_after_batch = [n for n in cols if n not in fetched]
        except Exception:
            fetched = {}
            missing_after_batch = list(cols)

    for name in cols:
        try:
            if name in fetched:
                series = fetched[name]
            elif name in missing_after_batch:
                series = data_source.load_column(name)
            else:
                continue
        except Exception as exc:
            reports.append(
                InputColumnReport(
                    column=name,
                    passed=False,
                    row_count=0,
                    non_null_ratio=0.0,
                    instrument_count=0,
                    message=f"加载失败: {exc}",
                )
            )
            continue

        if not isinstance(series, pd.Series):
            reports.append(
                InputColumnReport(
                    column=name,
                    passed=False,
                    row_count=0,
                    non_null_ratio=0.0,
                    instrument_count=0,
                    message=f"期望 Series，得到 {type(series).__name__}",
                )
            )
            continue

        row_count, non_null_ratio, inst_count = _series_stats(series)
        inf_ratio = 0.0
        if row_count > 0:
            numeric = pd.to_numeric(series, errors="coerce")
            inf_ratio = float(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) / float(
                row_count
            )

        passed = True
        msgs: list[str] = []
        if row_count < th.min_rows:
            passed = False
            msgs.append(f"行数 {row_count} < {th.min_rows}")
        if non_null_ratio < th.min_non_null_ratio:
            passed = False
            msgs.append(f"非空率 {non_null_ratio:.4f} < {th.min_non_null_ratio}")
        if inst_count < th.min_instruments:
            passed = False
            msgs.append(f"标的数 {inst_count} < {th.min_instruments}")
        if inf_ratio > th.max_inf_ratio:
            passed = False
            msgs.append(f"inf 比例 {inf_ratio:.4f} > {th.max_inf_ratio}")

        reports.append(
            InputColumnReport(
                column=name,
                passed=passed,
                row_count=row_count,
                non_null_ratio=non_null_ratio,
                instrument_count=inst_count,
                message="OK" if passed else "; ".join(msgs),
            )
        )

    return InputDQReport(columns=reports)


def assert_input_dq(
    data_source: Any,
    columns: set[str] | list[str],
    *,
    thresholds: InputDQThresholds | None = None,
    raise_on_fail: bool = True,
) -> InputDQReport:
    report = evaluate_input_columns(data_source, columns, thresholds=thresholds)
    if not report.passed and raise_on_fail:
        raise InputDQError(report)
    return report
