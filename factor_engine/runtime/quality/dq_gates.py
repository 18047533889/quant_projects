# -*- coding: utf-8 -*-
"""因子产出数据质量门禁（轻量 DQ Gate）。

设计原则：不引入 Great Expectations 重量依赖；在 materialize / pipeline 出口做可配置校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DQThresholds:
    min_coverage: float = 0.05
    max_nan_ratio: float = 0.95
    max_inf_ratio: float = 0.0
    max_abs_value: float = 1e8
    min_rows: int = 1
    min_instruments_per_day: int = 1


@dataclass
class DQCheckResult:
    name: str
    passed: bool
    message: str
    value: float | int | None = None


@dataclass
class FactorDQReport:
    checks: list[DQCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "value": c.value,
                }
                for c in self.checks
            ],
        }


class FactorDQError(RuntimeError):
    """DQ 门禁未通过。"""

    def __init__(self, report: FactorDQReport):
        self.report = report
        failed = [c for c in report.checks if not c.passed]
        msgs = "; ".join(f"{c.name}: {c.message}" for c in failed)
        super().__init__(f"因子 DQ 校验失败: {msgs}")


def evaluate_factor_dq(
    result: pd.Series,
    *,
    thresholds: DQThresholds | None = None,
    preserve_invalid_rows: bool = False,
) -> FactorDQReport:
    """对 MultiIndex(timestamp, instrument) Series 做产出质量检查。

    ``preserve_invalid_rows=True`` 时：inf 不参与 inf_ratio 门禁（将在落盘时
    转为 ``is_valid=0``）；覆盖率/缺失率仅统计有限值。
    """
    th = thresholds or DQThresholds()
    checks: list[DQCheckResult] = []

    if not isinstance(result, pd.Series):
        checks.append(DQCheckResult("type", False, f"期望 Series，得到 {type(result).__name__}"))
        return FactorDQReport(checks=checks)

    n = len(result)
    checks.append(
        DQCheckResult(
            "min_rows",
            n >= th.min_rows,
            f"行数 {n} {'>=' if n >= th.min_rows else '<'} {th.min_rows}",
            n,
        )
    )

    if n == 0:
        return FactorDQReport(checks=checks)

    vals = result.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(vals)
    non_null = int(finite.sum())
    nan_ratio = 1.0 - non_null / n
    coverage = non_null / n

    checks.append(
        DQCheckResult(
            "coverage",
            coverage >= th.min_coverage,
            f"覆盖率 {coverage:.4f} {'>=' if coverage >= th.min_coverage else '<'} {th.min_coverage}",
            round(coverage, 6),
        )
    )
    checks.append(
        DQCheckResult(
            "nan_ratio",
            nan_ratio <= th.max_nan_ratio,
            f"缺失率 {nan_ratio:.4f} {'<=' if nan_ratio <= th.max_nan_ratio else '>'} {th.max_nan_ratio}",
            round(nan_ratio, 6),
        )
    )

    inf_count = int(np.isinf(vals).sum())
    inf_ratio = inf_count / n
    if preserve_invalid_rows:
        checks.append(
            DQCheckResult(
                "inf_ratio",
                True,
                f"inf 比例 {inf_ratio:.4f}（preserve_invalid_rows 模式跳过门禁）",
                round(inf_ratio, 6),
            )
        )
    else:
        checks.append(
            DQCheckResult(
                "inf_ratio",
                inf_ratio <= th.max_inf_ratio,
                f"inf 比例 {inf_ratio:.4f}",
                round(inf_ratio, 6),
            )
        )

    if non_null > 0:
        finite_vals = vals[finite]
        abs_max = float(np.nanmax(np.abs(finite_vals))) if len(finite_vals) else 0.0
        checks.append(
            DQCheckResult(
                "abs_max",
                abs_max <= th.max_abs_value,
                f"|value| max {abs_max:.4g} <= {th.max_abs_value:.4g}",
                abs_max,
            )
        )

    if isinstance(result.index, pd.MultiIndex) and result.index.nlevels >= 2:
        ts_level = result.index.get_level_values(0)
        inst_per_day = result.groupby(ts_level).apply(lambda s: s.notna().sum())
        min_inst = int(inst_per_day.min()) if len(inst_per_day) else 0
        checks.append(
            DQCheckResult(
                "min_instruments_per_day",
                min_inst >= th.min_instruments_per_day,
                f"每日最少非空标的 {min_inst} >= {th.min_instruments_per_day}",
                min_inst,
            )
        )

        if len(ts_level.unique()) >= 2:
            dup = result.index.duplicated().sum()
            checks.append(
                DQCheckResult(
                    "unique_keys",
                    dup == 0,
                    f"重复 (timestamp, instrument) 键: {dup}",
                    int(dup),
                )
            )

    return FactorDQReport(checks=checks)


def assert_factor_dq(
    result: pd.Series,
    *,
    thresholds: DQThresholds | None = None,
    raise_on_fail: bool = True,
    preserve_invalid_rows: bool = False,
) -> FactorDQReport:
    report = evaluate_factor_dq(
        result,
        thresholds=thresholds,
        preserve_invalid_rows=preserve_invalid_rows,
    )
    if raise_on_fail and not report.passed:
        raise FactorDQError(report)
    return report
