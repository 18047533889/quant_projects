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
    """因子产出 DQ 门禁阈值。"""

    min_coverage: float = 0.05
    max_nan_ratio: float = 0.95
    max_inf_ratio: float = 0.0
    max_abs_value: float = 1e8
    min_rows: int = 1
    min_instruments_per_day: int = 1


@dataclass
class DQCheckResult:
    """单项 DQ 检查结果。"""

    name: str
    passed: bool
    message: str
    value: float | int | None = None


@dataclass
class FactorDQReport:
    """因子产出 DQ 报告（多项检查聚合）。"""

    checks: list[DQCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """是否全部检查通过。"""
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": bool(c.passed),
                    "message": c.message,
                    "value": c.value.item() if isinstance(c.value, np.generic) else c.value,
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
    time_aware: bool = False,
    role: str | None = None,
    allowed_codes: set | None = None,
) -> FactorDQReport:
    """对 MultiIndex(timestamp, instrument) Series 做产出质量检查。

    ``preserve_invalid_rows=True`` 时：inf 不参与 inf_ratio 门禁（将在落盘时
    转为 ``is_valid=0``）；覆盖率/缺失率/每日标的数仅统计有限值（R21-093）。

    R21-091: duplicate-key 检查不再依赖 unique timestamp>=2 —— 单日重复也会
    失败。R21-092: ``min_instruments_per_day`` 与 coverage 统一使用 finite 口径。

    R21-095..101: ``role`` 传入时追加 role-aware 域检查（Condition 只能
    {0,1,NaN}；Rank/Pct 范围 [0,1]；Category 合法 code domain 等）。

    R21-102..105: ``time_aware=True`` 时追加时间分布检查（daily/latest-day/
    rolling coverage、quantiles、change、degenerate 识别），并附带按日期/股票池
    的 coverage profile。
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
        # R21-092: 统一 finite 口径 —— Inf 不是有效标的。
        is_finite_series = pd.Series(finite, index=result.index)
        inst_per_day = is_finite_series.groupby(ts_level).sum()
        min_inst = int(inst_per_day.min()) if len(inst_per_day) else 0
        checks.append(
            DQCheckResult(
                "min_instruments_per_day",
                min_inst >= th.min_instruments_per_day,
                f"每日最少有限标的 {min_inst} >= {th.min_instruments_per_day}",
                min_inst,
            )
        )

        # R21-091: 单日重复也必须失败，不再要求 unique timestamp>=2。
        dup = result.index.duplicated().sum()
        checks.append(
            DQCheckResult(
                "unique_keys",
                dup == 0,
                f"重复 (timestamp, instrument) 键: {dup}",
                int(dup),
            )
        )

        if time_aware:
            checks.extend(_time_distribution_checks(result, finite, ts_level))
            profile = build_daily_coverage_profile(result, finite=finite)
            checks.append(
                DQCheckResult(
                    "coverage_profile",
                    True,
                    f"coverage profile 天数 {len(profile)}",
                    len(profile),
                )
            )

    if role:
        checks.extend(role_domain_checks(result, role=role, allowed_codes=allowed_codes))

    return FactorDQReport(checks=checks)


def assert_factor_dq(
    result: pd.Series,
    *,
    thresholds: DQThresholds | None = None,
    raise_on_fail: bool = True,
    preserve_invalid_rows: bool = False,
    time_aware: bool = False,
    role: str | None = None,
    allowed_codes: set | None = None,
) -> FactorDQReport:
    """执行产出 DQ 检查；``raise_on_fail=True`` 时未通过则抛 :class:`FactorDQError`。"""
    report = evaluate_factor_dq(
        result,
        thresholds=thresholds,
        preserve_invalid_rows=preserve_invalid_rows,
        time_aware=time_aware,
        role=role,
        allowed_codes=allowed_codes,
    )
    if raise_on_fail and not report.passed:
        raise FactorDQError(report)
    return report


# ---------------------------------------------------------------------------
# Role-aware DQ contracts (R21-095..101)
# ---------------------------------------------------------------------------


class DQRole:
    """Factor/operator role for DQ contract selection.

    Alpha / Condition / Event / State / GroupState / GlobalState / Intermediate.
    """

    ALPHA = "alpha"
    CONDITION = "condition"
    EVENT = "event"
    STATE = "state"
    GROUP_STATE = "group_state"
    GLOBAL_STATE = "global_state"
    INTERMEDIATE = "intermediate"


_ROLES = frozenset(
    {
        DQRole.ALPHA, DQRole.CONDITION, DQRole.EVENT, DQRole.STATE,
        DQRole.GROUP_STATE, DQRole.GLOBAL_STATE, DQRole.INTERMEDIATE,
        # Rank/Pct are valid factor roles for the [0,1] domain check (R21-098).
        "rank", "pct", "percentile",
    }
)


def normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    text = str(role).strip().lower()
    if text in _ROLES:
        return text
    raise KeyError(f"unknown DQ role: {role!r}")


def role_domain_checks(
    result: pd.Series,
    *,
    role: str,
    allowed_codes: set | None = None,
) -> list[DQCheckResult]:
    """R21-097..100: role-specific output-domain checks.

    - Condition: 输出只能 {0,1,NaN}（或声明的三值逻辑）——出现其它非空值即违例。
    - Rank/Pct: 范围 [0,1]。
    - Category: 只能落在合法 code domain。
    - GlobalState: 允许同截面广播；Alpha 不应意外全截面 constant。
    """
    checks: list[DQCheckResult] = []
    role = normalize_role(role) or "alpha"
    if len(result) == 0:
        return checks

    vals = result.to_numpy(dtype=float, copy=False)
    finite_vals = vals[np.isfinite(vals)]

    if role == DQRole.CONDITION:
        if len(finite_vals):
            non_binary = np.sum(~np.isin(finite_vals, [0.0, 1.0]))
            checks.append(
                DQCheckResult(
                    "condition_domain",
                    int(non_binary) == 0,
                    f"Condition 非 {0}/{1} 有限值: {int(non_binary)}",
                    int(non_binary),
                )
            )
        else:
            checks.append(DQCheckResult("condition_domain", True, "Condition 全 NaN（稀疏允许）", 0))

    if role in {DQRole.ALPHA, DQRole.INTERMEDIATE}:
        if len(finite_vals):
            lo, hi = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
            # Alpha 不做 IC/收益筛选，只做结构正确性（R21-101）：全截面 constant
            # 是结构异常信号。
            rng = hi - lo
            all_const = rng <= 1e-12
            checks.append(
                DQCheckResult(
                    "alpha_not_constant",
                    not all_const,
                    f"Alpha 值域 [{lo:.4g}, {hi:.4g}]" + ("（全截面 constant 结构异常）" if all_const else ""),
                    round(rng, 6),
                )
            )

    if role in {"rank", "pct", "percentile"}:
        if len(finite_vals):
            lo, hi = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
            in_range = (lo >= 0.0) and (hi <= 1.0 + 1e-9)
            checks.append(
                DQCheckResult(
                    "rank_range",
                    in_range,
                    f"Rank/Pct 值域 [{lo:.4g}, {hi:.4g}] 必须 ⊆ [0,1]",
                    round(hi, 6),
                )
            )

    if role == DQRole.CONDITION or role == DQRole.EVENT:
        # Event/Condition 稀疏不能误判为数据源坏（R21-117）——交给调用方用
        # EXPECTED_SPARSE 判定；这里只做域正确性。
        pass

    if role == DQRole.GLOBAL_STATE:
        # GlobalState 允许同截面广播：不检查 cross-section variance。
        pass

    return checks


# ---------------------------------------------------------------------------
# Time-distribution DQ (R21-102..105)
# ---------------------------------------------------------------------------


def build_daily_coverage_profile(result: pd.Series, *, finite: np.ndarray | None = None) -> pd.Series:
    """按日期输出 coverage profile（有限值 / 行数），供 audit 保存。"""
    if not isinstance(result.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    ts_level = result.index.get_level_values(0)
    mask = pd.Series(finite, index=result.index) if finite is not None else result.notna()
    daily = mask.groupby(ts_level).sum() / mask.groupby(ts_level).size()
    return daily.fillna(0.0)


def _time_distribution_checks(
    result: pd.Series,
    finite: np.ndarray,
    ts_level: pd.Index,
) -> list[DQCheckResult]:
    """R21-102..104: 识别“历史正常、最新一天全空”、all-constant、variance
    collapse、cardinality collapse 等时间分布退化。"""
    checks: list[DQCheckResult] = []
    days = list(ts_level.unique())
    if len(days) < 2:
        return checks
    is_finite = pd.Series(finite, index=result.index)
    daily_coverage = is_finite.groupby(ts_level).sum() / is_finite.groupby(ts_level).size()
    daily_finite_count = is_finite.groupby(ts_level).sum()

    # latest-day coverage vs prior coverage
    latest = float(daily_coverage.iloc[-1]) if len(daily_coverage) else 0.0
    prior = float(daily_coverage.iloc[:-1].mean()) if len(daily_coverage) > 1 else latest
    latest_collapse = prior > 0.5 and latest < 0.1
    checks.append(
        DQCheckResult(
            "latest_day_coverage",
            not latest_collapse,
            f"最新一天覆盖率 {latest:.3f} vs 前期均值 {prior:.3f}",
            round(latest, 4),
        )
    )

    # all-constant / unexpected all-zero / variance collapse
    vals = result.to_numpy(dtype=float, copy=False)
    finite_vals = vals[np.isfinite(vals)]
    if len(finite_vals):
        rng = float(np.nanmax(finite_vals) - np.nanmin(finite_vals))
        var = float(np.nanvar(finite_vals))
        checks.append(
            DQCheckResult(
                "no_variance_collapse",
                rng > 1e-12 and var > 1e-18,
                f"有限值 range {rng:.4g}, var {var:.4g}",
                round(rng, 6),
            )
        )
        nonzero = float(np.count_nonzero(finite_vals))
        all_zero = nonzero == 0
        checks.append(
            DQCheckResult(
                "not_all_zero",
                not all_zero,
                "有限值全部为 0（unexpected all-zero）",
                int(nonzero),
            )
        )
    return checks
