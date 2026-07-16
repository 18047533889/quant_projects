# -*- coding: utf-8 -*-
"""PIT 标签层：forward return 与特征窗口隔离（mining / backtest 侧）。

本模块提供标签窗口规格（``LabelWindowSpec``）、特征/标签时间轴对齐、
forward return 构造，以及标签 DSL 的 PiT 安全审计（禁止前视算子进入标签流水线）。
标签可使用未来收益，但须与特征 IR 在 bar 维度显式隔离（``gap_bars``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from runtime.pit_audit import PitSafetyError, audit_ir


@dataclass(frozen=True)
class LabelWindowSpec:
    """标签窗口规格：预测 horizon、特征 lookback 与隔离 gap。"""

    horizon_bars: int
    feature_lookback_bars: int
    gap_bars: int = 0

    @property
    def min_separation_bars(self) -> int:
        """特征窗口结束与标签起始之间的最小 bar 间隔（``gap_bars`` 下界为 0）。"""
        return max(0, int(self.gap_bars))

    def validate_no_overlap(self) -> None:
        """校验 horizon / lookback 参数合法（不检查实际 index 重叠）。"""
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars 必须 > 0")
        if self.feature_lookback_bars < 0:
            raise ValueError("feature_lookback_bars 不能为负")


def assert_label_feature_no_overlap(
    *,
    feature_end_bar: int,
    label_start_bar: int,
    spec: LabelWindowSpec,
) -> None:
    """断言标签起始 bar 在特征窗口结束之后（含可选 gap）。

    Parameters
    ----------
    feature_end_bar : int
        特征窗口最后一根 bar 的索引。
    label_start_bar : int
        标签窗口第一根 bar 的索引。
    spec : LabelWindowSpec
        含 ``gap_bars`` 的窗口规格。

    Raises
    ------
    PitSafetyError
        ``label_start_bar < feature_end_bar + gap`` 时抛出。
    """
    spec.validate_no_overlap()
    required = feature_end_bar + spec.min_separation_bars
    if label_start_bar < required:
        raise PitSafetyError(
            [
                f"label_feature_overlap: feature_end={feature_end_bar}, "
                f"label_start={label_start_bar}, gap={spec.min_separation_bars}",
            ]
        )


def build_forward_return_series(
    close: pd.Series,
    *,
    horizon: int = 1,
) -> pd.Series:
    """构造 forward return（仅用于标签；不可作为因子特征输入）。

    Parameters
    ----------
    close : pd.Series
        MultiIndex ``(timestamp, instrument)`` 收盘价序列。
    horizon : int
        前瞻 bar 数（默认 1）。

    Returns
    -------
    pd.Series
        ``close[t+h] / close[t] - 1``，与输入同 index 结构。
    """
    h = max(1, int(horizon))
    panel = close.unstack("instrument")
    fwd = panel.shift(-h) / panel - 1.0
    return fwd.stack(future_stack=True)


def default_mining_label_config(
    *,
    horizon_bars: int = 5,
    feature_lookback_bars: int = 20,
    gap_bars: int = 1,
    return_column: str = "close",
) -> dict[str, Any]:
    """挖掘/回测默认标签配置（与特征窗口显式隔离）。

    Parameters
    ----------
    horizon_bars : int
        标签前瞻 bar 数（默认 5）。
    feature_lookback_bars : int
        特征最大 lookback（默认 20，仅写入元数据）。
    gap_bars : int
        特征结束与标签起始之间的隔离 bar 数（默认 1）。
    return_column : str
        收益计算所用价格列（默认 ``"close"``）。

    Returns
    -------
    dict[str, Any]
        含 ``label_formula``、``horizon_bars``、``gap_bars`` 等的 JSON 可序列化配置。
    """
    spec = LabelWindowSpec(
        horizon_bars=horizon_bars,
        feature_lookback_bars=feature_lookback_bars,
        gap_bars=gap_bars,
    )
    spec.validate_no_overlap()
    return {
        "schema_version": "factor_engine.label_pit.v1",
        "return_column": return_column,
        "horizon_bars": spec.horizon_bars,
        "feature_lookback_bars": spec.feature_lookback_bars,
        "gap_bars": spec.gap_bars,
        "label_formula": f"ts_pct({return_column}, {spec.horizon_bars})",
        "pit_note": (
            "标签使用未来收益；特征 IR 须通过 pit_audit，且 label_start >= feature_end + gap"
        ),
    }


def validate_label_formula_for_pit(formula: str, *, enforce: bool = True) -> dict[str, Any]:
    """校验标签 DSL：禁止负 lag / Lead 等前视算子进入标签流水线。

    Parameters
    ----------
    formula : str
        标签 DSL 公式字符串。
    enforce : bool
        为 ``True`` 时违规抛出 ``PitSafetyError``；否则仅写入 report。

    Returns
    -------
    dict[str, Any]
        含 ``ok``、``pit_safe``、``violations`` 等字段的审计报告。
    """
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer
    from runtime.pit_audit import assert_pit_safe

    expr = parse_expr(str(formula or "").strip())
    analysis = Analyzer().lower(expr)
    report = assert_pit_safe(
        analysis.ir,
        enforce=enforce,
        forbid_forward_fill=True,
    )
    return {
        "ok": report.passed,
        "formula": formula,
        "pit_safe": report.passed,
        "violations": list(report.violations),
    }


def align_feature_and_label_windows(
    feature_index: pd.Index,
    label_index: pd.Index,
    *,
    spec: LabelWindowSpec,
) -> tuple[pd.Index, pd.Index]:
    """裁剪特征/标签 index，保证时间轴上无重叠区间。

    Parameters
    ----------
    feature_index : pd.Index
        特征可用时间戳 index。
    label_index : pd.Index
        标签可用时间戳 index。
    spec : LabelWindowSpec
        含 ``gap_bars`` 的窗口规格。

    Returns
    -------
    tuple[pd.Index, pd.Index]
        ``(feature_index, trimmed_label_index)``；必要时裁剪标签侧。
    """
    spec.validate_no_overlap()
    if len(feature_index) == 0 or len(label_index) == 0:
        return feature_index, label_index

    feature_end = pd.Timestamp(feature_index.max())
    label_start = pd.Timestamp(label_index.min())
    min_label_start = feature_end + pd.Timedelta(days=spec.min_separation_bars)
    if label_start < min_label_start:
        trimmed_label = label_index[label_index >= min_label_start]
        return feature_index, trimmed_label
    return feature_index, label_index
