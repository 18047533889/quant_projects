# -*- coding: utf-8 -*-
"""PIT 标签层：forward return 与特征窗口隔离（mining / backtest 侧）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from runtime.pit_audit import PitSafetyError, audit_ir


@dataclass(frozen=True)
class LabelWindowSpec:
    """标签窗口规格。"""

    horizon_bars: int
    feature_lookback_bars: int
    gap_bars: int = 0

    @property
    def min_separation_bars(self) -> int:
        return max(0, int(self.gap_bars))

    def validate_no_overlap(self) -> None:
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
    """断言标签起始 bar 在特征窗口结束之后（含可选 gap）。"""
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
    """构造 forward return（仅用于标签；不可作为因子特征输入）。"""
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
    """挖掘/回测默认标签配置（与特征窗口显式隔离）。"""
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
    """校验标签 DSL：禁止负 lag / Lead 等前视算子进入标签流水线。"""
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
    """裁剪特征/标签 index，保证时间轴上无重叠区间。"""
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
