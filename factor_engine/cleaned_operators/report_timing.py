# -*- coding: utf-8 -*-
"""Report disclosure timing / revision magnitude (2026-08 V3).

* ``report_filing_delay_surprise``  — robust z of the current filing delay
  against the firm's own historical delay distribution (P1).
* ``report_revision_magnitude``     — ``(x_new - x_old)/(|x_old| + scale)`` with
  ``scale`` the historical median |x_old| (P2 research).  The preferred path is
  *source-layer materialisation* from real revision vintages; the runtime
  operator consumes the previously-published value as a supplied panel and never
  guesses it from the time series.

``delay`` / ``x`` / ``prev_x`` are supplied panels (filing delay in days;
current published value; previously published value).  Deterministic,
prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental_period",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "fundamental_period", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:fundamental",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# report_filing_delay_surprise
# ---------------------------------------------------------------------------

def _delay_surprise_series(delay: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = delay.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        past = delay[lo:t]
        cur = delay[t]
        if not np.isfinite(cur):
            continue
        finite = past[np.isfinite(past)]
        if finite.size < mp:
            continue
        med = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - med)))
        scale = mad if mad > _EPS else float(np.std(finite))
        if scale <= _EPS:
            continue
        out[t] = float((cur - med) / scale)
    return out


@register_operator(
    name="report_filing_delay_surprise",
    category="fundamental_period",
    business_category="fundamental_period",
    canonical="report_filing_delay_surprise",
    source="report_timing",
)
class ReportFilingDelaySurprise(SeriesOperator):
    """当前披露延迟相对公司历史延迟分布的稳健 z 得分。

    只在该公司出现新的披露延迟值（``delay`` 输入为有限值）的日期输出；高 = 延迟
    显著晚于历史习惯（披露风险信号），低 = 异常早。PIT 安全（只用严格过去窗口）。
    """

    metadata = _metadata(
        "report_filing_delay_surprise",
        "披露延迟稳健 z（相对公司历史延迟分布）。",
        ["delay", "window", "min_periods"],
        unit="zscore",
        cost=2,
    )

    def _calculate_series(
        self, delay: pd.DataFrame, window: int = 60, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        dv = delay.to_numpy(dtype=float)
        rows, cols = dv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _delay_surprise_series(dv[:, c], window, min_periods)
        return frame_like(delay, out)


# ---------------------------------------------------------------------------
# report_revision_magnitude (P2 research)
# ---------------------------------------------------------------------------

def _revision_magnitude_series(x: np.ndarray, prev_x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    n = x.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    for t in range(n):
        cur = x[t]
        prev = prev_x[t]
        if not np.isfinite(cur) or not np.isfinite(prev):
            continue
        lo = max(0, t - w)
        history = prev_x[lo:t]
        finite = np.abs(history[np.isfinite(history)])
        if finite.size < mp:
            continue
        scale = float(np.median(finite))
        denom = abs(prev) + scale + _EPS
        out[t] = float((cur - prev) / denom)
    return out


@register_operator(
    name="report_revision_magnitude",
    category="fundamental_period",
    business_category="fundamental_period",
    canonical="report_revision_magnitude",
    source="report_timing",
    status="experimental",
)
class ReportRevisionMagnitude(SeriesOperator):
    """报表修订幅度 ``(x_new - x_old)/(|x_old| + scale)``（Research）。

    ``prev_x`` 是来源层提供的上一披露值（绝不能从时序猜）；``scale`` 为历史
    |x_old| 的中位数。只对同一 ReportPeriod 的新旧披露有意义。P2 / Research，
    首选在数据源层用真实修订历史 materialize，避免 AST 运行时扫描上百字段。
    """

    metadata = _metadata(
        "report_revision_magnitude",
        "报表修订幅度 (x_new-x_old)/(|x_old|+scale)。",
        ["x", "prev_x", "window", "min_periods"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, x: pd.DataFrame, prev_x: pd.DataFrame, window: int = 60, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        pv = prev_x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _revision_magnitude_series(xv[:, c], pv[:, c], window, min_periods)
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"report_filing_delay_surprise"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"report_revision_magnitude"}
    )


_register_surface()
