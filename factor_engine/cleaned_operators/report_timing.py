# -*- coding: utf-8 -*-
"""Report disclosure timing / revision magnitude (2026-08 V3).

* ``report_filing_delay_surprise``  — robust z of the current filing delay
  against the firm's own historical delay distribution.  ``window`` is the
  number of *past filing events* (event clock, review R4-16) so quarterly
  filers are never starved of history (P1).
* ``report_revision_magnitude``     — ``(x_new - x_old)/(|x_old| + scale)`` with
  ``scale`` the historical median |x_old| over the last ``window`` revision
  events (P2 research).  ``current_period_id`` / ``prev_period_id`` must be
  supplied and equal — only same-ReportPeriod old/new disclosures are
  meaningful (review R4-84).  The preferred path is *source-layer
  materialisation* from real revision vintages; the runtime operator consumes
  the previously-published value as a supplied panel and never guesses it from
  the time series.

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
    """Robust z of the current filing delay against the past filing delays.

    ``window`` is the number of *past filing events* (event clock, review
    R4-16) — not trading days.  A quarterly filer issues ~4 filings/year, so a
    "last 60 trading days" window could never accumulate ``min_periods``
    disclosures and the factor was NaN forever.  Here the reference set is the
    last ``window`` rows where ``delay`` is finite, strictly before ``t``.
    """
    n = delay.shape[0]
    ne = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    event_idx = np.flatnonzero(np.isfinite(delay))
    for t in range(n):
        cur = delay[t]
        if not np.isfinite(cur):
            continue
        past = event_idx[event_idx < t]
        if past.size < mp:
            continue
        past_delays = delay[past[-ne:]]
        finite = past_delays[np.isfinite(past_delays)]
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

    ``window`` 是"过去披露事件数"（事件时钟，R4-16），不是交易日：季度财报每年仅
    约 4 次，若用"最近 60 交易日"窗口永远凑不够 5 次披露 → 长期 NaN。典型取值
    ``window`` ∈ {4, 8, 12}。只在该公司出现新的披露延迟值（``delay`` 为有限值）
    的日期输出；高 = 延迟显著晚于历史习惯，低 = 异常早。PIT 安全（只用严格过去事件）。
    """

    metadata = _metadata(
        "report_filing_delay_surprise",
        "披露延迟稳健 z（相对公司历史延迟分布；window=过去披露事件数）。",
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

def _revision_magnitude_series(
    x: np.ndarray,
    prev_x: np.ndarray,
    cur_period: np.ndarray,
    prev_period: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    n = x.shape[0]
    ne = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    rev_idx = np.flatnonzero(np.isfinite(prev_x))
    for t in range(n):
        cur = x[t]
        prev = prev_x[t]
        if not np.isfinite(cur) or not np.isfinite(prev):
            continue
        # R4-84: (x_new - x_old) is only meaningful for the SAME ReportPeriod;
        # the runtime cannot infer it, so it must be supplied and equal —
        # otherwise fail closed.
        if pd.isna(cur_period[t]) or pd.isna(prev_period[t]) or cur_period[t] != prev_period[t]:
            continue
        # Event clock (R4-16): the historical scale uses the last ``window``
        # revision events (rows with a finite prev_x), not the last trading days.
        past = rev_idx[rev_idx < t]
        if past.size < mp:
            continue
        finite = np.abs(prev_x[past[-ne:]])
        finite = finite[np.isfinite(finite)]
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
    |x_old| 的中位数（事件时钟：最近 ``window`` 个 revision event）。必须同时提供
    ``current_period_id`` / ``prev_period_id`` 并保证相等——只有同一 ReportPeriod
    的新旧披露才有意义（R4-84），不满足 → NaN（fail closed）。P2 / Research，
    首选在数据源层用真实修订历史 materialize，避免 AST 运行时扫描上百字段。
    """

    metadata = _metadata(
        "report_revision_magnitude",
        "报表修订幅度 (x_new-x_old)/(|x_old|+scale)，须同 ReportPeriod。",
        ["x", "prev_x", "current_period_id", "prev_period_id", "window", "min_periods"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        prev_x: pd.DataFrame,
        current_period_id: pd.DataFrame,
        prev_period_id: pd.DataFrame,
        window: int = 60,
        min_periods: int = 5,
        **_: Any,
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        pv = prev_x.to_numpy(dtype=float)
        cp = current_period_id.to_numpy()
        pp = prev_period_id.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _revision_magnitude_series(xv[:, c], pv[:, c], cp[:, c], pp[:, c], window, min_periods)
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"report_filing_delay_surprise"})
    _surface.extend_research_only({"report_revision_magnitude"})


_register_surface()
