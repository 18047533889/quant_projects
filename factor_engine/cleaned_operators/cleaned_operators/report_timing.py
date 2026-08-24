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

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12

# Audit #48: ``window`` is a REPORT/EVENT count, never trading bars.  Quarterly
# filers issue ~4 filings a year, so a "last 60 reports" window would silently
# span ~15 years.  Defaults are redesigned to a per-report-frequency grid
# (4/8/12/20 reports); the value remains free (no hard choices) so callers may
# pass any event count (e.g. 40) without being rejected.
_REPORT_WINDOW_SPEC = ParamSpec(dtype=int, min=2, default=8)


def _validate_indicator(arr: np.ndarray, canonical: str, name: str) -> None:
    """A filing/revision event indicator must be a strict boolean {0, 1} or NaN."""
    finite = arr[np.isfinite(arr)]
    bad = finite[(finite != 0.0) & (finite != 1.0)]
    if bad.size:
        raise ValueError(
            f"{canonical} requires {name} to be a strict boolean indicator "
            f"(0/1 or NaN); found non-boolean finite value {float(bad[0])!r}"
        )


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
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
        param_specs=param_specs or {},
    )


# ---------------------------------------------------------------------------
# report_filing_delay_surprise
# ---------------------------------------------------------------------------

def _filing_event_rows(delay: np.ndarray, filing_event: np.ndarray | None) -> np.ndarray:
    """Rows that are REAL filing events (audit #46).

    A filing event is NEVER inferred from "the field is finite" — a
    forward-filled daily panel has a finite value every day, so that test would
    manufacture a filing every single day.  Only two sources are accepted:

    * an explicit ``filing_event`` indicator (``1`` = a filing was observed), or
    * sparse VINTAGE rows, detected as a finite -> (missing) -> finite transition
      (the first finite value after a run of missing values).  A genuinely
      sparse vintage panel therefore works, while a forward-filled panel
      (finite every day) produces NO events and fails closed to NaN.
    """
    n = delay.shape[0]
    if filing_event is not None:
        fe = np.asarray(filing_event, dtype=float)
        return (fe == 1.0) & np.isfinite(delay)
    rows = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isfinite(delay[t]) and (t == 0 or not np.isfinite(delay[t - 1])):
            rows[t] = True
    return rows


def _delay_surprise_series(
    delay: np.ndarray,
    window: int,
    min_periods: int,
    filing_event: np.ndarray | None = None,
) -> np.ndarray:
    """Robust z of the current filing delay against the past filing delays.

    ``window`` is the number of *past filing events* (event clock, review
    R4-16) — not trading days.  A quarterly filer issues ~4 filings/year, so a
    "last 60 trading days" window could never accumulate ``min_periods``
    disclosures and the factor was NaN forever.  Here the reference set is the
    last ``window`` REAL filing events (see :func:`_filing_event_rows`),
    strictly before ``t``; the output fires only on a real filing row.
    """
    n = delay.shape[0]
    ne = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    ev_rows = _filing_event_rows(delay, filing_event)
    event_idx = np.flatnonzero(ev_rows)
    for t in range(n):
        if not ev_rows[t]:
            continue
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
    按报告频率取 4/8/12/20（audit #48）。只在 REAL filing 事件日输出
    （``filing_event`` 显式标记，或 ``delay`` 的稀疏 vintage 行，见 audit #46）——
    绝不在「字段为有限值」的每一天凭空造出 filing 事件；前向填充日面板（每天有限）
    不产生任何事件 → 输出恒 NaN（fail closed）。PIT 安全（只用严格过去事件）。
    """

    metadata = _metadata(
        "report_filing_delay_surprise",
        "披露延迟稳健 z（相对公司历史延迟分布；window=过去披露事件数）。",
        ["delay", "filing_event", "window", "min_periods"],
        unit="zscore",
        cost=2,
        param_specs={"window": _REPORT_WINDOW_SPEC},
    )

    def _calculate_series(
        self,
        delay: pd.DataFrame,
        filing_event: pd.DataFrame | None = None,
        window: int = 8,
        min_periods: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        dv = delay.to_numpy(dtype=float)
        fv = None
        if filing_event is not None:
            fv = filing_event.to_numpy(dtype=float)
            _validate_indicator(fv, "report_filing_delay_surprise", "filing_event")
        rows, cols = dv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            fe_c = None if fv is None else fv[:, c]
            out[:, c] = _delay_surprise_series(dv[:, c], window, min_periods, fe_c)
        return frame_like(delay, out)


# ---------------------------------------------------------------------------
# report_revision_magnitude (P2 research)
# ---------------------------------------------------------------------------

def _revision_rows(
    x: np.ndarray,
    prev_x: np.ndarray,
    revision_event: np.ndarray | None,
) -> np.ndarray:
    """Rows that are REAL revision events (audit #47).

    ``prev_x`` being finite does NOT mean a revision happened today — a
    forward-filled panel carries the previous vintage's value every single day.
    A revision is only accepted from:

    * an explicit ``revision_event`` indicator (``1`` = a revision was
      observed), or
    * sparse VINTAGE rows of ``x`` (finite after missing -> a new published
      value arrived), where ``prev_x`` is also finite on the same row.
    """
    n = x.shape[0]
    if revision_event is not None:
        re = np.asarray(revision_event, dtype=float)
        return (re == 1.0) & np.isfinite(x) & np.isfinite(prev_x)
    rows = np.zeros(n, dtype=bool)
    for t in range(n):
        if (
            np.isfinite(x[t])
            and np.isfinite(prev_x[t])
            and (t == 0 or not np.isfinite(x[t - 1]))
        ):
            rows[t] = True
    return rows


def _revision_magnitude_series(
    x: np.ndarray,
    prev_x: np.ndarray,
    cur_period: np.ndarray,
    prev_period: np.ndarray,
    window: int,
    min_periods: int,
    revision_event: np.ndarray | None = None,
) -> np.ndarray:
    n = x.shape[0]
    ne = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.full(n, np.nan)
    rev_rows = _revision_rows(x, prev_x, revision_event)
    rev_idx = np.flatnonzero(rev_rows)
    for t in range(n):
        if not rev_rows[t]:
            continue
        cur = x[t]
        prev = prev_x[t]
        # Audit #49: a magnitude is only meaningful when the pair is a typed
        # RevisionPair — the SAME ReportPeriod's prior vintage (cur_period ==
        # prev_period) with both revisions finite.  A bare ``prev_x`` is never
        # accepted as proof of a revision; without a matching event row this
        # loop does not even reach here.  Fail closed on any mismatch.
        if pd.isna(cur_period[t]) or pd.isna(prev_period[t]) or cur_period[t] != prev_period[t]:
            continue
        # Event clock (R4-16): the historical scale uses the last ``window``
        # REAL revision events, not the last trading days.
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
    |x_old| 的中位数（事件时钟：最近 ``window`` 个 REAL revision event）。修订
    只在显式 ``revision_event`` 标记日（或 ``x`` 的稀疏 vintage 行）发生
    （audit #47）——``prev_x`` 有限 ≠ 今天发生修订。每个修订日必须是 typed
    ``RevisionPair``：``current_period_id == prev_period_id``（同一 ReportPeriod）
    且两个值均有限（audit #49），不满足 → NaN（fail closed）。P2 / Research，
    首选在数据源层用真实修订历史 materialize。
    """

    metadata = _metadata(
        "report_revision_magnitude",
        "报表修订幅度 (x_new-x_old)/(|x_old|+scale)，须同 ReportPeriod + 显式修订事件。",
        ["x", "prev_x", "current_period_id", "prev_period_id", "revision_event", "window", "min_periods"],
        unit="ratio",
        cost=2,
        param_specs={"window": _REPORT_WINDOW_SPEC},
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        prev_x: pd.DataFrame,
        current_period_id: pd.DataFrame,
        prev_period_id: pd.DataFrame,
        revision_event: pd.DataFrame | None = None,
        window: int = 8,
        min_periods: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        pv = prev_x.to_numpy(dtype=float)
        cp = current_period_id.to_numpy()
        pp = prev_period_id.to_numpy()
        rv = None
        if revision_event is not None:
            rv = revision_event.to_numpy(dtype=float)
            _validate_indicator(rv, "report_revision_magnitude", "revision_event")
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            re_c = None if rv is None else rv[:, c]
            out[:, c] = _revision_magnitude_series(
                xv[:, c], pv[:, c], cp[:, c], pp[:, c], window, min_periods, re_c
            )
        return frame_like(x, out)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"report_filing_delay_surprise"})
    _surface.extend_research_only({"report_revision_magnitude"})


_register_surface()


# Audit #8: report-timing history is measured in REPORT observations, not
# trading bars — ``window=60`` means 60 filing/revision EVENTS (a ~15-year span
# for a quarterly filer).  A bar-window warmup cannot derive a report count, so
# the declared history is ``report_count`` (conservative full-history for DAG
# composition).  Default report grid is 8 reports (audit #48).
def _declare_stateful_contracts() -> None:
    from factor_engine.runtime.execution_contract import declare_stateful

    declare_stateful(
        "report_filing_delay_surprise",
        state_model="recursive",
        chunking="required_full_history",
        history_kind="report_count",
        history_count=8,
    )
    declare_stateful(
        "report_revision_magnitude",
        state_model="recursive",
        chunking="required_full_history",
        history_kind="report_count",
        history_count=8,
    )


_declare_stateful_contracts()
