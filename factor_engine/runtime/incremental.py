# -*- coding: utf-8 -*-
"""Incremental factor production with watermark and causal history contracts."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import pandas as pd

from factor_engine.cleaned_operators.operator_policy import effective_lookback
from factor_engine.runtime.execution_contract import (
    ExecutionContract,
    FULL_HISTORY_LOOKBACK_SENTINEL,
    HistoryRequirement,
    is_full_history_lookback,
)
from factor_engine.storage.time_window import (
    resolve_incremental_window_for_bar_freq,
    slice_series_time_window,
)

_PENDING_INCREMENTAL_HISTORY: ContextVar[dict[str, str] | None] = ContextVar(
    "factor_engine_pending_incremental_history", default=None
)
_INCREMENTAL_HISTORY_CERTIFICATE: ContextVar[bool] = ContextVar(
    "factor_engine_incremental_history_certificate", default=False
)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _stage_incremental_history_contract(plan: "IncrementalPlan") -> None:
    """Stage expected window bounds; do not certify until narrowing occurs."""
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
    if (
        plan.is_full_run
        or plan.full_history_required
        or plan.load_start is None
        or plan.output_start is None
    ):
        _PENDING_INCREMENTAL_HISTORY.set(None)
        return
    _PENDING_INCREMENTAL_HISTORY.set(
        {
            "factor_id": plan.factor_id,
            "load_start": _iso(plan.load_start) or "",
            "load_end": _iso(plan.load_end) or "",
            "output_start": _iso(plan.output_start) or "",
            "output_end": _iso(plan.output_end) or "",
        }
    )


def certify_narrowed_incremental_window(
    *,
    start_date: Any,
    end_date: Any,
) -> bool:
    """Certify only when the actual narrowed source matches the staged plan."""
    pending = _PENDING_INCREMENTAL_HISTORY.get()
    if not pending:
        return False
    actual_start = _iso(start_date) or ""
    actual_end = _iso(end_date) or ""
    if actual_start != pending["load_start"] or actual_end != pending["load_end"]:
        _PENDING_INCREMENTAL_HISTORY.set(None)
        _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
        return False
    _PENDING_INCREMENTAL_HISTORY.set(None)
    _INCREMENTAL_HISTORY_CERTIFICATE.set(True)
    return True


def clear_incremental_history_contract() -> None:
    _PENDING_INCREMENTAL_HISTORY.set(None)
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)


def consume_incremental_history_certificate() -> bool:
    """Consume and clear the current task's one-shot certificate."""
    value = bool(_INCREMENTAL_HISTORY_CERTIFICATE.get())
    _INCREMENTAL_HISTORY_CERTIFICATE.set(False)
    return value


@dataclass(frozen=True)
class IncrementalPlan:
    factor_id: str
    lookback_bars: int
    load_start: pd.Timestamp | None
    load_end: pd.Timestamp | None
    output_start: pd.Timestamp | None
    output_end: pd.Timestamp | None
    watermark_end: str | None
    is_full_run: bool
    factor_freq: str | None = None
    source_bar_freq: str | None = None
    window_mode: str = "daily"
    full_history_required: bool = False
    # P0-01 / P2-01: the two halves of the temporal dependency + state contract.
    # ``backward_history`` = how many source bars are read before ``output_start``
    # (warm-up).  ``forward_impact`` = how many future output bars a changed
    # source bar affects (None = unbounded: recursive/stateful/event-clock
    # propagation to the end of the series).  ``state_requirement`` /
    # ``checkpoint_requirement`` mirror ``ExecutionContract``.
    backward_history: int = 0
    forward_impact: int | None = 0
    state_requirement: str = "stateless"
    checkpoint_requirement: str | None = None
    history_kind: str = "finite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "lookback_bars": self.lookback_bars,
            "load_start": None if self.load_start is None else self.load_start.isoformat(),
            "load_end": None if self.load_end is None else self.load_end.isoformat(),
            "output_start": None if self.output_start is None else self.output_start.isoformat(),
            "output_end": None if self.output_end is None else self.output_end.isoformat(),
            "watermark_end": self.watermark_end,
            "is_full_run": self.is_full_run,
            "factor_freq": self.factor_freq,
            "source_bar_freq": self.source_bar_freq,
            "window_mode": self.window_mode,
            "full_history_required": self.full_history_required,
            "backward_history": self.backward_history,
            "forward_impact": self.forward_impact,
            "state_requirement": self.state_requirement,
            "checkpoint_requirement": self.checkpoint_requirement,
            "history_kind": self.history_kind,
        }


def build_incremental_plan(
    *,
    factor_id: str,
    analysis_lookback: int,
    watermark: dict | None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    recompute_tail_bars: int | None = None,
    market: str | None = None,
    calendar=None,
    factor_freq: str | None = None,
    source_bar_freq: str | None = None,
    history: HistoryRequirement | None = None,
    execution: ExecutionContract | None = None,
    forward_impact: int | None = None,
) -> IncrementalPlan:
    """Build an incremental plan, forcing full replay for recursive factors.

    P0-01/P0-05/P2-01: the full-history decision comes from a real
    ``HistoryRequirement`` / ``ExecutionContract`` when provided (never the 1e9
    integer sentinel).  ``forward_impact`` (future output bars a changed source
    bar affects, ``None`` = unbounded) is persisted onto the plan so the
    scheduler/executor know the output window must reach forward past
    ``affected_end``.
    """
    from factor_engine.storage.trading_calendar import get_trading_calendar

    raw_lookback = int(analysis_lookback)
    if history is not None:
        full_history_required = bool(history.is_full_history)
    elif execution is not None:
        full_history_required = bool(execution.requires_full_history)
    else:
        # Legacy serialized-analysis compatibility: the 1e9 sentinel means
        # full-history replay (the sentinel is NOT a window size — see
        # ``execution_contract.FULL_HISTORY_LOOKBACK_SENTINEL``).
        full_history_required = is_full_history_lookback(raw_lookback)
    if history is not None and not history.is_full_history:
        # P0-05: the HistoryRequirement is authoritative for the warm-up rows
        # too — an overriding finite requirement wins over a raw legacy sentinel
        # (which is not a real window size and must never reach the calendar).
        finite_lookback = max(0, int(history.rows))
    elif is_full_history_lookback(raw_lookback):
        finite_lookback = 0  # the 1e9 sentinel is a marker, never a window size
    else:
        finite_lookback = 0 if full_history_required else max(0, raw_lookback)
    state_model = (
        execution.state_model if execution is not None else (
            "recursive" if full_history_required else "stateless"
        )
    )
    checkpoint_schema = execution.checkpoint_schema if execution is not None else None
    history_kind = history.kind if history is not None else (
        "full_history" if full_history_required else "finite"
    )
    fwd = None if (full_history_required or forward_impact is None) else max(0, int(forward_impact))

    watermark_end: str | None = None
    if since:
        watermark_end = str(since)
    elif watermark is not None:
        raw = watermark.get("end_date")
        if raw:
            watermark_end = str(raw)

    window_watermark = None if full_history_required else watermark_end
    load_lookback = effective_lookback(
        finite_lookback,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        extra=lookback_extra,
    )
    tail_bars = (
        max(1, finite_lookback + 1)
        if recompute_tail_bars is None
        else max(0, int(recompute_tail_bars))
    )

    calendar = calendar if calendar is not None else get_trading_calendar(market)
    window = resolve_incremental_window_for_bar_freq(
        watermark_end=window_watermark,
        lookback_bars=load_lookback,
        since=None,
        end_date=end_date,
        recompute_tail_bars=tail_bars,
        calendar=calendar,
        bar_freq=source_bar_freq,
    )
    window_mode = str(window.get("window_mode") or "daily")
    is_full = full_history_required or watermark_end is None

    plan = IncrementalPlan(
        factor_id=factor_id,
        lookback_bars=load_lookback,
        load_start=window["load_start"],
        load_end=window["load_end"],
        output_start=window["output_start"],
        output_end=window["output_end"],
        watermark_end=watermark_end,
        is_full_run=is_full,
        factor_freq=factor_freq,
        source_bar_freq=source_bar_freq,
        window_mode="full_history" if full_history_required else window_mode,
        full_history_required=full_history_required,
        backward_history=finite_lookback,
        forward_impact=fwd,
        state_requirement=state_model,
        checkpoint_requirement=checkpoint_schema,
        history_kind=history_kind,
    )
    _stage_incremental_history_contract(plan)
    return plan


def slice_factor_result_for_incremental(
    result: pd.Series,
    plan: IncrementalPlan,
) -> pd.Series:
    if plan.is_full_run or plan.output_start is None:
        return result
    return slice_series_time_window(
        result,
        start=plan.output_start,
        end=plan.output_end,
        exclusive_start=False,
    )
