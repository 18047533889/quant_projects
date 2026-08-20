# -*- coding: utf-8
"""Cumulative/expanding 增量 state 契约（第二阶段 production）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IncrementalTier = Literal["batch_only", "stateful_incremental"]


@dataclass(frozen=True)
class CumulativeStateSpec:
    tier: IncrementalTier = "batch_only"
    state_fields: tuple[str, ...] = ("last_ts", "count", "sum", "mean", "m2")


CUMULATIVE_STATE_OPS: dict[str, CumulativeStateSpec] = {
    "cum_sum": CumulativeStateSpec(),
    "cum_max": CumulativeStateSpec(),
    "cum_min": CumulativeStateSpec(),
    "cum_prod": CumulativeStateSpec(),
    "expanding_mean": CumulativeStateSpec(),
    "expanding_std": CumulativeStateSpec(),
    "expanding_sum": CumulativeStateSpec(),
}


def cumulative_batch_only_phase1() -> bool:
    return CumulativeStateSpec().tier == "batch_only"
