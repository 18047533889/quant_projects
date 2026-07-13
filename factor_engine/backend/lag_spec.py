# -*- coding: utf-8
"""Lag/Delay 语义：行滞后 vs 时间滞后。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LagKind = Literal["row_delay", "trading_day_delay", "calendar_day_delay", "previous_valid"]


@dataclass(frozen=True)
class LagSpec:
    kind: LagKind
    requires_trading_day_aligned_input: bool = False


# 第一阶段：delay/ts_delta/ts_pct 均认证为 row_delay
PHASE1_LAG_SPECS: dict[str, LagSpec] = {
    "delay": LagSpec(kind="row_delay", requires_trading_day_aligned_input=True),
    "delay": LagSpec(kind="row_delay", requires_trading_day_aligned_input=True),
    "ts_delta": LagSpec(kind="row_delay", requires_trading_day_aligned_input=True),
    "ts_pct": LagSpec(kind="row_delay", requires_trading_day_aligned_input=True),
}

FUTURE_LAG_ALIASES: dict[str, LagKind] = {
    "row_delay": "row_delay",
    "trading_day_delay": "trading_day_delay",
    "calendar_day_delay": "calendar_day_delay",
    "previous_valid": "previous_valid",
}


def lag_spec_for(canon: str) -> LagSpec:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PHASE1_LAG_SPECS.get(name, LagSpec(kind="row_delay"))


def delay_is_row_delay() -> bool:
    return lag_spec_for("delay").kind == "row_delay"
