# -*- coding: utf-8
"""累计/扩展窗口契约：cum_* / expanding_* / count。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CountKind = Literal["expanding_non_null_count"]
CumDeltaKind = Literal["delta_from_first_valid"]
CumResetPolicy = Literal["per_instrument_full_history"]


@dataclass(frozen=True)
class CountSpec:
    """``count(x)`` = 按 instrument 累计非 NULL 行数（非总行数）。"""

    kind: CountKind = "expanding_non_null_count"


@dataclass(frozen=True)
class CumDeltaSpec:
    """``cum_delta(x)`` = x - first_valid(x) per instrument。"""

    kind: CumDeltaKind = "delta_from_first_valid"
    anchor: Literal["first_non_null"] = "first_non_null"


@dataclass(frozen=True)
class CumProdSpec:
    """``cum_prod``：NULL 跳过；Inf/overflow 由 backend 浮点语义决定。"""

    null_skips: bool = True
    overflow_policy: Literal["propagate_inf", "to_null"] = "propagate_inf"


@dataclass(frozen=True)
class CumResetSpec:
    """日频第一阶段：仅 per-instrument 全历史累计，不按 session/年度重置。"""

    policy: CumResetPolicy = "per_instrument_full_history"


COUNT_SPEC = CountSpec()
CUM_DELTA_SPEC = CumDeltaSpec()
CUM_PROD_SPEC = CumProdSpec()
CUM_RESET_SPEC = CumResetSpec()


def count_is_expanding_non_null() -> bool:
    return COUNT_SPEC.kind == "expanding_non_null_count"


def cum_delta_from_first_valid() -> bool:
    return CUM_DELTA_SPEC.kind == "delta_from_first_valid"


def cum_reset_per_instrument_only() -> bool:
    return CUM_RESET_SPEC.policy == "per_instrument_full_history"


def cum_current_row_null_outputs_null() -> bool:
    """``cum_*`` / ``expanding_*``（除 ``count``）：当前行输入 NULL → 输出行 NULL。"""
    return True
