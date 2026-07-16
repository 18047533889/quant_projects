# -*- coding: utf-8
"""排序、重复键与输出稳定顺序契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OrderingSpec:
    """Long-table 主键与 window 排序键。"""

    primary_key: tuple[str, ...] = ("ts", "inst")
    window_order_by: tuple[str, ...] = ("ts",)
    tie_breaker: tuple[str, ...] = ("row_seq",)
    duplicate_key_policy: str = "error"
    output_sort_key: tuple[str, ...] = ("ts", "inst")


ORDERING_SPEC = OrderingSpec()

ORDER_DEPENDENT_OPS: frozenset[str] = frozenset(
    """
    delay ts_delta ts_pct ffill bfill
    cum_sum cum_max cum_min cum_prod cum_delta
    expanding_sum expanding_mean expanding_std expanding_rank count
    ts_argmax ts_argmin ts_rank
    ema ema ewm_std ewm_var
    first last
    """.split()
)


def production_output_sorted_by_ts_inst() -> bool:
    """PolarsLong collect 后须 ``sort([ts, inst])``。"""
    return True


def duplicate_ts_inst_keys_forbidden() -> bool:
    return ORDERING_SPEC.duplicate_key_policy == "error"


def window_order_columns(*, has_row_seq: bool = False) -> tuple[str, ...]:
    if has_row_seq:
        return ("ts", "row_seq")
    return ORDERING_SPEC.window_order_by


def order_dependent_op(canon: str) -> bool:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return name in ORDER_DEPENDENT_OPS
