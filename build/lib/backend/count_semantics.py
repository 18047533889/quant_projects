# -*- coding: utf-8
"""Count 类算子语义拆分（避免单一模糊 count）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CountKind = Literal[
    "count_rows",
    "count_non_null",
    "count_finite",
    "count_true",
    "rolling_count_non_null",
    "expanding_count_non_null",
]


@dataclass(frozen=True)
class CountOpSpec:
    kind: CountKind
    null_counts: bool = False
    nan_counts: bool = False
    inf_counts: bool = False


COUNT_OP_SPECS: dict[str, CountOpSpec] = {
    "count": CountOpSpec(kind="expanding_count_non_null", null_counts=False),
    "count_non_null": CountOpSpec(kind="expanding_count_non_null"),
    "count_finite": CountOpSpec(kind="expanding_count_non_null", inf_counts=False),
    "count_rows": CountOpSpec(kind="count_rows", null_counts=True, nan_counts=True),
    "rolling_count_non_null": CountOpSpec(kind="rolling_count_non_null"),
}


def count_op_spec_for(canon: str) -> CountOpSpec:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return COUNT_OP_SPECS.get(name, CountOpSpec(kind="count_non_null"))


def canonical_count_is_expanding_non_null() -> bool:
    return count_op_spec_for("count").kind == "expanding_count_non_null"
