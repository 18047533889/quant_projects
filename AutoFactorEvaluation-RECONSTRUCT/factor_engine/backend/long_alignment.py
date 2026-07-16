# -*- coding: utf-8
"""Long-table 索引对齐契约：shape-preserving join，禁止隐式删行。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

_TS = "ts"
_INST = "inst"
_VAL = "_v"


class AlignmentError(ValueError):
    """(ts, inst) 索引违反对齐契约。"""


def assert_unique_keys(lf: "pl.LazyFrame", *, context: str = "") -> None:
    """执行前校验 (ts, inst) 唯一；重复 key 直接报错。"""
    import polars as pl

    dup = (
        lf.group_by([_TS, _INST])
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    if dup:
        msg = f"对齐契约违反：({ _TS}, {_INST}) 存在 {dup} 组重复 key"
        if context:
            msg = f"{context}: {msg}"
        raise AlignmentError(msg)


def anchor_left_join_binary(left: "pl.LazyFrame", right: "pl.LazyFrame", *, right_col: str = "_y") -> "pl.LazyFrame":
    """以左操作数为 anchor 做 LEFT JOIN；右缺失 → NULL。"""
    assert_unique_keys(left, context="binary join left")
    assert_unique_keys(right, context="binary join right")
    return left.join(
        right.rename({_VAL: right_col}),
        on=[_TS, _INST],
        how="left",
    )


def anchor_left_join_triple(
    left: "pl.LazyFrame",
    mid: "pl.LazyFrame",
    right: "pl.LazyFrame",
) -> "pl.LazyFrame":
    """三元算子：左 anchor 串联 LEFT JOIN。"""
    return anchor_left_join_binary(left, mid, right_col="_ym").join(
        right.rename({_VAL: "_y"}),
        on=[_TS, _INST],
        how="left",
    )


def anchor_join_sql(left_sql: str, right_sql: str, *, left_alias: str = "l", right_alias: str = "r") -> str:
    """SQL anchor LEFT JOIN（左操作数保留全部 key）。"""
    return f"FROM ({left_sql}) {left_alias} LEFT JOIN ({right_sql}) {right_alias} USING (ts, inst)"


def anchor_join_value_group_sql(value_sql: str, group_sql: str) -> str:
    """截面/组内：value 为 anchor，group LEFT JOIN。"""
    return f"FROM ({value_sql}) x LEFT JOIN ({group_sql}) g USING (ts, inst)"
