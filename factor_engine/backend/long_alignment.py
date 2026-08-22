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


def _key_set(lf: "pl.LazyFrame"):
    """Materialize only the binary alignment keys for an exact-set check."""
    return lf.select([_TS, _INST]).collect()


def assert_exact_key_set(
    left: "pl.LazyFrame", right: "pl.LazyFrame", *, context: str = ""
) -> None:
    """Require identical, unique ``(ts, inst)`` key sets in both operands.

    A left join is not sufficient for binary long-panel arithmetic: it can
    silently turn an absent right observation into NULL.  Keep this check
    explicit and small (only keys are collected), while reporting directional
    counts and representative keys to make bad inputs actionable.
    """
    assert_unique_keys(left, context=f"{context} left" if context else "left")
    assert_unique_keys(right, context=f"{context} right" if context else "right")
    left_keys = _key_set(left)
    right_keys = _key_set(right)
    left_set = set(zip(left_keys[_TS].to_list(), left_keys[_INST].to_list()))
    right_set = set(zip(right_keys[_TS].to_list(), right_keys[_INST].to_list()))
    missing_right = left_set - right_set
    missing_left = right_set - left_set
    if missing_right or missing_left:
        def sample(keys):
            return sorted((repr(ts), repr(inst)) for ts, inst in keys)[:5]

        prefix = f"{context}: " if context else ""
        raise AlignmentError(
            f"{prefix}binary key sets are not exactly aligned; "
            f"left_count={len(left_set)}, right_count={len(right_set)}, "
            f"missing_right={len(missing_right)} sample={sample(missing_right)}, "
            f"missing_left={len(missing_left)} sample={sample(missing_left)}"
        )


def anchor_left_join_binary(left: "pl.LazyFrame", right: "pl.LazyFrame", *, right_col: str = "_y") -> "pl.LazyFrame":
    """Join binary long inputs only after exact key-set validation."""
    assert_exact_key_set(left, right, context="binary join")
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
    """Join three long inputs only after exact key-set validation."""
    assert_exact_key_set(left, mid, context="triple join left/mid")
    assert_exact_key_set(left, right, context="triple join left/right")
    return left.join(
        mid.rename({_VAL: "_ym"}),
        on=[_TS, _INST],
        how="left",
    ).join(
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
