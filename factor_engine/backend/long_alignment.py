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


def _compatible_time_keys(left, right):
    """Unify naive temporal representations without truncating precision."""
    import polars as pl
    a, b = left.collect_schema()[_TS], right.collect_schema()[_TS]
    if a == b:
        return left, right
    def naive(dtype):
        return dtype == pl.Date or (isinstance(dtype, pl.Datetime) and dtype.time_zone is None)
    if naive(a) and naive(b):
        units = [d.time_unit for d in (a,b) if isinstance(d,pl.Datetime)]
        unit = max(units, key=lambda u: {"ms":0,"us":1,"ns":2}[u])
        target = pl.Datetime(unit)
        return (left.with_columns(pl.col(_TS).cast(target, strict=True)),
                right.with_columns(pl.col(_TS).cast(target, strict=True)))
    return left, right


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
    left, right = _compatible_time_keys(left, right)
    left_keys = _key_set(left)
    right_keys = _key_set(right)
    # Validate each materialized key projection once. Re-collecting a lazy
    # map_groups subtree for uniqueness and then for equality executes the
    # factor twice. Python tuple sets also multiply full-panel memory usage.
    for side, keys in (("left", left_keys), ("right", right_keys)):
        duplicate = keys.is_duplicated()
        if duplicate.any():
            count = keys.filter(duplicate).unique().height
            raise AlignmentError(f"{context} {side}: 对齐契约违反：(ts, inst) 存在 {count} 组重复 key")
    import polars as pl
    try:
        missing_right = left_keys.join(right_keys, on=[_TS,_INST], how="anti", nulls_equal=True)
        missing_left = right_keys.join(left_keys, on=[_TS,_INST], how="anti", nulls_equal=True)
    except pl.exceptions.SchemaError as exc:
        raise AlignmentError(f"{context}: binary key schemas are not aligned: {exc}") from exc
    if missing_right.height or missing_left.height:
        def sample(keys):
            return [(repr(ts),repr(inst)) for ts,inst in keys.sort([_TS,_INST]).head(5).iter_rows()]

        prefix = f"{context}: " if context else ""
        raise AlignmentError(
            f"{prefix}binary key sets are not exactly aligned; "
            f"left_count={left_keys.height}, right_count={right_keys.height}, "
            f"missing_right={missing_right.height} sample={sample(missing_right)}, "
            f"missing_left={missing_left.height} sample={sample(missing_left)}"
        )


def anchor_left_join_binary(left: "pl.LazyFrame", right: "pl.LazyFrame", *, right_col: str = "_y") -> "pl.LazyFrame":
    """Join binary long inputs only after exact key-set validation."""
    left, right = _compatible_time_keys(left, right)
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
    left, mid = _compatible_time_keys(left, mid)
    left, right = _compatible_time_keys(left, right)
    left, mid = _compatible_time_keys(left, mid)
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
