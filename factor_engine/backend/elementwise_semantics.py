# -*- coding: utf-8
"""元素级算子语义：max/min NULL、比较 NULL 传播、protected 域裁剪。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd


def comparison_null_propagates() -> bool:
    """gt/lt/eq/ge/le/ne：任一输入 NULL/NaN → 输出 NULL。"""
    return True


def protected_log_null_preserved() -> bool:
    """protected_log：NULL 保持 NULL，仅对有限值做 domain clip。"""
    return True


def protected_div_null_preserved() -> bool:
    """protected_div：NULL 输入保持 NULL，仅 |denom|<=eps → default。"""
    return True


def max_horizontal_polars(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    """maximum：任一 NULL → NULL（对齐 DuckDB GREATEST）。"""
    import polars as pl

    return (
        pl.when(left.is_null() | right.is_null())
        .then(None)
        .otherwise(pl.max_horizontal(left, right))
    )


def min_horizontal_polars(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    """minimum：任一 NULL → NULL。"""
    import polars as pl

    return (
        pl.when(left.is_null() | right.is_null())
        .then(None)
        .otherwise(pl.min_horizontal(left, right))
    )


def _both_finite(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    import polars as pl

    return left.is_not_null() & right.is_not_null() & ~left.is_nan() & ~right.is_nan()


def compare_polars(op: str, left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    """比较算子：有效输入 → 0/1；NULL/NaN → NULL。"""
    import polars as pl

    cmp_map = {
        "gt": left > right,
        "lt": left < right,
        "eq": left == right,
        "ge": left >= right,
        "le": left <= right,
        "ne": left != right,
    }
    cond = cmp_map[op]
    valid = _both_finite(left, right)
    return pl.when(~valid).then(None).when(cond).then(1.0).otherwise(0.0)


def compare_sql(op: str, left_col: str, right_col: str, *, dialect_is_clickhouse: bool = False) -> str:
    """比较 SQL：NULL/NaN → NULL。"""
    sym = {"gt": ">", "lt": "<", "eq": "=", "ge": ">=", "le": "<=", "ne": "<>"}[op]
    if dialect_is_clickhouse and op == "ne":
        sym = "!="
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    valid = (
        f"({left_col} IS NOT NULL AND {right_col} IS NOT NULL "
        f"AND NOT {isnan_fn}({left_col}) AND NOT {isnan_fn}({right_col}))"
    )
    return f"CASE WHEN NOT ({valid}) THEN NULL WHEN {left_col} {sym} {right_col} THEN 1.0 ELSE 0.0 END"


def protected_log_polars(val: "pl.Expr", *, eps: float) -> "pl.Expr":
    """log(max(x, eps))，NULL 保持 NULL。"""
    import polars as pl

    log_eps = pl.lit(eps).log()
    return (
        pl.when(val.is_null())
        .then(None)
        .when(val <= eps)
        .then(log_eps)
        .otherwise(val.log())
    )


def protected_log_sql(value_col: str, *, eps: float, ln_fn: str = "LN") -> str:
    lit = repr(float(eps))
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {value_col} <= {lit} THEN {ln_fn}({lit}) "
        f"ELSE {ln_fn}({value_col}) END"
    )


def protected_div_polars(numer: "pl.Expr", denom: "pl.Expr", *, eps: float, default: float) -> "pl.Expr":
    """|denom|<=eps → default；NULL 输入 → NULL。"""
    import polars as pl

    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(None)
        .when(denom.abs() <= eps)
        .then(default)
        .otherwise(numer / denom)
    )


def protected_div_sql(
    left_col: str,
    right_col: str,
    *,
    eps: float,
    default: float,
    abs_fn: str = "abs",
) -> str:
    lit_eps = repr(float(eps))
    lit_def = repr(float(default))
    return (
        f"CASE WHEN {left_col} IS NULL OR {right_col} IS NULL THEN NULL "
        f"WHEN {abs_fn}({right_col}) <= {lit_eps} THEN {lit_def} "
        f"ELSE {left_col} / {right_col} END"
    )


def div_or_default_polars(numer: "pl.Expr", denom: "pl.Expr", *, eps: float, default: float) -> "pl.Expr":
    """|denom|<=eps 或 NULL 输入 → default（旧 protected_div 填充语义）。"""
    import polars as pl

    return (
        pl.when(denom.abs() <= eps)
        .then(default)
        .when(numer.is_null() | denom.is_null())
        .then(default)
        .otherwise(numer / denom)
    )


def div_or_default_sql(
    left_col: str,
    right_col: str,
    *,
    eps: float,
    default: float,
    abs_fn: str = "abs",
) -> str:
    lit_eps = repr(float(eps))
    lit_def = repr(float(default))
    core = (
        f"CASE WHEN {abs_fn}({right_col}) <= {lit_eps} THEN {lit_def} "
        f"ELSE {left_col} / {right_col} END"
    )
    return f"COALESCE({core}, {lit_def})"


def log_fill_invalid_polars(val: "pl.Expr", *, eps: float) -> "pl.Expr":
    """NULL/非法域 → log(epsilon)（旧 protected_log 填充语义）。"""
    import polars as pl

    log_eps = pl.lit(eps).log()
    return pl.when(val.is_null() | (val <= eps)).then(log_eps).otherwise(val.log())


def log_fill_invalid_sql(value_col: str, *, eps: float, ln_fn: str = "LN") -> str:
    lit = repr(float(eps))
    return (
        f"{ln_fn}(CASE WHEN {value_col} IS NULL OR {value_col} <= {lit} "
        f"THEN {lit} ELSE {value_col} END)"
    )


def is_infinite_polars_expr(col: str = "_v") -> "pl.Expr":
    """is_infinite：±Inf → 1；NULL/NaN/有限 → 0。"""
    import polars as pl

    return (
        pl.when(pl.col(col).is_null() | pl.col(col).is_nan())
        .then(0.0)
        .otherwise(pl.col(col).is_infinite().cast(pl.Float64))
    )


def is_finite_polars_expr(col: str = "_v") -> "pl.Expr":
    """is_finite：有限 → 1；NULL/NaN/±Inf → 0。"""
    import polars as pl

    return (
        pl.when(pl.col(col).is_null() | pl.col(col).is_nan())
        .then(0.0)
        .otherwise(pl.col(col).is_finite().cast(pl.Float64))
    )


def is_infinite_sql(value_col: str = "_v", *, dialect_is_clickhouse: bool = False) -> str:
    finite_fn = "isFinite" if dialect_is_clickhouse else "isfinite"
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    return (
        f"CASE WHEN {value_col} IS NULL OR {isnan_fn}({value_col}) THEN 0.0 "
        f"WHEN NOT {finite_fn}({value_col}) THEN 1.0 ELSE 0.0 END"
    )


def _pandas_null_mask(left: "pd.Series | pd.DataFrame", right: "pd.Series | pd.DataFrame"):
    import pandas as pd

    return left.isna() | right.isna()


def max_horizontal_pandas(
    left: "pd.Series | pd.DataFrame", right: "pd.Series | pd.DataFrame"
) -> "pd.Series | pd.DataFrame":
    """maximum：任一 NULL/NaN → NULL（对齐 DuckDB GREATEST）。"""
    import numpy as np
    import pandas as pd

    null_mask = _pandas_null_mask(left, right)
    raw = np.maximum(
        left.to_numpy(dtype=np.float64, copy=False),
        right.to_numpy(dtype=np.float64, copy=False),
    )
    if isinstance(left, pd.DataFrame):
        return pd.DataFrame(raw, index=left.index, columns=left.columns).mask(null_mask)
    return pd.Series(raw, index=left.index, name=getattr(left, "name", None)).mask(null_mask)


def min_horizontal_pandas(
    left: "pd.Series | pd.DataFrame", right: "pd.Series | pd.DataFrame"
) -> "pd.Series | pd.DataFrame":
    """minimum：任一 NULL/NaN → NULL。"""
    import numpy as np
    import pandas as pd

    null_mask = _pandas_null_mask(left, right)
    raw = np.minimum(
        left.to_numpy(dtype=np.float64, copy=False),
        right.to_numpy(dtype=np.float64, copy=False),
    )
    if isinstance(left, pd.DataFrame):
        return pd.DataFrame(raw, index=left.index, columns=left.columns).mask(null_mask)
    return pd.Series(raw, index=left.index, name=getattr(left, "name", None)).mask(null_mask)


def compare_pandas(
    op: str,
    left: "pd.Series | pd.DataFrame",
    right: "pd.Series | pd.DataFrame",
) -> "pd.Series | pd.DataFrame":
    """比较算子：有效输入 → 0/1；NULL/NaN → NULL。"""
    import numpy as np
    import pandas as pd

    larr = left.to_numpy(dtype=np.float64, copy=False)
    if isinstance(right, (int, float, np.floating)):
        rscalar = float(right)
        valid = left.notna().to_numpy() & ~np.isnan(larr)
        cmp_map = {
            "gt": larr > rscalar,
            "lt": larr < rscalar,
            "eq": larr == rscalar,
            "ge": larr >= rscalar,
            "le": larr <= rscalar,
            "ne": larr != rscalar,
        }
        raw = np.where(valid, cmp_map[op].astype(np.float64), np.nan)
    else:
        rarr = right.to_numpy(dtype=np.float64, copy=False)
        valid = left.notna().to_numpy() & right.notna().to_numpy() & ~np.isnan(larr) & ~np.isnan(rarr)
        cmp_map = {
            "gt": larr > rarr,
            "lt": larr < rarr,
            "eq": larr == rarr,
            "ge": larr >= rarr,
            "le": larr <= rarr,
            "ne": larr != rarr,
        }
        raw = np.where(valid, cmp_map[op].astype(np.float64), np.nan)
    if isinstance(left, pd.DataFrame):
        return pd.DataFrame(raw, index=left.index, columns=left.columns)
    return pd.Series(raw, index=left.index, name=getattr(left, "name", None))


def protected_log_pandas(x: "pd.Series | pd.DataFrame", *, epsilon: float) -> "pd.Series | pd.DataFrame":
    """log(max(x, epsilon))，NULL 保持 NULL。"""
    import numpy as np
    import pandas as pd

    arr = x.to_numpy(dtype=np.float64, copy=False)
    valid = x.notna().to_numpy()
    clipped = np.where(arr <= epsilon, epsilon, arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(valid, np.log(clipped), np.nan)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(raw, index=x.index, columns=x.columns)
    return pd.Series(raw, index=x.index, name=getattr(x, "name", None))


def protected_div_pandas(
    x: "pd.Series | pd.DataFrame",
    y: "pd.Series | pd.DataFrame",
    *,
    epsilon: float,
    default: float,
) -> "pd.Series | pd.DataFrame":
    """|denom|<=eps → default；NULL 输入 → NULL。"""
    import numpy as np
    import pandas as pd

    null_mask = x.isna() | y.isna()
    out = x / y.where(y.abs() > epsilon)
    small = y.abs() <= epsilon
    out = out.mask(null_mask)
    out = out.mask(small, other=default)
    if isinstance(x, pd.DataFrame):
        return out.replace([np.inf, -np.inf], default)
    return out.replace([np.inf, -np.inf], default)


def log_fill_invalid_pandas(x: "pd.Series | pd.DataFrame", *, epsilon: float) -> "pd.Series | pd.DataFrame":
    """NULL/非法域 → log(epsilon)。"""
    import numpy as np
    import pandas as pd

    arr = x.to_numpy(dtype=np.float64, copy=True)
    safe = np.where(np.isnan(arr), epsilon, np.where(arr <= epsilon, epsilon, arr))
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.log(safe)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(raw, index=x.index, columns=x.columns)
    return pd.Series(raw, index=x.index, name=getattr(x, "name", None))


def div_or_default_pandas(
    x: "pd.Series | pd.DataFrame",
    y: "pd.Series | pd.DataFrame",
    *,
    epsilon: float,
    default: float,
) -> "pd.Series | pd.DataFrame":
    """NULL 或 |denom|<=eps → default。"""
    import numpy as np
    import pandas as pd

    out = x / y.where(y.abs() > epsilon)
    small = y.abs() <= epsilon
    out = out.where(~small, other=default)
    out = out.fillna(default)
    return out.replace([np.inf, -np.inf], default)


def is_infinite_pandas(x: "pd.Series | pd.DataFrame") -> "pd.Series | pd.DataFrame":
    """is_infinite：±Inf → 1；NULL/NaN/有限 → 0。"""
    import numpy as np
    import pandas as pd

    arr = x.to_numpy(dtype=np.float64, copy=False)
    raw = np.where(np.isinf(arr), 1.0, 0.0)
    raw = np.where(x.isna().to_numpy() | np.isnan(arr), 0.0, raw)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(raw, index=x.index, columns=x.columns)
    return pd.Series(raw, index=x.index, name=getattr(x, "name", None))


def is_finite_pandas(x: "pd.Series | pd.DataFrame") -> "pd.Series | pd.DataFrame":
    """is_finite：有限 → 1；NULL/NaN/±Inf → 0。"""
    import numpy as np
    import pandas as pd

    arr = x.to_numpy(dtype=np.float64, copy=False)
    valid_finite = np.isfinite(arr)
    raw = np.where(valid_finite, 1.0, 0.0)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(raw, index=x.index, columns=x.columns)
    return pd.Series(raw, index=x.index, name=getattr(x, "name", None))


def is_finite_sql(value_col: str = "_v", *, dialect_is_clickhouse: bool = False) -> str:
    finite_fn = "isFinite" if dialect_is_clickhouse else "isfinite"
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    return (
        f"CASE WHEN {value_col} IS NULL OR {isnan_fn}({value_col}) THEN 0.0 "
        f"WHEN {finite_fn}({value_col}) THEN 1.0 ELSE 0.0 END"
    )
