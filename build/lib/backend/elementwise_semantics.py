# -*- coding: utf-8
"""元素级算子语义：max/min NULL、比较 NULL 传播、protected 域裁剪。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd


# ---------------------------------------------------------------------------
# R40 #199/#200: ProtectedDivisionSemantics —— protected_div / div_or_default
# 跨 backend 的**单一真值表**（防止各 emitter 各自处理 Inf/overflow 而分叉）。
# ---------------------------------------------------------------------------

_NullPolicy = Literal["null", "default"]
_InfPolicy = Literal["default", "null", "zero"]
_OverflowPolicy = Literal["default", "inf", "null"]


@dataclass(frozen=True)
class ComputePrecisionPolicy:
    """R40 #254：隐式 float64 计算的精度合同（进 factor identity）。

    旧实现把输入隐式升级到 float64，无任何声明 —— 同一表达式在
    ``float32`` 输入与 ``float64`` 输入下的计算精度/结果漂移完全不可见。
    本对象声明：

    * ``compute_dtype`` — 实际计算类型（默认 float64）；
    * ``input_dtype_identity_preserved`` — 输入 dtype 是否保留在 factor
      identity 中（True 表示 float32 vs float64 输入视为不同身份）；
    * ``output_storage_precision`` — 输出存储精度。

    ``identity_payload`` 进 factor identity / parity evidence。
    """

    compute_dtype: str = "float64"
    input_dtype_identity_preserved: bool = True
    output_storage_precision: str = "float64"

    def identity_payload(self) -> dict[str, object]:
        return {
            "compute_dtype": self.compute_dtype,
            "input_dtype_identity_preserved": self.input_dtype_identity_preserved,
            "output_storage_precision": self.output_storage_precision,
        }


DEFAULT_COMPUTE_PRECISION = ComputePrecisionPolicy()


def compute_precision_policy() -> ComputePrecisionPolicy:
    """默认 ComputePrecisionPolicy（所有元素级算子共享）。"""
    return DEFAULT_COMPUTE_PRECISION


def compute_promotes_float32_to_float64() -> bool:
    """#254：输入 float32 是否提升到 float64 计算（True = 默认行为）。"""
    return DEFAULT_COMPUTE_PRECISION.compute_dtype == "float64"


@dataclass(frozen=True)
class ProtectedDivisionSemantics:
    """R40 #199：protected 除法的完整真值表。

    字段解释（输入为 numer / denom）：
    * ``numerator_null`` / ``denominator_null`` — NULL/NaN 输入；
    * ``numerator_inf`` — 有限 denom 下的 ±Inf numer（避免 Inf 泄漏进输出）；
    * ``denominator_inf`` — 有限 numer 下的 ±Inf denom（数学上 x/∞ = 0）；
    * ``both_inf`` — Inf/Inf（不确定形式 → NaN/NULL）；
    * ``small_denominator`` — ``|denom| <= eps``；
    * ``overflow`` — 有限/有限 但商溢出到 ±Inf。

    默认真值表（与 pandas ``protected_div_pandas`` 现行行为一致）：
    NULL 输入 → NULL；Inf numer → default；Inf denom → 0；Inf/Inf → NULL；
    small denom → default；overflow → default。

    ``div_or_default`` 是同一真值表的**填充变体**：NULL 输入 → default
    （fillna），其余与 protected_div 相同。
    """

    numerator_null: _NullPolicy = "null"
    denominator_null: _NullPolicy = "null"
    numerator_inf: _InfPolicy = "default"
    denominator_inf: _InfPolicy = "zero"
    both_inf: _NullPolicy = "null"
    small_denominator: _NullPolicy = "default"
    overflow: _OverflowPolicy = "default"

    @property
    def fill_nulls(self) -> bool:
        """True 表示 NULL 输入也填充 default（div_or_default 语义）。"""
        return self.numerator_null == "default" or self.denominator_null == "default"


#: protected_div 权威真值表。
PROTECTED_DIV_SEMANTICS = ProtectedDivisionSemantics()
#: div_or_default 权威真值表（NULL 输入 → default）。
DIV_OR_DEFAULT_SEMANTICS = ProtectedDivisionSemantics(
    numerator_null="default",
    denominator_null="default",
)


def protected_division_truth_table(
    numer: "pd.Series | pd.DataFrame",
    denom: "pd.Series | pd.DataFrame",
    *,
    eps: float,
    default: float,
    semantics: ProtectedDivisionSemantics = PROTECTED_DIV_SEMANTICS,
) -> "pd.Series | pd.DataFrame":
    """按真值表逐元素计算 protected 除法（numpy 实现，pandas emitter 复用）。

    完全复刻 pandas 参考算法（真值表默认值与之逐位一致）：

    protected_div:
        1. raw = x / y  （small denom 处先置 NaN —— ``y.where(y.abs()>eps)``）；
        2. NULL 输入 -> NULL（NaN）；
        3. small denom -> default；
        4. ±Inf / overflow 结果 -> default。

    div_or_default:
        1. small denom -> default；
        2. NULL 输入 -> default（``fillna``）；
        3. ±Inf / overflow -> default。

    返回的 Series/DataFrame 中，NULL 语义对应 NaN。
    """
    import numpy as np
    import pandas as pd

    na = numer.to_numpy(dtype=np.float64, copy=False)
    db = denom.to_numpy(dtype=np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = na / db
    out = raw.copy()

    numer_nan = np.isnan(na)
    denom_nan = np.isnan(db)
    numer_inf = np.isinf(na)
    denom_inf = np.isinf(db)
    small = np.abs(db) <= eps

    # 1) 先处理 NULL（protected_div: NaN；div_or_default: 之后 fillna default）。
    if semantics.numerator_null == "default" or semantics.denominator_null == "default":
        out[numer_nan | denom_nan] = default
    else:
        out[numer_nan | denom_nan] = np.nan

    # 2) small denominator -> default。
    out[small & ~(numer_nan | denom_nan)] = default

    # 3) ±Inf 输入 / 溢出结果。
    if semantics.numerator_inf == "default":
        out[numer_inf & ~denom_inf] = default
    if semantics.denominator_inf == "zero":
        out[denom_inf & ~numer_inf] = 0.0
    elif semantics.denominator_inf == "default":
        out[denom_inf & ~numer_inf] = default
    # both_inf -> NULL（NaN），除非 div_or_default（已有 default）。
    if semantics.both_inf == "null":
        out[numer_inf & denom_inf] = np.nan
    # overflow（有限/有限 但 |raw|=Inf）-> default。
    if semantics.overflow == "default":
        out[np.isinf(out) & ~(numer_inf | denom_inf) & ~(numer_nan | denom_nan)] = default
    elif semantics.overflow == "null":
        out[np.isinf(out) & ~(numer_inf | denom_inf) & ~(numer_nan | denom_nan)] = np.nan

    # div_or_default 变体：最终 fillna(default) —— Inf/Inf 等 NaN 也填 default。
    if semantics.numerator_null == "default" or semantics.denominator_null == "default":
        out[np.isnan(out)] = default

    if isinstance(numer, pd.DataFrame):
        return pd.DataFrame(out, index=numer.index, columns=numer.columns)
    return pd.Series(out, index=numer.index, name=getattr(numer, "name", None))


def protected_division_semantics_for(canonical: str) -> ProtectedDivisionSemantics:
    """算子 -> 权威真值表（#199/#200 单一权威，供所有 emitter 引用）。"""
    return DIV_OR_DEFAULT_SEMANTICS if canonical == "div_or_default" else PROTECTED_DIV_SEMANTICS


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
    """maximum：任一 NULL/NaN → NULL；±Inf 是**数学值**（max(1, Inf)=Inf）。

    R40 #201：Inf 在 element-math 家族（max/min）中是数学值；统计家族
    （mean/std/var）把 Inf 当无效样本。声明见
    :func:`inf_semantics_for` / :func:`element_math_vs_stats_inf_semantics`。
    """
    import polars as pl

    missing = left.is_null() | right.is_null() | left.is_nan() | right.is_nan()
    return pl.when(missing).then(None).otherwise(pl.max_horizontal(left, right))


def min_horizontal_polars(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    """minimum：任一 NULL/NaN → NULL；±Inf 是数学值（R40 #201）。"""
    import polars as pl

    missing = left.is_null() | right.is_null() | left.is_nan() | right.is_nan()
    return pl.when(missing).then(None).otherwise(pl.min_horizontal(left, right))


def _both_finite(left: "pl.Expr", right: "pl.Expr") -> "pl.Expr":
    import polars as pl

    # pandas scalar_compare._compare masks out every NON-FINITE input (Inf
    # included), so a comparison involving Inf is NULL there.  Match it.
    return (
        left.is_not_null()
        & right.is_not_null()
        & ~left.is_nan()
        & ~right.is_nan()
        & ~left.is_infinite()
        & ~right.is_infinite()
    )


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
    """比较 SQL：NULL/NaN/**Inf** → NULL。

    R40 #198: 旧实现只查 ``IS NOT NULL`` + ``NOT isnan(col)``，SQL 侧把 ±Inf
    当成数学值参与比较，与 Polars 的 ``_both_finite``（排除 Inf）不一致。
    现在补齐 ``NOT isinf(col)`` —— Inf 输入在四个 backend 上都返回 NULL。
    """
    sym = {"gt": ">", "lt": "<", "eq": "=", "ge": ">=", "le": "<=", "ne": "<>"}[op]
    if dialect_is_clickhouse and op == "ne":
        sym = "!="
    isnan_fn = "isNaN" if dialect_is_clickhouse else "isnan"
    isinf_fn = "isInfinite" if dialect_is_clickhouse else "isinf"
    valid = (
        f"({left_col} IS NOT NULL AND {right_col} IS NOT NULL "
        f"AND NOT {isnan_fn}({left_col}) AND NOT {isnan_fn}({right_col}) "
        f"AND NOT {isinf_fn}({left_col}) AND NOT {isinf_fn}({right_col}))"
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
    """|denom|<=eps → default；NULL/Inf/Inf → NULL；Inf numer → default；
    Inf denom → 0；溢出 → default。

    R40 #199: 与 :data:`PROTECTED_DIV_SEMANTICS` 真值表逐位一致（旧实现把
    ±Inf 输入一律 NULL，与 pandas 的「Inf numer → default / Inf denom → 0」
    分叉）。
    """
    import polars as pl

    raw = numer / denom
    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(None)
        .when(denom.abs() <= eps)
        .then(default)
        .when(numer.is_infinite() & denom.is_infinite())
        .then(None)  # both_inf -> NULL (NaN)
        .when(numer.is_infinite())
        .then(default)  # Inf numerator (finite denom) -> default
        .when(denom.is_infinite())
        .then(0.0)  # Inf denominator (finite numer) -> 0
        .when(raw.is_infinite())
        .then(default)  # overflow
        .otherwise(raw)
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
    """|denom|<=eps 或 NULL/NaN 输入 → default；Inf/overflow → default。

    R40 #200: 与 :data:`DIV_OR_DEFAULT_SEMANTICS` 真值表一致——旧实现不处理
    Inf 输入（Inf numer 会泄漏成 Inf）与 overflow。
    """
    import polars as pl

    raw = numer / denom
    return (
        pl.when(numer.is_null() | denom.is_null())
        .then(default)  # div_or_default: NULL -> default (fillna)
        .when(denom.abs() <= eps)
        .then(default)
        .when(numer.is_infinite() & denom.is_infinite())
        .then(default)  # Inf/Inf -> default (filled NULL)
        .when(numer.is_infinite())
        .then(default)
        .when(denom.is_infinite())
        .then(0.0)
        .when(raw.is_infinite())
        .then(default)  # overflow
        .otherwise(raw)
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
    """比较算子：有效输入 → 0/1；NULL/NaN/**±Inf** → NULL。

    R40 #198：与 Polars ``_both_finite`` / SQL ``compare_sql`` 对齐 —— ±Inf 在
    比较中不是数学值，而是无效输入（返回 NULL/NaN）。
    """
    import numpy as np
    import pandas as pd

    larr = left.to_numpy(dtype=np.float64, copy=False)
    if isinstance(right, (int, float, np.floating)):
        rscalar = float(right)
        valid = left.notna().to_numpy() & ~np.isnan(larr) & ~np.isinf(larr)
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
        valid = (
            left.notna().to_numpy() & right.notna().to_numpy()
            & ~np.isnan(larr) & ~np.isnan(rarr)
            & ~np.isinf(larr) & ~np.isinf(rarr)
        )
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


# ---------------------------------------------------------------------------
# R40 #201: element-math vs stats —— Inf 语义在 OperatorContract 中显式声明。
# ---------------------------------------------------------------------------

#: element-math 家族（max/min/比较…）：±Inf 是数学值（max(1, Inf)=Inf）。
ELEMENT_MATH_CANONICALS: frozenset[str] = frozenset(
    {"maximum", "minimum", "max", "min", "max_horizontal", "min_horizontal"}
)
#: stats 家族（mean/std/var/corr/beta…）：Inf 是无效样本，不得进入统计量。
_STATS_INF_INVALID = "inf_is_invalid_sample"


def element_math_vs_stats_inf_semantics() -> dict[str, str]:
    """#201: 两类语义的显式声明 —— optimizer 不得假设全库统一。

    返回 ``{canonical_family: policy}``，policy ∈
    ``{"inf_is_mathematical_value", "inf_is_invalid_sample"}``。
    """
    return {
        "element_math": "inf_is_mathematical_value",
        "stats": _STATS_INF_INVALID,
    }


def inf_semantics_for(canonical: str) -> str:
    """单个 canonical 的 Inf 语义（#201）。

    element-math 家族 -> ``"inf_is_mathematical_value"``；其余（统计 / 聚合）->
    ``"inf_is_invalid_sample"``。``OperatorContract`` 应把该声明写进算子的
    数值语义合同。
    """
    name = str(canonical or "").strip()
    if name in ELEMENT_MATH_CANONICALS:
        return "inf_is_mathematical_value"
    return _STATS_INF_INVALID


# ---------------------------------------------------------------------------
# R40 #202: 共享 edge corpus（NULL/NaN/±Inf/±0/subnormal/1e308/1e-308/
# 零分母/近零分母）—— 供 Pandas/Polars/DuckDB/ClickHouse 可执行 parity。
# ---------------------------------------------------------------------------

def edge_value_corpus() -> dict[str, float]:
    """#202: 命名 edge 值集合（可执行 parity 测试的单一输入源）。

    NULL 用 ``float("nan")`` 表示（SQL 层 NULL 与 NaN 的区分由各 emitter
    自行处理）；``pos_zero``/``neg_zero`` 区分 signed zero；``subnormal`` 是
    最小正 subnormal；``near_zero_denom`` 是 near-zero 分母（> eps 但极小）。
    """
    import numpy as np

    return {
        "nan": float("nan"),
        "null": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "pos_zero": 0.0,
        "neg_zero": -0.0,
        "subnormal": float(np.nextafter(0.0, 1.0)),
        "huge": 1e308,
        "tiny": 1e-308,
        "one": 1.0,
        "near_zero_denom": 1e-300,
    }


def edge_value_names() -> tuple[str, ...]:
    return tuple(sorted(edge_value_corpus().keys()))
