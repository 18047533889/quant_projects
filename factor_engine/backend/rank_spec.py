# -*- coding: utf-8
"""Rank 系列统一契约：tie、pct 公式、singleton、NULL/Inf 策略。

Polars / DuckDB / Pandas 实现须引用 ``rank_spec_for(canon)``，禁止各路径硬编码不同 rank helper。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

if False:  # TYPE_CHECKING without import cycle
    import polars as pl

RankMethod = Literal["average", "min", "max", "first", "ordinal"]
PctFormula = Literal["rank_over_n", "rank_minus1_over_nminus1"]
RankNullPolicy = Literal["exclude", "bottom", "error"]
RankInfPolicy = Literal["exclude", "participate"]


@dataclass(frozen=True)
class RankSpec:
    """截面/组内/时序 rank 的公开语义契约。"""

    method: RankMethod = "average"
    pct_formula: PctFormula = "rank_over_n"
    null_policy: RankNullPolicy = "exclude"
    singleton_value: float = 0.5
    inf_policy: RankInfPolicy = "exclude"
    order_dependent_tie: bool = False


# canonical → RankSpec（单一事实来源）
RANK_SPECS: dict[str, RankSpec] = {
    # 截面 0-1：(rank-1)/(n-1)；单元素 → 0.5
    "rank": RankSpec(
        method="average",
        pct_formula="rank_minus1_over_nminus1",
        null_policy="exclude",
        singleton_value=0.5,
        # PARITY-SWEEP-R56: finite-mask authority (cs_rank_01 / NEW-255): ±Inf is
        # not a rankable extreme and stays NULL; the cross-section base counts
        # only np.isfinite cells.
        inf_policy="exclude",
    ),
    # 截面/组内百分位：rank/n；NULL 不参与分母
    "rank_pct": RankSpec(
        method="average",
        pct_formula="rank_over_n",
        null_policy="exclude",
        singleton_value=1.0,
        # PARITY-SWEEP-R56: pandas authority ``CrossSectionalRank._calculate_series
        # = x.rank(pct=True, axis=1)`` ranks over NaN-skipped rows but INCLUDES
        # ±Inf as a valid extreme (D gets rank/n=1.0).  The old default
        # ``inf_policy="exclude"`` made polars_long / SQL drop the Inf row while
        # the pandas backend ranked it — 10% of the cross-section diverged on the
        # Inf day of test_cs_family_systematic_parity.
        inf_policy="participate",
    ),
    "cs_pct_rank": RankSpec(
        method="average",
        pct_formula="rank_over_n",
        null_policy="exclude",
        singleton_value=1.0,
        inf_policy="participate",
    ),
    "group_rank": RankSpec(
        method="average",
        pct_formula="rank_over_n",
        null_policy="exclude",
        singleton_value=1.0,
        # PARITY-SWEEP-R56: group_rank follows the same pandas-percentile authority
        # (rank/n over NaN-skipped rows, Inf participates as an extreme).
        inf_policy="participate",
    ),
    # 滚动百分位：窗口内 average rank / 窗口内非 NULL 计数
    "ts_rank": RankSpec(
        method="average",
        pct_formula="rank_over_n",
        null_policy="exclude",
        singleton_value=1.0,
        # PARITY-SWEEP-R56: pandas ``rolling.rank(pct=True)`` treats ±Inf as a
        # missing sample — the Inf cell outputs NaN and Inf does NOT participate
        # in the window rank (unlike cs_pct_rank / rank_pct which rank over the
        # full NaN-skipped row and keep Inf as an extreme).  Keep exclude.
        inf_policy="exclude",
    ),
}


def rank_spec_for(canon: str) -> RankSpec:
    """查询算子 rank 契约；未知算子回退 average + rank_over_n。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # PARITY-SWEEP-R56: resolve only ONE hop so an exact-duplicate alias
    # (rank_pct -> cs_pct_rank) picks up the alias target's own RankSpec instead
    # of falling through to a default.  ``_aliases`` may still be empty during
    # very early imports, in which case the direct canonical name is used.
    name = OperatorRegistry._aliases.get(canon, canon)
    if name in RANK_SPECS:
        return RANK_SPECS[name]
    # One-hop alias resolution (e.g. rank_pct -> cs_pct_rank) before giving up.
    resolved = OperatorRegistry.resolve_canonical_optional(name)
    return RANK_SPECS.get(resolved, RankSpec())


def rank_tie_method(canon: str) -> str:
    """并列策略字符串（Polars/SQL ``method`` 参数）。"""
    return rank_spec_for(canon).method


def rank_ignore_nan(canon: str) -> bool:
    """NaN 不参与排名（输出仍为 NULL）。"""
    return rank_spec_for(canon).null_policy == "exclude"


def polars_cs_rank_expr(
    value_col: str,
    *,
    partition_cols: Sequence[str],
    order_by: str,
    spec: RankSpec | None = None,
    canon: str = "rank",
) -> "pl.Expr":
    """截面/组内 rank expr；``partition_cols`` 如 ``(_TS,)`` 或 ``(_TS, _GRP)``。"""
    import polars as pl

    from factor_engine.backend.stat_valid import polars_rank_input, polars_row_stat_invalid

    sp = spec or rank_spec_for(canon)
    # PARITY-SWEEP-R56: NaN exclusion follows RankSpec.null_policy; ±Inf follows
    # RankSpec.inf_policy.  cs_pct_rank / rank_pct / group_rank / ts_rank set
    # inf_policy="participate" (pandas ``x.rank(pct=True, axis=1)`` ranks over
    # NaN-skipped rows but keeps ±Inf as a rankable extreme — D gets rank/n=1.0);
    # the finite-mask ops cs_rank_01 / rank keep inf_policy="exclude" (NEW-255).
    exclude_nan = sp.inf_policy == "exclude"
    invalid = polars_row_stat_invalid(value_col, exclude_nan=exclude_nan)
    rank_input = polars_rank_input(value_col, exclude_nan=exclude_nan)
    r = rank_input.rank(method=sp.method).over(*partition_cols, order_by=order_by)
    n = rank_input.count().over(*partition_cols, order_by=order_by)
    if sp.pct_formula == "rank_minus1_over_nminus1":
        return (
            pl.when(invalid)
            .then(None)
            .when(n == 0)
            .then(None)
            .when(n == 1)
            .then(sp.singleton_value)
            .otherwise((r - 1.0) / (n - 1.0))
        )
    denom = pl.when(n > 0).then(n.cast(pl.Float64)).otherwise(None)
    frac = r / denom
    return pl.when(invalid).then(None).when(n == 0).then(None).otherwise(frac)


def polars_ts_rank_expr(
    value_col: str,
    *,
    window: int,
    inst_col: str,
    ts_col: str,
    min_periods: int = 1,
    spec: RankSpec | None = None,
    canon: str = "ts_rank",
) -> "pl.Expr":
    """滚动百分位 rank：窗口内 average rank / 非 NULL 计数；``min_periods`` 不足 → NULL。"""
    import polars as pl

    from factor_engine.backend.stat_valid import polars_rank_input, polars_row_stat_invalid, rank_excludes_nan

    sp = spec or rank_spec_for(canon)
    # PARITY-SWEEP-R56: same NaN/Inf split as polars_cs_rank_expr — NaN follows
    # null_policy, ±Inf follows inf_policy (ts_rank: Inf participates).
    exclude_nan = sp.inf_policy == "exclude"
    w = int(window)
    mp = max(int(min_periods), 1)
    rank_input = polars_rank_input(value_col, exclude_nan=exclude_nan)
    invalid = polars_row_stat_invalid(value_col, exclude_nan=exclude_nan)
    rank = rank_input.rolling_rank(window_size=w, min_samples=1, method=sp.method).over(
        inst_col, order_by=ts_col
    )
    cnt = (
        rank_input.is_not_null()
        .cast(pl.Float64)
        .rolling_sum(window_size=w, min_samples=1)
        .over(inst_col, order_by=ts_col)
    )
    return (
        pl.when(invalid)
        .then(None)
        .when(cnt < mp)
        .then(None)
        .otherwise(rank / cnt)
    )
