# -*- coding: utf-8 -*-
"""
时序滚动算子（**沿时间轴、按单标的** 在 panel 上计算）。

语义
----
对每个标的列独立做 rolling / EWM / 滞后窗口统计；**不是**截面 rank（见 ``cross_sectional.py``）。
DSL 常用 ``ts_*`` 前缀：``ts_mean``、``ts_std``、``ts_corr``、``ts_rank``、``ts_regression`` 等。

本模块还包含
------------
- 移动平均族：``SMA`` / ``WMA`` / ``EMA``（别名见 ``_aliases.py``）；
- 衰减加权：``decay_linear``（``ts_decay_linear``）、``ts_decay``、``ts_sum_decay``；
- ``m_*`` / ``tm_*``：窗口内 top-N、分位数、beta 等扩展滚动统计；
- 与 ``shift_diff_cum.py`` 部分重叠的 ``ts_delay`` / ``ts_delta``（以 registry canonical 为准）。

输入：宽表 ``DataFrame``，index=交易日，columns=instrument。
输出：同形 panel。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from factor_engine.cleaned_operators._causal import causal_lag
from factor_engine.cleaned_operators._rolling_fast import (
    cum_top_n_mean,
    cum_top_n_sum,
    check_wma_partial_policy,
    rolling_beta,
    rolling_bottom_n_mean,
    rolling_bottom_n_sum,
    rolling_linear_weighted,
    rolling_regression,
    rolling_top_n_mean,
    rolling_top_n_mean_window,
    rolling_top_n_std,
    rolling_top_n_sum,
    rolling_top_n_sum_window,
    wma_partial_policy_digest,
)
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)

import numpy as np
import pandas as pd
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# ---------------------------------------------------------------------------
# TopKContract (R40 #196).  A top-k / bottom-k / nlargest / nsmallest operator
# must DECLARE its sample policy and tie policy; an undeclared tie policy is
# rejected in production because tie-breaking by row/column iteration order is
# not a deterministic execution identity.
# ---------------------------------------------------------------------------
class TopKTiePolicy:
    STABLE_INSTRUMENT_KEY = "stable_instrument_key"   # ties break on canonical instrument key
    INCLUDE_ALL_TIES = "include_all_ties"             # all tied values enter the selection
    AVERAGE_WEIGHT = "average_weight"                 # tied values share the remaining slot weight


TOPK_TIE_POLICIES = frozenset({
    TopKTiePolicy.STABLE_INSTRUMENT_KEY,
    TopKTiePolicy.INCLUDE_ALL_TIES,
    TopKTiePolicy.AVERAGE_WEIGHT,
})


class TopKSamplePolicy:
    FINITE_ONLY = "finite_only"
    NAN_AND_INF_ARE_MISSING = "nan_and_inf_are_missing"


TOPK_SAMPLE_POLICIES = frozenset({TopKSamplePolicy.FINITE_ONLY, TopKSamplePolicy.NAN_AND_INF_ARE_MISSING})


@dataclass(frozen=True)
class TopKContract:
    """Declared sample/tie policy for a top-k selection operator (R40 #196).

    ``sample_policy=FINITE_ONLY`` means ±Inf is not a valid observation (a top
    ``k`` selection that silently ranks Inf above every finite value is not an
    economically meaningful top-k); ``tie_policy`` fixes how equal sort values
    are ordered so two executions on the same data always select the same rows.
    """

    sample_policy: str = TopKSamplePolicy.FINITE_ONLY
    tie_policy: str | None = None  # MUST be declared for production top-k


def check_topk_contract(
    contract: TopKContract,
    *,
    production: bool = True,
    canonical: str = "",
) -> None:
    """Production gate: a top-k operator must declare a deterministic tie policy.

    Raises ``ValueError`` when ``tie_policy`` is undeclared in production (R40
    #196 — undeclared tie-breaking is row-order dependent and must not enter a
    production surface).  Research callers pass ``production=False`` to degrade.
    """
    if contract.sample_policy not in TOPK_SAMPLE_POLICIES:
        raise ValueError(
            f"{canonical or 'topk'}: unknown sample_policy {contract.sample_policy!r} "
            f"(expected one of {sorted(TOPK_SAMPLE_POLICIES)})"
        )
    if production and contract.tie_policy is None:
        raise ValueError(
            f"{canonical or 'topk'}: production top-k requires a declared tie_policy "
            f"(one of {sorted(TOPK_TIE_POLICIES)}); undeclared tie-breaking is "
            "row-order dependent and rejected"
        )
    if contract.tie_policy is not None and contract.tie_policy not in TOPK_TIE_POLICIES:
        raise ValueError(
            f"{canonical or 'topk'}: unknown tie_policy {contract.tie_policy!r} "
            f"(expected one of {sorted(TOPK_TIE_POLICIES)})"
        )


def _select_top_k_with_tie_policy(
    values: pd.Series,
    k: int,
    *,
    ascending: bool,
    tie_policy: str | None,
) -> "pd.Index":
    """Deterministic top-k selection honoring a :class:`TopKContract`.

    ``values`` is a Series indexed by instrument key.  Missing (±Inf/NaN under
    FINITE_ONLY) are dropped; ties under ``STABLE_INSTRUMENT_KEY`` break on the
    canonical instrument key (ascending) so the selection is independent of the
    input column order.
    """
    mask = np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
    clean = values[mask]
    if clean.empty or k < 1:
        return clean.index[:0]
    if tie_policy == TopKTiePolicy.INCLUDE_ALL_TIES:
        # Include every instrument whose value equals the k-th selected value.
        ordered = clean.sort_values(ascending=ascending, kind="mergesort")
        if len(ordered) <= k:
            return ordered.index
        kth = ordered.iloc[k - 1]
        return ordered.index[ordered <= kth if not ascending else ordered >= kth]
    if tie_policy == TopKTiePolicy.AVERAGE_WEIGHT:
        # Take the first k after a stable instrument-key tie-break (weights are
        # applied downstream by the caller).
        ordered = clean.sort_values(ascending=ascending, kind="stable")
        # stable in the sense of deterministic: primary value, secondary key.
        ordered = ordered.iloc[_stable_topk_order(ordered, ascending)]
        return ordered.index[:k]
    # STABLE_INSTRUMENT_KEY (default) / explicit: value first, instrument key second.
    ordered = clean.sort_values(ascending=ascending, kind="stable")
    ordered = ordered.iloc[_stable_topk_order(ordered, ascending)]
    return ordered.index[:k]


def _stable_topk_order(ordered: pd.Series, ascending: bool) -> np.ndarray:
    """Deterministic index order: value (per ``ascending``) then instrument key.

    pandas ``sort_values(kind="stable")`` preserves the *original* order of tied
    rows; sorting (value, instrument-key) tuples canonically makes the tie-break
    independent of the input column order so the same cross-section always
    selects the same rows.
    """
    vals = ordered.to_numpy(dtype=np.float64, copy=False)
    keys = [str(i) for i in ordered.index]
    decorated = list(enumerate(zip(vals, keys)))
    if ascending:
        decorated.sort(key=lambda t: (t[1][0], t[1][1]))
    else:
        decorated.sort(key=lambda t: (-t[1][0], t[1][1]))
    return np.asarray([pos for pos, _ in decorated], dtype=np.int64)


# ---------------------------------------------------------------------------
# SupportPolicy (R40 #193).  ``min_periods`` (the minimum number of valid
# observations a window needs) is an ECONOMIC part of a rolling operator's
# definition — a change from ``min_periods=1`` to ``min_periods=3`` changes the
# output.  ``SupportPolicy`` makes the support explicit so it enters the
# operator semantic version / factor identity / evidence.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupportPolicy:
    """Declared window-support policy (R40 #193)."""

    min_observations: int = 1
    require_contiguous: bool = False
    partial_window: str = "allow"       # allow | reject
    ddof: int = 0                       # standard-deviation degrees of freedom

    def digest(self) -> str:
        import hashlib as _hashlib

        return _hashlib.sha256(
            f"support|{self.min_observations}|{self.require_contiguous}|"
            f"{self.partial_window}|{self.ddof}".encode("utf-8")
        ).hexdigest()[:12]


#: Canonical support policies for the rolling family.
SUPPORT_POLICIES: dict[str, SupportPolicy] = {
    "ts_mean": SupportPolicy(min_observations=1, ddof=0),
    "ts_std": SupportPolicy(min_observations=2, ddof=1),
    "ts_corr": SupportPolicy(min_observations=3, ddof=1),
    "ts_rank": SupportPolicy(min_observations=1, ddof=0),
}


def support_policy_for(canonical: str) -> SupportPolicy | None:
    return SUPPORT_POLICIES.get(canonical)


def check_support_policy(policy: SupportPolicy | None, *, canonical: str) -> None:
    """Production gate: a rolling statistic must declare its SupportPolicy."""
    if policy is None:
        raise ValueError(
            f"{canonical}: no declared SupportPolicy; min_periods is an economic "
            "parameter and must enter the semantic version (R40 #193)"
        )


# ---------------------------------------------------------------------------
# EWMContract (R40 #194).  The exponential-moving-average family hard-coded
# ``ewm(span=span, adjust=False)``.  The EWM policy is now an explicit contract
# (adjust / ignore_na / min_periods / decay mapping / bias / seed policy) whose
# ``digest()`` enters the operator semantic version and evidence — changing the
# policy automatically invalidates old cache/materialization identities.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EWMContract:
    """Explicit EWM policy for the EMA family (R40 #194)."""

    canonical: str
    decay_mapping: str = "span"          # span | alpha | halflife
    adjust: bool = False                 # ewm(adjust=...)
    ignore_na: bool = False
    min_periods: int = 0
    bias: bool = False
    ddof: int = 1                        # std/var correction
    seed_policy: str = "ewm_zero_initialization"  # how the recursion is seeded

    def to_kwargs(self, span: float | None = None) -> dict[str, Any]:
        """Build the ``ewm(...)`` keyword mapping for this contract."""
        kw: dict[str, Any] = {"adjust": self.adjust, "ignore_na": self.ignore_na, "min_periods": self.min_periods}
        if self.decay_mapping == "span":
            if span is None:
                raise ValueError("EWMContract(decay_mapping='span') needs span")
            kw["span"] = span
        elif self.decay_mapping == "alpha":
            kw["alpha"] = span  # alpha passed via the span slot for span-aliased canons
        elif self.decay_mapping == "halflife":
            kw["halflife"] = span
        return kw

    def digest(self) -> str:
        import hashlib as _hashlib

        canonical = _hashlib.sha256(
            f"ewm|{self.canonical}|{self.decay_mapping}|{self.adjust}|{self.ignore_na}"
            f"|{self.min_periods}|{self.bias}|{self.ddof}|{self.seed_policy}".encode()
        ).hexdigest()[:12]
        return canonical


#: Canonical EWM contracts (single source of truth for the EMA family).
EWM_CONTRACTS: dict[str, EWMContract] = {
    "ts_ema": EWMContract(canonical="ts_ema", decay_mapping="span", adjust=False),
}


def ewm_contract_for(canonical: str) -> EWMContract | None:
    return EWM_CONTRACTS.get(canonical)


def check_ewm_contract(contract: EWMContract | None, *, canonical: str) -> None:
    """Production gate: an EMA-family canonical MUST declare an EWM contract.

    An operator that computes an exponential moving average without a declared
    EWM contract cannot prove cross-backend parity and is rejected in
    production (R40 #194).
    """
    if contract is None:
        raise ValueError(
            f"{canonical}: EMA-family operator has no declared EWMContract; "
            "production requires the explicit adjust/ignore_na/decay/seed policy"
        )


# ---------------------------------------------------------------------------
# StatisticalSamplePolicy (R40 #187).  Every statistical operator
# (rank/mean/std/zscore/neutralize/regression/group) must use the SAME missing
# policy: a ``FINITE_ONLY`` sample is a value in ``np.isfinite`` (NaN AND ±Inf
# are missing).  ``uniform_finite_mask`` is the single mask provider so a window
# full of Inf never passes ``min_periods`` and yields a garbage statistic.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatisticalSamplePolicy:
    """Missing-sample policy for statistical operators (R40 #187)."""

    valid: str = "finite_only"   # finite_only | nan_is_missing
    min_count: int = 0           # minimum valid samples for the statistic to exist

    def mask(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Boolean mask of valid samples (``True`` = valid)."""
        arr = frame.to_numpy(dtype=np.float64, copy=False)
        if self.valid == "finite_only":
            valid = np.isfinite(arr)
        else:
            valid = ~np.isnan(arr)  # Inf counts as valid under nan_is_missing
        return pd.DataFrame(valid, index=frame.index, columns=frame.columns)

    def masked(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with every invalid sample set to NaN (for pandas rolling)."""
        mask = self.mask(frame)
        return frame.where(mask)


STATISTICAL_SAMPLE_POLICY_DEFAULT = StatisticalSamplePolicy(valid="finite_only", min_count=0)


def uniform_finite_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """The single uniform missing mask provider for all statistical operators.

    ``True`` = valid (finite); ``False`` = NaN or ±Inf (missing).  Using this
    everywhere guarantees the SAME sample policy across rank/mean/std/zscore/
    neutralize/regression/group statistics (R40 #187).
    """
    return STATISTICAL_SAMPLE_POLICY_DEFAULT.mask(frame)


def check_statistical_sample_policy(policy: StatisticalSamplePolicy, *, canonical: str = "") -> None:
    """Production gate: a statistical operator must declare its sample policy.

    An undeclared policy is rejected in production because NaN-only masking
    (``notna``) silently counts ±Inf as a valid observation, corrupting every
    window statistic that includes one.
    """
    if policy.valid not in {"finite_only", "nan_is_missing"}:
        raise ValueError(
            f"{canonical or 'stat'}: unknown sample policy {policy.valid!r}; "
            "expected 'finite_only' or 'nan_is_missing'"
        )


# 重复实现：见 ts_mean；dedupe 注销
# @register_operator(name="SMA", category="time_series", business_category="time_series", canonical="SMA", source="factor_dsl_np")
class SMA(SeriesOperator):
    """简单移动平均（SMA）；dedupe 后别名指向 ``ts_mean``。"""

    metadata = OperatorMetadata(
        name="SMA",
        category="time_series",
        description="计算n期简单移动平均（与m_avg相同）",
        examples=["SMA(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "sma", "simple"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        # R40 #187: uniform FINITE_ONLY sample policy — mask ±Inf to NaN so a
        # window "full" of Inf never yields an Inf mean (previously a lone Inf
        # passed min_periods=1 and produced Inf).
        return x.where(uniform_finite_mask(x)).rolling(window=window, min_periods=1).mean()



# canonical=WMA backend=pandas_numpy selected=WMA source=time_series/m_ops.py
@register_operator(name="WMA", category="time_series", business_category="time_series", canonical="WMA", source="factor_dsl_np")
class WMA(SeriesOperator):
    """加权移动平均（线性衰减权重）。"""

    metadata = OperatorMetadata(
        name="WMA",
        category="time_series",
        description="计算n期加权移动平均",
        examples=["WMA(close, 10)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "wma", "weighted"]
    )

    # R40 #195: the partial-warmup policy digest is part of the operator
    # contract / factor identity — a policy change invalidates cache identity.
    wma_partial_policy_digest = wma_partial_policy_digest()

    def _calculate_series(self, x: pd.DataFrame, window: int = 10, **kwargs) -> pd.DataFrame:
        check_wma_partial_policy(kwargs.get("partial_policy"))
        return rolling_linear_weighted(x, window)



_AGGR_TOP_N_FUNCS = frozenset({"sum", "avg", "mean", "max", "min", "std", "count"})


# canonical=aggr_top_n backend=pandas_numpy selected=aggr_top_n source=time_series/topn_ops.py
@register_operator(name="aggr_top_n", category="cross_sectional", business_category="cross_sectional_routing", canonical="aggr_top_n", source="factor_dsl_np")
class AggrTopN(SeriesOperator):
    """自定义 Top-N **跨截面路由**聚合（按排序列选取前 N 标的聚合）。

    Routing 状态：**全局 per-day**——每个交易日把当日全截面（所有标的）按
    ``sort_col`` 排序取前 ``top`` 名，再对选中标的的 ``x`` 做 ``aggr_func``
    聚合，并把聚合结果广播回被选中的位置（未选中的标的位置为 NaN）。不是
    逐标的时序窗口，也不是按组路由。支持 ``aggr_func``：
    ``sum`` / ``avg`` / ``mean`` / ``max`` / ``min`` / ``std`` / ``count``。
    未知 ``aggr_func`` 直接抛 ``ValueError``（绝不静默回退到 ``sum``）。"""

    metadata = OperatorMetadata(
        name="aggr_top_n", category="cross_sectional",
        description="自定义Top-N跨截面路由聚合（按 sort_col 取前 N 标的聚合，结果广播回选中位置）",
        examples=["aggr_top_n('sum', close, volume, 10, True)"],
        param_names=["aggr_func", "x", "sort_col", "top", "asc", "tie_policy"], return_type="series",
        tags=["cross_sectional", "routing", "top_n", "aggregate"]
    )
    def _calculate_series(self, aggr_func: str = "sum", x: pd.DataFrame = None,
                          sort_col: pd.DataFrame = None, top: int = 10,
                          asc: bool = True, tie_policy: str | None = None,
                          **kwargs) -> pd.DataFrame:
        aggr = str(aggr_func).lower()
        if aggr not in _AGGR_TOP_N_FUNCS:
            raise ValueError(
                f"aggr_top_n: unsupported aggr_func={aggr_func!r}; supported "
                f"{sorted(_AGGR_TOP_N_FUNCS)}"
            )
        if x is None:
            return pd.DataFrame()
        if sort_col is None:
            sort_col = x
        # R40 #196: an undeclared tie policy degrades to the deterministic
        # stable-instrument-key policy so the selection is never row-order
        # dependent; the compile-time gate (check_topk_contract) rejects an
        # undeclared policy in production.
        eff_tie = tie_policy if tie_policy is not None else TopKTiePolicy.STABLE_INSTRUMENT_KEY
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for idx in x.index:
            row_x = x.loc[idx]
            row_sort = sort_col.loc[idx] if idx in sort_col.index else row_x
            # R40 #196: FINITE_ONLY sample policy — ±Inf is not a valid
            # observation (a top-k that ranks Inf above every finite value is
            # not economically meaningful).  np.isfinite (not notna) enforces it.
            valid_mask = pd.Series(
                np.isfinite(row_x.to_numpy(dtype=np.float64, copy=False))
                & np.isfinite(row_sort.to_numpy(dtype=np.float64, copy=False)),
                index=x.columns,
            )
            if int(valid_mask.sum()) == 0:
                continue
            valid_x = row_x[valid_mask]
            valid_sort = row_sort[valid_mask]
            sorted_cols = _select_top_k_with_tie_policy(
                valid_sort, int(top), ascending=bool(asc), tie_policy=eff_tie
            )
            selected = valid_x[sorted_cols]
            if aggr == "sum":
                result.loc[idx, sorted_cols] = selected.sum()
            elif aggr == "avg" or aggr == "mean":
                result.loc[idx, sorted_cols] = selected.mean()
            elif aggr == "max":
                result.loc[idx, sorted_cols] = selected.max()
            elif aggr == "min":
                result.loc[idx, sorted_cols] = selected.min()
            elif aggr == "std":
                result.loc[idx, sorted_cols] = selected.std()
            else:  # "count"
                result.loc[idx, sorted_cols] = selected.count()
        return result



# canonical=cum_top_n_avg backend=pandas_numpy selected=cum_top_n_avg source=time_series/topn_ops.py
@register_operator(name="cum_top_n_avg", category="time_series", business_category="time_series", canonical="cum_top_n_avg", source="factor_dsl_np")
class CumTopNAvg(SeriesOperator):
    """扩展窗口内前 N 大值的均值。"""

    metadata = OperatorMetadata(
        name="cum_top_n_avg", category="time_series",
        description="累积前N大值均值",
        examples=["cum_top_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "cumulative", "top_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return cum_top_n_mean(x, n)



# canonical=cum_top_n_sum backend=pandas_numpy selected=cum_top_n_sum source=time_series/topn_ops.py
@register_operator(name="cum_top_n_sum", category="time_series", business_category="time_series", canonical="cum_top_n_sum", source="factor_dsl_np")
class CumTopNSum(SeriesOperator):
    """扩展窗口内前 N 大值的求和。"""

    metadata = OperatorMetadata(
        name="cum_top_n_sum", category="time_series",
        description="累积前N大值求和",
        examples=["cum_top_n_sum(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "cumulative", "top_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return cum_top_n_sum(x, n)



# canonical=decay_linear backend=pandas_numpy selected=ts_decay_linear source=time_series/ts_ops.py
@register_operator(name="ts_decay_linear", category="time_series", business_category="time_series", canonical="ts_decay_linear", source="factor_dsl_np")
class TSDecayLinear(SeriesOperator):
    """线性衰减加权滚动平均（``ts_decay_linear`` canonical）。"""

    metadata = OperatorMetadata(
        name="ts_decay_linear", category="time_series",
        description="线性加权滚动平均",
        examples=["ts_decay_linear(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "linear"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_linear_weighted(x, window)

# aliases: DECAY_LINEAR, TS_DECAY_LINEAR



# canonical=ema backend=pandas_numpy selected=EMA source=time_series/m_ops.py
@register_operator(name="EMA", category="time_series", business_category="time_series", canonical="ts_ema", source="factor_dsl_np")
class EMA(SeriesOperator):
    """指数移动平均（EWM，``adjust=False``）。"""

    metadata = OperatorMetadata(
        name="EMA",
        category="time_series",
        description="计算n期指数移动平均",
        examples=["EMA(close, 12)", "EMA(close, 26)"],
        param_names=["x", "span"],
        return_type="series",
        tags=["time_series", "ema", "exponential"]
    )

    def _calculate_series(self, x: pd.DataFrame, span: int = 12, **kwargs) -> pd.DataFrame:
        # R40 #194: the EWM policy comes from the declared EWMContract (single
        # source of truth) instead of a hard-coded ``ewm(span=span, adjust=False)``.
        contract = ewm_contract_for("ts_ema")
        check_ewm_contract(contract, canonical="ts_ema")
        assert contract is not None
        return x.ewm(**contract.to_kwargs(span=float(span))).mean()



# canonical=ts_argmax backend=pandas_numpy selected=ts_argmax source=time_series/m_ops.py
@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="factor_dsl_np")
class TSArgmax(SeriesOperator):
    """滚动窗口内最大值距当前 bar 的 bar 数（age，0=当前/最新 bar，并列取最新）。

    R19-039..042/049 origin/tie 收敛：``ts_argmax`` canonical 语义为 **age**
    （0=当前/最新 bar），与共享 kernel ``rolling_days_since_extreme``、GTJA
    compat 以及 SQL 后端一致；并列极值取 **最新** occurrence。需要
    "0=窗口最旧 bar" 的 index 语义请用 canonical ``ts_argmax_index_from_oldest``。

    唯一 canonical 参数为 ``window``；``d`` 是解析层别名（``param_aliases``），
    映射到 ``window``。所有参数经 binder 规范化后进入 kernel，kernel 不再做
    本地 ``int(...)`` cast。"""

    metadata = OperatorMetadata(
        name="ts_argmax",
        category="time_series",
        description="窗口最大值距当前 bar 的 bar 数（0=当前，并列取最近）",
        examples=["ts_argmax(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "argmax"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_days_since_extreme
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # ``d`` is a declared parser-level alias for ``window`` (param_aliases);
        # the central gate validates it against window's ParamSpec, so no local
        # ``int(...)`` cast is needed here.
        w = strict_int(kwargs.get("d", window), "window", minimum=1)
        return rolling_days_since_extreme(x, w, maximum=True)



# canonical=ts_argmin backend=pandas_numpy selected=ts_argmin source=time_series/m_ops.py
@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="factor_dsl_np")
class TSArgmin(SeriesOperator):
    """滚动窗口内最小值距当前 bar 的 bar 数（age，0=当前/最新 bar，并列取最新）。

    R19-039..042/049 origin/tie 收敛：``ts_argmin`` canonical 语义为 **age**
    （0=当前/最新 bar），并列极值取 **最新** occurrence。需要 "0=窗口最旧 bar"
    的 index 语义请用 canonical ``ts_argmin_index_from_oldest``。

    唯一 canonical 参数为 ``window``；``d`` 是解析层别名（``param_aliases``），
    映射到 ``window``。所有参数经 binder 规范化后进入 kernel，kernel 不再做
    本地 ``int(...)`` cast。"""

    metadata = OperatorMetadata(
        name="ts_argmin",
        category="time_series",
        description="窗口最小值距当前 bar 的 bar 数（0=当前，并列取最近）",
        examples=["ts_argmin(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "argmin"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_days_since_extreme
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # ``d`` is a declared parser-level alias for ``window`` (param_aliases).
        w = strict_int(kwargs.get("d", window), "window", minimum=1)
        return rolling_days_since_extreme(x, w, maximum=False)



# canonical=m_beta backend=pandas_numpy selected=m_beta source=time_series/m_ops.py
@register_operator(name="m_beta", category="time_series", business_category="time_series", canonical="ts_beta", source="factor_dsl_np")
class MovingBeta(SeriesOperator):
    """两变量滚动 Beta：``Cov(y,x)/Var(x)``。

    R19-033..035: the SINGLE beta kernel authority is
    :func:`cleaned_operators._rolling_fast.rolling_beta` — shared by
    ``Beta`` / ``ts_beta`` / ``rolling_beta``.  ``min_periods`` default is the
    reviewed value ``5`` (2-sample slopes are not statistically meaningful) and
    paired-finite masking (``np.isfinite``, not ``notna``) keeps ``±Inf`` out of
    the observation count.
    """

    metadata = OperatorMetadata(
        name="m_beta",
        category="time_series",
        description="计算两变量的n期移动Beta",
        examples=["m_beta(returns, market, 20)"],
        param_names=["y", "x", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "moving", "beta"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=2, default=5, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        window: int = 20,
        min_periods: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(window, "window", minimum=2)
        mp = strict_int(min_periods, "min_periods", minimum=2)
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        return rolling_beta(y, x, window=w, min_periods=mp)



# canonical=m_bottom_n_avg backend=pandas_numpy selected=m_bottom_n_avg source=time_series/topn_ops.py
@register_operator(name="m_bottom_n_avg", category="time_series", business_category="time_series", canonical="ts_bottom_n_avg", source="factor_dsl_np")
class MovingBottomNAvg(SeriesOperator):
    """滚动窗口内后 N 小值的均值。"""

    metadata = OperatorMetadata(
        name="m_bottom_n_avg", category="time_series",
        description="滚动窗口内后N小值的均值",
        examples=["m_bottom_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "bottom_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_bottom_n_mean(x, n)



# canonical=m_bottom_n_sum backend=pandas_numpy selected=m_bottom_n_sum source=time_series/topn_ops.py
@register_operator(name="m_bottom_n_sum", category="time_series", business_category="time_series", canonical="ts_bottom_n_sum", source="factor_dsl_np")
class MovingBottomNSum(SeriesOperator):
    """滚动窗口内后 N 小值的求和。"""

    metadata = OperatorMetadata(
        name="m_bottom_n_sum", category="time_series",
        description="滚动窗口内后N小值的求和",
        examples=["m_bottom_n_sum(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "bottom_n", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_bottom_n_sum(x, n)



# canonical=m_mad backend=pandas_numpy selected=m_mad source=time_series/m_ops.py
@register_operator(name="m_mad", category="time_series", business_category="time_series", canonical="ts_mad", source="factor_dsl_np")
class MovingMAD(SeriesOperator):
    """滚动平均绝对离差（MAD）。"""

    metadata = OperatorMetadata(
        name="m_mad",
        category="time_series",
        description="计算n期移动平均绝对离差",
        examples=["m_mad(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "mad"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        median = x.rolling(window=window, min_periods=1).median()
        return (x - median).abs().rolling(window=window, min_periods=1).mean()



# canonical=m_median backend=pandas_numpy selected=m_median source=time_series/m_ops.py
@register_operator(name="m_median", category="time_series", business_category="time_series", canonical="ts_median", source="factor_dsl_np")
class MovingMedian(SeriesOperator):
    """滚动中位数。"""

    metadata = OperatorMetadata(
        name="m_median",
        category="time_series",
        description="计算n期移动中位数",
        examples=["m_median(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "median"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).median()



# canonical=ts_pct backend=pandas_numpy selected=ts_pct source=time_series/m_ops.py
@register_operator(name="ts_pct", category="time_series", business_category="time_series", canonical="ts_pct", source="factor_dsl_np")
class TSPctChange(SeriesOperator):
    """d 期变化率：``x_t / x_{t-d} - 1``。"""

    metadata = OperatorMetadata(
        name="ts_pct",
        category="time_series",
        description="d 期变化率",
        examples=["ts_pct(close, 1)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "pct_change", "pit_safe"]
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        periods = strict_integer(d, "d", minimum=1)
        prev = x.shift(periods)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = x / prev - 1.0
        return out.where(prev.notna() & (prev != 0))



# canonical=ts_log_return backend=pandas_numpy selected=ts_log_return source=time_series/m_ops.py
@register_operator(name="ts_log_return", category="time_series", business_category="time_series", canonical="ts_log_return", source="factor_dsl_np")
class TSLogReturn(SeriesOperator):
    """d 期对数收益率：``ln(x_t / x_{t-d})``。"""

    metadata = OperatorMetadata(
        name="ts_log_return",
        category="time_series",
        description="d 期对数收益率 ln(x_t / x_{t-d})",
        examples=["ts_log_return(close, 1)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "returns", "log", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 1, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        n = strict_integer(d, "d", minimum=1)
        prev = x.shift(n)
        # NEW-200: a log return requires CURRENT and PREVIOUS prices both STRICTLY
        # > 0.  The old code only replaced ``prev == 0`` — a negative price pair
        # (negative/negative -> positive ratio) then produced a legal ``log``
        # value from an invalid price, and 0/negative pairs silently became NaN
        # with no data-quality signal.  Price <= 0 is not a missing value; it is
        # invalid input, censored to NaN (PositivePrice domain).
        valid = (x > 0) & (prev > 0)
        ratio = x / prev
        result = np.log(ratio.where(valid & np.isfinite(ratio)))
        return result.replace([np.inf, -np.inf], np.nan)



# canonical=ts_sharpe backend=pandas_numpy selected=ts_sharpe source=time_series/m_ops.py
@register_operator(name="ts_sharpe", category="time_series", business_category="time_series", canonical="ts_sharpe", source="factor_dsl_np")
class TSSharpe(SeriesOperator):
    """滚动夏普比率（年化，可配置 ``ann_factor``）。"""

    metadata = OperatorMetadata(
        name="ts_sharpe",
        category="time_series",
        description="滚动夏普：mean/std * sqrt(ann_factor)",
        examples=["ts_sharpe(returns, 60)"],
        # R19-056/057: min_periods is NOT exposed on the public surface — the
        # central ``polars_daily_native`` polars backend pins a 3-param contract,
        # so declaring a 4th positional param on the pandas reference would trip
        # the R4-100 backend-arity audit.  The support floor stays a fixed
        # internal policy (max(2, window//3)); a caller-supplied min_periods is
        # rejected by the strict unknown-kwarg gate.  ann_factor is a period-per-
        # year policy knob, NOT a search dimension.
        param_names=["x", "window", "ann_factor"],
        return_type="series",
        tags=["time_series", "sharpe", "pit_safe"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True,
                                param_role=ParamRole.HORIZON),
            "ann_factor": ParamSpec(dtype=float, min=1, default=252.0, searchable=False,
                                    param_role=ParamRole.SESSION_POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        ann_factor: float = 252.0,
        min_periods: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer

        w = strict_integer(window, "window", minimum=2)
        mp = (
            strict_integer(min_periods, "min_periods", minimum=2)
            if min_periods is not None
            else max(2, w // 3)
        )
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        # R19-058: annualization factor is the number of periods per year — it
        # must be strictly positive (>= 1).  ``ann_factor=0`` used to silently
        # zero out the whole Sharpe; the old ``minimum=0.0`` gate is tightened.
        scale = strict_finite_scalar(ann_factor, "ann_factor", minimum=1.0)
        mean = x.rolling(window=w, min_periods=mp).mean()
        std = x.rolling(window=w, min_periods=mp).std(ddof=1)
        zero_vol = std.eq(0) | std.isna()
        sharpe = mean / std.replace(0, np.nan)
        from factor_engine.backend.numeric_semantics import ts_sharpe_zero_std_is_null

        if ts_sharpe_zero_std_is_null():
            sharpe = sharpe.mask(zero_vol, np.nan)
        else:
            sharpe = sharpe.mask(zero_vol & mean.gt(0), np.inf)
            sharpe = sharpe.mask(zero_vol & mean.le(0), 0.0)
        return sharpe * np.sqrt(scale)


# canonical=ts_autocorr backend=pandas_numpy selected=ts_autocorr source=time_series/m_ops.py
@register_operator(
    name="ts_autocorr",
    category="time_series",
    business_category="time_series",
    canonical="ts_autocorr",
    source="factor_dsl_np")
class TSAutocorr(SeriesOperator):
    """滚动自相关系数：窗口内 ``corr(x, x.shift(lag))``。"""

    metadata = OperatorMetadata(
        name="ts_autocorr",
        category="time_series",
        description="滚动自相关 corr(x_t, x_{t-lag}) within window",
        examples=["ts_autocorr(returns, 20, 1)"],
        param_names=["x", "window", "lag", "min_periods"],
        return_type="series",
        tags=["time_series", "autocorrelation", "pit_safe"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True,
                             param_role=ParamRole.ECONOMIC),
            "min_periods": ParamSpec(dtype=int, min=2, default=None, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        lag: int = 1,
        min_periods: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        k = strict_integer(lag, "lag", minimum=1)
        if k >= w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("lag must be < window")
        mp = (
            strict_integer(min_periods, "min_periods", minimum=2)
            if min_periods is not None
            else max(2, w // 3)
        )
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        y = x.shift(k)
        return x.rolling(window=w, min_periods=mp).corr(y)


# canonical=ts_quantile backend=pandas_numpy selected=ts_quantile source=time_series/m_ops.py
@register_operator(name="ts_quantile", category="time_series", business_category="time_series", canonical="ts_quantile", source="factor_dsl_np")
class TSQuantile(SeriesOperator):
    """滚动窗口分位数 ``Q_q``。"""

    metadata = OperatorMetadata(
        name="ts_quantile",
        category="time_series",
        description="滚动窗口分位数",
        examples=["ts_quantile(returns, 20, 0.75)"],
        param_names=["x", "d", "q"],
        return_type="series",
        tags=["time_series", "moving", "percentile"],
        # R19-050: hidden ``window``/``p`` aliases are now explicit metadata
        # declarations (binder canonicalizes + validates them against d/q).
        param_aliases={"window": "d", "p": "q"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "q": ParamSpec(dtype=float, min=0, max=1, default=0.5, searchable=True,
                           param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, q: float = 0.5, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int, strict_probability

        # R19-050..052: ``window``/``p`` are declared aliases -> binder-validated
        # against the canonical specs; kernel only uses the canonical names and
        # never hand-casts ``float()``.  ``q`` must be strictly in [0,1].
        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        quantile = strict_probability(kwargs.get("p", q), "q")
        return x.rolling(window=w, min_periods=1).quantile(quantile)



# canonical=m_top_n_avg backend=pandas_numpy selected=m_top_n_avg source=time_series/topn_ops.py
@register_operator(name="m_top_n_avg", category="time_series", business_category="time_series", canonical="ts_top_n_avg", source="factor_dsl_np")
class MovingTopNAvg(SeriesOperator):
    """滚动窗口内前 N 大值的均值。"""

    metadata = OperatorMetadata(
        name="m_top_n_avg", category="time_series",
        description="滚动窗口内前N大值的均值",
        examples=["m_top_n_avg(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "average"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_mean(x, n)



# canonical=m_top_n_std backend=pandas_numpy selected=m_top_n_std source=time_series/topn_ops.py
@register_operator(name="m_top_n_std", category="time_series", business_category="time_series", canonical="ts_top_n_std", source="factor_dsl_np")
class MovingTopNStd(SeriesOperator):
    """滚动窗口内前 N 大值的标准差。"""

    metadata = OperatorMetadata(
        name="m_top_n_std", category="time_series",
        description="滚动窗口内前N大值的标准差",
        examples=["m_top_n_std(volume, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "top_n", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_std(x, n)



# canonical=ts_topk_sum backend=pandas_numpy selected=ts_topk_sum source=time_series/topn_ops.py
@register_operator(name="ts_topk_sum", category="time_series", business_category="time_series", canonical="ts_topk_sum", source="factor_dsl_np")
class TSTopKSum(SeriesOperator):
    """滚动窗口内 Top-K 求和。"""

    metadata = OperatorMetadata(
        name="ts_topk_sum", category="time_series",
        description="滚动窗口内 Top-K 求和",
        examples=["ts_topk_sum(volume, 20, 5)"],
        param_names=["x", "d", "k"], return_type="series",
        tags=["time_series", "top_n", "sum"],
        # R19-050: hidden ``window``/``n`` aliases declared explicitly.
        param_aliases={"window": "d", "n": "k"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=None, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, d: int = 20, k: int | None = None, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.operator_errors import OperatorParameterError
        from factor_engine.cleaned_operators._rolling_fast import rolling_top_n_sum_window
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # ``window``/``n`` are declared aliases for ``d``/``k``; binder validates
        # them against the canonical specs, so no local ``int(...)`` cast.
        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        k_eff = k if k is not None else kwargs.get("n", d)
        top_k = strict_int(k_eff, "k", minimum=1)
        if top_k > w:
            raise OperatorParameterError("k must be <= window")
        return rolling_top_n_sum_window(x, w, top_k)



# canonical=m_var backend=pandas_numpy selected=m_var source=time_series/m_ops.py
@register_operator(name="m_var", category="time_series", business_category="time_series", canonical="ts_var", source="factor_dsl_np")
class MovingVariance(SeriesOperator):
    """滚动方差。

    Parameters:
    - window: rolling window size
    - ddof: Delta Degrees of Freedom. ddof=1 (default) for sample variance, ddof=0 for population variance
    - min_periods: minimum number of observations required (default=1)
    """

    metadata = OperatorMetadata(
        name="m_var",
        category="time_series",
        description="计算n期移动方差",
        examples=["m_var(returns, 20)", "m_var(returns, 20, ddof=0)"],
        param_names=["x", "window", "ddof", "min_periods"],
        return_type="series",
        tags=["time_series", "moving", "variance"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, ddof: int = 1, min_periods: int = 1, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=min_periods).var(ddof=ddof)



# 重复实现：见 ts_zscore；dedupe 注销
# @register_operator(name="m_zscore", category="time_series", business_category="time_series", canonical="m_zscore", source="factor_dsl_np")
class MovingZscore(SeriesOperator):
    """滚动窗口 Z-Score 标准化（dedupe 后别名指向 ``ts_zscore``）。"""

    metadata = OperatorMetadata(
        name="m_zscore",
        category="time_series",
        description="移动窗口内的Z-Score标准化",
        examples=["m_zscore(close, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "zscore"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std().replace(0, 1)
        return (x - mean) / std



# canonical=tm_top_n_avg backend=pandas_numpy selected=tm_top_n_avg source=time_series/topn_ops.py
@register_operator(name="tm_top_n_avg", category="time_series", business_category="time_series", canonical="tm_top_n_avg", source="factor_dsl_np")
class TimeWindowTopNAvg(SeriesOperator):
    """指定时间窗口内前 N 大值的均值。"""

    metadata = OperatorMetadata(
        name="tm_top_n_avg", category="time_series",
        description="时间窗口内前N大值的均值",
        examples=["tm_top_n_avg(close, 20, 5)"],
        param_names=["x", "time_window", "n"], return_type="series",
        tags=["time_series", "top_n", "average", "time_window"]
    )
    def _calculate_series(self, x: pd.DataFrame, time_window: int = 20, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_mean_window(x, time_window, n)



# canonical=tm_top_n_sum backend=pandas_numpy selected=tm_top_n_sum source=time_series/topn_ops.py
@register_operator(name="tm_top_n_sum", category="time_series", business_category="time_series", canonical="tm_top_n_sum", source="factor_dsl_np")
class TimeWindowTopNSum(SeriesOperator):
    """指定时间窗口内前 N 大值的求和。"""

    metadata = OperatorMetadata(
        name="tm_top_n_sum", category="time_series",
        description="时间窗口内前N大值的求和",
        examples=["tm_top_n_sum(close, 20, 5)"],
        param_names=["x", "time_window", "n"], return_type="series",
        tags=["time_series", "top_n", "sum", "time_window"]
    )
    def _calculate_series(self, x: pd.DataFrame, time_window: int = 20, n: int = 5, **kwargs) -> pd.DataFrame:
        return rolling_top_n_sum_window(x, time_window, n)



# canonical=ts_corr backend=pandas_numpy selected=ts_corr source=time_series/ts_ops.py

# helper for ts_corr
class TSCorrelation(SeriesOperator):
    """滚动 Pearson 相关系数（``ts_corr`` 基类）。"""

    metadata = OperatorMetadata(
        name="ts_correlation", category="time_series",
        description="滚动相关系数 (与m_cor相同)",
        examples=["ts_correlation(close, volume, 20)"],
        param_names=["x", "y", "window", "min_periods"], return_type="series",
        tags=["time_series", "ts_", "correlation"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20, min_periods: int = 2, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(window, "window", minimum=2)
        mp = strict_int(min_periods, "min_periods", minimum=2)
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        # R19-030: Numba fastpath and pandas slow path share the SAME
        # current-row policy (see _rolling_fast.CURRENT_ROW_POLICY): a window
        # statistic may exist at a row whose current x/y pair is missing, as
        # long as >= min_periods finite pairs are present.  The old slow path
        # forced current-row-missing -> NaN via ``.where(valid)``, contradicting
        # the Numba kernel.  ``.where(valid)`` is removed so both paths only NaN
        # when the window's valid pairs are insufficient.
        use_numba = os.environ.get("FACTOR_ENGINE_USE_NUMBA", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if use_numba:
            try:
                from factor_engine.backend.numba_kernels import rolling_corr_panel

                fast = rolling_corr_panel(
                    x.to_numpy(dtype=float),
                    y.to_numpy(dtype=float),
                    w,
                    min_count=mp,
                )
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        # Pairwise rolling semantics treat both NaN and +/-Inf as missing,
        # matching the finite-pair contract used by the Polars implementation.
        x_finite = x.where(np.isfinite(x))
        y_finite = y.where(np.isfinite(y))
        return x_finite.rolling(window=w, min_periods=mp).corr(y_finite)

@register_operator(name="ts_corr", category="time_series", business_category="time_series", canonical="ts_corr", source="factor_dsl_np")
class TSCorr(TSCorrelation):
    """滚动相关系数（``ts_corr`` canonical）。"""

    metadata = OperatorMetadata(
        name="ts_corr", category="time_series",
        description="滚动相关系数 (ts_correlation的别名)",
        examples=["ts_corr(close, volume, 20)"],
        param_names=["x", "y", "window", "min_periods"], return_type="series",
        tags=["time_series", "ts_", "corr"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=2, default=2, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

# aliases: TS_CORR, correlation, m_cor, ts_correlation



# canonical=ts_cov backend=pandas_numpy selected=ts_cov source=time_series/ts_ops.py
@register_operator(name="ts_cov", category="time_series", business_category="time_series", canonical="ts_cov", source="factor_dsl_np")
class TSCov(SeriesOperator):
    """滚动协方差。"""

    metadata = OperatorMetadata(
        name="ts_cov", category="time_series",
        description="滚动协方差 (与m_cov相同)",
        examples=["ts_cov(returns, market, 20)"],
        param_names=["x", "y", "window", "ddof", "min_periods"], return_type="series",
        tags=["time_series", "ts_", "cov"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "ddof": ParamSpec(dtype=int, min=0, default=1),
            "min_periods": ParamSpec(dtype=int, min=1, default=None),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, window: int = 20,
                         ddof: int = 1, min_periods: int | None = None, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(window, "window", minimum=2)
        ddof_val = strict_int(ddof, "ddof", minimum=0)

        # Default min_periods to 2 if not specified (need at least 2 pairs)
        if min_periods is None:
            min_p = 2
        else:
            min_p = strict_int(min_periods, "min_periods", minimum=1)

        # R19-030: same current-row policy as ts_corr — a window covariance may
        # exist at a row whose current pair is missing (>= min_periods finite
        # pairs required).  ``.where(valid)`` removed for cross-backend parity.
        # Pandas rolling.cov supports ddof parameter directly
        return x.rolling(window=w, min_periods=min_p).cov(y, ddof=ddof_val)

# aliases: TS_COV, m_cov, ts_covariance



# 重复实现：见 ts_decay_linear；dedupe 注销
# @register_operator(name="ts_decay", ...)
class TSDecay(SeriesOperator):
    """线性衰减加权滚动（``ts_decay_linear`` 别名，dedupe 注销）。"""

    metadata = OperatorMetadata(
        name="ts_decay", category="time_series",
        description="衰减操作 (与ts_decay_linear相同)",
        examples=["ts_decay(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return rolling_linear_weighted(x, window)



# canonical=ts_decay_exp_window backend=pandas_numpy selected=ts_decay_exp_window source=time_series/ts_ops.py
@register_operator(name="ts_decay_exp_window", category="time_series", business_category="time_series", canonical="ts_decay_exp_window", source="factor_dsl_np")
class TSDecayExpWindow(SeriesOperator):
    """指数加权滚动平均。"""

    metadata = OperatorMetadata(
        name="ts_decay_exp_window", category="time_series",
        description="指数加权滚动",
        examples=["ts_decay_exp_window(volume, 10, 0.5)"],
        param_names=["x", "window", "alpha"], return_type="series",
        tags=["time_series", "ts_", "decay", "exp"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=10, searchable=True,
                                param_role=ParamRole.HORIZON),
            # R19-048: 0 < alpha <= 1.  alpha=0 would zero every weight except
            # the newest, alpha<0 flips weight signs and alpha>1 makes the OLDEST
            # bar dominate (reversing the decay direction).  Binder constrains to
            # [0,1]; kernel rejects the ``0`` endpoint (strictly > 0).
            "alpha": ParamSpec(dtype=float, min=0, max=1, default=0.5, searchable=True,
                               param_role=ParamRole.ECONOMIC),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 10, alpha: float = 0.5, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.operator_errors import OperatorParameterError
        from factor_engine.cleaned_operators._rolling_fast import rolling_age_weighted
        from factor_engine.cleaned_operators.common.strict_params import strict_float, strict_int

        w = strict_int(window, "window", minimum=1)
        a = strict_float(alpha, "alpha")
        if not (0.0 < a <= 1.0):
            raise OperatorParameterError(
                f"alpha must satisfy 0 < alpha <= 1, got {alpha!r}"
            )
        # R19-045/047: same shared age-weighted policy as ts_sum_decay.
        weights = np.array([a ** i for i in range(w)][::-1], dtype=np.float64)
        return rolling_age_weighted(x, w, weights)



# canonical=ts_delay backend=pandas_numpy selected=ts_delay source=time_series/ts_ops.py
@register_operator(name="ts_delay", category="time_series", business_category="time_series", canonical="ts_delay", source="factor_dsl_np")
class TSDelay(SeriesOperator):
    """n 期因果滞后（PIT-safe，负滞后返回 NaN）。"""

    metadata = OperatorMetadata(
        name="ts_delay", category="time_series",
        description="n期滞后 (与Ref相同)",
        examples=["ts_delay(close, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delay"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        return causal_lag(x, strict_integer(n, "n", minimum=0))

# aliases: DELAY, Delay, Ref, delay, m_delay, shift



# canonical=ts_delta backend=pandas_numpy selected=ts_delta source=time_series/ts_ops.py
@register_operator(name="ts_delta", category="time_series", business_category="time_series", canonical="ts_delta", source="factor_dsl_np")
class TSDelta(SeriesOperator):
    """n 期差分：``x - lag(x, n)``。"""

    metadata = OperatorMetadata(
        name="ts_delta", category="time_series",
        description="n期差分 (与Delta相同)",
        examples=["ts_delta(close, 1)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delta"]
    )
    def _calculate_series(self, x: pd.DataFrame, n: int = 1, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        return x - causal_lag(x, strict_integer(n, "n", minimum=1))

# aliases: Delta, Diff, TS_DELTA, pct_change



# canonical=ts_kurt backend=pandas_numpy selected=m_kurt source=time_series/m_ops.py
@register_operator(name="m_kurt", category="time_series", business_category="time_series", canonical="ts_kurt", source="factor_dsl_np")
class MovingKurt(SeriesOperator):
    """滚动峰度。"""

    metadata = OperatorMetadata(
        name="m_kurt",
        category="time_series",
        description="计算n期移动峰度",
        examples=["m_kurt(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "kurtosis"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).kurt()

# aliases: TS_KURT



# canonical=ts_max backend=pandas_numpy selected=ts_max source=time_series/ts_ops.py
@register_operator(name="ts_max", category="time_series", business_category="time_series", canonical="ts_max", source="factor_dsl_np")
class TSMax(SeriesOperator):
    """滚动最大值。"""

    metadata = OperatorMetadata(
        name="ts_max", category="time_series",
        description="滚动最大值 (与m_max相同)",
        examples=["ts_max(high, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "max"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).max()

# aliases: Max, TS_MAX, m_max, max



# canonical=ts_mean backend=pandas_numpy selected=ts_mean source=time_series/ts_ops.py
@register_operator(name="ts_mean", category="time_series", business_category="time_series", canonical="ts_mean", source="factor_dsl_np")
class TSMean(SeriesOperator):
    """滚动均值（支持 Numba 加速路径）。"""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.time_series:TSMean:v1",
        emitter_identity="pandas.rolling.mean:v1",
        parameter_domain_hash="ts_mean.window:int:min=1",
        semantic_contract_hash="ts_mean:min_periods=1:axis=time:v1",
    )

    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.routing import numba_enabled_for_op

        if numba_enabled_for_op("ts_mean", window=int(window)):
            try:
                from factor_engine.backend.numba_kernels import rolling_mean_panel

                fast = rolling_mean_panel(x.to_numpy(dtype=float), int(window), min_count=1)
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=1).mean()

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=pandas_numpy selected=ts_min source=time_series/ts_ops.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMin(SeriesOperator):
    """滚动最小值。"""

    metadata = OperatorMetadata(
        name="ts_min", category="time_series",
        description="滚动最小值 (与m_min相同)",
        examples=["ts_min(low, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "min"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).min()

# aliases: Min, TS_MIN, m_min, min



# canonical=ts_product backend=pandas_numpy selected=ts_product source=time_series/ts_ops.py
@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="factor_dsl_np")
class TSProduct(SeriesOperator):
    """滚动乘积（对数域累加实现）。"""

    metadata = OperatorMetadata(
        name="ts_product", category="time_series",
        description="滚动乘积（保留零与负号，signed + zero-safe）",
        examples=["ts_product(volume + 1, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "product"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 5, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_signed_product
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # R19-043/044: signed + zero-safe product.  The old log-domain
        # implementation broke on ``0`` (``[2,0,3]`` -> NaN) and negatives
        # (``[-2,-3]`` -> NaN).  ``rolling_signed_product`` maintains
        # zero_count / sign_parity / sum_log_abs -> ``[2,0,3]``=0,
        # ``[-2,-3]``=6, ``[-2,3]``=-6.
        w = strict_int(window, "window", minimum=1)
        return rolling_signed_product(x, w)



# canonical=ts_rank backend=pandas_numpy selected=ts_rank source=time_series/ts_ops.py
@register_operator(name="ts_rank", category="time_series", business_category="time_series", canonical="ts_rank", source="factor_dsl_np")
class TSRank(SeriesOperator):
    """滚动百分位排名（支持 Numba 加速路径）。"""

    metadata = OperatorMetadata(
        name="ts_rank", category="time_series",
        description="滚动排名 (与m_rank相同)",
        examples=["ts_rank(volume, 10)"],
        param_names=["x", "window", "min_periods"], return_type="series",
        tags=["time_series", "ts_", "rank"],
        # R19-124 hidden-kwargs 收口: ``min_periods`` 从隐藏 kwargs 收为声明参数
        # (SUPPORT_POLICY,非搜索维度)。
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=1, default=1, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.routing import numba_enabled_for_op
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        mp = strict_integer(min_periods, "min_periods", minimum=1)
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        if numba_enabled_for_op("ts_rank", window=w):
            try:
                from factor_engine.backend.numba_kernels import rolling_rank_pct_panel

                fast = rolling_rank_pct_panel(
                    x.to_numpy(dtype=float), w, min_count=mp
                )
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=w, min_periods=mp).rank(pct=True)

# aliases: TS_RANK, m_rank



# canonical=ts_regression backend=pandas_numpy selected=ts_regression source=time_series/ts_ops.py
@register_operator(name="ts_regression", category="time_series", business_category="time_series", canonical="ts_regression", source="factor_dsl_np")
class TSRegression(SeriesOperator):
    """滚动 OLS 回归（slope/intercept/r²/residual）。"""

    metadata = OperatorMetadata(
        name="ts_regression", category="time_series",
        description="滚动回归",
        examples=["ts_regression(returns, market, 252, 1, 'slope')"],
        param_names=["y", "x", "window", "lag", "retval"], return_type="series",
        tags=["time_series", "ts_", "regression"]
    )
    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 252,
                          lag: int = 0, retval: str = 'slope', **kwargs) -> pd.DataFrame:
        return rolling_regression(
            y, x, window=window, min_periods=3, lag=lag, retval=retval
        )

# aliases: TS_REGRESSION_SLOPE, ts_regression_slope



# canonical=ts_skew backend=pandas_numpy selected=m_skew source=time_series/m_ops.py
@register_operator(name="m_skew", category="time_series", business_category="time_series", canonical="ts_skew", source="factor_dsl_np")
class MovingSkew(SeriesOperator):
    """滚动偏度。"""

    metadata = OperatorMetadata(
        name="m_skew",
        category="time_series",
        description="计算n期移动偏度",
        examples=["m_skew(returns, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "moving", "skewness"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).skew()

# aliases: TS_SKEW



# canonical=ts_std backend=pandas_numpy selected=ts_std source=time_series/ts_ops.py

# helper for ts_std
class TSStdDev(SeriesOperator):
    """滚动标准差（``ts_std`` 基类）。"""

    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.routing import numba_enabled_for_op

        if numba_enabled_for_op("ts_std", window=int(window)):
            try:
                from factor_engine.backend.numba_kernels import rolling_std_panel

                fast = rolling_std_panel(x.to_numpy(dtype=float), int(window), min_count=1)
                if fast is not None:
                    return pd.DataFrame(fast, index=x.index, columns=x.columns)
            except Exception:
                pass
        return x.rolling(window=window, min_periods=1).std()

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStd(TSStdDev):
    """滚动标准差（``ts_std`` canonical）。"""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_std",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.PANDAS_REFERENCE,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.time_series:TSStd:v1",
        emitter_identity="pandas.rolling.std:v1",
        parameter_domain_hash="ts_std.window:int:min=1",
        semantic_contract_hash="ts_std:min_periods=1:axis=time:v1",
    )

    metadata = OperatorMetadata(
        name="ts_std", category="time_series",
        description="滚动标准差 (ts_std_dev的别名)",
        examples=["ts_std(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )

# aliases: Std, TS_STD, m_std, std, ts_std_dev, ts_stddev



# canonical=ts_sum backend=pandas_numpy selected=ts_sum source=time_series/ts_ops.py
@register_operator(name="ts_sum", category="time_series", business_category="time_series", canonical="ts_sum", source="factor_dsl_np")
class TSSum(SeriesOperator):
    """滚动求和。"""

    metadata = OperatorMetadata(
        name="ts_sum", category="time_series",
        description="滚动求和 (与m_sum相同)",
        examples=["ts_sum(volume, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "sum"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        return x.rolling(window=window, min_periods=1).sum()

# aliases: TS_SUM, m_sum



# canonical=ts_sum_decay backend=pandas_numpy selected=ts_sum_decay source=time_series/ts_ops.py
@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_np")
class TSSumDecay(SeriesOperator):
    """指数衰减权重滚动求和。"""

    metadata = OperatorMetadata(
        name="ts_sum_decay", category="time_series",
        description="衰减求和（加权归一；age-weighted 加权平均）",
        examples=["ts_sum_decay(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "sum"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_age_weighted
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # R19-045/046: age-weighted operators share ONE partial-window policy
        # (see _rolling_fast.AGE_WEIGHTED_PARTIAL_POLICY): oldest_to_newest
        # weights, partial windows use the newest-L age slots, missing values
        # are skipped + reweighted.  Weights are NOT pre-normalized — the kernel
        # renormalizes each window.
        w = strict_int(window, "window", minimum=1)
        weights = np.array([2 ** (i / w) for i in range(w)])
        return rolling_age_weighted(x, w, weights)



# canonical=ts_zscore backend=pandas_numpy selected=ts_zscore source=time_series/ts_ops.py
@register_operator(name="ts_zscore", category="time_series", business_category="time_series", canonical="ts_zscore", source="factor_dsl_np")
class TSZScore(SeriesOperator):
    """滚动 Z-Score 标准化。"""

    metadata = OperatorMetadata(
        name="ts_zscore", category="time_series",
        description="滚动Z-Score (与m_zscore相同)",
        examples=["ts_zscore(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "zscore"]
    )
    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.backend.numeric_semantics import zscore_zero_std_fill

        # Treat infinities as missing values, matching the finite-only
        # statistical contract used by the selectable Polars authority.
        finite_x = x.replace([np.inf, -np.inf], np.nan)
        mean = finite_x.rolling(window=window, min_periods=1).mean()
        std = finite_x.rolling(window=window, min_periods=1).std()

        # R40 Parity Fix: When std=0 or NULL, return zero_fill (0.0) to match Polars backend
        # and numeric_semantics policy. Must avoid division by zero entirely.
        zero_fill = zscore_zero_std_fill("ts_zscore")

        # Mask where std is valid (not null and not zero)
        valid_std_mask = (std.notna()) & (std != 0)

        # Initialize result as NaN everywhere
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)

        # Compute zscore only where std is valid and non-zero
        result[valid_std_mask] = ((finite_x - mean) / std)[valid_std_mask]

        # Where std is exactly zero (constant window), use zero_fill
        zero_std_mask = (std.notna()) & (std == 0) & finite_x.notna()
        result[zero_std_mask] = zero_fill

        return result


def _apply_colwise_kernel(x: pd.DataFrame, fn, **kwargs) -> pd.DataFrame:
    """对 panel 每列应用 numpy 一维内核函数。

    参数:
        x: 输入宽表 panel。
        fn: 接收一维 numpy 数组的核函数。
        **kwargs: 传给 ``fn`` 的额外关键字参数。

    返回:
        逐列计算后的 ``pd.DataFrame``。
    """
    return x.apply(lambda s: fn(s.values, **kwargs) if kwargs else fn(s.values))


@register_operator(name="price_spread_deviation", category="time_series", business_category="time_series", canonical="price_spread_deviation", source="factor_dsl_np")
class PriceSpreadDeviation(SeriesOperator):
    """相对窗口均值偏离度。"""

    metadata = OperatorMetadata(
        name="price_spread_deviation",
        category="time_series",
        description="相对窗口均值偏离: x / mean(x, d) - 1",
        examples=["price_spread_deviation(close, 20)"],
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "deviation"],
        # R19-050: hidden ``window`` alias declared explicitly.
        param_aliases={"window": "d"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import price_spread_deviation_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(kwargs.get("window", d), "d", minimum=1)
        return _apply_colwise_kernel(x, price_spread_deviation_, d=w)


@register_operator(name="ts_moment", category="time_series", business_category="time_series", canonical="ts_moment", source="factor_dsl_np")
class TSMoment(SeriesOperator):
    """滚动 k 阶中心矩。"""

    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="窗口 k 阶中心矩（k>=2；k=1 恒为 0，不具信息量）",
        examples=["ts_moment(close, 20, 3)"],
        param_names=["x", "d", "k"],
        return_type="series",
        tags=["time_series", "moment"],
        # R19-050..053: k>=2 — central moment of order 1 is identically 0.
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=2, default=3, searchable=True,
                           param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, k: int = 3, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_moment_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(
            x, ts_moment_, d=strict_int(d, "d", minimum=1), k=strict_int(k, "k", minimum=2)
        )


@register_operator(name="rank_corr", category="time_series", business_category="time_series", canonical="rank_corr", source="factor_dsl_np")
class RankCorr(SeriesOperator):
    """滚动窗口秩相关系数（时序，`d>0`）。

    R19-016..018 origin 收敛：``rank_corr`` 的 canonical 语义仅为**时序窗口**
    （窗口内独立 rank + paired finite，trailing window，prefix-invariant）。旧的
    ``d=0`` full-sample 截面广播（整段历史 rank → 对 t 时点使用 t+1..T 数据，
    full-sample look-ahead）已被彻底移除——需要逐日跨股票截面秩相关的算子请用
    独立的 ``cs_rank_corr`` canonical（GLOBAL_STATE 诊断，非个股 alpha）。
    """

    metadata = OperatorMetadata(
        name="rank_corr",
        category="time_series",
        description="滚动窗口秩相关系数（d>0 时序窗口；d=0 截面模式已移除，跨股票秩相关用 cs_rank_corr）",
        examples=["rank_corr(close, volume, 20)"],
        param_names=["x", "y", "d"],
        return_type="series",
        tags=["time_series", "correlation"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_rank_corr_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(d, "d", minimum=1)
        result = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            if col in y.columns:
                result[col] = ts_rank_corr_(x[col].values, y[col].values, w)
        return result


@register_operator(name="ts_poly2_coeff", category="time_series", business_category="time_series", canonical="ts_poly2_coeff", source="factor_dsl_np")
class TSPoly2Coeff(SeriesOperator):
    """时间二次拟合二次项系数。"""

    metadata = OperatorMetadata(
        name="ts_poly2_coeff",
        category="time_series",
        description="时间二次拟合二次项系数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "regression"],
        param_specs={
            "d": ParamSpec(dtype=int, min=3, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_poly2_coeff_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(x, ts_poly2_coeff_, d=strict_int(d, "d", minimum=3))


@register_operator(name="ts_poly2_resid", category="time_series", business_category="time_series", canonical="ts_poly2_resid", source="factor_dsl_np")
class TSPoly2Resid(SeriesOperator):
    """二次拟合窗口末残差。"""

    metadata = OperatorMetadata(
        name="ts_poly2_resid",
        category="time_series",
        description="二次拟合残差",
        param_names=["y", "x", "d"],
        return_type="series",
        tags=["time_series", "regression"],
        param_specs={
            "d": ParamSpec(dtype=int, min=3, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_poly2_resid_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        w = strict_int(d, "d", minimum=3)
        result = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
        for col in y.columns:
            if col in x.columns:
                result[col] = ts_poly2_resid_(y[col].values, x[col].values, w)
        return result


# R22-058: causal siblings of the in-sample ts_poly2_coeff/resid.  The model is
# fit on data STRICTLY <= t-1; the current observation is used only for
# evaluation (R22-054..055).  These are the mineable poly2 forms (DIRECT_ALPHA);
# the in-sample *_coeff/*_resid stay RESEARCH_TOOL.
@register_operator(name="ts_poly2_prior_coeff", category="time_series", business_category="time_series", canonical="ts_poly2_prior_coeff", source="factor_dsl_np")
class TSPoly2PriorCoeff(SeriesOperator):
    """二次拟合二次项系数（prior 窗口 [t-d, t-1]，严格因果）。"""

    metadata = OperatorMetadata(
        name="ts_poly2_prior_coeff",
        category="time_series",
        description="二次拟合二次项系数（拟合窗口 <= t-1，严格因果）",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "regression", "causal_prior"],
        param_specs={
            "d": ParamSpec(dtype=int, min=3, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_poly2_prior_coeff_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(x, ts_poly2_prior_coeff_, d=strict_int(d, "d", minimum=3))


@register_operator(name="ts_poly2_forecast_error", category="time_series", business_category="time_series", canonical="ts_poly2_forecast_error", source="factor_dsl_np")
class TSPoly2ForecastError(SeriesOperator):
    """二次拟合一步外预测误差（prior 拟合，当前观测仅评估）。"""

    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error",
        category="time_series",
        description="二次拟合一步外预测误差（拟合窗口 <= t-1，当前观测仅评估）",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "regression", "causal_prior"],
        param_specs={
            "d": ParamSpec(dtype=int, min=3, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_poly2_forecast_error_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(x, ts_poly2_forecast_error_, d=strict_int(d, "d", minimum=3))


@register_operator(name="ts_poly2_forecast_error_z", category="time_series", business_category="time_series", canonical="ts_poly2_forecast_error_z", source="factor_dsl_np")
class TSPoly2ForecastErrorZ(SeriesOperator):
    """标准化一步外预测误差（除以 in-sample 残差 std）。"""

    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error_z",
        category="time_series",
        description="标准化一步外预测误差（除以 in-sample 残差 std）",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "regression", "causal_prior"],
        param_specs={
            "d": ParamSpec(dtype=int, min=3, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_poly2_forecast_error_z_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(x, ts_poly2_forecast_error_z_, d=strict_int(d, "d", minimum=3))


@register_operator(name="digital_count", category="time_series", business_category="time_series", canonical="digital_count", source="factor_dsl_np")
class DigitalCount(SeriesOperator):
    """连续小波动片段计数。"""

    metadata = OperatorMetadata(
        name="digital_count",
        category="time_series",
        description="连续小波动片段计数",
        param_names=["x", "d", "threshold", "run"],
        return_type="series",
        tags=["time_series", "microstructure"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, min=0, default=0.01, searchable=True,
                                   param_role=ParamRole.STATE_THRESHOLD),
            "run": ParamSpec(dtype=int, min=1, default=3, searchable=True,
                             param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, d: int = 20, threshold: float = 0.01, run: int = 3, **kwargs
    ) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import digital_count_
        from factor_engine.cleaned_operators.common.strict_params import strict_float, strict_int

        return _apply_colwise_kernel(
            x,
            digital_count_,
            d=strict_int(d, "d", minimum=1),
            threshold=strict_float(threshold, "threshold", minimum=0),
            run=strict_int(run, "run", minimum=1),
        )


@register_operator(name="ts_max_buildup", category="time_series", business_category="time_series", canonical="ts_max_buildup", source="factor_dsl_np")
class TSMaxBuildup(SeriesOperator):
    """窗口内持续创新高次数。"""

    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="time_series",
        description="窗口内持续创新高次数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "momentum"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                           param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, d: int = 20, **kwargs) -> pd.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import ts_max_buildup_
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        return _apply_colwise_kernel(x, ts_max_buildup_, d=strict_int(d, "d", minimum=1))


try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore
from factor_engine.cleaned_operators.base_polars import (
    Operator as PolarsOperator,
    OperatorMetadata as PolarsOperatorMetadata,
    SeriesOperator as PolarsSeriesOperator,
    register_operator as register_polars_operator,
    apply_numba_rolling,
    apply_numba_zscore,
    apply_numba_rank,
)
# polars blocks below reuse names Operator/SeriesOperator/register_operator via aliases
Operator = PolarsOperator
OperatorMetadata = PolarsOperatorMetadata
SeriesOperator = PolarsSeriesOperator
register_operator = register_polars_operator


# canonical=decay_linear backend=polars selected=ts_decay_linear source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_decay_linear",
    category="time_series",
    business_category="time_series",
    canonical="ts_decay_linear",
    source="factor_dsl_np",
    backend="polars")
class TSDecayLinearPolars(SeriesOperator):
    """Polars 线性衰减加权滚动平均。"""

    metadata = OperatorMetadata(
        name="ts_decay_linear", category="time_series",
        description="线性加权滚动平均",
        examples=["ts_decay_linear(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "linear"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        wlen = max(1, int(window))
        weights = np.arange(1, wlen + 1, dtype=float)

        def linear_decay(s):
            arr = np.asarray(s, dtype=float)
            # 权重按完整窗口（含 NaN 位置）切片，再按有效掩码——与 pandas
            # ``_linear_weighted_1d_numpy`` 一致（NaN 位置权重丢弃，其余保持
            # 原位权重）。不能按有效数量重切片（会错配权重）。
            ww = weights[-len(arr):]
            valid = np.isfinite(arr)
            if not np.any(valid):
                return None
            seg = arr[valid]
            w = ww[valid]
            return float(np.dot(seg, w) / w.sum())

        return x.with_columns([
            pl.col(c).rolling_map(linear_decay, window_size=wlen, min_samples=1).alias(c)
            for c in numeric_cols
        ])

# aliases: DECAY_LINEAR, TS_DECAY_LINEAR



# Legacy Polars ts_corr registration intentionally removed (R2-P0-018).
# The repaired centered finite-pair implementation in polars_ts_rolling.py is
# the sole normal-bootstrap Polars authority; the pandas TSCorr above remains
# the reference/shared surface.

# aliases: TS_CORR, correlation, m_cor, ts_correlation


# Legacy Polars ts_cov registration intentionally removed (R2-P0-019).
# The repaired centered finite-pair implementation in polars_ts_rolling.py is
# the sole normal-bootstrap Polars authority; the pandas TSCov above remains
# the reference/shared surface.


# canonical=ts_decay backend=polars selected=ts_decay source=time_series/ts_ops_polars.py
# 重复实现：见 ts_decay_linear；dedupe 注销
# @register_operator(name="ts_decay", ...)
class TSDecayPolars(SeriesOperator):
    """Polars 线性衰减滚动（``ts_decay_linear`` 别名）。"""

    metadata = OperatorMetadata(
        name="ts_decay", category="time_series",
        description="衰减操作 (与ts_decay_linear相同)",
        examples=["ts_decay(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        return TSDecayLinearPolars()._calculate_series(x, window, **kwargs)



# canonical=ts_decay_exp_window backend=polars selected=ts_decay_exp_window source=time_series/ts_ops_polars.py
@register_operator(name="ts_decay_exp_window", category="time_series", business_category="time_series", canonical="ts_decay_exp_window", source="factor_dsl_np")
class TSDecayExpWindowPolars(SeriesOperator):
    """Polars 指数加权滚动。"""

    metadata = OperatorMetadata(
        name="ts_decay_exp_window", category="time_series",
        description="指数加权滚动",
        examples=["ts_decay_exp_window(volume, 10, 0.5)"],
        param_names=["x", "window", "alpha"], return_type="series",
        tags=["time_series", "ts_", "decay", "exp"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=10, searchable=True,
                                param_role=ParamRole.HORIZON),
            "alpha": ParamSpec(dtype=float, min=0, max=1, default=0.5, searchable=True,
                               param_role=ParamRole.ECONOMIC),
        },
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 10, alpha: float = 0.5, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.operator_errors import OperatorParameterError
        from factor_engine.cleaned_operators._rolling_fast import rolling_age_weighted
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.common.strict_params import strict_float, strict_int

        # R19-045/047/048: same shared age-weighted policy / alpha domain as the
        # pandas version, via the pandas bridge for exact numeric parity.
        w = strict_int(window, "window", minimum=1)
        a = strict_float(alpha, "alpha")
        if not (0.0 < a <= 1.0):
            raise OperatorParameterError(
                f"alpha must satisfy 0 < alpha <= 1, got {alpha!r}"
            )
        weights = np.array([a ** i for i in range(w)][::-1], dtype=np.float64)
        return panel_pandas_bridge(
            x, lambda pdf: rolling_age_weighted(pdf, w, weights)
        )



# canonical=ts_delay backend=polars selected=ts_delay source=time_series/ts_ops_polars.py
@register_operator(name="ts_delay", category="time_series", business_category="time_series", canonical="ts_delay", source="factor_dsl_np")
class TSDelayPolars(SeriesOperator):
    """Polars n 期滞后（负滞后返回 NaN）。"""

    metadata = OperatorMetadata(
        name="ts_delay", category="time_series",
        description="n期滞后 (与Ref相同)",
        examples=["ts_delay(close, 5)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delay"]
    )
    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        lag = strict_integer(n, "n", minimum=0)
        return x.with_columns([
            pl.col(c).shift(lag).alias(c) for c in numeric_cols
        ])

# aliases: DELAY, Delay, Ref, delay, m_delay, shift



# canonical=ts_delta backend=polars selected=ts_delta source=time_series/ts_ops_polars.py
@register_operator(name="ts_delta", category="time_series", business_category="time_series", canonical="ts_delta", source="factor_dsl_np")
class TSDeltaPolars(SeriesOperator):
    """Polars n 期差分。"""

    metadata = OperatorMetadata(
        name="ts_delta", category="time_series",
        description="n期差分 (与Delta相同)",
        examples=["ts_delta(close, 1)"],
        param_names=["x", "n"], return_type="series",
        tags=["time_series", "ts_", "delta"]
    )
    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        lag = strict_integer(n, "n", minimum=1)
        return x.with_columns([
            (pl.col(c) - pl.col(c).shift(lag)).alias(c) for c in numeric_cols
        ])

# aliases: Delta, Diff, TS_DELTA, pct_change



# canonical=ts_kurt backend=polars selected=ts_kurtosis source=time_series/ts_ops_polars.py
@register_operator(name="ts_kurtosis", category="time_series", business_category="time_series", canonical="ts_kurt", source="factor_dsl_np")
class TSKurtosisPolars(SeriesOperator):
    """Polars rolling kurtosis with pandas ``rolling.kurt`` semantics.

    Constant full windows return ``-3.0`` to preserve the canonical pandas
    authority's established degenerate-window behavior.
    """

    metadata = OperatorMetadata(
        name="ts_kurtosis", category="time_series",
        description="滚动峰度 (与m_kurt相同)",
        examples=["ts_kurtosis(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "kurtosis"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge

        w = int(kwargs.get("d", window))
        return panel_pandas_bridge(x, lambda pdf: pdf.rolling(window=w, min_periods=1).kurt())

# aliases: TS_KURT



# canonical=ts_max backend=polars selected=ts_max source=time_series/ts_ops_polars.py
@register_operator(name="ts_max", category="time_series", business_category="time_series", canonical="ts_max", source="factor_dsl_np")
class TSMaxPolars(SeriesOperator):
    """Polars 滚动最大值。"""

    metadata = OperatorMetadata(
        name="ts_max", category="time_series",
        description="滚动最大值 (与m_max相同)",
        examples=["ts_max(high, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "max"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_max(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Max, TS_MAX, m_max, max



# canonical=ts_mean backend=polars selected=ts_mean source=time_series/ts_ops_polars.py
@register_operator(name="ts_mean", category="time_series", business_category="time_series", canonical="ts_mean", source="factor_dsl_np")
class TSMeanPolars(SeriesOperator):
    """Polars 滚动均值。"""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=False,
        stateful=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.time_series:TSMeanPolars:v1",
        emitter_identity="polars.Expr.rolling_mean:v1",
        parameter_domain_hash="ts_mean.window:int:min=1",
        semantic_contract_hash="ts_mean:min_samples=1:axis=time:v1",
    )

    metadata = OperatorMetadata(
        name="ts_mean", category="time_series",
        description="滚动均值 (与m_avg相同)",
        examples=["ts_mean(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "mean"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Mean, TS_MEAN, m_avg, mean



# canonical=ts_min backend=polars selected=ts_min source=time_series/ts_ops_polars.py
@register_operator(name="ts_min", category="time_series", business_category="time_series", canonical="ts_min", source="factor_dsl_np")
class TSMinPolars(SeriesOperator):
    """Polars 滚动最小值。"""

    metadata = OperatorMetadata(
        name="ts_min", category="time_series",
        description="滚动最小值 (与m_min相同)",
        examples=["ts_min(low, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "min"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_min(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: Min, TS_MIN, m_min, min



# canonical=ts_product backend=polars selected=ts_product source=time_series/ts_ops_polars.py
@register_operator(name="ts_product", category="time_series", business_category="time_series", canonical="ts_product", source="factor_dsl_np")
class TSProductPolars(SeriesOperator):
    """Polars 滚动乘积。"""

    metadata = OperatorMetadata(
        name="ts_product", category="time_series",
        description="滚动乘积（保留零与负号，signed + zero-safe）",
        examples=["ts_product(volume + 1, 5)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "product"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=5, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_signed_product
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # R19-043/044: signed + zero-safe product, synced to the pandas kernel.
        w = strict_int(window, "window", minimum=1)
        return panel_pandas_bridge(
            x, lambda pdf: rolling_signed_product(pdf, w)
        )



# canonical=ts_rank backend=polars selected=ts_rank source=time_series/ts_ops_polars.py
@register_operator(name="ts_rank", category="time_series", business_category="time_series", canonical="ts_rank", source="factor_dsl_np")
class TSRankPolars(SeriesOperator):
    """Polars 滚动百分位排名。"""

    metadata = OperatorMetadata(
        name="ts_rank", category="time_series",
        description="滚动排名 (与m_rank相同)",
        examples=["ts_rank(volume, 10)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "rank"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        numeric_cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        # 使用 Polars 的 rolling_apply
        def _rolling_rank_pct(s):
            if len(s) == 0:
                return None
            ranks = s.rank(method="average")
            return float(ranks[-1] / len(s))

        return x.with_columns([
            pl.col(c).rolling_map(_rolling_rank_pct, window_size=window, min_periods=1).alias(c)
            for c in numeric_cols
        ])

# aliases: TS_RANK, m_rank



# canonical=ts_skew backend=polars selected=ts_skewness source=time_series/ts_ops_polars.py
@register_operator(name="ts_skewness", category="time_series", business_category="time_series", canonical="ts_skew", source="factor_dsl_np")
class TSSkewnessPolars(SeriesOperator):
    """Polars 滚动偏度。"""

    metadata = OperatorMetadata(
        name="ts_skewness", category="time_series",
        description="滚动偏度 (与m_skew相同)",
        examples=["ts_skewness(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "skewness"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_skew(window_size=w, bias=False, min_samples=1).alias(c)
            for c in cols
        ])

# aliases: TS_SKEW



# canonical=ts_std backend=polars selected=ts_std source=time_series/ts_ops_polars.py

# helper for ts_std
class TSStdDev(SeriesOperator):
    """滚动标准差 (与m_std相同)"""
    metadata = OperatorMetadata(
        name="ts_std_dev", category="time_series",
        description="滚动标准差 (与m_std相同)",
        examples=["ts_std_dev(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_std(window_size=window, min_samples=1).alias(c) for c in cols
        ])

@register_operator(name="ts_std", category="time_series", business_category="time_series", canonical="ts_std", source="factor_dsl_np")
class TSStdPolars(TSStdDev):
    """Polars 滚动标准差。"""

    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_std",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=False,
        stateful=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.time_series:TSStdPolars:v1",
        emitter_identity="polars.Expr.rolling_std:v1",
        parameter_domain_hash="ts_std.window:int:min=1",
        semantic_contract_hash="ts_std:min_samples=1:axis=time:v1",
    )

    metadata = OperatorMetadata(
        name="ts_std", category="time_series",
        description="滚动标准差 (ts_std_dev的别名)",
        examples=["ts_std(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "std"]
    )

# aliases: Std, TS_STD, m_std, std, ts_std_dev, ts_stddev



# canonical=ts_sum backend=polars selected=ts_sum source=time_series/ts_ops_polars.py
@register_operator(name="ts_sum", category="time_series", business_category="time_series", canonical="ts_sum", source="factor_dsl_np")
class TSSumPolars(SeriesOperator):
    """Polars 滚动求和。"""

    metadata = OperatorMetadata(
        name="ts_sum", category="time_series",
        description="滚动求和 (与m_sum相同)",
        examples=["ts_sum(volume, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "sum"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in x.columns if c not in ['date', 'stock_code']]
        return x.with_columns([
            pl.col(c).rolling_sum(window_size=window, min_samples=1).alias(c) for c in cols
        ])

# aliases: TS_SUM, m_sum



# canonical=ts_sum_decay backend=polars selected=ts_sum_decay source=time_series/ts_ops_polars.py
@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_np")
class TSSumDecayPolars(SeriesOperator):
    """Polars 衰减权重滚动求和。"""

    metadata = OperatorMetadata(
        name="ts_sum_decay", category="time_series",
        description="衰减求和（加权归一；age-weighted 加权平均）",
        examples=["ts_sum_decay(returns, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "decay", "sum"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
        },
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._rolling_fast import rolling_age_weighted
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.common.strict_params import strict_int

        # R19-045/046: shared age-weighted policy (newest-L partial anchor +
        # skip-missing reweight), synced to the pandas kernel.
        w = strict_int(window, "window", minimum=1)
        weights = np.array([2 ** (i / w) for i in range(w)])
        return panel_pandas_bridge(
            x, lambda pdf: rolling_age_weighted(pdf, w, weights)
        )



# canonical=ts_zscore backend=polars selected=ts_zscore source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_zscore",
    category="time_series",
    business_category="time_series",
    canonical="ts_zscore",
    source="factor_dsl_np",
    backend="polars")
class TSZScorePolars(SeriesOperator):
    """Polars 滚动 Z-Score。"""

    metadata = OperatorMetadata(
        name="ts_zscore", category="time_series",
        description="滚动Z-Score (与m_zscore相同)",
        examples=["ts_zscore(close, 20)"],
        param_names=["x", "window"], return_type="series",
        tags=["time_series", "ts_", "zscore"]
    )
    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        if "min_periods" in kwargs:
            raise TypeError("ts_zscore does not expose min_periods")

        def zscore_fn(values: pl.Series) -> float:
            raw = np.asarray(values.to_numpy(), dtype=float)
            finite = raw[np.isfinite(raw)]
            if finite.size < 2:
                return np.nan
            current = raw[-1]
            if not np.isfinite(current):
                return np.nan
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1))
            if std == 0.0:
                return 0.0
            return float((current - mean) / std)

        numeric_cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        return x.with_columns([
            pl.col(c).rolling_map(
                zscore_fn, window_size=w, min_samples=1
            ).alias(c)
            for c in numeric_cols
        ])



# canonical=ts_sharpe backend=polars selected=ts_sharpe source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_sharpe",
    category="time_series",
    business_category="time_series",
    canonical="ts_sharpe",
    source="factor_dsl_np",
    backend="polars")
class TSSharpePolars(SeriesOperator):
    """Polars 滚动夏普比率。"""

    metadata = OperatorMetadata(
        name="ts_sharpe",
        category="time_series",
        description="滚动夏普：mean/std * sqrt(ann_factor)",
        examples=["ts_sharpe(returns, 60)"],
        param_names=["x", "window", "ann_factor"],
        return_type="series",
        tags=["time_series", "sharpe", "pit_safe"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=60, searchable=True,
                                param_role=ParamRole.HORIZON),
            "ann_factor": ParamSpec(dtype=float, min=1, default=252.0, searchable=False,
                                    param_role=ParamRole.SESSION_POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 60,
        ann_factor: float = 252.0,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer

        w = strict_integer(window, "window", minimum=2)
        mp = (
            strict_integer(min_periods, "min_periods", minimum=2)
            if min_periods is not None
            else max(2, w // 3)
        )
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        # R19-058: ann_factor is periods-per-year, must be strictly positive.
        ann_factor = strict_finite_scalar(ann_factor, "ann_factor", minimum=1.0)
        cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        sqrt_af = float(np.sqrt(ann_factor))
        return x.with_columns(
            [
                pl.when(
                    (pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1) == 0)
                    & (pl.col(c).rolling_mean(window_size=w, min_samples=mp) > 0)
                )
                .then(float("inf"))
                .when(pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1) == 0)
                .then(0.0)
                .otherwise(
                    pl.col(c).rolling_mean(window_size=w, min_samples=mp)
                    / pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1)
                )
                .mul(sqrt_af)
                .alias(c)
                for c in cols
            ]
        )


# canonical=ts_autocorr backend=polars selected=ts_autocorr source=time_series/ts_ops_polars.py
@register_operator(
    name="ts_autocorr",
    category="time_series",
    business_category="time_series",
    canonical="ts_autocorr",
    source="factor_dsl_np",
    backend="polars")
class TSAutocorrPolars(SeriesOperator):
    """Polars 滚动自相关。"""

    metadata = OperatorMetadata(
        name="ts_autocorr",
        category="time_series",
        description="滚动自相关 corr(x_t, x_{t-lag}) within window",
        examples=["ts_autocorr(returns, 20, 1)"],
        param_names=["x", "window", "lag", "min_periods"],
        return_type="series",
        tags=["time_series", "autocorrelation", "pit_safe"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                                param_role=ParamRole.HORIZON),
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True,
                             param_role=ParamRole.ECONOMIC),
            "min_periods": ParamSpec(dtype=int, min=2, default=None, searchable=False,
                                     param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        window: int = 20,
        lag: int = 1,
        min_periods: int | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        k = strict_integer(lag, "lag", minimum=1)
        if k >= w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("lag must be < window")
        mp = (
            strict_integer(min_periods, "min_periods", minimum=2)
            if min_periods is not None
            else max(2, w // 3)
        )
        if mp > w:
            from factor_engine.backend.operator_errors import OperatorParameterError
            raise OperatorParameterError("min_periods must be <= window")
        cols = [c for c in x.columns if c not in ["date", "stock_code"]]
        return x.with_columns(
            [
                pl.rolling_corr(
                    pl.col(c),
                    pl.col(c).shift(k),
                    window_size=w,
                    min_samples=mp,
                ).alias(c)
                for c in cols
            ]
        )


# R22-058: causal poly2 siblings are mineable composition/terminal alphas on the
# extended authoring surface (the in-sample ts_poly2_coeff/resid stay research
# tools).  Their DirectUse verdicts are DIRECT_ALPHA.
def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface
    _surface.extend_extended_only(
        ["ts_poly2_prior_coeff", "ts_poly2_forecast_error", "ts_poly2_forecast_error_z", "WMA"]
    )

_register_surface()
