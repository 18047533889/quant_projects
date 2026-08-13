# -*- coding: utf-8 -*-
"""Relation, index and event aggregation operators.

Input contract: every operator consumes *pre-aggregated* daily panels
(``timestamp x instrument``), one row per instrument-day.  No source-side
relation-row access happens here — ``storage/sources/relation`` remains the
owner of raw row aggregation.

The ``relation_hhi``/``relation_entropy``/``relation_topk_sum``/
``relation_rank_weighted_sum`` operators accept up to 10 *ranked* panels
(``s1`` = top holder, ``s2`` = second, …) so per-instrument concentration can
be computed from pre-aggregated rank columns.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    category: str,
    domain: str,
    unit: str,
    output_unit: str | None = None,
    param_specs: dict[str, ParamSpec] | None = None,
    relational_specs: list[RelationalParamSpec] | None = None,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    metadata = OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1", *extra_tags,
        ],
        output_unit=output_unit,
        param_specs=dict(param_specs or {}),
        relational_specs=list(relational_specs or []),
    )
    return metadata


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def strict_relation_align(*panels: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """R24-005: fail-closed strict-axes gate for multi-panel relation inputs.

    Public relation operators MUST pass every panel through this gate before any
    numpy pairing.  It forbids the silent alignment the R24 audit targets:
    ``reindex_like`` / ``reindex`` / ``align(outer|inner)`` / position-only
    ``.to_numpy()`` pairing.  Checks, against the FIRST panel:

    * DataFrame type for every panel;
    * index EXACT equality (not a reindexable subset / union);
    * columns EXACT equality (instrument identity — a shuffled column order is
      a mismatch, never auto-repaired);
    * unique index / unique columns (a duplicated axis is ambiguous);
    * index timezone consistency.

    Any violation raises ``TypeError``/``ValueError`` — never normalizes.
    """
    if not panels:
        raise ValueError("strict_relation_align requires at least one panel")
    base = panels[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError(
            f"relation panel 0 is {type(base).__name__}, expected pandas DataFrame"
        )
    for position, panel in enumerate(panels[1:], start=1):
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(
                f"relation panel {position} is {type(panel).__name__}, "
                "expected pandas DataFrame"
            )
        if not panel.index.equals(base.index):
            raise ValueError(
                f"relation panel {position} index != panel 0 "
                "(fail-closed; no silent reindex)"
            )
        if not panel.columns.equals(base.columns):
            raise ValueError(
                f"relation panel {position} columns != panel 0 "
                "(fail-closed; no silent reindex)"
            )
    if not base.index.is_unique:
        raise ValueError("relation panel index is not unique (fail-closed)")
    if not base.columns.is_unique:
        raise ValueError("relation panel columns are not unique (fail-closed)")
    try:
        tzs = {getattr(idx, "tz", None) for p in panels for idx in (p.index,)}
    except Exception:  # pragma: no cover - non-datetime index
        tzs = {None}
    if len(tzs) > 1:
        raise ValueError(
            "relation panels have inconsistent index timezones (fail-closed)"
        )
    return panels


class HolderRankMissingSemantic(str, Enum):
    """R24-015: why a holder-rank slot is empty.

    Only :attr:`STRUCTURAL_ZERO` / :attr:`OUTSIDE_TOP_K` may be treated as a
    numeric 0.  :attr:`NOT_REPORTED` / :attr:`SOURCE_MISSING` / :attr:`UNKNOWN`
    must fail closed (NaN / coverage fail) — a blank slot is NOT evidence of a
    zero holding.
    """

    STRUCTURAL_ZERO = "structural_zero"
    NOT_REPORTED = "not_reported"
    SOURCE_MISSING = "source_missing"
    OUTSIDE_TOP_K = "outside_top_k"
    UNKNOWN = "unknown"

    @classmethod
    def permits_zero(cls, value: str) -> bool:
        """R24-016: only structural zero / outside-top-k may count as 0."""
        return value in (cls.STRUCTURAL_ZERO.value, cls.OUTSIDE_TOP_K.value)


def _stack_panels(*panels: pd.DataFrame) -> np.ndarray:
    """Align positional panels to the first panel and stack to (N, rows, cols).

    Panels are financial panels (``timestamp x instrument``) that must share the
    exact same date axis and instrument columns.  A silent ``reindex`` could
    re-pair a row to a different date (or an instrument column to a different
    name) after an upstream misalignment, manufacturing a spurious concentration /
    mobility value.  Fail closed instead (R11 #122, hardened by R24-005).
    """
    panels = strict_relation_align(*panels)
    arrays = [np.asarray(p.to_numpy(dtype=float)) for p in panels]
    return np.stack(arrays, axis=0)


def _finite_weights(panel: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
    values = panel.to_numpy(dtype=float)
    out = np.where(np.isfinite(values) & (values > threshold), values, np.nan)
    return out


_MISSING_SEMANTIC_CHOICES = (
    "structural_zero", "outside_top_k", "not_reported", "source_missing", "unknown",
)


def _rank_values(stacked: np.ndarray, missing_semantic: str) -> np.ndarray:
    """R24-016/017: apply the holder-rank missing semantics to stacked slots.

    ``structural_zero`` / ``outside_top_k`` → empty slots count as 0
    (structurally absent holders contribute nothing).  ``not_reported`` /
    ``source_missing`` / ``unknown`` → empty slots fail closed: NaN stays NaN
    and the aggregation mask excludes the cell (never a guessed 0).
    """
    if missing_semantic not in _MISSING_SEMANTIC_CHOICES:
        raise ValueError(
            f"missing_semantic must be one of {_MISSING_SEMANTIC_CHOICES!r}; "
            f"got {missing_semantic!r}"
        )
    if HolderRankMissingSemantic.permits_zero(missing_semantic):
        return np.nan_to_num(stacked, nan=0.0)
    # fail-closed: NaN stays NaN (it contributes neither to the sum nor the
    # share denominator, and any cell the caller must see as unknown stays NaN)
    return stacked


# ---------------------------------------------------------------------------
# Cross-sectional / rank aggregation operators (scope: cs)
# ---------------------------------------------------------------------------


@register_operator(
    name="relation_hhi",
    category="relation",
    business_category="relation",
    canonical="relation_hhi",
    source="relation.ops",
    status="experimental",
)
class RelationHhi(SeriesOperator):
    """按名次面板计算持股集中度 HHI = Σw_i²（w_i = s_i / Σs）。

    R24-012..014: this is the *observed-top-k* economic definition — the share
    denominator is the observed top-k subtotal, NOT the company's total shares.
    The company-ownership HHI (denominator = total shares) is the separate
    canonical ``holder_company_ownership_hhi``; the two are NEVER aliased.
    ``missing_semantic`` (R24-015..017) controls how an empty rank slot is
    treated: only ``structural_zero`` / ``outside_top_k`` may count as 0;
    ``not_reported`` / ``source_missing`` / ``unknown`` fail closed (NaN).
    """

    metadata = _metadata(
        "relation_hhi",
        "名次面板持股集中度 HHI（observed top-k 口径，分母=已观测 top-k 合计；"
        "公司总股本口径见 holder_company_ownership_hhi）。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
         "missing_semantic"],
        category="relation",
        domain="relation",
        unit="ratio",
        extra_tags=("output_domain:bounded_0_1",),
        param_specs={
            "missing_semantic": ParamSpec(
                dtype=str,
                choices=("structural_zero", "outside_top_k", "not_reported",
                         "source_missing", "unknown"),
                default="outside_top_k",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, *args: pd.DataFrame, missing_semantic: str = "outside_top_k", **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_hhi requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = _rank_values(stacked, missing_semantic)
        total = values.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = values / total if total != 0 else np.nan
            hhi = np.sum(shares * shares, axis=0)
        hhi = np.where(total > 0, hhi, np.nan)
        return _frame_like(base, hhi)


@register_operator(
    name="holder_observed_topk_hhi",
    category="relation",
    business_category="shareholder",
    canonical="holder_observed_topk_hhi",
    source="relation.ops",
    status="experimental",
)
class HolderObservedTopkHhi(SeriesOperator):
    """R24-011..013: 股东集中度 HHI，权重分母 = observed top-k subtotal。

    This is the second of the two distinct HHI economic definitions.  The first
    (``holder_company_ownership_hhi``) uses the company's total shares as the
    denominator; this one normalizes by the observed top-k subtotal first
    (Σ (s_i / Σ_topk s)²).  Per R24-014 the two are SEPARATE canonicals — never
    aliased to one another.  ``relation_hhi`` keeps the same formula as a
    relation-domain name; this is the explicit holder-domain canonical.
    """

    metadata = _metadata(
        "holder_observed_topk_hhi",
        "前十大股东观测 top-k 口径 HHI（分母=观测 top-k 合计）。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
         "missing_semantic"],
        category="relation",
        domain="shareholder",
        unit="ratio",
        extra_tags=("output_domain:bounded_0_1", "hhi_topk_denominator"),
        param_specs={
            "missing_semantic": ParamSpec(
                dtype=str,
                choices=_MISSING_SEMANTIC_CHOICES,
                default="outside_top_k",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, *args: pd.DataFrame, missing_semantic: str = "outside_top_k", **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("holder_observed_topk_hhi requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = _rank_values(stacked, missing_semantic)
        total = values.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = values / total if total != 0 else np.nan
            hhi = np.sum(shares * shares, axis=0)
        hhi = np.where(total > 0, hhi, np.nan)
        return _frame_like(base, hhi)


@register_operator(
    name="relation_entropy",
    category="relation",
    business_category="relation",
    canonical="relation_entropy",
    source="relation.ops",
    status="experimental",
)
class RelationEntropy(SeriesOperator):
    """按名次面板计算持股/权重分布熵（归一化到 [0,1]）。"""

    metadata = _metadata(
        "relation_entropy",
        "名次面板持股分布归一化熵。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
         "missing_semantic"],
        category="relation",
        domain="relation",
        unit="ratio",
        extra_tags=("output_domain:bounded_0_1",),
        param_specs={
            "missing_semantic": ParamSpec(
                dtype=str,
                choices=("structural_zero", "outside_top_k", "not_reported",
                         "source_missing", "unknown"),
                default="outside_top_k",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, *args: pd.DataFrame, missing_semantic: str = "outside_top_k", **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_entropy requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = _rank_values(stacked, missing_semantic)
        total = values.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = values / total if total != 0 else np.nan
            entropy = -np.sum(shares * np.log(np.where(shares > 0, shares, 1.0)), axis=0)
        count = (np.isfinite(values)).sum(axis=0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(count > 1, entropy / np.log(count), 0.0)
        return _frame_like(base, np.where(total > 0, normalized, np.nan))


@register_operator(
    name="relation_topk_sum",
    category="relation",
    business_category="relation",
    canonical="relation_topk_sum",
    source="relation.ops",
    status="experimental",
)
class RelationTopkSum(SeriesOperator):
    """前 K 名持股合计（K = 提供的名次面板数）。"""

    metadata = _metadata(
        "relation_topk_sum",
        "前 K 名持股合计。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
         "missing_semantic"],
        category="relation",
        domain="relation",
        # R11 #118: a plain sum carries the unit of its addends (the rank-panel
        # share values), NOT a uniform ratio.
        unit="same_as:value",
        output_unit="same_as:value",
        param_specs={
            "missing_semantic": ParamSpec(
                dtype=str,
                choices=("structural_zero", "outside_top_k", "not_reported",
                         "source_missing", "unknown"),
                default="outside_top_k",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, *args: pd.DataFrame, missing_semantic: str = "outside_top_k", **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_topk_sum requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = _rank_values(stacked, missing_semantic)
        total = np.nansum(values, axis=0)
        total = np.where(np.isfinite(values).sum(axis=0) > 0, total, np.nan)
        return _frame_like(base, total)


@register_operator(
    name="relation_rank_weighted_sum",
    category="relation",
    business_category="relation",
    canonical="relation_rank_weighted_sum",
    source="relation.ops",
    status="experimental",
)
class RelationRankWeightedSum(SeriesOperator):
    """逆名次权重平均：Σ (s_i / i) / Σ (1/i)。"""

    metadata = _metadata(
        "relation_rank_weighted_sum",
        "逆名次加权持股均值。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
         "missing_semantic"],
        category="relation",
        domain="relation",
        # R11 #118: a weighted MEAN carries the unit of the weighted values
        # (same_as:value), not a uniform ratio.
        unit="same_as:value",
        output_unit="same_as:value",
        param_specs={
            "missing_semantic": ParamSpec(
                dtype=str,
                choices=("structural_zero", "outside_top_k", "not_reported",
                         "source_missing", "unknown"),
                default="outside_top_k",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, *args: pd.DataFrame, missing_semantic: str = "outside_top_k", **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_rank_weighted_sum requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = _rank_values(stacked, missing_semantic)
        ranks = np.arange(1, stacked.shape[0] + 1, dtype=float)[:, None, None]
        weights = 1.0 / ranks if ranks != 0 else np.nan
        finite = np.isfinite(values)
        weighted = np.nansum(np.where(finite, values * weights, 0.0), axis=0)
        weight_sum = np.sum(np.where(finite, weights, 0.0), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = weighted / weight_sum if weight_sum != 0 else np.nan
        out = np.where(np.isfinite(values).sum(axis=0) > 0, out, np.nan)
        return _frame_like(base, out)


@register_operator(
    name="relation_category_share",
    category="relation",
    business_category="relation",
    canonical="relation_category_share",
    source="relation.ops",
    status="experimental",
)
class RelationCategoryShare(SeriesOperator):
    """个股 value 占同日期同类目 value 总和的比例。

    R24-008: the input ``value`` must be a non-negative activity / weight /
    amount (a "share" is only well-defined for non-negative weights).  For a
    signed contribution see ``relation_category_signed_contribution``.
    R24-010: the output is a true share → ``output_domain=bounded_0_1``.
    R24-004/005: the two panels are strict-aligned — a ``reindex_like`` /
    position-only pairing is forbidden and raises on any axis mismatch.
    """

    metadata = _metadata(
        "relation_category_share",
        "value 占同类别总和的比例（value 须为非负权重/活动量/金额）。",
        ["value", "category"],
        category="relation",
        domain="relation",
        unit="ratio",
        extra_tags=("output_domain:bounded_0_1",),
    )

    def _calculate_series(self, value: pd.DataFrame, category: pd.DataFrame, **_: Any) -> pd.DataFrame:
        value, category = strict_relation_align(value, category)
        vv = value.to_numpy(dtype=float)
        cv = category.to_numpy()
        if np.any((vv < 0) & np.isfinite(vv)):
            raise ValueError(
                "relation_category_share requires NON-NEGATIVE value input "
                "(R24-008); negative / signed contributions must use "
                "relation_category_signed_contribution"
            )
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            labels = pd.unique(cv[row])
            for label in labels:
                idx = cv[row] == label
                group_sum = float(np.nansum(vv[row][idx]))
                if not np.isfinite(group_sum) or group_sum == 0:
                    continue
                out[row][idx] = vv[row][idx] / group_sum
        return _frame_like(value, out)


@register_operator(
    name="relation_category_signed_contribution",
    category="relation",
    business_category="relation",
    canonical="relation_category_signed_contribution",
    source="relation.ops",
    status="experimental",
)
class RelationCategorySignedContribution(SeriesOperator):
    """组内 signed value 占同组绝对量合计的贡献（可 <0 或 >1，非 share）。"""

    metadata = _metadata(
        "relation_category_signed_contribution",
        "signed value / Σ|value| over category（输出可 <0 或 >1；非 share）。",
        ["value", "category"],
        category="relation",
        domain="relation",
        unit="ratio",
        # R24-009: the output of the SIGNED form is a contribution, not a share
        # — it may fall outside [0,1], so it must not be tagged bounded_0_1.
    )

    def _calculate_series(self, value: pd.DataFrame, category: pd.DataFrame, **_: Any) -> pd.DataFrame:
        value, category = strict_relation_align(value, category)
        vv = value.to_numpy(dtype=float)
        cv = category.to_numpy()
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            labels = pd.unique(cv[row])
            for label in labels:
                idx = cv[row] == label
                abs_sum = float(np.nansum(np.abs(vv[row][idx])))
                if not np.isfinite(abs_sum) or abs_sum == 0:
                    continue
                out[row][idx] = vv[row][idx] / abs_sum
        return _frame_like(value, out)


@register_operator(
    name="relation_peer_weighted_mean_ex_self",
    category="relation",
    business_category="relation",
    canonical="relation_peer_weighted_mean_ex_self",
    source="relation.ops",
    status="experimental",
)
class RelationPeerWeightedMeanExSelf(SeriesOperator):
    """组内除自身外其余成员的 value 权重加权均值（leave-one-out peer）。"""

    metadata = _metadata(
        "relation_peer_weighted_mean_ex_self",
        "组内除自身外其余成员的权重加权均值。",
        ["value", "weight", "group"],
        category="relation",
        domain="relation",
        # R11 #118: a weighted MEAN carries the unit of the weighted values
        # (same_as:value), not a uniform ratio.
        unit="same_as:value",
        output_unit="same_as:value",
    )

    def _calculate_series(self, value: pd.DataFrame, weight: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        value, weight, group = strict_relation_align(value, weight, group)
        vv = value.to_numpy(dtype=float)
        wv = weight.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            g_row = gv[row]
            for label in pd.unique(g_row):
                group_idx = np.flatnonzero(g_row == label)
                # 有效样本：value 与 weight 均有效且 weight > 0
                valid = (
                    np.isfinite(vv[row][group_idx])
                    & np.isfinite(wv[row][group_idx])
                    & (wv[row][group_idx] > 0)
                )
                valid_idx = group_idx[valid]
                if valid_idx.size == 0:
                    continue
                weights = wv[row][valid_idx]
                total_w = float(np.sum(weights))
                if not np.isfinite(total_w) or total_w <= 0.0:
                    continue
                weighted = float(np.sum(weights * vv[row][valid_idx]))
                for j in valid_idx:
                    own_w = wv[row][j]
                    denom = total_w - own_w
                    if denom <= 0.0:
                        continue
                    out[row][j] = (weighted - own_w * vv[row][j]) / denom
        return _frame_like(value, out)


# ---------------------------------------------------------------------------
# Per-instrument time-series relation/index/event operators (scope: ts)
# ---------------------------------------------------------------------------


@register_operator(
    name="relation_entry_count",
    category="relation",
    business_category="relation",
    canonical="relation_entry_count",
    source="relation.ops",
    status="experimental",
)
class RelationEntryCount(SeriesOperator):
    """窗口内 membership 从 0→1 的进入次数。"""

    metadata = _metadata(
        "relation_entry_count",
        "窗口内 0→1 进入次数。",
        ["member", "window", "missing_policy"],
        category="relation",
        domain="relation",
        unit="count",
        # R11 #120: ``missing_policy="false"`` turns unknown membership into
        # "not member", manufacturing false exits/entries.  It is NOT a
        # production membership interpretation: the call contract allows only
        # ``break`` (NaN breaks the transition sequence) and hides the legacy
        # ``false`` option from the search surface.
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "missing_policy": ParamSpec(choices=("break",), searchable=False),
        },
    )

    def _calculate_series(self, member: pd.DataFrame, window: int = 60, missing_policy: str = "break", **_: Any) -> pd.DataFrame:
        return _transition_count(member, window, forward=True, missing_policy=missing_policy)


@register_operator(
    name="relation_exit_count",
    category="relation",
    business_category="relation",
    canonical="relation_exit_count",
    source="relation.ops",
    status="experimental",
)
class RelationExitCount(SeriesOperator):
    """窗口内 membership 从 1→0 的退出次数。"""

    metadata = _metadata(
        "relation_exit_count",
        "窗口内 1→0 退出次数。",
        ["member", "window", "missing_policy"],
        category="relation",
        domain="relation",
        unit="count",
        # R11 #120: see RelationEntryCount — ``missing_policy`` is restricted to
        # ``break`` (the legacy ``false`` / NaN-as-nonmember interpretation is
        # explicitly non-production).
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "missing_policy": ParamSpec(choices=("break",), searchable=False),
        },
    )

    def _calculate_series(self, member: pd.DataFrame, window: int = 60, missing_policy: str = "break", **_: Any) -> pd.DataFrame:
        return _transition_count(member, window, forward=False, missing_policy=missing_policy)


def _transition_count(
    member: pd.DataFrame, window: int, forward: bool, missing_policy: str = "break"
) -> pd.DataFrame:
    w = int(window)
    if missing_policy not in {"break", "false"}:
        raise ValueError("missing_policy must be 'break' or 'false'")
    mv = member.to_numpy(dtype=float)
    rows, cols = mv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            segment = mv[start : row + 1, col]
            valid = np.isfinite(segment)
            if valid.sum() < 2:
                continue
            if missing_policy == "break":
                # 逐对有效：仅当相邻两 bar 均已知时才计一次进入/退出。NaN 打断序列，
                # 避免把缺失当作非成员而制造虚假转换（与 index_reconstitution_churn 一致）。
                current = segment[1:] != 0
                previous = segment[:-1] != 0
                pair_valid = valid[1:] & valid[:-1]
                entry = pair_valid & current & ~previous
                exit_ = pair_valid & ~current & previous
            else:  # legacy "false": NaN 视为非成员
                truth = valid & (segment != 0)
                entry = truth[1:] & ~truth[:-1]
                exit_ = ~truth[1:] & truth[:-1]
            transitions = np.sum(entry) if forward else np.sum(exit_)
            out[row, col] = float(transitions)
    return _frame_like(member, out)


@register_operator(
    name="relation_weighted_change",
    category="relation",
    business_category="relation",
    canonical="relation_weighted_change",
    source="relation.ops",
    status="experimental",
)
class RelationWeightedChange(SeriesOperator):
    """权重加权的值变化：(value - 前一期 value) * weight。"""

    metadata = _metadata(
        "relation_weighted_change",
        "(value - delay(value,1)) * weight。",
        ["value", "weight"],
        category="relation",
        domain="relation",
        # R11 #118: (Δvalue) * weight carries unit(value)*unit(weight) — the
        # product of the two input units, NOT a uniform ratio.
        unit="unit(value)*unit(weight)",
        output_unit="unit(value)*unit(weight)",
    )

    def _calculate_series(self, value: pd.DataFrame, weight: pd.DataFrame, **_: Any) -> pd.DataFrame:
        # R24-004/005: ``delta * weight`` would silently reindex on mismatched
        # axes — strict-align both panels first, then pair positions.
        value, weight = strict_relation_align(value, weight)
        vv = value.to_numpy(dtype=float)
        wv = weight.to_numpy(dtype=float)
        delta = np.full(vv.shape, np.nan, dtype=float)
        delta[1:, :] = vv[1:, :] - vv[:-1, :]
        return _frame_like(value, delta * wv)


# ---------------------------------------------------------------------------
# Index membership / weight operators
# ---------------------------------------------------------------------------


@register_operator(
    name="index_member",
    category="index",
    business_category="index",
    canonical="index_member",
    source="relation.ops",
    status="experimental",
)
class IndexMember(SeriesOperator):
    """指数成分标记：1 = 成分股，0 = 确定非成分股，NaN = 数据未知。"""

    metadata = _metadata(
        "index_member",
        "指数成分标记（member!=0→1，member==0→0）。",
        ["member"],
        category="index",
        domain="index",
        unit="boolean",
    )

    def _calculate_series(self, member: pd.DataFrame, **_: Any) -> pd.DataFrame:
        mv = member.to_numpy(dtype=float)
        out = np.where(np.isfinite(mv) & (mv != 0), 1.0, np.where(np.isfinite(mv), 0.0, np.nan))
        return _frame_like(member, out)


@register_operator(
    name="index_weight_change",
    category="index",
    business_category="index",
    canonical="index_weight_change",
    source="relation.ops",
    status="experimental",
)
class IndexWeightChange(SeriesOperator):
    """指数权重变化：weight - delay(weight, window)。"""

    metadata = _metadata(
        "index_weight_change",
        "指数权重变化 weight - delay(weight,window)。",
        ["weight", "window"],
        category="index",
        domain="index",
        unit="ratio",
        # review §17: strict-integer ``window`` — 5.1 must be rejected, never
        # silently ``int(window)``-truncated to 5.
        param_specs={"window": ParamSpec(dtype=int, min=1)},
    )

    def _calculate_series(self, weight: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        return weight - weight.shift(w)


@register_operator(
    name="index_entry_exit_event",
    category="index",
    business_category="index",
    canonical="index_entry_exit_event",
    source="relation.ops",
    status="experimental",
)
class IndexEntryExitEvent(SeriesOperator):
    """指数纳入/剔除事件：0→1 记 +1，1→0 记 -1（有符号状态，非布尔掩码）。"""

    metadata = _metadata(
        "index_entry_exit_event",
        "纳入 +1 / 剔除 -1（有符号状态事件；0=无事件）。",
        ["member"],
        category="index",
        domain="index",
        # R11 #119: the output is a SIGNED STATE event (-1 exit / 0 none /
        # +1 entry), not a boolean mask.  A boolean unit would let the searcher
        # treat -1 as truthy; the signed-state domain keeps the semantics.
        unit="state_signed",
        output_unit="state_signed",
        extra_tags=("signed_state",),
    )

    def _calculate_series(self, member: pd.DataFrame, **_: Any) -> pd.DataFrame:
        mv = member.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.zeros((rows, cols), dtype=float)
        for col in range(cols):
            prev_state: bool | None = None
            for row in range(rows):
                value = mv[row, col]
                if not np.isfinite(value):
                    out[row, col] = np.nan
                    prev_state = None
                    continue
                state = value != 0
                if prev_state is None:
                    out[row, col] = 0.0
                elif state and not prev_state:
                    out[row, col] = 1.0
                elif not state and prev_state:
                    out[row, col] = -1.0
                else:
                    out[row, col] = 0.0
                prev_state = state
        return _frame_like(member, out)


@register_operator(
    name="index_membership_age",
    category="index",
    business_category="index",
    canonical="index_membership_age",
    source="relation.ops",
    status="experimental",
)
class IndexMembershipAge(SeriesOperator):
    """距指数纳入以来经过的交易行数。

    R24-036..038: a provider gap (NaN membership) must NOT reset the entry age.
    ``last_confirmed_membership_state`` / ``last_confirmed_entry`` /
    ``unknown_gap_start`` are maintained per instrument.  When membership
    recovers to the same state after a gap, the entry age is NOT reset; and
    when the gap makes the exact entry timing unprovable, the output is NaN
    (fail-closed) unless ``output_mode="lower_bound"`` emits the floor age.
    """

    metadata = _metadata(
        "index_membership_age",
        "距纳入以来经过的行数（provider gap 不重置 entry age；gap 内不可证时 NaN）。",
        ["member", "max_lookback", "output_mode"],
        category="index",
        domain="index",
        unit="count",
        # review §17: ``max_lookback`` is a strict-integer row cap — 5.1 must be
        # rejected, never ``int(max_lookback)``-truncated.  ``None`` (the
        # declared default) means uncapped.
        param_specs={
            "max_lookback": ParamSpec(dtype=int, min=1, default=None),
            "output_mode": ParamSpec(
                dtype=str,
                choices=("exact", "lower_bound"),
                default="exact",
                searchable=False,
            ),
        },
    )

    def _calculate_series(self, member: pd.DataFrame, max_lookback: Any = None, output_mode: str = "exact", **_: Any) -> pd.DataFrame:
        limit = None if max_lookback is None else int(max_lookback)
        if output_mode not in ("exact", "lower_bound"):
            raise ValueError(f"output_mode must be 'exact' or 'lower_bound', got {output_mode!r}")
        mv = member.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            # R24-036: per-instrument confirmed-membership state machine.
            last_confirmed_state: bool | None = None
            last_confirmed_entry = -1
            in_gap = False
            gap_unresolved = False
            for row in range(rows):
                value = mv[row, col]
                if not np.isfinite(value):
                    # Provider gap begins (do NOT reset last_confirmed_entry).
                    if last_confirmed_state is not None and not in_gap:
                        in_gap = True
                    out[row, col] = np.nan
                    continue
                state = value != 0
                if in_gap:
                    in_gap = False
                    if last_confirmed_state is None:
                        # No prior confirmed state at all: cold-start.
                        last_confirmed_state = state
                        if state:
                            last_confirmed_entry = row
                        out[row, col] = np.nan
                        continue
                    if state == last_confirmed_state and state:
                        # Recovered to the SAME member state after a provider
                        # gap.  R24-037: never reset the entry age.  R24-038:
                        # whether an exit happened inside the gap is
                        # unprovable → exact age stays NaN (the ambiguity is
                        # permanent in a daily panel); lower_bound emits the
                        # floor age row - last_confirmed_entry.
                        if output_mode == "lower_bound":
                            distance = row - last_confirmed_entry
                            if limit is None or distance < limit:
                                out[row, col] = float(distance)
                        else:
                            gap_unresolved = True
                        continue
                    if state and not last_confirmed_state:
                        # Genuine entry happened during the gap; the exact entry
                        # row is unprovable → NaN (exact) / floor 0 (lower_bound).
                        last_confirmed_state = True
                        last_confirmed_entry = row
                        if output_mode == "lower_bound":
                            out[row, col] = 0.0
                        else:
                            gap_unresolved = True
                        continue
                    # state == False: exited (or never member) — no age; the
                    # confirmed non-member state ends the (possibly gap-unknown)
                    # member run and resets the entry anchor.
                    last_confirmed_state = False
                    last_confirmed_entry = -1
                    gap_unresolved = False
                    continue
                last_confirmed_state = state
                if state:
                    # R24-038: an unresolved gap keeps the exact entry age NaN.
                    if gap_unresolved and output_mode == "exact":
                        continue
                    if last_confirmed_entry < 0:
                        last_confirmed_entry = row
                    distance = row - last_confirmed_entry
                    if limit is None or distance < limit:
                        out[row, col] = float(distance)
                else:
                    # 已剔除：不再累计成分股年龄；已确认的非成员解除 gap 疑义。
                    out[row, col] = np.nan
                    last_confirmed_entry = -1
                    gap_unresolved = False
        return _frame_like(member, out)


# ---------------------------------------------------------------------------
# Event / timing operators
# ---------------------------------------------------------------------------

# review §17: every event operator's ``window`` / ``event_effective_lag`` are
# strict-integer row counts — 5.1 is rejected, never ``int(...)``-truncated.
# ``event_effective_lag`` is non-negative and must stay inside the accumulation
# window (a lag at or past the window degenerates the "since event" window).
_EVENT_WINDOW_PARAMS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=1),
    "event_effective_lag": ParamSpec(dtype=int, min=0),
}
_EVENT_WINDOW_RELATION = RelationalParamSpec(
    "event_effective_lag < window",
    "event operators require event_effective_lag < window "
    "(window={window}, event_effective_lag={event_effective_lag})",
)


@register_operator(
    name="event_cumulative_return_past",
    category="event",
    business_category="event",
    canonical="event_cumulative_return_past",
    source="relation.ops",
    status="experimental",
)
class EventCumulativeReturnPast(SeriesOperator):
    """事件触发后过去窗口内收益累计（causal）：Σ ret where event。"""

    metadata = _metadata(
        "event_cumulative_return_past",
        "事件触发后（含 event_effective_lag 错位）窗口内收益累计（算术和；复利见 event_return_since_last）。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        lag = max(0, int(event_effective_lag))
        ret, event = strict_relation_align(ret, event)
        rv = ret.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_event = -1
            for row in range(rows):
                if np.isfinite(ev[row, col]) and ev[row, col] != 0:
                    last_event = row
                if last_event >= 0:
                    start = last_event + lag
                    if start <= row and row - last_event < w + lag:
                        segment = rv[start : row + 1, col]
                        valid = np.isfinite(segment)
                        if valid.any():
                            out[row, col] = float(np.nansum(segment))
        return _frame_like(ret, out)


@register_operator(
    name="event_abnormal_return_past",
    category="event",
    business_category="event",
    canonical="event_abnormal_return_past",
    source="relation.ops",
    status="experimental",
)
class EventAbnormalReturnPast(SeriesOperator):
    """事件触发后窗口内超额收益累计：Σ (ret - benchmark_ret) where event。"""

    metadata = _metadata(
        "event_abnormal_return_past",
        "事件触发后（含 event_effective_lag 错位）窗口内超额收益累计（算术和）。",
        ["ret", "benchmark_ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, benchmark_ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        lag = max(0, int(event_effective_lag))
        ret, benchmark_ret, event = strict_relation_align(ret, benchmark_ret, event)
        rv = ret.to_numpy(dtype=float)
        bv = benchmark_ret.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        abnormal = rv - bv
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_event = -1
            for row in range(rows):
                if np.isfinite(ev[row, col]) and ev[row, col] != 0:
                    last_event = row
                if last_event >= 0:
                    start = last_event + lag
                    if start <= row and row - last_event < w + lag:
                        segment = abnormal[start : row + 1, col]
                        valid = np.isfinite(segment)
                        if valid.any():
                            out[row, col] = float(np.nansum(segment))
        return _frame_like(ret, out)


def _event_since_last_agg(
    ret: pd.DataFrame,
    event: pd.DataFrame,
    window: int,
    lag: int,
    mode: str,
) -> pd.DataFrame:
    """自最近一次事件（含 event_effective_lag 错位）起的聚合。

    ``mode``: ``"compounded"``=复利 Π(1+r)-1、``"sum"``=算术和 Σr、
    ``"logsum"``=Σln(1+r)、``"count"``=有效 bar 数。只统计窗口内的有限收益。
    """
    w = int(window)
    lag = max(0, int(lag))
    ret, event = strict_relation_align(ret, event)
    rv = ret.to_numpy(dtype=float)
    ev = event.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        last_event = -1
        for row in range(rows):
            if np.isfinite(ev[row, col]) and ev[row, col] != 0:
                last_event = row
            if last_event < 0:
                continue
            start = last_event + lag
            if start <= row and row - last_event < w + lag:
                segment = rv[start : row + 1, col]
                finite = np.isfinite(segment)
                if mode == "count":
                    out[row, col] = float(finite.sum())
                elif not finite.any():
                    continue
                elif mode == "compounded":
                    out[row, col] = float(np.prod(1.0 + segment[finite]) - 1.0)
                elif mode == "sum":
                    out[row, col] = float(np.sum(segment[finite]))
                elif mode == "logsum":
                    # ``ln(1+ret)`` is undefined for ret <= -1 (impossible
                    # price move or mis-scaled bp input).  Drop those bars and
                    # output NaN when nothing valid remains (review §7.7);
                    # invalid returns must never produce -inf/NaN contamination.
                    valid = np.isfinite(segment) & (segment > -1.0)
                    if not valid.any():
                        continue
                    out[row, col] = float(np.sum(np.log1p(segment[valid])))
    return _frame_like(ret, out)


@register_operator(
    name="event_return_since_last",
    category="event",
    business_category="event",
    canonical="event_return_since_last",
    source="relation.ops",
    status="experimental",
)
class EventReturnSinceLast(SeriesOperator):
    """自最近事件起的**复利**收益 Π(1+r)-1（事件触发后含错位）。"""

    metadata = _metadata(
        "event_return_since_last",
        "自最近事件（含 event_effective_lag 错位）起的复利收益 Π(1+r)-1。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        return _event_since_last_agg(ret, event, window, event_effective_lag, "compounded")


@register_operator(
    name="event_arithmetic_return_sum",
    category="event",
    business_category="event",
    canonical="event_arithmetic_return_sum",
    source="relation.ops",
    status="experimental",
)
class EventArithmeticReturnSum(SeriesOperator):
    """自最近事件起的**算术**收益和 Σ ret。"""

    metadata = _metadata(
        "event_arithmetic_return_sum",
        "自最近事件（含 event_effective_lag 错位）起的算术收益和。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        return _event_since_last_agg(ret, event, window, event_effective_lag, "sum")


@register_operator(
    name="event_log_return_sum",
    category="event",
    business_category="event",
    canonical="event_log_return_sum",
    source="relation.ops",
    status="experimental",
)
class EventLogReturnSum(SeriesOperator):
    """自最近事件起的对数收益和 Σ ln(1+ret)。"""

    metadata = _metadata(
        "event_log_return_sum",
        "自最近事件（含 event_effective_lag 错位）起的对数收益和。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        return _event_since_last_agg(ret, event, window, event_effective_lag, "logsum")


@register_operator(
    name="event_active_count",
    category="event",
    business_category="event",
    canonical="event_active_count",
    source="relation.ops",
    status="experimental",
)
class EventActiveCount(SeriesOperator):
    """自最近事件起的有效 bar 数。"""

    metadata = _metadata(
        "event_active_count",
        "自最近事件（含 event_effective_lag 错位）起的有效观测 bar 数。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="count",
        param_specs=_EVENT_WINDOW_PARAMS,
        relational_specs=[_EVENT_WINDOW_RELATION],
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        return _event_since_last_agg(ret, event, window, event_effective_lag, "count")


@register_operator(
    name="fin_applicability_mask",
    category="event",
    business_category="event",
    canonical="fin_applicability_mask",
    source="relation.ops",
    status="experimental",
)
class FinApplicabilityMask(SeriesOperator):
    """适用性掩码：value 有效且 > threshold → 1，否则 0。"""

    metadata = _metadata(
        "fin_applicability_mask",
        "value 有效且 > threshold 时为 1。",
        ["value", "threshold"],
        category="event",
        domain="fundamental",
        unit="boolean",
    )

    def _calculate_series(self, value: pd.DataFrame, threshold: float = 0.0, **_: Any) -> pd.DataFrame:
        thr = float(threshold)
        values = value.to_numpy(dtype=float)
        # Unknown / unreported values (NaN) must stay NaN — never converted to
        # "0 = not applicable" (audit §4.6: missing is not a confirmed no).
        finite = np.isfinite(values)
        out = np.full(values.shape, np.nan, dtype=float)
        out[finite & (values > thr)] = 1.0
        out[finite & (values <= thr)] = 0.0
        return _frame_like(value, out)


def _day_diff_frame(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Elementwise calendar-day difference ``right - left``.

    Accepts datetime64 panels (converted per element) or numeric row-position
    panels.  Never relies on ``DataFrame.dt``.  R24-005: the two panels are
    strict-aligned first — position-only pairing on mismatched axes raises.
    """
    left, right = strict_relation_align(left, right)
    a = left.to_numpy()
    b = right.to_numpy()
    out = np.full(a.shape, np.nan, dtype=float)
    try:
        ad = a.astype("datetime64[ns]")
        bd = b.astype("datetime64[ns]")
        delta = np.where(np.timedelta64(1, "D") != 0, (bd - ad) / np.timedelta64(1, "D"), np.nan)
        valid = ~np.isnat(ad) & ~np.isnat(bd)
        out[valid] = np.asarray(delta[valid], dtype=float)
    except (ValueError, TypeError):
        out = b.astype(float) - a.astype(float)
    return _frame_like(left, out)


@register_operator(
    name="calendar_day_diff",
    category="event",
    business_category="event",
    canonical="calendar_day_diff",
    source="relation.ops",
    status="experimental",
)
class CalendarDayDiff(SeriesOperator):
    """两个日期面板的自然日差（date2 - date1，不含交易日历）。"""

    metadata = _metadata(
        "calendar_day_diff",
        "date2 - date1（自然日差）。",
        ["date1", "date2"],
        category="event",
        domain="calendar",
        unit="count",
    )

    def _calculate_series(self, date1: pd.DataFrame, date2: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _day_diff_frame(date1, date2)


@register_operator(
    name="fin_announcement_lag",
    category="event",
    business_category="event",
    canonical="fin_announcement_lag",
    source="relation.ops",
    status="experimental",
)
class FinAnnouncementLag(SeriesOperator):
    """公告相对报告期末的自然日滞后：pub_date - period_end_date。"""

    metadata = _metadata(
        "fin_announcement_lag",
        "pub_date - period_end_date（自然日天数）。",
        ["period_end_date", "pub_date"],
        category="event",
        domain="fundamental",
        unit="count",
    )

    def _calculate_series(self, period_end_date: pd.DataFrame, pub_date: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _day_diff_frame(period_end_date, pub_date)


@register_operator(
    name="trading_day_diff",
    category="event",
    business_category="calendar",
    canonical="trading_day_diff",
    source="relation.ops",
    status="experimental",
)
class TradingDayDiff(SeriesOperator):
    """两个日期面板的交易日差（date2 - date1，基于交易日历）。

    NOTE: 当前实现为简化版，直接使用 calendar-day 差值 × 0.7 近似交易日差。
    完整实现需要接入真实交易日历服务（如 calendar.TradingCalendar）。
    """

    metadata = _metadata(
        "trading_day_diff",
        "date2 - date1（交易日天数；当前简化实现使用自然日 × 0.7）。",
        ["date1", "date2"],
        category="event",
        domain="calendar",
        unit="count",
    )

    def _calculate_series(self, date1: pd.DataFrame, date2: pd.DataFrame, **_: Any) -> pd.DataFrame:
        date1, date2 = strict_relation_align(date1, date2)
        d1 = date1.to_numpy()
        d2 = date2.to_numpy()
        out = np.full(d1.shape, np.nan, dtype=float)

        try:
            d1_dt = d1.astype("datetime64[D]")
            d2_dt = d2.astype("datetime64[D]")
            valid = ~np.isnat(d1_dt) & ~np.isnat(d2_dt)
            # 计算自然日差
            calendar_days = (d2_dt - d1_dt).astype("timedelta64[D]").astype(float)
            # 交易日约为自然日的 70%（简化假设：周末 + 节假日）
            # 完整实现应查询真实交易日历
            out[valid] = np.round(calendar_days[valid] * 0.7)
        except (ValueError, TypeError):
            # Fallback: 如果不是日期类型，直接做数值差
            out = d2.astype(float) - d1.astype(float)

        return _frame_like(date1, out)


def _row_entity_ids(panel: pd.DataFrame) -> list[set]:
    """Row-wise non-null entity id sets from a pre-aggregated entity panel."""
    arr = panel.to_numpy(dtype=object)
    rows = []
    for row in arr:
        seen = set()
        for value in row:
            if value is None:
                continue
            text = str(value)
            if text == "nan" or text == "None":
                continue
            seen.add(text)
        rows.append(seen)
    return rows


@register_operator(
    name="relation_distinct_count",
    category="relation",
    business_category="relation",
    canonical="relation_distinct_count",
    source="relation.ops",
    status="experimental",
)
class RelationDistinctCount(SeriesOperator):
    """同一股票快照中的不同实体数量（逐行去重）。"""

    metadata = _metadata(
        "relation_distinct_count",
        "每个 instrument-day 快照中不同 entity_id 的数量。",
        ["entity_ids"],
        category="relation",
        domain="shareholder",
        unit="count",
    )

    def _calculate_series(self, entity_ids: pd.DataFrame, **_: Any) -> pd.DataFrame:
        counts = np.array([len(s) for s in _row_entity_ids(entity_ids)], dtype=float)
        return _frame_like(entity_ids, counts.reshape(-1, 1).repeat(entity_ids.shape[1], axis=1))


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return np.nan
    return np.where(len(union)) != 0, float(len(a & b) / len(union)), np.nan)


@register_operator(
    name="relation_overlap_ratio",
    category="relation",
    business_category="relation",
    canonical="relation_overlap_ratio",
    source="relation.ops",
    status="experimental",
)
class RelationOverlapRatio(SeriesOperator):
    """当前快照与上期快照的实体重叠比例（默认 Jaccard）。"""

    metadata = _metadata(
        "relation_overlap_ratio",
        "当前/上期实体集合的重叠比例（jaccard 或 overlap）。",
        ["current_ids", "previous_ids", "method"],
        category="relation",
        domain="shareholder",
        unit="ratio",
    )

    def _calculate_series(self, current_ids: pd.DataFrame, previous_ids: pd.DataFrame, method: str = "jaccard", **_: Any) -> pd.DataFrame:
        current_ids, previous_ids = strict_relation_align(current_ids, previous_ids)
        cur = _row_entity_ids(current_ids)
        prev = _row_entity_ids(previous_ids)
        out = []
        for a, b in zip(cur, prev):
            if method == "jaccard":
                out.append(_jaccard(a, b))
            elif method == "overlap":
                inter = len(a & b)
                out.append(float(inter / min(len(a), len(b))) if min(len(a), len(b)) else np.nan)
            else:
                raise ValueError(f"unknown overlap method: {method!r}")
        arr = np.asarray(out, dtype=float).reshape(-1, 1)
        return _frame_like(current_ids, arr.repeat(current_ids.shape[1], axis=1))


@register_operator(
    name="index_weight",
    category="index",
    business_category="index",
    canonical="index_weight",
    source="relation.ops",
    status="experimental",
)
class IndexWeight(SeriesOperator):
    """指数成分权重（可选按截面归一化）。"""

    metadata = _metadata(
        "index_weight",
        "指数成分权重；normalize=True 时按每个横截面归一化为 1。",
        ["weight", "normalize"],
        category="index",
        domain="index",
        unit="weight",
    )

    def _calculate_series(self, weight: pd.DataFrame, normalize: bool = True, **_: Any) -> pd.DataFrame:
        arr = np.where(np.isfinite(weight), weight.to_numpy(dtype=float), np.nan)
        if bool(normalize):
            denom = np.nansum(arr, axis=1, keepdims=True)
            denom = np.where(np.abs(denom) > 1e-12, denom, np.nan)
            arr = arr / denom if denom != 0 else np.nan
        return _frame_like(weight, arr)


import cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.extend_extended_only(set(['relation_hhi', 'relation_entropy', 'relation_topk_sum', 'relation_rank_weighted_sum', 'relation_category_share', 'relation_peer_weighted_mean_ex_self', 'relation_entry_count', 'relation_exit_count', 'relation_weighted_change', 'index_member', 'index_weight_change', 'index_entry_exit_event', 'index_membership_age', 'event_cumulative_return_past', 'event_abnormal_return_past', 'fin_applicability_mask', 'fin_announcement_lag', 'relation_distinct_count', 'relation_overlap_ratio', 'index_weight']))


from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.extend_extended_only({
        "relation_distinct_count", "relation_overlap_ratio", "index_weight",
        "event_return_since_last", "event_arithmetic_return_sum",
        "event_log_return_sum", "event_active_count",
        # R24-009/011: new relation/shareholder canonicals
        "relation_category_signed_contribution", "holder_observed_topk_hhi",
        # Phase 2 (2026-08-12): trading day calendar diff
        "trading_day_diff",
    })

# ``event_compounded_return`` 是 ``event_return_since_last`` 的语义别名（复利口径）。
from cleaned_operators.registry import OperatorRegistry as _registry  # noqa: E402
_registry.register_alias("event_compounded_return", "event_return_since_last")
