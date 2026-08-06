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

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, category: str, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _stack_panels(*panels: pd.DataFrame) -> np.ndarray:
    """Align positional panels to the first panel and stack to (N, rows, cols)."""
    base = panels[0]
    arrays = [np.asarray(p.reindex(index=base.index, columns=base.columns).to_numpy(dtype=float)) for p in panels]
    return np.stack(arrays, axis=0)


def _finite_weights(panel: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
    values = panel.to_numpy(dtype=float)
    out = np.where(np.isfinite(values) & (values > threshold), values, np.nan)
    return out


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
    """按名次面板计算持股集中度 HHI = Σw_i²（w_i = s_i / Σs）。"""

    metadata = _metadata(
        "relation_hhi",
        "名次面板持股集中度 HHI。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
        category="relation",
        domain="relation",
        unit="ratio",
    )

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_hhi requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = np.nan_to_num(stacked, nan=0.0)
        total = values.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = values / total
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
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
        category="relation",
        domain="relation",
        unit="ratio",
    )

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_entropy requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        values = np.nan_to_num(stacked, nan=0.0)
        total = values.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = values / total
            entropy = -np.sum(shares * np.log(np.where(shares > 0, shares, 1.0)), axis=0)
        count = (np.isfinite(stacked)).sum(axis=0).astype(float)
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
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
        category="relation",
        domain="relation",
        unit="ratio",
    )

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_topk_sum requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        total = np.nansum(stacked, axis=0)
        total = np.where(np.isfinite(stacked).sum(axis=0) > 0, total, np.nan)
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
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
        category="relation",
        domain="relation",
        unit="ratio",
    )

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 2:
            raise ValueError("relation_rank_weighted_sum requires at least two ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        ranks = np.arange(1, stacked.shape[0] + 1, dtype=float)[:, None, None]
        weights = 1.0 / ranks
        weighted = np.nansum(stacked * weights, axis=0)
        weight_sum = np.sum(weights, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = weighted / weight_sum
        out = np.where(np.isfinite(stacked).sum(axis=0) > 0, out, np.nan)
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
    """个股 value 占同日期同类目 value 总和的比例。"""

    metadata = _metadata(
        "relation_category_share",
        "value 占同类别总和的比例。",
        ["value", "category"],
        category="relation",
        domain="relation",
        unit="ratio",
    )

    def _calculate_series(self, value: pd.DataFrame, category: pd.DataFrame, **_: Any) -> pd.DataFrame:
        value, category = value.reindex_like(category), category
        vv = value.to_numpy(dtype=float)
        cv = category.to_numpy()
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
        unit="ratio",
    )

    def _calculate_series(self, value: pd.DataFrame, weight: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        vv = value.to_numpy(dtype=float)
        wv = weight.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = vv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            g_row = gv[row]
            for label in pd.unique(g_row):
                idx = np.flatnonzero(g_row == label)
                weights = wv[row][idx]
                total_w = float(np.nansum(weights))
                if not np.isfinite(total_w) or total_w <= 0.0:
                    continue
                weighted = float(np.nansum(weights * vv[row][idx]))
                for j in idx:
                    own_w = wv[row][j]
                    if not np.isfinite(own_w) or not np.isfinite(vv[row][j]):
                        continue
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
        ["member", "window"],
        category="relation",
        domain="relation",
        unit="count",
    )

    def _calculate_series(self, member: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _transition_count(member, window, forward=True)


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
        ["member", "window"],
        category="relation",
        domain="relation",
        unit="count",
    )

    def _calculate_series(self, member: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
        return _transition_count(member, window, forward=False)


def _transition_count(member: pd.DataFrame, window: int, forward: bool) -> pd.DataFrame:
    w = int(window)
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
            truth = valid & (segment != 0)
            transitions = np.sum(truth[1:] & ~truth[:-1]) if forward else np.sum(~truth[1:] & truth[:-1])
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
        unit="ratio",
    )

    def _calculate_series(self, value: pd.DataFrame, weight: pd.DataFrame, **_: Any) -> pd.DataFrame:
        delta = value - value.shift(1)
        return delta * weight


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
    """指数成分标记：member 非零处保留为 1，否则 NaN。"""

    metadata = _metadata(
        "index_member",
        "指数成分标记（member!=0 → 1）。",
        ["member"],
        category="index",
        domain="index",
        unit="boolean",
    )

    def _calculate_series(self, member: pd.DataFrame, **_: Any) -> pd.DataFrame:
        valid = np.isfinite(member.to_numpy(dtype=float))
        out = np.where(valid & (member.to_numpy(dtype=float) != 0), 1.0, np.nan)
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
    """指数纳入/剔除事件：0→1 记 +1，1→0 记 -1。"""

    metadata = _metadata(
        "index_entry_exit_event",
        "纳入 +1 / 剔除 -1。",
        ["member"],
        category="index",
        domain="index",
        unit="boolean",
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
    """距指数纳入以来经过的交易行数。"""

    metadata = _metadata(
        "index_membership_age",
        "距纳入以来经过的行数。",
        ["member", "max_lookback"],
        category="index",
        domain="index",
        unit="count",
    )

    def _calculate_series(self, member: pd.DataFrame, max_lookback: Any = None, **_: Any) -> pd.DataFrame:
        limit = None if max_lookback is None else int(max_lookback)
        mv = member.to_numpy(dtype=float)
        rows, cols = mv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_entry = -1
            for row in range(rows):
                if np.isfinite(mv[row, col]):
                    if mv[row, col] != 0 and (row == 0 or last_entry < 0 or (not np.isfinite(mv[row - 1, col])) or mv[row - 1, col] == 0):
                        last_entry = row
                if last_entry >= 0:
                    distance = row - last_entry
                    if limit is None or distance < limit:
                        out[row, col] = float(distance)
        return _frame_like(member, out)


# ---------------------------------------------------------------------------
# Event / timing operators
# ---------------------------------------------------------------------------


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
        "事件触发时点后窗口内收益累计。",
        ["ret", "event", "window"],
        category="event",
        domain="event",
        unit="return",
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        rv = ret.to_numpy(dtype=float)
        ev = event.to_numpy(dtype=float)
        rows, cols = rv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            last_event = -1
            for row in range(rows):
                if np.isfinite(ev[row, col]) and ev[row, col] != 0:
                    last_event = row
                if last_event >= 0 and row - last_event < w:
                    segment = rv[last_event : row + 1, col]
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
        "事件触发时点后窗口内超额收益累计。",
        ["ret", "benchmark_ret", "event", "window"],
        category="event",
        domain="event",
        unit="return",
    )

    def _calculate_series(self, ret: pd.DataFrame, benchmark_ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
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
                if last_event >= 0 and row - last_event < w:
                    segment = abnormal[last_event : row + 1, col]
                    valid = np.isfinite(segment)
                    if valid.any():
                        out[row, col] = float(np.nansum(segment))
        return _frame_like(ret, out)


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
        out = np.where(np.isfinite(values) & (values > thr), 1.0, 0.0)
        return _frame_like(value, out)


@register_operator(
    name="trading_day_diff",
    category="event",
    business_category="event",
    canonical="trading_day_diff",
    source="relation.ops",
    status="experimental",
)
class TradingDayDiff(SeriesOperator):
    """两个日期面板的交易日差（数值面板按行位置差处理）。"""

    metadata = _metadata(
        "trading_day_diff",
        "date2 - date1（交易日差）。",
        ["date1", "date2"],
        category="event",
        domain="calendar",
        unit="count",
    )

    def _calculate_series(self, date1: pd.DataFrame, date2: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if hasattr(date1.index, "dtype") and str(date1.index.dtype).startswith("datetime"):
            pass
        if str(date1.dtypes.iloc[0]).startswith("datetime"):
            return (date2 - date1).dt.days.astype(float)
        return date2.astype(float) - date1.astype(float)


@register_operator(
    name="fin_announcement_lag",
    category="event",
    business_category="event",
    canonical="fin_announcement_lag",
    source="relation.ops",
    status="experimental",
)
class FinAnnouncementLag(SeriesOperator):
    """公告相对报告期末的滞后天数：pub_date - period_end_date。"""

    metadata = _metadata(
        "fin_announcement_lag",
        "pub_date - period_end_date（天数）。",
        ["period_end_date", "pub_date"],
        category="event",
        domain="fundamental",
        unit="count",
    )

    def _calculate_series(self, period_end_date: pd.DataFrame, pub_date: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if str(period_end_date.dtypes.iloc[0]).startswith("datetime"):
            return (pub_date - period_end_date).dt.days.astype(float)
        return pub_date.astype(float) - period_end_date.astype(float)


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
    return float(len(a & b) / len(union))


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
            arr = arr / denom
        return _frame_like(weight, arr)


import cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.EXTENDED_ONLY_CANONICALS = frozenset(set(_surface.EXTENDED_ONLY_CANONICALS) | set(['relation_hhi', 'relation_entropy', 'relation_topk_sum', 'relation_rank_weighted_sum', 'relation_category_share', 'relation_peer_weighted_mean_ex_self', 'relation_entry_count', 'relation_exit_count', 'relation_weighted_change', 'index_member', 'index_weight_change', 'index_entry_exit_event', 'index_membership_age', 'event_cumulative_return_past', 'event_abnormal_return_past', 'fin_applicability_mask', 'trading_day_diff', 'fin_announcement_lag', 'relation_distinct_count', 'relation_overlap_ratio', 'index_weight']))


from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS)
    | {"relation_distinct_count", "relation_overlap_ratio", "index_weight"}
)
