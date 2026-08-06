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
        "名次面板持股集中度 HHI（仅已披露前十大股东口径，非全体股东结构）。",
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
        finite = np.isfinite(stacked)
        weighted = np.nansum(np.where(finite, stacked * weights, 0.0), axis=0)
        weight_sum = np.sum(np.where(finite, weights, 0.0), axis=0)
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
        ["member", "window"],
        category="relation",
        domain="relation",
        unit="count",
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
        ["member", "window"],
        category="relation",
        domain="relation",
        unit="count",
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
                value = mv[row, col]
                if not np.isfinite(value):
                    last_entry = -1
                    out[row, col] = np.nan
                    continue
                if value != 0:
                    if last_entry < 0:
                        last_entry = row
                    distance = row - last_entry
                    if limit is None or distance < limit:
                        out[row, col] = float(distance)
                else:
                    # 已剔除：不再累计成分股年龄
                    out[row, col] = np.nan
                    last_entry = -1
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
        "事件触发后（含 event_effective_lag 错位）窗口内收益累计（算术和；复利见 event_return_since_last）。",
        ["ret", "event", "window", "event_effective_lag"],
        category="event",
        domain="event",
        unit="return",
    )

    def _calculate_series(self, ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        lag = max(0, int(event_effective_lag))
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
    )

    def _calculate_series(self, ret: pd.DataFrame, benchmark_ret: pd.DataFrame, event: pd.DataFrame, window: int = 20, event_effective_lag: int = 1, **_: Any) -> pd.DataFrame:
        w = int(window)
        lag = max(0, int(event_effective_lag))
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
    panels.  Never relies on ``DataFrame.dt``.
    """
    a = left.to_numpy()
    b = right.to_numpy()
    out = np.full(a.shape, np.nan, dtype=float)
    try:
        ad = a.astype("datetime64[ns]")
        bd = b.astype("datetime64[ns]")
        delta = (bd - ad) / np.timedelta64(1, "D")
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
_surface.EXTENDED_ONLY_CANONICALS = frozenset(set(_surface.EXTENDED_ONLY_CANONICALS) | set(['relation_hhi', 'relation_entropy', 'relation_topk_sum', 'relation_rank_weighted_sum', 'relation_category_share', 'relation_peer_weighted_mean_ex_self', 'relation_entry_count', 'relation_exit_count', 'relation_weighted_change', 'index_member', 'index_weight_change', 'index_entry_exit_event', 'index_membership_age', 'event_cumulative_return_past', 'event_abnormal_return_past', 'fin_applicability_mask', 'fin_announcement_lag', 'relation_distinct_count', 'relation_overlap_ratio', 'index_weight']))


from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    set(_surface.EXTENDED_ONLY_CANONICALS)
    | {
        "relation_distinct_count", "relation_overlap_ratio", "index_weight",
        "event_return_since_last", "event_arithmetic_return_sum",
        "event_log_return_sum", "event_active_count",
    }
)

# ``event_compounded_return`` 是 ``event_return_since_last`` 的语义别名（复利口径）。
from cleaned_operators.registry import OperatorRegistry as _registry  # noqa: E402
_registry.register_alias("event_compounded_return", "event_return_since_last")
