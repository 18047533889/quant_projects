# -*- coding: utf-8 -*-
"""Relation-distribution and group-shape operators (2026-08 pack, group 5).

Two sub-families:

* ``relation_*`` — up to 10 ranked relation panels (``s1..s10``) or a single
  pre-aggregated concentration panel.  Concentration/mobility/shape statistics
  are computed per (date, instrument); rolling variants are causal.
* ``group_*`` — cross-sectional shape of a group's members on the current row
  (skew, kurtosis, quantile spread, tail ratio).

Missing-value policy: NaN is never treated as 0 for concentration weights
(relation panels with a missing rank are excluded from that row's total); a
group with fewer than the required members returns NaN.
"""
from __future__ import annotations

import math
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
from cleaned_operators.rolling_pack import check_window, frame_like, register_polars_bridge


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    category: str,
    unit: str,
    output_unit: str | None = None,
    param_specs: dict[str, Any] | None = None,
    relational_specs: list[RelationalParamSpec] | None = None,
) -> OperatorMetadata:
    metadata = OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:relation",
            f"unit:{unit}", "cost:2",
        ],
        output_unit=output_unit,
        param_specs=dict(param_specs or {}),
        relational_specs=list(relational_specs or []),
    )
    return metadata


def _stack_panels(*panels: pd.DataFrame) -> np.ndarray:
    """Stack positional panels to (N, rows, cols) without reindexing.

    R11 #122: panels are ``timestamp x instrument`` relation panels that must
    share the exact same date axis and instrument columns.  A silent
    ``reindex`` (union) could re-pair a row to a different date or an instrument
    column to a different name after an upstream misalignment, silently
    corrupting the per-cell concentration / mobility statistic.  Different axes
    raise instead.
    """
    base = panels[0]
    for position, panel in enumerate(panels[1:], start=1):
        if not panel.index.equals(base.index) or not panel.columns.equals(base.columns):
            raise ValueError(
                f"relation panel {position} has a different index/columns than "
                "panel 0 (fail-closed; no silent reindex)"
            )
    arrays = [
        np.asarray(p.to_numpy(dtype=float))
        for p in panels
    ]
    return np.stack(arrays, axis=0)


def _has_spread(values: np.ndarray) -> bool:
    valid = values[np.isfinite(values)]
    return valid.size >= 2 and float(np.std(valid, ddof=0)) >= 1e-12


def _stack_id_panels(*panels: pd.DataFrame) -> np.ndarray:
    """Stack object-dtype identity panels to (N, rows, cols) without reindexing.

    Same fail-closed axis contract as :func:`_stack_panels` (R11 #122).
    """
    base = panels[0]
    for position, panel in enumerate(panels[1:], start=1):
        if not panel.index.equals(base.index) or not panel.columns.equals(base.columns):
            raise ValueError(
                f"relation identity panel {position} has a different index/columns "
                "than panel 0 (fail-closed; no silent reindex)"
            )
    return np.stack([np.asarray(p.to_numpy(dtype=object)) for p in panels], axis=0)


def _id_key(value: Any) -> str | None:
    """Normalise an identity cell to a string key, or None when missing."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value)
    if s in ("", "nan", "None", "NaN", "<NA>", "NaT"):
        return None
    return s


def _id_value_map(
    ids: np.ndarray, values: np.ndarray
) -> dict[str, float] | None:
    """Build ``{id: value}`` for one cell across ranked slots (fail-closed).

    Returns ``None`` when a present identity has a missing value (unknown
    holding is not a confirmed 0) or the same identity repeats with conflicting
    values — mirroring the shareholder churn snapshot-validity contract.
    """
    out: dict[str, float] = {}
    for k in range(ids.shape[0]):
        key = _id_key(ids[k])
        if key is None:
            continue
        v = values[k]
        if not np.isfinite(v):
            return None
        if key in out:
            if abs(out[key] - float(v)) > 1e-12:
                return None
            continue
        out[key] = float(v)
    return out


@register_operator(
    name="relation_topk_concentration",
    category="relation",
    business_category="relation",
    canonical="relation_topk_concentration",
    source="relation.distribution")
class RelationTopkConcentration(SeriesOperator):
    """名次面板前 k 名占比（rank-slot 语义）：分子=Σ(s1..sk)，前 k 个名次槽任一缺失 ⇒ NaN。

    Rank-slot semantics: the numerator is the sum of the FIRST k RANK SLOTS
    (s1..sk) — never the k largest values.  If ANY of the first k slots is
    non-finite on a cell, that cell is NaN (an unknown top-1 cannot be
    backfilled by top-6).  The denominator is the sum over ALL provided rank
    slots; a missing rank slot is excluded from the denominator, never treated
    as 0.
    """

    metadata = _metadata(
        "relation_topk_concentration",
        "前 k 名关系值占比（rank-slot：分子=Σ(s1..sk)；前 k 个名次槽任一缺失 ⇒ NaN；缺失名次不计入分母）。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "k"],
        category="relation",
        unit="ratio",
        # review §17: ``k`` is a strict-integer count of rank slots — 5.1 must be
        # rejected, never silently ``int(k)``-truncated to 5.
        param_specs={"k": ParamSpec(dtype=int, min=1)},
    )

    def _calculate_series(
        self, s1: pd.DataFrame | None = None, s2: pd.DataFrame | None = None,
        s3: pd.DataFrame | None = None, s4: pd.DataFrame | None = None,
        s5: pd.DataFrame | None = None, s6: pd.DataFrame | None = None,
        s7: pd.DataFrame | None = None, s8: pd.DataFrame | None = None,
        s9: pd.DataFrame | None = None, s10: pd.DataFrame | None = None,
        k: int = 5, **_: Any,
    ) -> pd.DataFrame:
        panels = [p for p in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) if isinstance(p, pd.DataFrame)]
        if len(panels) < 2:
            raise ValueError("relation_topk_concentration requires at least two ranked panels")
        top_k = int(k)
        if top_k < 1:
            raise ValueError("k must be >= 1")
        base = panels[0]
        stacked = _stack_panels(*panels)
        n, rows, cols = stacked.shape
        # Feasibility: k must fit the actual number of relation slots.  A k that
        # exceeds the available panels used to be silently capped, making
        # ``relation_topk_concentration(..., k=5)`` and ``k=10`` indistinguishable
        # in the surface.  Fail closed instead (review P1-121).
        if top_k > n:
            raise ValueError(
                f"relation_topk_concentration: k={top_k} exceeds the number of "
                f"available relation panels ({n})"
            )
        # Relation share/weight semantics require non-negative values; a negative
        # share would corrupt the top-k concentration ratio (review P1-121).
        if np.any(stacked[np.isfinite(stacked)] < 0.0):
            raise ValueError(
                "relation_topk_concentration: relation share/weight must be non-negative"
            )
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                vals = stacked[:, r, c]
                # Rank-slot numerator: the first k SLOTS (s1..sk) are the top-k
                # ranks.  ANY missing slot among them fails the cell closed —
                # an unknown top-1 must never be backfilled by a lower rank.
                top_slots = vals[:top_k]
                if not np.all(np.isfinite(top_slots)):
                    continue
                finite = vals[np.isfinite(vals)]
                if finite.size < 2:
                    continue
                total = float(np.sum(finite))
                if total <= 0.0:
                    continue
                top = float(np.sum(top_slots))
                out[r, c] = top / total
        return frame_like(base, out)


def _skew(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size < 3 or not _has_spread(valid):
        return np.nan
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=0))
    return float(np.mean((valid - mean) ** 3) / (std ** 3))


def _kurtosis(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size < 4 or not _has_spread(valid):
        return np.nan
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=0))
    return float(np.mean((valid - mean) ** 4) / (std ** 4))


@register_operator(
    name="relation_distribution_skew",
    category="relation",
    business_category="relation",
    canonical="relation_distribution_skew",
    source="relation.distribution")
class RelationDistributionSkew(SeriesOperator):
    """名次面板截面偏度（同一行 10 个名次值的偏度）。"""

    metadata = _metadata(
        "relation_distribution_skew",
        "名次面板截面偏度（variadic：接收 3 个及以上关系面板列数组）。",
        ["relations"],
        category="relation",
        # R11 #123: skewness is a standardized third moment — dimensionless,
        # never a raw "level".
        unit="dimensionless",
        output_unit="dimensionless",
    )
    # R5-06: genuinely variadic (3+ ranked panels in one positional slot).
    metadata.tags = list(metadata.tags) + ["variadic"]

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 3:
            raise ValueError("relation_distribution_skew requires at least three ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        rows, cols = stacked.shape[1], stacked.shape[2]
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = _skew(stacked[:, r, c])
        return frame_like(base, out)


def _pearson_kurtosis(values: np.ndarray) -> float:
    """Pearson kurtosis ``E[(X-μ)⁴]/σ⁴`` — ≈3 for a normal sample.

    Review §16: the historical ``relation_distribution_kurtosis`` name implied
    excess kurtosis (normal ≈ 0) while the formula was the fourth standardized
    moment (normal ≈ 3).  The canonical spelling now says exactly what it
    computes.
    """
    return _kurtosis(values)


@register_operator(
    name="relation_distribution_pearson_kurtosis",
    category="relation",
    business_category="relation",
    canonical="relation_distribution_pearson_kurtosis",
    source="relation.distribution")
class RelationDistributionPearsonKurtosis(SeriesOperator):
    """名次面板截面峰度（Pearson：E[(X-μ)⁴]/σ⁴，正态≈3）。"""

    metadata = _metadata(
        "relation_distribution_pearson_kurtosis",
        "名次面板截面峰度（Pearson：E[(X-μ)⁴]/σ⁴，正态≈3；variadic：接收 4 个及以上关系面板列数组）。",
        ["relations"],
        category="relation",
        # R11 #123: kurtosis is a standardized fourth moment — dimensionless.
        unit="dimensionless",
        output_unit="dimensionless",
    )
    # R5-06: genuinely variadic (4+ ranked panels in one positional slot).
    metadata.tags = list(metadata.tags) + ["variadic"]

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 4:
            raise ValueError("relation_distribution_pearson_kurtosis requires at least four ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        rows, cols = stacked.shape[1], stacked.shape[2]
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                out[r, c] = _pearson_kurtosis(stacked[:, r, c])
        return frame_like(base, out)


@register_operator(
    name="relation_distribution_excess_kurtosis",
    category="relation",
    business_category="relation",
    canonical="relation_distribution_excess_kurtosis",
    source="relation.distribution")
class RelationDistributionExcessKurtosis(SeriesOperator):
    """名次面板截面超额峰度 = Pearson 峰度 − 3（正态≈0）。"""

    metadata = _metadata(
        "relation_distribution_excess_kurtosis",
        "名次面板截面超额峰度 = Pearson 峰度 − 3（正态≈0；variadic：接收 4 个及以上关系面板列数组）。",
        ["relations"],
        category="relation",
        # R11 #123: kurtosis is a standardized fourth moment — dimensionless.
        unit="dimensionless",
        output_unit="dimensionless",
    )
    # R5-06: genuinely variadic (4+ ranked panels in one positional slot).
    metadata.tags = list(metadata.tags) + ["variadic"]

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) < 4:
            raise ValueError("relation_distribution_excess_kurtosis requires at least four ranked panels")
        base = args[0]
        stacked = _stack_panels(*args)
        rows, cols = stacked.shape[1], stacked.shape[2]
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                kurt = _pearson_kurtosis(stacked[:, r, c])
                out[r, c] = kurt - 3.0 if np.isfinite(kurt) else np.nan
        return frame_like(base, out)


def _delta(panel: np.ndarray, window: int) -> np.ndarray:
    rows, cols = panel.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        prev = r - window
        if prev < 0:
            continue
        for c in range(cols):
            if np.isfinite(panel[r, c]) and np.isfinite(panel[prev, c]):
                out[r, c] = panel[r, c] - panel[prev, c]
    return out


@register_operator(
    name="relation_hhi_change",
    category="relation",
    business_category="relation",
    canonical="relation_hhi_change",
    source="relation.distribution")
class RelationHhiChange(SeriesOperator):
    """集中度窗口变化：value[t] - value[t-window]。"""

    metadata = _metadata(
        "relation_hhi_change",
        "集中度 HHI 窗口变化。",
        ["hhi", "window"],
        category="relation",
        unit="ratio",
        # review §17: ``window`` is a strict-integer span — 5.1 must be rejected,
        # never silently ``int(window)``-truncated.  A change over a single bar
        # collides with the plain delta family, so the window spans >= 2 points.
        param_specs={"window": ParamSpec(dtype=int, min=2)},
    )

    def _calculate_series(self, hhi: pd.DataFrame, window: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        return frame_like(hhi, _delta(hhi.to_numpy(dtype=float), w))


@register_operator(
    name="relation_entropy_change",
    category="relation",
    business_category="relation",
    canonical="relation_entropy_change",
    source="relation.distribution")
class RelationEntropyChange(SeriesOperator):
    """分布熵窗口变化：value[t] - value[t-window]。"""

    metadata = _metadata(
        "relation_entropy_change",
        "分布熵窗口变化。",
        ["entropy", "window"],
        category="relation",
        unit="level",
        # review §17: strict-integer ``window`` — 5.1 is rejected, not truncated.
        # A change over a single bar collides with the plain delta family, so the
        # window spans >= 2 points.
        param_specs={"window": ParamSpec(dtype=int, min=2)},
    )

    def _calculate_series(self, entropy: pd.DataFrame, window: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        return frame_like(entropy, _delta(entropy.to_numpy(dtype=float), w))


@register_operator(
    name="relation_concentration_acceleration",
    category="relation",
    business_category="relation",
    canonical="relation_concentration_acceleration",
    source="relation.distribution")
class RelationConcentrationAcceleration(SeriesOperator):
    """集中度二阶差分：(v[t]-v[t-w]) - (v[t-w]-v[t-2w])。"""

    metadata = _metadata(
        "relation_concentration_acceleration",
        "集中度二阶差分（加速度）。",
        ["hhi", "window"],
        category="relation",
        unit="ratio",
        # R11 #125: the second difference (v[t]-v[t-w]) - (v[t-w]-v[t-2w])
        # needs 2*window prior rows; declared as a compound history formula so
        # the history planner never under-provisions warm-up.
        param_specs={
            # review §17: strict-integer ``window`` — 5.1 is rejected, never
            # ``int(window)``-truncated.  A second difference needs at least two
            # bars apart, so the window spans >= 2 points.
            "window": ParamSpec(
                dtype=int,
                min=2,
                history_formula="2 * window",
            ),
        },
    )

    def _calculate_series(self, hhi: pd.DataFrame, window: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        v = hhi.to_numpy(dtype=float)
        rows, cols = v.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            prev = r - w
            prev2 = r - 2 * w
            if prev < 0 or prev2 < 0:
                continue
            for c in range(cols):
                if all(np.isfinite(t) for t in (v[r, c], v[prev, c], v[prev2, c])):
                    out[r, c] = (v[r, c] - v[prev, c]) - (v[prev, c] - v[prev2, c])
        return frame_like(hhi, out)


def _mean_panel_change(stacked: np.ndarray, window: int) -> np.ndarray:
    n, rows, cols = stacked.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        prev = r - window
        if prev < 0:
            continue
        for c in range(cols):
            diffs: list[float] = []
            for i in range(n):
                a = stacked[i, r, c]
                b = stacked[i, prev, c]
                if np.isfinite(a) and np.isfinite(b):
                    diffs.append(abs(float(a - b)))
            if diffs:
                out[r, c] = float(np.mean(diffs))
    return out


@register_operator(
    name="relation_rank_mobility",
    category="relation",
    business_category="relation",
    canonical="relation_rank_mobility",
    source="relation.distribution")
class RelationRankMobility(SeriesOperator):
    """名次**槽位**（slot）移动性：窗口内各名次槽绝对变化的均值。

    R11 #124: this is SLOT mobility — ``rank1`` today may belong to a different
    entity than ``rank1`` five days ago, so ``rank1_value_t - rank1_value_{t-w}``
    is NOT the same entity's mobility.  For the entity-matched analogue (same
    ShareholderId across periods) use ``relation_rank_entity_mobility``.
    """

    metadata = _metadata(
        "relation_rank_mobility",
        "名次槽位移动性（slot）：窗口内各名次槽绝对变化的均值；实体匹配见 relation_rank_entity_mobility。",
        ["rank1", "rank2", "rank3", "rank4", "rank5", "rank6", "rank7", "rank8", "rank9", "rank10", "window"],
        category="relation",
        # R11 #182: rank mobility is measured in rank steps — the mean absolute
        # change of rank values across time, so the unit is the rank step, not a
        # bare level.  slot-identity: values at a fixed rank slot across time
        # are not the same entity, so this must never be read as entity churn.
        unit="rank",
        output_unit="rank",
        param_specs={"window": ParamSpec(dtype=int, min=1)},
    )
    metadata.tags = list(metadata.tags) + ["slot_identity"]

    def _calculate_series(
        self, rank1: pd.DataFrame | None = None, rank2: pd.DataFrame | None = None,
        rank3: pd.DataFrame | None = None, rank4: pd.DataFrame | None = None,
        rank5: pd.DataFrame | None = None, rank6: pd.DataFrame | None = None,
        rank7: pd.DataFrame | None = None, rank8: pd.DataFrame | None = None,
        rank9: pd.DataFrame | None = None, rank10: pd.DataFrame | None = None,
        window: int = 5, **_: Any,
    ) -> pd.DataFrame:
        panels = [
            p for p in (rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8, rank9, rank10)
            if isinstance(p, pd.DataFrame)
        ]
        if len(panels) < 2:
            raise ValueError("relation_rank_mobility requires at least two ranked panels")
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        base = panels[0]
        return frame_like(base, _mean_panel_change(_stack_panels(*panels), w))


@register_operator(
    name="relation_rank_entity_mobility",
    category="relation",
    business_category="relation",
    canonical="relation_rank_entity_mobility",
    source="relation.distribution")
class RelationRankEntityMobility(SeriesOperator):
    """实体匹配的名次值移动性：跨期同一实体 |value_cur - value_prev| 的均值。

    R11 #124: the slot-based ``relation_rank_mobility`` compares the value at a
    fixed rank slot across time, which may be two DIFFERENT entities.  This
    operator pairs holders by ShareholderId across the current and previous
    snapshots and averages the absolute change over the common entity set, so a
    pure rank swap with unchanged holdings yields zero mobility here.

    Output unit: the operator consumes share/weight VALUE panels (s1..s10 /
    p1..p10) and emits a change of those values, so the output carries the share
    unit — ``same_as:share``, matching the sibling ``relation_share_mobility``.
    """

    metadata = _metadata(
        "relation_rank_entity_mobility",
        "实体匹配的名次值移动性：对跨期共同实体取 |Δvalue| 均值。",
        [
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
            "sid1", "sid2", "sid3", "sid4", "sid5", "sid6", "sid7", "sid8", "sid9", "sid10",
            "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
            "psid1", "psid2", "psid3", "psid4", "psid5", "psid6", "psid7", "psid8", "psid9", "psid10",
        ],
        category="relation",
        # R11 #182: entity-matched mobility is a change of the share/weight
        # VALUE panels (s1..s10 / p1..p10) themselves — the output carries the
        # share unit (matching relation_share_mobility), never an ambiguous
        # "same_as:value" reference to a nonexistent input.
        unit="same_as:share",
        output_unit="same_as:share",
    )
    metadata.tags = list(metadata.tags) + ["entity_identity"]

    def _calculate_series(self, *args: pd.DataFrame, **_: Any) -> pd.DataFrame:
        if len(args) != 40:
            raise ValueError(
                "relation_rank_entity_mobility requires 40 panels "
                "(10 current values + 10 current ids + 10 previous values + 10 previous ids)"
            )
        base = args[0]
        cur = _stack_panels(*args[:10])
        cur_id = _stack_id_panels(*args[10:20])
        prev = _stack_panels(*args[20:30])
        prev_id = _stack_id_panels(*args[30:40])
        _, rows, cols = cur.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                cur_map = _id_value_map(cur_id[:, r, c], cur[:, r, c])
                prev_map = _id_value_map(prev_id[:, r, c], prev[:, r, c])
                if cur_map is None or prev_map is None:
                    continue
                common = [k for k in cur_map if k in prev_map]
                if not common:
                    continue
                out[r, c] = float(
                    np.mean([abs(cur_map[k] - prev_map[k]) for k in common])
                )
        return frame_like(base, out)


@register_operator(
    name="relation_share_mobility",
    category="relation",
    business_category="relation",
    canonical="relation_share_mobility",
    source="relation.distribution")
class RelationShareMobility(SeriesOperator):
    """份额面板移动性：窗口内各份额绝对变化的均值。"""

    metadata = _metadata(
        "relation_share_mobility",
        "份额面板平均绝对变化。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "window"],
        category="relation",
        # R11 #182: share mobility preserves the share unit — it is a change of
        # shares, not a ratio between unrelated quantities.
        unit="same_as:share",
        output_unit="same_as:share",
        # review §17: strict-integer ``window`` — 5.1 must be rejected, never
        # silently ``int(window)``-truncated to 5.
        param_specs={"window": ParamSpec(dtype=int, min=1)},
    )

    def _calculate_series(
        self, s1: pd.DataFrame | None = None, s2: pd.DataFrame | None = None,
        s3: pd.DataFrame | None = None, s4: pd.DataFrame | None = None,
        s5: pd.DataFrame | None = None, s6: pd.DataFrame | None = None,
        s7: pd.DataFrame | None = None, s8: pd.DataFrame | None = None,
        s9: pd.DataFrame | None = None, s10: pd.DataFrame | None = None,
        window: int = 5, **_: Any,
    ) -> pd.DataFrame:
        panels = [p for p in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) if isinstance(p, pd.DataFrame)]
        if len(panels) < 2:
            raise ValueError("relation_share_mobility requires at least two ranked panels")
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        base = panels[0]
        return frame_like(base, _mean_panel_change(_stack_panels(*panels), w))


# ---------------------------------------------------------------------------
# Group cross-sectional shape
# ---------------------------------------------------------------------------

def _valid_label(value: Any) -> bool:
    """True for a usable group label (numeric or categorical, NaN/None excluded)."""
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(float(value)):
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _group_shape(
    x: pd.DataFrame,
    group: pd.DataFrame,
    fn: Any,
) -> pd.DataFrame:
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    gv = group.to_numpy(dtype=object)
    xv = x.to_numpy(dtype=float)
    for r in range(xv.shape[0]):
        row_group = gv[r]
        row_x = xv[r]
        for label in set(v for v in row_group if _valid_label(v)):
            mask = row_group == label
            values = row_x[mask]
            valid = values[np.isfinite(values)]
            if valid.size < 3:
                continue
            value = fn(valid)
            if np.isfinite(value):
                result.iloc[r, np.flatnonzero(mask)] = value
    return result


@register_operator(
    name="group_skewness",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_skewness",
    source="relation.distribution")
class GroupSkewness(SeriesOperator):
    """组内截面偏度。"""

    metadata = _metadata(
        "group_skewness",
        "组内成员截面偏度。",
        ["x", "group"],
        category="cross_sectional",
        # R11 #123: standardized third moment — dimensionless.
        unit="dimensionless",
        output_unit="dimensionless",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _group_shape(x, group, _skew)


@register_operator(
    name="group_kurtosis",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_kurtosis",
    source="relation.distribution")
class GroupKurtosis(SeriesOperator):
    """组内截面峰度。"""

    metadata = _metadata(
        "group_kurtosis",
        "组内成员截面峰度。",
        ["x", "group"],
        category="cross_sectional",
        # R11 #123: standardized fourth moment — dimensionless.
        unit="dimensionless",
        output_unit="dimensionless",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _group_shape(x, group, _kurtosis)


@register_operator(
    name="group_quantile_spread",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_quantile_spread",
    source="relation.distribution")
class GroupQuantileSpread(SeriesOperator):
    """组内分位数距：(Q_high - Q_low)。"""

    metadata = _metadata(
        "group_quantile_spread",
        "组内 Q_high - Q_low。",
        ["x", "group", "q_low", "q_high"],
        category="cross_sectional",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, q_low: float = 0.25, q_high: float = 0.75, **_: Any) -> pd.DataFrame:
        ql, qh = float(q_low), float(q_high)
        if not 0.0 < ql < qh < 1.0:
            raise ValueError("require 0 < q_low < q_high < 1")

        def _fn(values: np.ndarray) -> float:
            if values.size < 3 or not _has_spread(values):
                return np.nan
            return float(np.quantile(values, qh) - np.quantile(values, ql))

        return _group_shape(x, group, _fn)


@register_operator(
    name="group_tail_ratio",
    category="cross_sectional",
    business_category="group_neutralization",
    canonical="group_tail_ratio",
    source="relation.distribution")
class GroupTailRatio(SeriesOperator):
    """组内尾部比：abs(Q_high)/abs(Q_low)，Q_low≈0 返回 NaN。"""

    metadata = _metadata(
        "group_tail_ratio",
        "组内右尾/左尾分位数绝对值比。",
        ["x", "group", "q_low", "q_high"],
        category="cross_sectional",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, q_low: float = 0.05, q_high: float = 0.95, **_: Any) -> pd.DataFrame:
        ql, qh = float(q_low), float(q_high)
        if not 0.0 < ql < qh < 1.0:
            raise ValueError("require 0 < q_low < q_high < 1")

        def _fn(values: np.ndarray) -> float:
            if values.size < 3 or not _has_spread(values):
                return np.nan
            q_lo = float(np.quantile(values, ql))
            q_hi = float(np.quantile(values, qh))
            if abs(q_lo) < 1e-12:
                return np.nan
            return float(abs(q_hi) / abs(q_lo))

        return _group_shape(x, group, _fn)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    _surface.extend_extended_only({
            "relation_topk_concentration", "relation_distribution_skew",
            "relation_distribution_pearson_kurtosis",
            "relation_distribution_excess_kurtosis",
            "relation_hhi_change",
            "relation_entropy_change", "relation_concentration_acceleration",
            "relation_rank_mobility", "relation_rank_entity_mobility",
            "relation_share_mobility",
            "group_skewness", "group_kurtosis", "group_quantile_spread",
            "group_tail_ratio",
        })
    # Review §16 honest rename: the old ``relation_distribution_kurtosis``
    # spelling is now a DEPRECATED ALIAS of the Pearson canonical (normal ≈ 3).
    # It is no longer an active canonical, so it must leave the extended
    # partition — otherwise layer_governance's exact-classification check counts
    # an inactive name (the alias) against the static surface.
    _surface.retract_extended_only({"relation_distribution_kurtosis"})
    for _canon in (
        "relation_topk_concentration", "relation_distribution_skew",
        "relation_distribution_pearson_kurtosis",
        "relation_distribution_excess_kurtosis",
        "relation_hhi_change",
        "relation_entropy_change", "relation_concentration_acceleration",
        "relation_rank_mobility", "relation_rank_entity_mobility",
        "relation_share_mobility",
        "group_skewness", "group_kurtosis", "group_quantile_spread",
        "group_tail_ratio",
    ):
        register_polars_bridge(_canon)

    # Review §16: old spelling -> deprecated alias of the honest Pearson
    # canonical.  Prefer rename_canonical (migrates first-registered identity /
    # governance) when another layer already registered the historical spelling;
    # otherwise register the alias directly.
    if "relation_distribution_kurtosis" in OperatorRegistry._operators:
        OperatorRegistry.rename_canonical(
            "relation_distribution_kurtosis",
            "relation_distribution_pearson_kurtosis",
        )
    elif "relation_distribution_pearson_kurtosis" in OperatorRegistry._operators:
        OperatorRegistry.register_alias(
            "relation_distribution_kurtosis",
            "relation_distribution_pearson_kurtosis",
            replacement_reason="review §16 renamed: the formula is Pearson "
            "kurtosis E[(X-mu)^4]/sigma^4 (normal ~ 3), not excess kurtosis "
            "(normal ~ 0); see relation_distribution_pearson_kurtosis / "
            "relation_distribution_excess_kurtosis",
        )


_register_surface()
