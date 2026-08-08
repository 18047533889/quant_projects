# -*- coding: utf-8 -*-
"""Cross-sectional spectral crowding operators (2026-08 V2, P1).

Within each group (industry / sector / relation), each date, the members'
2..4 feature vectors are stacked, z-scored in the group and SVD'd.  One
decomposition yields three regime / crowding signals broadcast to every member:

* ``group_feature_mode_share``        — σ1²/Σσ²: how dominant the top mode is
  (group collapsing into one common trading pattern).
* ``group_feature_effective_rank``    — exp(-Σ p log p), p=σ²/Σσ²: how many
  effective dimensions the group uses.
* ``group_feature_mode_localization`` — Σ u1^4 (inverse participation ratio of the
  top left singular vector across members): a leader-only mode is localized in
  a few names.

The eigenspectrum of the within-group correlation structure switches with market
regime (RMT literature on the A-share market); these make that switch a daily
cross-sectional state.  Deterministic (SVD, no randomness).
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12

# P0 (round 7): a 2..3 member group makes a mechanically low-rank standardized
# matrix, so mode_share ≈ 1 is a math artifact rather than crowding.  Require a
# group to have at least ``max(6, 2*d+2)`` valid members before its spectrum
# enters mining (below that we emit NaN).
_MIN_MEMBER_BASE = 6
_MIN_MEMBER_PER_FEATURE = 2

# P1 (round 7): when the two leading singular values are nearly degenerate the
# top singular vector is unstable — mode-localization readouts need a minimum
# relative gap (σ1-σ2)/σ1 or they are NaN.
_REL_GAP_THRESHOLD = 0.10

# P1 (round 7): breadth-stability gate.  A group whose valid-member count swings
# wildly from day to day is changing because of coverage noise, not economic
# structure; if today's count collapses to < 0.5 of the trailing median we fail
# the spectrum output for that day (NaN).
_BREADTH_STABILITY_WINDOW = 60
_BREADTH_RATIO_THRESHOLD = 0.5


def _min_members(d: int) -> int:
    """Minimum valid-member count for a trustworthy spectrum of ``d`` features."""
    return max(_MIN_MEMBER_BASE, 2 * int(d) + _MIN_MEMBER_PER_FEATURE)


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="group_structure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "group_structure", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _spectrum_stats(Z: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """SVD spectrum -> (mode_share, effective_rank, mode_localization,
    spectral_gap, second_mode_localization)."""
    n, d = Z.shape
    # P0 (round 7): a 2..3 member group's standardized matrix is mechanically
    # low-rank; the spectrum is a math artifact, not crowding.  Gate on the
    # member floor so tiny groups never enter mining.
    if n < _min_members(d) or d < 2:
        return None
    try:
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if S.shape[0] < 2 or not np.all(np.isfinite(S)):
        return None
    s2 = S * S
    total = float(s2.sum())
    if total <= _EPS:
        return None
    p = s2 / total
    mode_share = float(p[0])
    p_pos = p[p > 0.0]
    eff_rank = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    n_members = int(U.shape[0])
    # P1 (round 7): the top-two singular values' *relative* gap.  When the top
    # modes are nearly degenerate the leading singular vector is unstable and
    # mode-localization readouts are not a stable property of the group.
    s0 = float(S[0])
    s1 = float(S[1]) if S.shape[0] > 1 else 0.0
    rel_gap = (s0 - s1) / max(s0, _EPS) if s0 > _EPS else 0.0
    spectral_gap = float((s2[0] - s2[1]) / total)
    u1 = U[:, 0]
    ipr1 = float(np.sum(u1 * u1 * u1 * u1))
    # P1 (round 7): normalize the first-mode IPR the same way mode 2 already is
    # — raw Σu^4 has a 1/N baseline that varies with group size; the [0,1]
    # standardized form keeps baselines comparable across group sizes.
    if n_members > 1:
        localization = (n_members * ipr1 - 1.0) / (n_members - 1.0)
    else:
        localization = np.nan
    if rel_gap < _REL_GAP_THRESHOLD:
        localization = np.nan
    u2 = U[:, 1]
    ipr2 = float(np.sum(u2 * u2 * u2 * u2))
    if n_members > 1:
        second_loc = (n_members * ipr2 - 1.0) / (n_members - 1.0)
    else:
        second_loc = np.nan
    if rel_gap < _REL_GAP_THRESHOLD:
        second_loc = np.nan
    return mode_share, eff_rank, localization, spectral_gap, second_loc


def _group_spectrum_series(feats: np.ndarray, group: np.ndarray, which: str) -> np.ndarray:
    rows, cols, d = feats.shape
    min_members = _min_members(d)
    out = np.full((rows, cols), np.nan, dtype=float)
    # P1 (round 7): per-group trailing history of valid-member counts so extreme
    # membership swings (30 members today, 7 tomorrow) are treated as coverage
    # noise instead of economic change.
    breadth_history: dict[Any, deque] = {}
    for t in range(rows):
        row = feats[t]
        g_row = group[t]
        labels = pd.unique(g_row)
        for label in labels:
            idx = np.flatnonzero(g_row == label)
            if idx.size < min_members:
                continue
            Z = row[idx].astype(float)
            # Audit P1-G: only members with ALL features finite enter the group
            # statistic — one NaN member must not poison the whole group.  The
            # excluded members keep NaN (never a fabricated group value).
            finite_members = np.all(np.isfinite(Z), axis=1)
            Zv = Z[finite_members]
            # P0 (round 7): member floor after NaN filtering.
            if Zv.shape[0] < min_members:
                continue
            # P1 (round 7): breadth-stability gate — when the current valid
            # member count collapses below 0.5 of the trailing median, the
            # spectral change is coverage noise; fail closed to NaN.
            hist = breadth_history.setdefault(label, deque(maxlen=_BREADTH_STABILITY_WINDOW))
            if len(hist) >= 3:
                trailing_median = float(np.median(list(hist)))
                if trailing_median > 0.0 and Zv.shape[0] < _BREADTH_RATIO_THRESHOLD * trailing_median:
                    hist.append(int(Zv.shape[0]))
                    continue
            hist.append(int(Zv.shape[0]))
            med = np.median(Zv, axis=0)
            mad = 1.4826 * np.median(np.abs(Zv - med), axis=0)
            scale = np.where(mad > _EPS, mad, np.std(Zv, axis=0))
            if np.any(~np.isfinite(scale)) or np.any(scale <= _EPS):
                continue
            Zv = (Zv - med) / scale
            stats = _spectrum_stats(Zv)
            if stats is None:
                continue
            if which == "mode_share":
                value = stats[0]
            elif which == "effective_rank":
                value = stats[1]
            elif which == "localization":
                value = stats[2]
            elif which == "spectral_gap":
                value = stats[3]
            elif which == "second_mode_localization":
                value = stats[4]
            else:
                value = np.nan
            out[t, idx[finite_members]] = value
    return out


def _group_membership_series(feats: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-member broadcast of the group's valid-member count and coverage ratio.

    ``valid_member_count`` is the number of group members with ALL features
    finite on that date; ``coverage_ratio`` is that count over the group's total
    row count.  These are diagnostics (they are NOT gated by the spectrum member
    floor) so a user can see exactly why a spectrum output is NaN.
    """
    rows, cols, d = feats.shape
    cnt = np.full((rows, cols), np.nan, dtype=float)
    cov = np.full((rows, cols), np.nan, dtype=float)
    for t in range(rows):
        row = feats[t]
        g_row = group[t]
        labels = pd.unique(g_row)
        for label in labels:
            idx = np.flatnonzero(g_row == label)
            if idx.size == 0:
                continue
            Z = row[idx].astype(float)
            finite_members = np.all(np.isfinite(Z), axis=1)
            valid = int(finite_members.sum())
            cnt[t, idx] = valid
            cov[t, idx] = valid / idx.size
    return cnt, cov


@register_operator(
    name="group_feature_valid_member_count",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_valid_member_count",
    source="group_spectrum",
)
class GroupFeatureValidMemberCount(SeriesOperator):
    """组内有效成员数（当日全部特征有限的成员数）。

    P1 诊断输出：配合 ``group_feature_*`` 谱输出使用——谱输出在有效成员数低于
    ``max(6, 2d+2)`` 或相对尾部中位数塌缩时是 NaN，此算子告诉用户那一天到底有
    多少成员可用（覆盖噪声 vs 经济变化的判别）。
    """

    metadata = _metadata(
        "group_feature_valid_member_count",
        "组内当日有效成员数（诊断：谱输出为何 NaN）。",
        ["f1", "f2", "f3", "group"],
        unit="count",
        cost=2,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        cnt, _ = _group_membership_series(feats, group.to_numpy())
        return frame_like(f1, cnt)


@register_operator(
    name="group_feature_coverage_ratio",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_coverage_ratio",
    source="group_spectrum",
)
class GroupFeatureCoverageRatio(SeriesOperator):
    """组内当日数据覆盖比例 ``valid_members / group_rows``。

    P1 诊断输出：成员行本身是 NaN（停牌/未上市）使覆盖率 < 1；覆盖率剧烈波动
    说明谱输出变化是覆盖噪声而非经济结构变化。与 ``group_feature_valid_member_count``
    一起构成组谱的稳定性诊断。
    """

    metadata = _metadata(
        "group_feature_coverage_ratio",
        "组内当日覆盖比例 valid/rows（谱稳定性诊断）。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        _, cov = _group_membership_series(feats, group.to_numpy())
        return frame_like(f1, cov)


@register_operator(
    name="group_feature_mode_share",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_mode_share",
    source="group_spectrum",
)
class GroupFeatureModeShare(SeriesOperator):
    """组内特征谱 top-mode 占比 ``σ1²/Σσ²``。

    每组每日对成员的特征矩阵做组内 z-score + SVD；接近 1 = 整个组的横截面
    变异集中在一个共同方向（群体坍缩成共同交易模式 / 单因子结构）。组内每只
    股票得到同一值。P1。
    """

    metadata = _metadata(
        "group_feature_mode_share",
        "组内特征谱 top-mode 占比 σ1²/Σσ²（拥挤/同步化）。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "mode_share"))


@register_operator(
    name="group_feature_effective_rank",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_effective_rank",
    source="group_spectrum",
)
class GroupFeatureEffectiveRank(SeriesOperator):
    """组内特征谱有效秩 ``exp(-Σ p log p)``。

    p = σ²/Σσ²。有效秩低（接近 1）= 组内只有一两个有效维度（风格解体 /
    抱团）；高 = 组内结构分散多元。与 mode_share 互补（mode_share 只看头名，
    有效秩看整体谱宽）。组内每只股票同一值。P1。
    """

    metadata = _metadata(
        "group_feature_effective_rank",
        "组内特征谱有效秩 exp(-Σp log p)。",
        ["f1", "f2", "f3", "group"],
        unit="count",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "effective_rank"))


@register_operator(
    name="group_feature_mode_localization",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_mode_localization",
    source="group_spectrum",
)
class GroupFeatureModeLocalization(SeriesOperator):
    """组内主导模式的成员局域化 ``(N·Σu1⁴ - 1)/(N-1)``（[0,1]）。

    top 左奇异向量 u1 的 IPR 做 [0,1] 标准化（与第二模式一致），使不同组大小的
    基线可比（原始 Σu1⁴ 的随机基线是 1/N，随组大小漂移）。接近 1 = 主导模式由
    极少数股票驱动（leader-only rally）；低 = 模式均匀分布在全组。前两奇异值近
    简并（相对谱隙 < 0.1）时首特征向量不稳定，输出 NaN。组内每只股票同一值。P1。
    """

    metadata = _metadata(
        "group_feature_mode_localization",
        "组内 top-mode 局域化 (N·Σu1⁴-1)/(N-1)（[0,1]，谱隙门槛）。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "localization"))


@register_operator(
    name="group_feature_spectral_gap",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_spectral_gap",
    source="group_spectrum",
)
class GroupFeatureSpectralGap(SeriesOperator):
    """组内谱隙 ``(σ1² - σ2²) / Σσ²``（前两主导模式之间的相对差距）。

    高 = 市场只有一个绝对 dominant 方向（单主题抱团）；低 = 第二主题与第一
    主题规模接近（两个主要风格正在竞争 / 风格切换前兆）。与 mode_share（只看
    第一名绝对占比）互补：谱隙看第一名与第二名的*相对*距离。组内每只股票同一值。
    """

    metadata = _metadata(
        "group_feature_spectral_gap",
        "组内谱隙 (σ1²-σ2²)/Σσ²（单主题 vs 双主题竞争）。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "spectral_gap"))


@register_operator(
    name="group_feature_second_mode_localization",
    category="group_structure",
    business_category="group_structure",
    canonical="group_feature_second_mode_localization",
    source="group_spectrum",
)
class GroupFeatureSecondModeLocalization(SeriesOperator):
    """组内第二主导模式的成员局域化 ``(N·Σu2⁴ - 1)/(N-1)``（[0,1]）。

    第一特征向量常是 market/sector 共同模式；第二特征向量更像组内分裂轴。高 =
    第二主题主要由少数股票驱动（个别龙头的二分结构）；低 = 板块系统性二分
    （均匀两派）。与 ``group_feature_mode_localization``（第一模式）互补。
    """

    metadata = _metadata(
        "group_feature_second_mode_localization",
        "组内第二模式局域化 (N·Σu2⁴-1)/(N-1)。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "second_mode_localization"))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    # P1-G rename: the SVD is over the members' FEATURE matrix (not a
    # correlation matrix), so the canonical names are group_feature_*.  The old
    # group_corr_* names are pure aliases (kept for compatibility).
    _surface.extend_extended_only({
            "group_feature_mode_share",
            "group_feature_effective_rank",
            "group_feature_mode_localization",
            "group_feature_spectral_gap",
            "group_feature_second_mode_localization",
            "group_feature_valid_member_count",
            "group_feature_coverage_ratio",
        })
    _corr_to_feature = {
        "group_corr_mode_share": "group_feature_mode_share",
        "group_corr_effective_rank": "group_feature_effective_rank",
        "group_corr_mode_localization": "group_feature_mode_localization",
        "group_corr_spectral_gap": "group_feature_spectral_gap",
        "group_corr_second_mode_localization": "group_feature_second_mode_localization",
    }
    for alias, canonical in _corr_to_feature.items():
        try:
            OperatorRegistry.register_alias(
                alias,
                canonical,
                replacement_reason=(
                    "renamed: SVD over the members' feature matrix, not a "
                    "correlation-matrix eigenspectrum"
                ),
            )
        except Exception:  # pragma: no cover - idempotent across reloads
            pass


_register_surface()
