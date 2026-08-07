# -*- coding: utf-8 -*-
"""Cross-sectional spectral crowding operators (2026-08 V2, P1).

Within each group (industry / sector / relation), each date, the members'
2..4 feature vectors are stacked, z-scored in the group and SVD'd.  One
decomposition yields three regime / crowding signals broadcast to every member:

* ``group_corr_mode_share``        — σ1²/Σσ²: how dominant the top mode is
  (group collapsing into one common trading pattern).
* ``group_corr_effective_rank``    — exp(-Σ p log p), p=σ²/Σσ²: how many
  effective dimensions the group uses.
* ``group_corr_mode_localization`` — Σ u1^4 (inverse participation ratio of the
  top left singular vector across members): a leader-only mode is localized in
  a few names.

The eigenspectrum of the within-group correlation structure switches with market
regime (RMT literature on the A-share market); these make that switch a daily
cross-sectional state.  Deterministic (SVD, no randomness).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


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


def _spectrum_stats(Z: np.ndarray) -> tuple[float, float, float] | None:
    """SVD spectrum -> (mode_share, effective_rank, mode_localization)."""
    n, d = Z.shape
    if n < 2 or d < 2:
        return None
    try:
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if S.shape[0] < 1 or not np.all(np.isfinite(S)):
        return None
    s2 = S * S
    total = float(s2.sum())
    if total <= _EPS:
        return None
    p = s2 / total
    mode_share = float(p[0])
    p_pos = p[p > 0.0]
    eff_rank = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    u1 = U[:, 0]
    localization = float(np.sum(u1 * u1 * u1 * u1))
    return mode_share, eff_rank, localization


def _group_spectrum_series(feats: np.ndarray, group: np.ndarray, which: str) -> np.ndarray:
    rows, cols, d = feats.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for t in range(rows):
        row = feats[t]
        g_row = group[t]
        labels = pd.unique(g_row)
        for label in labels:
            idx = np.flatnonzero(g_row == label)
            if idx.size < 2:
                continue
            Z = row[idx].astype(float)
            med = np.median(Z, axis=0)
            mad = 1.4826 * np.median(np.abs(Z - med), axis=0)
            scale = np.where(mad > _EPS, mad, np.std(Z, axis=0))
            if np.any(~np.isfinite(scale)) or np.any(scale <= _EPS):
                continue
            Z = (Z - med) / scale
            stats = _spectrum_stats(Z)
            if stats is None:
                continue
            if which == "mode_share":
                value = stats[0]
            elif which == "effective_rank":
                value = stats[1]
            else:
                value = stats[2]
            out[t, idx] = value
    return out


@register_operator(
    name="group_corr_mode_share",
    category="group_structure",
    business_category="group_structure",
    canonical="group_corr_mode_share",
    source="group_spectrum",
)
class GroupCorrModeShare(SeriesOperator):
    """组内特征谱 top-mode 占比 ``σ1²/Σσ²``。

    每组每日对成员的特征矩阵做组内 z-score + SVD；接近 1 = 整个组的横截面
    变异集中在一个共同方向（群体坍缩成共同交易模式 / 单因子结构）。组内每只
    股票得到同一值。P1。
    """

    metadata = _metadata(
        "group_corr_mode_share",
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
    name="group_corr_effective_rank",
    category="group_structure",
    business_category="group_structure",
    canonical="group_corr_effective_rank",
    source="group_spectrum",
)
class GroupCorrEffectiveRank(SeriesOperator):
    """组内特征谱有效秩 ``exp(-Σ p log p)``。

    p = σ²/Σσ²。有效秩低（接近 1）= 组内只有一两个有效维度（风格解体 /
    抱团）；高 = 组内结构分散多元。与 mode_share 互补（mode_share 只看头名，
    有效秩看整体谱宽）。组内每只股票同一值。P1。
    """

    metadata = _metadata(
        "group_corr_effective_rank",
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
    name="group_corr_mode_localization",
    category="group_structure",
    business_category="group_structure",
    canonical="group_corr_mode_localization",
    source="group_spectrum",
)
class GroupCorrModeLocalization(SeriesOperator):
    """组内主导模式的成员局域化 ``Σ u1^4``（inverse participation ratio）。

    top 左奇异向量 u1 在成员上的四次方和：接近 1 = 主导模式由极少数股票驱动
    （leader-only rally）；低 = 模式均匀分布在全组。组内每只股票同一值。P1。
    """

    metadata = _metadata(
        "group_corr_mode_localization",
        "组内 top-mode 局域化 Σu1^4（leader-only vs 均匀）。",
        ["f1", "f2", "f3", "group"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, group: pd.DataFrame, **_: Any
    ) -> pd.DataFrame:
        feats = np.stack([f.to_numpy(dtype=float) for f in (f1, f2, f3)], axis=2)
        return frame_like(f1, _group_spectrum_series(feats, group.to_numpy(), "localization"))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "group_corr_mode_share",
            "group_corr_effective_rank",
            "group_corr_mode_localization",
        }
    )


_register_surface()
