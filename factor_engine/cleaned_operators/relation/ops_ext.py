# -*- coding: utf-8 -*-
"""Relation-graph centrality primitives (2026-08-08 Gemini round).

``relation_pagerank_centrality`` follows the same production-feasible input
contract as ``relation_diffusion_score``: until the relation layer exposes a
true PIT typed graph (node/edge/direction/validity), the group panel defines
the adjacency.  Within each day the group's members form a complete weighted
graph whose edge flow into a node is proportional to the node's signal
(``x``); PageRank centrality then ranks names by signal-weighted link
structure.  ``damping`` is fixed (default 0.85), never searched.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.group_key import is_missing_group_key
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_DAMPING_DEFAULT = 0.85
_MAX_ITER = 100
_CONV = 1e-10


def _metadata(
    name: str,
    description: str,
    params: list[str],
    param_types: dict | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="relation",
        description=description,
        param_names=params,
        param_types=dict(param_types or {}),
        return_type="series",
        tags=[
            "relation", "daily", "pit_safe", "causal", "deterministic",
            f"signature:{','.join(params)}->series", "domain:relation",
            "unit:level", "cost:6",
        ],
    )


def _pagerank_series(
    xv: np.ndarray,
    gv: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Signal-weighted centrality on the group's complete graph (analytic).

    R11 #121: the group panel defines a COMPLETE graph whose edge flow into a
    node is proportional to the node's signal — every row of the adjacency is
    the same vector ``sig/S``, so the PageRank iteration collapses to the closed
    form ``pr[i] = (1-d)/m + d·sig[i]/S``.  There is no need for the iterative
    dense-matrix solve, and without real edge weights an iterative PageRank
    would be pure ceremony on a rank-1 matrix.  Resolve analytically instead.
    """
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        g_row = gv[r]
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            # R11 P1-10: unified missing-group-key rule (None / NaN / pd.NA /
            # NaT / "") so no phantom group can form.
            if is_missing_group_key(lab):
                continue
            positions.setdefault(lab, []).append(i)
        for lab, members in positions.items():
            if len(members) < 2:
                continue
            idx = np.asarray(members, dtype=int)
            sig = xv[r, idx]
            if not np.all(np.isfinite(sig)):
                continue
            m = idx.size
            total = float(np.sum(np.maximum(sig, _EPS)))
            if total <= _EPS:
                continue
            teleport = (1.0 - damping) / m
            # Analytic complete-graph PageRank: pr[i] = (1-d)/m + d*sig[i]/S.
            pr = teleport + damping * (np.maximum(sig, _EPS) / total)
            for pos, i in enumerate(idx):
                out[r, i] = float(pr[pos])
    return out


@register_operator(
    name="group_signal_attraction_share",
    category="relation",
    business_category="relation",
    canonical="group_signal_attraction_share",
    source="relation.ops_ext",
    # P0-008: research-only (fake complete-graph PageRank demotion).  ``experimental``
    # status is what keeps production_hardening._mark_experimental from certifying a
    # research-surface op — the six-gate governance invariant requires certified ops
    # to be daily/extended, and this op is intentionally not.
    status="experimental",
)
class GroupSignalAttractionShare(SeriesOperator):
    """组内信号吸引份额（无 PIT 关系图时的诚实命名，P1-28）。

    真正的 ``relation_pagerank_centrality`` 需要一张 PIT 关系图
    （source/target/weight/direction/validity）。在没有这张图之前，每日 group
    内以信号 ``x`` 构造的完全图令每个节点的出链分布完全相同，PageRank 退化为
    "组内 positive signal 归一化 + teleport 收缩"，并不利用任何"谁连谁"的图
    结构。因此本实现如实命名为 ``group_signal_attraction_share``；
    ``relation_pagerank_centrality`` 保留为 deprecated research-only 别名。
    ``damping`` 固定 0.85，不进入参数搜索面（P1-29）。
    """

    metadata = _metadata(
        "group_signal_attraction_share",
        "组内信号吸引份额（damping 固定 0.85）。",
        ["x", "group", "damping"],
        param_types={"damping": float},
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group: pd.DataFrame,
        damping: float = _DAMPING_DEFAULT,
        **_: Any,
    ) -> pd.DataFrame:
        x, group = x.copy(), group.copy()
        if not group.index.equals(x.index) or not group.columns.equals(x.columns):
            group = group.reindex(index=x.index, columns=x.columns)
        d = float(damping)
        if not (0.0 < d < 1.0):
            raise ValueError("group_signal_attraction_share requires 0 < damping < 1")
        return frame_like(
            x,
            _pagerank_series(
                x.to_numpy(dtype=float),
                group.to_numpy(),
                d,
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface
    from cleaned_operators.registry import OperatorRegistry

    # P0-008: both names RESEARCH_ONLY.  The group-panel adjacency is a fake
    # complete graph (edge flow proportional to node signal), so the output is
    # closer to a signal-weighted share than to a true network PageRank.  It
    # must not sit on the extended/daily mining surface until a real relation
    # graph layer (node/edge/weight/direction/validity) exists.
    _surface.extend_research_only({"group_signal_attraction_share"})
    from cleaned_operators.rolling_pack import register_polars_udf

    # Only the NEW canonical gets a real polars backend.  ``relation_pagerank_
    # centrality`` is a pure ALIAS below — registering a polars backend for it
    # would resurrect it as an active canonical with no explicit policy and break
    # finalize_layer_governance for every session.
    register_polars_udf("group_signal_attraction_share")
    try:
        OperatorRegistry.register_alias(
            "relation_pagerank_centrality",
            "group_signal_attraction_share",
            replacement_reason="renamed: no PIT relation graph yet, so the group "
            "complete-graph form is group_signal_attraction_share, not PageRank",
        )
    except Exception:  # pragma: no cover - idempotent across reloads
        pass


_register_surface()
