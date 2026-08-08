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
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_DAMPING_DEFAULT = 0.85
_MAX_ITER = 100
_CONV = 1e-10


def _metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="relation",
        description=description,
        param_names=params,
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
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        g_row = gv[r]
        positions: dict[Any, list[int]] = {}
        for i in range(cols):
            lab = g_row[i]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
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
            # edge flow i -> j proportional to node j's signal (j attracts links)
            A = np.maximum(sig, _EPS)[None, :] * np.ones((m, 1))
            rowsum = A.sum(axis=1, keepdims=True)
            rowsum[rowsum <= _EPS] = _EPS
            A = A / rowsum
            pr = np.full(m, 1.0 / m, dtype=float)
            for _ in range(_MAX_ITER):
                new_pr = (1.0 - damping) / m + damping * (A.T @ pr)
                if np.abs(new_pr - pr).max() < _CONV:
                    pr = new_pr
                    break
                pr = new_pr
            for pos, i in enumerate(idx):
                out[r, i] = float(pr[pos])
    return out


@register_operator(
    name="relation_pagerank_centrality",
    category="relation",
    business_category="relation",
    canonical="relation_pagerank_centrality",
    source="relation.ops_ext",
    status="implemented",
)
class RelationPagerankCentrality(SeriesOperator):
    """组内信号加权 PageRank 中心度（group 邻接的生产务实形态）。

    每日每个 group 内以信号 ``x`` 加权构造完全图，运行阻尼 PageRank，返回
    组内每只股票的中心度排名。无 peer 或信号非有限 → fail-closed NaN。
    ``damping`` 固定 0.85，不进入参数搜索。
    """

    metadata = _metadata(
        "relation_pagerank_centrality",
        "组内信号加权 PageRank 中心度（damping 固定 0.85）。",
        ["x", "group", "damping"],
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
            raise ValueError("relation_pagerank_centrality requires 0 < damping < 1")
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

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"relation_pagerank_centrality"}
    )
    from cleaned_operators.rolling_pack import register_polars_udf

    register_polars_udf("relation_pagerank_centrality")


_register_surface()
