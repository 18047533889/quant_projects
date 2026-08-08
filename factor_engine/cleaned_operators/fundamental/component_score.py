# -*- coding: utf-8 -*-
"""Component scoring operator (Piotroski-style composite scores).

``fin_component_score`` sums per-component boolean contributions with an
optional weight.  It is a generic daily-panel operator: components can be any
comparable signal (profitability, leverage, liquidity, growth…).  Missing
components are ignored rather than treated as zero score; a completely empty
row yields NaN so the score never silently claims a stock with no data scored
zero.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental",
        description=(
            "按方向条件对组件求和评分（如 Piotroski 类 F-score）。"
            "component_directions / score_weights 为可选标量参数。"
        ),
        param_names=params,
        return_type="series",
        tags=[
            "fundamental", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:score",
            "unit:score", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _score_component(component: np.ndarray, direction: str) -> np.ndarray:
    """Map a component array to +1 / -1 / 0 contributions.

    direction: "up" (value > 0 → +1), "down" (value < 0 → +1),
    "pos" (non-zero → +1) and "neg" (non-zero → -1).
    """
    if direction == "pos":
        return np.where(np.isfinite(component) & (component != 0), 1.0, np.nan)
    if direction == "neg":
        return np.where(np.isfinite(component) & (component != 0), -1.0, np.nan)
    if direction == "up":
        return np.where(np.isfinite(component) & (component > 0), 1.0, np.nan)
    if direction == "down":
        return np.where(np.isfinite(component) & (component < 0), 1.0, np.nan)
    raise ValueError(f"unknown component direction: {direction!r}")


@register_operator(
    name="fin_component_score",
    category="fundamental",
    business_category="fundamental",
    canonical="fin_component_score",
    source="component_score",
    backend="pandas_numpy",
    status="experimental",
)
class FinComponentScore(SeriesOperator):
    """对若干组件按方向求和评分（支持 weights 与 component_directions 参数）。"""

    metadata = _metadata("fin_component_score", ["components"])

    def _calculate_series(
        self,
        components: pd.DataFrame,
        component_directions: str | list[str] | None = None,
        score_weights: list[float] | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        comp_df = components.apply(pd.to_numeric, errors="coerce") if isinstance(components, pd.DataFrame) else components
        arr = np.where(np.isfinite(comp_df), comp_df.to_numpy(dtype=float), np.nan)
        n_components = arr.shape[1]
        if isinstance(component_directions, str):
            directions = [component_directions] * n_components
        elif component_directions is None:
            directions = ["up"] * n_components
        else:
            directions = list(component_directions)
            if len(directions) != n_components:
                raise ValueError("component_directions length must equal number of component columns")
        if score_weights is None:
            weights_list = [1.0] * n_components
        else:
            weights_list = [float(w) for w in score_weights]
            if len(weights_list) != n_components:
                raise ValueError("score_weights length must equal number of component columns")

        contributions = np.stack(
            [_score_component(arr[:, i], directions[i]) * weights_list[i] for i in range(n_components)],
            axis=1,
        )
        with np.errstate(invalid="ignore"):
            scored = np.nansum(contributions, axis=1)
        any_finite = np.any(np.isfinite(contributions), axis=1)
        scored = np.where(any_finite, scored, np.nan)
        return _frame_like(components, scored[:, None].repeat(n_components, axis=1))


__all__ = ["fin_component_score"]


import cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.extend_extended_only({"fin_component_score"})


from cleaned_operators import operator_surface as _surface  # noqa: E402

_surface.extend_extended_only({"fin_component_score"})
