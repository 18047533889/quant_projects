# -*- coding: utf-8 -*-
"""Native Polars backend for the cross-sectional component score operator."""
from __future__ import annotations

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _cols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _SKIP]


def _score_component(component: np.ndarray, direction: str) -> np.ndarray:
    if direction == "pos":
        return np.where(np.isfinite(component) & (component != 0), 1.0, np.nan)
    if direction == "neg":
        return np.where(np.isfinite(component) & (component != 0), -1.0, np.nan)
    if direction == "up":
        return np.where(np.isfinite(component) & (component > 0), 1.0, np.nan)
    if direction == "down":
        return np.where(np.isfinite(component) & (component < 0), 1.0, np.nan)
    raise ValueError(f"unknown component direction: {direction!r}")


def fin_component_score(components, component_directions=None, score_weights=None):
    cols = _cols(components)
    n = components.height
    arr = np.stack([components[c].to_numpy() for c in cols], axis=1)
    if isinstance(component_directions, str):
        directions = [component_directions] * len(cols)
    elif component_directions is None:
        directions = ["up"] * len(cols)
    else:
        directions = list(component_directions)
        if len(directions) != len(cols):
            raise ValueError("component_directions length must equal number of component columns")
    if score_weights is None:
        weights = [1.0] * len(cols)
    else:
        weights = [float(w) for w in score_weights]
        if len(weights) != len(cols):
            raise ValueError("score_weights length must equal number of component columns")
    contributions = np.stack(
        [_score_component(arr[:, i], directions[i]) * weights[i] for i in range(len(cols))],
        axis=1,
    )
    with np.errstate(invalid="ignore"):
        scored = np.nansum(contributions, axis=1)
    any_finite = np.any(np.isfinite(contributions), axis=1)
    scored = np.where(any_finite, scored, np.nan)
    repeated = scored[:, None].repeat(len(cols), axis=1)
    return pl.DataFrame({c: repeated[:, i] for i, c in enumerate(cols)})


metadata = OperatorMetadata(
    name="fin_component_score",
    category="fundamental_period",
    description="Weighted score of component directions.",
    param_names=["components", "component_directions", "score_weights"],
    return_type="series",
    tags=["fundamental", "polars", "native"],
)


def _calculate_series(self, components, component_directions=None, score_weights=None, **kwargs):
    return fin_component_score(components, component_directions, score_weights)


cls = type(
    "PolarsComponentScore",
    (SeriesOperator,),
    {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
)
register_operator(
    name="fin_component_score",
    category="fundamental_period",
    business_category="fundamental",
    canonical="fin_component_score",
    source="polars_component_score",
    backend="polars",
    status="production",
)(cls)
