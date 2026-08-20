# -*- coding: utf-8 -*-
"""Native Polars backend for the multi-panel component score operator.

R25-184 axis fix: each component is a SEPARATE ``date x instrument`` panel.
The score is computed PER CELL across the component slots — never by treating
the columns of one wide DataFrame as components (that would score instruments
as components in a standard panel).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})
MAX_COMPONENTS = 8


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


def fin_component_score(*panels, component_directions=None, score_weights=None, missing_policy="score_available"):
    # Each positional arg is a separate date x instrument component panel.
    provided = [p for p in panels if p is not None]
    if not provided:
        raise ValueError("fin_component_score requires at least one component panel")
    base = provided[0]
    n_components = len(provided)
    cols = _cols(base)
    rows = base.height
    arrays = []
    for frame in provided:
        arr = np.stack([frame[c].to_numpy() for c in cols], axis=1)
        arrays.append(arr)
    stacked = np.stack(arrays, axis=2)  # (rows, cols, components)
    if isinstance(component_directions, str):
        directions = [component_directions] * n_components
    elif component_directions is None:
        directions = ["up"] * n_components
    else:
        directions = list(component_directions)
        if len(directions) != n_components:
            raise ValueError("component_directions length must equal number of component panels")
    if score_weights is None:
        weights = [1.0] * n_components
    else:
        weights = [float(w) for w in score_weights]
        if len(weights) != n_components:
            raise ValueError("score_weights length must equal number of component panels")
    contributions = np.stack(
        [_score_component(stacked[:, :, i], directions[i]) * weights[i] for i in range(n_components)],
        axis=2,
    )
    if missing_policy == "require_full":
        # R30 §18 parity: a score is only comparable when every component is
        # finite at that cell; a missing component makes it UNKNOWN (NaN).
        all_finite = np.all(np.isfinite(contributions), axis=2)
        with np.errstate(invalid="ignore"):
            scored = np.nansum(contributions, axis=2)
        scored = np.where(all_finite, scored, np.nan)
    else:
        with np.errstate(invalid="ignore"):
            scored = np.nansum(contributions, axis=2)
        any_finite = np.any(np.isfinite(contributions), axis=2)
        scored = np.where(any_finite, scored, np.nan)
    return pl.DataFrame({c: scored[:, i] for i, c in enumerate(cols)})


metadata = OperatorMetadata(
    name="fin_component_score",
    category="fundamental_period",
    description="Weighted multi-panel component score (date×instrument per component).",
    param_names=[f"component_{i}" for i in range(1, MAX_COMPONENTS + 1)]
    + ["component_directions", "score_weights", "missing_policy"],
    return_type="series",
    tags=["fundamental", "polars", "native", "component_stack_axis", "multi_panel_components"],
)


def _calculate_series(self, *args, **kwargs):
    return fin_component_score(*args, **kwargs)


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
