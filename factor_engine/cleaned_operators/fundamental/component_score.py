# -*- coding: utf-8 -*-
"""Component scoring operator (Piotroski-style composite scores), multi-panel.

R25-078..082 / R25-184 (axis fix): the operator takes up to
``MAX_COMPONENTS`` SEPARATE ``date x instrument`` panels, one per component.
In the standard FactorEngine panel the columns ARE instruments, so a single
wide DataFrame whose columns are components would treat instruments as
components and broadcast one cross-sectional score back onto every stock column
— an implicit shape that is forbidden.  With per-component panels the score is
computed PER CELL across the component slots, so every instrument gets its own
genuine score.

A caller that passes a single ``date x instrument`` DataFrame is now
interpreted as component_1 (a legitimate single-component score); passing the
same panel where an explicit ComponentStack (component x date x instrument)
semantic type is required is rejected by the typed binding layer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.alignment import align_panel_inputs
from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

MAX_COMPONENTS = 8
_COMPONENT_PARAMS = [f"component_{i}" for i in range(1, MAX_COMPONENTS + 1)]


def _metadata(name: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="fundamental",
        description=(
            "按方向条件对多个 date×instrument 组件面板逐 cell 求和评分（如 Piotroski 类 "
            "F-score）。每个 component 是一个独立 date×instrument 面板；缺失组件该 cell 不计分，"
            "全缺失为 NaN。component_directions / score_weights / missing_policy 为可选标量参数。"
            "R30 §18: 条件满足→+1，条件不满足(完整数据)→0，非有限→NaN（三值逻辑）。"
        ),
        param_names=params,
        return_type="series",
        tags=[
            "fundamental", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:score",
            "unit:score", "cost:1",
            # R25-184: multi-panel component contract — each component is a
            # date×instrument panel.  The old single-wide-DataFrame (columns =
            # components) shape is REPLACED; a standard date×instrument panel is
            # a single component, never a component stack.
            "component_stack_axis", "multi_panel_components",
        ],
    )


def _score_component(component: np.ndarray, direction: str) -> np.ndarray:
    """Map a component array to +1 / -1 / 0 contributions.

    Three-valued logic (R30 §18, P0-013):
      * TRUE  (a finite value satisfies the direction)  -> +1 / -1;
      * FALSE (a finite value does NOT satisfy it)      -> 0  (NOT NaN);
      * UNKNOWN / MISSING (non-finite input)            -> NaN.
    The old ``np.where(finite & condition, 1.0, np.nan)`` collapsed FALSE into
    MISSING, so a fully-observed component that simply did not meet the
    condition became NaN and dragged the whole score down as if it were unknown.

    direction: "up" (value > 0 → +1), "down" (value < 0 → +1),
    "pos" (non-zero → +1) and "neg" (non-zero → -1).
    """
    finite = np.isfinite(component)
    if direction == "pos":
        cond = component != 0
    elif direction == "neg":
        cond = component != 0
    elif direction == "up":
        cond = component > 0
    elif direction == "down":
        cond = component < 0
    else:
        raise ValueError(f"unknown component direction: {direction!r}")
    out = np.zeros_like(component, dtype=float)
    out[finite & cond] = 1.0
    if direction == "neg":
        out[finite & cond] = -1.0
    out[~finite] = np.nan
    return out


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
    """对若干 date×instrument 组件面板按方向逐 cell 求和评分（多 panel 轴契约）。"""

    metadata = _metadata("fin_component_score", [*_COMPONENT_PARAMS, "component_directions", "score_weights", "missing_policy"])

    def _calculate_series(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        # The panel inputs are the positional component slots; the scalar params
        # arrive as kwargs (or trailing positional).  Separate them strictly:
        # every DataFrame/None positional is a component panel, the two scalar
        # params are never panels.
        panels: list[pd.DataFrame | None] = []
        for value in args:
            if value is None:
                panels.append(None)
            elif isinstance(value, pd.DataFrame):
                panels.append(value)
            elif isinstance(value, (str, list, tuple)) or value is None:
                # scalar param bound positionally — stop treating as panel
                break
            else:
                panels.append(value)

        directions = kwargs.get("component_directions")
        missing_policy = kwargs.get("missing_policy", "score_available")
        if missing_policy not in ("score_available", "require_full"):
            raise ValueError(
                f"missing_policy must be 'score_available' or 'require_full', got {missing_policy!r}"
            )
        weights = kwargs.get("score_weights")
        provided = [p for p in panels if p is not None]
        if not provided:
            raise ValueError("fin_component_score requires at least one component panel")
        # Strict alignment: every provided component panel must share the exact
        # date x instrument grid (R25-082 — a mismatched ComponentStack input is
        # a caller bug, never a silent reindex).
        aligned = align_panel_inputs(*provided, names=[f"component_{i}" for i in range(1, len(provided) + 1)])
        base = aligned[0]
        n_components = len(aligned)
        if isinstance(directions, str):
            direction_list = [directions] * n_components
        elif directions is None:
            direction_list = ["up"] * n_components
        else:
            direction_list = list(directions)
            if len(direction_list) != n_components:
                raise ValueError("component_directions length must equal number of component panels")
        if weights is None:
            weight_list = [1.0] * n_components
        else:
            weight_list = [float(w) for w in weights]
            if len(weight_list) != n_components:
                raise ValueError("score_weights length must equal number of component panels")

        arrays = [f.astype(float).to_numpy(dtype=float) for f in aligned]
        stacked = np.stack(arrays, axis=2)  # (rows, cols, components)
        contributions = np.stack(
            [
                _score_component(stacked[:, :, i], direction_list[i]) * weight_list[i]
                for i in range(n_components)
            ],
            axis=2,
        )
        if missing_policy == "require_full":
            # R30 §18: a score is only comparable when the component set is
            # complete at that cell — any missing component makes the cell
            # UNKNOWN (NaN), never a partial sum.
            all_finite = np.all(np.isfinite(contributions), axis=2)
            with np.errstate(invalid="ignore"):
                scored = np.nansum(contributions, axis=2)
            scored = np.where(all_finite, scored, np.nan)
        else:
            # ``score_available`` (default): visible components score, missing
            # ones are skipped (nansum).  R30 §18: the effective component
            # count must be observable so partial scores are not compared to
            # full scores as if equal — we expose it via the returned DataFrame
            # ``attrs`` for downstream comparability checks.
            with np.errstate(invalid="ignore"):
                scored = np.nansum(contributions, axis=2)
            any_finite = np.any(np.isfinite(contributions), axis=2)
            scored = np.where(any_finite, scored, np.nan)
        out = pd.DataFrame(scored, index=base.index, columns=base.columns, dtype=float)
        # R30 §18: record the effective (finite) component count per cell so
        # downstream consumers can refuse to compare partial and full scores.
        effective = np.sum(np.isfinite(contributions), axis=2)
        out.attrs["effective_component_count"] = effective
        out.attrs["missing_policy"] = missing_policy
        return out


__all__ = ["fin_component_score"]


import factor_engine.cleaned_operators.operator_surface as _surface  # noqa: E402
_surface.extend_extended_only({"fin_component_score"})
