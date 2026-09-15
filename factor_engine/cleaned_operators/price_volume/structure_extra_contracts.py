"""Explicit contracts for the twelve supplemental chart-pattern signatures."""
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

EXTRA_PATTERNS = frozenset({
    "pattern_triple_top", "pattern_triple_bottom", "pattern_123_bull", "pattern_123_bear",
    "pattern_rounding_bottom", "pattern_rounding_top", "pattern_cup", "pattern_cup_handle",
    "pattern_bull_pennant", "pattern_bear_pennant", "pattern_breakout_retest", "pattern_breakdown_retest",
})

def extra_contract(name, params):
    panel_set = {"close", "high", "low", "volume"}
    panel_params = tuple(p for p in params if p in panel_set)
    integer_minima = {
        "left_window": 1, "right_window": 1, "history_window": 1,
        "min_spacing": 1, "max_spacing": 1, "max_wait": 1,
        "cup_window": 5, "handle_window": 2, "impulse_window": 2, "pennant_window": 3,
        "window": 2 if name in {"pattern_breakout_retest", "pattern_breakdown_retest"} else 5,
    }
    specs = {}
    for p in params:
        if p in panel_set:
            continue
        if p in integer_minima:
            specs[p] = ParamSpec(dtype=int, min=integer_minima[p], param_role=ParamRole.HORIZON,
                                 history_formula="left_window + right_window + history_window" if p=="left_window" else None)
        else:
            specs[p] = ParamSpec(dtype=float, min=1e-12 if p=="tolerance" and name.startswith("pattern_triple") else 0.,
                                 param_role=ParamRole.STATE_THRESHOLD)
    return dict(panel_params=panel_params, scalar_params=tuple(specs), param_specs=specs)
