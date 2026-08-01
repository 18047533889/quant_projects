# -*- coding: utf-8 -*-
"""Japanese candlestick aliases over one adaptive semantic engine."""
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry

_PATTERNS={
"cdl_2_crows":"2_crows","cdl_3_inside":"3_inside","cdl_3_outside":"3_outside","cdl_3_line_strike":"3_line_strike","cdl_3_stars_south":"3_stars_south","cdl_abandoned_baby":"abandoned_baby","cdl_advance_block":"advance_block","cdl_belt_hold":"belt_hold","cdl_breakaway":"breakaway","cdl_closing_marubozu":"closing_marubozu","cdl_counterattack":"counterattack","cdl_doji_star":"doji_star","cdl_evening_doji_star":"evening_doji_star","cdl_high_wave":"high_wave","cdl_homing_pigeon":"homing_pigeon","cdl_identical_3_crows":"identical_3_crows","cdl_in_neck":"in_neck","cdl_on_neck":"on_neck","cdl_thrusting":"thrusting","cdl_ladder_bottom":"ladder_bottom","cdl_long_legged_doji":"long_legged_doji","cdl_long_line":"long_line","cdl_short_line":"short_line","cdl_matching_low":"matching_low","cdl_mat_hold":"mat_hold","cdl_rickshaw_man":"rickshaw_man","cdl_rise_fall_3_methods":"rise_fall_3_methods","cdl_separating_lines":"separating_lines","cdl_stalled_pattern":"stalled_pattern","cdl_stick_sandwich":"stick_sandwich","cdl_takuri":"takuri","cdl_tasuki_gap":"tasuki_gap","cdl_tristar":"tristar","cdl_unique_3_river":"unique_3_river","cdl_upside_gap_2_crows":"upside_gap_2_crows","cdl_xside_gap_3_methods":"xside_gap_3_methods",
}
for name,pattern in _PATTERNS.items():
    FactorRecipeRegistry.register(FactorRecipe(
        name,"candlestick",f"Adaptive Japanese candlestick pattern: {pattern}",
        f"candlestick_pattern(open, high, low, close, '{pattern}', body_window, shadow_window, penetration)",
        ("open","high","low","close","body_window","shadow_window","penetration"),
        status="production",
    ))
