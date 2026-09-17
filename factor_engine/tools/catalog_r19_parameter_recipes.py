"""Fail-closed repairs for reviewed catalog formulas with obsolete parameters."""
from __future__ import annotations

import ast
import copy

_MAX_BYTES = 65_536
_STATE_WINDOW = {
    "category_age", "category_frequency", "category_transition_rate",
    "category_transition_surprise", "state_episode_age", "state_flip_age",
}
_PIVOT_HISTORY = {
    "ts_last_pivot_high", "ts_last_pivot_low", "ts_pivot_high_age", "ts_pivot_low_age",
}
_STRUCTURE_POINTS = {
    "ts_resistance_level", "ts_support_level", "ts_resistance_slope", "ts_support_slope",
    "ts_distance_to_resistance", "ts_distance_to_support", "ts_resistance_break", "ts_support_break",
}
_TAIL_SINGLE = {
    "ts_quantile_crossing_spectral_concentration", "ts_extremogram",
    "ts_extremal_dependence_decay",
}
_FISCAL_SIGNAL = {
    "fiscal_ar_resid_std", "fiscal_autocorr", "fiscal_direction_consistency",
    "fiscal_reversal_ratio", "fiscal_sign_consistency", "fiscal_standardized_surprise",
}
_TWO_WINDOW_DISTRIBUTION = {
    "ts_quantile_transport_slope", "ts_quantile_transport_curvature", "ts_mmd_rbf_shift",
}
_KNN_MIN_20 = {
    "cs_knn_local_linear_residual", "cs_knn_tangent_residual", "cs_knn_local_gradient_norm",
}
_EVENT_WINDOW = {
    "event_frequency", "event_fano_factor", "event_interval_memory",
    "event_local_variation", "event_fano_excess", "event_hawkes_branching_ratio_proxy",
}


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(map(len, lines[:lineno - 1])) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    assert node.end_lineno is not None and node.end_col_offset is not None
    return _offset(lines, node.lineno, node.col_offset), _offset(lines, node.end_lineno, node.end_col_offset)


def _lit(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Return a repaired formula and explicit audit records; unrecognised shapes are unchanged."""
    if not isinstance(formula, str) or not isinstance(logic, str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode()) > _MAX_BYTES:
        raise ValueError("formula exceeds migration input budget")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula, []
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []

    def replace(node: ast.AST, text: str) -> None:
        a, b = _span(lines, node); edits.append((a, b, text))

    def append(call: ast.Call, text: str) -> None:
        _, b = _span(lines, call); edits.append((b - 1, b - 1, text))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        kws = {k.arg: k for k in node.keywords if k.arg is not None}
        if len(kws) != len(node.keywords):
            continue

        if name in _FISCAL_SIGNAL and "x" in kws and "signal" not in kws:
            keyword = kws["x"]
            start = _offset(lines, keyword.lineno, keyword.col_offset)
            edits.append((start, start + 1, "signal"))
            changes.append(f"PARAMETER_ALIAS {name}: obsolete keyword x -> declared keyword signal; value expression unchanged")

        if name == "KAMA" and not node.keywords:
            if len(node.args) == 2:
                append(node, ", 2, 30")
                changes.append("SEMANTIC_REDESIGN KAMA: original KAMA(x, er_window) omitted fast/slow windows -> canonical fast_window=2, slow_window=30")
            elif len(node.args) == 5:
                a, b = _span(lines, node)
                close_a, close_b = _span(lines, node.args[3])
                edits.append((a, b, f"KAMA({formula[close_a:close_b]}, 10, 2, 30)"))
                changes.append("SEMANTIC_REDESIGN KAMA: original invalid OHLCV signature -> close-only KAMA(close, er_window=10, fast_window=2, slow_window=30); open/high/low/volume inputs removed")
            continue
        if name == "MACD_hist" and len(node.args) == 5 and not node.keywords:
            a, b = _span(lines, node)
            close_a, close_b = _span(lines, node.args[3])
            edits.append((a, b, f"MACD_hist({formula[close_a:close_b]}, 12, 26, 9)"))
            changes.append("SEMANTIC_REDESIGN MACD_hist: original invalid OHLCV signature -> close-only MACD_hist(close, fast=12, slow=26, signal=9); open/high/low/volume inputs removed")
            continue
        if name == "ts_rolling_sr_gaussian_mean_shift_score" and "side" in kws and _lit(kws["side"].value, "two_sided"):
            up = copy.deepcopy(node); down = copy.deepcopy(node)
            next(k for k in up.keywords if k.arg == "side").value = ast.Constant("up")
            next(k for k in down.keywords if k.arg == "side").value = ast.Constant("down")
            a, b = _span(lines, node)
            edits.append((a, b, f"flex_max({ast.unparse(up)}, {ast.unparse(down)})"))
            changes.append("SEMANTIC_REDESIGN ts_rolling_sr_gaussian_mean_shift_score: unsupported side='two_sided' -> maximum of supported up/down directional scores")
            continue
        if name == "candlestick_pattern" and "pattern" in kws and _lit(kws["pattern"].value, "hammer"):
            inactive = {"body_window", "shadow_window", "penetration"}
            if inactive.intersection(kws):
                repaired = copy.deepcopy(node)
                repaired.keywords = [k for k in repaired.keywords if k.arg not in inactive]
                a, b = _span(lines, node); edits.append((a, b, ast.unparse(repaired)))
                changes.append("PARAMETER_CANONICALIZATION candlestick_pattern hammer: removed inactive body_window/shadow_window/penetration knobs; canonical defaults preserve one AST")
                continue
        if name in {"group_peer_deviation_index", "group_peer_beta_deviation"} and len(node.args) == 2 and not node.keywords:
            value_a, value_b = _span(lines, node.args[0])
            group_a, group_b = _span(lines, node.args[1])
            value = formula[value_a:value_b]; group = formula[group_a:group_b]
            a, b = _span(lines, node)
            edits.append((a, b, f"subtract({value}, group_ex_self_mean({value}, {group}))"))
            if name == "group_peer_deviation_index":
                changes.append("SEMANTIC_REDESIGN group_peer_deviation_index: original supplied one signal plus IndustryCode to a five-signal aggregation -> same signal minus equal-weight industry peer mean excluding self; absent d2/d3/d4/d5 were not fabricated")
            else:
                changes.append("SEMANTIC_REDESIGN group_peer_beta_deviation: original omitted weight -> beta minus equal-weight industry peer mean excluding self; no weight field fabricated")
            continue
        if (
            name == "ts_multiscale_permutation_entropy_slope"
            and "window" in kws and "order" in kws and "min_patterns" in kws
            and _lit(kws["window"].value, 120) and _lit(kws["order"].value, 3)
            and _lit(kws["min_patterns"].value, 20)
        ):
            replace(kws["window"].value, "256")
            changes.append("SEMANTIC_REDESIGN ts_multiscale_permutation_entropy_slope: original window=120 -> minimal feasible window=256 for fixed scales [1,2,4,8], preserving order=3 and min_patterns=20; scale 8 then has 32 coarse points / 30 ordinal patterns, meeting max(min_patterns, 5*3!)=30; observation horizon changed non-equivalently")
            continue
        if name in _TWO_WINDOW_DISTRIBUTION and "window" in kws and _lit(kws["window"].value, 60) and len(node.keywords) == 1:
            repaired = copy.deepcopy(node)
            repaired.keywords = [ast.keyword(arg="recent_window", value=ast.Constant(20)), ast.keyword(arg="old_window", value=ast.Constant(40))]
            a, b = _span(lines, node); edits.append((a, b, ast.unparse(repaired)))
            changes.append(f"SEMANTIC_REDESIGN {name}: original undifferentiated window=60 -> recent_window=20 and old_window=40; this defines a 20-vs-prior-40 distribution shift and is not equivalent to the old expression")
            continue
        if name == "ts_downside_deviation" and len(node.args) == 4 and not node.keywords and _lit(node.args[1], 0.0) and _lit(node.args[2], 20) and _lit(node.args[3], 10):
            a, b = _span(lines, node)
            x_a, x_b = _span(lines, node.args[0]); x = formula[x_a:x_b]
            edits.append((a, b, f"ts_downside_deviation({x}, 20, 0.0, 10)"))
            changes.append("PARAMETER_REBIND ts_downside_deviation: original positional (x, target=0.0, window=20, min_periods=10) was bound against canonical (x, window, target, min_periods); reordered to window=20, target=0.0, min_periods=10 without changing intended values")
            continue
        if name == "ts_state_integral" and "upper" in kws and "lower" in kws and _lit(kws["upper"].value, 1.0) and isinstance(kws["lower"].value, ast.UnaryOp) and isinstance(kws["lower"].value.op, ast.USub) and _lit(kws["lower"].value.operand, 1.0):
            replace(kws["lower"].value, "0.0")
            changes.append("SEMANTIC_REDESIGN ts_state_integral: original signed lower=-1.0 is invalid because thresholds apply to |z|; redefined lower=0.0 with upper=1.0, retaining full within-state excess magnitude")
            continue
        if name in _KNN_MIN_20 and "k" in kws and _lit(kws["k"].value, 10):
            replace(kws["k"].value, "20")
            changes.append(f"SEMANTIC_REDESIGN {name}: original k=10 below supported cross-sectional minimum -> k=20; neighborhood definition changed non-equivalently")
            continue
        if name == "ts_markov_transition_surprisal" and "min_periods" in kws and _lit(kws["min_periods"].value, 3) and "min_count" not in kws:
            keyword = kws["min_periods"]
            start = _offset(lines, keyword.lineno, keyword.col_offset)
            edits.append((start, start + len("min_periods"), "min_count"))
            changes.append("PARAMETER_REBIND ts_markov_transition_surprisal: obsolete min_periods=3 -> formal transition-support policy min_count=3; min_state_support/min_history retain canonical defaults")
            continue
        if name == "ts_autocorr_decay_half_life" and len(node.args) == 5 and not node.keywords and _lit(node.args[1], 60) and _lit(node.args[2], 1) and _lit(node.args[3], 20) and _lit(node.args[4], 0.05):
            replace(node.args[2], "10"); replace(node.args[3], "False"); replace(node.args[4], "20")
            changes.append("SEMANTIC_REDESIGN ts_autocorr_decay_half_life: original five-position shape misbound 20 as use_abs and 0.05 as min_periods -> use_abs=False, min_periods=20; obsolete 0.05 threshold removed; impossible one-lag regression -> canonical max_lag=10")
            continue
        if name in _EVENT_WINDOW and len(node.args) == 2 and not node.keywords and not isinstance(node.args[1], ast.Constant):
            replace(node.args[1], "60")
            changes.append(f"SEMANTIC_REDESIGN {name}: original response panel was passed in scalar window position -> chosen explicit window=60 (not the operator default); response panel removed because this event-process statistic has no response input")
            continue

        if name == "cs_rank_gaussian" and len(node.args) == 2 and not node.keywords and _lit(node.args[1], 3.0):
            replace(node.args[1], "'blom'")
            changes.append("SEMANTIC_REDESIGN cs_rank_gaussian: original unsupported method=3.0 -> method='blom' (canonical plotting-position estimator)")
        elif name in _STATE_WINDOW and len(node.args) == 1 and not node.keywords:
            append(node, ", 252")
            changes.append(f"SEMANTIC_REDESIGN {name}: original omitted window -> explicit 252-trading-day lifecycle window")
        elif name in _PIVOT_HISTORY and len(node.args) == 3 and not node.keywords:
            append(node, ", 250")
            changes.append(f"SEMANTIC_REDESIGN {name}: original omitted history_window -> documented 250-bar bounded history")
        elif name in _STRUCTURE_POINTS and not node.keywords:
            expected = 3 if name in {"ts_resistance_level", "ts_support_level", "ts_resistance_slope", "ts_support_slope"} else 4
            if len(node.args) == expected:
                append(node, ", 250, 3")
                changes.append(f"SEMANTIC_REDESIGN {name}: original omitted history_window/points -> bounded 250-bar history with 3 confirmed pivots")
            elif len(node.args) == expected + 1:
                append(node, ", 3")
                changes.append(f"SEMANTIC_REDESIGN {name}: original omitted points -> minimum 3 confirmed pivots")
        elif name == "ts_quantile_skew" and len(node.args) == 6 and not node.keywords and _lit(node.args[-1], 0.9):
            for target, text in zip(node.args[2:], ("0.1", "0.5", "0.9", "5")):
                replace(target, text)
            changes.append("SEMANTIC_REDESIGN ts_quantile_skew: original five-quantile shape (0.1,0.25,0.75,0.9) misbound 0.9 as min_periods -> canonical Bowley (0.1,0.5,0.9), min_periods=5")
        elif name == "ts_conditional_transfer_entropy" and "bins" in kws and _lit(kws["bins"].value, 5):
            replace(kws["bins"].value, "2"); changes.append("SEMANTIC_REDESIGN ts_conditional_transfer_entropy: original bins=5 -> supported bins=2")
        elif name == "ts_markov_entropy_production" and "bins" in kws and _lit(kws["bins"].value, 6):
            replace(kws["bins"].value, "5"); changes.append("SEMANTIC_REDESIGN ts_markov_entropy_production: original bins=6 -> nearest supported bins=5")
        elif name == "ts_active_information_storage" and "bins" in kws and _lit(kws["bins"].value, 6):
            replace(kws["bins"].value, "2"); changes.append("SEMANTIC_REDESIGN ts_active_information_storage: original bins=6 -> supported bins=2")
        elif name in _TAIL_SINGLE and "quantile" in kws and _lit(kws["quantile"].value, 0.9) and "side" in kws and _lit(kws["side"].value, "upper"):
            replace(kws["quantile"].value, "0.1"); changes.append(f"SEMANTIC_REDESIGN {name}: original upper threshold quantile=0.9 -> API upper-tail probability=0.1 (same Q0.9 threshold)")
        elif name in {"ts_cross_extremogram", "ts_cross_quantilogram"}:
            for q, side in (("target_q", "target_side"), ("source_q", "source_side")):
                if q in kws and side in kws and _lit(kws[q].value, 0.9) and _lit(kws[side].value, "upper"):
                    replace(kws[q].value, "0.1"); changes.append(f"SEMANTIC_REDESIGN {name}.{q}: original upper threshold 0.9 -> upper-tail probability 0.1")
        elif name == "ts_gap_fill_ratio" and len(node.args) == 4 and not node.keywords and _lit(node.args[3], 0.01):
            replace(node.args[3], "60")
            changes.append("SEMANTIC_REDESIGN ts_gap_fill_ratio: original fourth positional 0.01 was invalid as window -> documented window=60; obsolete tolerance assumption removed")
        elif name == "ts_transition_count" and len(node.args) == 3 and not node.keywords and _lit(node.args[2], 0.0):
            replace(node.args[2], "'break'")
            changes.append("SEMANTIC_REDESIGN ts_transition_count: original numeric missing_policy=0.0 -> canonical missing_policy='break'")

    ordered = sorted(edits)
    if any(b > c for (_, b, _), (c, _, _) in zip(ordered, ordered[1:])):
        return formula, []
    migrated = formula
    for a, b, text in reversed(ordered):
        migrated = migrated[:a] + text + migrated[b:]
    return migrated, changes
