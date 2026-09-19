"""Narrow R19 repairs for reviewed legacy operator spellings/shapes.

Only exact catalog shapes are rewritten.  The five-input technical calls came
from an erroneous generic OHLCV template; their irrelevant inputs are removed
and the registered operator's documented default horizon is made explicit.
"""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_r17_technical_recipes import (
    migrate_catalog_r17_technical_formula,
)
from factor_engine.tools.catalog_r18_technical_recipes import (
    migrate_catalog_r18_technical_formula,
)

_MAX_FORMULA_BYTES = 65_536
_TABLE = "StockDailyBarAdj"
_OHLCV = ("open", "high", "low", "close", "volume")


def _field(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == name
        and len(node.keywords) == 1
        and node.keywords[0].arg == "table"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == _TABLE
    )


def _legacy_ohlcv(node: ast.Call) -> bool:
    return not node.keywords and len(node.args) == 5 and all(
        _field(arg, name) for arg, name in zip(node.args, _OHLCV)
    )


def _aroon(close: str, window: int, direction: str) -> str:
    full = window + 1
    age = "ts_argmax" if direction == "up" else "ts_argmin"
    return (
        f"multiply(100.0, divide(subtract({window}, "
        f"{age}({close}, {full}, {full})), {window}))"
    )


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Return a repaired formula and an explicit audit trail."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula, []
    changes: list[str] = []

    class Lower(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript):
            node = self.generic_visit(node)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "IndustryCode"
                and isinstance(node.slice, ast.Name)
                and node.slice.id in {"sw_l1", "sw_l2", "sw_l3", "jq_l1", "jq_l2", "zjw"}
            ):
                taxonomy = node.slice.id
                if taxonomy == "sw_l1":
                    changes.append(
                        "EXACT_TAXONOMY_RECIPE IndustryCode[sw_l1]: use the registered "
                        "StockIndustry industry_code field; runtime required-filter default "
                        "is explicitly sw_l1"
                    )
                else:
                    changes.append(
                        f"R19_V1 SEMANTIC_REDESIGN IndustryCode[{taxonomy}] -> "
                        "registered StockIndustry industry_code default sw_l1: requested "
                        "taxonomy is unavailable in formula field binding; coarser/default "
                        "grouping is NOT equivalent"
                    )
                return ast.parse(
                    "field('industry_code', table='StockIndustry')", mode="eval"
                ).body
            return node

        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)
            if not isinstance(node.func, ast.Name):
                return node
            if node.func.id == "ts_persistence_entropy" and len(node.args) == 1:
                keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                if len(keywords) == len(node.keywords) and set(keywords) == {
                    "window", "tau", "dim"
                }:
                    window, tau, dim = (
                        keywords["window"], keywords["tau"], keywords["dim"]
                    )
                    if all(
                        isinstance(value, ast.Constant)
                        and type(value.value) is int
                        and value.value > 0
                        for value in (window, tau, dim)
                    ):
                        replacement = ast.Call(
                            func=ast.Name(id="ts_permutation_entropy", ctx=ast.Load()),
                            args=[node.args[0]],
                            keywords=[
                                ast.keyword(arg="window", value=window),
                                ast.keyword(arg="order", value=dim),
                                ast.keyword(arg="delay", value=tau),
                                ast.keyword(arg="normalize", value=ast.Constant(True)),
                            ],
                        )
                        changes.append(
                            "R19_V1 SEMANTIC_REDESIGN ts_persistence_entropy -> "
                            "ts_permutation_entropy: preserves rolling ordinal-complexity "
                            "intent but is not topological persistence entropy"
                        )
                        return replacement
            if (
                node.func.id == "intraday_volume_clock_path_roughness"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "MinuteVolume"
            ):
                changes.append(
                    "R19_V1 SEMANTIC_REDESIGN intraday_volume_clock_path_roughness "
                    "-> 1 - intraday_volume_clock_path_efficiency: adds real adjusted "
                    "minute_close price input and fixed buckets=16; original omitted price"
                )
                return ast.parse(
                    "subtract(1.0, intraday_volume_clock_path_efficiency("
                    "field('minute_close', table='StockMinuteBarAdj'), "
                    "field('minute_volume', table='StockMinuteBarAdj'), 16))",
                    mode="eval",
                ).body
            if (
                node.func.id == "intraday_volume_clock_path_efficiency"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "MinuteVolume"
            ):
                changes.append(
                    "R19_V1 SEMANTIC_REDESIGN intraday_volume_clock_path_efficiency: "
                    "adds real adjusted minute_close price input, uses MinuteVolume as "
                    "activity, and fixes buckets=16 because original omitted price"
                )
                return ast.parse(
                    "intraday_volume_clock_path_efficiency("
                    "field('minute_close', table='StockMinuteBarAdj'), "
                    "field('minute_volume', table='StockMinuteBarAdj'), 16)",
                    mode="eval",
                ).body
            if (
                node.func.id == "max_drawdown"
                and len(node.args) == 1
                and not node.keywords
                and (
                    isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "ret"
                    or _field(node.args[0], "ret")
                    or _field(node.args[0], "close")
                )
            ):
                changes.append(
                    "R19_V2 SEMANTIC_REDESIGN max_drawdown(ret-or-close) -> "
                    "ts_max_drawdown(adjusted close, 252): replaces undefined "
                    "full-history return compounding with a fixed 252-session "
                    "adjusted-price drawdown horizon"
                )
                return ast.parse(
                    "ts_max_drawdown(field('close', table='StockDailyBarAdj'), 252)",
                    mode="eval",
                ).body
            if (
                node.func.id in {"overnight_return", "open_gap", "close_gap", "open_close_return", "open_to_vwap_return"}
                and len(node.args) == 4
                and not node.keywords
                and all(
                    _field(arg, name)
                    for arg, name in zip(node.args, ("open", "close", "pre_close", "vwap"))
                )
            ):
                open_, close, pre_close, vwap = map(ast.unparse, node.args)
                replacements = {
                    "overnight_return": f"overnight_return({open_}, {pre_close})",
                    "open_gap": f"overnight_return({open_}, {pre_close})",
                    "close_gap": f"open_close_return({open_}, {close})",
                    "open_close_return": f"open_close_return({open_}, {close})",
                    "open_to_vwap_return": f"open_to_vwap_return({open_}, {vwap})",
                }
                changes.append(
                    f"R19_V2 TEMPLATE_REPAIR {node.func.id}: malformed generic "
                    "open/close/pre_close/vwap call reduced to the registered return "
                    "decomposition inputs; irrelevant template fields removed"
                )
                return ast.parse(replacements[node.func.id], mode="eval").body
            if (
                node.func.id == "ts_quantile_regression_beta"
                and len(node.args) == 2
                and not node.keywords
            ):
                y, x = map(ast.unparse, node.args)
                changes.append(
                    "R19_V2 SEMANTIC_REDESIGN ts_quantile_regression_beta -> "
                    "ts_quantile_regression_slope: preserves conditional median "
                    "slope direction with explicit window=120, q=0.5, min_periods=5; "
                    "registered estimator implementation differs"
                )
                return ast.parse(
                    f"ts_quantile_regression_slope({y}, {x}, 120, 0.5, 5)",
                    mode="eval",
                ).body
            if (
                node.func.id == "ts_transition_count"
                and len(node.args) == 3
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "ret"
                and isinstance(node.args[1], ast.Constant)
                and type(node.args[1].value) is int
                and node.args[1].value == 20
                and isinstance(node.args[2], ast.Constant)
                and node.args[2].value == "break"
            ):
                changes.append(
                    "R19_V3 SEMANTIC_REDESIGN ts_transition_count(ret, 20, 'break') "
                    "-> ts_transition_count(gt(ret, 0.0), 20, 'break'): the operator "
                    "requires a ConditionBool; defines ret > 0 as the positive state, "
                    "which is NOT equivalent to accepting continuous returns"
                )
                return ast.parse(
                    "ts_transition_count(gt(ret, 0.0), 20, 'break')", mode="eval"
                ).body
            if not _legacy_ohlcv(node):
                return node
            name = node.func.id
            open_, high, low, close, volume = map(ast.unparse, node.args)
            replacements = {
                "RSI": f"RSI_WILDER({close}, 14)",
                "ATR": f"ATR_WILDER({high}, {low}, {close}, 14)",
                "MACD": f"MACD_line({close}, 12, 26, 9)",
                "MACD_line": f"MACD_line({close}, 12, 26, 9)",
                "MACD_signal": f"MACD_signal({close}, 12, 26, 9)",
                "MOM": f"subtract({close}, ts_delay({close}, 10))",
                "ROC": (
                    f"multiply(100.0, safe_div_null("
                    f"subtract({close}, ts_delay({close}, 10)), "
                    f"ts_delay({close}, 10)))"
                ),
                "BollingerUpper": (
                    f"add(ts_mean({close}, 20), multiply(2.0, ts_std({close}, 20)))"
                ),
                "BollingerLower": (
                    f"subtract(ts_mean({close}, 20), multiply(2.0, ts_std({close}, 20)))"
                ),
                "AROON_up": _aroon(close, 25, "up"),
                "AROON_down": _aroon(close, 25, "down"),
                "FisherTransform": f"FisherTransform({high}, {low}, 9)",
                "CoppockCurve": f"CoppockCurve({close}, 14, 11, 10)",
                "QQE": f"QQE({close}, 14, 5, 4.236)",
                "ElderRay": f"ElderRay({high}, {low}, {close}, 13)",
            }
            replacement = replacements.get(name)
            if name == "AROON":
                canonical = f"AROON({close}, 25)"
                replacement = migrate_catalog_r17_technical_formula(
                    canonical, enabled=True
                ).formula
            elif name == "StochasticD":
                canonical = f"StochasticD({high}, {low}, {close}, 14)"
                replacement = migrate_catalog_r17_technical_formula(
                    canonical, enabled=True
                ).formula
            elif name == "TRIX":
                canonical = f"TRIX({close}, 15)"
                replacement = migrate_catalog_r18_technical_formula(
                    canonical, enabled=True
                ).formula
            elif name == "ADXR":
                canonical = f"ADXR({high}, {low}, {close}, 14)"
                replacement = migrate_catalog_r18_technical_formula(
                    canonical, enabled=True
                ).formula
            if replacement is None:
                return node
            changes.append(
                f"R19_V1 SEMANTIC_REPAIR {name}: erroneous generic OHLCV call replaced "
                "by the registered indicator signature with its documented default "
                "horizon; unused template inputs removed"
            )
            return ast.parse(replacement, mode="eval").body

    lowered = ast.fix_missing_locations(Lower().visit(tree))
    if not changes:
        return formula, []
    return ast.unparse(lowered), changes
