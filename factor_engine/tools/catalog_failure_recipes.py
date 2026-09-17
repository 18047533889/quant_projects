"""Opt-in repairs for reviewed catalog calls with obsolete signatures.

Every repair in this module is a semantic redesign, not an equivalence claim.
The legacy arguments are retained in each audit message, while the formula is
rewritten only for an exact reviewed shape corroborated by its factor logic.
"""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    line = lines[lineno - 1]
    column = len(line.encode("utf-8")[:byte_col].decode("utf-8"))
    return sum(len(item) for item in lines[: lineno - 1]) + column


def _span(formula: str, lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (
        _offset(lines, node.lineno, node.col_offset),
        _offset(lines, node.end_lineno, node.end_col_offset),
    )


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def _has_logic(logic: str, *needles: str) -> bool:
    normalized = logic.casefold()
    return any(needle.casefold() in normalized for needle in needles)


def _keyword_map(node: ast.Call) -> dict[str, ast.keyword] | None:
    if any(keyword.arg is None for keyword in node.keywords):
        return None
    result = {keyword.arg: keyword for keyword in node.keywords}
    return result if len(result) == len(node.keywords) else None


def migrate_catalog_failure_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Repair only exact, user-reviewed failure recipes.

    With ``enabled=False`` this API is a strict no-op.  Unsupported spellings,
    already canonical calls, and calls whose logic does not corroborate the
    redesign fail closed.
    """

    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        keywords = _keyword_map(node)
        if keywords is None:
            continue

        if name == "ts_ordinal_irreversibility":
            if (
                len(node.args) == 1
                and set(keywords) == {"window", "order", "delay", "min_periods"}
                and _literal(keywords["window"].value, 120)
                and _literal(keywords["order"].value, 3)
                and _literal(keywords["delay"].value, 1)
                and _literal(keywords["min_periods"].value, 5)
                and _has_logic(logic, "irrevers", "不可逆", "时间反演")
            ):
                target = keywords["min_periods"]
                start = _offset(lines, target.lineno, target.col_offset)
                edits.append((start, start + len("min_periods"), "min_patterns"))
                changes.append(
                    "SEMANTIC_REDESIGN ts_ordinal_irreversibility: "
                    "min_periods=5 -> min_patterns=5; window=120; order=3; "
                    "delay=1; support unit changes from observations to ordinal patterns"
                )
            continue

        if name == "ts_extreme_cluster_ratio":
            if (
                not node.keywords
                and len(node.args) == 6
                and all(
                    _literal(argument, value)
                    for argument, value in zip(node.args[1:], (60, 2.0, 0.0, 5, 5))
                )
                and _has_logic(logic, "extreme", "cluster", "极端", "集中", "聚集")
            ):
                first_start, first_end = _span(formula, lines, node.args[0])
                replacement = (
                    f"ts_extreme_cluster_ratio({formula[first_start:first_end]}, 60, "
                    "'quantile', 0.9, 'absolute', 5)"
                )
                start, end = _span(formula, lines, node)
                edits.append((start, end, replacement))
                changes.append(
                    "SEMANTIC_REDESIGN ts_extreme_cluster_ratio: legacy "
                    "(window=60, 2.0, 0.0, 5, 5) -> canonical "
                    "(window=60, threshold='quantile', q=0.9, side='absolute', "
                    "min_periods=5); invalid legacy positions had no supported meaning"
                )
            continue

        if name == "ts_mmd_rbf_shift":
            if (
                len(node.args) == 1
                and set(keywords) == {"window"}
                and _literal(keywords["window"].value, 60)
                and _has_logic(logic, "mmd", "分布变化", "分布漂移", "distribution shift")
            ):
                first_start, first_end = _span(formula, lines, node.args[0])
                replacement = (
                    f"ts_mmd_rbf_shift({formula[first_start:first_end]}, "
                    "recent_window=20, old_window=40)"
                )
                start, end = _span(formula, lines, node)
                edits.append((start, end, replacement))
                changes.append(
                    "SEMANTIC_REDESIGN ts_mmd_rbf_shift: window=60 -> "
                    "recent_window=20, old_window=40; uses canonical default split "
                    "over the stated 60-row total horizon"
                )
            continue

        if name == "ts_tail_ratio":
            if (
                not node.keywords
                and len(node.args) == 5
                and all(
                    _literal(argument, value)
                    for argument, value in zip(node.args[1:], (60, 2.0, 0.0, 5))
                )
                and _has_logic(logic, "tail", "尾部", "长尾")
            ):
                first_start, first_end = _span(formula, lines, node.args[0])
                replacement = (
                    f"ts_tail_ratio({formula[first_start:first_end]}, 60, 0.05, 0.95, 5)"
                )
                start, end = _span(formula, lines, node)
                edits.append((start, end, replacement))
                changes.append(
                    "SEMANTIC_REDESIGN ts_tail_ratio: legacy q_low=2.0, q_high=0.0 "
                    "-> canonical defaults q_low=0.05, q_high=0.95; window=60; "
                    "min_periods=5; invalid legacy quantiles had no supported meaning"
                )
            continue

        if name == "KeltnerPosition":
            if (
                not node.keywords
                and len(node.args) == 6
                and ast.dump(node.args[2]) == ast.dump(node.args[3])
                and _literal(node.args[4], 20)
                and _literal(node.args[5], 2.0)
                and _has_logic(logic, "keltner", "channel", "通道", "位置")
            ):
                parts = []
                for argument in node.args[:3]:
                    arg_start, arg_end = _span(formula, lines, argument)
                    parts.append(formula[arg_start:arg_end])
                replacement = f"KeltnerPosition({', '.join(parts)}, 20, 20, 2.0)"
                start, end = _span(formula, lines, node)
                edits.append((start, end, replacement))
                changes.append(
                    "SEMANTIC_REDESIGN KeltnerPosition: duplicated close in ema_window "
                    "slot -> ema_window=20, atr_window=20, multiplier=2.0; equal "
                    "horizons follow the reviewed channel-position recipe"
                )

    ordered = sorted(edits)
    if any(start < previous_end for (_, previous_end, _), (start, _, _) in zip(ordered, ordered[1:])):
        return RecipeMigration(formula, ())

    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
