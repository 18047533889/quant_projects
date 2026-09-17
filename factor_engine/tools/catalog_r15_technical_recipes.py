"""Opt-in, exact-shape Stochastic %K recipes for catalog R15."""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536
_TABLE = "StockDailyBarAdj"
_OHLCV = ("open", "high", "low", "close", "volume")


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[:lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return _offset(lines, node.lineno, node.col_offset), _offset(
        lines, node.end_lineno, node.end_col_offset
    )


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def _field(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
        and len(node.args) == 1
        and _literal(node.args[0], name)
        and len(node.keywords) == 1
        and node.keywords[0].arg == "table"
        and _literal(node.keywords[0].value, _TABLE)
    )


def _source(formula: str, lines: list[str], node: ast.AST) -> str:
    start, end = _span(lines, node)
    return formula[start:end]


def _window(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
        return node.value
    return None


def migrate_catalog_r15_technical_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Expand reviewed StochasticK calls into equivalent supported primitives."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula, ())
    normalized_logic = logic.casefold()

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StochasticK"
            and not node.keywords
        ):
            continue
        high_node: ast.AST | None = None
        low_node: ast.AST | None = None
        close_node: ast.AST | None = None
        window: int | None = None
        shape = ""
        decision_kind = ""
        if (
            len(node.args) == 4
            and _field(node.args[0], "high")
            and _field(node.args[1], "low")
            and _field(node.args[2], "close")
        ):
            high_node, low_node, close_node = node.args[:3]
            window = _window(node.args[3])
            shape = "qualified StockDailyBarAdj high/low/close"
            decision_kind = "EXACT_SHAPE_RECIPE"
        elif (
            len(node.args) == 5
            and ("stochastic" in normalized_logic or "stoch" in normalized_logic)
            and all(_field(arg, name) for arg, name in zip(node.args, _OHLCV))
        ):
            high_node, low_node, close_node = node.args[1:4]
            window = 14
            shape = "StockDailyBarAdj OHLCV catalog bundle"
            decision_kind = "EXPLICIT_CONTRACT_REPAIR"
        if high_node is None or low_node is None or close_node is None or window is None:
            continue
        high = _source(formula, lines, high_node)
        low = _source(formula, lines, low_node)
        close = _source(formula, lines, close_node)
        rolling_high = f"ts_max({high}, {window})"
        denominator = f"subtract({rolling_high}, ts_min({low}, {window}))"
        numerator = f"subtract({close}, ts_min({low}, {window}))"
        # Canonical StochasticK masks an exactly-zero range only.  safe_div_null is
        # intentionally not used for the ratio because its positive epsilon
        # would also erase tiny nonzero ranges and its finite mask would erase
        # infinities that raw canonical division preserves.
        replacement = (
            f"where(eq({denominator}, 0.0), safe_div_null(0.0, 0.0), "
            f"multiply(100.0, divide({numerator}, {denominator})))"
        )
        start, end = _span(lines, node)
        edits.append((start, end, replacement))
        changes.append(
            f"{decision_kind} StochasticK: {shape}; window={window}; "
            "rolling high/low min_periods=1; exact zero range maps to null; "
            "tiny nonzero ranges and raw nonfinite division follow canonical semantics"
            + (
                "; malformed five-argument catalog call restored to the formal "
                "high/low/close contract with assumed default window 14; open/volume discarded"
                if decision_kind == "EXPLICIT_CONTRACT_REPAIR" else ""
            )
        )

    ordered = sorted(edits)
    if any(next_start < end for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])):
        return RecipeMigration(formula, ())
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
