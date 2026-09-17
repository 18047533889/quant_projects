"""R19 opt-in repairs for reviewed typed-IR catalog failures.

The recipes preserve the original signal expression.  When the catalog used
return-only spectral entropy for an arbitrary series, the fallback is the
closely related spectral-flatness statistic (high means broadband/noisy, low
means narrow-band/structured); this is deliberately recorded as non-equivalent.
"""
from __future__ import annotations

import ast

_MAX_FORMULA_BYTES = 65_536
_ACTIVITY_FIELDS = frozenset({
    "amount", "volume", "turnover", "turnover_ratio", "minute_amount",
    "minute_volume",
})
_PRICE_FIELDS = frozenset({"close", "adj_close", "vwap"})
_BOOL_CALLS = frozenset({
    "and_", "or_", "not_", "eq", "ne", "gt", "ge", "lt", "le",
    "ashare_limit_up_touch", "ashare_limit_down_touch", "ashare_limit_failed",
    "ashare_limit_one_price", "ashare_open_at_upper_limit",
    "ashare_limit_open_failed",
})


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (_offset(lines, node.lineno, node.col_offset),
            _offset(lines, node.end_lineno, node.end_col_offset))


def _source(formula: str, lines: list[str], node: ast.AST) -> str:
    start, end = _span(lines, node)
    return formula[start:end]


def _is_ret(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"ret", "ret_1d"}
    if not (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "field" and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"ret", "return"}
    ):
        return False
    table = next((kw.value.value for kw in node.keywords
                  if kw.arg == "table" and isinstance(kw.value, ast.Constant)), None)
    return table == "StockDailyBarAdj"


def _bare_field(node: ast.AST, allowed: frozenset[str]) -> bool:
    """Recognize only catalog shorthands whose semantic kind is authoritative."""
    return isinstance(node, ast.Name) and node.id in allowed


def _is_bool_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in _BOOL_CALLS
    )


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Return a migrated formula and explicit audit messages.

    This function is intentionally syntax-directed and idempotent.  It never
    asserts a semantic type, disables validation, or substitutes an unrelated
    return series for the original input.
    """
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
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "ts_spectral_entropy" and node.args:
            x = node.args[0]
            if _is_ret(x):
                continue
            if _bare_field(x, _ACTIVITY_FIELDS):
                replacement_name = "ts_activity_spectral_entropy"
                message = (
                    "TYPE_REPAIR ts_spectral_entropy: nonnegative activity input retained; "
                    "use canonical ts_activity_spectral_entropy"
                )
            elif _bare_field(x, _PRICE_FIELDS):
                replacement_name = "ts_detrended_level_spectral_entropy"
                message = (
                    "TYPE_REPAIR ts_spectral_entropy: continuous adjusted price input retained; "
                    "use canonical ts_detrended_level_spectral_entropy"
                )
            else:
                replacement_name = "ts_spectral_flatness"
                message = (
                    "SEMANTIC_REDESIGN_NON_EQUIVALENT ts_spectral_entropy: original generic "
                    "series retained; return-only entropy replaced by spectral flatness, which "
                    "preserves the high=broadband/noisy, low=structured direction"
                )
            start, end = _span(lines, node.func)
            edits.append((start, end, replacement_name))
            changes.append(message)
        elif (
            node.func.id == "event_historical_response_mean"
            and len(node.args) >= 2 and _is_ret(node.args[1]) and not _is_ret(node.args[0])
        ):
            first = _source(formula, lines, node.args[0])
            second = _source(formula, lines, node.args[1])
            # Every accepted call here has a formal EventBool or MaskBool
            # producer declaration in Analyzer; unknown event-like names fail closed.
            if not _is_bool_expression(node.args[0]):
                continue
            event = first
            tail = [_source(formula, lines, arg) for arg in node.args[2:]]
            tail.extend(
                f"{kw.arg}={_source(formula, lines, kw.value)}"
                for kw in node.keywords if kw.arg is not None
            )
            args = [second, event, *tail]
            replacement = f"event_historical_response_mean({', '.join(args)})"
            start, end = _span(lines, node)
            edits.append((start, end, replacement))
            changes.append(
                "TYPE_REPAIR event_historical_response_mean: moved return response to the "
                "response slot and event signal to the event slot"
            )

    ordered = sorted(edits)
    if any(next_start < end for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])):
        return formula, []
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return migrated, changes
