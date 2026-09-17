"""Reviewed R17 repairs for malformed four-input A-share limit event calls."""
from __future__ import annotations

import ast

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536
_RAW = "StockDailyBar"
_ADJ = "StockDailyBarAdj"


def _field(node: ast.AST, name: str, tables: set[str]) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "field" and len(node.args) == 1 and len(node.keywords) == 1
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == name
        and any(
            keyword.arg == "table" and isinstance(keyword.value, ast.Constant)
            and keyword.value.value in tables for keyword in node.keywords
        )
    )


def _supports(logic: str, *words: str) -> bool:
    value = logic.casefold()
    return any(word.casefold() in value for word in words)


def _supports_direction(logic: str, *, positive: tuple[str, ...], negative: tuple[str, ...]) -> bool:
    """Require explicit direction evidence and reject any contradictory label."""
    normalized = logic.casefold()
    return (
        any(word.casefold() in normalized for word in positive)
        and not any(word.casefold() in normalized for word in negative)
    )


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[:lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    return _offset(lines,node.lineno,node.col_offset), _offset(lines,node.end_lineno,node.end_col_offset)


def migrate_catalog_r17_limit_formula(
    formula: str, *, logic: str = "", enabled: bool = False,
) -> RecipeMigration:
    """Rewrite only the reviewed OHLC+limits malformed shape; otherwise no-op."""
    if not isinstance(formula,str) or not isinstance(logic,str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode()) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled:
        return RecipeMigration(formula,())
    tree=ast.parse(formula,mode="eval"); lines=formula.splitlines(keepends=True) or [""]
    edits=[]; changes=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.keywords or len(node.args)!=4:
            continue
        name=node.func.id; a=node.args
        common=(
            _field(a[0],"open",{_RAW}) and _field(a[1],"close",{_RAW})
            and _field(a[2],"high_limit",{_RAW,_ADJ})
            and _field(a[3],"low_limit",{_RAW,_ADJ})
        )
        if not common: continue
        if name=="ashare_limit_up_touch" and _supports_direction(
            logic,
            positive=("涨停", "up limit", "limit_up"),
            negative=("跌停", "down limit", "limit_down", "failed limit", "failed_limit", "炸板"),
        ):
            replacement="ashare_limit_up_touch(field('high', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"
            change="SEMANTIC_REDESIGN ashare_limit_up_touch: reviewed OHLC+limits bundle selects raw intraday high and raw official high_limit; open/close/low_limit discarded; canonical default tick_tolerance=0.005 made explicit"
        elif name=="ashare_limit_down_touch" and _supports_direction(
            logic,
            positive=("跌停", "down limit", "limit_down"),
            negative=("涨停", "up limit", "limit_up", "failed limit", "failed_limit", "炸板"),
        ):
            replacement="ashare_limit_down_touch(field('low', table='StockDailyBar'), field('low_limit', table='StockDailyBar'), 0.005)"
            change="SEMANTIC_REDESIGN ashare_limit_down_touch: reviewed OHLC+limits bundle selects raw intraday low and raw official low_limit; open/close/high_limit discarded; canonical default tick_tolerance=0.005 made explicit"
        elif name=="ashare_limit_failed" and _supports_direction(
            logic,
            positive=("炸板", "failed limit", "failed_limit", "未封住"),
            negative=("跌停", "down limit", "limit_down"),
        ):
            replacement="ashare_limit_failed(field('high', table='StockDailyBar'), field('close', table='StockDailyBar'), field('high_limit', table='StockDailyBar'), 0.005)"
            change="SEMANTIC_REDESIGN ashare_limit_failed: reviewed OHLC+limits bundle selects raw intraday high, raw close, and raw official high_limit; open/low_limit discarded; canonical default tick_tolerance=0.005 made explicit"
        else: continue
        start,end=_span(lines,node); edits.append((start,end,replacement)); changes.append(change)
    ordered=sorted(edits)
    if any(b<e for (_,e,_),(b,_,_) in zip(ordered,ordered[1:])):
        return RecipeMigration(formula,())
    out=formula
    for start,end,replacement in reversed(ordered): out=out[:start]+replacement+out[end:]
    return RecipeMigration(out,tuple(changes))
