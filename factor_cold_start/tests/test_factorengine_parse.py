from __future__ import annotations

import ast

from api.dsl_parser import parse_expr
from cleaned_operators.operator_surface import (
    DAILY_CANONICALS,
    EXTENDED_ONLY_CANONICALS,
    RESEARCH_ONLY_CANONICALS,
)
from factor_cold_start.catalog import load_catalog


def _assert_no_negative_lags(formula: str) -> None:
    positions = {
        "ts_delay": 1,
        "ts_delta": 1,
        "ts_pct": 1,
        "ts_log_return": 1,
        "ts_autocorr": 2,
    }
    for node in ast.walk(ast.parse(formula, mode="eval")):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        pos = positions.get(node.func.id)
        if pos is None or pos >= len(node.args):
            continue
        value = node.args[pos]
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            assert value.value >= 0


def test_all_daily_formulas_parse_on_daily_surface() -> None:
    for market in ("ashare", "us"):
        for row in load_catalog(market, "daily"):
            parse_expr(row.formula, surface="daily")
            assert set(row.operators) <= set(DAILY_CANONICALS)
            _assert_no_negative_lags(row.formula)


def test_all_extended_formulas_parse_on_compat_surface_without_research_ops() -> None:
    allowed = set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS)
    for market in ("ashare", "us"):
        for row in load_catalog(market, "extended"):
            parse_expr(row.formula, surface="compat")
            assert set(row.operators) <= allowed
            _assert_no_negative_lags(row.formula)


def test_all_research_formulas_parse_on_compat_research_surface() -> None:
    allowed = set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS) | set(RESEARCH_ONLY_CANONICALS)
    for market in ("ashare", "us"):
        for row in load_catalog(market, "research"):
            parse_expr(row.formula, surface="compat_research")
            assert set(row.operators) <= allowed
            _assert_no_negative_lags(row.formula)
