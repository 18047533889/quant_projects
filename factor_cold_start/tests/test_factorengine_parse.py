from __future__ import annotations

import ast

from api.dsl_parser import parse_expr
from factor_cold_start.catalog import load_catalog
from factor_cold_start.production_admission import admit_factor


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


def test_all_default_daily_formulas_are_production_admitted() -> None:
    for market in ("ashare", "us"):
        rows = load_catalog(market, "daily")
        assert rows, market
        for row in rows:
            # Daily-output production factors may use operators promoted from the
            # former Extended authoring surface, therefore parsing uses compat.
            parse_expr(row.formula, surface="compat")
            admission = admit_factor(row)
            assert admission.eligible, (row.factor_id, admission.violations)
            assert admission.canonical_operators, row.factor_id
            assert all(admission.backend_map.values()), row.factor_id
            _assert_no_negative_lags(row.formula)


def test_extended_archive_parses_but_makes_no_production_claim() -> None:
    for market in ("ashare", "us"):
        rows = load_catalog(market, "extended")
        assert rows, market
        for row in rows:
            parse_expr(row.formula, surface="compat")
            _assert_no_negative_lags(row.formula)
