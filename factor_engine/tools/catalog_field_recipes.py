"""Opt-in, reviewed field recipes for catalog formulas.

The rewrites here are deliberately narrower than ordinary alias binding.  A
recipe is enabled only when the catalog row's reviewed source-table context
proves the physical source.  Unavailable aggregates and benchmark identities
are left untouched.
"""
from __future__ import annotations

import ast
import re

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536

_FLOW_FIELDS = {
    "OperatingRevenue_SP": ("StockIncome", "operating_revenue"),
    "OperatingCost_SP": ("StockIncome", "operating_cost"),
    "SaleExpense_SP": ("StockIncome", "selling_expense"),
    "AdministrationExpense_SP": ("StockIncome", "administration_expense"),
    "FinancialExpense_SP": ("StockIncome", "financial_expense"),
    "AssetImpairmentLoss_SP": ("StockIncome", "asset_impairment_loss"),
    "CreditImpairmentLoss_SP": ("StockIncome", "credit_impairment_loss"),
    "SustOperateNetProfit_SP": ("StockIncome", "sust_operate_net_profit"),
    "NetOperateCashFlow_SP": ("StockCashFlow", "operating_cash_flow"),
}

_BALANCE_PERIOD_CALLS = {
    "fin_delta_noa",
    "fin_goodwill_risk_score",
    "fiscal_pct_change",
}
_BALANCE_FIELDS = {
    "AccountReceivable",
    "CashEquivalents",
    "GoodWill",
    "Inventories",
    "LongtermLoan",
    "ShorttermLoan",
    "TaxsPayable",
    "TotalAssets",
    "TotalLiability",
}


def _offset(lines: list[str], lineno: int, byte_column: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_column]
    column = len(prefix.decode("utf-8"))
    return sum(len(line) for line in lines[: lineno - 1]) + column


def _period(table: str) -> str:
    return f'field("report_period_end_date", table="{table}")'


def _quarter(table: str, field_name: str) -> str:
    period = _period(table)
    return (
        f'fin_quarter_from_cumulative(field("{field_name}", table="{table}"), '
        f"{period}, ashare_fiscal_quarter_from_period_end({period}))"
    )


def _trace(original: str, decision: str, new: str) -> str:
    return f"ORIGINAL={original}; DECISION={decision}; NEW={new}"


def migrate_catalog_field_formula(
    formula: str, *, logic: str = "", tables: str = "", enabled: bool = False
) -> RecipeMigration:
    """Apply reviewed, source-qualified field recipes, failing closed.

    ``logic`` is accepted for the migration pipeline's common context API.  It
    is retained as audit context, never interpreted to invent a physical field.
    """

    if not enabled:
        return RecipeMigration(formula, ())
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str) or not isinstance(tables, str):
        raise TypeError("logic and tables must be strings")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")

    reviewed_tables = frozenset(
        token for part in re.split(r"[|,]", tables) if (token := part.strip())
    )
    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        replacement: str | None = None
        decision: str | None = None

        flow = _FLOW_FIELDS.get(node.id)
        if flow is not None and flow[0] in reviewed_tables:
            replacement = _quarter(*flow)
            decision = f"{flow[0]} cumulative-YTD flow -> canonical single quarter"
        elif node.id == "FiscalAvgAssets" and "StockBalance" in reviewed_tables:
            period = _period("StockBalance")
            replacement = (
                'fin_average_balance(field("total_assets", table="StockBalance"), '
                f"{period})"
            )
            decision = "StockBalance TotalAssets adjacent-fiscal-period average"
        elif node.id == "FiscalPeriodId":
            table = _period_table_for_node(node, parents, reviewed_tables)
            if table is not None:
                replacement = _period(table)
                decision = f"fiscal ordinal from {table}.ReportPeriodEndDate"
        elif (
            node.id in {"PubDate", "ReportPeriodEndDate"}
            and "StockIndicator" in reviewed_tables
            and not reviewed_tables.intersection(
                {"StockBalance", "StockIncome", "StockCashFlow"}
            )
        ):
            semantic = "pub_date" if node.id == "PubDate" else "report_period_end_date"
            replacement = f'field("{semantic}", table="StockIndicator")'
            decision = "original_tables uniquely selects StockIndicator event clock"

        if replacement is None or decision is None:
            continue
        start = _offset(lines, node.lineno, node.col_offset)
        end = _offset(lines, node.end_lineno, node.end_col_offset)
        edits.append((start, end, replacement))
        changes.append(_trace(formula[start:end], decision, replacement))

    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))


def _period_table_for_node(
    node: ast.Name,
    parents: dict[ast.AST, ast.AST],
    reviewed_tables: frozenset[str],
) -> str | None:
    current: ast.AST | None = node
    while current is not None and not isinstance(current, ast.Call):
        current = parents.get(current)
    if isinstance(current, ast.Call):
        names = {item.id for item in ast.walk(current) if isinstance(item, ast.Name)}
        flow_tables = {
            table for name, (table, _field) in _FLOW_FIELDS.items() if name in names
        }
        if len(flow_tables) == 1 and next(iter(flow_tables)) in reviewed_tables:
            return next(iter(flow_tables))
        if isinstance(current.func, ast.Name):
            callee = current.func.id
            if (
                callee in _BALANCE_PERIOD_CALLS
                and "StockBalance" in reviewed_tables
                and (callee != "fiscal_pct_change" or names.intersection(_BALANCE_FIELDS))
            ):
                return "StockBalance"
    return None
