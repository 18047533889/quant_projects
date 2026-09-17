"""Source-backed repairs for legacy ``*_SP`` fiscal-flow shorthand."""
from __future__ import annotations

import ast

_MAX_BYTES = 65_536

# Legacy name -> (registered table, registered semantic field).
_CUMULATIVE_FIELDS = {
    "OperatingRevenue_SP": ("StockIncome", "operating_revenue"),
    "OperatingProfit_SP": ("StockIncome", "operating_profit"),
    "TotalProfit_SP": ("StockIncome", "total_profit"),
    "IncomeTaxExpense_SP": ("StockIncome", "income_tax_expense"),
    "NpParentCompanyOwners_SP": ("StockIncome", "np_parent_company_owners"),
    "SaleExpense_SP": ("StockIncome", "selling_expense"),
    "AdministrationExpense_SP": ("StockIncome", "administration_expense"),
    "RdExpenses_SP": ("StockIncome", "rd_expenses"),
    "FairValueVariableIncome_SP": ("StockIncome", "fair_value_variable_income"),
    "InvestmentIncome_SP": ("StockIncome", "investment_income"),
    "AssetDealIncome_SP": ("StockIncome", "asset_deal_income"),
    "OtherEarnings_SP": ("StockIncome", "other_earnings"),
    "NetOperateCashFlow_SP": ("StockCashFlow", "operating_cash_flow"),
    "FixIntanOtherAssetAcquiCash_SP": ("StockCashFlow", "fix_intan_other_asset_acquis_cash"),
    "GoodsSaleAndServiceRenderCash_SP": ("StockCashFlow", "goods_sale_and_service_render_cash"),
    "GoodsAndServicesCashPaid_SP": ("StockCashFlow", "goods_and_services_cash_paid"),
    "StaffBehalfPaid_SP": ("StockCashFlow", "staff_behalf_paid"),
    "TaxPayments_SP": ("StockCashFlow", "tax_payments"),
    "InvestWithdrawalCash_SP": ("StockCashFlow", "invest_withdrawal_cash"),
    "InvestProceeds_SP": ("StockCashFlow", "invest_proceeds"),
    "FixIntanOtherAssetDispoCash_SP": ("StockCashFlow", "fix_intan_other_asset_dispo_cash"),
    "NetCashFromSubCompany_SP": ("StockCashFlow", "net_cash_from_sub_company"),
    "CashFromBorrowing_SP": ("StockCashFlow", "cash_from_borrowing"),
    "CashFromBondsIssue_SP": ("StockCashFlow", "cash_from_bonds_issue"),
    "BorrowingRepayment_SP": ("StockCashFlow", "borrowing_repayment"),
}


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(map(len, lines[:lineno - 1])) + len(prefix.decode("utf-8"))


def _period(table: str) -> str:
    return f'field("report_period_end_date", table="{table}")'


def _quarter(table: str, field_name: str) -> str:
    period = _period(table)
    return (
        f'fin_quarter_from_cumulative(field("{field_name}", table="{table}"), '
        f'{period}, ashare_fiscal_quarter_from_period_end({period}))'
    )


def migrate_formula(formula: str, logic: str = "") -> tuple[str, list[str]]:
    """Expand only registry-proven fiscal shorthand; unknown names are preserved."""
    if not isinstance(formula, str) or not isinstance(logic, str):
        raise TypeError("formula and logic must be strings")
    if len(formula.encode()) > _MAX_BYTES:
        raise ValueError("formula exceeds migration input budget")
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return formula, []
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
        if isinstance(parents.get(node), ast.Call) and parents[node].func is node:
            continue
        replacement = None
        source = _CUMULATIVE_FIELDS.get(node.id)
        if source is not None:
            replacement = _quarter(*source)
            decision = f"registered {source[0]}.{source[1]} cumulative-YTD flow -> PIT single quarter"
        elif node.id == "FiscalAvgEquity":
            period = _period("StockBalance")
            replacement = f'fin_average_balance(field("total_owner_equities", table="StockBalance"), {period})'
            decision = "registered StockBalance.total_owner_equities -> adjacent fiscal-period average equity"
        else:
            continue
        start = _offset(lines, node.lineno, node.col_offset)
        end = _offset(lines, node.end_lineno, node.end_col_offset)
        edits.append((start, end, replacement))
        changes.append(f"FIELD_RECIPE ORIGINAL={formula[start:end]}; DECISION={decision}; NEW={replacement}")
    migrated = formula
    for start, end, replacement in sorted(edits, reverse=True):
        migrated = migrated[:start] + replacement + migrated[end:]
    return migrated, changes
