# -*- coding: utf-8 -*-
"""Fundamental recipes over PIT-aware, field-agnostic primitives.

Recipes in this module accept *single-report-period* flow inputs.  Fiscal-YTD
cumulative filing fields must use the explicit ``*_from_cumulative`` recipes in
``fundamental_cumulative.py`` or be converted with
``fin_quarter_from_cumulative`` first.
"""
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _recipe(
    name,
    description,
    expression,
    parameters,
    *,
    category="fundamental",
    status="production",
    replacement_for=(),
):
    return FactorRecipe(
        name,
        category,
        description,
        expression,
        tuple(parameters),
        status=status,
        replacement_for=tuple(replacement_for),
    )


_RECIPES = (
    _recipe(
        "current_ratio",
        "流动比率",
        "fin_ratio(current_assets, current_liabilities)",
        ("current_assets", "current_liabilities"),
        replacement_for=("current_ratio",),
    ),
    _recipe(
        "quick_ratio",
        "速动比率",
        "fin_ratio(current_assets - inventory, current_liabilities)",
        ("current_assets", "inventory", "current_liabilities"),
        replacement_for=("quick_ratio",),
    ),
    _recipe(
        "debt_to_equity",
        "负债权益比",
        "fin_ratio(total_debt, total_equity)",
        ("total_debt", "total_equity"),
        replacement_for=("debt_to_equity",),
    ),
    _recipe(
        "operating_margin",
        "营业利润率",
        "fin_ratio(operating_income, revenue)",
        ("operating_income", "revenue"),
        replacement_for=("operating_margin",),
    ),
    _recipe(
        "gross_margin",
        "毛利率",
        "fin_ratio(gross_profit, revenue)",
        ("gross_profit", "revenue"),
    ),
    _recipe(
        "net_margin",
        "净利率",
        "fin_ratio(net_income, revenue)",
        ("net_income", "revenue"),
    ),
    _recipe(
        "roa",
        "资产收益率（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(net_income, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("net_income", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "roe",
        "净资产收益率（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(net_income, period_id, periods_per_year), fin_average_balance(total_equity, period_id, 2))",
        ("net_income", "total_equity", "period_id", "periods_per_year"),
    ),
    _recipe(
        "roic_proxy",
        "投入资本回报率代理（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(nopat, period_id, periods_per_year), fin_average_balance(invested_capital, period_id, 2))",
        ("nopat", "invested_capital", "period_id", "periods_per_year"),
    ),
    _recipe(
        "gross_profitability",
        "毛利润资产收益（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(gross_profit, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("gross_profit", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "operating_profitability",
        "营业利润资产收益（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(operating_income, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("operating_income", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cash_profitability",
        "经营现金流资产收益（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("operating_cash_flow", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "accruals",
        "应计质量（单季度流量输入）",
        "fin_accrual_ratio(fin_ttm_quarterly(net_income, period_id, periods_per_year), fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        (
            "net_income",
            "operating_cash_flow",
            "total_assets",
            "period_id",
            "periods_per_year",
        ),
    ),
    _recipe(
        "asset_growth",
        "资产增长",
        "fin_yoy(total_assets, period_id, periods_per_year)",
        ("total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "inventory_growth",
        "存货增长",
        "fin_yoy(inventory, period_id, periods_per_year)",
        ("inventory", "period_id", "periods_per_year"),
    ),
    _recipe(
        "receivables_growth",
        "应收增长",
        "fin_yoy(receivables, period_id, periods_per_year)",
        ("receivables", "period_id", "periods_per_year"),
    ),
    _recipe(
        "revenue_growth",
        "收入增长",
        "fin_yoy(revenue, period_id, periods_per_year)",
        ("revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "earnings_growth",
        "利润增长",
        "fin_yoy(net_income, period_id, periods_per_year)",
        ("net_income", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cashflow_growth",
        "经营现金流增长",
        "fin_yoy(operating_cash_flow, period_id, periods_per_year)",
        ("operating_cash_flow", "period_id", "periods_per_year"),
    ),
    _recipe(
        "rd_intensity",
        "研发强度（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(research_development, period_id, periods_per_year), fin_ttm_quarterly(revenue, period_id, periods_per_year))",
        ("research_development", "revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "capex_intensity",
        "资本开支强度（单季度流量输入）",
        "fin_ratio(abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",
        ("capex", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "free_cash_flow_profitability",
        "自由现金流资产收益（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",
        (
            "operating_cash_flow",
            "capex",
            "total_assets",
            "period_id",
            "periods_per_year",
        ),
    ),
    _recipe(
        "net_debt_ratio",
        "净债务资产比",
        "fin_ratio(short_debt + long_debt - cash, total_assets)",
        ("short_debt", "long_debt", "cash", "total_assets"),
    ),
    _recipe(
        "interest_coverage",
        "利息保障倍数（单季度流量输入）",
        "fin_ratio(fin_ttm_quarterly(operating_income, period_id, periods_per_year), abs(fin_ttm_quarterly(interest_expense, period_id, periods_per_year)))",
        ("operating_income", "interest_expense", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cash_conversion",
        "现金利润转化率（单季度流量输入）",
        "fin_cash_conversion(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_ttm_quarterly(net_income, period_id, periods_per_year))",
        ("operating_cash_flow", "net_income", "period_id", "periods_per_year"),
    ),
    _recipe(
        "margin_expansion",
        "营业利润率改善",
        "fin_diff(fin_ratio(operating_income, revenue), period_id, periods)",
        ("operating_income", "revenue", "period_id", "periods"),
    ),
    _recipe(
        "leverage_change",
        "杠杆变化",
        "fin_diff(fin_ratio(total_debt, total_assets), period_id, periods)",
        ("total_debt", "total_assets", "period_id", "periods"),
    ),
    _recipe(
        "capex_growth",
        "资本开支增长",
        "fin_growth(abs(capex), period_id, periods)",
        ("capex", "period_id", "periods"),
    ),
    _recipe(
        "asset_turnover",
        "资产周转率（单季度流量输入）",
        "fin_turnover(fin_ttm_quarterly(revenue, period_id, periods_per_year), total_assets, period_id, 2)",
        ("revenue", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "inventory_turnover",
        "存货周转率（单季度流量输入）",
        "fin_turnover(fin_ttm_quarterly(cost_of_goods_sold, period_id, periods_per_year), inventory, period_id, 2)",
        ("cost_of_goods_sold", "inventory", "period_id", "periods_per_year"),
    ),
    _recipe(
        "receivable_turnover",
        "应收周转率（单季度流量输入）",
        "fin_turnover(fin_ttm_quarterly(revenue, period_id, periods_per_year), receivables, period_id, 2)",
        ("revenue", "receivables", "period_id", "periods_per_year"),
    ),
    _recipe(
        "revenue_receivables_divergence",
        "收入与应收增长背离",
        "fin_divergence(revenue, receivables, period_id, periods)",
        ("revenue", "receivables", "period_id", "periods"),
    ),
    _recipe(
        "earnings_cashflow_divergence",
        "利润与现金流增长背离",
        "fin_divergence(net_income, operating_cash_flow, period_id, periods)",
        ("net_income", "operating_cash_flow", "period_id", "periods"),
    ),
    _recipe(
        "quality_growth_combo",
        "增长×现金质量组合（单季度流量输入）",
        "fin_yoy(revenue, period_id, periods_per_year) * fin_cash_conversion(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_ttm_quarterly(net_income, period_id, periods_per_year))",
        (
            "revenue",
            "operating_cash_flow",
            "net_income",
            "period_id",
            "periods_per_year",
        ),
    ),
    _recipe(
        "size_neutralize",
        "市值中性化残差",
        "cs_multi_resid(factor, market_cap)",
        ("factor", "market_cap"),
        category="cross_sectional",
        status="pending",
        replacement_for=("size_neutralize",),
    ),
)

for _item in _RECIPES:
    FactorRecipeRegistry.register(_item)
