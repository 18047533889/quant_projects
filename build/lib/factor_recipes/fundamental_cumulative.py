# -*- coding: utf-8 -*-
"""Fundamental recipes for fiscal-YTD cumulative filing fields."""
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _register(name, description, expression, parameters):
    FactorRecipeRegistry.register(
        FactorRecipe(
            name=name,
            category="fundamental",
            description=description,
            expression=expression,
            parameters=tuple(parameters),
            status="production",
        )
    )


_COMMON_PROFITABILITY = (
    "period_id",
    "fiscal_quarter",
    "periods_per_year",
)

_register(
    "roa_from_cumulative",
    "资产收益率（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(net_income, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
    ("net_income", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "roe_from_cumulative",
    "净资产收益率（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(net_income, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_equity, period_id, 2))",
    ("net_income", "total_equity", *_COMMON_PROFITABILITY),
)
_register(
    "gross_profitability_from_cumulative",
    "毛利润资产收益（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(gross_profit, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
    ("gross_profit", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "operating_profitability_from_cumulative",
    "营业利润资产收益（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(operating_income, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
    ("operating_income", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "cash_profitability_from_cumulative",
    "经营现金流资产收益（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(operating_cash_flow, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
    ("operating_cash_flow", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "accruals_from_cumulative",
    "应计质量（财政年度累计流量输入）",
    "fin_accrual_ratio(fin_ttm_cumulative(net_income, period_id, fiscal_quarter, periods_per_year), fin_ttm_cumulative(operating_cash_flow, period_id, fiscal_quarter, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
    (
        "net_income",
        "operating_cash_flow",
        "total_assets",
        *_COMMON_PROFITABILITY,
    ),
)
_register(
    "rd_intensity_from_cumulative",
    "研发强度（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(research_development, period_id, fiscal_quarter, periods_per_year), fin_ttm_cumulative(revenue, period_id, fiscal_quarter, periods_per_year))",
    ("research_development", "revenue", *_COMMON_PROFITABILITY),
)
_register(
    "capex_intensity_from_cumulative",
    "资本开支强度（财政年度累计流量输入）",
    "fin_ratio(abs(fin_ttm_cumulative(capex, period_id, fiscal_quarter, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",
    ("capex", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "free_cash_flow_profitability_from_cumulative",
    "自由现金流资产收益（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(operating_cash_flow, period_id, fiscal_quarter, periods_per_year) - abs(fin_ttm_cumulative(capex, period_id, fiscal_quarter, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",
    (
        "operating_cash_flow",
        "capex",
        "total_assets",
        *_COMMON_PROFITABILITY,
    ),
)
_register(
    "interest_coverage_from_cumulative",
    "利息保障倍数（财政年度累计流量输入）",
    "fin_ratio(fin_ttm_cumulative(operating_income, period_id, fiscal_quarter, periods_per_year), abs(fin_ttm_cumulative(interest_expense, period_id, fiscal_quarter, periods_per_year)))",
    ("operating_income", "interest_expense", *_COMMON_PROFITABILITY),
)
_register(
    "cash_conversion_from_cumulative",
    "现金利润转化率（财政年度累计流量输入）",
    "fin_cash_conversion(fin_ttm_cumulative(operating_cash_flow, period_id, fiscal_quarter, periods_per_year), fin_ttm_cumulative(net_income, period_id, fiscal_quarter, periods_per_year))",
    ("operating_cash_flow", "net_income", *_COMMON_PROFITABILITY),
)
_register(
    "asset_turnover_from_cumulative",
    "资产周转率（财政年度累计流量输入）",
    "fin_turnover(fin_ttm_cumulative(revenue, period_id, fiscal_quarter, periods_per_year), total_assets, period_id, 2)",
    ("revenue", "total_assets", *_COMMON_PROFITABILITY),
)
_register(
    "inventory_turnover_from_cumulative",
    "存货周转率（财政年度累计流量输入）",
    "fin_turnover(fin_ttm_cumulative(cost_of_goods_sold, period_id, fiscal_quarter, periods_per_year), inventory, period_id, 2)",
    ("cost_of_goods_sold", "inventory", *_COMMON_PROFITABILITY),
)
_register(
    "receivable_turnover_from_cumulative",
    "应收周转率（财政年度累计流量输入）",
    "fin_turnover(fin_ttm_cumulative(revenue, period_id, fiscal_quarter, periods_per_year), receivables, period_id, 2)",
    ("revenue", "receivables", *_COMMON_PROFITABILITY),
)
