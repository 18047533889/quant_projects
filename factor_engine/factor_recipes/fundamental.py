# -*- coding: utf-8 -*-
"""Fundamental domain recipes built from generic PIT-aware ``fin_*`` primitives.

Inputs named as flows are assumed single-report-period values unless the caller
first converts a cumulative filing field.  Domain ratios are recipes; the
primitive layer stays field-agnostic.
"""
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry

_RECIPES=(
    FactorRecipe("current_ratio","fundamental","流动比率","fin_ratio(current_assets, current_liabilities)",( "current_assets","current_liabilities"),replacement_for=("current_ratio",)),
    FactorRecipe("quick_ratio","fundamental","速动比率","fin_ratio(current_assets - inventory, current_liabilities)",( "current_assets","inventory","current_liabilities"),replacement_for=("quick_ratio",)),
    FactorRecipe("debt_to_equity","fundamental","负债权益比","fin_ratio(total_debt, total_equity)",( "total_debt","total_equity"),replacement_for=("debt_to_equity",)),
    FactorRecipe("operating_margin","fundamental","营业利润率","fin_ratio(operating_income, revenue)",( "operating_income","revenue"),replacement_for=("operating_margin",)),
    FactorRecipe("gross_margin","fundamental","毛利率","fin_ratio(gross_profit, revenue)",( "gross_profit","revenue")),
    FactorRecipe("net_margin","fundamental","净利率","fin_ratio(net_income, revenue)",( "net_income","revenue")),
    FactorRecipe("roa","fundamental","资产收益率","fin_ratio(fin_ttm(net_income, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",("net_income","total_assets","period_id","periods_per_year")),
    FactorRecipe("roe","fundamental","净资产收益率","fin_ratio(fin_ttm(net_income, period_id, periods_per_year), fin_average_balance(total_equity, period_id, 2))",("net_income","total_equity","period_id","periods_per_year")),
    FactorRecipe("roic_proxy","fundamental","投入资本回报率代理","fin_ratio(fin_ttm(nopat, period_id, periods_per_year), fin_average_balance(invested_capital, period_id, 2))",("nopat","invested_capital","period_id","periods_per_year")),
    FactorRecipe("gross_profitability","fundamental","毛利润资产收益","fin_ratio(fin_ttm(gross_profit, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",("gross_profit","total_assets","period_id","periods_per_year")),
    FactorRecipe("operating_profitability","fundamental","营业利润资产收益","fin_ratio(fin_ttm(operating_income, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",("operating_income","total_assets","period_id","periods_per_year")),
    FactorRecipe("cash_profitability","fundamental","经营现金流资产收益","fin_ratio(fin_ttm(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",("operating_cash_flow","total_assets","period_id","periods_per_year")),
    FactorRecipe("accruals","fundamental","应计质量","fin_accrual_ratio(fin_ttm(net_income, period_id, periods_per_year), fin_ttm(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",("net_income","operating_cash_flow","total_assets","period_id","periods_per_year")),
    FactorRecipe("asset_growth","fundamental","资产增长","fin_yoy(total_assets, period_id, periods_per_year)",( "total_assets","period_id","periods_per_year")),
    FactorRecipe("inventory_growth","fundamental","存货增长","fin_yoy(inventory, period_id, periods_per_year)",( "inventory","period_id","periods_per_year")),
    FactorRecipe("receivables_growth","fundamental","应收增长","fin_yoy(receivables, period_id, periods_per_year)",( "receivables","period_id","periods_per_year")),
    FactorRecipe("revenue_growth","fundamental","收入增长","fin_yoy(revenue, period_id, periods_per_year)",( "revenue","period_id","periods_per_year")),
    FactorRecipe("earnings_growth","fundamental","利润增长","fin_yoy(net_income, period_id, periods_per_year)",( "net_income","period_id","periods_per_year")),
    FactorRecipe("cashflow_growth","fundamental","经营现金流增长","fin_yoy(operating_cash_flow, period_id, periods_per_year)",( "operating_cash_flow","period_id","periods_per_year")),
    FactorRecipe("rd_intensity","fundamental","研发强度","fin_ratio(fin_ttm(research_development, period_id, periods_per_year), fin_ttm(revenue, period_id, periods_per_year))",("research_development","revenue","period_id","periods_per_year")),
    FactorRecipe("capex_intensity","fundamental","资本开支强度","fin_ratio(abs(fin_ttm(capex, period_id, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",("capex","total_assets","period_id","periods_per_year")),
    FactorRecipe("free_cash_flow_profitability","fundamental","自由现金流资产收益","fin_ratio(fin_ttm(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm(capex, period_id, periods_per_year)), fin_average_balance(total_assets, period_id, 2))",("operating_cash_flow","capex","total_assets","period_id","periods_per_year")),
    FactorRecipe("net_debt_ratio","fundamental","净债务资产比","fin_ratio(short_debt + long_debt - cash, total_assets)",( "short_debt","long_debt","cash","total_assets")),
    FactorRecipe("interest_coverage","fundamental","利息保障倍数","fin_ratio(fin_ttm(operating_income, period_id, periods_per_year), abs(fin_ttm(interest_expense, period_id, periods_per_year)))",("operating_income","interest_expense","period_id","periods_per_year")),
    FactorRecipe("cash_conversion","fundamental","现金利润转化率","fin_cash_conversion(fin_ttm(operating_cash_flow, period_id, periods_per_year), fin_ttm(net_income, period_id, periods_per_year))",("operating_cash_flow","net_income","period_id","periods_per_year")),
    FactorRecipe("margin_expansion","fundamental","营业利润率改善","fin_diff(fin_ratio(operating_income, revenue), period_id, periods)",( "operating_income","revenue","period_id","periods")),
    FactorRecipe("leverage_change","fundamental","杠杆变化","fin_diff(fin_ratio(total_debt, total_assets), period_id, periods)",( "total_debt","total_assets","period_id","periods")),
    FactorRecipe("capex_growth","fundamental","资本开支增长","fin_growth(abs(capex), period_id, periods)",( "capex","period_id","periods")),
    FactorRecipe("asset_turnover","fundamental","资产周转率","fin_turnover(fin_ttm(revenue, period_id, periods_per_year), total_assets, period_id, 2)",( "revenue","total_assets","period_id","periods_per_year")),
    FactorRecipe("inventory_turnover","fundamental","存货周转率","fin_turnover(fin_ttm(cost_of_goods_sold, period_id, periods_per_year), inventory, period_id, 2)",( "cost_of_goods_sold","inventory","period_id","periods_per_year")),
    FactorRecipe("receivable_turnover","fundamental","应收周转率","fin_turnover(fin_ttm(revenue, period_id, periods_per_year), receivables, period_id, 2)",( "revenue","receivables","period_id","periods_per_year")),
    FactorRecipe("revenue_receivables_divergence","fundamental","收入与应收增长背离","fin_divergence(revenue, receivables, period_id, periods)",( "revenue","receivables","period_id","periods")),
    FactorRecipe("earnings_cashflow_divergence","fundamental","利润与现金流增长背离","fin_divergence(net_income, operating_cash_flow, period_id, periods)",( "net_income","operating_cash_flow","period_id","periods")),
    FactorRecipe("quality_growth_combo","fundamental","增长×现金质量组合","fin_yoy(revenue, period_id, periods_per_year) * fin_cash_conversion(fin_ttm(operating_cash_flow, period_id, periods_per_year), fin_ttm(net_income, period_id, periods_per_year))",("revenue","operating_cash_flow","net_income","period_id","periods_per_year")),
    FactorRecipe("size_neutralize","cross_sectional","市值中性化残差","cs_multi_resid(factor, market_cap)",( "factor","market_cap"),status="pending",replacement_for=("size_neutralize",)),
)
for _recipe in _RECIPES:FactorRecipeRegistry.register(_recipe)
