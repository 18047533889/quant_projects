# -*- coding: utf-8 -*-
"""Fundamental factor recipes built from PIT-aware period primitives."""
from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


_RECIPES = (
    FactorRecipe("current_ratio", "fundamental", "流动比率", "safe_div_null(current_assets, current_liabilities)", ("current_assets", "current_liabilities"), replacement_for=("current_ratio",)),
    FactorRecipe("quick_ratio", "fundamental", "速动比率", "safe_div_null(current_assets - inventory, current_liabilities)", ("current_assets", "inventory", "current_liabilities"), replacement_for=("quick_ratio",)),
    FactorRecipe("debt_to_equity", "fundamental", "负债权益比", "safe_div_null(total_debt, total_equity)", ("total_debt", "total_equity"), replacement_for=("debt_to_equity",)),
    FactorRecipe("operating_margin", "fundamental", "营业利润率", "safe_div_null(operating_income, revenue)", ("operating_income", "revenue"), replacement_for=("operating_margin",)),
    FactorRecipe("gross_profitability", "fundamental", "毛利润资产收益", "safe_div_null(ttm_from_quarterly(gross_profit, period_id), period_average(total_assets, period_id, 2))", ("gross_profit", "total_assets", "period_id"), status="pending"),
    FactorRecipe("operating_profitability", "fundamental", "营业利润资产收益", "safe_div_null(ttm_from_quarterly(operating_income, period_id), period_average(total_assets, period_id, 2))", ("operating_income", "total_assets", "period_id"), status="pending"),
    FactorRecipe("cash_profitability", "fundamental", "经营现金流资产收益", "safe_div_null(ttm_from_quarterly(operating_cash_flow, period_id), period_average(total_assets, period_id, 2))", ("operating_cash_flow", "total_assets", "period_id"), status="pending"),
    FactorRecipe("accruals", "fundamental", "总应计项", "safe_div_null(ttm_from_quarterly(net_income, period_id) - ttm_from_quarterly(operating_cash_flow, period_id), period_average(total_assets, period_id, 2))", ("net_income", "operating_cash_flow", "total_assets", "period_id"), status="pending"),
    FactorRecipe("asset_growth", "fundamental", "资产同比增长", "period_change(total_assets, period_id, 4, 'ratio')", ("total_assets", "period_id"), status="pending"),
    FactorRecipe("inventory_growth", "fundamental", "存货同比增长", "period_change(inventory, period_id, 4, 'ratio')", ("inventory", "period_id"), status="pending"),
    FactorRecipe("receivables_growth", "fundamental", "应收账款同比增长", "period_change(receivables, period_id, 4, 'ratio')", ("receivables", "period_id"), status="pending"),
    FactorRecipe("rd_intensity", "fundamental", "研发强度", "safe_div_null(ttm_from_quarterly(research_development, period_id), ttm_from_quarterly(revenue, period_id))", ("research_development", "revenue", "period_id"), status="pending"),
    FactorRecipe("capex_intensity", "fundamental", "资本开支强度", "safe_div_null(abs(ttm_from_quarterly(capex, period_id)), period_average(total_assets, period_id, 2))", ("capex", "total_assets", "period_id"), status="pending"),
    FactorRecipe("free_cash_flow_profitability", "fundamental", "自由现金流资产收益", "safe_div_null(ttm_from_quarterly(operating_cash_flow, period_id) - abs(ttm_from_quarterly(capex, period_id)), period_average(total_assets, period_id, 2))", ("operating_cash_flow", "capex", "total_assets", "period_id"), status="pending"),
    FactorRecipe("net_debt_ratio", "fundamental", "净债务资产比", "safe_div_null(short_debt + long_debt - cash, total_assets)", ("short_debt", "long_debt", "cash", "total_assets")),
    FactorRecipe("interest_coverage", "fundamental", "利息保障倍数", "safe_div_null(ttm_from_quarterly(operating_income, period_id), abs(ttm_from_quarterly(interest_expense, period_id)))", ("operating_income", "interest_expense", "period_id"), status="pending"),
    FactorRecipe("cash_conversion", "fundamental", "现金利润转化率", "safe_div_null(ttm_from_quarterly(operating_cash_flow, period_id), ttm_from_quarterly(net_income, period_id))", ("operating_cash_flow", "net_income", "period_id"), status="pending"),
    FactorRecipe("size_neutralize", "cross_sectional", "市值中性化残差", "cs_multi_resid(factor, market_cap)", ("factor", "market_cap"), status="pending", replacement_for=("size_neutralize",)),
)

for _recipe in _RECIPES:
    FactorRecipeRegistry.register(_recipe)
