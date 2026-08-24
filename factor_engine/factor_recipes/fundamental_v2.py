# -*- coding: utf-8 -*-
"""Fundamental v2 recipes (2026-08 operator expansion, class C).

All ``status="production"`` recipes reference only certified leaf operators
(``fin_*`` family plus daily primitives).  Recipes that depend on the newly
added atomic/relation operators live in ``shareholder_relative.py`` as
experimental until those operators gain three-backend evidence.
"""
from __future__ import annotations

from factor_engine.factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _recipe(
    name,
    description,
    expression,
    parameters,
    *,
    category="fundamental_v2",
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
    # ---- valuation / capital ------------------------------------------------
    _recipe("sales_yield", "市销率收益率 1/PS", "fin_ratio(1, ps_ratio)", ("ps_ratio",)),
    _recipe("cashflow_yield_1", "市现率收益率 1/PCF", "fin_ratio(1, pcf_ratio)", ("pcf_ratio",)),
    _recipe("cashflow_yield_2", "第二口径市现率收益率 1/PCF2", "fin_ratio(1, pcf_ratio_2)", ("pcf_ratio_2",)),
    _recipe(
        "pe_ttm_lyr_spread",
        "PE_TTM 与 PE_LYR 收益率差",
        "fin_ratio(1, pe_ratio_ttm) - fin_ratio(1, pe_ratio_lyr)",
        ("pe_ratio_ttm", "pe_ratio_lyr"),
    ),
    _recipe(
        "pcf_definition_spread",
        "两种 PCF 收益率差",
        "fin_ratio(1, pcf_ratio) - fin_ratio(1, pcf_ratio_2)",
        ("pcf_ratio", "pcf_ratio_2"),
    ),
    _recipe("free_float_cap_ratio", "自由流通市值占比", "fin_ratio(free_market_cap, market_cap)", ("free_market_cap", "market_cap")),
    _recipe("circulating_cap_ratio", "流通市值占比", "fin_ratio(circulating_market_cap, market_cap)", ("circulating_market_cap", "market_cap")),
    _recipe("free_market_cap_ratio", "自由流通/流通市值", "fin_ratio(free_market_cap, circulating_market_cap)", ("free_market_cap", "circulating_market_cap")),
    _recipe("amount_to_market_cap", "成交额/总市值", "fin_ratio(amount, market_cap)", ("amount", "market_cap")),
    _recipe("amount_to_free_market_cap", "成交额/自由流通市值", "fin_ratio(amount, free_market_cap)", ("amount", "free_market_cap")),
    _recipe("capital_change_ratio", "总股本环比变化", "fin_diff(total_capital, period_id, 1)", ("total_capital", "period_id")),
    _recipe("circulating_cap_change_ratio", "流通股本环比变化", "fin_diff(circulating_capital, period_id, 1)", ("circulating_capital", "period_id")),

    # ---- liquidity / leverage ----------------------------------------------
    _recipe("cash_ratio", "现金比率", "fin_ratio(cash_equivalents, current_liabilities)", ("cash_equivalents", "current_liabilities")),
    _recipe("working_capital", "营运资本", "current_assets - current_liabilities", ("current_assets", "current_liabilities")),
    _recipe("working_capital_to_assets", "营运资本/总资产", "fin_ratio(current_assets - current_liabilities, total_assets)", ("current_assets", "current_liabilities", "total_assets")),
    _recipe("debt_to_assets", "资产负债率", "fin_ratio(total_liabilities, total_assets)", ("total_liabilities", "total_assets")),
    _recipe("equity_to_assets", "权益资产比", "fin_ratio(total_owner_equities, total_assets)", ("total_owner_equities", "total_assets")),
    _recipe("net_debt", "净债务", "shortterm_loan + longterm_loan + bonds_payable + lease_liability - cash_equivalents", ("shortterm_loan", "longterm_loan", "bonds_payable", "lease_liability", "cash_equivalents")),
    _recipe("net_debt_to_assets", "净债务/总资产", "fin_ratio(shortterm_loan + longterm_loan + bonds_payable + lease_liability - cash_equivalents, total_assets)", ("shortterm_loan", "longterm_loan", "bonds_payable", "lease_liability", "cash_equivalents", "total_assets")),

    # ---- asset structure ---------------------------------------------------
    _recipe("cash_to_assets", "货币资金/总资产", "fin_ratio(cash_equivalents, total_assets)", ("cash_equivalents", "total_assets")),
    _recipe("receivable_to_assets", "应收/总资产", "fin_ratio(bill_receivable + account_receivable + receivable_fin, total_assets)", ("bill_receivable", "account_receivable", "receivable_fin", "total_assets")),
    _recipe("inventory_to_assets", "存货/总资产", "fin_ratio(inventory, total_assets)", ("inventory", "total_assets")),
    _recipe("fixed_assets_to_assets", "固定资产/总资产", "fin_ratio(fixed_assets, total_assets)", ("fixed_assets", "total_assets")),
    _recipe("intangible_to_assets", "无形资产/总资产", "fin_ratio(intangible_assets, total_assets)", ("intangible_assets", "total_assets")),
    _recipe("goodwill_to_assets", "商誉/总资产", "fin_ratio(goodwill, total_assets)", ("goodwill", "total_assets")),
    _recipe("goodwill_to_equity", "商誉/权益", "fin_ratio(goodwill, total_owner_equities)", ("goodwill", "total_owner_equities")),
    _recipe("contract_assets_to_assets", "合同资产/总资产", "fin_ratio(contract_assets, total_assets)", ("contract_assets", "total_assets")),
    _recipe("right_of_use_assets_to_assets", "使用权资产/总资产", "fin_ratio(usufruct_assets, total_assets)", ("usufruct_assets", "total_assets")),

    # ---- profitability -----------------------------------------------------
    _recipe(
        "parent_net_margin",
        "归母净利率",
        "fin_ratio(fin_ttm_quarterly(np_parent_company_owners, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("np_parent_company_owners", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "equity_turnover",
        "权益周转率",
        "fin_turnover(fin_ttm_quarterly(operating_revenue, period_id, periods_per_year), total_owner_equities, period_id, 2)",
        ("operating_revenue", "total_owner_equities", "period_id", "periods_per_year"),
    ),
    _recipe("parent_profit_share", "归母利润占比", "fin_ratio(np_parent_company_owners, net_profit)", ("np_parent_company_owners", "net_profit")),
    _recipe("adjusted_profit_ratio", "扣非利润/归母利润", "fin_ratio(adjusted_profit, np_parent_company_owners)", ("adjusted_profit", "np_parent_company_owners")),

    # ---- expense / R&D -----------------------------------------------------
    _recipe(
        "selling_expense_ratio",
        "销售费用率",
        "fin_ratio(fin_ttm_quarterly(selling_expense, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("selling_expense", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "administration_expense_ratio",
        "管理费用率",
        "fin_ratio(fin_ttm_quarterly(administration_expense, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("administration_expense", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "financial_expense_ratio",
        "财务费用率",
        "fin_ratio(fin_ttm_quarterly(financial_expense, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("financial_expense", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "rd_to_gross_profit",
        "研发/毛利",
        "fin_ratio(fin_ttm_quarterly(rd_expenses, period_id, periods_per_year), fin_ttm_quarterly(gross_profit, period_id, periods_per_year))",
        ("rd_expenses", "gross_profit", "period_id", "periods_per_year"),
    ),
    _recipe("rd_growth", "研发支出增长", "fin_yoy(rd_expenses, period_id, periods_per_year)", ("rd_expenses", "period_id", "periods_per_year")),
    _recipe(
        "rd_growth_acceleration",
        "研发支出增长加速",
        "fin_growth_acceleration(rd_expenses, period_id, short_periods, long_periods)",
        ("rd_expenses", "period_id", "short_periods", "long_periods"),
    ),

    # ---- cash flow quality ------------------------------------------------
    _recipe(
        "cfo_margin",
        "经营现金流利润率",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("operating_cash_flow", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cfo_to_net_profit",
        "经营现金流/净利润",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_ttm_quarterly(net_profit, period_id, periods_per_year))",
        ("operating_cash_flow", "net_profit", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cfo_to_parent_profit",
        "经营现金流/归母净利润",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_ttm_quarterly(np_parent_company_owners, period_id, periods_per_year))",
        ("operating_cash_flow", "np_parent_company_owners", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cfo_to_assets",
        "经营现金流/总资产",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("operating_cash_flow", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "capex_to_revenue",
        "资本开支/营收",
        "fin_ratio(abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("capex", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "free_cash_flow",
        "自由现金流",
        "fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm_quarterly(capex, period_id, periods_per_year))",
        ("operating_cash_flow", "capex", "period_id", "periods_per_year"),
    ),
    _recipe(
        "free_cash_flow_margin",
        "自由现金流利润率",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year))",
        ("operating_cash_flow", "capex", "operating_revenue", "period_id", "periods_per_year"),
    ),
    _recipe(
        "free_cash_flow_yield",
        "自由现金流收益率",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), market_cap)",
        ("operating_cash_flow", "capex", "market_cap", "period_id", "periods_per_year"),
    ),
    _recipe(
        "earnings_cash_gap",
        "利润现金流缺口（相对总资产）",
        "fin_ratio(fin_ttm_quarterly(np_parent_company_owners, period_id, periods_per_year) - fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("np_parent_company_owners", "operating_cash_flow", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "total_accruals_cashflow_method",
        "应计利润（现金流法，相对总资产）",
        "fin_accrual_ratio(fin_ttm_quarterly(np_parent_company_owners, period_id, periods_per_year), fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("np_parent_company_owners", "operating_cash_flow", "total_assets", "period_id", "periods_per_year"),
    ),

    # ---- turnover / efficiency --------------------------------------------
    _recipe(
        "payable_turnover",
        "应付周转率",
        "fin_turnover(fin_ttm_quarterly(operating_cost, period_id, periods_per_year), accounts_payable, period_id, 2)",
        ("operating_cost", "accounts_payable", "period_id", "periods_per_year"),
    ),
    _recipe(
        "days_sales_outstanding",
        "应收周转天数",
        "fin_ratio(365, receivable_turnover(revenue, receivables, period_id, periods_per_year))",
        ("revenue", "receivables", "period_id", "periods_per_year"),
    ),
    _recipe(
        "days_inventory_outstanding",
        "存货周转天数",
        "fin_ratio(365, inventory_turnover(cost_of_goods_sold, inventory, period_id, periods_per_year))",
        ("cost_of_goods_sold", "inventory", "period_id", "periods_per_year"),
    ),
    _recipe(
        "days_payables_outstanding",
        "应付周转天数",
        "fin_ratio(365, payable_turnover(cost_of_goods_sold, accounts_payable, period_id, periods_per_year))",
        ("cost_of_goods_sold", "accounts_payable", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cash_conversion_cycle",
        "现金转换周期",
        "fin_ratio(365, receivable_turnover(revenue, receivables, period_id, periods_per_year)) + fin_ratio(365, inventory_turnover(cost_of_goods_sold, inventory, period_id, periods_per_year)) - fin_ratio(365, payable_turnover(cost_of_goods_sold, accounts_payable, period_id, periods_per_year))",
        ("revenue", "receivables", "cost_of_goods_sold", "inventory", "accounts_payable", "period_id", "periods_per_year"),
    ),

    # ---- DuPont / invested capital ----------------------------------------
    _recipe(
        "equity_multiplier",
        "权益乘数",
        "fin_ratio(fin_average_balance(total_assets, period_id, 2), fin_average_balance(total_owner_equities, period_id, 2))",
        ("total_assets", "total_owner_equities", "period_id"),
    ),
    _recipe(
        "dupont_roe",
        "杜邦 ROE",
        "fin_ratio(fin_ttm_quarterly(np_parent_company_owners, period_id, periods_per_year), fin_ttm_quarterly(operating_revenue, period_id, periods_per_year)) * fin_turnover(fin_ttm_quarterly(operating_revenue, period_id, periods_per_year), total_assets, period_id, 2) * fin_ratio(fin_average_balance(total_assets, period_id, 2), fin_average_balance(total_owner_equities, period_id, 2))",
        ("np_parent_company_owners", "operating_revenue", "total_assets", "total_owner_equities", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cash_roa",
        "现金流 ROA",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_assets, period_id, 2))",
        ("operating_cash_flow", "total_assets", "period_id", "periods_per_year"),
    ),
    _recipe(
        "cash_roe",
        "现金流 ROE",
        "fin_ratio(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year), fin_average_balance(total_owner_equities, period_id, 2))",
        ("operating_cash_flow", "total_owner_equities", "period_id", "periods_per_year"),
    ),

    # ---- growth / stability -----------------------------------------------
    _recipe("gross_profit_growth", "毛利增长", "fin_yoy(gross_profit, period_id, periods_per_year)", ("gross_profit", "period_id", "periods_per_year")),
    _recipe("operating_profit_growth", "营业利润增长", "fin_yoy(operating_profit, period_id, periods_per_year)", ("operating_profit", "period_id", "periods_per_year")),
    _recipe("parent_profit_growth", "归母净利润增长", "fin_yoy(np_parent_company_owners, period_id, periods_per_year)", ("np_parent_company_owners", "period_id", "periods_per_year")),
    _recipe("cfo_growth", "经营现金流增长", "fin_yoy(operating_cash_flow, period_id, periods_per_year)", ("operating_cash_flow", "period_id", "periods_per_year")),
    _recipe(
        "fcf_growth",
        "自由现金流增长",
        "fin_yoy(fin_ttm_quarterly(operating_cash_flow, period_id, periods_per_year) - abs(fin_ttm_quarterly(capex, period_id, periods_per_year)), period_id, periods_per_year)",
        ("operating_cash_flow", "capex", "period_id", "periods_per_year"),
    ),
    _recipe(
        "revenue_growth_acceleration",
        "营收增长加速",
        "fin_growth_acceleration(operating_revenue, period_id, short_periods, long_periods)",
        ("operating_revenue", "period_id", "short_periods", "long_periods"),
    ),
    _recipe(
        "profit_growth_acceleration",
        "利润增长加速",
        "fin_growth_acceleration(net_profit, period_id, short_periods, long_periods)",
        ("net_profit", "period_id", "short_periods", "long_periods"),
    ),
    _recipe(
        "cfo_growth_acceleration",
        "现金流增长加速",
        "fin_growth_acceleration(operating_cash_flow, period_id, short_periods, long_periods)",
        ("operating_cash_flow", "period_id", "short_periods", "long_periods"),
    ),
    _recipe(
        "revenue_growth_stability",
        "营收增长稳定性",
        "fin_growth_stability(operating_revenue, period_id, growth_periods, window_periods)",
        ("operating_revenue", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "profit_growth_stability",
        "利润增长稳定性",
        "fin_growth_stability(net_profit, period_id, growth_periods, window_periods)",
        ("net_profit", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "margin_stability",
        "利润率稳定性",
        "fin_growth_volatility(fin_ratio(operating_profit, operating_revenue), period_id, growth_periods, window_periods)",
        ("operating_profit", "operating_revenue", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "roe_stability",
        "ROE 稳定性",
        "fin_growth_volatility(roe, period_id, growth_periods, window_periods)",
        ("roe", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "cfo_stability",
        "现金流稳定性",
        "fin_growth_volatility(operating_cash_flow, period_id, growth_periods, window_periods)",
        ("operating_cash_flow", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "profitability_persistence",
        "盈利持续性",
        "fin_growth_persistence(net_profit, period_id, growth_periods, window_periods)",
        ("net_profit", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "cashflow_persistence",
        "现金流持续性",
        "fin_growth_persistence(operating_cash_flow, period_id, growth_periods, window_periods)",
        ("operating_cash_flow", "period_id", "growth_periods", "window_periods"),
    ),
    _recipe(
        "margin_persistence",
        "利润率持续性",
        "fin_growth_persistence(fin_ratio(operating_profit, operating_revenue), period_id, growth_periods, window_periods)",
        ("operating_profit", "operating_revenue", "period_id", "growth_periods", "window_periods"),
    ),
)

for _item in _RECIPES:
    FactorRecipeRegistry.register(_item)
