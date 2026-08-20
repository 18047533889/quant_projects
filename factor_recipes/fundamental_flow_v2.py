# -*- coding: utf-8 -*-
"""Fundamental and fund-flow recipe expansion (2026-08).

Completes the C-class recipe surface for factor mining:

* financial field ratios (production): ``gross_profit``, ``invested_capital``,
  ``roic``, ``total_accruals``, ``capex``
* shareholder / fund-flow structure (experimental until the relation data
  source is installed): category shares, pledge/freeze ratios, holder churn,
  float-vs-total gaps, industry ex-self mean

Shareholder recipes reference relation operators that are currently
``SOURCE_BLOCKED_CANONICALS``, so they are registered as ``experimental`` and
remain outside the production admission gate until their source dataset exists.
"""
from __future__ import annotations

from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _recipe(
    name,
    description,
    expression,
    parameters,
    *,
    category="fundamental",
    status="production",
):
    return FactorRecipe(
        name,
        category,
        description,
        expression,
        tuple(parameters),
        status=status,
    )


_RECIPES = (
    # ------------------------------------------------------------------
    # Financial field ratios (operator leaves are certified)
    # ------------------------------------------------------------------
    _recipe(
        "gross_profit",
        "毛利 = 营业收入 - 营业成本",
        "subtract(operating_revenue, operating_cost)",
        ("operating_revenue", "operating_cost"),
    ),
    _recipe(
        "capex",
        "资本开支（物理字段直通）",
        "fix_intan_other_asset_acquisition_cash",
        ("fix_intan_other_asset_acquisition_cash",),
        status="experimental",
    ),
    _recipe(
        "total_accruals",
        "总应计 = 归母净利润 - 经营现金流",
        "subtract(parent_profit, net_operate_cash_flow)",
        ("parent_profit", "net_operate_cash_flow"),
    ),
    _recipe(
        "invested_capital",
        "投入资本 = 股东权益 + 有息负债 - 货币资金",
        "subtract(add(total_equity, interest_bearing_debt), cash_equivalents)",
        ("total_equity", "interest_bearing_debt", "cash_equivalents"),
    ),
    _recipe(
        "roic",
        "投入资本回报率 ROIC（税前代理；税后口径请自行传入 NOPAT 字段）",
        "safe_div_null(operating_profit, fin_average_balance(subtract(add(total_equity, interest_bearing_debt), cash_equivalents), period_id, 2))",
        ("operating_profit", "total_equity", "interest_bearing_debt", "cash_equivalents", "period_id"),
    ),
    # ------------------------------------------------------------------
    # Shareholder / fund-flow structure (source-blocked → experimental)
    # ------------------------------------------------------------------
    _recipe(
        "top1_holder_ratio",
        "第一大股东持股比例",
        "relation_topk_sum(share_ratio, shareholder_rank, 1)",
        ("share_ratio", "shareholder_rank"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "institutional_holder_ratio",
        "机构投资者持股比例",
        "relation_category_share(share_ratio, holder_category, institution)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "fund_holder_ratio",
        "基金持股比例",
        "relation_category_share(share_ratio, holder_category, fund)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "insurance_holder_ratio",
        "保险持股比例",
        "relation_category_share(share_ratio, holder_category, insurance)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "qfii_holder_ratio",
        "QFII 持股比例",
        "relation_category_share(share_ratio, holder_category, qfii)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "state_owned_holder_ratio",
        "国有法人持股比例",
        "relation_category_share(share_ratio, holder_category, state_owned)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "natural_person_holder_ratio",
        "自然人持股比例",
        "relation_category_share(share_ratio, holder_category, natural_person)",
        ("share_ratio", "holder_category"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "pledged_share_ratio",
        "质押股份占总股本比例",
        "safe_div_null(share_pledge, total_capital)",
        ("share_pledge", "total_capital"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "frozen_share_ratio",
        "冻结股份占总股本比例",
        "safe_div_null(share_freeze, total_capital)",
        ("share_freeze", "total_capital"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "pledged_holder_ratio",
        "质押股份占股东持股比例",
        "safe_div_null(share_pledge, share_number)",
        ("share_pledge", "share_number"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "frozen_holder_ratio",
        "冻结股份占股东持股比例",
        "safe_div_null(share_freeze, share_number)",
        ("share_freeze", "share_number"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "holder_overlap_ratio",
        "股东集合重叠比例（Jaccard）",
        "relation_overlap_ratio(current_ids, previous_ids, jaccard)",
        ("current_ids", "previous_ids"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "holder_entry_count",
        "新进股东数量",
        "relation_entry_count(current_ids, previous_ids)",
        ("current_ids", "previous_ids"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "holder_exit_count",
        "退出股东数量",
        "relation_exit_count(current_ids, previous_ids)",
        ("current_ids", "previous_ids"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "holder_churn_rate",
        "股东换手率",
        "safe_div_null(add(relation_entry_count(current_ids, previous_ids), relation_exit_count(current_ids, previous_ids)), add(relation_distinct_count(current_ids), relation_distinct_count(previous_ids)))",
        ("current_ids", "previous_ids"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "float_total_top10_gap",
        "流通前十大 - 总股本前十大持股比例",
        "subtract(top10_float_ratio, top10_total_ratio)",
        ("top10_float_ratio", "top10_total_ratio"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "float_total_hhi_gap",
        "流通 HHI - 总股本 HHI",
        "subtract(float_holder_hhi, total_holder_hhi)",
        ("float_holder_hhi", "total_holder_hhi"),
        category="shareholder",
        status="experimental",
    ),
    _recipe(
        "industry_ex_self_mean",
        "行业同伴均值（剔除自身）",
        "group_ex_self_mean(stock_value, industry)",
        ("stock_value", "industry"),
        category="index_industry",
        status="experimental",
    ),
)


def _register_all() -> None:
    for item in _RECIPES:
        FactorRecipeRegistry.register(item)


_register_all()
