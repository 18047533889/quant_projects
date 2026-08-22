# -*- coding: utf-8 -*-
"""Shareholder, index and industry-relative recipes (2026-08 expansion).

These recipes depend on the newly added atomic/relation operators
(``relation_hhi``, ``group_ex_self_mean``, ``ts_beta_if`` …) and are therefore
``experimental`` until those operators gain three-backend evidence.  They are
available to the research mining tier but are skipped by the production recipe
admission gate.
"""
from __future__ import annotations

from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _recipe(
    name,
    description,
    expression,
    parameters,
    *,
    category="shareholder_relative",
    status="experimental",
):
    return FactorRecipe(name, category, description, expression, tuple(parameters), status=status)


_RECIPES = (
    # ---- shareholder concentration (variadic ranked panels) ----------------
    _recipe(
        "holder_hhi",
        "前十大股东持股集中度 HHI",
        "relation_hhi(top1_share_ratio, top2_share_ratio, top3_share_ratio, top4_share_ratio, top5_share_ratio, top6_share_ratio, top7_share_ratio, top8_share_ratio, top9_share_ratio, top10_share_ratio)",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio", "top4_share_ratio", "top5_share_ratio", "top6_share_ratio", "top7_share_ratio", "top8_share_ratio", "top9_share_ratio", "top10_share_ratio"),
    ),
    _recipe(
        "top3_holder_ratio",
        "前三大股东持股比例",
        "relation_topk_sum(top1_share_ratio, top2_share_ratio, top3_share_ratio)",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio"),
    ),
    _recipe(
        "top5_holder_ratio",
        "前五大股东持股比例",
        "relation_topk_sum(top1_share_ratio, top2_share_ratio, top3_share_ratio, top4_share_ratio, top5_share_ratio)",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio", "top4_share_ratio", "top5_share_ratio"),
    ),
    _recipe(
        "top10_holder_ratio",
        "前十大股东持股比例",
        "relation_topk_sum(top1_share_ratio, top2_share_ratio, top3_share_ratio, top4_share_ratio, top5_share_ratio, top6_share_ratio, top7_share_ratio, top8_share_ratio, top9_share_ratio, top10_share_ratio)",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio", "top4_share_ratio", "top5_share_ratio", "top6_share_ratio", "top7_share_ratio", "top8_share_ratio", "top9_share_ratio", "top10_share_ratio"),
    ),
    _recipe(
        "holder_entropy",
        "前十大股东持股熵",
        "relation_entropy(top1_share_ratio, top2_share_ratio, top3_share_ratio, top4_share_ratio, top5_share_ratio, top6_share_ratio, top7_share_ratio, top8_share_ratio, top9_share_ratio, top10_share_ratio)",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio", "top4_share_ratio", "top5_share_ratio", "top6_share_ratio", "top7_share_ratio", "top8_share_ratio", "top9_share_ratio", "top10_share_ratio"),
    ),
    _recipe(
        "effective_holder_count",
        "有效股东数（HHI 倒数）",
        "fin_ratio(1, relation_hhi(top1_share_ratio, top2_share_ratio, top3_share_ratio, top4_share_ratio, top5_share_ratio, top6_share_ratio, top7_share_ratio, top8_share_ratio, top9_share_ratio, top10_share_ratio))",
        ("top1_share_ratio", "top2_share_ratio", "top3_share_ratio", "top4_share_ratio", "top5_share_ratio", "top6_share_ratio", "top7_share_ratio", "top8_share_ratio", "top9_share_ratio", "top10_share_ratio"),
    ),
    _recipe(
        "index_weight_hhi",
        "指数成分权重集中度",
        "relation_hhi(index_weight)",
        ("index_weight",),
    ),
    _recipe(
        "index_entry_return",
        "指数纳入后收益",
        "event_cumulative_return_past(ret, index_entry_exit_event(index_member), window)",
        ("ret", "index_member", "window"),
    ),
    _recipe(
        "index_membership_age_recipe",
        "指数纳入年龄",
        "index_membership_age(index_member, max_lookback)",
        ("index_member", "max_lookback"),
    ),

    # ---- index / industry relative ----------------------------------------
    _recipe("stock_excess_return_to_index", "个股对指数超额收益", "subtract(ret, index_return)", ("ret", "index_return")),
    _recipe(
        "industry_relative_return",
        "行业相对收益（剔除自身）",
        "subtract(ret, group_ex_self_mean(ret, industry_code))",
        ("ret", "industry_code"),
    ),
    _recipe(
        "industry_relative_roe",
        "行业相对 ROE（剔除自身）",
        "subtract(roe, group_ex_self_mean(roe, industry_code))",
        ("roe", "industry_code"),
    ),
    _recipe(
        "industry_relative_valuation",
        "行业相对估值（剔除自身）",
        "subtract(earnings_yield, group_ex_self_mean(earnings_yield, industry_code))",
        ("earnings_yield", "industry_code"),
    ),
    _recipe(
        "index_downside_beta",
        "指数下行 beta",
        "ts_beta_if(stock_return, index_return, lt(index_return, 0), window, min_periods)",
        ("stock_return", "index_return", "window", "min_periods"),
    ),
    _recipe(
        "index_residual_momentum",
        "指数残差动量",
        "ts_regression_resid(stock_return, index_return, window, min_periods)",
        ("stock_return", "index_return", "window", "min_periods"),
    ),
    _recipe(
        "robust_pe_zscore",
        "估值稳健 z-score",
        "ts_robust_zscore(pe_ratio, window, center, scale, clip)",
        ("pe_ratio", "window", "center", "scale", "clip"),
    ),
    _recipe(
        "robust_turnover_zscore",
        "换手稳健 z-score",
        "ts_robust_zscore(turnover_ratio, window, center, scale, clip)",
        ("turnover_ratio", "window", "center", "scale", "clip"),
    ),
)

for _item in _RECIPES:
    FactorRecipeRegistry.register(_item)
