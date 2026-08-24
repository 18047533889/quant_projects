# -*- coding: utf-8 -*-
"""Profitability derived recipes using Phase 0 certified fields.

These recipes use Phase 0 fields (``gross_profit``, ``nopat``) that were added
to the catalog and certified for direct use, rather than computing them from
components like ``operating_revenue - operating_cost``.

All recipes have ``status="production"`` because they reference only certified
leaf operators and Phase 0 catalog fields.
"""
from __future__ import annotations

from factor_engine.factor_recipes.registry import FactorRecipe, FactorRecipeRegistry


def _recipe(
    name,
    description,
    expression,
    parameters,
    *,
    category="profitability_derived",
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
    # ---- Phase 0 field-based profitability ratios --------------------------
    _recipe(
        "gross_profit_derived",
        "毛利（直接引用 Phase 0 字段）",
        "gross_profit",
        ("gross_profit",),
    ),
    _recipe(
        "gross_margin_derived",
        "毛利率（基于 Phase 0 gross_profit）",
        "fin_ratio(gross_profit, operating_revenue)",
        ("gross_profit", "operating_revenue"),
    ),
    _recipe(
        "operating_margin_derived",
        "营业利润率（基于 Phase 0 operating_profit）",
        "fin_ratio(operating_profit, operating_revenue)",
        ("operating_profit", "operating_revenue"),
    ),
    _recipe(
        "net_margin_derived",
        "净利率（基于 Phase 0 net_profit）",
        "fin_ratio(net_profit, operating_revenue)",
        ("net_profit", "operating_revenue"),
    ),
    # ---- Industry relative --------------------------------------------------
    _recipe(
        "industry_peer_mean",
        "行业同行均值（通用跨截面分组均值）",
        "group_mean(metric, industry_code)",
        ("metric", "industry_code"),
    ),
)

for _item in _RECIPES:
    FactorRecipeRegistry.register(_item)
