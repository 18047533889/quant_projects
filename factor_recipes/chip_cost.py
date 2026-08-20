# -*- coding: utf-8 -*-
"""Chip-cost recipes composed from the turnover-survival primitives (2026-08).

These are *factor recipes*, not canonical primitives: each is a 1-2 layer DSL
expression over ``ts_turnover_reference_price``.  The two standard
capital-gains-overhang conventions are kept as distinct names:

* ``capital_gains_overhang`` — ``(P_t - RP_t) / P_t``  (gain relative to price;
  Grinblatt-Han style aggregate unrealised gain).
* ``cost_basis_gap``         — ``(P_t - RP_t) / RP_t`` (gain relative to the
  average acquisition cost).

Do not merge the two: they have different denominators and different factor
econometrics.
"""
from __future__ import annotations

from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry

_RECIPES = (
    FactorRecipe(
        "capital_gains_overhang",
        "time_series_chip_cost",
        "资本利得悬置: (P - 换手存活加权平均成本) / P",
        "divide(subtract(price, ts_turnover_reference_price(price, turnover, window)), price)",
        ("price", "turnover", "window"),
    ),
    FactorRecipe(
        "cost_basis_gap",
        "time_series_chip_cost",
        "成本基差: (P - 换手存活加权平均成本) / 平均成本",
        "divide(subtract(price, ts_turnover_reference_price(price, turnover, window)), ts_turnover_reference_price(price, turnover, window))",
        ("price", "turnover", "window"),
    ),
)

for _recipe in _RECIPES:
    FactorRecipeRegistry.register(_recipe)
