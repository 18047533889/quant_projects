# -*- coding: utf-8 -*-
"""Reusable factor recipes composed from production primitive operators."""
from factor_engine.factor_recipes.registry import FactorRecipe, FactorRecipeRegistry
from factor_engine.factor_recipes.fundamental_ratios import (
    current_ratio,
    debt_to_equity,
    operating_margin,
    quick_ratio,
)

# Import declarative recipe catalogs after the registry is available.
from factor_engine.factor_recipes import fundamental as _fundamental  # noqa: F401,E402
from factor_engine.factor_recipes import fundamental_cumulative as _fundamental_cumulative  # noqa: F401,E402
from factor_engine.factor_recipes import technical as _technical  # noqa: F401,E402
from factor_engine.factor_recipes import candlestick_v2 as _candlestick_v2  # noqa: F401,E402
from factor_engine.factor_recipes import fundamental_v2 as _fundamental_v2  # noqa: F401,E402
from factor_engine.factor_recipes import fundamental_flow_v2 as _fundamental_flow_v2  # noqa: F401,E402
from factor_engine.factor_recipes import shareholder_relative as _shareholder_relative  # noqa: F401,E402
from factor_engine.factor_recipes import new_stage_recipes as _new_stage_recipes  # noqa: F401,E402
from factor_engine.factor_recipes import chip_cost as _chip_cost  # noqa: F401,E402
from factor_engine.factor_recipes import profitability_derived as _profitability_derived  # noqa: F401,E402

__all__ = [
    "FactorRecipe",
    "FactorRecipeRegistry",
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "operating_margin",
]
