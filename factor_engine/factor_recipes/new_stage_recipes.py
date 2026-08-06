# -*- coding: utf-8 -*-
"""Next-stage factor recipes (2026-08).

Fixed interval / time-structure / growth definitions composed from the new
stage primitive operators.  Recipes referencing experimental operators are
marked ``experimental`` so they stay out of the default production expansion.
"""
from __future__ import annotations

from factor_recipes.registry import FactorRecipe, FactorRecipeRegistry

_RECIPES: list[FactorRecipe] = []


def _recipe(name, description, expression, parameters, *, category="intraday_microstructure", status="experimental"):
    return FactorRecipe(
        name,
        category,
        description,
        expression,
        tuple(parameters),
        status=status,
        replacement_for=(),
    )


# ---------------------------------------------------------------------------
# § Fixed intraday time-window recipes (single generic kernel)
# ---------------------------------------------------------------------------

_INTRA = [
    ("intra_opening_15m_return", "开盘 15 分钟收益", 570, 585),
    ("intra_opening_30m_return", "开盘 30 分钟收益", 570, 600),
    ("intra_morning_return", "上午收益", 570, 690),
    ("intra_pre_lunch_30m_return", "午前 30 分钟收益", 660, 690),
    ("intra_afternoon_open_30m_return", "午后开盘 30 分钟收益", 780, 810),
    ("intra_closing_30m_return", "尾盘 30 分钟收益", 870, 900),
    ("intra_closing_15m_return", "尾盘 15 分钟收益", 885, 900),
]
for _name, _desc, _start, _end in _INTRA:
    _RECIPES.append(
        _recipe(_name, _desc, f"intra_interval_return(close, {_start}, {_end})", ("close",))
    )

_RECIPES.extend(
    [
        _recipe(
            "intra_morning_afternoon_return_spread",
            "上午收益 - 下午收益",
            "subtract(intra_morning_return(close), intra_interval_return(close, 780, 900))",
            ("close",),
        ),
        _recipe(
            "intra_morning_close_continuation",
            "上午方向 × 尾盘 30 分钟（延续）",
            "multiply(sign(intra_morning_return(close)), intra_closing_30m_return(close))",
            ("close",),
        ),
        _recipe(
            "intra_morning_close_reversal",
            "-上午方向 × 尾盘 30 分钟（反转）",
            "multiply(neg(sign(intra_morning_return(close))), intra_closing_30m_return(close))",
            ("close",),
        ),
        _recipe(
            "intra_open_close_pressure",
            "开盘 30 分钟收益 - 尾盘 30 分钟收益",
            "subtract(intra_opening_30m_return(close), intra_closing_30m_return(close))",
            ("close",),
        ),
        _recipe(
            "intra_front_back_volume_ratio",
            "前半日成交量 / 后半日成交量",
            "safe_div_null(intra_interval_volume_share(volume, 570, 690), intra_interval_volume_share(volume, 780, 900))",
            ("volume",),
        ),
    ]
)

# ---------------------------------------------------------------------------
# § Asset / account growth recipes (generic fin_growth kernel)
# ---------------------------------------------------------------------------

_FIN_GROWTH = [
    ("fin_total_asset_growth", "总资产增长率", "total_assets"),
    ("fin_operating_asset_growth", "经营性资产增长率", "operating_assets"),
    ("fin_fixed_asset_growth", "固定资产增长率", "fixed_assets"),
    ("fin_inventory_growth", "存货增长率", "inventories"),
    ("fin_receivable_growth", "应收账款增长率", "account_receivable"),
    ("fin_goodwill_growth", "商誉增长率", "goodwill"),
    ("fin_intangible_growth", "无形资产增长率", "intangible_assets"),
    ("fin_construction_in_progress_growth", "在建工程增长率", "construction_in_progress"),
]
for _name, _desc, _field in _FIN_GROWTH:
    _RECIPES.append(
        _recipe(_name, _desc, f"fin_growth({_field}, period_id, 1)", (_field, "period_id"),
                category="fundamental")
    )

for _recipe_obj in _RECIPES:
    FactorRecipeRegistry.register(_recipe_obj)
