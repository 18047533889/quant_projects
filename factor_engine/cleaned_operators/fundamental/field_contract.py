# -*- coding: utf-8
"""基本面比率算子的字段口径契约（research / experimental 门禁用）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PeriodType = Literal["quarterly", "cumulative", "ttm"]
StatementType = Literal["balance", "income", "cashflow", "mixed"]
SemanticType = Literal["flow", "stock", "rate"]


@dataclass(frozen=True)
class FundamentalFieldContract:
    """单字段期望的财报口径。

    ``semantic_type`` (round-3 item 30) declares what the field MEANS:
    ``flow`` (single-period income/cash-flow), ``stock`` (balance-sheet
    point-in-time) or ``rate`` (a Return/Rate/ratio).  Threshold-relative score
    operators require ``rate`` / ``flow`` fields — never a raw Price/Volume
    (``Price > 0`` / ``Volume > 0`` are almost always True).
    """

    period_type: PeriodType
    statement_type: StatementType
    pit_required: bool = True
    semantic_type: SemanticType = "flow"


# canonical → {param_name: contract}
FUNDAMENTAL_RATIO_FIELD_CONTRACTS: dict[str, dict[str, FundamentalFieldContract]] = {
    "operating_margin": {
        "operating_income": FundamentalFieldContract("quarterly", "income", semantic_type="flow"),
        "revenue": FundamentalFieldContract("quarterly", "income", semantic_type="flow"),
    },
    "current_ratio": {
        "current_assets": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
        "current_liabilities": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
    },
    "quick_ratio": {
        "current_assets": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
        "inventory": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
        "current_liabilities": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
    },
    "debt_to_equity": {
        "total_debt": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
        "total_equity": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
    },
    "piotroski_f_score": {
        "roa": FundamentalFieldContract("quarterly", "income", semantic_type="rate"),
        "ocf": FundamentalFieldContract("quarterly", "cashflow", semantic_type="rate"),
        "net_profit": FundamentalFieldContract("quarterly", "income", semantic_type="rate"),
        "leverage": FundamentalFieldContract("quarterly", "mixed", semantic_type="rate"),
        "current_ratio": FundamentalFieldContract("quarterly", "mixed", semantic_type="rate"),
        "total_capital": FundamentalFieldContract("quarterly", "balance", semantic_type="stock"),
        "gross_margin": FundamentalFieldContract("quarterly", "income", semantic_type="rate"),
        "asset_turnover": FundamentalFieldContract("quarterly", "mixed", semantic_type="rate"),
    },
}


def check_fundamental_ratio_field_contracts() -> list[str]:
    """比率算子 param_names 须覆盖 field contract 声明。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canon, fields in FUNDAMENTAL_RATIO_FIELD_CONTRACTS.items():
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None or OperatorRegistry.get(canon) is None:
            continue
        params = set(entry.get("param_names") or [])
        for field_name in fields:
            if field_name not in params:
                errors.append(
                    f"基本面比率 {canon!r} param_names 缺少字段契约 {field_name!r}"
                )
    return errors
