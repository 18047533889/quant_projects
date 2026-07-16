# -*- coding: utf-8
"""基本面比率算子的字段口径契约（research / experimental 门禁用）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PeriodType = Literal["quarterly", "cumulative", "ttm"]
StatementType = Literal["balance", "income", "cashflow", "mixed"]


@dataclass(frozen=True)
class FundamentalFieldContract:
    """单字段期望的财报口径。"""

    period_type: PeriodType
    statement_type: StatementType
    pit_required: bool = True


# canonical → {param_name: contract}
FUNDAMENTAL_RATIO_FIELD_CONTRACTS: dict[str, dict[str, FundamentalFieldContract]] = {
    "operating_margin": {
        "operating_income": FundamentalFieldContract("quarterly", "income"),
        "revenue": FundamentalFieldContract("quarterly", "income"),
    },
    "current_ratio": {
        "current_assets": FundamentalFieldContract("quarterly", "balance"),
        "current_liabilities": FundamentalFieldContract("quarterly", "balance"),
    },
    "quick_ratio": {
        "current_assets": FundamentalFieldContract("quarterly", "balance"),
        "inventory": FundamentalFieldContract("quarterly", "balance"),
        "current_liabilities": FundamentalFieldContract("quarterly", "balance"),
    },
    "debt_to_equity": {
        "total_debt": FundamentalFieldContract("quarterly", "balance"),
        "total_equity": FundamentalFieldContract("quarterly", "balance"),
    },
}


def check_fundamental_ratio_field_contracts() -> list[str]:
    """比率算子 param_names 须覆盖 field contract 声明。"""
    from cleaned_operators.registry import OperatorRegistry

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
