# -*- coding: utf-8 -*-
"""Fundamental ratio recipes composed from primitive operators.

These are factor recipes, not primitive canonicals.  Input fields must already
be point-in-time aligned by data_access.
"""
from __future__ import annotations

from typing import Any

from factor_engine.cleaned_operators.registry import OperatorRegistry


def _calculate(name: str, backend: str, *args: Any):
    operator = OperatorRegistry.get(name, backend=backend)
    if operator is None:
        raise RuntimeError(f"required primitive operator missing: {name}/{backend}")
    return operator.calculate(*args)


def operating_margin(operating_income, revenue, *, backend: str = "pandas_numpy"):
    return _calculate("safe_div_null", backend, operating_income, revenue)


def current_ratio(current_assets, current_liabilities, *, backend: str = "pandas_numpy"):
    return _calculate("safe_div_null", backend, current_assets, current_liabilities)


def quick_ratio(current_assets, inventory, current_liabilities, *, backend: str = "pandas_numpy"):
    numerator = _calculate("subtract", backend, current_assets, inventory)
    return _calculate("safe_div_null", backend, numerator, current_liabilities)


def debt_to_equity(total_debt, total_equity, *, backend: str = "pandas_numpy"):
    return _calculate("safe_div_null", backend, total_debt, total_equity)
