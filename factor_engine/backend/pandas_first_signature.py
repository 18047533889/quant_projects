# -*- coding: utf-8 -*-
"""Production call validation for backend-independent Pandas-first operators.

The semantic production tier intentionally does not require DuckDB or Polars,
but it still requires bounded, deterministic call signatures.  This module
validates the literal/static parameters of the reviewed Pandas-first promotion
set after the logical plan has been canonicalized.
"""
from __future__ import annotations

import math
from typing import Any

from planner.logical_plan import PlanNode


_WINDOW_NAMES = ("window", "d", "n", "span", "period", "periods")


def _literal(child: Any) -> Any | None:
    if getattr(child, "op", None) != "literal":
        return None
    return (getattr(child, "attrs", None) or {}).get("value")


def _static_params(node: PlanNode, canonical: str) -> dict[str, Any]:
    """Collect literal positional and keyword parameters using registry metadata."""
    from cleaned_operators.registry import OperatorRegistry

    catalog = OperatorRegistry._catalog.get(canonical, {})
    names = list(catalog.get("param_names") or [])
    values: dict[str, Any] = {}
    for index, name in enumerate(names):
        if index >= len(node.inputs):
            break
        value = _literal(node.inputs[index])
        if value is not None:
            values[str(name)] = value
    values.update(dict(node.attrs or {}))
    return values


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not _finite(value):
        return None
    number = float(value)
    if int(number) != number:
        return None
    return int(number)


def _first(values: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def validate_pandas_first_call(canonical: str, node: PlanNode) -> tuple[bool, str]:
    """Validate one promoted call; dynamic panel inputs are validated by runtime DQ."""
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS

    if canonical not in PANDAS_FIRST_PRODUCTION_CANONICALS:
        return True, ""

    values = _static_params(node, canonical)
    window_raw = _first(values, _WINDOW_NAMES)
    window: int | None = None
    if window_raw is not None:
        window = _integer(window_raw)
        if window is None or window <= 0:
            return False, f"{canonical}: rolling/state window must be a positive integer"

    min_periods_raw = values.get("min_periods")
    if min_periods_raw is not None:
        min_periods = _integer(min_periods_raw)
        if min_periods is None or min_periods <= 0:
            return False, f"{canonical}: min_periods must be a positive integer"
        if window is not None and min_periods > window:
            return False, f"{canonical}: min_periods must be <= window"

    for name in ("q", "quantile", "lower", "upper"):
        if name in values:
            if not _finite(values[name]) or not 0.0 <= float(values[name]) <= 1.0:
                return False, f"{canonical}: {name} must be in [0, 1]"

    if "k" in values:
        k = _integer(values["k"])
        if k is None or k <= 0:
            return False, f"{canonical}: k must be a positive integer"
        if canonical == "ts_moment" and k > 8:
            return False, "ts_moment: production supports moment order k<=8"
        if canonical == "ts_topk_sum" and window is not None and k > window:
            return False, "ts_topk_sum: k must be <= window"

    if "ddof" in values:
        ddof = _integer(values["ddof"])
        if ddof not in {0, 1}:
            return False, f"{canonical}: ddof must be 0 or 1"

    if "decimals" in values:
        decimals = _integer(values["decimals"])
        if decimals is None or abs(decimals) > 12:
            return False, "round: production decimals must be an integer in [-12, 12]"

    if "to" in values and not _finite(values["to"]):
        return False, "scale: to must be finite"

    if "add_intercept" in values and not isinstance(values["add_intercept"], bool):
        return False, f"{canonical}: add_intercept must be boolean"

    # All other literal numeric tuning parameters must at least be finite. This
    # catches NaN/Inf configuration leaks without guessing a narrower semantic
    # domain that the operator contract has not declared.
    for name, value in values.items():
        if name in {"null_policy", "nan_policy", "zero_std_policy"}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not _finite(value):
            return False, f"{canonical}: {name} must be finite"

    return True, ""


def check_pandas_first_plan_signatures(plan: Any) -> list[str]:
    """Walk a canonical plan and return bounded-signature violations."""
    from cleaned_operators.production_tiers import PANDAS_FIRST_PRODUCTION_CANONICALS
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op:
            canonical = OperatorRegistry._aliases.get(op, op)
            if canonical in PANDAS_FIRST_PRODUCTION_CANONICALS:
                ok, message = validate_pandas_first_call(canonical, node)
                if not ok:
                    errors.append(message)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return errors
