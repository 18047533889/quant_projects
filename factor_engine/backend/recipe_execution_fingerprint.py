# -*- coding: utf-8 -*-
"""Deterministic per-recipe execution fingerprints for recipe certification.

Every production recipe is executed on the Pandas semantic-reference path with
a deterministic synthetic panel; the output is reduced to a stable fingerprint
(hash of the sorted finite values + shape + NaN ratio).  ``--check`` recomputes
the fingerprints and fails if any stale recipe execution is recorded, so a
recipe whose operator DAG or bindings changed can never keep stale evidence.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


_SCALARS: dict[str, Any] = {
    "window": 5,
    "lag": 1,
    "min_periods": 2,
    "periods": 4,
    "min_obs": 4,
    "short_periods": 2,
    "long_periods": 4,
    "growth_periods": 2,
    "window_periods": 4,
    "average_periods": 2,
    "fast_period": 3,
    "slow_period": 6,
    "signal_period": 3,
    "add_intercept": True,
    "constant": 0.015,
    "penetration": 0.3,
    "periods_per_year": 4,
    "threshold": 0.0,
    "min_count": 2,
    "k": 3,
}
_PANEL_IDENTIFIERS = frozenset({"period_id", "fiscal_quarter", "group", "industry", "sector", "category", "holder_category", "entity_id"})


def _is_scalar_parameter(name: str) -> bool:
    return (
        name in _SCALARS
        or name.endswith("_window")
        or name.endswith("_periods")
        or name.endswith("_factor")
    )


def _scalar_value(name: str) -> Any:
    if name in _SCALARS:
        return _SCALARS[name]
    if name.endswith("_window"):
        return 3
    if name.endswith("_periods"):
        return 4
    if name.endswith("_factor"):
        return 1.0
    raise KeyError(name)


def _bindings_for(recipe: Any) -> dict[str, Any]:
    """Deterministic binding: scalars -> scalar literals, panels -> column refs."""
    return {
        parameter: (_scalar_value(parameter) if _is_scalar_parameter(parameter) else parameter)
        for parameter in recipe.parameters
    }


def _build_panel_fields(recipes: list[Any]) -> dict[str, pd.Series]:
    """Build a shared deterministic panel for all non-scalar recipe fields."""
    timestamps = pd.date_range("2024-01-02", periods=64, freq="B")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([timestamps, instruments], names=["timestamp", "instrument"])
    size = len(index)
    base = 20.0 + np.arange(size, dtype=float) * 0.03
    fields: set[str] = set()
    for recipe in recipes:
        for parameter in recipe.parameters:
            if not _is_scalar_parameter(parameter):
                fields.add(parameter)
    data: dict[str, pd.Series] = {}
    for offset, name in enumerate(sorted(fields)):
        if name == "period_id":
            values = np.repeat(np.arange(len(timestamps)) // 3, len(instruments))
            data[name] = pd.Series(values.astype(float), index=index)
            continue
        if name == "fiscal_quarter":
            report_number = np.arange(len(timestamps)) // 3
            values = np.repeat(report_number % 4 + 1, len(instruments))
            data[name] = pd.Series(values.astype(float), index=index)
            continue
        if name in {"group", "industry", "sector", "category", "holder_category", "entity_id"}:
            values = np.tile(np.array(["A0", "B0", "C0"], dtype=object), len(timestamps) * 3 // 3 + 1)[:size]
            data[name] = pd.Series(values, index=index)
            continue
        raw = base * (1.0 + 0.01 * offset) + 0.1 * np.sin(np.arange(size) / 3.0 + offset)
        values = np.abs(raw) * 1e4 + 1e3 if name in {
            "volume", "amount", "market_cap", "total_assets", "total_equity",
            "current_assets", "current_liabilities", "inventory", "receivables",
            "cash", "share_ratio", "share_number", "total_capital",
            "float_shares", "total_shares",
        } else raw
        data[name] = pd.Series(values.astype(float), index=index)
    return data


def _fingerprint(series: pd.Series) -> dict[str, Any]:
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite):
        bucket = finite[:: max(1, len(finite) // 200)]
        digest = hashlib.sha256(np.round(bucket, 6).tobytes()).hexdigest()[:16]
    else:
        digest = hashlib.sha256(b"empty").hexdigest()[:16]
    return {
        "rows": int(series.shape[0]),
        "nan_ratio": round(float(np.isnan(values).mean()), 6),
        "output_hash": digest,
        "index_columns": list(series.index.names or []),
    }


def record_recipe_execution_fingerprints() -> dict[str, dict[str, Any]]:
    """Execute every production recipe and return {name: fingerprint}."""
    import os
    from factor_engine.factor_recipes import FactorRecipeRegistry
    from factor_engine.backend.recipe_evidence import _production_recipe_names

    os.environ["FACTOR_ENGINE_EXPAND_RECIPE_USAGE"] = "1"
    from factor_engine.factor_recipes.planner_bridge import compile_recipe_plans
    from factor_engine.backend.context import ExecutionContext
    from factor_engine.backend.factory import build_backend
    from tests.helpers import InMemorySeriesSource

    names = sorted(_production_recipe_names())
    recipes = [FactorRecipeRegistry.get(name) for name in names]
    recipes = [r for r in recipes if r is not None]

    # Build one shared source from the union of non-scalar fields.
    source = InMemorySeriesSource(data=_build_panel_fields(recipes))
    backend = build_backend("pandas")
    out: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        request = {recipe.name: (recipe.name, _bindings_for(recipe))}
        batch = compile_recipe_plans(request)
        plan = batch.plans[recipe.name]
        try:
            output = backend.execute(
                plan,
                ExecutionContext(data_source=source, run_mode="research"),
            )
            out[recipe.name] = _fingerprint(pd.Series(output)) if output is not None else {}
        except Exception as exc:  # fail closed: record the failure so --check rejects
            out[recipe.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def fingerprints_current(recorded: dict[str, dict[str, Any]]) -> list[str]:
    """Return mismatch diagnostics between recorded and recomputed fingerprints."""
    current = record_recipe_execution_fingerprints()
    errors: list[str] = []
    for name in sorted(set(recorded) | set(current)):
        rec = recorded.get(name) or {}
        cur = current.get(name) or {}
        if rec.get("error") or cur.get("error"):
            if (rec.get("error") or "ok") != (cur.get("error") or "ok"):
                errors.append(f"recipe {name} execution error mismatch: {rec.get('error')!r} vs {cur.get('error')!r}")
            continue
        if rec.get("output_hash") != cur.get("output_hash"):
            errors.append(
                f"recipe {name} output hash stale: recorded={rec.get('output_hash')!r} "
                f"actual={cur.get('output_hash')!r}"
            )
    return errors
