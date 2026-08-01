#!/usr/bin/env python3
"""Fail-closed validation for all FactorEngine cold-start catalogs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from _bootstrap import REPO_ROOT
from factor_cold_start.catalog import load_all_catalogs, load_catalog
from factor_cold_start.generator import MARKET_FIELDS, SURFACE_DSL, existing_formula_hashes
from factor_cold_start.model import formula_dependencies

MIN_COUNTS = {
    ("ashare", "daily"): 900,
    ("us", "daily"): 1050,
    ("ashare", "extended"): 1400,
    ("us", "extended"): 1450,
    ("ashare", "research"): 250,
    ("us", "research"): 250,
}


def _allowed_fields(market: str, tier: str) -> set[str]:
    allowed = set(MARKET_FIELDS[market]["core"])
    if tier != "core":
        allowed.update(MARKET_FIELDS[market].get(tier, set()))
    return allowed


def _literal_number(call: ast.Call, position: int) -> float | None:
    if position >= len(call.args):
        return None
    node = call.args[position]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        if isinstance(node.operand.value, (int, float)):
            return -float(node.operand.value)
    return None


def _assert_causal(formula: str, factor_id: str) -> None:
    tree = ast.parse(formula, mode="eval")
    blocked_names = {"lead", "bfill", "backfill", "future", "shift_forward"}
    lag_positions = {
        "ts_delay": 1,
        "ts_delta": 1,
        "ts_pct": 1,
        "ts_log_return": 1,
        "ts_autocorr": 2,
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in blocked_names:
            raise ValueError(f"{factor_id}: blocked future-looking operator {name}")
        position = lag_positions.get(name)
        if position is not None:
            value = _literal_number(node, position)
            if value is not None and value < 0:
                raise ValueError(f"{factor_id}: negative lag/period in {name}")


def _surface_sets() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )

    targets = {
        "daily": set(DAILY_CANONICALS),
        "extended": set(EXTENDED_ONLY_CANONICALS),
        "research": set(RESEARCH_ONLY_CANONICALS),
    }
    allowed = {
        "daily": set(DAILY_CANONICALS),
        "extended": set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS),
        "research": set(DAILY_CANONICALS) | set(EXTENDED_ONLY_CANONICALS) | set(RESEARCH_ONLY_CANONICALS),
    }
    return targets, allowed


def validate_catalog(
    market: str,
    surface: str,
    *,
    manifest_fields: set[str],
    existing: set[str],
) -> list[str]:
    from api.dsl_parser import parse_expr

    targets, allowed_by_surface = _surface_sets()
    rows = load_catalog(market, surface)
    errors: list[str] = []
    minimum = MIN_COUNTS[(market, surface)]
    if len(rows) < minimum:
        errors.append(f"{market}/{surface}: expected at least {minimum}, got {len(rows)}")
    ids = [row.factor_id for row in rows]
    hashes = [row.formula_hash for row in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"{market}/{surface}: duplicate ids")
    if len(hashes) != len(set(hashes)):
        errors.append(f"{market}/{surface}: duplicate formulas")

    covered_ops: set[str] = set()
    for row in rows:
        try:
            parse_expr(row.formula, surface=SURFACE_DSL[surface])
        except Exception as exc:
            errors.append(f"{row.factor_id}: parser failed: {type(exc).__name__}: {exc}")
            continue
        ops, fields = formula_dependencies(row.formula)
        covered_ops.update(ops)
        if ops != row.operators:
            errors.append(f"{row.factor_id}: operator metadata mismatch")
        if fields != row.required_fields:
            errors.append(f"{row.factor_id}: field metadata mismatch")
        unknown_fields = set(fields) - manifest_fields
        if unknown_fields:
            errors.append(f"{row.factor_id}: fields absent from canonical manifest: {sorted(unknown_fields)}")
        unavailable = set(fields) - _allowed_fields(market, row.availability_tier)
        if unavailable:
            errors.append(f"{row.factor_id}: fields outside {market}/{row.availability_tier}: {sorted(unavailable)}")
        blocked_ops = set(ops) - allowed_by_surface[surface]
        if blocked_ops:
            errors.append(f"{row.factor_id}: operators outside {surface} authoring contract: {sorted(blocked_ops)}")
        if row.formula_hash in existing:
            errors.append(f"{row.factor_id}: exact duplicate of GTJA/Week2 formula")
        try:
            _assert_causal(row.formula, row.factor_id)
        except ValueError as exc:
            errors.append(str(exc))
        if market == "ashare" and any(field.startswith("ret__") or field == "adj_factor" for field in fields):
            errors.append(f"{row.factor_id}: US-only field in A-share catalog")
        if market == "us" and "factor" in fields:
            errors.append(f"{row.factor_id}: A-share adjustment factor in US catalog")

    missing_surface_ops = targets[surface] - covered_ops
    if missing_surface_ops:
        errors.append(
            f"{market}/{surface}: missing active {surface} operators: {sorted(missing_surface_ops)}"
        )
    return errors


def validate_all() -> list[str]:
    manifest = json.loads(
        (REPO_ROOT / "factor_engine" / "docs" / "field_manifest.json").read_text(encoding="utf-8")
    )
    manifest_fields = set(manifest["fields"])
    existing = existing_formula_hashes(REPO_ROOT)
    errors: list[str] = []
    for market in ("ashare", "us"):
        for surface in ("daily", "extended", "research"):
            errors.extend(
                validate_catalog(
                    market,
                    surface,
                    manifest_fields=manifest_fields,
                    existing=existing,
                )
            )

    targets, _ = _surface_sets()
    active = set().union(*targets.values())
    used = {op for row in load_all_catalogs() for op in row.operators}
    missing = active - used
    if missing:
        errors.append(f"global active-operator coverage incomplete: {sorted(missing)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="validate all six generated catalogs")
    parser.add_argument("--market", choices=("ashare", "us"))
    parser.add_argument("--surface", choices=("daily", "extended", "research"), default="daily")
    args = parser.parse_args()
    if args.all or args.market is None:
        errors = validate_all()
    else:
        manifest = json.loads(
            (REPO_ROOT / "factor_engine" / "docs" / "field_manifest.json").read_text(encoding="utf-8")
        )
        errors = validate_catalog(
            args.market,
            args.surface,
            manifest_fields=set(manifest["fields"]),
            existing=existing_formula_hashes(REPO_ROOT),
        )
    if errors:
        for error in errors[:150]:
            print(error)
        if len(errors) > 150:
            print(f"... {len(errors) - 150} more errors")
        return 1
    print("cold-start catalogs valid; all active daily/extended/research operators covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
