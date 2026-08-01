#!/usr/bin/env python3
"""Fail-closed validation for production and candidate cold-start catalogs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from _bootstrap import REPO_ROOT
from factor_cold_start.catalog import load_candidate_catalog, load_catalog
from factor_cold_start.generator import MARKET_FIELDS, existing_formula_hashes
from factor_cold_start.model import formula_dependencies
from factor_cold_start.production_admission import admit_factor

CANDIDATE_MIN_COUNTS = {
    ("ashare", "daily"): 800,
    ("us", "daily"): 950,
    ("ashare", "extended"): 1300,
    ("us", "extended"): 1400,
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
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _assert_causal(formula: str, factor_id: str) -> None:
    tree = ast.parse(formula, mode="eval")
    blocked_names = {
        "lead",
        "Lead",
        "next",
        "bfill",
        "causal_bfill",
        "backfill",
        "fillna_interpolate",
        "interpolate",
        "future",
        "shift_forward",
        "shuffle",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in blocked_names:
            raise ValueError(f"{factor_id}: blocked future/noncausal operator {name}")
        position = {
            "ts_delay": 1,
            "ts_delta": 1,
            "ts_pct": 1,
            "ts_log_return": 1,
            "ts_autocorr": 2,
        }.get(name)
        if position is not None:
            value = _literal_number(node, position)
            if value is not None and value < 0:
                raise ValueError(f"{factor_id}: negative lag/period in {name}")


def _validate_rows(
    market: str,
    label: str,
    rows,
    *,
    manifest_fields: set[str],
    existing: set[str],
    require_production: bool,
) -> list[str]:
    from api.dsl_parser import parse_expr

    errors: list[str] = []
    ids = [row.factor_id for row in rows]
    hashes = [row.formula_hash for row in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"{market}/{label}: duplicate ids")
    if len(hashes) != len(set(hashes)):
        errors.append(f"{market}/{label}: duplicate formulas")

    for row in rows:
        try:
            parse_expr(row.formula, surface="compat")
        except Exception as exc:
            errors.append(
                f"{row.factor_id}: parser failed: {type(exc).__name__}: {exc}"
            )
            continue

        ops, fields = formula_dependencies(row.formula)
        if ops != row.operators:
            errors.append(f"{row.factor_id}: operator metadata mismatch")
        if fields != row.required_fields:
            errors.append(f"{row.factor_id}: field metadata mismatch")
        unknown_fields = set(fields) - manifest_fields
        if unknown_fields:
            errors.append(
                f"{row.factor_id}: fields absent from canonical manifest: "
                f"{sorted(unknown_fields)}"
            )
        unavailable = set(fields) - _allowed_fields(market, row.availability_tier)
        if unavailable:
            errors.append(
                f"{row.factor_id}: fields outside {market}/{row.availability_tier}: "
                f"{sorted(unavailable)}"
            )
        if row.formula_hash in existing:
            errors.append(f"{row.factor_id}: exact duplicate of GTJA/Week2 formula")
        try:
            _assert_causal(row.formula, row.factor_id)
        except ValueError as exc:
            errors.append(str(exc))
        if market == "ashare" and any(
            field.startswith("ret__") or field == "adj_factor" for field in fields
        ):
            errors.append(f"{row.factor_id}: US-only field in A-share catalog")
        if market == "us" and "factor" in fields:
            errors.append(f"{row.factor_id}: A-share adjustment factor in US catalog")

        if require_production:
            admission = admit_factor(row)
            if not admission.eligible:
                errors.append(
                    f"{row.factor_id}: production admission failed: "
                    + "; ".join(admission.violations)
                )
            elif not admission.certified_backends:
                errors.append(
                    f"{row.factor_id}: production admission has no certified backend rows"
                )
    return errors


def validate_catalog(
    market: str,
    surface: str,
    *,
    manifest_fields: set[str],
    existing: set[str],
) -> list[str]:
    errors: list[str] = []
    if surface == "daily":
        rows = load_catalog(market, "daily")
        if not rows:
            errors.append(f"{market}/daily: production catalog is empty")
        errors.extend(
            _validate_rows(
                market,
                "daily-production",
                rows,
                manifest_fields=manifest_fields,
                existing=existing,
                require_production=True,
            )
        )
        return errors

    rows = load_candidate_catalog(market, "extended")
    minimum = CANDIDATE_MIN_COUNTS[(market, "extended")]
    if len(rows) < minimum:
        errors.append(
            f"{market}/extended-candidates: expected at least {minimum}, got {len(rows)}"
        )
    errors.extend(
        _validate_rows(
            market,
            "extended-candidates",
            rows,
            manifest_fields=manifest_fields,
            existing=existing,
            require_production=False,
        )
    )
    return errors


def validate_candidate_scale() -> list[str]:
    errors: list[str] = []
    for key, minimum in CANDIDATE_MIN_COUNTS.items():
        rows = load_candidate_catalog(*key)
        if len(rows) < minimum:
            errors.append(f"{key}: expected at least {minimum}, got {len(rows)}")
    return errors


def validate_all() -> list[str]:
    manifest = json.loads(
        (REPO_ROOT / "factor_engine" / "docs" / "field_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_fields = set(manifest["fields"])
    existing = existing_formula_hashes(REPO_ROOT)
    errors = validate_candidate_scale()
    for market in ("ashare", "us"):
        for surface in ("daily", "extended"):
            errors.extend(
                validate_catalog(
                    market,
                    surface,
                    manifest_fields=manifest_fields,
                    existing=existing,
                )
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="validate all catalogs")
    parser.add_argument("--market", choices=("ashare", "us"))
    parser.add_argument("--surface", choices=("daily", "extended"), default="daily")
    args = parser.parse_args()

    if args.all or args.market is None:
        errors = validate_all()
    else:
        manifest = json.loads(
            (REPO_ROOT / "factor_engine" / "docs" / "field_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        errors = validate_catalog(
            args.market,
            args.surface,
            manifest_fields=set(manifest["fields"]),
            existing=existing_formula_hashes(REPO_ROOT),
        )
    if errors:
        for error in errors[:100]:
            print(error)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors")
        return 1
    print("cold-start catalogs valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
