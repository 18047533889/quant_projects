# -*- coding: utf-8 -*-
"""Static production-factor usage inventory and 90-day review policy."""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from cleaned_operators.registry import OperatorRegistry
from factor_recipes.registry import FactorRecipeRegistry

FUSED_COMPOSITES = frozenset({
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
})


@dataclass(frozen=True)
class UsageEntry:
    name: str
    kind: str
    call_count: int
    manifest_count: int
    last_seen: str | None
    examples: tuple[str, ...]


@dataclass(frozen=True)
class OperatorUsageReport:
    schema_version: str
    reference_timestamp: str
    lookback_days: int
    scanned_manifest_count: int
    operator_usage: Mapping[str, UsageEntry]
    recipe_usage: Mapping[str, UsageEntry]
    fused_composite_review: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_timestamp": self.reference_timestamp,
            "lookback_days": self.lookback_days,
            "scanned_manifest_count": self.scanned_manifest_count,
            "operator_usage": {name: asdict(value) for name, value in sorted(self.operator_usage.items())},
            "recipe_usage": {name: asdict(value) for name, value in sorted(self.recipe_usage.items())},
            "fused_composite_review": dict(sorted(self.fused_composite_review.items())),
        }


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def extract_call_names(formula: str) -> list[str]:
    """Extract direct DSL call names from a Python-like factor expression."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return []
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def _canonical_operator(name: str) -> str | None:
    canonical = OperatorRegistry._aliases.get(name, name)
    return canonical if canonical in OperatorRegistry._operators else None


def build_usage_report(
    manifests: Iterable[tuple[Path, Mapping[str, Any]]],
    *,
    as_of: datetime | None = None,
    lookback_days: int = 90,
) -> OperatorUsageReport:
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    records = list(manifests)
    timestamps = [
        timestamp
        for _, payload in records
        if (timestamp := _parse_timestamp(payload.get("born_timestamp") or payload.get("created_at"))) is not None
    ]
    reference = as_of.astimezone(timezone.utc) if as_of is not None else (
        max(timestamps) if timestamps else datetime(1970, 1, 1, tzinfo=timezone.utc)
    )
    operator_counts: dict[str, int] = {}
    operator_manifests: dict[str, set[str]] = {}
    operator_last: dict[str, datetime] = {}
    operator_examples: dict[str, list[str]] = {}
    recipe_counts: dict[str, int] = {}
    recipe_manifests: dict[str, set[str]] = {}
    recipe_last: dict[str, datetime] = {}
    recipe_examples: dict[str, list[str]] = {}

    for path, payload in records:
        formula = payload.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            continue
        timestamp = _parse_timestamp(payload.get("born_timestamp") or payload.get("created_at"))
        manifest_id = str(payload.get("candidate_id") or path.as_posix())
        for called in extract_call_names(formula):
            recipe = FactorRecipeRegistry.get(called)
            if recipe is not None:
                recipe_counts[called] = recipe_counts.get(called, 0) + 1
                recipe_manifests.setdefault(called, set()).add(manifest_id)
                if timestamp is not None:
                    recipe_last[called] = max(recipe_last.get(called, timestamp), timestamp)
                examples = recipe_examples.setdefault(called, [])
                if len(examples) < 3 and formula not in examples:
                    examples.append(formula)
                continue
            canonical = _canonical_operator(called)
            if canonical is None:
                continue
            operator_counts[canonical] = operator_counts.get(canonical, 0) + 1
            operator_manifests.setdefault(canonical, set()).add(manifest_id)
            if timestamp is not None:
                operator_last[canonical] = max(operator_last.get(canonical, timestamp), timestamp)
            examples = operator_examples.setdefault(canonical, [])
            if len(examples) < 3 and formula not in examples:
                examples.append(formula)

    operator_usage = {
        name: UsageEntry(
            name=name,
            kind="operator",
            call_count=count,
            manifest_count=len(operator_manifests.get(name, set())),
            last_seen=operator_last[name].isoformat() if name in operator_last else None,
            examples=tuple(operator_examples.get(name, [])),
        )
        for name, count in operator_counts.items()
    }
    recipe_usage = {
        name: UsageEntry(
            name=name,
            kind="recipe",
            call_count=count,
            manifest_count=len(recipe_manifests.get(name, set())),
            last_seen=recipe_last[name].isoformat() if name in recipe_last else None,
            examples=tuple(recipe_examples.get(name, [])),
        )
        for name, count in recipe_counts.items()
    }

    threshold = reference - timedelta(days=lookback_days)
    review: dict[str, str] = {}
    for canonical in sorted(FUSED_COMPOSITES):
        usage = operator_usage.get(canonical)
        if usage is None or usage.call_count == 0:
            review[canonical] = "review_unused"
        elif usage.last_seen is None:
            review[canonical] = "review_missing_timestamp"
        elif _parse_timestamp(usage.last_seen) < threshold:
            review[canonical] = "review_dormant"
        else:
            review[canonical] = "retain_active"

    return OperatorUsageReport(
        schema_version="operator_usage.v1",
        reference_timestamp=reference.isoformat(),
        lookback_days=lookback_days,
        scanned_manifest_count=len(records),
        operator_usage=operator_usage,
        recipe_usage=recipe_usage,
        fused_composite_review=review,
    )


def scan_manifest_files(root: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    records: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("expression_type") not in {None, "dsl"}:
            continue
        if isinstance(payload.get("formula"), str):
            records.append((path, payload))
    return records
