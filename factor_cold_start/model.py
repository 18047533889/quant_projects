"""Immutable cold-start factor records and formula introspection helpers."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def formula_tree_key(formula: str) -> str:
    """Return a whitespace-insensitive structural key for exact DSL deduplication."""
    tree = ast.parse(str(formula).strip(), mode="eval")
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def formula_hash(formula: str) -> str:
    return hashlib.sha256(formula_tree_key(formula).encode("utf-8")).hexdigest()


def formula_dependencies(formula: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(operators, fields)`` from the Python-expression compatible DSL."""
    tree = ast.parse(str(formula).strip(), mode="eval")
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    fields = names - call_names - {"True", "False", "None"}
    return tuple(sorted(call_names)), tuple(sorted(fields))


@dataclass(frozen=True)
class ColdStartFactor:
    factor_id: str
    market: str
    surface: str
    formula: str
    family: str
    subfamily: str
    horizon: int | None
    complexity: str
    availability_tier: str
    rationale: str
    direction_hint: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    formula_hash: str = ""
    operators: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.market not in {"ashare", "us"}:
            raise ValueError(f"unsupported market: {self.market}")
        if self.surface not in {"daily", "extended", "research"}:
            raise ValueError(f"unsupported surface: {self.surface}")
        if self.complexity not in {"basic", "moderate", "composite"}:
            raise ValueError(f"unsupported complexity: {self.complexity}")
        formula = self.formula.strip()
        if not formula:
            raise ValueError("formula must be non-empty")
        ops, fields = formula_dependencies(formula)
        digest = formula_hash(formula)
        if self.formula_hash and self.formula_hash != digest:
            raise ValueError(f"formula hash mismatch: {self.factor_id}")
        if self.operators and tuple(sorted(self.operators)) != ops:
            raise ValueError(f"operator dependency mismatch: {self.factor_id}")
        if self.required_fields and tuple(sorted(self.required_fields)) != fields:
            raise ValueError(f"field dependency mismatch: {self.factor_id}")
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "formula_hash", digest)
        object.__setattr__(self, "operators", ops)
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "market": self.market,
            "surface": self.surface,
            "formula": self.formula,
            "formula_hash": self.formula_hash,
            "family": self.family,
            "subfamily": self.subfamily,
            "horizon": self.horizon,
            "complexity": self.complexity,
            "availability_tier": self.availability_tier,
            "required_fields": list(self.required_fields),
            "operators": list(self.operators),
            "rationale": self.rationale,
            "direction_hint": self.direction_hint,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ColdStartFactor":
        return cls(
            factor_id=str(row["factor_id"]),
            market=str(row["market"]),
            surface=str(row["surface"]),
            formula=str(row["formula"]),
            formula_hash=str(row.get("formula_hash") or ""),
            family=str(row["family"]),
            subfamily=str(row["subfamily"]),
            horizon=None if row.get("horizon") is None else int(row["horizon"]),
            complexity=str(row["complexity"]),
            availability_tier=str(row["availability_tier"]),
            required_fields=tuple(row.get("required_fields") or ()),
            operators=tuple(row.get("operators") or ()),
            rationale=str(row.get("rationale") or ""),
            direction_hint=str(row.get("direction_hint") or "unknown"),
            metadata=dict(row.get("metadata") or {}),
        )


def ensure_unique(records: Iterable[ColdStartFactor]) -> tuple[ColdStartFactor, ...]:
    rows = tuple(records)
    ids = [row.factor_id for row in rows]
    hashes = [row.formula_hash for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cold-start factor ids")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate cold-start factor formulas")
    return rows
