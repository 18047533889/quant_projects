"""Opt-in conversion of complete ``operator[key=value]`` catalog sketches."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_MAX_FORMULA_BYTES = 65_536
_SKETCH = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]\s*", re.DOTALL)
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class SketchMigration:
    formula: str
    converted: bool
    status: str


def _split_assignments(text: str) -> list[str] | None:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif char in ",;" and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        return None
    parts.append(text[start:].strip())
    return parts if all(parts) else None


def convert_complete_parameter_sketch(formula: str) -> SketchMigration:
    """Perform only the shape-safe bracket-to-keyword-call conversion."""

    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    match = _SKETCH.fullmatch(formula)
    if match is None:
        return SketchMigration(formula, False, "not_complete_sketch")
    name, body = match.groups()
    parts = _split_assignments(body)
    if not parts:
        return SketchMigration(formula, False, "invalid_assignment_list")
    assignments: list[tuple[str, str]] = []
    seen: set[str] = set()
    for part in parts:
        if "=" not in part:
            return SketchMigration(formula, False, "invalid_assignment")
        key, value = (piece.strip() for piece in part.split("=", 1))
        if _KEY.fullmatch(key) is None or key in seen:
            return SketchMigration(formula, False, "invalid_or_duplicate_key")
        if not value:
            return SketchMigration(formula, False, "invalid_value")
        try:
            ast.parse(value, mode="eval")
        except SyntaxError:
            return SketchMigration(formula, False, "invalid_value")
        seen.add(key)
        assignments.append((key, value))
    candidate = f"{name}(" + ", ".join(f"{key}={value}" for key, value in assignments) + ")"
    return SketchMigration(candidate, True, "syntax_converted")


def migrate_catalog_sketch_formula(
    formula: str,
    *,
    enabled: bool = False,
    parser: object | None = None,
    surface: str = "compat_research",
) -> SketchMigration:
    """Convert only when a frozen DSL parser accepts the resulting call.

    Bulk callers should supply one ``DSLParser`` session through ``parser`` so
    every row is checked against the same immutable registry snapshot.
    """

    if not enabled:
        if not isinstance(formula, str):
            raise TypeError("formula must be a string")
        return SketchMigration(formula, False, "disabled")
    converted = convert_complete_parameter_sketch(formula)
    if not converted.converted:
        return converted

    if parser is None:
        from factor_engine.api.dsl_parser import DSLParser

        parser = DSLParser(surface=surface)
    try:
        parser.parse(converted.formula)  # type: ignore[attr-defined]
    except (RuntimeError, SyntaxError, TypeError, ValueError):
        return SketchMigration(formula, False, "dsl_validation_failed")
    return SketchMigration(converted.formula, True, "converted")
