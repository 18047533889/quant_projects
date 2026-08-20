"""R21-152..160: formula schema versioning, migration registry, canonical
textual serializer.

- persisted ``source_expr`` carries ``formula_schema_version`` (R21-152).
- operator renames / param renames / dialect upgrades live in a migration
  registry with ``deprecate_since / remove_after / replacement`` (R21-153..155).
- ``migrate_formula(from_version, to_version)`` rewrites a stored formula
  (R21-156).
- ``canonical_serialize`` derives a unique textual form from a typed Expr/IR so
  ``parse -> canonical serialize -> parse`` is identity (R21-157..160); the
  original user ``source_expr`` and the ``canonical_expr`` are both kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FORMULA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperatorMigration:
    replacement: str
    deprecate_since: str
    remove_after: str
    param_renames: dict[str, str] = None  # type: ignore[assignment]


_MIGRATIONS: dict[str, OperatorMigration] = {}


def register_operator_migration(
    operator: str,
    replacement: str,
    *,
    deprecate_since: str = "1.0",
    remove_after: str = "2.0",
    param_renames: dict[str, str] | None = None,
) -> None:
    """R21-153/154: declare an operator rename / param rename / removal window."""
    _MIGRATIONS[operator] = OperatorMigration(
        replacement=replacement,
        deprecate_since=deprecate_since,
        remove_after=remove_after,
        param_renames=dict(param_renames or {}),
    )


def operator_replacement(operator: str) -> str | None:
    migration = _MIGRATIONS.get(operator)
    return migration.replacement if migration else None


def is_deprecated(operator: str, *, since: str | None = None) -> bool:
    migration = _MIGRATIONS.get(operator)
    if migration is None:
        return False
    if since is None:
        return True
    # coarse tuple compare of dotted versions
    def _key(v: str):
        return tuple(int(p) if p.isdigit() else 0 for p in v.split("."))
    return _key(since) >= _key(migration.deprecate_since)


def migrate_formula(formula: str, *, from_version: int, to_version: int) -> str:
    """R21-156: migrate a stored formula from one schema version to another.

    Schema v1 -> v1 is identity (only renames recorded in the registry are
    applied when a replacement is registered and not yet removed).
    """
    if from_version > to_version:
        raise ValueError("cannot migrate a formula to an older schema version")
    if to_version > FORMULA_SCHEMA_VERSION:
        raise ValueError(f"target formula schema {to_version} not supported")
    if not _MIGRATIONS:
        return formula
    import re

    out = formula
    for operator, migration in _MIGRATIONS.items():
        out = re.sub(rf"\b{re.escape(operator)}\s*\(", migration.replacement + "(", out)
    return out


# ---------------------------------------------------------------------------
# Canonical textual serializer (R21-157..160)
# ---------------------------------------------------------------------------


def canonical_serialize(expr: Any) -> str:
    """Unique textual representation of a typed Expr/IR.

    ``parse -> canonical_serialize -> parse`` is identity for the supported
    node kinds (CleanedCall / FieldRef / Literal / binary ops).  Used for
    catalog diff, migration, human review and reproducibility — separate from
    the user's original ``source_expr``.
    """
    kind = type(expr).__name__

    if kind == "Literal":
        value = expr.value
        if isinstance(value, str):
            return json_dumps(value)
        return repr(value)

    if kind == "FieldRef":
        name = getattr(expr, "name", None) or getattr(expr, "canonical_name", None) or "?"
        return _field_ref(name)

    if kind == "CleanEdCall" or kind == "CleanedCall":
        op = getattr(expr, "op", None)
        args = list(getattr(expr, "args", None) or [])
        kd = getattr(expr, "kwargs_dict", None)
        if callable(kd):
            kwargs = dict(kd()) if getattr(expr, "kwargs", None) is None else dict(getattr(expr, "kwargs", {}) or {})
        else:
            kwargs = dict(kd or {})
        arg_s = ", ".join(canonical_serialize(a) for a in args)
        kw_s = ", ".join(f"{k}={canonical_serialize(v)}" for k, v in sorted(kwargs.items()))
        parts = [p for p in (arg_s, kw_s) if p]
        return f"{op}({', '.join(parts)})"

    # Fallback for Expr subclasses exposing .children/.args recursively.
    op = getattr(expr, "op", None) or getattr(expr, "name", None)
    children = getattr(expr, "children", None)
    if children:
        return f"{op}({', '.join(canonical_serialize(c) for c in children)})"

    return repr(expr)


def _field_ref(name: str) -> str:
    if "(" in name or ")" in name or "," in name:
        return json_dumps(name)
    return name


def json_dumps(value: str) -> str:
    import json

    return json.dumps(value)


def persist_source_expr(source_expr: str, expr: Any) -> dict[str, str]:
    """R21-160: keep BOTH the user's source_expr and the canonical expr.

    The canonical expr is the reproducibility key; the source_expr is the
    human/migration-friendly form.
    """
    return {
        "source_expr": source_expr,
        "canonical_expr": canonical_serialize(expr),
        "formula_schema_version": str(FORMULA_SCHEMA_VERSION),
    }
