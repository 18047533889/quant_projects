"""Evidence invalidation for explicitly reviewed catalog formula changes."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def apply_reviewed_formula(
    row: Mapping[str, Any], formula: str, changes: Sequence[str]
) -> dict[str, Any]:
    """Return a new row; a changed formula cannot inherit old passing evidence.

    The caller retains the immutable input checkpoint and writes the old/new
    formula plus decisions to its review ledger. This does not certify a rewrite.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("reviewed formula must be a nonempty string")
    if isinstance(changes, (str, bytes)) or not all(isinstance(x, str) for x in changes):
        raise ValueError("changes must be a sequence of decision strings")
    prior = row.get("migration_changes") or "[]"
    notes = json.loads(prior) if isinstance(prior, str) else list(prior)
    if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
        raise ValueError("existing migration ledger must contain decision strings")
    result = dict(row)
    result["migration_changes"] = json.dumps(
        list(dict.fromkeys([*notes, *changes])), ensure_ascii=False
    )
    result["current_formula"] = formula
    if formula == row.get("current_formula"):
        return result
    for key in (
        "static_reason", "bindings", "current_fields", "current_tables",
        "compile_reason", "compile_evidence_file", "compile_validation_scope",
        "execution_reason", "value_count", "finite_count", "result_hash",
        "retry_after_batch_abort", "batch_abort_error_type", "batch_abort_error",
        "backend_path", "execution_evidence_file", "execution_validation_scope",
        "execution_window", "execution_symbol_count",
    ):
        result[key] = ""
    for key in ("static_status", "compile_status", "execution_status"):
        result[key] = "NOT_RUN"
    result["execution_reason"] = "Formula changed under explicit review; previous evidence invalidated"
    return result


def validate_review_source(path, expected_sha256: str) -> dict[str, int]:
    """Verify a pinned checkpoint and exhaust its gzip before any promotion."""
    import csv
    import gzip
    import hashlib
    from pathlib import Path

    source = Path(path)
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("source checkpoint SHA256 mismatch")
    ids, source_rows = set(), set()
    count = 0
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_row", "id", "original_formula", "current_formula",
                    "execution_status", "execution_validation_scope"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("source checkpoint missing identity/evidence fields")
        for row in reader:
            source_row = int(row["source_row"])
            if source_row in source_rows:
                raise ValueError("duplicate source row")
            source_rows.add(source_row)
            count += 1
            if row["id"]:
                if row["id"] in ids:
                    raise ValueError("duplicate factor ID")
                ids.add(row["id"])
            if ("donchian_position(" in row["original_formula"]
                    and row["execution_status"] in {"EXECUTED", "EXECUTED_ALL_NONFINITE"}
                    and (not row["execution_validation_scope"]
                         or "historical" in row["execution_validation_scope"])):
                raise ValueError("stale Donchian execution evidence in source")
    return {"rows": count, "unique_factor_ids": len(ids)}
