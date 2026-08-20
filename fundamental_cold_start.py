# -*- coding: utf-8
"""Strict cold-start contract for the single-default fundamental library."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_FILENAME = "fundamental_factors_single_default_1288.csv"
_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_ROOTS = (_ROOT, _ROOT / "data", _ROOT.parent, _ROOT.parent / "data")
_REQUIRED_COLUMNS = ("factor_id", "dsl")


@dataclass(frozen=True)
class ColdStartReport:
    status: str
    filename: str
    path: str | None
    factor_count: int
    error: str | None
    source_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "filename": self.filename,
            "path": self.path,
            "factor_count": self.factor_count,
            "error": self.error,
            "source_sha256": self.source_sha256,
        }


def locate_cold_start_file() -> Path | None:
    for root in _SEARCH_ROOTS:
        candidate = root / EXPECTED_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_cold_start(path: str | Path | None = None) -> tuple[list[dict[str, str]], ColdStartReport]:
    source = Path(path) if path is not None else locate_cold_start_file()
    if source is None:
        return [], ColdStartReport("unavailable", EXPECTED_FILENAME, None, 0, "missing_cold_start_source", None)
    if source.name != EXPECTED_FILENAME:
        raise ValueError(f"only {EXPECTED_FILENAME!r} is accepted as the cold-start source")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing = sorted(set(_REQUIRED_COLUMNS) - set(fields))
            if missing:
                raise ValueError(f"missing required columns: {missing}")
            rows = [dict(row) for row in reader]
        ids = [str(row["factor_id"]).strip() for row in rows]
        formulas = [str(row["dsl"]).strip() for row in rows]
        if any(not value for value in ids + formulas):
            raise ValueError("factor_id and dsl must be non-empty")
        if len(ids) != len(set(ids)):
            raise ValueError("factor_id values must be unique")
        if len(formulas) != len(set(formulas)):
            raise ValueError("dsl values must be unique")
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return [], ColdStartReport("unavailable", EXPECTED_FILENAME, str(source), 0, f"invalid_cold_start_source: {exc}", None)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return rows, ColdStartReport("available", EXPECTED_FILENAME, str(source), len(rows), None, digest)


def write_cold_start_report(path: str | Path) -> Path:
    rows, report = load_cold_start()
    target = Path(path)
    target.write_text(json.dumps({"report": report.as_dict(), "factor_ids": [row["factor_id"] for row in rows]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
