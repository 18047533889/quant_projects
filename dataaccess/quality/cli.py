"""Minimal registry contract quality checks and CLI."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pyarrow as pa


@dataclass(frozen=True)
class QualityReport:
    dataset: str
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    sampled: bool = False


def validate_table(
    dataset: str,
    table: pa.Table,
    *,
    required_columns: tuple[str, ...] = (),
    primary_key: tuple[str, ...] = (),
    quality_mode: str = "strict",
) -> QualityReport:
    failures: list[str] = []
    checks: list[str] = []
    missing = [name for name in required_columns if name not in table.column_names]
    checks.append("required_columns")
    if missing:
        failures.append(f"missing columns: {missing}")
    if primary_key:
        checks.append("primary_key")
        if any(name not in table.column_names for name in primary_key):
            failures.append(f"primary key columns missing: {list(primary_key)}")
        else:
            frame = table.select(list(primary_key)).to_pandas()
            if frame.duplicated().any():
                failures.append("duplicate primary keys")
    if quality_mode not in {"off", "warn", "strict"}:
        raise ValueError("quality_mode must be off, warn, or strict")
    if quality_mode == "off":
        failures = []
    return QualityReport(dataset, not failures, tuple(checks), tuple(failures))


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate a data_access Arrow/Parquet contract")
    parser.add_argument("dataset")
    parser.add_argument("path")
    parser.add_argument("--required-column", action="append", default=[])
    parser.add_argument("--primary-key", action="append", default=[])
    args = parser.parse_args()
    table = pa.Table.from_pandas(__import__("pandas").read_parquet(args.path), preserve_index=False)
    report = validate_table(
        args.dataset,
        table,
        required_columns=tuple(args.required_column),
        primary_key=tuple(args.primary_key),
    )
    print(json.dumps(asdict(report), ensure_ascii=False))
    return 0 if report.passed else 1
