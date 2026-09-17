"""Streaming real-engine compile preflight for the imported factor catalog.

This checks DSL migration, parsing, analysis, lowering, and optimization only.
It deliberately does not execute plans or certify numerical results.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable

from factor_engine.storage.sources.datasource import DataSource


class _NoReadDataSource(DataSource):
    """Compile-only source: any accidental data access fails closed."""

    def _reject(self, method: str, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"compile preflight must not call data source method {method}")

    def load_column(self, *args: Any, **kwargs: Any) -> Any:
        return self._reject("load_column", *args, **kwargs)

    def load_columns(self, *args: Any, **kwargs: Any) -> Any:
        return self._reject("load_columns", *args, **kwargs)

    def prefetch_columns(self, *args: Any, **kwargs: Any) -> Any:
        return self._reject("prefetch_columns", *args, **kwargs)

    def scan_polars_long(self, *args: Any, **kwargs: Any) -> Any:
        return self._reject("scan_polars_long", *args, **kwargs)

    def scan_index_long(self, *args: Any, **kwargs: Any) -> Any:
        return self._reject("scan_index_long", *args, **kwargs)


def build_runtime():
    """Build one frozen parser and one real research-mode compile engine."""
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.backend.pandas_backend import PandasBackend
    from factor_engine.cleaned_operators import load_all
    from factor_engine.runtime.engine import FactorEngine

    load_all()
    return (
        DSLParser(surface="compat_research"),
        FactorEngine(PandasBackend(), _NoReadDataSource(), run_mode="research"),
    )


def compile_record(
    record: dict[str, Any], *, parser: Any, engine: Any,
    semantic_redesign_intraday_limit: bool = False,
) -> dict[str, Any]:
    """Compile one catalog row and return an audit row with the exact error."""
    from factor_engine.api.factor import Factor
    from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
    from factor_engine.tools.catalog_recipe_migration import (
        migrate_catalog_recipe_formula, redesign_catalog_intraday_limit_formula,
    )
    from factor_engine.tools.catalog_sketch_migration import migrate_catalog_sketch_formula

    row = {key: record.get(key) for key in ("source_row", "id", "formula")}
    try:
        factor_id = record.get("id")
        formula = record.get("formula")
        if not factor_id:
            raise ValueError("NON_FACTOR_OR_MISSING_ID")
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError("MISSING_FORMULA")

        # Keep this order identical to audit_catalog.py.
        sketch = migrate_catalog_sketch_formula(formula, enabled=True, parser=parser)
        redesign = redesign_catalog_intraday_limit_formula(
            sketch.formula, enabled=semantic_redesign_intraday_limit
        )
        price = migrate_adjusted_price_fields(redesign.formula, market="ashare")
        recipe = migrate_catalog_recipe_formula(price.formula)
        changes = (("parameter sketch -> keyword call",) if sketch.converted else ())
        changes += tuple(redesign.changes) + tuple(price.changes) + tuple(recipe.changes)
        row.update(
            current_formula=recipe.formula,
            migration_changes=changes,
            sketch_status=sketch.status,
        )
        expr = parser.parse(recipe.formula)
        factor = Factor(
            name=str(factor_id),
            expr=expr,
            source_expr=recipe.formula,
            surface="compat_research",
        )
        engine.compile(factor)
        row["status"] = "COMPILED"
    except Exception as exc:
        row.update(
            status="COMPILE_FAILED",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return row


def audit_stream(
    records: Iterable[dict[str, Any]],
    dst: Any,
    *,
    parser: Any,
    engine: Any,
    limit: int = 0,
    semantic_redesign_intraday_limit: bool = False,
    progress_every: int = 1000,
) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    failures: collections.Counter[tuple[str, str]] = collections.Counter()
    examples: dict[tuple[str, str], dict[str, Any]] = {}
    started = time.monotonic()
    processed = 0
    for record in records:
        if limit and processed >= limit:
            break
        row = compile_record(
            record, parser=parser, engine=engine,
            semantic_redesign_intraday_limit=semantic_redesign_intraday_limit,
        )
        processed += 1
        counts[row["status"]] += 1
        if row["status"] == "COMPILE_FAILED":
            # Summary cardinality and size stay bounded; per-row output keeps exact errors.
            key = (row["error_type"], row["error"][:500])
            failures[key] += 1
            if len(examples) < 100:
                examples.setdefault(
                    key,
                    {"source_row": row.get("source_row"), "formula": row.get("formula")},
                )
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        if progress_every and processed % progress_every == 0:
            print(
                json.dumps(
                    {"processed": processed, "counts": counts, "seconds": time.monotonic() - started}
                ),
                flush=True,
            )
    elapsed = time.monotonic() - started
    return {
        "processed": processed,
        "counts": dict(counts),
        "seconds": elapsed,
        "rows_per_second": processed / elapsed if elapsed else None,
        "failures": [
            {
                "error_type": key[0],
                "error": key[1],
                "count": count,
                "example": examples.get(key),
            }
            for key, count in failures.most_common(100)
        ],
        "failure_groups_truncated": len(failures) > 100,
        "scope": "compile_preflight_only_not_execution_certification",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.jsonl.gz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--semantic-redesign-intraday-limit", action="store_true")
    args = ap.parse_args()
    if args.start < 0 or args.limit < 0:
        ap.error("--start and --limit must be non-negative")

    base = Path(__file__).resolve().parent
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = base / input_path
    if not output_path.is_absolute():
        output_path = base / output_path

    parser, engine = build_runtime()
    with gzip.open(input_path, "rt", encoding="utf-8") as src, gzip.open(
        output_path, "wt", encoding="utf-8"
    ) as dst:
        records = itertools.islice((json.loads(line) for line in src), args.start, None)
        summary = audit_stream(
            records, dst, parser=parser, engine=engine, limit=args.limit,
            semantic_redesign_intraday_limit=args.semantic_redesign_intraday_limit,
        )
    summary["start"] = args.start
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
