"""Recompile the reviewed 429 R20 operator/parameter proposals against current code."""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evidence/factor_catalog_20260915"))

from compile_catalog import build_runtime
from smoke_catalog import bind_fields
from factor_engine.api.factor import Factor
from factor_engine.tools.catalog_r20_ops_params_recipes import migrate_formula
from factor_engine.tools.catalog_r19_parameter_recipes import migrate_formula as migrate_r19_parameters
from factor_engine.tools.catalog_r19_remaining_params_recipes import migrate_formula as migrate_r19_remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-before-formula", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    counts: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    dsl_parser, engine = build_runtime()
    with args.input.open(encoding="utf-8") as src, args.output.open("x", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            row = json.loads(line)
            seed_formula = row["before_formula"] if args.from_before_formula else row["current_formula"]
            formula, notes = seed_formula, []
            migrations = (migrate_r19_parameters, migrate_r19_remaining, migrate_r19_parameters, migrate_r19_remaining) if args.from_before_formula else ()
            for migration in migrations:
                formula, added_notes = migration(formula)
                notes.extend(added_notes)
            for _ in range(4):
                migrated, added_notes = migrate_formula(formula, row.get("current_definition", ""))
                notes.extend(added_notes)
                if migrated == formula:
                    break
                formula = migrated
            record = dict(row)
            record["current_formula"] = formula
            prior_notes = [] if args.from_before_formula else row.get("changes", [])
            record["changes"] = list(dict.fromkeys([*prior_notes, *notes]))
            record.update(compile_status="COMPILED", error="", bindings=[])
            try:
                expr = dsl_parser.parse(formula)
                bindings, failures = bind_fields(expr)
                record["bindings"] = bindings
                if failures:
                    raise ValueError("FIELD_BINDING_FAILED: " + json.dumps(failures, ensure_ascii=False))
                engine.compile(Factor(name=row["id"], expr=expr, source_expr=formula, surface="compat_research"))
            except Exception as exc:
                record.update(compile_status="COMPILE_FAILED", error=f"{type(exc).__name__}: {exc}")
            counts[record["compile_status"]] += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            if line_number % 100 == 0:
                dst.flush()
                print(json.dumps({"rows": line_number, "counts": counts, "seconds": time.monotonic() - started}), flush=True)
    print(json.dumps({"rows": sum(counts.values()), "counts": counts, "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
