"""Bounded current-DSL audit of an imported factor recipe catalog (not execution)."""
from __future__ import annotations
import argparse
import collections
import gzip
import json
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--semantic-redesign-intraday-limit",
        action="store_true",
        help="opt in to the reviewed 3-argument intraday-limit semantic redesign",
    )
    args = ap.parse_args()
    from factor_engine.cleaned_operators import load_all
    from factor_engine.api.dsl_parser import DSLParser
    from factor_engine.tools.catalog_migration import migrate_adjusted_price_fields
    from factor_engine.tools.catalog_recipe_migration import (
        migrate_catalog_recipe_formula,
        redesign_catalog_intraday_limit_formula,
    )
    from factor_engine.tools.catalog_sketch_migration import migrate_catalog_sketch_formula
    from factor_engine.expr.column import ColumnRef
    from factor_engine.fields.resolver import resolve_market_field
    load_all()
    parser = DSLParser(surface="compat_research")
    base = Path(__file__).parent
    counts = collections.Counter()
    failures = collections.Counter()
    fields = collections.Counter()
    examples = {}
    started = time.monotonic()
    output = base / args.output
    with gzip.open(base / "input.jsonl.gz", "rt", encoding="utf-8") as src, gzip.open(output, "wt", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if args.limit and index >= args.limit:
                break
            record = json.loads(line)
            row = {k: record[k] for k in ("source_row", "id", "formula")}
            if not record["id"]:
                row.update(status="NON_FACTOR_RECORD", error="Source contains a mechanism/task record, not an identified executable factor")
                counts[row["status"]] += 1
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            try:
                if not record["id"]:
                    raise ValueError("NON_FACTOR_OR_MISSING_ID")
                if not record["formula"].strip():
                    raise ValueError("MISSING_FORMULA")
                sketch = migrate_catalog_sketch_formula(record["formula"], enabled=True, parser=parser)
                redesign = redesign_catalog_intraday_limit_formula(
                    sketch.formula, enabled=args.semantic_redesign_intraday_limit
                )
                migration = migrate_adjusted_price_fields(redesign.formula, market="ashare")
                recipe = migrate_catalog_recipe_formula(migration.formula)
                row.update(current_formula=recipe.formula, migration_changes=(("parameter sketch -> keyword call",) if sketch.converted else ()) + redesign.changes + migration.changes + recipe.changes, sketch_status=sketch.status)
                expr = parser.parse(recipe.formula)
                leaves = {}
                stack = [expr]
                while stack:
                    node = stack.pop()
                    if isinstance(node, ColumnRef):
                        leaves[(node.name, getattr(node, "table", None))] = node
                    stack.extend(node.children())
                    for _, value in getattr(node, "kwargs", ()):
                        if hasattr(value, "children"):
                            stack.append(value)
                bindings = []
                unresolved = []
                for (name, _table), leaf in sorted(leaves.items(), key=lambda item: str(item[0])):
                    fields[name] += 1
                    try:
                        result = resolve_market_field(leaf, "ashare", strict=True)
                        spec = result.spec
                        bindings.append({"input": name, "table": spec.table, "column": spec.source_name,
                                         "unit": spec.unit, "source_unit": spec.source_unit,
                                         "scale": spec.scale_to_canonical, "frequency": spec.frequency})
                    except (ValueError, KeyError, TypeError) as exc:
                        unresolved.append({"input": name, "error": str(exc)[:250]})
                row.update(status="FIELDS_UNRESOLVED" if unresolved else "PARSED_FIELDS_BOUND",
                           bindings=bindings, unresolved=unresolved)
            except Exception as exc:
                row.update(status="PARSE_FAILED", error_type=type(exc).__name__, error=str(exc)[:500])
                key = row["error_type"] + ": " + row["error"]
                failures[key] += 1
                examples.setdefault(key, {"source_row": record["source_row"], "formula": record["formula"]})
            counts[row["status"]] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (index + 1) % 1000 == 0:
                print(json.dumps({"processed": index + 1, "counts": counts, "seconds": time.monotonic()-started}), flush=True)
    summary = {"counts": counts, "seconds": time.monotonic()-started, "fields": fields.most_common(),
               "failures": [{"error": key, "count": value, "example": examples[key]} for key, value in failures.most_common()]}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"counts": counts, "seconds": summary["seconds"], "top_failures": summary["failures"][:15]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
