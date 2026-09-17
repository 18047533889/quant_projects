"""Explicit, bounded reviewed DSL revision; compilation is not execution."""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import apply_reviewed_formula, validate_review_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--expected-source-sha256", required=True)
    ap.add_argument("--output-prefix", required=True, type=Path)
    ap.add_argument("--max-changes", type=int, default=5000)
    ap.add_argument(
        "--recipe-set",
        choices=("r12", "r13", "r13_nested", "r13_technical", "r14_technical", "r15_technical", "r16_technical"),
        default="r12",
    )
    args = ap.parse_args()
    source_validation = validate_review_source(args.checkpoint, args.expected_source_sha256)
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    from factor_engine.tools.catalog_failure_recipes import migrate_catalog_failure_formula
    from factor_engine.tools.catalog_field_recipes import migrate_catalog_field_formula
    from factor_engine.tools.catalog_additional_failure_recipes import migrate_catalog_additional_failure_formula
    parser, engine = build_runtime()
    paths = {
        "checkpoint": Path(str(args.output_prefix) + ".csv.gz"),
        "decisions": Path(str(args.output_prefix) + ".decisions.jsonl.gz"),
        "input": Path(str(args.output_prefix) + ".input.jsonl.gz"),
        "manifest": Path(str(args.output_prefix) + ".manifest.json"),
    }
    staged = {k: Path(str(v) + ".partial") for k, v in paths.items()}
    for p in (*paths.values(), *staged.values()):
        if p.exists():
            raise FileExistsError(p)
    counts = collections.Counter()
    ids = set()
    try:
        with gzip.open(args.checkpoint, "rt", encoding="utf-8-sig", newline="") as src, \
             gzip.open(staged["checkpoint"], "xt", encoding="utf-8-sig", newline="") as dst, \
             gzip.open(staged["decisions"], "xt", encoding="utf-8") as decisions, \
             gzip.open(staged["input"], "xt", encoding="utf-8") as inputs:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for original in reader:
                counts["rows"] += 1
                row = original
                if original["id"]:
                    if original["id"] in ids:
                        raise ValueError("duplicate factor ID")
                    ids.add(original["id"])
                    needs_review = (
                        original["execution_status"] in {
                            "COMPILE_FAILED", "EXECUTION_FAILED", "PREPARE_FAILED",
                            "COMPILE_TIMEOUT", "EXECUTION_TIMEOUT",
                        }
                        or original.get("compile_status") == "COMPILE_FAILED"
                        or original.get("static_status") in {"FIELDS_UNRESOLVED", "PARSE_FAILED"}
                    )
                    if not needs_review:
                        writer.writerow(row)
                        counts["execution:" + row["execution_status"]] += 1
                        continue
                    try:
                        if args.recipe_set == "r16_technical":
                            from factor_engine.tools.catalog_r15_technical_recipes import (
                                migrate_catalog_r15_technical_formula,
                            )
                            from factor_engine.tools.catalog_r16_technical_recipes import (
                                migrate_catalog_r16_technical_formula,
                            )
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_catalog_r15_technical_formula(
                                original["current_formula"],
                                logic=original.get("logic", ""), enabled=True,
                            )
                            additional = migrate_catalog_r16_technical_formula(
                                first.formula, logic=original.get("logic", ""), enabled=True,
                            )
                            second = RecipeMigration(additional.formula, ())
                        elif args.recipe_set == "r15_technical":
                            from factor_engine.tools.catalog_r15_technical_recipes import (
                                migrate_catalog_r15_technical_formula,
                            )
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_catalog_r15_technical_formula(
                                original["current_formula"],
                                logic=original.get("logic", ""), enabled=True,
                            )
                            additional = RecipeMigration(first.formula, ())
                            second = RecipeMigration(first.formula, ())
                        elif args.recipe_set == "r14_technical":
                            from factor_engine.tools.catalog_r14_technical_recipes import (
                                migrate_catalog_r14_technical_formula,
                            )
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_catalog_r14_technical_formula(
                                original["current_formula"],
                                logic=original.get("logic", ""), enabled=True,
                            )
                            additional = RecipeMigration(first.formula, ())
                            second = RecipeMigration(first.formula, ())
                        elif args.recipe_set == "r13_technical":
                            from factor_engine.tools.catalog_r13_technical_recipes import (
                                migrate_catalog_r13_technical_formula,
                            )
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_catalog_r13_technical_formula(
                                original["current_formula"],
                                logic=original.get("logic", ""), enabled=True,
                            )
                            additional = RecipeMigration(first.formula, ())
                            second = RecipeMigration(first.formula, ())
                        elif args.recipe_set == "r13_nested":
                            from factor_engine.tools.catalog_r13_nested_recipes import migrate_r13_nested_formula
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_r13_nested_formula(
                                original["current_formula"], logic=original.get("logic", ""), enabled=True,
                            )
                            additional = RecipeMigration(first.formula, ())
                            second = RecipeMigration(first.formula, ())
                        elif args.recipe_set == "r13":
                            from factor_engine.tools.catalog_r13_expression_recipes import migrate_catalog_r13_expression_formula
                            from factor_engine.tools.catalog_r13_parameter_recipes import migrate_catalog_r13_parameter_formula
                            from factor_engine.tools.catalog_recipe_migration import RecipeMigration
                            first = migrate_catalog_r13_expression_formula(
                                original["current_formula"], logic=original.get("logic", ""),
                                enabled=True,
                            )
                            additional = migrate_catalog_r13_parameter_formula(
                                first.formula, logic=original.get("logic", ""), enabled=True,
                            )
                            second = RecipeMigration(additional.formula, ())
                        else:
                            first = migrate_catalog_failure_formula(
                                original["current_formula"], logic=original.get("logic", ""),
                                enabled=True,
                            )
                            additional = migrate_catalog_additional_failure_formula(
                                first.formula, logic=original.get("logic", ""), enabled=True,
                            )
                            second = migrate_catalog_field_formula(
                                additional.formula, logic=original.get("logic", ""),
                                tables=original.get("original_tables", ""), enabled=True,
                            )
                    except (ValueError, SyntaxError) as exc:
                        counts["migration_rejected"] += 1
                        decisions.write(json.dumps({
                            "source_row": original["source_row"], "id": original["id"],
                            "status": "MIGRATION_REJECTED", "error": str(exc)[:500],
                        }, ensure_ascii=False) + "\n")
                    else:
                        if second.formula != original["current_formula"]:
                            counts["changed"] += 1
                            if counts["changed"] > args.max_changes:
                                raise RuntimeError("review change budget exceeded")
                            row = apply_reviewed_formula(
                                original, second.formula,
                                tuple(first.changes) + tuple(additional.changes) + tuple(second.changes),
                            )
                            row["compile_validation_scope"] = f"{args.recipe_set}_exact_formula_no_read_preflight"
                            try:
                                expr = parser.parse(row["current_formula"])
                                bindings, failures = bind_fields(expr)
                                row["bindings"] = json.dumps(bindings, ensure_ascii=False)
                                row["current_fields"] = json.dumps(
                                    sorted({b["input"] for b in bindings}), ensure_ascii=False
                                )
                                row["current_tables"] = json.dumps(
                                    sorted({b["table"] for b in bindings if b["table"]}),
                                    ensure_ascii=False,
                                )
                                row["static_status"] = "FIELDS_UNRESOLVED" if failures else "PARSED_FIELDS_BOUND"
                                row["static_reason"] = json.dumps(failures, ensure_ascii=False) if failures else ""
                                engine.compile(Factor(
                                    name=row["id"], expr=expr, source_expr=row["current_formula"],
                                    surface="compat_research",
                                ))
                                row["compile_status"] = "COMPILED"
                            except Exception as exc:
                                if row["static_status"] == "NOT_RUN":
                                    row["static_status"] = "PARSE_FAILED"
                                    row["static_reason"] = str(exc)[:1000]
                                row["compile_status"] = "COMPILE_FAILED"
                                row["compile_reason"] = type(exc).__name__ + ": " + str(exc)[:1000]
                            row["compile_evidence_file"] = paths["decisions"].name
                            decision = {
                                "source_row": row["source_row"], "id": row["id"],
                                "original_formula": row["original_formula"],
                                "previous_formula": original["current_formula"],
                                "current_formula": row["current_formula"],
                                "logic": row.get("logic", ""),
                                "changes": list(first.changes) + list(additional.changes) + list(second.changes),
                                "previous_execution_status": original["execution_status"],
                                "previous_execution_evidence": original["execution_evidence_file"],
                                "static_status": row["static_status"],
                                "static_reason": row["static_reason"],
                                "compile_status": row["compile_status"],
                                "compile_reason": row["compile_reason"],
                                "status": "REVIEWED_FORMULA_CHANGED_NOT_EXECUTED",
                            }
                            decisions.write(json.dumps(decision, ensure_ascii=False) + "\n")
                            projected = {
                                "source_row": int(row["source_row"]), "id": row["id"],
                                "formula": row["current_formula"],
                                "fields": row["original_fields"], "tables": row["original_tables"],
                                **{k: row.get(k, "") for k in ("domain", "pit", "logic", "batch")},
                            }
                            inputs.write(json.dumps(projected, ensure_ascii=False) + "\n")
                            counts["changed_static:" + row["static_status"]] += 1
                            counts["changed_compile:" + row["compile_status"]] += 1
                            if counts["changed"] % 25 == 0:
                                print(json.dumps(dict(counts)), flush=True)
                if tuple(row[k] for k in ("source_row", "id", "original_formula")) != tuple(original[k] for k in ("source_row", "id", "original_formula")):
                    raise ValueError("review changed immutable row identity")
                writer.writerow(row)
                counts["execution:" + row["execution_status"]] += 1
        if counts["rows"] != source_validation["rows"] or len(ids) != source_validation["unique_factor_ids"]:
            raise ValueError("review/source record count mismatch")
        # Consume gzip footers before promoting any evidence.
        for kind in ("checkpoint", "decisions", "input"):
            with gzip.open(staged[kind], "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
        hashes = {k: hashlib.sha256(staged[k].read_bytes()).hexdigest()
                  for k in ("checkpoint", "decisions", "input")}
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.expected_source_sha256:
            raise ValueError("source changed during review")
        manifest = {
            "recipe_set": args.recipe_set,
            "counts": dict(counts), "unique_factor_ids": len(ids),
            "source_checkpoint": str(args.checkpoint),
            "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "output_sha256": hashes, "scope": "reviewed_changes_compile_not_execution",
            "authorization": "user requested explanation-backed DSL completion; explicit decisions retained",
        }
        with staged["manifest"].open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        # Manifest is published last: consumers must require it and verify hashes.
        for key in paths:
            staged[key].rename(paths[key])
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    finally:
        for p in staged.values():
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
