"""Build a pinned R18 technical-recipe checkpoint from the delivered R17c."""
from __future__ import annotations

import argparse, collections, csv, gzip, hashlib, json, os, sys, tempfile, time
from pathlib import Path

from factor_engine.tools.catalog_r18_technical_recipes import migrate_catalog_r18_technical_formula


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _invalidate_execution(row: dict[str, str], reason: str) -> None:
    history = {key: row.get(key, "") for key in (
        "compile_status", "compile_evidence_file", "execution_status",
        "execution_evidence_file", "result_hash",
    )}
    changes = json.loads(row.get("migration_changes") or "[]")
    changes.append("R18_EVIDENCE_INVALIDATION: " + reason + "; prior=" + json.dumps(history, sort_keys=True))
    row["migration_changes"] = json.dumps(changes, ensure_ascii=False)
    row.update(execution_status="NOT_RUN", execution_reason=reason)
    for key in (
        "value_count", "finite_count", "result_hash", "retry_after_batch_abort",
        "batch_abort_error_type", "batch_abort_error", "backend_path",
        "execution_evidence_file", "execution_validation_scope", "execution_window",
        "execution_symbol_count",
    ):
        row[key] = ""


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--expected-input-sha",required=True); ap.add_argument("--output-prefix",type=Path,required=True)
    ap.add_argument("--expected-changed", type=int, required=True)
    ap.add_argument("--expected-recipe-sha", required=True)
    ap.add_argument("--deadline-seconds",type=float,default=150.0); args=ap.parse_args(argv)
    recipe=Path(__file__).resolve().parents[2]/"factor_engine/tools/catalog_r18_technical_recipes.py"
    if sha256(recipe)!=args.expected_recipe_sha: raise ValueError("recipe SHA mismatch")
    output=Path(str(args.output_prefix)+".csv.gz"); manifest=Path(str(args.output_prefix)+".manifest.json")
    evidence=Path(str(args.output_prefix)+".compile.jsonl.gz")
    for path in (output,manifest,evidence):
        if path.exists(): raise FileExistsError(path)
    if sha256(args.input)!=args.expected_input_sha: raise ValueError("input SHA mismatch")
    helper=Path(__file__).resolve().parents[1]/"factor_catalog_20260915"; sys.path.insert(0,str(helper))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime(); started=time.monotonic(); counts=collections.Counter(); execution_counts=collections.Counter(); kinds=collections.Counter(); seen=set(); ids=set(); changed_identities=set()
    with tempfile.TemporaryDirectory(prefix=".r18-",dir=args.output_prefix.parent) as tmp:
        staged=Path(tmp)/output.name; staged_evidence=Path(tmp)/evidence.name
        with gzip.open(args.input,"rt",encoding="utf-8-sig",newline="") as src, gzip.open(staged,"xt",encoding="utf-8-sig",newline="") as dst, gzip.open(staged_evidence,"xt",encoding="utf-8") as ev:
            reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames); writer.writeheader(); rows=0
            for original in reader:
                rows+=1; ident=(int(original["source_row"]),original.get("id") or "")
                if ident in seen: raise ValueError(f"duplicate identity {ident}")
                seen.add(ident); ids.add(ident[1]) if ident[1] else None
                row=dict(original); formula=row["current_formula"]
                try: migration=migrate_catalog_r18_technical_formula(formula,enabled=True)
                except (SyntaxError,ValueError): migration=None
                if migration and migration.changes:
                    formula=migration.formula
                    changes=json.loads(row.get("migration_changes") or "[]")
                    changes.extend(migration.changes)
                    row["migration_changes"]=json.dumps(changes,ensure_ascii=False)
                    kinds["technical_rows"]+=1
                    kinds["technical_changes"]+=len(migration.changes)
                changed=formula!=original["current_formula"]
                if not changed:
                    writer.writerow(row); counts[row.get("compile_status") or ""]+=1; execution_counts[row.get("execution_status") or ""]+=1; continue
                changed_identities.add(ident)
                row["current_formula"]=formula
                _invalidate_execution(row,"current_formula changed; prior evidence retained only in migration history")
                if time.monotonic()-started>=args.deadline_seconds: raise TimeoutError("R18 compile deadline exceeded")
                record={"source_row":ident[0],"id":ident[1],"executed_formula":formula,"formula_sha256":hashlib.sha256(formula.encode()).hexdigest()}
                row["bindings"]="[]"; row["current_fields"]=""; row["current_tables"]=""
                row["static_status"]=""; row["static_reason"]=""
                expr=None; phase="parse"
                try:
                    expr=parser.parse(formula); phase="bind"; bindings,failures=bind_fields(expr)
                    record["field_binding_count"]=len(bindings); record["field_binding_failures"]=failures
                    row["bindings"]=json.dumps(bindings,ensure_ascii=False)
                    fields=sorted({f"{x['table']}.{x['column']}" for x in bindings}); row["current_fields"]="|".join(fields)
                    row["current_tables"]="|".join(sorted({x["table"] for x in bindings}))
                    if failures:
                        row["static_status"]="FIELDS_UNRESOLVED"; row["static_reason"]="; ".join(map(str,failures))
                        raise ValueError("FIELD_BINDING_FAILED")
                    row["static_status"]="PARSED_FIELDS_BOUND"; row["static_reason"]=""
                    phase="compile"
                    engine.compile(Factor(name=ident[1],expr=expr,source_expr=formula,surface="compat_research"))
                    row.update(compile_status="COMPILED",compile_reason=""); record["compile_status"]="COMPILED"
                except Exception as exc:
                    if phase == "parse":
                        row["static_status"]="PARSE_FAILED"; row["static_reason"]=str(exc)[:2000]
                    elif phase == "bind":
                        row["static_status"]="FIELDS_UNRESOLVED"; row["static_reason"]=str(exc)[:2000]
                    row.update(compile_status="COMPILE_FAILED",compile_reason=str(exc)[:2000])
                    record.update(compile_status="COMPILE_FAILED",error_type=type(exc).__name__,error=str(exc)[:2000])
                row["compile_evidence_file"]=evidence.name; row["compile_validation_scope"]="formula-matched compile-only; no data read; not execution evidence"
                ev.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n"); writer.writerow(row); counts[row["compile_status"]]+=1; execution_counts[row["execution_status"]]+=1
        if rows!=114132 or len(seen)!=114132: raise ValueError("row coverage mismatch")
        if len(changed_identities)!=args.expected_changed: raise ValueError(f"unexpected migration counts {kinds}")
        if sha256(recipe)!=args.expected_recipe_sha: raise ValueError("recipe changed during build")
        if sha256(args.input)!=args.expected_input_sha: raise ValueError("input changed during build")
        payload={"input_checkpoint":str(args.input),"input_checkpoint_sha256":args.expected_input_sha,"output_checkpoint":str(output),"output_checkpoint_sha256":sha256(staged),"compile_evidence":str(evidence),"compile_evidence_sha256":sha256(staged_evidence),"source_rows":rows,"unique_nonempty_factor_ids":len(ids),"changed_unique_rows":len(changed_identities),"migration_counts":dict(kinds),"compile_status_counts_all_rows_including_239_blank_ids":dict(counts),"execution_status_counts_all_rows_including_239_blank_ids":dict(execution_counts),"scope":"R18 exact technical recipes; changed formulas recompiled and prior execution evidence invalidated","recipe_sha256":args.expected_recipe_sha}
        staged_manifest=Path(tmp)/manifest.name; staged_manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        os.link(staged,output); os.link(staged_evidence,evidence); os.link(staged_manifest,manifest)
    print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return payload


if __name__=="__main__": main()
