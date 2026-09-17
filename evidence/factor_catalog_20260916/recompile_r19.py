"""Bounded, no-read current-code compile evidence for R19 repairs."""
import argparse, collections, csv, gzip, hashlib, importlib, json, sys, time
from pathlib import Path

def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--modules", nargs="*", default=[])
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    source=root/"evidence/factor_catalog_20260916/factor_catalog_review_r18c.csv.gz"
    assert digest(source)=="ccf6a0effe8a09471491b94d3845767309bb4ebc380bf5cd3c016b680144c7c3"
    sys.path.insert(0,str(root/"evidence/factor_catalog_20260915"))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    modules=[importlib.import_module(m) for m in args.modules]
    migrations=[m.migrate_formula for m in modules]
    code_files=[Path(m.__file__) for m in modules]+[
        root/"factor_engine/api/dsl_parser.py",
        root/"factor_engine/backend/long_frame.py",
        root/"factor_engine/cleaned_operators/price_volume/candle_pattern_engine_v2.py",
        root/"factor_engine/cleaned_operators/common/polars_candle_patterns.py",
        root/"factor_engine/ir/analyzer.py",
        root/"factor_engine/ir/types.py",
        root/"factor_engine/cleaned_operators/gemini_v2_common.py",
        root/"factor_engine/cleaned_operators/base.py",
        root/"factor_engine/cleaned_operators/markov_dynamics.py",
        root/"factor_engine/cleaned_operators/sequence_complexity.py",
        root/"factor_engine/cleaned_operators/state_event.py",
        root/"factor_engine/cleaned_operators/fiscal_event_ops.py",
        root/"factor_engine/planner/lowerings/next_stage.py",
        root/"factor_engine/planner/lowerings/ashare.py",
        root/"factor_engine/planner/canonicalize_params.py",
        Path(__file__),
    ]
    code_hashes={str(p.relative_to(root)):digest(p) for p in code_files}
    from factor_engine.tools.catalog_field_recipes import migrate_catalog_field_formula
    from factor_engine.tools.catalog_sketch_migration import convert_complete_parameter_sketch
    parser,engine=build_runtime()
    prefix=Path(args.prefix)
    output=Path(str(prefix)+".jsonl.gz")
    summary=Path(str(prefix)+".summary.json")
    if output.exists() or summary.exists(): raise FileExistsError(prefix)
    counts=collections.Counter(); changed=0; start=time.monotonic()
    with gzip.open(source,"rt",encoding="utf-8-sig",newline="") as f, gzip.open(output,"xt",encoding="utf-8") as out:
        for row in csv.DictReader(f):
            if not row["id"] or row["compile_status"]!="COMPILE_FAILED": continue
            if args.limit and sum(counts.values())>=args.limit: break
            formula=row["current_formula"]; changes=[]
            if migrations:
                sketch=convert_complete_parameter_sketch(formula)
                if sketch.converted:
                    formula=sketch.formula
                    changes.append("EXACT_SKETCH_SYNTAX: complete parameter assignment sketch converted to keyword call")
                try:
                    field_fix=migrate_catalog_field_formula(
                        formula,logic=row.get("logic",""),tables=row.get("original_tables",""),
                        enabled=True,
                    )
                    formula=field_fix.formula
                    changes.extend(field_fix.changes)
                except SyntaxError:
                    pass
            for migrate in migrations:
                formula,notes=migrate(formula,row.get("logic",""))
                changes.extend(notes)
            record={"source_row":int(row["source_row"]),"id":row["id"],
                    "before_formula":row["current_formula"],"current_formula":formula,
                    "changes":changes,"status":"COMPILED","error":""}
            changed+=formula!=row["current_formula"]
            try:
                expr=parser.parse(formula)
                bindings,failures=bind_fields(expr)
                record["bindings"]=bindings
                record["binding_failures"]=failures
                if failures: raise ValueError("FIELD_BINDING_FAILED: "+json.dumps(failures,ensure_ascii=False))
                engine.compile(Factor(name=row["id"],expr=expr,source_expr=formula,surface="compat_research"))
            except Exception as exc:
                record.update(status="COMPILE_FAILED",error=type(exc).__name__+": "+str(exc))
            counts[record["status"]]+=1
            out.write(json.dumps(record,ensure_ascii=False)+"\n")
            if sum(counts.values())%100==0:
                out.flush()
                print(json.dumps({"counts":counts,"changed":changed,"seconds":time.monotonic()-start}),flush=True)
    result={"source_sha256":digest(source),"output_sha256":digest(output),
            "counts":counts,"changed":changed,"seconds":time.monotonic()-start,
            "modules":args.modules,"scope":"real_engine_no_read_compile_not_execution",
            "code_hashes_before":code_hashes,
            "code_hashes_after":{str(p.relative_to(root)):digest(p) for p in code_files}}
    with summary.open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=="__main__": main()
