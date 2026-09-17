"""Apply already reviewed R19 exact-shape migrations to newly exposed failures."""
import argparse,collections,gzip,importlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"evidence/factor_catalog_20260915"))
from compile_catalog import build_runtime
from smoke_catalog import bind_fields
from factor_engine.api.factor import Factor

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,nargs="+",required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    with gzip.open(ROOT/"evidence/factor_catalog_20260916/r20_original_context.jsonl.gz","rt") as f:
        assigned={int(json.loads(x)["source_row"]) for x in f}
    summary=json.loads((ROOT/"evidence/factor_catalog_20260916/r19-combined-r5.summary.json").read_text())
    migrations=[importlib.import_module(m).migrate_formula for m in summary["modules"]]
    parser,engine=build_runtime();counts=collections.Counter();started=time.monotonic()
    with gzip.open(args.output,"xt",encoding="utf-8") as out:
        for p in args.inputs:
            with gzip.open(p,"rt") as f:
                for line in f:
                    prior=json.loads(line)
                    if prior["status"]=="COMPILED" or prior["source_row"] in assigned:continue
                    formula=prior["current_formula"];notes=[]
                    for migrate in migrations:
                        formula,changes=migrate(formula,"")
                        notes.extend(changes)
                    record={k:prior[k] for k in ("source_row","id","before_formula")}
                    record.update(current_formula=formula,changes=notes,status="COMPILED",error="",previous_error=prior["error"])
                    try:
                        expr=parser.parse(formula);bindings,failures=bind_fields(expr)
                        record.update(bindings=bindings,binding_failures=failures)
                        if failures:raise ValueError("FIELD_BINDING_FAILED: "+json.dumps(failures,ensure_ascii=False))
                        engine.compile(Factor(name=record["id"],expr=expr,source_expr=formula,surface="compat_research"))
                    except Exception as exc:
                        record.update(status="COMPILE_FAILED",error=type(exc).__name__+": "+str(exc))
                    counts[record["status"]]+=1
                    out.write(json.dumps(record,ensure_ascii=False)+"\n")
                    if sum(counts.values())%100==0:
                        out.flush();print(json.dumps({"counts":counts,"seconds":time.monotonic()-started}),flush=True)
    print(json.dumps({"counts":counts,"seconds":time.monotonic()-started}),flush=True)
if __name__=="__main__":main()
