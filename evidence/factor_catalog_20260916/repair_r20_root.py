"""Bounded compilation of root-owned additional signature corrections."""
import collections,gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"evidence/factor_catalog_20260915"))
from compile_catalog import build_runtime
from smoke_catalog import bind_fields
from factor_engine.api.factor import Factor
from factor_engine.tools.catalog_r20_root_recipes import migrate_formula
E=ROOT/"evidence/factor_catalog_20260916"

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True,type=Path);args=ap.parse_args()
    rows=[]
    for p in sorted(E.glob("r20-extra-s?.jsonl.gz")):
        for x in gzip.open(p,"rt"):
            r=json.loads(x)
            if r["status"]=="COMPILED" or "FIELD_BINDING_FAILED" in r["error"]:continue
            if "intra_" in r["error"] or "MinuteOpen,MinuteHigh,MinuteLow,MinuteClose,MinuteVolume,MinuteAmount,MinuteVwap" in r["current_formula"]:continue
            rows.append(r)
    prior={int(r["source_row"]):r for r in map(json.loads,gzip.open(E/"r20_original_context.jsonl.gz","rt"))}
    for x in open(E/"r20_ops_params_nonlegacy_failures.jsonl"):
        r=json.loads(x);r["before_formula"]=prior[int(r["source_row"])]["current_formula"];rows.append(r)
    parser,engine=build_runtime();counts=collections.Counter();start=time.monotonic()
    with gzip.open(args.output,"xt",encoding="utf-8") as out:
        for r in rows:
            from factor_engine.tools.catalog_r19_parameter_recipes import migrate_formula as old_params
            from factor_engine.tools.catalog_r19_remaining_params_recipes import migrate_formula as old_remaining
            formula=r["current_formula"]; notes=[]
            for migrate in (old_params,old_remaining,old_params,old_remaining,migrate_formula):
                formula,newnotes=migrate(formula)
                notes.extend(newnotes)
            rec={k:r[k] for k in ("source_row","id","before_formula")}
            rec.update(current_formula=formula,changes=r.get("changes",[])+notes,status="COMPILED",error="")
            try:
                expr=parser.parse(formula);bindings,failed=bind_fields(expr)
                rec.update(bindings=bindings,binding_failures=failed)
                if failed:raise ValueError("FIELD_BINDING_FAILED: "+json.dumps(failed,ensure_ascii=False))
                engine.compile(Factor(name=r["id"],expr=expr,source_expr=formula,surface="compat_research"))
            except Exception as exc:rec.update(status="COMPILE_FAILED",error=type(exc).__name__+": "+str(exc))
            counts[rec["status"]]+=1;out.write(json.dumps(rec,ensure_ascii=False)+"\n")
            if sum(counts.values())%100==0:out.flush();print(json.dumps({"counts":counts,"seconds":time.monotonic()-start}),flush=True)
    print(json.dumps({"counts":counts,"seconds":time.monotonic()-start}),flush=True)
if __name__=="__main__":main()
