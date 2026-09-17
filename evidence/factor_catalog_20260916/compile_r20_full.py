"""Identity-pinned streaming real-engine compilation; no data reads or migrations."""
import argparse, collections, csv, gzip, hashlib, json, sys, time
from pathlib import Path

SOURCE_SHA = "190a34ada8b380103e65f59b9cdf503f022c930838a0856e28824ab0b1e6d330"
def digest(path):
    with open(path,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--source-sha",default=SOURCE_SHA)
    ap.add_argument("--prefix",type=Path,required=True)
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--shards",type=int,default=1)
    ap.add_argument("--proposals",type=Path,nargs="*",default=[])
    args=ap.parse_args()
    if not 0<=args.shard<args.shards: raise ValueError("invalid shard")
    if digest(args.source)!=args.source_sha: raise ValueError("source hash mismatch")
    proposals={}
    for p in args.proposals:
        with (gzip.open(p,"rt") if p.suffix == ".gz" else p.open("r",encoding="utf-8")) as f:
            for line in f:
                r=json.loads(line); key=(int(r["source_row"]),r["id"])
                if key in proposals: raise ValueError("duplicate proposal")
                if r["current_formula"]!=r["before_formula"] and not r["changes"]:
                    raise ValueError("undocumented rewrite")
                proposals[key]=r
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/"evidence/factor_catalog_20260915"))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    paths=[root/p for p in (
        "factor_engine/api/dsl_parser.py","factor_engine/ir/analyzer.py",
        "factor_engine/ir/types.py","factor_engine/cleaned_operators/base.py",
        "factor_engine/cleaned_operators/operator_surface.py",
        "factor_engine/cleaned_operators/gemini_v2_common.py",
        "factor_engine/planner/canonicalize_params.py",
        "factor_engine/planner/lowerings/next_stage.py",
        "factor_engine/planner/lowerings/ashare.py",
        "evidence/factor_catalog_20260915/smoke_catalog.py",
        "evidence/factor_catalog_20260915/compile_catalog.py")]
    # Pin all engine/source implementation files, including dynamically registered kernels.
    paths = sorted(set(paths + list((root/"factor_engine").rglob("*.py")) + list((root/"data_access").rglob("*.py"))))
    paths.append(Path(__file__))
    hashes=lambda:{str(p.relative_to(root)):digest(p) for p in paths}
    before=hashes()
    parser,engine=build_runtime()
    output=Path(str(args.prefix)+".jsonl.gz")
    summary=Path(str(args.prefix)+".summary.json")
    if output.exists() or summary.exists(): raise FileExistsError(args.prefix)
    start=time.monotonic(); counts=collections.Counter(); used=set(); total=0
    with gzip.open(args.source,"rt",encoding="utf-8-sig",newline="") as src, gzip.open(output,"xt",encoding="utf-8") as dst:
        for row in csv.DictReader(src):
            if not row["id"]: continue
            index=total; total+=1
            if index%args.shards!=args.shard: continue
            key=(int(row["source_row"]),row["id"]); proposal=proposals.get(key)
            formula=row["current_formula"]; changes=[]
            if proposal:
                if proposal["before_formula"]!=formula: raise ValueError("proposal formula mismatch")
                formula=proposal["current_formula"]; changes=proposal["changes"]; used.add(key)
            from factor_engine.tools.catalog_r20_return_units import migrate_formula as repair_units
            formula, unit_notes = repair_units(formula)
            changes = list(changes) + unit_notes
            event={"source_row":key[0],"id":key[1],"before_formula":row["current_formula"],
                   "current_formula":formula,"changes":changes,"status":"COMPILED","error":""}
            try:
                expr=parser.parse(formula); bindings,failures=bind_fields(expr)
                event.update(bindings=bindings,binding_failures=failures)
                if failures: raise ValueError("FIELD_BINDING_FAILED: "+json.dumps(failures,ensure_ascii=False))
                engine.compile(Factor(name=key[1],expr=expr,source_expr=formula,surface="compat_research"))
            except Exception as exc:
                event.update(status="COMPILE_FAILED",error=type(exc).__name__+": "+str(exc))
            dst.write(json.dumps(event,ensure_ascii=False)+"\n")
            counts[event["status"]]+=1
            if sum(counts.values())%500==0:
                dst.flush()
                print(json.dumps({"shard":args.shard,"counts":counts,"seconds":time.monotonic()-start}),flush=True)
                # Drop per-engine plan caches; registry loading is cached.
                parser,engine=build_runtime()
    result={"source_sha256":args.source_sha,"output_sha256":digest(output),
            "shard":args.shard,"shards":args.shards,"source_factor_count":total,
            "proposal_count":len(used),"counts":dict(counts),"seconds":time.monotonic()-start,
            "scope":"real_engine_strict_field_bind_no_read_compile_not_execution",
            "code_hashes_before":before,"code_hashes_after":hashes(),
            "proposal_hashes":{str(p):digest(p) for p in args.proposals}}
    with summary.open("x") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
