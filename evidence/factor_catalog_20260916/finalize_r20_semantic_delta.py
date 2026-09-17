"""Apply independently recompiled semantic deltas to complete frozen-code evidence."""
import argparse,collections,csv,gzip,hashlib,itertools,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"evidence/factor_catalog_20260915"))
from reconcile_r20_full import update_row
csv.field_size_limit(32*1024*1024)
SOURCE_SHA="190a34ada8b380103e65f59b9cdf503f022c930838a0856e28824ab0b1e6d330"
def digest(p):
    with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):
    with (gzip.open(p,"rt") if p.suffix==".gz" else p.open()) as f:
        yield from map(json.loads,f)
def verify_code(hashes):
    actual={str(p.relative_to(ROOT)) for directory in ("factor_engine","data_access") for p in (ROOT/directory).rglob("*.py")}
    expected={p for p in hashes if p.startswith(("factor_engine/","data_access/"))}
    if actual!=expected:raise ValueError("engine source set changed after full compile")
    for p,h in hashes.items():
        if digest(ROOT/p)!=h:raise ValueError("code drift: "+p)
def validate_delta(base,old,proposal,expected_previous):
    key=(int(base["source_row"]),base["id"])
    if key!=(int(old["source_row"]),old["id"]) or key!=(int(proposal["source_row"]),proposal["id"]):
        raise ValueError("delta identity mismatch")
    if proposal["before_formula"]!=old["current_formula"]:raise ValueError("delta original formula mismatch")
    if base["current_formula"]!=expected_previous:raise ValueError("delta predecessor mismatch")
    if not proposal.get("changes"):raise ValueError("undocumented delta")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--proposals",type=Path,nargs="+",required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    E=ROOT/"evidence/factor_catalog_20260916"
    original=E/"factor_catalog_review_r19_final.csv.gz"
    merged_path=E/"r20-merged-final.jsonl.gz"
    merged_sha="3e178337ed59211c1a66dd196ccd61f3ba559ce1f5cff14cce0530e9137e09b5"
    manifest_path=Path(str(args.base)+".manifest.json")
    spec=json.loads(manifest_path.read_text());hashes=spec["code_hashes"]
    base_sha=digest(args.base);manifest_sha=digest(manifest_path)
    if base_sha!=spec["output_sha256"] or digest(original)!=SOURCE_SHA or digest(merged_path)!=merged_sha:
        raise ValueError("input hash mismatch")
    verify_code(hashes)
    proposals={};pins={}
    for p in args.proposals:
        pins[str(p)]=digest(p)
        for r in read(p):
            key=(int(r["source_row"]),r["id"])
            if key in proposals:raise ValueError("duplicate delta")
            proposals[key]=r
    merged={(int(r["source_row"]),r["id"]):r for r in read(merged_path)}
    if not set(proposals)<=set(merged):raise ValueError("delta outside reviewed set")
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    from factor_engine.tools.catalog_r20_return_units import migrate_formula
    parser,engine=build_runtime();events={};failures=[]
    for key,proposal in sorted(proposals.items()):
        formula,unit_notes=migrate_formula(proposal["current_formula"])
        expr=parser.parse(formula);bindings,bad=bind_fields(expr)
        if bad:raise ValueError((key,bad))
        try:engine.compile(Factor(name=key[1],expr=expr,source_expr=formula,surface="compat_research"))
        except Exception as exc:
            failures.append({"id":key[1],"error":type(exc).__name__+": "+str(exc)});continue
        previous,unused=migrate_formula(merged[key]["current_formula"])
        old_notes=list(merged[key]["changes"]);all_notes=list(proposal["changes"])
        if all_notes[:len(old_notes)]!=old_notes:raise ValueError("delta dropped predecessor notes")
        notes=all_notes[len(old_notes):]+unit_notes
        if formula!=previous and not notes:raise ValueError("undocumented semantic edit")
        events[key]={"source_row":key[0],"id":key[1],"before_formula":previous,"current_formula":formula,
                     "changes":notes,"status":"COMPILED","error":"","bindings":bindings,"binding_failures":[]}
    if failures:raise ValueError(json.dumps(failures,ensure_ascii=False))
    verify_code(hashes)
    part=Path(str(args.output)+".part");out_manifest=Path(str(args.output)+".manifest.json")
    evidence=Path(str(args.output)+".delta.jsonl.gz")
    if any(p.exists() for p in (args.output,part,out_manifest,evidence)):raise FileExistsError(args.output)
    compiled=collections.Counter();execution=collections.Counter();changed=0;rows=0;seen=set();actual_delta_changes=0
    try:
        with gzip.open(args.base,"rt",encoding="utf-8-sig",newline="") as bf,gzip.open(original,"rt",encoding="utf-8-sig",newline="") as of,gzip.open(part,"xt",encoding="utf-8-sig",newline="") as dst:
            br=csv.DictReader(bf);orr=csv.DictReader(of);writer=csv.DictWriter(dst,fieldnames=br.fieldnames);writer.writeheader()
            for base,old in itertools.zip_longest(br,orr):
                if base is None or old is None:raise ValueError("row coverage mismatch")
                rows+=1
                key=(int(base["source_row"]),base["id"])
                if key!=(int(old["source_row"]),old["id"]):raise ValueError("source row alignment")
                if key in events:
                    event=events[key]
                    validate_delta(base,old,proposals[key],event["before_formula"])
                    old_evidence=base.get("compile_evidence_file","")
                    actual_delta_changes+=update_row(base,event,old_evidence+"|"+str(evidence))
                    base["compile_validation_scope"]="R20 complete frozen-code compile plus hash-matched real-engine semantic delta compilation; not execution certification"
                    seen.add(key)
                if base["id"]:
                    compiled[base["compile_status"]]+=1
                    changed+=base["current_formula"]!=old["current_formula"]
                execution[base["execution_status"]]+=1;writer.writerow(base)
        if rows!=114132 or compiled!={"COMPILED":113893} or seen!=set(proposals):
            raise ValueError(("final coverage/compile mismatch",rows,compiled,len(seen)))
        if digest(args.base)!=base_sha or digest(manifest_path)!=manifest_sha or digest(original)!=SOURCE_SHA:
            raise ValueError("inputs changed during finalize")
        for p,h in pins.items():
            if digest(p)!=h:raise ValueError("proposals changed during finalize")
        verify_code(hashes)
        with gzip.open(evidence,"xt") as f:
            for key in sorted(events):f.write(json.dumps(events[key],ensure_ascii=False)+"\n")
        result={"source_sha256":SOURCE_SHA,"output_sha256":digest(part),"rows":rows,
                "changed_this_round":changed,"compile_counts_nonempty_ids":dict(compiled),
                "execution_counts":dict(execution),"code_hashes":hashes,
                "base_manifest":str(manifest_path),"base_manifest_sha256":manifest_sha,
                "base_checkpoint_sha256":base_sha,"delta_proposal_hashes":pins,
                "delta_evidence":str(evidence),"delta_evidence_sha256":digest(evidence),
                "delta_compiled":len(events),"delta_formula_changes":actual_delta_changes,
                "finalizer_sha256":digest(Path(__file__)),
                "scope":"All 113893 final formulas real-engine compiled on identical source hashes, complete base compilation plus semantic delta recompile; not full execution"}
        os.link(part,args.output)
        with out_manifest.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!="code_hashes"},ensure_ascii=False))
    finally:
        if part.exists():part.unlink()
if __name__=="__main__":main()
