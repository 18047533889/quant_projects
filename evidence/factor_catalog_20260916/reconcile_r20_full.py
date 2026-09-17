"""Publish a no-overwrite catalog checkpoint from complete, pinned compile evidence."""
import argparse, collections, csv, gzip, hashlib, json, os
from pathlib import Path

SOURCE_SHA="190a34ada8b380103e65f59b9cdf503f022c930838a0856e28824ab0b1e6d330"
CLEAR=("value_count","finite_count","result_hash","retry_after_batch_abort",
       "batch_abort_error_type","batch_abort_error","backend_path",
       "execution_evidence_file","execution_validation_scope","execution_window",
       "execution_symbol_count")

def digest(path):
    with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def update_row(row,event,evidence):
    if event["before_formula"]!=row["current_formula"]:raise ValueError("formula identity mismatch")
    if event["status"] not in {"COMPILED","COMPILE_FAILED"}:raise ValueError("nonterminal compile")
    if event["status"]=="COMPILE_FAILED" and not event.get("error"):raise ValueError("missing failure reason")
    if event["status"]=="COMPILED" and (event.get("error") or event.get("binding_failures")):
        raise ValueError("false successful status")
    changed=event["current_formula"]!=row["current_formula"]
    notes=list(json.loads(row.get("migration_changes") or "[]"))
    if changed and not event["changes"]:raise ValueError("undocumented formula edit")
    if changed or (event["status"]=="COMPILE_FAILED" and row.get("execution_status") in {"EXECUTED","EXECUTED_ALL_NONFINITE"}):
        prior={k:row.get(k,"") for k in ("execution_status","execution_reason",*CLEAR)}
        notes.append("R20_EXECUTION_INVALIDATED: "+json.dumps(prior,ensure_ascii=False,sort_keys=True))
        for k in CLEAR:row[k]=""
        row["execution_status"]="NOT_RUN" if event["status"]=="COMPILED" else "COMPILE_FAILED"
        row["execution_reason"]="DSL changed; small-sample execution pending" if event["status"]=="COMPILED" else event["error"]
    elif row.get("execution_status")=="COMPILE_FAILED" and event["status"]=="COMPILED":
        row["execution_status"]="NOT_RUN"
        row["execution_reason"]="Current-code compilation repaired; execution pending"
        for k in CLEAR:row[k]=""
    row["current_formula"]=event["current_formula"]
    row["compile_status"]=event["status"];row["compile_reason"]=event["error"]
    row["compile_evidence_file"]=evidence
    row["compile_validation_scope"]="R20 pinned current-code real-engine no-read compile; not execution certification"
    notes.extend(event["changes"])
    row["migration_changes"]=json.dumps(notes,ensure_ascii=False)
    if "bindings" in event:
        row["bindings"]=json.dumps(event["bindings"],ensure_ascii=False)
        if not event.get("binding_failures"):
            row["current_fields"]="|".join(sorted({b["table"]+"."+b["column"] for b in event["bindings"]}))
            row["current_tables"]="|".join(sorted({b["table"] for b in event["bindings"]}))
            row["static_status"]="PARSED_FIELDS_BOUND";row["static_reason"]=""
        else:
            row["static_status"]="FIELD_BINDING_FAILED"
            row["static_reason"]=json.dumps(event["binding_failures"],ensure_ascii=False)
            row["current_fields"]="";row["current_tables"]=""
    elif changed:
        row["bindings"]="[]";row["current_fields"]="";row["current_tables"]=""
        row["static_status"]="PARSE_FAILED";row["static_reason"]=event["error"]
    return changed


def apply_runtime_abort(row,event,evidence):
    if (int(row["source_row"]),row["id"])!=(int(event["source_row"]),event["id"]):
        raise ValueError("runtime identity mismatch")
    if event.get("executed_formula")!=row["current_formula"]:
        raise ValueError("runtime formula mismatch")
    if event.get("status")!="BATCH_ABORTED" or not event.get("error"):
        raise ValueError("expected explicit batch-abort evidence")
    notes=list(json.loads(row.get("migration_changes") or "[]"))
    prior={k:row.get(k,"") for k in ("execution_status","execution_reason",*CLEAR)}
    notes.append("R20_PRIOR_EXECUTION_BEFORE_AUTO_TEST: "+json.dumps(prior,ensure_ascii=False,sort_keys=True))
    for k in CLEAR:row[k]=""
    row["execution_status"]="BATCH_ABORTED"
    row["execution_reason"]="Auto batch aborted, not an individual factor diagnosis: "+event.get("error_type","")+": "+event["error"]
    row["execution_evidence_file"]=evidence
    row["execution_validation_scope"]="R20 current-code real-data run_many(auto), one batch of 16; physical-plan rejection; no successful materialization"
    row["execution_window"]=json.dumps(event.get("execution_window",[]))
    row["execution_symbol_count"]=str(event.get("execution_symbol_count",""))
    row["migration_changes"]=json.dumps(notes,ensure_ascii=False)

def events(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        previous=0
        for line in f:
            row=json.loads(line); number=int(row["source_row"])
            if number<=previous: raise ValueError("unordered/duplicate shard row")
            previous=number
            yield row

def main():
    import heapq
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--prefixes",nargs="+",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--aborted-smoke",type=Path)
    args=ap.parse_args()
    if digest(args.source)!=SOURCE_SHA: raise ValueError("source mismatch")
    paths=[Path(str(p)+".jsonl.gz") for p in args.prefixes]
    summaries=[json.loads(Path(str(p)+".summary.json").read_text()) for p in args.prefixes]
    counts_expected=collections.Counter()
    hashes=summaries[0]["code_hashes_before"]
    total_shards=len(summaries)
    if sorted(s["shard"] for s in summaries)!=list(range(total_shards)): raise ValueError("incomplete shards")
    for p,s in zip(paths,summaries):
        if s["shards"]!=total_shards or s["source_factor_count"]!=113893:
            raise ValueError("shard scope mismatch")
        if s["source_sha256"]!=SOURCE_SHA or s["output_sha256"]!=digest(p):
            raise ValueError("evidence hash mismatch")
        if s["code_hashes_before"]!=hashes or s["code_hashes_after"]!=hashes:
            raise ValueError("code drift: rerun final compile on stable code")
        counts_expected.update(s["counts"])
    if sum(counts_expected.values())!=113893: raise ValueError("incomplete full compilation")
    aborts={};abort_sha=None;aborts_used=set()
    if args.aborted_smoke:
        abort_sha=digest(args.aborted_smoke)
        smoke_summary=json.loads(Path(str(args.aborted_smoke).replace(".jsonl.gz",".summary.json")).read_text())
        if smoke_summary.get("requested_backend")!="auto" or not smoke_summary.get("batch_only") or smoke_summary.get("batch_size")!=16 or smoke_summary.get("counts")!={"BATCH_ABORTED":16}:
            raise ValueError("unexpected runtime smoke scope")
        with gzip.open(args.aborted_smoke,"rt") as f:
            for event in map(json.loads,f):
                key=(int(event["source_row"]),event["id"])
                if key in aborts:raise ValueError("duplicate runtime evidence")
                aborts[key]=event
        if len(aborts)!=16:raise ValueError("incomplete runtime smoke")
    stream=heapq.merge(*(events(p) for p in paths),key=lambda r:int(r["source_row"]))
    manifest=Path(str(args.output)+".manifest.json")
    part=Path(str(args.output)+".part")
    if any(p.exists() for p in (args.output,manifest,part)): raise FileExistsError(args.output)
    counts=collections.Counter(); execution=collections.Counter(); changed=0; rows=0
    seen_ids=set()
    try:
        with gzip.open(args.source,"rt",encoding="utf-8-sig",newline="") as src, gzip.open(part,"xt",encoding="utf-8-sig",newline="") as dst:
            reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                rows+=1
                if row["id"]:
                    event=next(stream,None)
                    if event is None or (int(event["source_row"]),event["id"])!=(int(row["source_row"]),row["id"]):
                        raise ValueError("evidence coverage/identity mismatch")
                    if row["id"] in seen_ids: raise ValueError("duplicate factor ID")
                    seen_ids.add(row["id"])
                    changed+=update_row(row,event,"|".join(map(str,paths)))
                    counts[row["compile_status"]]+=1
                runtime_key=(int(row["source_row"]),row["id"])
                if runtime_key in aborts:
                    apply_runtime_abort(row,aborts[runtime_key],str(args.aborted_smoke))
                    aborts_used.add(runtime_key)
                execution[row["execution_status"]]+=1
                writer.writerow(row)
        if next(stream,None) is not None or rows!=114132 or counts!=counts_expected:
            raise ValueError("coverage mismatch")
        if aborts_used!=set(aborts):raise ValueError("runtime evidence IDs not covered")
        if args.aborted_smoke and digest(args.aborted_smoke)!=abort_sha:raise ValueError("runtime evidence changed")
        if digest(args.source)!=SOURCE_SHA: raise ValueError("source changed")
        for p,s in zip(paths,summaries):
            if digest(p)!=s["output_sha256"]: raise ValueError("evidence changed")
        result={"source_sha256":SOURCE_SHA,"output_sha256":digest(part),
                "rows":rows,"changed_this_round":changed,
                "runtime_abort_evidence_sha256":abort_sha,"runtime_rows_updated":len(aborts_used),
                "compile_counts_nonempty_ids":dict(counts),"execution_counts":dict(execution),
                "scope":"All 113893 factor DSLs strict field binding and real-engine compilation on current code; not full execution certification",
                "code_hashes":hashes,"shard_summaries":summaries}
        os.link(part,args.output)
        with manifest.open("x") as f: json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k not in {"shard_summaries","code_hashes"}},ensure_ascii=False))
    finally:
        if part.exists(): part.unlink()

if __name__=="__main__": main()
