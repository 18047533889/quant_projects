"""Publish a no-overwrite catalog checkpoint from complete, pinned compile evidence."""
import argparse, collections, csv, gzip, hashlib, json, os
from pathlib import Path

SOURCE_SHA="ccf6a0effe8a09471491b94d3845767309bb4ebc380bf5cd3c016b680144c7c3"
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
        notes.append("R19_EXECUTION_INVALIDATED: "+json.dumps(prior,ensure_ascii=False,sort_keys=True))
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
    row["compile_validation_scope"]="R19 pinned current-code real-engine no-read compile; not execution certification"
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

def main():
    ap=argparse.ArgumentParser()
    for key in ("source","evidence","summary","output"):ap.add_argument("--"+key,required=True,type=Path)
    args=ap.parse_args()
    if digest(args.source)!=SOURCE_SHA:raise ValueError("source hash mismatch")
    summary=json.loads(args.summary.read_text())
    if summary["source_sha256"]!=SOURCE_SHA or summary["output_sha256"]!=digest(args.evidence):
        raise ValueError("compile evidence hash mismatch")
    if summary["code_hashes_before"]!=summary["code_hashes_after"]:raise ValueError("compile code changed during run")
    if sum(summary["counts"].values())!=4212:raise ValueError("incomplete 4212 audit")
    selected={};ev_counts=collections.Counter()
    with gzip.open(args.evidence,"rt") as f:
        for line in f:
            event=json.loads(line);key=(int(event["source_row"]),event["id"])
            if key in selected:raise ValueError("duplicate identity")
            selected[key]=event;ev_counts[event["status"]]+=1
    if dict(ev_counts)!=summary["counts"]:raise ValueError("summary counts mismatch")
    manifest=Path(str(args.output)+".manifest.json")
    part=Path(str(args.output)+".part")
    if args.output.exists() or manifest.exists() or part.exists():raise FileExistsError(args.output)
    counts=collections.Counter();execution=collections.Counter();changed=0;rows=0
    try:
        with gzip.open(args.source,"rt",encoding="utf-8-sig",newline="") as src, gzip.open(part,"xt",encoding="utf-8-sig",newline="") as dst:
            reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                rows+=1;key=(int(row["source_row"]),row["id"])
                event=selected.pop(key,None)
                if event is not None:
                    if row["compile_status"]!="COMPILE_FAILED":raise ValueError("selection changed")
                    changed+=update_row(row,event,str(args.evidence))
                elif row["id"] and row["compile_status"]=="COMPILE_FAILED":raise ValueError("uncovered failed row")
                if row["id"]:counts[row["compile_status"]]+=1
                execution[row["execution_status"]]+=1
                writer.writerow(row)
        if selected or rows!=114132 or sum(counts.values())!=113893:raise ValueError("coverage mismatch")
        if counts["COMPILED"]!=109681+ev_counts["COMPILED"]:raise ValueError("compiled reconciliation mismatch")
        if digest(args.source)!=SOURCE_SHA or digest(args.evidence)!=summary["output_sha256"]:raise ValueError("input changed")
        result={"source_sha256":SOURCE_SHA,"compile_evidence_sha256":digest(args.evidence),
                "output_sha256":digest(part),"rows":rows,"changed_this_round":changed,
                "compile_counts_nonempty_ids":dict(counts),"execution_counts":dict(execution),
                "scope":"Only 4212 prior failures current-code recompiled; other compile and execution evidence remain historical",
                "code_hashes":summary["code_hashes_after"]}
        os.link(part,args.output)
        with manifest.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False))
    finally:
        if part.exists():part.unlink()

if __name__=="__main__":main()
