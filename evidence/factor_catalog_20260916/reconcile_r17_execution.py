"""Reconcile one closed, fully pinned mixed-outcome R17 execution batch."""
from __future__ import annotations

import argparse, collections, csv, gzip, hashlib, json, os, tempfile
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import validate_review_source

SUCCESS = "EXECUTED"
ALL_NONFINITE = "EXECUTED_ALL_NONFINITE"
FAILURES = {"BATCH_ABORTED", "EXECUTION_FAILED", "PREPARE_FAILED", "NATIVE_CRASH"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(row: dict) -> tuple[int, str]:
    return int(row["source_row"]), row.get("id") or ""


def load_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _hex_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _counts(value: str) -> dict[str, int]:
    return {key: int(raw) for part in value.split(",") if part for key, raw in (part.split("=", 1),)}


def _list(value):
    return value if isinstance(value, list) else [value]


def reconcile(args):
    checkpoint=Path(args.checkpoint)
    group_values=[_list(getattr(args,name)) for name in ("input","input_sha256","execution","execution_sha256","summary","summary_sha256","expected_records","expected_batch_counts")]
    lengths={len(values) for values in group_values}
    if len(lengths)!=1: raise ValueError("execution group argument counts do not align")
    groups=[]; pins=[(checkpoint,args.checkpoint_sha256,"checkpoint")]
    for request,input_pin,execution,execution_pin,summary_path,summary_pin,expected_records,expected_counts in zip(*group_values):
        request=Path(request); execution=Path(execution); summary_path=Path(summary_path)
        groups.append((request,input_pin,execution,execution_pin,summary_path,summary_pin,int(expected_records),_counts(expected_counts)))
        pins.extend(((request,input_pin,"input"),(execution,execution_pin,"execution"),(summary_path,summary_pin,"summary")))
    for path,pin,label in pins:
        if sha256(path)!=pin: raise ValueError(f"pinned {label} hash mismatch")
    validation=validate_review_source(checkpoint,args.checkpoint_sha256)
    request_map={}; result_map={}; group_for_identity={}; actual=collections.Counter(); evidence_groups=[]
    for group_index,(request,input_pin,execution,execution_pin,summary_path,summary_pin,expected_records,expected_batch_counts) in enumerate(groups):
        requests=load_jsonl(request); results=load_jsonl(execution)
        if len(requests)!=expected_records or len(results)!=expected_records: raise ValueError("record count mismatch")
        local_requests={identity(row):row for row in requests}; local_results={identity(row):row for row in results}
        if len(local_requests)!=len(requests) or len(local_results)!=len(results): raise ValueError("duplicate execution identity")
        if set(local_requests)!=set(local_results): raise ValueError("input/execution identities mismatch")
        if set(request_map)&set(local_requests): raise ValueError("duplicate execution identity across groups")
        local_counts=collections.Counter()
        for key,item in local_results.items():
            status=item.get("status"); local_counts[status]+=1; actual[status]+=1
            if status not in {SUCCESS,ALL_NONFINITE,*FAILURES}: raise ValueError(f"non-terminal execution status {key}: {status}")
            if item.get("executed_formula")!=local_requests[key].get("formula"): raise ValueError(f"execution formula mismatch {key}")
            if status==SUCCESS:
                values,finite=item.get("value_count"),item.get("finite_count")
                if type(values) is not int or type(finite) is not int or not 0 < finite <= values: raise ValueError(f"invalid successful values {key}")
                if not _hex_digest(item.get("result_hash")): raise ValueError(f"invalid successful result hash {key}")
            elif status==ALL_NONFINITE:
                values,finite=item.get("value_count"),item.get("finite_count")
                if type(values) is not int or values <= 0 or type(finite) is not int or finite != 0:
                    raise ValueError(f"invalid all-nonfinite values {key}")
                if not _hex_digest(item.get("result_hash")):
                    raise ValueError(f"invalid all-nonfinite result hash {key}")
            elif not isinstance(item.get("error_type"),str) or not item["error_type"].strip() or not isinstance(item.get("error"),str) or not item["error"].strip():
                raise ValueError(f"failure lacks actual error evidence {key}")
            group_for_identity[key]=group_index
        if dict(local_counts)!=expected_batch_counts: raise ValueError(f"execution counts mismatch {local_counts}")
        summary=json.loads(summary_path.read_text())
        if summary.get("complete") is not True or summary.get("processed")!=expected_records or summary.get("requested_limit")!=expected_records or summary.get("counts")!=expected_batch_counts:
            raise ValueError("execution summary mismatch")
        if Path(summary.get("input","")).name!=request.name or Path(summary.get("output","")).name!=execution.name: raise ValueError("summary input/output mismatch")
        request_map.update(local_requests); result_map.update(local_results)
        evidence_groups.append({"input":str(request),"input_sha256":input_pin,"execution":str(execution),"execution_sha256":execution_pin,"summary":str(summary_path),"summary_sha256":summary_pin,"records":expected_records,"counts":dict(sorted(local_counts.items())),"window":summary.get("window"),"symbol_count":len(summary.get("symbols") or [])})

    output=Path(str(args.output_prefix)+".csv.gz"); manifest=Path(str(args.output_prefix)+".manifest.json")
    if output.exists() or manifest.exists(): raise FileExistsError(output if output.exists() else manifest)
    expected_catalog_counts=_counts(args.expected_catalog_counts); seen=set(); catalog_counts=collections.Counter()
    with tempfile.TemporaryDirectory(prefix=".r17-execution-",dir=output.parent) as tmp:
        staged=Path(tmp)/output.name
        with gzip.open(checkpoint,"rt",encoding="utf-8-sig",newline="") as src,gzip.open(staged,"xt",encoding="utf-8-sig",newline="") as dst:
            reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames); writer.writeheader(); rows=0
            for row in reader:
                rows+=1; key=identity(row); item=result_map.get(key)
                if item is not None:
                    group=evidence_groups[group_for_identity[key]]; execution=Path(group["execution"])
                    formula=request_map[key]["formula"]
                    if row.get("current_formula")!=formula or item["executed_formula"]!=formula: raise ValueError(f"checkpoint formula mismatch {key}")
                    seen.add(key); old={name:row.get(name,"") for name in ("execution_status","execution_reason","execution_evidence_file","execution_validation_scope","value_count","finite_count","result_hash")}
                    changes=json.loads(row.get("migration_changes") or "[]")
                    changes.append("R17_EXECUTION_RECONCILE: prior="+json.dumps(old,ensure_ascii=False,sort_keys=True)+"; evidence="+execution.name)
                    row["migration_changes"]=json.dumps(changes,ensure_ascii=False)
                    status=item["status"]; row["execution_status"]=status; row["execution_evidence_file"]=execution.name
                    row["execution_validation_scope"]="pinned_source_request_output_summary; identity_and_current_formula_matched; research_execution_not_production_certification"
                    row["execution_window"]=json.dumps(item.get("execution_window",group.get("window")),ensure_ascii=False)
                    row["execution_symbol_count"]=item.get("execution_symbol_count",group.get("symbol_count",0))
                    row["retry_after_batch_abort"]=item.get("retry_after_batch_abort","")
                    row["batch_abort_error_type"]=item.get("batch_abort_error_type","")
                    row["batch_abort_error"]=item.get("batch_abort_error","")
                    if status==SUCCESS:
                        row.update(execution_reason="",value_count=item["value_count"],finite_count=item["finite_count"],result_hash=item["result_hash"],backend_path=json.dumps(item.get("backend_path"),ensure_ascii=False))
                    elif status==ALL_NONFINITE:
                        row.update(execution_reason="execution completed with zero finite values",value_count=item["value_count"],finite_count=item["finite_count"],result_hash=item["result_hash"],backend_path=json.dumps(item.get("backend_path"),ensure_ascii=False))
                    else:
                        row.update(execution_reason=f"{item['error_type']}: {item['error']}",value_count="",finite_count="",result_hash="",backend_path="")
                catalog_counts[row.get("execution_status") or ""]+=1; writer.writerow(row)
        if seen!=set(result_map): raise ValueError("checkpoint does not cover all execution identities")
        if rows!=validation["rows"] or dict(catalog_counts)!=expected_catalog_counts: raise ValueError(f"catalog coverage/status mismatch rows={rows} counts={catalog_counts}")
        for path,pin,label in pins:
            if sha256(path)!=pin: raise ValueError(f"{label} changed during reconciliation")
        payload={"input_checkpoint":str(checkpoint),"input_checkpoint_sha256":args.checkpoint_sha256,"execution_groups":evidence_groups,"execution_records":len(result_map),"batch_execution_status_counts":dict(sorted(actual.items())),"source_rows":rows,"unique_factor_ids":validation["unique_factor_ids"],"output_execution_status_counts":dict(sorted(catalog_counts.items())),"output_checkpoint":str(output),"output_checkpoint_sha256":sha256(staged),"scope":"multi_group_pinned_mixed_terminal_research_execution_evidence; no NOT_RUN target remains"}
        staged_manifest=Path(tmp)/manifest.name; staged_manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        os.link(staged,output); os.link(staged_manifest,manifest)
    return payload


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--checkpoint-sha256",required=True); p.add_argument("--expected-catalog-counts",required=True)
    for name in ("input","execution","summary"): p.add_argument("--"+name,required=True,type=Path,action="append")
    for name in ("input-sha256","execution-sha256","summary-sha256","expected-batch-counts"): p.add_argument("--"+name,required=True,action="append")
    p.add_argument("--expected-records",required=True,type=int,action="append"); p.add_argument("--output-prefix",required=True,type=Path)
    args=p.parse_args(argv); payload=reconcile(args); print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return payload


if __name__=="__main__": main()
