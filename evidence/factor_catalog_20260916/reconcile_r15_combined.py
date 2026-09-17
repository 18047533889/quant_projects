"""One-pass reconciliation of closed compile-only and execution evidence."""
from __future__ import annotations

import argparse, collections, csv, gzip, hashlib, json, os, tempfile
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import validate_review_source


def _sha(path: Path) -> str:
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def _identity(item): return int(item["source_row"]), item.get("id") or ""


def _summary_path(path: Path) -> Path:
    return path.with_suffix(".summary.json")


def _execution_summary_path(path: Path) -> Path:
    suffix = ".output.jsonl.gz"
    if path.name.endswith(suffix):
        return path.with_name(path.name[:-len(suffix)] + ".output.summary.json")
    return _summary_path(path)


def _counts(value: str) -> dict[str, int]:
    return {
        key: int(raw)
        for item in value.split(",") if item
        for key, raw in (item.split("=", 1),)
    }


def _load_compile(paths, source_sha):
    records, evidence = {}, []
    for path in paths:
        if path.suffixes[-2:] != [".jsonl", ".gz"] or path.name.endswith(".partial"):
            raise ValueError(f"compile shard is not closed gzip: {path}")
        summary_path = _summary_path(path)
        summary = json.loads(summary_path.read_text())
        actual_sha = _sha(path)
        if not summary.get("complete") or summary.get("processed") != summary.get("requested"):
            raise ValueError(f"compile shard is not closed: {path}")
        if summary.get("input_sha256") != source_sha or summary.get("output_sha256") != actual_sha:
            raise ValueError(f"compile shard hash/source mismatch: {path}")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line); ident = _identity(item)
                if ident in records: raise ValueError(f"duplicate compile identity {ident}")
                formula = item.get("executed_formula")
                if item.get("source_csv_sha256") != source_sha:
                    raise ValueError(f"compile source SHA mismatch {ident}")
                if item.get("formula_sha256") != hashlib.sha256(formula.encode()).hexdigest():
                    raise ValueError(f"compile formula hash mismatch {ident}")
                if item.get("compile_status") not in {"COMPILED", "COMPILE_FAILED"}:
                    raise ValueError(f"invalid compile status {ident}")
                records[ident] = item; count += 1
        if count != summary["processed"]: raise ValueError(f"compile count mismatch: {path}")
        evidence.append({"path":str(path),"sha256":actual_sha,"summary":str(summary_path),
                         "summary_sha256":_sha(summary_path),"records":count})
    return records, evidence


def _load_execution(paths, expected_hashes=None):
    records, evidence = {}, []
    if expected_hashes is not None and len(expected_hashes) != len(paths):
        raise ValueError("execution hash pin count mismatch")
    for index, path in enumerate(paths):
        if path.suffixes[-2:] != [".jsonl", ".gz"] or path.name.endswith(".partial"):
            raise ValueError(f"execution shard is not closed gzip: {path}")
        summary_path = _execution_summary_path(path)
        summary = json.loads(summary_path.read_text())
        actual_sha = _sha(path)
        if expected_hashes is not None and actual_sha != expected_hashes[index]:
            raise ValueError(f"execution pinned hash mismatch: {path}")
        if summary.get("complete") is not True or type(summary.get("processed")) is not int or summary["processed"] <= 0 or summary.get("processed") != summary.get("requested_limit"):
            raise ValueError(f"execution shard is not closed: {path}")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item=json.loads(line); ident=_identity(item)
                if ident in records: raise ValueError(f"duplicate execution identity {ident}")
                if item.get("status") != "EXECUTED":
                    raise ValueError(f"non-success execution evidence {ident}")
                if not item.get("executed_formula"):
                    raise ValueError(f"missing execution formula {ident}")
                values, finite = item.get("value_count"), item.get("finite_count")
                if type(values) is not int or type(finite) is not int or not 0 < finite <= values:
                    raise ValueError(f"invalid successful execution counts {ident}")
                result_hash = item.get("result_hash")
                if not isinstance(result_hash, str) or len(result_hash) != 64 or any(c not in "0123456789abcdef" for c in result_hash):
                    raise ValueError(f"invalid execution result hash {ident}")
                records[ident]=(item,path.name); count += 1
        if count != summary["processed"]: raise ValueError(f"execution count mismatch: {path}")
        if summary.get("counts") != {"EXECUTED": count}:
            raise ValueError(f"execution summary status counts mismatch: {path}")
        if _sha(path) != actual_sha:
            raise ValueError(f"execution shard changed during read: {path}")
        evidence.append({"path":str(path),"sha256":actual_sha,"summary":str(summary_path),
                         "summary_sha256":_sha(summary_path),"records":count})
    return records, evidence


def reconcile(args):
    source=Path(args.checkpoint); prefix=Path(args.output_prefix)
    output=Path(str(prefix)+".csv.gz"); manifest_path=Path(str(prefix)+".manifest.json")
    for p in (output,manifest_path):
        if p.exists(): raise FileExistsError(p)
    validation=validate_review_source(source,args.expected_checkpoint_sha)
    compile_records,compile_evidence=_load_compile([Path(x) for x in args.compile_evidence],args.expected_checkpoint_sha)
    execution_records,execution_evidence=_load_execution([Path(x) for x in args.execution_evidence], args.execution_sha256)
    if len(compile_records)!=args.expected_compile_records: raise ValueError("compile coverage count mismatch")
    if len(execution_records)!=args.expected_execution_records: raise ValueError("execution coverage count mismatch")
    compile_targets=set(); seen=set(); compile_counts=collections.Counter(); execution_counts=collections.Counter()
    with tempfile.TemporaryDirectory(prefix=".r15-combined-",dir=prefix.parent) as tmp:
        staged=Path(tmp)/output.name
        with gzip.open(source,"rt",encoding="utf-8-sig",newline="") as src, gzip.open(staged,"xt",encoding="utf-8-sig",newline="") as dst:
            reader=csv.DictReader(src); writer=csv.DictWriter(dst,fieldnames=reader.fieldnames); writer.writeheader(); rows=0
            for row in reader:
                rows+=1; ident=(int(row["source_row"]),row.get("id") or "")
                if ident in seen: raise ValueError(f"duplicate checkpoint identity {ident}")
                seen.add(ident)
                if row.get("compile_status")=="NOT_RUN" and ident[1]: compile_targets.add(ident)
                comp=compile_records.get(ident)
                if comp:
                    if comp["executed_formula"]!=row["current_formula"]: raise ValueError(f"compile formula mismatch {ident}")
                    row["compile_status"]=comp["compile_status"]
                    row["compile_reason"]=comp.get("error") or ""
                    row["compile_evidence_file"]="r15-compile-not-run-shards"
                    row["compile_validation_scope"]="compile_only_no_data_read_not_execution_evidence; formula=current_formula"
                exe=execution_records.get(ident)
                if exe:
                    item,name=exe
                    if item["executed_formula"]!=row["current_formula"]: raise ValueError(f"execution formula mismatch {ident}")
                    row.update(execution_status="EXECUTED",execution_reason="",value_count=item.get("value_count",""),
                               finite_count=item.get("finite_count",""),result_hash=item.get("result_hash",""),
                               retry_after_batch_abort=item.get("retry_after_batch_abort",""),
                               batch_abort_error_type=item.get("batch_abort_error_type",""),batch_abort_error=item.get("batch_abort_error",""),
                               backend_path=json.dumps(item.get("backend_path"),ensure_ascii=False),execution_evidence_file=name,
                               execution_validation_scope="formula_matched_real_data_research_smoke_not_production_or_current_code_certification",
                               execution_window=json.dumps(item.get("execution_window"),ensure_ascii=False) if item.get("execution_window") is not None else "",
                               execution_symbol_count=item.get("execution_symbol_count",""))
                compile_counts[row.get("compile_status") or ""]+=1; execution_counts[row.get("execution_status") or ""]+=1; writer.writerow(row)
        if compile_targets!=set(compile_records): raise ValueError("compile identities do not exactly cover NOT_RUN targets")
        if not set(execution_records).issubset(seen): raise ValueError("execution identity absent from checkpoint")
        if rows!=validation["rows"]: raise ValueError("row count changed")
        with gzip.open(staged,"rb") as f:
            while f.read(1024*1024): pass
        if _sha(source)!=args.expected_checkpoint_sha: raise ValueError("source changed during reconciliation")
        if dict(compile_counts)!=_counts(args.expected_compile_counts): raise ValueError(f"unexpected compile counts {compile_counts}")
        if dict(execution_counts)!=_counts(args.expected_execution_counts): raise ValueError(f"unexpected execution counts {execution_counts}")
        manifest={"input_checkpoint":str(source),"input_checkpoint_sha256":args.expected_checkpoint_sha,
                  "source_rows":rows,"unique_factor_ids":validation["unique_factor_ids"],"compile_records":len(compile_records),
                  "execution_records":len(execution_records),"compile_evidence":compile_evidence,"execution_evidence":execution_evidence,
                  "output_compile_status_counts":dict(sorted(compile_counts.items())),"output_execution_status_counts":dict(sorted(execution_counts.items())),
                  "output_checkpoint":str(output),"output_checkpoint_sha256":_sha(staged),
                  "scope":"combined_formula_matched_compile_only_and_research_execution_evidence",
                  "telemetry_caveats":["Williams auto40 r8 predates transfer telemetry correction: transfer actual_bytes may equal predicted_bytes and is not measured payload; numerical output evidence is retained."]}
        sm=Path(tmp)/manifest_path.name; sm.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        os.link(staged,output); os.link(sm,manifest_path)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--expected-checkpoint-sha",required=True)
    p.add_argument("--compile-evidence",action="append",required=True); p.add_argument("--execution-evidence",action="append",required=True)
    p.add_argument("--execution-sha256", action="append", required=True, help="Pinned SHA256 for each execution shard, in matching order")
    p.add_argument("--expected-compile-records",type=int,required=True); p.add_argument("--expected-execution-records",type=int,required=True)
    p.add_argument("--expected-compile-counts",required=True); p.add_argument("--expected-execution-counts",required=True); p.add_argument("--output-prefix",required=True)
    args=p.parse_args(argv); result=reconcile(args); print(json.dumps(result,ensure_ascii=False,sort_keys=True)); return result

if __name__=="__main__": main()
