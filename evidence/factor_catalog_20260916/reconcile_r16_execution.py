"""Reconcile one closed, pinned execution shard into an immutable checkpoint."""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, os, tempfile
from pathlib import Path
from factor_engine.tools.catalog_review_evidence import validate_review_source


def sha(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def ident(row):
    return int(row["source_row"]), row.get("id") or ""


def load_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--input-sha256", required=True)
    p.add_argument("--execution", required=True, type=Path)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--expected-records", required=True, type=int)
    p.add_argument("--expected-execution-counts", required=True)
    p.add_argument("--output-prefix", required=True, type=Path)
    a = p.parse_args()
    validation = validate_review_source(a.checkpoint, a.checkpoint_sha256)
    if sha(a.input) != a.input_sha256 or sha(a.execution) != a.execution_sha256:
        raise ValueError("pinned input/execution hash mismatch")
    input_rows = load_jsonl(a.input)
    executed = load_jsonl(a.execution)
    if len(input_rows) != a.expected_records or len(executed) != a.expected_records:
        raise ValueError("record count mismatch")
    inputs = {ident(row): row for row in input_rows}
    records = {ident(row): row for row in executed}
    if len(inputs) != a.expected_records or set(inputs) != set(records):
        raise ValueError("input/execution identities mismatch or duplicate")
    for key, row in records.items():
        if row.get("status") != "EXECUTED" or row.get("executed_formula") != inputs[key].get("formula"):
            raise ValueError(f"execution status/formula mismatch {key}")
        values, finite = row.get("value_count"), row.get("finite_count")
        if type(values) is not int or type(finite) is not int or not 0 < finite <= values:
            raise ValueError(f"invalid finite counts {key}")
        result_hash = row.get("result_hash")
        if not isinstance(result_hash, str) or len(result_hash) != 64:
            raise ValueError(f"invalid result hash {key}")
    summary_path = a.execution.with_name(a.execution.name.replace(".output.jsonl.gz", ".output.summary.json"))
    summary = json.loads(summary_path.read_text())
    if summary.get("complete") is not True or summary.get("processed") != a.expected_records or summary.get("counts") != {"EXECUTED": a.expected_records}:
        raise ValueError("execution summary mismatch")
    output = Path(str(a.output_prefix) + ".csv.gz")
    manifest = Path(str(a.output_prefix) + ".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(output if output.exists() else manifest)
    expected_counts = dict(item.split("=", 1) for item in a.expected_execution_counts.split(","))
    expected_counts = {k: int(v) for k, v in expected_counts.items()}
    seen, counts = set(), collections.Counter()
    with tempfile.TemporaryDirectory(prefix=".r16-execution-", dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        with gzip.open(a.checkpoint, "rt", encoding="utf-8-sig", newline="") as src, gzip.open(staged, "xt", encoding="utf-8-sig", newline="") as dst:
            reader = csv.DictReader(src); writer = csv.DictWriter(dst, fieldnames=reader.fieldnames); writer.writeheader(); rows = 0
            for row in reader:
                rows += 1; key = ident(row)
                if key in records:
                    item = records[key]
                    if row["current_formula"] != inputs[key]["formula"] or row["current_formula"] != item["executed_formula"]:
                        raise ValueError(f"checkpoint formula mismatch {key}")
                    seen.add(key)
                    row.update(
                        execution_status="EXECUTED", execution_reason="",
                        value_count=item["value_count"], finite_count=item["finite_count"],
                        result_hash=item["result_hash"],
                        retry_after_batch_abort=item.get("retry_after_batch_abort", ""),
                        batch_abort_error_type=item.get("batch_abort_error_type", ""),
                        batch_abort_error=item.get("batch_abort_error", ""),
                        backend_path=json.dumps(item.get("backend_path"), ensure_ascii=False),
                        execution_evidence_file=a.execution.name,
                        execution_validation_scope="formula_and_input_identity_matched_real_data_research_smoke_not_production_certification",
                        execution_window=json.dumps(item.get("execution_window"), ensure_ascii=False),
                        execution_symbol_count=item.get("execution_symbol_count", ""),
                    )
                counts[row.get("execution_status") or ""] += 1
                writer.writerow(row)
        if seen != set(records) or rows != validation["rows"] or dict(counts) != expected_counts:
            raise ValueError(f"coverage/row/status mismatch seen={len(seen)} rows={rows} counts={counts}")
        if sha(a.checkpoint) != a.checkpoint_sha256 or sha(a.input) != a.input_sha256 or sha(a.execution) != a.execution_sha256:
            raise ValueError("source evidence changed during reconciliation")
        payload = {
            "input_checkpoint": str(a.checkpoint), "input_checkpoint_sha256": a.checkpoint_sha256,
            "input_execution_request": str(a.input), "input_execution_request_sha256": a.input_sha256,
            "execution_evidence": str(a.execution), "execution_evidence_sha256": a.execution_sha256,
            "execution_summary": str(summary_path), "execution_summary_sha256": sha(summary_path),
            "execution_records": len(records), "source_rows": rows,
            "unique_factor_ids": validation["unique_factor_ids"],
            "output_execution_status_counts": dict(sorted(counts.items())),
            "output_checkpoint": str(output), "output_checkpoint_sha256": sha(staged),
            "scope": "formula_input_identity_and_hash_matched_research_execution_evidence_only",
        }
        staged_manifest = Path(tmp) / manifest.name
        staged_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.link(staged, output); os.link(staged_manifest, manifest)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
