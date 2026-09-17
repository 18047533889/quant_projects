from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path("/home/sunhaiwei/quant_projects")
E15 = ROOT / "evidence/factor_catalog_20260915"
E16 = ROOT / "evidence/factor_catalog_20260916"
SOURCE = E16 / "factor_catalog_review_checkpoint_r7_2_intraday_daily16546.csv.gz"
OUTPUT = E16 / "factor_catalog_review_checkpoint_r8_donchian_daily17460.csv.gz"
MANIFEST = E16 / "factor_catalog_review_checkpoint_r8_donchian_daily17460.csv.manifest.json"
VALIDATION = E16 / "factor_catalog_review_checkpoint_r8_donchian_daily17460.csv.validation.json"


def jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_record(item: dict, path: Path, scope: str) -> dict:
    return {
        "execution_status": item.get("status", "NOT_RUN"),
        "execution_reason": item.get("error") or "",
        "value_count": item.get("value_count", ""),
        "finite_count": item.get("finite_count", ""),
        "result_hash": item.get("result_hash", ""),
        "retry_after_batch_abort": item.get("retry_after_batch_abort", ""),
        "batch_abort_error_type": item.get("batch_abort_error_type", ""),
        "batch_abort_error": item.get("batch_abort_error", ""),
        "backend_path": json.dumps(item.get("backend_path"), ensure_ascii=False),
        "execution_evidence_file": path.name,
        "execution_validation_scope": scope,
        "execution_window": json.dumps(item.get("execution_window")) if item.get("execution_window") is not None else "",
        "execution_symbol_count": item.get("execution_symbol_count", ""),
        "executed_formula": item.get("executed_formula"),
    }


def main() -> None:
    for path in (OUTPUT, MANIFEST, VALIDATION):
        if path.exists():
            raise FileExistsError(path)

    r17_paths = sorted(E15.glob("resume-daily16546-external-r17-*.jsonl.gz"))
    if len(r17_paths) != 20:
        raise RuntimeError(f"expected 20 r17 chunks, got {len(r17_paths)}")

    ledger = [json.loads(line) for line in (E16 / "donchian-position-semantic-rerun-r3.ledger.jsonl").read_text().splitlines() if line]
    if len(ledger) != 14:
        raise RuntimeError(f"expected 14 r3 ledger entries, got {len(ledger)}")
    donchian_paths = [ROOT / entry["output"] for entry in ledger if entry["exit"] == 0]
    donchian_paths.append(E16 / "donchian-position-semantic-rerun-r4-offset100.jsonl.gz")
    if len(donchian_paths) != 13:
        raise RuntimeError(f"expected 13 successful Donchian chunks, got {len(donchian_paths)}")

    updates: dict[tuple[int, str], dict] = {}
    r17_counts = collections.Counter()
    for path in r17_paths:
        for item in jsonl(path):
            key = (int(item["source_row"]), item.get("id") or "")
            updates[key] = evidence_record(item, path, "closed_real_execution_r17_formula_matched")
            r17_counts[item.get("status", "NOT_RUN")] += 1

    donchian_counts = collections.Counter()
    donchian_keys: set[tuple[int, str]] = set()
    for path in donchian_paths:
        for item in jsonl(path):
            key = (int(item["source_row"]), item.get("id") or "")
            donchian_keys.add(key)
            updates[key] = evidence_record(item, path, "post_donchian_macro_fix_revision_bound")
            donchian_counts[item.get("status", "NOT_RUN")] += 1

    input_rows = list(jsonl(E16 / "donchian-position-semantic-rerun-input.jsonl.gz"))
    if len(input_rows) != 280:
        raise RuntimeError(f"expected 280 Donchian inputs, got {len(input_rows)}")
    crash_rows = input_rows[120:140]
    for item in crash_rows:
        key = (int(item["source_row"]), item.get("id") or "")
        donchian_keys.add(key)
        updates[key] = {
            "execution_status": "NATIVE_CRASH",
            "execution_reason": "bounded post-fix chunk exited -11; no result artifact promoted",
            "value_count": "",
            "finite_count": "",
            "result_hash": "",
            "retry_after_batch_abort": "",
            "batch_abort_error_type": "NATIVE_CRASH",
            "batch_abort_error": "process exit -11",
            "backend_path": "",
            "execution_evidence_file": "donchian-position-semantic-rerun-r3-120.log",
            "execution_validation_scope": "post_donchian_macro_fix_native_crash_no_result",
            "execution_window": "",
            "execution_symbol_count": "",
            "executed_formula": None,
        }
    donchian_counts["NATIVE_CRASH"] += len(crash_rows)
    if len(donchian_keys) != 280:
        raise RuntimeError(f"Donchian coverage is {len(donchian_keys)}, expected 280")

    counts = collections.Counter()
    seen_donchian: set[tuple[int, str]] = set()
    redesign_counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix=".checkpoint-r8-", dir=E16) as tempdir:
        staged = Path(tempdir) / OUTPUT.name
        with gzip.open(SOURCE, "rt", encoding="utf-8-sig", newline="") as source, gzip.open(staged, "xt", encoding="utf-8-sig", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (int(row["source_row"]), row.get("id") or "")
                replacement = updates.get(key)
                if replacement:
                    replacement = dict(replacement)
                    expected = replacement.pop("executed_formula", None)
                    if expected is not None and expected != row["current_formula"]:
                        raise RuntimeError(f"formula mismatch for {key}")
                    row.update(replacement)
                if "SEMANTIC_REDESIGN" in row.get("migration_changes", ""):
                    redesign_counts[row["static_status"]] += 1
                if key in donchian_keys:
                    seen_donchian.add(key)
                    if row["execution_validation_scope"] == "historical_formula_matched_not_current_code_certification":
                        raise RuntimeError(f"stale Donchian evidence retained for {key}")
                writer.writerow(row)
                counts["rows"] += 1
                counts["factor_ids"] += bool(row.get("id"))
                counts["static:" + row["static_status"]] += 1
                counts["compile:" + row["compile_status"]] += 1
                counts["execution:" + row["execution_status"]] += 1
        if seen_donchian != donchian_keys:
            raise RuntimeError(f"missing Donchian rows: {len(donchian_keys - seen_donchian)}")
        os.link(staged, OUTPUT)

    evidence_paths = r17_paths + donchian_paths + [E16 / "donchian-position-semantic-rerun-r3-120.log"]
    evidence = [{"path": str(path.relative_to(ROOT)), "records": sum(1 for _ in jsonl(path)) if path.suffix == ".gz" else None, "sha256": sha256(path)} for path in evidence_paths]
    validation = {
        "rows": counts["rows"],
        "factor_ids": counts["factor_ids"],
        "donchian_rows": len(seen_donchian),
        "donchian_status_counts": dict(donchian_counts),
        "donchian_old_evidence_rows": 0,
        "r17_status_counts": dict(r17_counts),
        "intraday_semantic_redesign_counts": dict(redesign_counts),
        "output_sha256": sha256(OUTPUT),
        "passed": counts["rows"] == 114132 and counts["factor_ids"] == 113893 and len(seen_donchian) == 280,
    }
    manifest = {
        "counts": dict(counts),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "evidence": evidence,
        "donchian_revision": json.loads((E16 / "donchian-position-semantic-rerun-r4-offset100.ledger.json").read_text())["source_sha256"],
        "intraday_semantic_redesign": {
            "approval_state": "approved_and_applied_opt_in_redesign",
            "authoritative_audit": "evidence/factor_catalog_20260916/full-parse-r9-intraday-redesign.jsonl.gz",
            "authoritative_audit_sha256": sha256(E16 / "full-parse-r9-intraday-redesign.jsonl.gz"),
            "counts": dict(redesign_counts),
            "historical_preapproval_ledger": "evidence/factor_catalog_20260916/intraday-limit-semantic-conflict-ledger-r1.json",
        },
        "scope": "checkpoint_not_final_all_factor_execution",
        "validation_scope": "mixed_historical_and_explicit_revision_bound_evidence",
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
