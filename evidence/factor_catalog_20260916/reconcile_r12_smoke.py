"""Merge only closed, formula-matched R12 smoke evidence into a pinned checkpoint.

This creates a new checkpoint and manifest.  It does not certify the current
code revision, and it never reads unpublished temporary smoke shards.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import validate_review_source


_SCOPE = "formula_matched_research_smoke_not_code_certification"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_closed_ledger(path: Path) -> tuple[list[dict], dict]:
    events = []
    with path.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid ledger JSON at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"ledger line {line_number} is not an object")
            events.append(event)
    if not events or events[-1].get("event") != "sweep_closed":
        raise ValueError("ledger is not closed")
    closed = events[-1]
    if closed.get("requested_count_completed") is not True:
        raise ValueError("ledger closed without completing its requested count")
    batches = [event for event in events[:-1] if event.get("event") == "batch_complete"]
    if not batches:
        raise ValueError("closed ledger contains no completed batches")
    outputs = [event.get("output") for event in batches]
    if not all(isinstance(output, str) and output for output in outputs):
        raise ValueError("completed batch has no output path")
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate completed batch output")
    processed = sum(int(event.get("processed", -1)) for event in batches)
    if processed != int(closed.get("processed", -2)):
        raise ValueError("ledger batch counts do not match closed processed count")
    if processed != int(closed.get("requested", -3)):
        raise ValueError("ledger did not close the requested record count")
    return batches, closed


def _evidence_record(item: dict, path: Path) -> dict[str, object]:
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
        "execution_validation_scope": _SCOPE,
        "execution_window": (
            json.dumps(item.get("execution_window"), ensure_ascii=False)
            if item.get("execution_window") is not None else ""
        ),
        "execution_symbol_count": item.get("execution_symbol_count", ""),
    }


def _load_updates(ledger: Path, batches: list[dict]) -> tuple[dict, list[dict]]:
    updates: dict[tuple[int, str], tuple[dict, str]] = {}
    evidence = []
    for batch in batches:
        raw_path = Path(batch["output"])
        path = raw_path if raw_path.is_absolute() else ledger.parent / raw_path
        if path.name.endswith(".partial") or path.suffixes[-2:] != [".jsonl", ".gz"]:
            raise ValueError(f"completed batch is not a closed jsonl gzip: {path}")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid evidence JSON in {path.name}:{line_number}"
                    ) from exc
                try:
                    key = (int(item["source_row"]), item.get("id") or "")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid evidence identity in {path.name}") from exc
                if key in updates:
                    raise ValueError(f"duplicate evidence identity {key}")
                formula = item.get("executed_formula")
                if not isinstance(formula, str) or not formula:
                    raise ValueError(f"missing executed_formula for {key}")
                updates[key] = (item, path.name)
                count += 1
        if count != int(batch.get("processed", -1)):
            raise ValueError(f"completed batch record count mismatch: {path.name}")
        evidence.append({
            "path": str(path), "records": count, "sha256": _sha256(path),
        })
    return updates, evidence


def reconcile(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    ledger = Path(args.ledger)
    prefix = Path(args.output_prefix)
    output = Path(str(prefix) + ".csv.gz")
    manifest_path = Path(str(prefix) + ".manifest.json")
    if checkpoint.resolve() in {output.resolve(), manifest_path.resolve()}:
        raise ValueError("output must be distinct from the source checkpoint")
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_validation = validate_review_source(checkpoint, args.expected_sha)
    batches, closed = _load_closed_ledger(ledger)
    updates, evidence = _load_updates(ledger, batches)
    remaining = set(updates)
    status_counts = collections.Counter()
    merged = 0

    with tempfile.TemporaryDirectory(prefix=".r12-reconcile-", dir=output.parent) as temp:
        staged_output = Path(temp) / output.name
        with gzip.open(checkpoint, "rt", encoding="utf-8-sig", newline="") as source, \
             gzip.open(staged_output, "xt", encoding="utf-8-sig", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            rows = 0
            for row in reader:
                rows += 1
                key = (int(row["source_row"]), row.get("id") or "")
                replacement = updates.get(key)
                if replacement is not None:
                    item, evidence_name = replacement
                    if item.get("id") != row.get("id"):
                        raise ValueError(f"factor ID mismatch for source row {key[0]}")
                    if item["executed_formula"] != row["current_formula"]:
                        raise ValueError(f"executed formula mismatch for {key}")
                    evidence_path = next(
                        Path(entry["path"]) for entry in evidence
                        if Path(entry["path"]).name == evidence_name
                    )
                    immutable = tuple(
                        row[name] for name in (
                            "source_row", "id", "original_formula", "current_formula"
                        )
                    )
                    row.update(_evidence_record(item, evidence_path))
                    if immutable != tuple(
                        row[name] for name in (
                            "source_row", "id", "original_formula", "current_formula"
                        )
                    ):
                        raise ValueError(f"immutable checkpoint fields changed for {key}")
                    remaining.remove(key)
                    merged += 1
                status_counts[row.get("execution_status") or ""] += 1
                writer.writerow(row)
        if remaining:
            raise ValueError(f"evidence rows absent from checkpoint: {len(remaining)}")
        if rows != source_validation["rows"]:
            raise ValueError("output row count differs from source checkpoint")
        # Exhaust the newly written gzip before publishing it.
        with gzip.open(staged_output, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        output_sha = _sha256(staged_output)
        manifest = {
            "input_checkpoint": str(checkpoint),
            "input_checkpoint_sha256": _sha256(checkpoint),
            "input_ledger": str(ledger),
            "input_ledger_sha256": _sha256(ledger),
            "closed_evidence": evidence,
            "source_rows": source_validation["rows"],
            "unique_factor_ids": source_validation["unique_factor_ids"],
            "merged_records": merged,
            "status_counts": dict(sorted(status_counts.items())),
            "output_checkpoint": str(output),
            "output_checkpoint_sha256": output_sha,
            "scope": _SCOPE,
            "ledger_closed_processed": closed["processed"],
        }
        staged_manifest = Path(temp) / manifest_path.name
        with staged_manifest.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        os.link(staged_output, output)
        os.link(staged_manifest, manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = reconcile(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
