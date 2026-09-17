"""Merge pinned compile-only records into a new catalog checkpoint.

Only compile status/reason/evidence fields are updated.  Static analysis,
execution evidence, formulas, and row identity remain byte-for-byte values from
the pinned source rows.  This is not execution or current-code certification.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import validate_review_source


_SCOPE = "compile_only_formula_matched_not_execution_or_code_certification"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _factor_family(formula: str) -> str:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return "unparsed"
    names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    if "fin_net_borrowing_cashflow" in names:
        return "fin_net_borrowing_cashflow"
    return names[0] if names else "literal_or_field"


def _load_evidence(path: Path, expected_sha: str):
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError("compile evidence SHA256 mismatch")
    records = {}
    status_counts = collections.Counter()
    code_hash_counts = collections.Counter()
    factor_counts = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evidence JSON at line {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"evidence line {line_number} is not an object")
            try:
                identity = (int(item["source_row"]), item.get("id") or "")
                formula = item["current_formula"]
                status = item["status"]
                code_hash = item["code_hash"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid compile evidence at line {line_number}") from exc
            if identity in records:
                raise ValueError(f"duplicate compile evidence identity {identity}")
            if not all(isinstance(value, str) and value for value in (formula, status, code_hash)):
                raise ValueError(f"incomplete compile evidence for {identity}")
            error = item.get("error") or ""
            if not isinstance(error, str):
                raise ValueError(f"compile error is not text for {identity}")
            records[identity] = item
            status_counts[status] += 1
            code_hash_counts[code_hash] += 1
            factor_counts[_factor_family(formula)] += 1
    if not records:
        raise ValueError("compile evidence is empty")
    return records, {
        "sha256": actual_sha,
        "records": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "code_hash_counts": dict(sorted(code_hash_counts.items())),
        "factor_counts": dict(sorted(factor_counts.items())),
    }


def reconcile(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint)
    evidence_path = Path(args.evidence)
    prefix = Path(args.output_prefix)
    output = Path(str(prefix) + ".csv.gz")
    manifest_path = Path(str(prefix) + ".manifest.json")
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    if output.resolve() == checkpoint.resolve():
        raise ValueError("output must differ from checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_validation = validate_review_source(
        checkpoint, args.expected_checkpoint_sha
    )
    evidence, evidence_summary = _load_evidence(
        evidence_path, args.expected_evidence_sha
    )
    remaining = set(evidence)
    output_compile_counts = collections.Counter()
    merged = 0

    with tempfile.TemporaryDirectory(prefix=".r12-compile-", dir=output.parent) as temp:
        staged_output = Path(temp) / output.name
        with gzip.open(checkpoint, "rt", encoding="utf-8-sig", newline="") as source, \
             gzip.open(staged_output, "xt", encoding="utf-8-sig", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            required_compile = {
                "compile_status", "compile_reason", "compile_evidence_file",
                "compile_validation_scope",
            }
            if not required_compile.issubset(reader.fieldnames or ()):
                raise ValueError("checkpoint lacks compile evidence fields")
            writer.writeheader()
            rows = 0
            for row in reader:
                rows += 1
                identity = (int(row["source_row"]), row.get("id") or "")
                item = evidence.get(identity)
                if item is not None:
                    if item["current_formula"] != row["current_formula"]:
                        raise ValueError(f"compile evidence formula mismatch for {identity}")
                    immutable_before = dict(row)
                    row["compile_status"] = item["status"]
                    row["compile_reason"] = item.get("error") or ""
                    row["compile_evidence_file"] = evidence_path.name
                    row["compile_validation_scope"] = (
                        f"{_SCOPE}; formula=current_formula; code_hash={item['code_hash']}"
                    )
                    allowed = required_compile
                    if any(
                        row[key] != value
                        for key, value in immutable_before.items()
                        if key not in allowed
                    ):
                        raise ValueError(f"non-compile field changed for {identity}")
                    remaining.remove(identity)
                    merged += 1
                output_compile_counts[row.get("compile_status") or ""] += 1
                writer.writerow(row)
        if remaining:
            raise ValueError(f"compile evidence rows absent from checkpoint: {len(remaining)}")
        if rows != source_validation["rows"]:
            raise ValueError("output row count differs from source checkpoint")
        with gzip.open(staged_output, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        final_checkpoint_sha = _sha256(checkpoint)
        if final_checkpoint_sha != args.expected_checkpoint_sha:
            raise ValueError("source checkpoint changed after validation")
        final_evidence_sha = _sha256(evidence_path)
        if final_evidence_sha != args.expected_evidence_sha:
            raise ValueError("compile evidence changed after validation")
        output_sha = _sha256(staged_output)
        manifest = {
            "input_checkpoint": str(checkpoint),
            "input_checkpoint_sha256": final_checkpoint_sha,
            "compile_evidence": str(evidence_path),
            "evidence_sha256": final_evidence_sha,
            "evidence_records": evidence_summary["records"],
            "evidence_status_counts": evidence_summary["status_counts"],
            "code_hash_counts": evidence_summary["code_hash_counts"],
            "code_hash_interpretation": (
                "per-record audit-time lowering/test snapshot; not a hash or "
                "certification of the current repository"
            ),
            "factor_counts": evidence_summary["factor_counts"],
            "merged_records": merged,
            "source_rows": source_validation["rows"],
            "unique_factor_ids": source_validation["unique_factor_ids"],
            "output_compile_status_counts": dict(sorted(output_compile_counts.items())),
            "output_checkpoint": str(output),
            "output_checkpoint_sha256": output_sha,
            "scope": _SCOPE,
            "unchanged_field_families": [
                "identity", "original_formula", "current_formula", "static", "execution"
            ],
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
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--expected-evidence-sha", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = reconcile(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
