"""Aggregate approved compile chunks and reconcile them into an R13 checkpoint."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from reconcile_r12_compile import reconcile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict) -> tuple[int, str]:
    return int(row["source_row"]), row.get("id") or ""


def _unsupported_targets(path: Path, expected_sha: str) -> dict[tuple[int, str], str]:
    if _sha256(path) != expected_sha:
        raise ValueError("source checkpoint SHA256 mismatch")
    targets = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not (row.get("compile_reason") or "").startswith("Unsupported function:"):
                continue
            try:
                ast.parse(row["current_formula"], mode="eval")
            except SyntaxError:
                continue
            identity = _identity(row)
            if identity in targets:
                raise ValueError(f"duplicate source identity {identity}")
            targets[identity] = row["current_formula"]
    return targets


def _current_formulas(path: Path, expected_sha: str) -> dict[tuple[int, str], str]:
    if _sha256(path) != expected_sha:
        raise ValueError("R13 checkpoint SHA256 mismatch")
    formulas = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = _identity(row)
            if identity in formulas:
                raise ValueError(f"duplicate R13 identity {identity}")
            formulas[identity] = row["current_formula"]
    return formulas


def _load_approved_chunks(
    manifests: list[Path], approved_hashes: list[str], expected_source_sha: str,
    expected_code_hash: str,
):
    if len(manifests) != len(approved_hashes) or not manifests:
        raise ValueError("chunk manifests and approved hashes must be non-empty and paired")
    records = {}
    summaries = []
    for manifest_path, approved_sha in zip(manifests, approved_hashes):
        actual_manifest_sha = _sha256(manifest_path)
        if actual_manifest_sha != approved_sha:
            raise ValueError(f"chunk manifest SHA256 mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_sha256") != expected_source_sha:
            raise ValueError(f"chunk source checkpoint SHA256 mismatch: {manifest_path}")
        if manifest.get("scope") != "current_code_no_read_compile_only_not_execution":
            raise ValueError(f"chunk scope is not approved compile-only evidence: {manifest_path}")
        if manifest.get("code_hash") != expected_code_hash:
            raise ValueError(f"chunk code hash mismatch: {manifest_path}")
        evidence_path = Path(manifest["evidence"])
        if not evidence_path.is_absolute():
            evidence_path = manifest_path.parent / evidence_path
        actual_evidence_sha = _sha256(evidence_path)
        if actual_evidence_sha != manifest["evidence_sha256"]:
            raise ValueError(f"chunk evidence SHA256 mismatch: {evidence_path}")
        bindings_path = Path(manifest["bindings"])
        if not bindings_path.is_absolute():
            bindings_path = manifest_path.parent / bindings_path
        actual_bindings_sha = _sha256(bindings_path)
        if actual_bindings_sha != manifest["bindings_sha256"]:
            raise ValueError(f"chunk bindings SHA256 mismatch: {bindings_path}")
        count = 0
        with gzip.open(evidence_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("code_hash") != expected_code_hash:
                    raise ValueError(f"record code hash mismatch at line {line_number}")
                identity = _identity(item)
                if identity in records:
                    raise ValueError(f"duplicate chunk evidence identity {identity}")
                records[identity] = item
                count += 1
        if count != int(manifest["records"]):
            raise ValueError(f"chunk record count mismatch: {manifest_path}")
        summaries.append({
            "manifest": str(manifest_path),
            "manifest_sha256": actual_manifest_sha,
            "evidence": str(evidence_path),
            "evidence_sha256": actual_evidence_sha,
            "records": count,
            "bindings": str(bindings_path),
            "bindings_sha256": actual_bindings_sha,
        })
    return records, summaries


def aggregate_and_reconcile(args: argparse.Namespace) -> dict:
    source = Path(args.source_checkpoint)
    checkpoint = Path(args.checkpoint)
    aggregate = Path(args.aggregate_evidence)
    prefix = Path(args.output_prefix)
    if aggregate.exists():
        raise FileExistsError(aggregate)
    targets = _unsupported_targets(source, args.expected_source_sha)
    if len(targets) != args.expected_targets:
        raise ValueError(f"unsupported target count {len(targets)} != {args.expected_targets}")
    current = _current_formulas(checkpoint, args.expected_checkpoint_sha)
    records, chunks = _load_approved_chunks(
        [Path(item) for item in args.chunk_manifest],
        args.expected_chunk_manifest_sha,
        args.expected_source_sha,
        args.expected_code_hash,
    )
    if set(records) != set(targets):
        missing = len(set(targets) - set(records))
        extra = len(set(records) - set(targets))
        raise ValueError(f"chunk identities do not exactly cover targets: missing={missing}, extra={extra}")
    excluded = {
        identity: "current_formula_changed_since_compile_source"
        for identity, old_formula in targets.items()
        if current.get(identity) != old_formula
    }
    if len(excluded) != args.expected_excluded:
        raise ValueError(f"excluded count {len(excluded)} != {args.expected_excluded}")
    included = []
    for identity, item in records.items():
        if item.get("current_formula") != targets[identity]:
            raise ValueError(f"chunk formula differs from source target for {identity}")
        if identity in excluded:
            continue
        if current.get(identity) != item["current_formula"]:
            raise ValueError(f"chunk formula differs from R13 current formula for {identity}")
        included.append(item)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(aggregate, "xt", encoding="utf-8") as stream:
            for item in sorted(included, key=_identity):
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        aggregate_sha = _sha256(aggregate)
        reconcile_args = argparse.Namespace(
            checkpoint=checkpoint,
            expected_checkpoint_sha=args.expected_checkpoint_sha,
            evidence=aggregate,
            expected_evidence_sha=aggregate_sha,
            output_prefix=prefix,
        )
        result = reconcile(reconcile_args)
    except Exception:
        aggregate.unlink(missing_ok=True)
        raise
    result["aggregation"] = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": args.expected_source_sha,
        "target_records": len(targets),
        "included_records": len(included),
        "excluded_records": len(excluded),
        "excluded_reason_counts": {"current_formula_changed_since_compile_source": len(excluded)},
        "excluded_identities": [list(identity) for identity in sorted(excluded)],
        "approved_chunks": chunks,
        "aggregate_evidence": str(aggregate),
        "aggregate_evidence_sha256": aggregate_sha,
    }
    result["code_hash_interpretation"] = (
        "per-record audit-time declared source-member snapshot, see approved chunk "
        "manifests; not whole current repository certification"
    )
    manifest_path = Path(str(prefix) + ".manifest.json")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temp_manifest = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_manifest, manifest_path)
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha", required=True)
    parser.add_argument("--chunk-manifest", action="append", required=True)
    parser.add_argument("--expected-chunk-manifest-sha", action="append", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--expected-excluded", type=int, required=True)
    parser.add_argument("--aggregate-evidence", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    result = aggregate_and_reconcile(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
