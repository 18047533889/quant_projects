"""Bounded current-code NoRead recompilation of parseable Unsupported rows."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "d9668cd615ea28c9008159a6cd84bc99c9e1d29e38e04d459b3c4f0d2abb75d5"
CODE_PATHS = (
    "factor_engine/api/dsl_parser.py",
    "factor_engine/ir/analyzer.py",
    "factor_engine/runtime/engine.py",
    "factor_engine/cleaned_operators/registry.py",
    "factor_engine/tools/catalog_r13_expression_recipes.py",
    "evidence/factor_catalog_20260915/compile_catalog.py",
    "evidence/factor_catalog_20260915/smoke_catalog.py",
    "evidence/factor_catalog_20260916/recompile_r13_unsupported.py",
)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _snapshot_hash(root: Path) -> tuple[str, dict[str, str]]:
    members = {name: _file_hash(root / name) for name in CODE_PATHS}
    payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), members


def _identity_hash(row: dict[str, str]) -> str:
    payload = json.dumps(
        [int(row["source_row"]), row.get("id") or "", row["current_formula"]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0 or args.limit > 200:
        parser.error("start must be non-negative and limit must be in 1..200")
    if _file_hash(args.checkpoint) != args.expected_source_sha256:
        raise ValueError("checkpoint SHA256 mismatch")

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "evidence/factor_catalog_20260915"))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor

    code_hash, code_members = _snapshot_hash(root)
    evidence_path = Path(str(args.output_prefix) + ".jsonl.gz")
    bindings_path = Path(str(args.output_prefix) + ".bindings.jsonl.gz")
    manifest_path = Path(str(args.output_prefix) + ".manifest.json")
    for path in (evidence_path, bindings_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    dsl_parser, engine = build_runtime()
    seen = selected = emitted = 0
    counts: Counter[str] = Counter()
    with gzip.open(args.checkpoint, "rt", encoding="utf-8-sig", newline="") as source, \
         gzip.open(evidence_path, "xt", encoding="utf-8") as evidence, \
         gzip.open(bindings_path, "xt", encoding="utf-8") as binding_stream:
        for row in csv.DictReader(source):
            if not (row.get("compile_reason") or "").startswith("Unsupported function"):
                continue
            try:
                ast.parse(row["current_formula"], mode="eval")
            except SyntaxError:
                continue
            if seen < args.start:
                seen += 1
                continue
            if selected >= args.limit:
                break
            selected += 1
            identity_hash = _identity_hash(row)
            bindings, failures = [], []
            expr = None
            try:
                expr = dsl_parser.parse(row["current_formula"])
                bindings, failures = bind_fields(expr)
            except Exception as exc:
                failures = [{
                    "stage": "dsl_parse", "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }]
            binding_stream.write(json.dumps({
                "source_row": int(row["source_row"]), "id": row.get("id") or "",
                "current_formula": row["current_formula"],
                "identity_sha256": identity_hash,
                "bindings": bindings, "binding_failures": failures,
            }, ensure_ascii=False, sort_keys=True) + "\n")
            status, error = "COMPILED", ""
            if expr is None:
                status = "COMPILE_FAILED"
                first = failures[0]
                error = f"{first['error_type']}: {first['error']}"[:2000]
            else:
                try:
                    engine.compile(Factor(
                        name=row.get("id") or f"source_row_{row['source_row']}",
                        expr=expr, source_expr=row["current_formula"],
                        surface="compat_research",
                    ))
                except Exception as exc:
                    status = "COMPILE_FAILED"
                    error = f"{type(exc).__name__}: {exc}"[:2000]
            record = {
                "source_row": int(row["source_row"]), "id": row.get("id") or "",
                "current_formula": row["current_formula"], "status": status,
                "error": error, "code_hash": code_hash,
                "identity_sha256": identity_hash,
            }
            evidence.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts[status] += 1
            emitted += 1
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_source_sha256,
        "start": args.start, "limit": args.limit, "records": emitted,
        "status_counts": dict(sorted(counts.items())),
        "code_hash": code_hash, "code_hash_members": code_members,
        "evidence": evidence_path.name,
        "evidence_sha256": _file_hash(evidence_path),
        "bindings": bindings_path.name,
        "bindings_sha256": _file_hash(bindings_path),
        "scope": "current_code_no_read_compile_only_not_execution",
        "selection": "compile_reason startswith Unsupported function and ast.parse(mode=eval) succeeds",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
