"""Build one pinned, consolidated R18c correction/evidence checkpoint."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from factor_engine.tools.catalog_review_evidence import validate_review_source


SUCCESS = "EXECUTED"
ALL_NONFINITE = "EXECUTED_ALL_NONFINITE"
FAILURES = {
    "BATCH_ABORTED", "EXECUTION_FAILED", "PREPARE_FAILED", "NATIVE_CRASH",
}
EXECUTION_FIELDS = (
    "value_count", "finite_count", "result_hash", "retry_after_batch_abort",
    "batch_abort_error_type", "batch_abort_error", "backend_path",
    "execution_evidence_file", "execution_validation_scope", "execution_window",
    "execution_symbol_count",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(row: dict) -> tuple[int, str]:
    return int(row["source_row"]), row.get("id") or ""


def load_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def counts_arg(value: str) -> dict[str, int]:
    result = {}
    for part in value.split(","):
        if not part:
            continue
        key, raw = part.split("=", 1)
        count = int(raw)
        if count:
            result[key] = count
    return result


def require_pin(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual}")


def clear_execution(row: dict[str, str], reason: str, evidence: str) -> None:
    old = {key: row.get(key, "") for key in (
        "execution_status", "execution_reason", "execution_evidence_file",
        "execution_validation_scope", "value_count", "finite_count", "result_hash",
    )}
    changes = json.loads(row.get("migration_changes") or "[]")
    changes.append(
        "R18C_COMPILE_INVALIDATION: " + reason + "; prior="
        + json.dumps(old, ensure_ascii=False, sort_keys=True)
        + "; evidence=" + evidence
    )
    row["migration_changes"] = json.dumps(changes, ensure_ascii=False)
    row["execution_status"] = "COMPILE_FAILED"
    row["execution_reason"] = reason
    for field in EXECUTION_FIELDS:
        row[field] = ""


def merge_execution(
    row: dict[str, str], item: dict, execution_name: str, summary: dict
) -> None:
    old = {key: row.get(key, "") for key in (
        "execution_status", "execution_reason", "execution_evidence_file",
        "execution_validation_scope", "value_count", "finite_count", "result_hash",
    )}
    changes = json.loads(row.get("migration_changes") or "[]")
    changes.append(
        "R18C_EXECUTION_RECONCILE: prior="
        + json.dumps(old, ensure_ascii=False, sort_keys=True)
        + "; evidence=" + execution_name
    )
    row["migration_changes"] = json.dumps(changes, ensure_ascii=False)
    status = item["status"]
    row["execution_status"] = status
    row["execution_evidence_file"] = execution_name
    row["execution_validation_scope"] = (
        "pinned_source_request_output_summary; identity_and_current_formula_matched; "
        "research_execution_not_production_certification"
    )
    row["execution_window"] = json.dumps(
        item.get("execution_window", summary.get("window")), ensure_ascii=False
    )
    row["execution_symbol_count"] = item.get(
        "execution_symbol_count", len(summary.get("symbols") or [])
    )
    row["retry_after_batch_abort"] = item.get("retry_after_batch_abort", "")
    row["batch_abort_error_type"] = item.get("batch_abort_error_type", "")
    row["batch_abort_error"] = item.get("batch_abort_error", "")
    if status == SUCCESS:
        row.update(
            execution_reason="",
            value_count=item["value_count"],
            finite_count=item["finite_count"],
            result_hash=item["result_hash"],
            backend_path=json.dumps(item.get("backend_path"), ensure_ascii=False),
        )
    elif status == ALL_NONFINITE:
        row.update(
            execution_reason="execution completed with zero finite values",
            value_count=item["value_count"],
            finite_count=item["finite_count"],
            result_hash=item["result_hash"],
            backend_path=json.dumps(item.get("backend_path"), ensure_ascii=False),
        )
    else:
        row.update(
            execution_reason=f"{item['error_type']}: {item['error']}",
            value_count="", finite_count="", result_hash="", backend_path="",
        )


def validate_terminal_result(item: dict, key, hex_digest) -> None:
    status = item.get("status")
    if status == SUCCESS:
        values, finite = item.get("value_count"), item.get("finite_count")
        if type(values) is not int or type(finite) is not int or not 0 < finite <= values:
            raise ValueError(f"invalid successful counts {key}")
        if not hex_digest(item.get("result_hash")):
            raise ValueError(f"invalid successful hash {key}")
    elif status == ALL_NONFINITE:
        values, finite = item.get("value_count"), item.get("finite_count")
        if type(values) is not int or values <= 0 or type(finite) is not int or finite != 0:
            raise ValueError(f"invalid all-nonfinite counts {key}")
        if not hex_digest(item.get("result_hash")):
            raise ValueError(f"invalid all-nonfinite hash {key}")
    elif status in FAILURES:
        if not item.get("error_type") or not item.get("error"):
            raise ValueError(f"failure lacks evidence {key}")
    else:
        raise ValueError(f"non-terminal execution result {key}: {status}")


def run_embedded_tests(hex_digest) -> None:
    digest = "a" * 64
    row = {
        "migration_changes": "[]", "execution_status": "BATCH_ABORTED",
        "execution_reason": "old", "execution_evidence_file": "old.jsonl.gz",
        "execution_validation_scope": "old", "value_count": "",
        "finite_count": "", "result_hash": "", "backend_path": "",
        "retry_after_batch_abort": "", "batch_abort_error_type": "",
        "batch_abort_error": "", "execution_window": "",
        "execution_symbol_count": "",
    }
    item = {
        "status": ALL_NONFINITE, "value_count": 2560, "finite_count": 0,
        "result_hash": digest, "backend_path": {"actual_backend": "pandas_numpy"},
    }
    validate_terminal_result(item, (1, "all-null"), hex_digest)
    merged = dict(row)
    merge_execution(
        merged, item, "closed.output.jsonl.gz",
        {"window": ["2025-01-01", "2026-04-30"], "symbols": ["A"]},
    )
    assert merged["execution_status"] == ALL_NONFINITE
    assert merged["value_count"] == 2560 and merged["finite_count"] == 0
    assert merged["result_hash"] == digest and "pandas_numpy" in merged["backend_path"]
    cleared = dict(merged)
    clear_execution(cleared, "compile failed", "compile.jsonl.gz")
    assert cleared["execution_status"] == "COMPILE_FAILED"
    assert all(cleared[field] == "" for field in EXECUTION_FIELDS)
    assert "closed.output.jsonl.gz" in cleared["migration_changes"]
    bad = dict(item, status=SUCCESS, finite_count=1, result_hash="G" * 64)
    try:
        validate_terminal_result(bad, (2, "bad-hash"), hex_digest)
    except ValueError as exc:
        assert "invalid successful hash" in str(exc)
    else:
        raise AssertionError("bad success hash was accepted")


def reconcile(args):
    checkpoint = Path(args.checkpoint)
    gaussian_tool = Path(args.gaussian_tool)
    reconcile_helper = Path(args.reconcile_helper)
    compile_input = Path(args.compile_input)
    compile_evidence = Path(args.compile_evidence)
    compile_summary_path = Path(args.compile_summary)
    valid_input = Path(args.valid_input)
    execution = Path(args.execution)
    execution_summary_path = Path(args.execution_summary)
    builder = Path(__file__).resolve()
    pins = (
        (checkpoint, args.checkpoint_sha256, "checkpoint"),
        (gaussian_tool, args.gaussian_tool_sha256, "Gaussian correction tool"),
        (reconcile_helper, args.reconcile_helper_sha256, "execution reconcile helper"),
        (compile_input, args.compile_input_sha256, "compile input"),
        (compile_evidence, args.compile_evidence_sha256, "compile evidence"),
        (compile_summary_path, args.compile_summary_sha256, "compile summary"),
        (valid_input, args.valid_input_sha256, "valid execution input"),
        (execution, args.execution_sha256, "execution output"),
        (execution_summary_path, args.execution_summary_sha256, "execution summary"),
    )
    for path, pin, label in pins:
        require_pin(path, pin, label)
    source_validation = validate_review_source(checkpoint, args.checkpoint_sha256)

    helper_spec = importlib.util.spec_from_file_location(
        "_r18c_pinned_reconcile_helper", reconcile_helper
    )
    if helper_spec is None or helper_spec.loader is None:
        raise ValueError("cannot load pinned execution reconcile helper")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    hex_digest = helper_module._hex_digest
    if (
        helper_module.SUCCESS != SUCCESS
        or helper_module.ALL_NONFINITE != ALL_NONFINITE
        or helper_module.FAILURES != FAILURES
    ):
        raise ValueError("pinned reconcile helper terminal-status contract mismatch")
    run_embedded_tests(hex_digest)

    module = importlib.import_module("factor_engine.tools.catalog_r18_gaussian_corrections")
    if Path(module.__file__).resolve() != gaussian_tool.resolve():
        raise ValueError("Gaussian correction import path mismatch")
    correct_gaussian_row = module.correct_gaussian_row

    compile_requests = load_jsonl(compile_input)
    compile_records = load_jsonl(compile_evidence)
    compile_summary = json.loads(compile_summary_path.read_text())
    request_map = {identity(row): row for row in compile_requests}
    compile_map = {identity(row): row for row in compile_records}
    if len(compile_requests) != 40 or len(request_map) != 40:
        raise ValueError("compile input must contain 40 unique identities")
    if len(compile_records) != 40 or set(compile_map) != set(request_map):
        raise ValueError("compile evidence identity mismatch")
    compile_counts = collections.Counter(row.get("status") for row in compile_records)
    if compile_counts != collections.Counter(COMPILED=22, COMPILE_FAILED=18):
        raise ValueError(f"unexpected compile counts {compile_counts}")
    if (
        compile_summary.get("processed") != 40
        or compile_summary.get("counts") != dict(compile_counts)
        or compile_summary.get("compile_output_sha256") != args.compile_evidence_sha256
        or compile_summary.get("pinned_input_sha256") != args.compile_input_sha256
        or compile_summary.get("valid_input_sha256") != args.valid_input_sha256
        or compile_summary.get("valid_formulas_unchanged_from_pinned_input") is not True
    ):
        raise ValueError("compile summary mismatch")
    compiled_ids = {key for key, row in compile_map.items() if row["status"] == "COMPILED"}
    failed_ids = set(compile_map) - compiled_ids
    for key, record in compile_map.items():
        request_formula = request_map[key]["formula"]
        if record.get("formula") != request_formula or record.get("current_formula") != request_formula:
            raise ValueError(f"compile formula mismatch {key}")
        if record.get("field_binding_failures") != []:
            raise ValueError(f"compile binding failure {key}")
        if key in failed_ids and not (record.get("error_type") and record.get("error")):
            raise ValueError(f"compile failure lacks evidence {key}")

    valid_requests = load_jsonl(valid_input)
    valid_map = {identity(row): row for row in valid_requests}
    if len(valid_requests) != 22 or set(valid_map) != compiled_ids:
        raise ValueError("valid execution input does not equal compiled subset")
    for key, row in valid_map.items():
        if row.get("formula") != request_map[key].get("formula"):
            raise ValueError(f"valid execution formula mismatch {key}")

    results = load_jsonl(execution)
    result_map = {identity(row): row for row in results}
    execution_summary = json.loads(execution_summary_path.read_text())
    expected_execution_counts = counts_arg(args.expected_execution_counts)
    actual_execution_counts = collections.Counter(row.get("status") for row in results)
    if len(results) != 22 or len(result_map) != 22 or set(result_map) != compiled_ids:
        raise ValueError("execution output identity mismatch")
    if dict(actual_execution_counts) != expected_execution_counts:
        raise ValueError(f"unexpected execution counts {actual_execution_counts}")
    if (
        execution_summary.get("complete") is not True
        or execution_summary.get("processed") != 22
        or execution_summary.get("requested_limit") != 22
        or execution_summary.get("counts") != expected_execution_counts
        or Path(execution_summary.get("input", "")).name != valid_input.name
        or Path(execution_summary.get("output", "")).name != execution.name
    ):
        raise ValueError("execution summary mismatch")
    for key, item in result_map.items():
        if item.get("executed_formula") != valid_map[key].get("formula"):
            raise ValueError(f"executed formula mismatch {key}")
        validate_terminal_result(item, key, hex_digest)

    output = Path(str(args.output_prefix) + ".csv.gz")
    manifest = Path(str(args.output_prefix) + ".manifest.json")
    gaussian_compile = Path(str(args.output_prefix) + ".gaussian.compile.jsonl.gz")
    for path in (output, manifest, gaussian_compile):
        if path.exists():
            raise FileExistsError(path)

    helper = Path(__file__).resolve().parents[1] / "factor_catalog_20260915"
    sys.path.insert(0, str(helper))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor

    parser, engine = build_runtime()
    expected_catalog_counts = counts_arg(args.expected_catalog_counts)
    expected_compile_counts = counts_arg(args.expected_nonempty_compile_counts)
    corrected_ids = set()
    seen_compile_ids = set()
    seen_execution_ids = set()
    catalog_counts = collections.Counter()
    nonempty_compile_counts = collections.Counter()
    gaussian_compile_counts = collections.Counter()
    builder_sha = sha256(builder)

    with tempfile.TemporaryDirectory(prefix=".r18c-", dir=output.parent) as tmp:
        staged_output = Path(tmp) / output.name
        staged_gaussian = Path(tmp) / gaussian_compile.name
        with (
            gzip.open(checkpoint, "rt", encoding="utf-8-sig", newline="") as src,
            gzip.open(staged_output, "xt", encoding="utf-8-sig", newline="") as dst,
            gzip.open(staged_gaussian, "xt", encoding="utf-8") as gev,
        ):
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            rows = 0
            for original in reader:
                rows += 1
                key = identity(original)
                row = correct_gaussian_row(original)
                gaussian_changed = row["current_formula"] != original["current_formula"]
                if gaussian_changed:
                    if key in request_map:
                        raise ValueError(f"Gaussian/40-row overlap {key}")
                    corrected_ids.add(key)
                    formula = row["current_formula"]
                    record = {
                        "source_row": key[0], "id": key[1], "formula": formula,
                        "formula_sha256": hashlib.sha256(formula.encode()).hexdigest(),
                    }
                    row["bindings"] = "[]"
                    row["current_fields"] = ""
                    row["current_tables"] = ""
                    row["static_status"] = ""
                    row["static_reason"] = ""
                    phase = "parse"
                    try:
                        expr = parser.parse(formula)
                        phase = "bind"
                        bindings, failures = bind_fields(expr)
                        record["field_binding_count"] = len(bindings)
                        record["field_binding_failures"] = failures
                        row["bindings"] = json.dumps(bindings, ensure_ascii=False)
                        fields = sorted({f"{x['table']}.{x['column']}" for x in bindings})
                        row["current_fields"] = "|".join(fields)
                        row["current_tables"] = "|".join(sorted({x["table"] for x in bindings}))
                        if failures:
                            raise ValueError("FIELD_BINDING_FAILED: " + "; ".join(map(str, failures)))
                        row["static_status"] = "PARSED_FIELDS_BOUND"
                        row["static_reason"] = ""
                        phase = "compile"
                        engine.compile(Factor(name=key[1], expr=expr, source_expr=formula, surface="compat_research"))
                        row["compile_status"] = "COMPILED"
                        row["compile_reason"] = ""
                        record["status"] = "COMPILED"
                    except Exception as exc:
                        if phase == "parse":
                            row["static_status"] = "PARSE_FAILED"
                            row["static_reason"] = str(exc)[:2000]
                        elif phase == "bind":
                            row["static_status"] = "FIELDS_UNRESOLVED"
                            row["static_reason"] = str(exc)[:2000]
                        row["compile_status"] = "COMPILE_FAILED"
                        row["compile_reason"] = str(exc)[:2000]
                        record.update(status="COMPILE_FAILED", error_type=type(exc).__name__, error=str(exc)[:2000])
                    row["compile_evidence_file"] = gaussian_compile.name
                    row["compile_validation_scope"] = (
                        "identity/hash-pinned Gaussian correction; formal runtime field-bind and compile"
                    )
                    if row["compile_status"] == "COMPILE_FAILED":
                        clear_execution(
                            row,
                            f"{record['error_type']}: {record['error']}",
                            gaussian_compile.name,
                        )
                    gaussian_compile_counts[row["compile_status"]] += 1
                    gev.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

                compile_record = compile_map.get(key)
                if compile_record is not None:
                    if row["current_formula"] != request_map[key]["formula"]:
                        raise ValueError(f"checkpoint/compile formula mismatch {key}")
                    seen_compile_ids.add(key)
                    row["compile_status"] = compile_record["status"]
                    row["compile_reason"] = "" if compile_record["status"] == "COMPILED" else (
                        f"{compile_record['error_type']}: {compile_record['error']}"
                    )
                    row["compile_evidence_file"] = compile_evidence.name
                    row["compile_validation_scope"] = (
                        "pinned 40-row formal runtime compile preflight; field bindings checked; no data read"
                    )
                    if key in failed_ids:
                        clear_execution(
                            row,
                            f"{compile_record['error_type']}: {compile_record['error']}",
                            compile_evidence.name,
                        )
                    else:
                        item = result_map[key]
                        merge_execution(row, item, execution.name, execution_summary)
                        seen_execution_ids.add(key)

                if row.get("id"):
                    nonempty_compile_counts[row.get("compile_status") or ""] += 1
                catalog_counts[row.get("execution_status") or ""] += 1
                writer.writerow(row)

        if rows != source_validation["rows"]:
            raise ValueError(f"row coverage mismatch: {rows}")
        if len(corrected_ids) != 3:
            raise ValueError(f"expected three Gaussian corrections, got {corrected_ids}")
        if gaussian_compile_counts != collections.Counter(COMPILE_FAILED=3):
            raise ValueError(f"unexpected Gaussian compile counts {gaussian_compile_counts}")
        if seen_compile_ids != set(compile_map) or seen_execution_ids != compiled_ids:
            raise ValueError("checkpoint does not cover all compile/execution identities")
        if dict(catalog_counts) != expected_catalog_counts:
            raise ValueError(f"execution status mismatch {catalog_counts}")
        if dict(nonempty_compile_counts) != expected_compile_counts:
            raise ValueError(f"nonempty compile status mismatch {nonempty_compile_counts}")
        for path, pin, label in pins:
            require_pin(path, pin, label)
        if sha256(builder) != builder_sha:
            raise ValueError("builder changed during publication")

        payload = {
            "input_checkpoint": str(checkpoint),
            "input_checkpoint_sha256": args.checkpoint_sha256,
            "output_checkpoint": str(output),
            "output_checkpoint_sha256": sha256(staged_output),
            "gaussian_correction_tool": str(gaussian_tool),
            "gaussian_correction_tool_sha256": args.gaussian_tool_sha256,
            "execution_reconcile_helper": str(reconcile_helper),
            "execution_reconcile_helper_sha256": args.reconcile_helper_sha256,
            "gaussian_corrected_rows": len(corrected_ids),
            "gaussian_compile_counts": dict(sorted(gaussian_compile_counts.items())),
            "gaussian_compile_evidence": str(gaussian_compile),
            "gaussian_compile_evidence_sha256": sha256(staged_gaussian),
            "compile_input": str(compile_input),
            "compile_input_sha256": args.compile_input_sha256,
            "compile_evidence": str(compile_evidence),
            "compile_evidence_sha256": args.compile_evidence_sha256,
            "compile_summary": str(compile_summary_path),
            "compile_summary_sha256": args.compile_summary_sha256,
            "valid_execution_input": str(valid_input),
            "valid_execution_input_sha256": args.valid_input_sha256,
            "execution_output": str(execution),
            "execution_output_sha256": args.execution_sha256,
            "execution_summary": str(execution_summary_path),
            "execution_summary_sha256": args.execution_summary_sha256,
            "mixed_compile_counts": dict(sorted(compile_counts.items())),
            "mixed_execution_counts": dict(sorted(actual_execution_counts.items())),
            "output_execution_status_counts": dict(sorted(catalog_counts.items())),
            "output_nonempty_compile_status_counts": dict(sorted(nonempty_compile_counts.items())),
            "source_rows": rows,
            "unique_factor_ids": source_validation["unique_factor_ids"],
            "builder_sha256": builder_sha,
            "scope": (
                "single consolidated R18c: three pinned Gaussian withdrawals, "
                "40-row compile preflight, and closed 22-row execution reconciliation"
            ),
        }
        staged_manifest = Path(tmp) / manifest.name
        staged_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.link(staged_output, output)
        os.link(staged_gaussian, gaussian_compile)
        os.link(staged_manifest, manifest)
    return payload


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--gaussian-tool", required=True)
    p.add_argument("--gaussian-tool-sha256", required=True)
    p.add_argument("--reconcile-helper", required=True)
    p.add_argument("--reconcile-helper-sha256", required=True)
    p.add_argument("--compile-input", required=True)
    p.add_argument("--compile-input-sha256", required=True)
    p.add_argument("--compile-evidence", required=True)
    p.add_argument("--compile-evidence-sha256", required=True)
    p.add_argument("--compile-summary", required=True)
    p.add_argument("--compile-summary-sha256", required=True)
    p.add_argument("--valid-input", required=True)
    p.add_argument("--valid-input-sha256", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--execution-summary", required=True)
    p.add_argument("--execution-summary-sha256", required=True)
    p.add_argument("--expected-execution-counts", required=True)
    p.add_argument("--expected-catalog-counts", required=True)
    p.add_argument("--expected-nonempty-compile-counts", required=True)
    p.add_argument("--output-prefix", required=True, type=Path)
    args = p.parse_args(argv)
    payload = reconcile(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
