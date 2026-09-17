"""Pinned withdrawal of unsupported R13 Gaussian-method assumptions."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from typing import Mapping


# source_row -> (id, exact current_formula sha256, expected corrected call count)
APPROVED_GAUSSIAN_CORRECTIONS = {
    26549: ("cold_r26_70bdca3572252fd6f6", "e430d9c37e2aab1fc5e5c071a8aec0d467933e28fe112e1eefd9844088b90da9", 1),
    26552: ("cold_r26_6ed57c3ac3dd077d00", "e6faa5b17f0eefd08a46f47a59d10dfa6547545ea24284dcbef129d5437edda3", 1),
    26905: ("cold_r26_b4520cee288df9a3f4", "8536cd947c06bcc581b59fc4fe3e0eb739aba7a79c23a4a29ff54caa5ab52cb0", 1),
}

_INVALIDATED_FIELDS = (
    "compile_evidence_file", "compile_validation_scope", "backend_path",
    "execution_evidence_file", "execution_validation_scope", "execution_window",
    "execution_symbol_count", "value_count", "finite_count", "result_hash",
    "retry_after_batch_abort", "batch_abort_error_type", "batch_abort_error",
)


def _row_number(row: Mapping[str, str]) -> int:
    return int(row.get("source_row") or row.get("\ufeffsource_row") or "")


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (_offset(lines, node.lineno, node.col_offset), _offset(lines, node.end_lineno, node.end_col_offset))


def correct_gaussian_row(row: Mapping[str, str]) -> dict[str, str]:
    """Withdraw the three identity/hash-pinned assumptions; otherwise no-op."""
    result = copy.deepcopy(dict(row))
    target = APPROVED_GAUSSIAN_CORRECTIONS.get(_row_number(row))
    if target is None:
        return result
    expected_id, expected_hash, expected_count = target
    formula = row["current_formula"]
    if row.get("id") != expected_id or hashlib.sha256(formula.encode()).hexdigest() != expected_hash:
        raise ValueError("approved Gaussian correction identity/hash mismatch")

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "cs_rank_gaussian" or len(node.args) != 2 or node.keywords:
            continue
        method = node.args[1]
        if isinstance(method, ast.Constant) and method.value == "blom":
            start, end = _span(lines, method)
            edits.append((start, end, "3.0"))
    if len(edits) != expected_count:
        raise ValueError("approved Gaussian correction call-count mismatch")
    for start, end, replacement in sorted(edits, reverse=True):
        formula = formula[:start] + replacement + formula[end:]
    result["current_formula"] = formula

    changes = json.loads(row.get("migration_changes") or "[]")
    history = {
        "prior_compile_status": row.get("compile_status", ""),
        "prior_compile_evidence_file": row.get("compile_evidence_file", ""),
        "prior_execution_status": row.get("execution_status", ""),
        "prior_execution_evidence_file": row.get("execution_evidence_file", ""),
        "corrected_calls": expected_count,
    }
    changes.append(
        "CORRECTION cs_rank_gaussian: withdrew unsupported numeric method=3.0 -> "
        "'blom' assumption back to unresolved numeric method=3.0; review text did not "
        "specify a canonical method; invalidated prior evidence="
        + json.dumps(history, ensure_ascii=False, sort_keys=True)
    )
    result["migration_changes"] = json.dumps(changes, ensure_ascii=False)
    result["compile_status"] = "NOT_RUN"
    result["compile_reason"] = "Gaussian method correction requires compile rerun"
    result["execution_status"] = "NOT_RUN"
    result["execution_reason"] = "Gaussian method correction invalidated prior execution evidence"
    for field in _INVALIDATED_FIELDS:
        if field in result:
            result[field] = ""
    return result
