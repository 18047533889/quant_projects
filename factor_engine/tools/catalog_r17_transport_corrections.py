"""Exact corrections for unsupported round-13 quantile-transport rewrites."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from typing import Mapping


# source_row -> (id, current_formula sha256, expected corrected call count)
APPROVED_TRANSPORT_CORRECTIONS = {
    54777: ("cold_c214b76051943f5e", "c2bf9ef76a7849e8eec17c0c13a9a063309b04296ed463a49ddea3a3b5d2ab5f", 1),
    55108: ("cold_e6d43efc83766a9a", "4a80ee394490de2e3ea5f8f83fbf6ea6de9b39f88ae0aa9f4ab57d25fa2df6d9", 1),
    55174: ("cold_475f81dd1bbad121", "a9ac8892e5cf74e3ec148342bd107499a380e59bb8464f637b25676f1584d42f", 1),
    55513: ("cold_b2a0fc9d26145ba1", "eaa9d2bc827219f892ea3286224dc947db1cb0ea185cbc79b38cd0e8c72d37e4", 1),
    56078: ("cold_38be5be647ac5d96", "2867a9165cab50f4441625a41c69850fed81525a14a3e865977ab2218d5f3f3b", 1),
    56079: ("cold_9ac2fdae404d4878", "a01da98c63ed71ebf6e1805965c929b96a887a1f6c873ed7ccd79c7458f9d46e", 1),
    56982: ("cold_18e59bd6319b1917", "311ab2f27823a22e9457ea3c140c16338642b3eb8e35e81a6abffa65f3015430", 1),
    57189: ("cold_44d1456b2a6a9b08", "22feedffaa5441d85bc8479dae430093682d0458b72d4c4d3960fc22307c066b", 1),
    57190: ("cold_e023c6196a7ad4ec", "0ec46e06b788af48a687b9268b5f998d1c571dd4b91567087535e0090fc81293", 1),
    57454: ("cold_bab8eae8b4a71d8a", "c911f9675ac86cd1a32df8f6b730912f056b51ca2f61ee84fb371eacaba75bdc", 1),
    57455: ("cold_712d03c8349cc90b", "b3df23adf962b12826b64a2291ab607bfe631082559c16f77c82bd17b922e8a5", 1),
    59919: ("cold_629def288e108baa", "bee2f4ff94a53e9ec3c1bcec464d7fcb4d3dc4a6021d34e9c666a740a130c8b7", 1),
    60423: ("cold_c04fa20d38fb5009", "dbd555c2942617a90ce0feef6be5cbe7768d35786ec0a037caf6ac851570dd7a", 1),
    60424: ("cold_1d6031b7c9f861a6", "5eb319754bcce10cd2426d8e5e9db282bc46a3964291b64352a4ad318b80a8d5", 1),
    60909: ("cold_7e7b9f808485eac2", "7dfb1c92dc878cc43299ec93e46081aa771b7a0b1529efd83fdca3f4eeeff945", 1),
    60995: ("cold_d1138863057645d1", "c4e86ccab6c9abeb375176567f999f57f48de2e46a0a014cdb59213d8e9c1282", 1),
    61105: ("cold_c9dacb871cd83602", "be59db2f585e12f8e2d62b0a0133e284467889e7903d93e5e65dbf2d241d4e49", 1),
    61117: ("cold_b68b88c3fe64f765", "1839169705e50656062c02f21aece035f922463121497097f62b23b238421a10", 1),
    61118: ("cold_dc4b6d4c12898572", "3a8bd4e35a2ac425b46cc39bb44a911304b05c9e19f8eba4647229b0cd7b7c63", 1),
    61272: ("cold_6e2db4e07c7b8f35", "5e0bc5467449bd86df83f3ca3a903b6558395bc75bd9fea9e4aae6c17a7e2381", 1),
    61842: ("cold_b57429911a43a2cd", "75158c7a4db2bafe01ec678214d7f66733050a087aa97d53370cf71170949583", 1),
    61843: ("cold_6b1709ae452a2291", "d82c521776b2ca356fed951f4df3ba26fc3fc6f9f22a787fc22b7b4a22dbed62", 1),
    62317: ("cold_dbf7ca63d4b4a1d7", "f5ab12ef72b94a1e6b80efd9ddecbeaf742cfa2b397caed9ccd95cd6a8feb177", 1),
    62966: ("cold_93964857ffe71348", "281a9763071208f2b2c9ce64ec594a8e8a34e723fa19e2acec4a4fc7dd55fe9f", 1),
    63460: ("cold_b1ee860baf5135ee", "7273ec0fe4ad2a45e0c671f0e35b2a07846cd4bfb91ab7e68e5550f044107679", 1),
    63461: ("cold_ac69ac5b0bae9414", "83921880722b0562e400abec5a09a0235e9cbcba868fb94012c3f2f2ac90ec0f", 1),
    63934: ("cold_7cd74d78c1357a75", "f4326292919a44e7de379b83e363d349f06cac91548e119ee2d5815848c23632", 1),
    64372: ("cold_cc016f4ea901e7d5", "eeab9849a55efe42df567b3c50eac24a949955e7dc148a8b92d682f00471af3f", 1),
    66532: ("cold_629320eca33d9b7f", "d967be446e079f48b562da187f3480d139a10f2679b37a336b279164f96c37b8", 2),
    66617: ("cold_3a7fb08819f21e95", "2511b2ccd31814a0fff3d5d6d7ac5c1fc8ff75244c4f5612ea23a628e277d203", 2),
}

_OPS = {"ts_quantile_transport_slope", "ts_quantile_transport_curvature"}
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
    return (
        _offset(lines, node.lineno, node.col_offset),
        _offset(lines, node.end_lineno, node.end_col_offset),
    )


def correct_transport_row(row: Mapping[str, str]) -> dict[str, str]:
    """Correct one identity/hash-pinned row, otherwise return it unchanged."""
    result = copy.deepcopy(dict(row))
    target = APPROVED_TRANSPORT_CORRECTIONS.get(_row_number(row))
    if target is None:
        return result
    expected_id, expected_hash, expected_count = target
    formula = row["current_formula"]
    actual_hash = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    if row.get("id") != expected_id or actual_hash != expected_hash:
        raise ValueError("approved transport correction identity/hash mismatch")

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in _OPS or len(node.args) != 1 or len(node.keywords) != 2:
            continue
        values = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        if set(values) != {"recent_window", "old_window"}:
            continue
        recent, old = values["recent_window"], values["old_window"]
        if not (
            isinstance(recent, ast.Constant) and type(recent.value) is int and recent.value == 20
            and isinstance(old, ast.Constant) and type(old.value) is int and old.value == 40
        ):
            continue
        arg_start, arg_end = _span(lines, node.args[0])
        start, end = _span(lines, node)
        edits.append((start, end, f"{node.func.id}({formula[arg_start:arg_end]}, window=60)"))
    if len(edits) != expected_count:
        raise ValueError("approved transport correction call-count mismatch")
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
        "CORRECTION ts_quantile_transport: withdrew unsupported "
        "recent_window=20, old_window=40 redesign back to unresolved window=60; "
        "scalar horizon did not prove the split; invalidated prior evidence="
        + json.dumps(history, ensure_ascii=False, sort_keys=True)
    )
    result["migration_changes"] = json.dumps(changes, ensure_ascii=False)
    result["compile_status"] = "NOT_RUN"
    result["compile_reason"] = "transport split correction requires compile rerun"
    result["execution_status"] = "NOT_RUN"
    result["execution_reason"] = "transport split correction invalidated prior execution evidence"
    for field in _INVALIDATED_FIELDS:
        if field in result:
            result[field] = ""
    return result
