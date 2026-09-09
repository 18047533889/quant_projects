"""Lossless ledger aggregation; reporting is never an execution receipt."""
from __future__ import annotations
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def observe(history, issue_id, row, *, source, source_bytes, spec_hash, manual=False):
    record = {
        "namespace": row.get("task_namespace", spec_hash),
        "spec_hash": row.get("spec_hash", spec_hash),
        "source_ledger": source,
        "source_ledger_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_commit": row.get("source_commit", row.get("commit")),
        "source_tree": row.get("source_tree", row.get("tree")),
        "assertion_kind": "MANUAL_ASSERTION" if manual else "HISTORICAL_REPORT",
        "original_conclusion": row.get("status", "NOT_RUN"),
        "original_record": row,
    }
    records = history.setdefault(issue_id, [])
    if record not in records:
        records.append(record)


def reconcile(records, *, current_tree, implementation_hashes=None):
    """Do not infer PASS from test paths, counts, manual labels or old receipts.

    A separate verifier must authenticate actual run artifacts. This aggregator
    deliberately has no VERIFIED path and preserves every original statement.
    """
    records = list(records)
    conclusions = {r["original_conclusion"] for r in records}
    domains = {(r["namespace"], r["spec_hash"], r["source_tree"]) for r in records}
    conflicts = []
    if len(conclusions) > 1:
        conflicts.append("conflicting_original_conclusions")
    if len(domains) > 1:
        conflicts.append("different_spec_or_tree_domains")
    hashes = implementation_hashes or {}
    for record in records:
        old = record["original_record"].get("implementation_hashes", {})
        if old and any(hashes.get(path) != sha for path, sha in old.items()):
            conflicts.append("implementation_changed_since_report")
    return {
        "status": "NEEDS_RECONCILIATION" if conflicts else (
            "MANUAL_ASSERTION" if any(r["assertion_kind"] == "MANUAL_ASSERTION" for r in records)
            else "NEEDS_REVALIDATION" if records else "NOT_RUN"),
        "current_execution_status": "NOT_RUN",
        "current_tree": current_tree,
        "conflicts": sorted(set(conflicts)),
        "source_records": records,
        "implementation_hashes": hashes,
    }
