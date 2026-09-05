# -*- coding: utf-8 -*-
"""Runtime fail-closed gate binding the Agent operator surface to CURRENT evidence.

R61-P0 #59: the evidence split-brain fix.  Every artifact in
``evidence/CURRENT.json`` carries a bound ``source_snapshot_id`` (the unique
SourceTreeIdentity captured when the artifact was actually produced).  A
runtime consumer (Agent mining surface, production allowlist export, ...) may
call :func:`require_current_evidence` to assert that the on-disk CURRENT file
is fresh against the LIVE tree before trusting any artifact-backed decision.

Fail-closed semantics: any problem — missing file, unreadable payload,
non-CURRENT status, stale/failed artifacts, unresolvable snapshot — raises
:class:`StaleAgentOperatorEvidence`.  There is no "proceed with warning"
path: a caller that cannot prove its evidence is current does not run.
"""
from __future__ import annotations

from typing import Any


class StaleAgentOperatorEvidence(RuntimeError):
    """Raised when CURRENT evidence cannot be proven fresh (fail-closed)."""


def current_evidence_report() -> dict[str, Any]:
    """Evaluate the on-disk CURRENT.json against the live tree, WITHOUT writing.

    Returns the evaluate() summary plus a machine ``all_current`` flag.
    """
    from evidence.current import build_current, evaluate, load_existing

    payload = load_existing()
    if payload is None:
        return {"all_current": False, "status": "MISSING", "reason": "evidence/CURRENT.json does not exist"}
    # Fold the live identities into the bound payload and decide honestly.
    live = build_current()
    payload["sha_bindings"] = live["sha_bindings"]
    payload["source_tree_identity"] = live["source_tree_identity"]
    payload["source_snapshot_id"] = live["source_snapshot_id"]
    result = evaluate(payload)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    all_current = bool(
        summary.get("stale") == 0
        and summary.get("failed") == 0
        and summary.get("current", 0) == summary.get("total", 0)
        and summary.get("total", 0) > 0
    )
    changed = (
        (result.get("evaluation") or {}).get("changed_artifacts", [])
        if isinstance(result, dict)
        else []
    )
    return {
        "all_current": all_current,
        "status": "CURRENT" if all_current else "STALE",
        "summary": summary,
        "changed_artifacts": changed,
    }


def require_current_evidence(*, allow_stale: bool = False) -> dict[str, Any]:
    """Fail-closed gate: raise unless every CURRENT.json artifact is CURRENT.

    ``allow_stale=True`` exists ONLY for research/diagnostic tooling; the
    production Agent surface must never pass it.
    """
    report = current_evidence_report()
    if report.get("all_current") or allow_stale:
        return report
    changed = report.get("changed_artifacts") or []
    summary = report.get("summary") or {}
    raise StaleAgentOperatorEvidence(
        "evidence/CURRENT.json is not CURRENT against the live source tree "
        f"(status={report.get('status')}, current={summary.get('current')}/"
        f"{summary.get('total')}, changed={changed}).  Re-run the artifact "
        "generators and `python -m evidence.current --write` before using the "
        "Agent operator surface."
    )


__all__ = ["StaleAgentOperatorEvidence", "current_evidence_report", "require_current_evidence"]
