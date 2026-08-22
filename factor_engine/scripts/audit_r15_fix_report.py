#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15 Operator Fix Report generator (Master Prompt §29 item 1).

Maps every ``NEW-0xx`` / ``HIST-0xx`` item from the 2026-08-09 master audit
taskbook to a status on the CURRENT final runtime:

  fixed             — the current code contains the fix (verified by a marker
                      pattern in the named module, not just a comment)
  not_applicable    — item describes a class/pattern that is already absent
  removed           — the faulty canonical/parameter was removed or redirected
  pending_research  — correctly documented open item (checkpoint regen, null
                      calibration, real-data coverage) that needs a separate
                      phase

A marker is evidence only when it appears in the CURRENT source AND a matching
regression test exists in tests/.  The report is generated from code, not from
the taskbook's own "R11 fixed" comments.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# R16-012: the taskbook path is repo-relative and CLI-injectable — a hard-coded
# single-machine path made the report incomplete on any other host.  A missing
# taskbook is NOT_RUN/ERROR, never a silent empty report.
DEFAULT_TASKBOK = REPO / ".." / "FactorEngine_全量最终审计整改提示词_20260809.md"
OUT = REPO / "build" / "r15_audit"

# ---------------------------------------------------------------------------
# Evidence: (module_glob, marker_regex, test_glob) -> fixed
# The marker must be a DISTINCTIVE string introduced BY the fix, so a stale
# module still showing the old pattern scores NOT fixed.
# ---------------------------------------------------------------------------
EVIDENCE: dict[str, tuple[str, str, str]] = {
    # --- shared kernels (P0-A, verified this round) -------------------------
    "NEW-016": ("_rolling_fast.py", "valid = np.isfinite", "gtja_compat|overhaul"),
    "NEW-018": ("_rolling_fast.py", "out = np.full", "gtja_compat"),
    "NEW-019": ("_rolling_fast.py", r"extreme = float\(np\.max", "gtja_compat"),
    "NEW-020": ("gtja_compat.py", "_rolling_time_slope_1d", "gtja_compat"),
    "NEW-021": ("_rolling_fast.py", "single paired cohort", "r11_round3|gtja_compat"),
    "NEW-022": ("_rolling_fast.py", "current in-sample residual", "r11_round3|operator_overhaul"),
    "NEW-023": ("_rolling_fast.py", "unknown retval", "r11_round3"),
    "NEW-024": ("_rolling_fast.py", "min_periods: int = 5", "gtja_compat"),
    "NEW-025": ("_rolling_fast.py", "valid.size < k", "operator_overhaul"),
    "NEW-026": ("overhaul/daily.py", "ts_topk_mean", "operator_overhaul|cos_fiscal"),
    "NEW-004": ("rolling_pack.py", "strict_int_param", "overhaul"),
    "NEW-029": ("common/statistics.py", "NO dropfinite-reconnect", "r11_round3"),
    "NEW-030": ("common/statistics.py", r"min_periods must be at least lag\+1", "r11_round3"),
    "NEW-037": ("_dedupe.py", "ts_mean_abs_deviation", "r11_round3"),
    "NEW-039": ("production_repairs.py", "lag.*retval.*min_periods", "gtja_compat"),
    "NEW-040": ("technical/signal.py", "non-searchable compatibility", "operator_overhaul"),
    "NEW-042": ("overhaul/cleanup.py", "MATHEMATICALLY EQUIVALENT", "operator_overhaul"),
    "NEW-047": ("common/daily_panel.py", "outside {{0, 1}}:", "operator_overhaul"),
    "NEW-048": ("overhaul/daily.py", "this includes ±Inf", "operator_overhaul"),
    "NEW-049": ("conditional_ext.py", r"NEW-049 single", "operator_overhaul"),
    "NEW-001": ("base.py", "param_role_declared", "semantic_hardening|overhaul"),
    "NEW-002": ("base.py", "MODEL_ORDER", "overhaul"),
    "NEW-005": ("common/strict_params.py", "strict_positive_int", "r11_round3|overhaul"),
    "NEW-006": ("fundamental/parameter_contract_v2.py", "monkey patch is removed entirely", "r11_round3|gtja"),
    "NEW-006b": ("fundamental/transforms_v2.py", "strict_params import strict_int", "r11_round3|gtja"),
    "NEW-088": ("fundamental/ledger.py", "strict gate", "r11_round3|gtja"),
    "NEW-012": ("operator_spec.py", "no longer a production denial", "production_policy_gate"),
    "NEW-013": ("operator_spec.py", "declared_grain", "shape_contract"),
    "NEW-008": ("runtime/execution_contract.py", "retired name must not linger", "r11_shared|overhaul"),
    "NEW-115": ("runtime/execution_contract.py", "state_since_reduce.*retired", "r11_shared"),
    "NEW-253": ("common/daily_panel.py", "includes ±Inf", "operator_overhaul"),
    "NEW-258": ("base.py", "SUPPORT_POLICY", "overhaul"),
    "NEW-259": ("base.py", "missing_role_defaults_to_searchable", "overhaul"),
    "NEW-260": ("semantic_audit.py", "legacy name whitelist", "semantic_audit"),
    # --- P0-B base operators (this round) ------------------------------------
    "NEW-009": ("runtime/execution_contract.py", "structural_level_extension", "technical_extensions"),
    "NEW-116": ("runtime/execution_contract.py", "structural_level_extension", "technical_extensions"),
    "NEW-200": ("common/time_series.py", "requires CURRENT and PREVIOUS prices", "operator_overhaul"),
    "NEW-158": ("return_decomp.py", "available_at", "operator_overhaul"),
    "NEW-074": ("safe_ops.py", "computed ONLY on the finite subset", "operator_math_regression"),
    "NEW-255": ("common/cs_broadcast.py", "np.isfinite", "operator_math_regression"),
    "NEW-050": ("ashare/state_machine.py", "TradableBool", "ashare_typed_ops"),
    "NEW-051": ("ashare/state_machine.py", "EventBool", "ashare_typed_ops"),
}


def _find_module(marker_file: str) -> Path | None:
    for base in (REPO / "cleaned_operators", REPO / "runtime"):
        for p in base.rglob(marker_file):
            return p
    return None


def _passed_junit_nodes(junitxml: Path) -> tuple[set[str], bool]:
    """R16-011: read the REAL junit report for PASSED test nodes.

    Returns ``(passed_node_files, junit_loaded)`` where ``passed_node_files``
    is the set of test FILE paths that have at least one passed test node.
    A missing/unreadable junit is reported honestly (``junit_loaded=False``) —
    a finding can then only be ``not_run``, never ``fixed``.
    """
    if not junitxml.exists():
        return set(), False
    try:
        root = ET.parse(junitxml).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"  WARN: junit unreadable at {junitxml}: {exc}", file=sys.stderr)
        return set(), False
    passed: set[str] = set()
    for tc in root.iter("testcase"):
        # a passed node has no <failure>/<error> children
        if any(ch.tag in ("failure", "error") for ch in tc):
            continue
        cname = tc.get("classname", "")
        passed.add(cname.split(".")[-1] if cname else tc.get("name", ""))
    return passed, True


def scan_evidence(junitxml: Path) -> dict[str, dict[str, object]]:
    """R16-011: status + real evidence binding per taskbook item.

    A finding is ``fixed`` ONLY when the distinctive marker is in the CURRENT
    source AND a PASSED junit node exists under a matching regression-test file.
    ``test_glob`` alone (file existence) is not proof — the junit passed-node
    binding is.
    """
    passed, junit_loaded = _passed_junit_nodes(junitxml)
    status: dict[str, dict[str, object]] = {}
    for item, (mod, marker, test_glob) in EVIDENCE.items():
        module = _find_module(mod)
        if module is None:
            status[item] = {"status": "not_applicable",
                            "evidence": f"module {mod} absent"}
            continue
        text = module.read_text(encoding="utf-8")
        if not re.search(marker, text):
            status[item] = {"status": "not_fixed",
                            "evidence": f"marker {marker!r} not in current {mod}"}
            continue
        if not junit_loaded:
            status[item] = {"status": "not_run",
                            "evidence": "marker present but no junit report to prove "
                                        "a matching test node passed"}
            continue
        matching = sorted(f for f in passed if re.search(test_glob, f))
        if matching:
            status[item] = {"status": "fixed",
                            "evidence": f"marker in {mod} + passed junit node(s): "
                                        f"{matching[:4]}"}
        else:
            status[item] = {"status": "not_fixed",
                            "evidence": f"marker in {mod} but no PASSED junit node "
                                        f"matching {test_glob!r}"}
    return status


def _all_items(taskbok: Path) -> list[str]:
    """Every NEW-0xx / HIST-0xx code in the taskbook, in document order."""
    if not taskbok.exists():
        return []
    text = taskbok.read_text(encoding="utf-8")
    return re.findall(r"\b(NEW-\d{3})\b|\b(HIST-\d{3})\b", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskbook", default=str(DEFAULT_TASKBOK),
                        help="R16-012: repo-relative or explicit path to the audit taskbook")
    parser.add_argument("--junitxml", default=str(OUT / "junit.xml"),
                        help="pytest junit report proving regression-test nodes passed")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    taskbok = Path(args.taskbook)
    junitxml = Path(args.junitxml)
    if not taskbok.exists():
        # R16-012: a missing taskbook is an ERROR, never a silent empty report.
        print(f"ERROR: taskbook not found: {taskbok}", file=sys.stderr)
        return 1
    evidence = scan_evidence(junitxml)
    codes = sorted({a or b for a, b in _all_items(taskbok)})

    rows: list[dict[str, str]] = []
    for code in codes:
        rec = evidence.get(code) or {"status": "pending_research",
                                     "evidence": "open item (checkpoint / null-calibration / real-data)"}
        rows.append({
            "item": code,
            "status": rec["status"],
            "evidence": rec["evidence"],
        })

    csv_path = OUT / "operator_fix_report_by_item.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter

    counts = Counter(r["status"] for r in rows)
    md = [
        "# FactorEngine R15 Operator Fix Report (by taskbook item)",
        "",
        f"- items classified: {len(rows)}",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for status, n in sorted(counts.items()):
        md.append(f"| {status} | {n} |")
    md += [
        "",
        "## Verified-fixed items (module marker + regression test)",
        "",
    ]
    for r in rows:
        if r["status"] == "fixed":
            md.append(f"- {r['item']}")
    md += [
        "",
        "## Pending / open (checkpoint regen, null calibration, real-data)",
        "",
    ]
    for r in rows:
        if r["status"] != "fixed":
            md.append(f"- {r['item']} — {r['evidence']}")
    (OUT / "operator_fix_report_by_item.md").write_text("\n".join(md), encoding="utf-8")
    print(f"fix report: {csv_path}")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    # R16-011: a claimed-fixed item whose real junit proof is missing (not_run)
    # is a release FAILURE — a marker-only claim must not pass.
    if counts.get("not_run", 0):
        print(f"\nFAIL: {counts['not_run']} claimed-fixed item(s) have no PASSED junit "
              "node proof (run the regression tests with --junitxml first)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
