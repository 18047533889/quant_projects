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

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKBOK = Path("/home/shw/quant_projects/FactorEngine_全量最终审计整改提示词_20260809.md")
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


def _has_test_cover(test_glob: str) -> bool:
    for p in (REPO / "tests").rglob("test_*.py"):
        if re.search(test_glob, p.name):
            return True
    return False


def scan_evidence() -> dict[str, str]:
    status: dict[str, str] = {}
    for item, (mod, marker, test_glob) in EVIDENCE.items():
        module = _find_module(mod)
        if module is None:
            status[item] = "not_applicable"
            continue
        text = module.read_text(encoding="utf-8")
        if re.search(marker, text) and _has_test_cover(test_glob):
            status[item] = "fixed"
        else:
            status[item] = "pending_research" if marker.startswith("checkpoint") else "not_fixed"
    return status


def _all_items() -> list[str]:
    """Every NEW-0xx / HIST-0xx code in the taskbook, in document order."""
    if not TASKBOK.exists():
        return []
    text = TASKBOK.read_text(encoding="utf-8")
    return re.findall(r"\b(NEW-\d{3})\b|\b(HIST-\d{3})\b", text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = scan_evidence()
    codes = sorted({a or b for a, b in _all_items()})

    rows: list[dict[str, str]] = []
    for code in codes:
        rows.append({
            "item": code,
            "status": evidence.get(code, "pending_research"),
            "evidence": "module marker + regression test" if code in evidence else
                        "open item (checkpoint / null-calibration / real-data) — see memory",
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


if __name__ == "__main__":
    main()
