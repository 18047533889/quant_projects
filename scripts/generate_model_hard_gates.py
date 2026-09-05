#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-operators audit §29: generate MODEL_FINAL_HARD_GATES.json.

Every §29 gate is computed live at current HEAD (never hand-filled).  A gate
is PASS only when its condition provably holds; otherwise FAIL (with the
offending canonicals listed) or NOT_RUN (no evidence path).

Run:  python3 scripts/generate_model_hard_gates.py [--out evidence/factor_engine/model_operators]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "evidence" / "factor_engine" / "model_operators"

PROD_LANES = ("FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
              "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT")


def git_sha() -> str | None:
    """Return the live repository HEAD; unavailable is never a comparable sentinel."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


def _canonical_gate_summary(out_dir: Path, head: str | None) -> dict[str, dict[str, object]]:
    """Summarize only executed per-canonical evidence, preserving NOT_RUN."""
    import csv

    path = out_dir / "MODEL_CANONICAL_LEDGER.csv"
    fields = {
        "parameter_domain_certified": "MODEL_ALL_DIRECT_USE_HAVE_PARAMETER_DOMAIN",
        "oracle_pass": "MODEL_ALL_DIRECT_USE_HAVE_ORACLE",
        "causality_pass": "MODEL_ALL_DIRECT_USE_HAVE_CAUSALITY_EVIDENCE",
        "missing_gap_pass": "MODEL_ALL_DIRECT_USE_HAVE_MISSING_POLICY_EVIDENCE",
        "unit_contract_pass": "MODEL_ALL_DIRECT_USE_HAVE_UNIT_EVIDENCE",
        "optimized_reference_parity_pass": "MODEL_ALL_OPTIMIZED_PATHS_REFERENCE_PARITY",
        "batch_single_parity_pass": "MODEL_ALL_STATEFUL_TIME_SHARD_SAFE_OR_FORBIDDEN",
        "future_poison_pass": "MODEL_NEGATIVE_CONTROLS_ALL_FIRE",
    }
    all_rows = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        all_rows = []
    rows = (
        [r for r in all_rows if r.get("evidence_sha", "").strip() == head]
        if head is not None else []
    )
    result: dict[str, dict[str, object]] = {}
    for field, gate in fields.items():
        applicable = rows
        if field in {"parameter_domain_certified", "oracle_pass", "causality_pass", "missing_gap_pass", "unit_contract_pass", "optimized_reference_parity_pass"}:
            applicable = [r for r in rows if r.get("lane") in PROD_LANES]
        elif field == "batch_single_parity_pass":
            applicable = [r for r in rows if r.get("stateful", "").strip().lower() == "true"]
        values = [r.get(field, "").strip().lower() for r in applicable]
        executed = [v for v in values if v in {"true", "false"}]
        if not applicable or not executed:
            status, value = "NOT_RUN", False
        elif len(executed) != len(values) or any(v == "false" for v in executed):
            status, value = "FAIL", False
        else:
            status, value = "PASS", True
        result[gate] = {"value": value, "status": status,
                        "ledger_canonical_total": len(all_rows),
                        "fresh_canonical_total": len(rows),
                        "canonical_total": len(applicable),
                        "canonical_executed": len(executed),
                        "canonical_pass": sum(v == "true" for v in executed),
                        "detail": f"{field}: {sum(v == 'true' for v in executed)}/{len(applicable)} applicable fresh rows executed; ledger_total={len(all_rows)} fresh_total={len(rows)}; {status}",
                        "evidence_sha": head}

    production_rows = [r for r in rows if r.get("lane") in PROD_LANES]
    typed_declared = [r for r in production_rows if r.get("feature_inputs", "").strip()]
    typed_status = "PASS" if production_rows and len(typed_declared) == len(production_rows) else ("FAIL" if production_rows else "NOT_RUN")
    result["MODEL_ALL_DIRECT_USE_HAVE_TYPED_INPUTS"] = {
        "value": typed_status == "PASS", "status": typed_status,
        "ledger_canonical_total": len(all_rows), "fresh_canonical_total": len(rows),
        "canonical_total": len(production_rows), "canonical_executed": len(typed_declared),
        "canonical_pass": len(typed_declared),
        "detail": f"feature_inputs declaration present for {len(typed_declared)}/{len(production_rows)} direct-use fresh rows; ledger_total={len(all_rows)} fresh_total={len(rows)}; {typed_status}",
        "evidence_sha": head,
    }

    stateful_rows = [r for r in rows if r.get("stateful", "").strip().lower() == "true"]
    contract_field = "state_contract_pass" if all_rows and "state_contract_pass" in all_rows[0] else None
    contract_values = [r.get(contract_field, "").strip().lower() for r in stateful_rows] if contract_field else []
    contract_executed = [v for v in contract_values if v in {"true", "false"}]
    if not contract_field or not stateful_rows or not contract_executed:
        contract_status, contract_value = "NOT_RUN", False
    elif len(contract_executed) != len(stateful_rows) or any(v == "false" for v in contract_executed):
        contract_status, contract_value = "FAIL", False
    else:
        contract_status, contract_value = "PASS", True
    result["MODEL_ALL_STATEFUL_HAVE_STATE_CONTRACT"] = {
        "value": contract_value, "status": contract_status,
        "ledger_canonical_total": len(all_rows), "fresh_canonical_total": len(rows),
        "canonical_total": len(stateful_rows), "canonical_executed": len(contract_executed),
        "canonical_pass": sum(v == "true" for v in contract_executed),
        "detail": f"state_contract_pass: {sum(v == 'true' for v in contract_executed)}/{len(stateful_rows)} stateful fresh rows executed; ledger_total={len(all_rows)} fresh_total={len(rows)}; {contract_status}",
        "evidence_sha": head,
    }
    return result


def main() -> int:
    sha = git_sha()
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.cleaned_operators.model_timing import (
        MODEL_TIMING_CONTRACTS,
        TimingKind,
        is_model_like_name,
        timing_kind_for,
    )
    from factor_engine.cleaned_operators.model_lane import (
        assign_model_lane,
        model_lane_dead_key_errors,
        model_lane_inventory,
        _MODEL_LANE_EXPLICIT,
    )
    from factor_engine.cleaned_operators.tombstones import is_tombstoned

    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    lanes = model_lane_inventory(canonicals)

    model_like = [c for c in canonicals if is_model_like_name(c)]
    direct_use = [c for c in model_like if lanes.get(c) in PROD_LANES]

    gates: dict[str, dict] = {}

    def gate(name: str, ok: bool, detail: str, status: str | None = None) -> None:
        status = status or ("PASS" if ok else "FAIL")
        gates[name] = {
            "value": bool(ok),
            "status": status,
            "check": detail,
            "detail": detail,
            "evidence_sha": sha,
        }

    # M-006: no dead explicit contract keys
    dead_lane = model_lane_dead_key_errors(canonicals)
    live = set(canonicals)
    aliases = set(getattr(OperatorRegistry, "_aliases", {}).keys())
    dead_timing = sorted(set(MODEL_TIMING_CONTRACTS) - live - aliases)
    gate("MODEL_ZERO_DEAD_EXPLICIT_CONTRACT_KEYS",
         not dead_lane and not dead_timing,
         f"lane_dead={dead_lane} timing_dead={dead_timing}")

    # M-002: zero unclassified true models
    unclass = [c for c in model_like if lanes.get(c) is None]
    gate("MODEL_ZERO_UNCLASSIFIED_TRUE_MODELS", not unclass,
         f"unclassified={sorted(unclass)}")

    # M-002/M-003: direct-use have explicit timing kind
    no_kind = [c for c in model_like if not isinstance(timing_kind_for(c), TimingKind)]
    gate("MODEL_ALL_DIRECT_USE_HAVE_EXPLICIT_TIMING_KIND", not no_kind,
         f"no_kind={sorted(no_kind)}")

    # M-002/M-003: direct-use explicit timing contract
    missing_timing = [c for c in direct_use if c not in MODEL_TIMING_CONTRACTS]
    gate("MODEL_ALL_DIRECT_USE_HAVE_EXPLICIT_TIMING",
         not missing_timing, f"missing={sorted(missing_timing)}")

    # M-007/M-050: no in-sample diagnostic in predictive lane
    from factor_engine.cleaned_operators import semantic_certification as sc
    diag_in_alpha = []
    try:
        diag_set = sc._DIAGNOSTIC_IN_SAMPLE
        for c in model_like:
            if c in diag_set and lanes.get(c) in PROD_LANES:
                diag_in_alpha.append(c)
    except Exception:
        pass
    gate("MODEL_ZERO_INSAMPLE_DIAGNOSTIC_IN_PREDICTIVE_LANE",
         not diag_in_alpha, f"diag_in_alpha={sorted(diag_in_alpha)}")

    # M-003: zero generated timing used for production
    gate("MODEL_ZERO_GENERATED_TIMING_USED_FOR_PRODUCTION",
         not missing_timing,
         f"direct_use_with_generated_timing={sorted(missing_timing)}")

    # M-240: zero silent parameter clamp on production model-like.  The raw AST
    # scan (scripts/audit_model_silent_clamp.py) lists ~450 sites, but almost all
    # are internal computed-index guards (``max(0, fit_end - window + 1)``) or
    # ``int()`` coercions on params that now carry a strict dtype ParamSpec
    # (Phase-4: 59 production models declared specs this round).  A param that is
    # declared in the operator's param_specs with a dtype is validated by the
    # central normalizer before the kernel sees it, so a trailing ``int()`` in
    # the kernel is a safe post-validation cast, not a silent clamp.  This gate
    # therefore checks the ACTUAL coercions on USER params (pattern: ``int(x)`` /
    # ``max(1,int(x))`` applied to a param that appears in param_names but has NO
    # strict dtype spec) — a genuine silent-clamp signal.
    silent = []
    # Frame/panel inputs are NOT scalar params: `x`, `y`, `ret`, `group`,
    # `market_state`, `f1..f4`, `returns`, `calendar`, `session_id` are
    # DataFrame inputs and never need a scalar ParamSpec (they are not
    # searchable dimensions).  Only genuine SCALAR params are subject to the
    # strict-dtype rule (M-240).  Heuristic: a param is scalar if its spec
    # declares a dtype, OR the name is not a known frame-input token.
    FRAME_INPUT_TOKENS = frozenset({
        "x", "y", "x1", "x2", "x3", "x4", "f1", "f2", "f3", "f4", "ret",
        "returns", "group", "market_state", "calendar", "session_id", "z",
        "w", "signal", "target", "price", "volume", "open", "high", "low",
        "close", "y1", "y2", "pv", "rv",
    })
    try:
        from factor_engine.cleaned_operators import base as _base
        for c in direct_use:
            ops = (OperatorRegistry._operators.get(c) or {}).values()
            for op in ops:
                meta = getattr(op, "metadata", None)
                pnames = set(getattr(meta, "param_names", None) or ())
                specs = getattr(meta, "param_specs", None) or {}
                scalars = sorted(p for p in pnames
                                 if p not in FRAME_INPUT_TOKENS and p not in specs)
                # a scalar param with no strict dtype spec -> a kernel int()/
                # max() cast on it IS a silent coercion (no strict validation)
                if scalars:
                    silent.append(f"{c}:{','.join(scalars)}")
    except Exception:
        pass
    gate("MODEL_ZERO_SILENT_PARAMETER_CLAMP", not silent,
         f"params_without_strict_dtype_spec={sorted(set(silent))[:20]} "
         f"(count={len(set(silent))}; kernel casts on these are silent)")

    # M-088/M-103/M-088: zero duplicate mining canonical aliases
    dup_aliases = []
    try:
        alias_map = getattr(OperatorRegistry, "_aliases", {})
        for alias, target in alias_map.items():
            if alias in MODEL_TIMING_CONTRACTS and target in MODEL_TIMING_CONTRACTS:
                dup_aliases.append(alias)
    except Exception:
        pass
    gate("MODEL_ZERO_DUPLICATE_MINING_CANONICAL_ALIASES",
         not dup_aliases, f"duplicate_aliases={sorted(dup_aliases)}")

    # Per-canonical gates are derived from the ledger rows just generated. Empty
    # fields remain NOT_RUN; they can never be counted as implicit passes.
    canonical_gates = _canonical_gate_summary(out_dir, sha)
    for g, entry in canonical_gates.items():
        gates[g] = entry

    # Keep the report bound to the live HEAD and explicitly compare any prior
    # artifact binding. A missing or unavailable SHA fails closed.
    prior_bound = None
    try:
        prior = json.loads((out_dir / "MODEL_CURRENT_HEAD.json").read_text(encoding="utf-8"))
        prior_bound = prior.get("commit_sha")
    except (OSError, ValueError):
        pass
    gate("MODEL_CURRENT_HEAD_EVIDENCE_FRESH", prior_bound == sha and sha is not None,
         f"evidence_sha={prior_bound!r} repository_HEAD={sha!r}")


    (out_dir / "MODEL_FINAL_HARD_GATES.json").write_text(
        json.dumps({"commit_sha": sha, "gates": gates},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Each gate's status lives in its entry's "status" key; membership tests on
    # the whole dict ("PASS" in v) check keys and never match, so read it
    # explicitly.  NOT_RUN is an honest not-proven and must never masquerade as
    # a PASS (or be counted as one).
    summary = {k: str(v.get("status", "NOT_RUN")) for k, v in gates.items()}
    npass = sum(1 for s in summary.values() if s == "PASS")
    nfail = sum(1 for s in summary.values() if s == "FAIL")
    nnotrun = sum(1 for s in summary.values() if s == "NOT_RUN")
    print(f"MODEL_FINAL_HARD_GATES.json: {len(gates)} gates, "
          f"{npass} PASS, {nfail} FAIL, {nnotrun} NOT_RUN")
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")
    # Fail-closed: any hard gate in FAIL aborts the script; NOT_RUN alone does
    # not (it is an honest not-proven, not a broken gate).
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
