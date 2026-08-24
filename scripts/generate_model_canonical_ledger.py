#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-operators audit §27: generate MODEL_CANONICAL_LEDGER.csv/parquet.

For every model-like canonical at current HEAD, emit the 35-field ledger row
(family / true_model_type / semantic_role / surface / lane / timing_kind /
explicit_timing / feature_inputs / label_input / label_horizon / stateful /
checkpointable / missing_policy / window_semantics / min_history /
min_effective_obs / input_units / output_unit / parameter_domain_certified /
oracle_required / oracle_pass / causality_pass / future_poison_pass /
label_poison_pass / missing_gap_pass / unit_contract_pass / backend_parity_pass /
batch_single_parity_pass / optimized_reference_parity_pass / performance_lane /
default_searchable / final_direct_use_ready / failure_reasons / evidence_sha).

``final_direct_use_ready`` is DERIVED from the sub-gates (never hand-filled):
  - explicit timing present AND
  - lane is a production lane AND
  - not tombstoned AND
  - parameter domain has at least one certified point (or no scalar params).

Evidence is bound to current git HEAD (M-001); ``evidence_sha`` = HEAD sha.

Run:  python3 scripts/generate_model_canonical_ledger.py [--out evidence/factor_engine/model_operators]
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "evidence" / "factor_engine" / "model_operators"

LEDGER_FIELDS = [
    "canonical", "family", "true_model_type", "semantic_role", "surface",
    "lane", "timing_kind", "explicit_timing", "feature_inputs", "label_input",
    "label_horizon", "stateful", "checkpointable", "missing_policy",
    "window_semantics", "min_history", "min_effective_obs", "input_units",
    "output_unit", "parameter_domain_certified", "oracle_required",
    "oracle_pass", "causality_pass", "future_poison_pass", "label_poison_pass",
    "missing_gap_pass", "unit_contract_pass", "backend_parity_pass",
    "batch_single_parity_pass", "optimized_reference_parity_pass",
    "performance_lane", "default_searchable", "final_direct_use_ready",
    "failure_reasons", "evidence_sha",
]

TRUE_MODEL_FAMILIES = frozenset({
    "pcr", "pls", "enet", "regime", "mixture", "ar", "kalman", "garch",
    "har", "dmd", "hankel", "ssa", "matrix_profile", "signature", "kernel",
    "hsic", "knn", "state", "first_passage", "lyapunov", "rqa",
    "transfer_entropy", "change_point", "regression",
})


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _surface_of(canonical: str) -> str:
    try:
        from factor_engine.cleaned_operators.operator_surface import classify_canonical
        return classify_canonical(canonical)
    except Exception:
        return "unclassified"


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
        model_family_of,
        timing_kind_for,
    )
    from factor_engine.cleaned_operators.model_lane import assign_model_lane, _category_of
    from factor_engine.cleaned_operators.model_contract import get_model_operator_contract

    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())

    rows: list[dict] = []
    model_like = [c for c in canonicals if is_model_like_name(c, _category_of(c))]
    for c in model_like:
        lane = assign_model_lane(c)
        timing_kind = timing_kind_for(c)
        explicit = c in MODEL_TIMING_CONTRACTS
        surface = _surface_of(c)
        family = model_family_of(c)
        mc = get_model_operator_contract(c)

        # derive final_direct_use_ready from sub-gates only (audit §27: never
        # hand-filled).  "Ready" = explicit timing + production lane + not
        # tombstoned + has a certified parameter-domain point.  Parameter-domain
        # certification is an evidence-phase artifact (R37 certifies 17
        # primitives, zero models so far) — so today 0/300 direct-use models are
        # "ready", which is the honest state the audit's §36 tri-state requires.
        failures: list[str] = []
        if not explicit:
            failures.append("no_explicit_timing")
        if lane not in ("FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
                        "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT"):
            failures.append("not_production_lane")
        if lane == "DELETE_TOMBSTONE":
            failures.append("tombstoned")
        # parameter domain: any certified point in the runtime store
        param_certified = False
        try:
            from factor_engine.runtime.parameter_domain_store import ParameterDomainCertificationStore
            store = ParameterDomainCertificationStore()
            param_certified = bool(store.operator_has_any_certified_region(c))
        except Exception:
            param_certified = False
        if not param_certified:
            failures.append("no_parameter_domain_point")
        final_ready = not failures

        row = {
            "canonical": c,
            "family": family,
            "true_model_type": family if family in TRUE_MODEL_FAMILIES else "statistical_estimator",
            "semantic_role": (mc.role if mc else ""),
            "surface": surface,
            "lane": lane,
            "timing_kind": timing_kind.value,
            "explicit_timing": str(explicit),
            "feature_inputs": ",".join(mc.feature_params) if mc else "",
            "label_input": (mc.label_param or "") if mc else "",
            "label_horizon": str((mc.label_horizon_param or "") if mc else ""),
            "stateful": str(bool(mc and mc.stateful)),
            "checkpointable": str(bool(mc and mc.checkpoint_supported)),
            "missing_policy": (mc.missing_policy if mc else ""),
            "window_semantics": "",
            "min_history": "",
            "min_effective_obs": "",
            "input_units": "",
            "output_unit": "",
            "parameter_domain_certified": str(param_certified),
            "oracle_required": "family_oracle",
            "oracle_pass": "", "causality_pass": "", "future_poison_pass": "",
            "label_poison_pass": "", "missing_gap_pass": "", "unit_contract_pass": "",
            "backend_parity_pass": "", "batch_single_parity_pass": "",
            "optimized_reference_parity_pass": "",
            "performance_lane": "expensive" if lane == "EXPENSIVE_CERTIFIED_ALPHA" else (
                "fast" if lane == "FAST_NATIVE_ALPHA" else ""),
            "default_searchable": "false" if lane in ("DIAGNOSTIC_RESEARCH", "DELETE_TOMBSTONE") else "true",
            "final_direct_use_ready": str(final_ready),
            "failure_reasons": "|".join(failures),
            "evidence_sha": sha,
        }
        rows.append(row)

    # write csv + parquet
    csv_path = out_dir / "MODEL_CANONICAL_LEDGER.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_dir / "MODEL_CANONICAL_LEDGER.parquet", index=False)
        parquet_ok = True
    except Exception:
        parquet_ok = False

    # MODEL_CURRENT_HEAD.json
    (out_dir / "MODEL_CURRENT_HEAD.json").write_text(json.dumps({
        "commit_sha": sha,
        "canonical_count": len(canonicals),
        "model_like_count": len(model_like),
        "explicit_timing_count": len(MODEL_TIMING_CONTRACTS),
        "final_direct_use_ready_count": sum(1 for r in rows if r["final_direct_use_ready"] == "True"),
        "generated_at": subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"],
                                       capture_output=True, text=True).stdout.strip(),
        "runtime_versions": {},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"MODEL_CANONICAL_LEDGER.csv: {len(rows)} rows -> {csv_path}")
    print(f"parquet: {parquet_ok}")
    print(f"final_direct_use_ready: {sum(1 for r in rows if r['final_direct_use_ready']=='True')}/{len(rows)}")
    print(f"bound to HEAD: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
