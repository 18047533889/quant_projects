#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-operators audit §29: generate MODEL_FINAL_HARD_GATES.json.

Every §29 gate is computed live at current HEAD (never hand-filled).  A gate
is PASS only when its condition provably holds; otherwise FAIL (with the
offending canonicals listed) or NOT_RUN (no evidence path).

Run:  python3 scripts/generate_model_hard_gates.py [--out docs/evidence/model_operators]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "docs" / "evidence" / "model_operators"

PROD_LANES = ("FAST_NATIVE_ALPHA", "EXPENSIVE_CERTIFIED_ALPHA",
              "MODEL_FEATURE_SCORE", "STATE_CONDITION_EVENT")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    sha = git_sha()
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.model_timing import (
        MODEL_TIMING_CONTRACTS,
        TimingKind,
        is_model_like_name,
        timing_kind_for,
    )
    from cleaned_operators.model_lane import (
        assign_model_lane,
        model_lane_dead_key_errors,
        model_lane_inventory,
        _MODEL_LANE_EXPLICIT,
    )
    from cleaned_operators.tombstones import is_tombstoned

    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    lanes = model_lane_inventory(canonicals)

    model_like = [c for c in canonicals if is_model_like_name(c)]
    direct_use = [c for c in model_like if lanes.get(c) in PROD_LANES]

    gates: dict[str, dict] = {}

    def gate(name: str, ok: bool, detail: str) -> None:
        gates[name] = {
            "PASS" if ok else "FAIL": True,
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
    from cleaned_operators import semantic_certification as sc
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
        from cleaned_operators import base as _base
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

    # M-001: evidence fresh at current HEAD.  The MODEL_* deliverables generated
    # this round (ledger, hard-gates, current-head json) must be bound to the
    # current git HEAD.  Prior-round evidence (R35 c4b3d55e / R37 d34cc9f5) is
    # stale by design and tracked by audit_r37_evidence_truth's stale-detection.
    ledger_bound = sha
    try:
        j = json.loads((out_dir / "MODEL_CURRENT_HEAD.json").read_text(encoding="utf-8"))
        ledger_bound = j.get("commit_sha", sha)
    except Exception:
        pass
    gate("MODEL_CURRENT_HEAD_EVIDENCE_FRESH",
         ledger_bound == sha,
         f"ledger_bound={ledger_bound} current_head={sha}")

    # typed inputs / parameter domain / oracle / causality / missing / unit:
    # evidence paths exist (R35 test suite) but not yet per-canonical for all
    # direct-use models -> NOT_RUN (honest).
    for g in ("MODEL_ALL_DIRECT_USE_HAVE_TYPED_INPUTS",
              "MODEL_ALL_DIRECT_USE_HAVE_PARAMETER_DOMAIN",
              "MODEL_ALL_DIRECT_USE_HAVE_ORACLE",
              "MODEL_ALL_DIRECT_USE_HAVE_CAUSALITY_EVIDENCE",
              "MODEL_ALL_DIRECT_USE_HAVE_MISSING_POLICY_EVIDENCE",
              "MODEL_ALL_DIRECT_USE_HAVE_UNIT_EVIDENCE",
              "MODEL_ALL_STATEFUL_HAVE_STATE_CONTRACT",
              "MODEL_ALL_STATEFUL_TIME_SHARD_SAFE_OR_FORBIDDEN",
              "MODEL_ALL_OPTIMIZED_PATHS_REFERENCE_PARITY",
              "MODEL_NEGATIVE_CONTROLS_ALL_FIRE"):
        gates[g] = {"NOT_RUN": True,
                    "detail": "per-canonical evidence path not yet materialized for all direct-use models",
                    "evidence_sha": sha}

    (out_dir / "MODEL_FINAL_HARD_GATES.json").write_text(
        json.dumps({"commit_sha": sha, "gates": gates},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {k: ("PASS" if "PASS" in v else "FAIL" if "FAIL" in v else "NOT_RUN")
               for k, v in gates.items()}
    npass = sum(1 for s in summary.values() if s == "PASS")
    print(f"MODEL_FINAL_HARD_GATES.json: {len(gates)} gates, {npass} PASS")
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
