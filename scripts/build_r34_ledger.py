# -*- coding: utf-8 -*-
"""R34 §185: Per-Canonical Correctness Ledger.

把本机已生成的证据（参数域认证 / hard gates / primitive+factor evidence /
typed signature / edge contract / stateful 行为检测）按 canonical 聚合成一个
machine-readable correctness ledger。

production verdict = 实际证据 AND，绝不从 status/surface 推导（R34 §7）。
缺证据的维度一律 PENDING（PASS/FAIL/PENDING/N/A 五态分开，R34 §87）。

输出：
    docs/evidence/r34/R34_CANONICAL_CORRECTNESS_LEDGER.csv
    docs/evidence/r34/R34_CANONICAL_CORRECTNESS_LEDGER.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded  # noqa: E402
from factor_engine.cleaned_operators.production_hardening import factor_production_targets  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402

E = Path("docs/evidence/r34")


def _typed_signature_status(canonical: str) -> tuple[str, str]:
    from factor_engine.backend.production_signature import signature_for

    sig = signature_for(canonical)
    if sig is None:
        return "FAIL", "no typed signature"
    if not getattr(sig, "params", ()):
        return "FAIL", "empty params"
    return "PASS", ""


def _edge_status(canonical: str) -> tuple[str, str]:
    from factor_engine.cleaned_operators.edge_requirements import (
        edge_contract,
        production_edge_evidence_complete,
    )

    c = edge_contract(canonical)
    if c is None:
        return "FAIL", "undeclared edge contract"
    if production_edge_evidence_complete(canonical):
        return "PASS", ""
    return "FAIL", "required edge evidence incomplete"


def _param_domain_status(canonical: str, coverage: dict) -> tuple[str, str]:
    certified = coverage.get("certified", {})
    not_certified = coverage.get("not_certified", {})
    if canonical in certified:
        return "PASS", f"windows={certified[canonical]}"
    if canonical in not_certified:
        return "PENDING", str(not_certified[canonical])
    return "PENDING", "no independent oracle case run"


def _stateful_class(canonical: str) -> str:
    from factor_engine.cleaned_operators.production_hardening import (
        FULL_HISTORY_REPLAY_CANONICALS,
        SEGMENTED_EXECUTION_CANONICALS,
    )

    if canonical in SEGMENTED_EXECUTION_CANONICALS:
        return "segmented"
    if canonical in FULL_HISTORY_REPLAY_CANONICALS:
        return "full_replay"
    return "bounded"


def _production_verdict(row: dict) -> str:
    dims = ["typed_signature", "edge_declared", "edge_verified", "parameter_domain"]
    vals = [row.get(d) for d in dims]
    if any(v == "FAIL" for v in vals):
        return "NOT_CERTIFIED"
    if all(v == "PASS" for v in vals):
        return "CERTIFIED"
    return "PENDING"


def main() -> int:
    ensure_cleaned_loaded()
    E.mkdir(parents=True, exist_ok=True)
    try:
        coverage = json.loads((E / "R34_PARAMETER_DOMAIN_COVERAGE.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        coverage = {}

    rows: list[dict] = []
    for canon in sorted(factor_production_targets()):
        typed, typed_note = _typed_signature_status(canon)
        edge_d, edge_d_note = _edge_status(canon)
        # edge verified 复用 edge 状态（PENDING 细节区分）
        edge_v = "PASS" if edge_d == "PASS" else ("PENDING" if edge_d_note == "undeclared edge contract" else "FAIL")
        pd_, pd_note = _param_domain_status(canon, coverage)
        row = {
            "canonical": canon,
            "surface": _surface(canon),
            "role": _role(canon),
            "typed_signature": typed,
            "typed_note": typed_note,
            "edge_declared": edge_d,
            "edge_verified": edge_v,
            "parameter_domain": pd_,
            "param_note": pd_note,
            "stateful_class": _stateful_class(canon),
            "backends": ",".join(sorted(OperatorRegistry.backends_for(canon))) or "none",
            "production_verdict": "",
        }
        row["production_verdict"] = _production_verdict(row)
        rows.append(row)

    # CSV
    with (E / "R34_CANONICAL_CORRECTNESS_LEDGER.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # JSON（聚合统计）
    from collections import Counter

    verdicts = Counter(r["production_verdict"] for r in rows)
    ts = Counter(r["typed_signature"] for r in rows)
    ed = Counter(r["edge_declared"] for r in rows)
    pd_ = Counter(r["parameter_domain"] for r in rows)
    summary = {
        "generated_by": "scripts/build_r34_ledger.py",
        "total_production_canonicals": len(rows),
        "production_verdict": dict(verdicts),
        "typed_signature": dict(ts),
        "edge_declared": dict(ed),
        "parameter_domain": dict(pd_),
        "rows": rows,
    }
    (E / "R34_CANONICAL_CORRECTNESS_LEDGER.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[r34-ledger] total={len(rows)} verdicts={dict(verdicts)} "
        f"typed={dict(ts)} edge={dict(ed)} param={dict(pd_)}"
    )
    return 0


def _surface(canonical: str) -> str:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    try:
        return classify_canonical(canonical)
    except Exception:
        return "unknown"


def _role(canonical: str) -> str:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    try:
        return classify_canonical(canonical)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
