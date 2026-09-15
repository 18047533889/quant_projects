#!/usr/bin/env python3
"""R07: honest candidate-math lane plus backend-parity JUnit ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANDIDATES = ("ts_skew", "ts_kurt", "ts_quantile", "is_nan")
BACKEND_TOKENS = frozenset({
    "pandas", "pandas_numpy", "polars", "polars_long", "duckdb", "duckdb_sql", "sql", "numba",
})


def _source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "factor_engine").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _independent_expected(canonical: str, values: np.ndarray) -> np.ndarray:
    """Small fixed-fixture oracle, independent of pandas rolling kernels.

    This is not a general parameter-domain certification: only the documented
    w=5, q=.5 fixture is covered. Decimal centered moments avoid sharing the
    implementation under test with the reference.
    """
    from decimal import Decimal, localcontext
    out = np.full((len(values), 1), np.nan)
    for i, value in enumerate(values):
        if canonical == "is_nan":
            out[i, 0] = float(float(value) != float(value))
            continue
        window = [float(v) for v in values[max(0, i - 4):i + 1] if np.isfinite(v)]
        n = len(window)
        if canonical == "ts_quantile":
            if n:
                ordered = sorted(window)
                out[i, 0] = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
            continue
        if n < (3 if canonical == "ts_skew" else 5):
            continue
        with localcontext() as context:
            context.prec = 50
            x = [Decimal(str(v)) for v in window]
            dn = Decimal(n)
            mean = sum(x) / dn
            centered = [v - mean for v in x]
            m2 = sum(v ** 2 for v in centered) / dn
            if canonical == "ts_skew":
                m3 = sum(v ** 3 for v in centered) / dn
                result = (dn * (dn - 1)).sqrt() / (dn - 2) * m3 / (m2 * m2.sqrt())
            else:
                m4 = sum(v ** 4 for v in centered) / dn
                result = (dn - 1) / ((dn - 2) * (dn - 3)) * ((dn + 1) * m4 / m2 ** 2 - 3 * (dn - 1))
            out[i, 0] = float(result)
    return out


def _candidate_lane() -> list[dict[str, Any]]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry, _impl_source_hash
    from factor_engine.cleaned_operators.math_certificate import _validation_equal
    from factor_engine.cleaned_operators.semantic_certification import operator_certification_for
    from factor_engine.backend.operator_call_capability import (
        CapabilityLevel,
        check_operator_call_capability,
    )

    load_all(include_research=True)
    values = np.array([1., 2., 3., 5., np.nan, 8., 13., 21., 34., 55., 89., 144.])
    panel = pd.DataFrame({"asset": values})
    rows: list[dict[str, Any]] = []
    for canonical in CANDIDATES:
        entry = OperatorRegistry._catalog.get(canonical, {})
        resolved = OperatorRegistry.resolve_canonical_strict(canonical)
        backends = list(OperatorRegistry.backends_for(resolved))
        op = OperatorRegistry.get(resolved, backend="pandas_numpy", mode="research")
        row: dict[str, Any] = {
            "advertised": canonical,
            "runtime_resolved": resolved,
            "registered_backends": backends,
            "backend_results": {backend: {"status": "NOT_RUN", "reason": "candidate lane executes pandas_numpy only"}
                                for backend in backends},
            "catalog_production_certified": entry.get("production_certified") is True,
            "production_registry_rejected": OperatorRegistry.get(
                resolved, backend="pandas_numpy", mode="production"
            ) is None,
            "six_gate_production_certified": operator_certification_for(
                resolved, dict(entry)
            ).production_certified,
        }
        production_decision = check_operator_call_capability(
            resolved, backend="pandas_numpy", production=True
        )
        row["production_capability"] = {
            "level": production_decision.level.value,
            "reason": production_decision.reason,
        }
        if op is None:
            row.update(status="NOT_RUN", reason="no pandas_numpy research binding")
            rows.append(row)
            continue
        row.update(
            implementation_module=type(op).__module__,
            implementation_class=type(op).__qualname__,
            implementation_hash=_impl_source_hash(op),
        )
        try:
            if canonical == "ts_skew":
                actual = op.calculate(panel.copy(), window=5)
            elif canonical == "ts_kurt":
                actual = op.calculate(panel.copy(), window=5)
            elif canonical == "ts_quantile":
                actual = op.calculate(panel.copy(), d=5, q=0.5)
            else:
                actual = op.calculate(panel.copy())
            expected = _independent_expected(canonical, values)
            ok = _validation_equal(
                np.asarray(actual), np.asarray(expected), rtol=1e-10, atol=1e-12
            )
            row["backend_results"]["pandas_numpy"] = {
                "status": "PASS_LIMITED" if ok else "FAIL",
                "oracle": "independent Decimal centered moments / sorted median / scalar NaN predicate; fixed fixture only",
                "fixture": {"rows": len(panel), "columns": 1, "contains_nan": True},
            }
            production_ok = (
                production_decision.level is not CapabilityLevel.PRODUCTION
                and not row["catalog_production_certified"]
                and not row["six_gate_production_certified"]
            )
            row["status"] = "PASS_LIMITED" if ok and production_ok else "FAIL"
            row["production_lane_status"] = "REJECTED_AS_EXPECTED" if production_ok else "FAIL_OPEN"
        except Exception as exc:  # evidence tooling must report, not conceal
            row.update(status="AUDIT_ERROR", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def _junit_ledger(path: Path) -> dict[str, Any]:
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all(include_research=True)
    known = set(OperatorRegistry.list_canonical()) | set(OperatorRegistry._aliases)
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        name = case.attrib.get("name", "")
        classname = case.attrib.get("classname", "")
        child = next(iter(case), None)
        status = "PASS"
        reason = ""
        if child is not None and child.tag in {"skipped", "failure", "error"}:
            status = {"skipped": "SKIPPED", "failure": "FAIL", "error": "ERROR"}[child.tag]
            reason = child.attrib.get("message", "") or (child.text or "").strip().splitlines()[0]
        params = name[name.find("[") + 1:name.rfind("]")].split("-") if "[" in name else []
        exact_canonicals = sorted({OperatorRegistry.resolve_canonical(p) for p in params if p in known})
        exact_backends = sorted({p for p in params if p in BACKEND_TOKENS})
        cases.append({
            "node": f"{classname}::{name}",
            "status": status,
            "reason": reason or None,
            "canonical_mapping": exact_canonicals or None,
            "backend_mapping": exact_backends or None,
            "mapping_note": "exact parametrized token only; null means unresolved, not inferred",
        })
    counts = Counter(row["status"] for row in cases)
    skip_reasons = Counter(row["reason"] for row in cases if row["status"] == "SKIPPED")
    return {
        "junit": str(path),
        "counts": dict(sorted(counts.items())),
        "skip_reason_counts": dict(sorted(skip_reasons.items(), key=lambda item: (-item[1], str(item[0])))),
        "nodes": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-hash-before", required=True,
                        help="Observed FE source hash immediately before the test run")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_hash_after = _source_tree_hash(repo)
    payload = {
        "schema": "r07-candidate-math-ledger-v1",
        "source_tree_hash_at_ledger": source_hash_after,
        "source_tree_hash_before": args.source_hash_before,
        "source_stability": ("MATCH" if args.source_hash_before == source_hash_after else "CHANGED"),
        "candidate_lane_scope": "four real unverified pandas_numpy bindings; other backends NOT_RUN",
        "candidate_lane": _candidate_lane(),
        "backend_parity": _junit_ledger(args.junit),
        "overall_status": "PARTIAL",
        "overall_reason": "Limited candidate and parity evidence only; not all operators, parameter domains, approved data or performance acceptance. See source_stability and per-node results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "candidate_statuses": {r["advertised"]: r["status"] for r in payload["candidate_lane"]},
        "backend_counts": payload["backend_parity"]["counts"],
        "source_tree_hash_at_ledger": payload["source_tree_hash_at_ledger"],
        "overall_status": payload["overall_status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
